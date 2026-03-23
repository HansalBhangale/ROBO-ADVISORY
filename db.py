"""
==============================================================================
MongoDB Atlas Integration — User & Portfolio Storage
==============================================================================
Handles user authentication, portfolio persistence, and model version tracking.

Usage:
    from db import MongoDB
    db = MongoDB()
    db.save_user("user@email.com", "John", risk_profile={...})
    db.save_portfolio("user@email.com", portfolio_data)
    portfolios = db.get_user_portfolios("user@email.com")
==============================================================================
"""

import os
import sys
import hashlib
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Try to import pymongo, graceful fallback if not installed
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


class MongoDB:
    """MongoDB Atlas connection manager."""

    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False

        if not PYMONGO_AVAILABLE:
            print("⚠️  pymongo not installed. Run: pip install pymongo python-dotenv")
            return

        uri = os.getenv("MONGODB_URI", "")
        if not uri:
            print("⚠️  MONGODB_URI not set in .env file")
            return

        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[os.getenv("MONGODB_DB", "robo_advisory")]
            self.connected = True
            print("✅ Connected to MongoDB Atlas")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"⚠️  MongoDB connection failed: {e}")
        except Exception as e:
            print(f"⚠️  MongoDB error: {e}")

    def is_connected(self):
        return self.connected and self.db is not None

    # =========================================================================
    # USER OPERATIONS
    # =========================================================================
    def save_user(self, email, name, password=None, risk_profile=None, risk_score=None, risk_category=None):
        """Create or update a user."""
        if not self.is_connected():
            return None

        user_data = {
            "email": email,
            "updated_at": datetime.utcnow(),
        }
        if name:
            user_data["name"] = name
            
        if password:
            hashed_pw = hashlib.sha256(password.encode('utf-8')).hexdigest()
            user_data["password_hash"] = hashed_pw

        if risk_profile:
            user_data["risk_profile"] = risk_profile
        if risk_score is not None:
            user_data["risk_score"] = risk_score
        if risk_category:
            user_data["risk_category"] = risk_category

        set_on_insert = {"created_at": datetime.utcnow()}
        if not name:
            set_on_insert["name"] = ""
        # If no password provided during insert, store empty string
        if not password:
            set_on_insert["password_hash"] = ""

        result = self.db.users.update_one(
            {"email": email},
            {"$set": user_data, "$setOnInsert": set_on_insert},
            upsert=True
        )
        return result

    def get_user(self, email):
        """Get user by email."""
        if not self.is_connected():
            return None
        return self.db.users.find_one({"email": email})

    def get_or_create_user(self, email, password, name=""):
        """Get existing user or create new one with password authentication.
        Returns: (user_dict, error_string)
        """
        user = self.get_user(email)
        hashed_pw = hashlib.sha256(password.encode('utf-8')).hexdigest()
        
        if user is None:
            # Create new user
            self.save_user(email, name, password=password)
            user = self.get_user(email)
            return user, None
        else:
            # Authenticate existing user
            stored_hash = user.get("password_hash", "")
            if stored_hash and stored_hash != hashed_pw:
                return None, "Invalid password combination for this email."
            elif not stored_hash and password:
                # If they didn't have a password before (legacy), set it now
                self.save_user(email, name, password=password)
            
            # Update name if provided and it was empty before
            if name and not user.get("name"):
                 self.save_user(email, name)
                 user = self.get_user(email)
                 
            return user, None

    # =========================================================================
    # PORTFOLIO OPERATIONS
    # =========================================================================
    def save_portfolio(self, email, portfolio_data, risk_score, risk_category,
                       capital, allocations, backtest=None):
        """Save a generated portfolio."""
        if not self.is_connected():
            return None

        user = self.get_user(email)
        if not user:
            return None

        # We removed auto-archiving here to allow multiple discrete active portfolios per user

        doc = {
            "user_email": email,
            "user_id": user["_id"],
            "created_at": datetime.utcnow(),
            "capital": capital,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "portfolio_metrics": {
                "eq_pct": portfolio_data.get("eq_pct"),
                "cash_pct": portfolio_data.get("cash_pct"),
                "port_ret": portfolio_data.get("port_ret"),
                "sharpe": portfolio_data.get("sharpe"),
            },
            "allocations": [
                {
                    "ticker": a["ticker"],
                    "name": a.get("name", a["ticker"]),
                    "weight_pct": a["weight_pct"],
                    "capital": a["capital"],
                    "shares": a["shares"],
                    "entry_price": a.get("current_price", 0),
                    "combined_score": a.get("combined_score"),
                    "signal": a.get("combined_signal"),
                    "predicted_return": a.get("predicted_return"),
                }
                for a in allocations
            ],
            "backtest": backtest,
            "status": "active",
        }

        result = self.db.portfolios.insert_one(doc)
        return result.inserted_id

    def get_user_portfolios(self, email, limit=10):
        """Get all portfolios for a user, most recent first."""
        if not self.is_connected():
            return []
        cursor = self.db.portfolios.find(
            {"user_email": email}
        ).sort("created_at", -1).limit(limit)
        return list(cursor)

    def get_latest_portfolio(self, email):
        """Get most recent active portfolio."""
        if not self.is_connected():
            return None
        return self.db.portfolios.find_one(
            {"user_email": email, "status": "active"},
            sort=[("created_at", -1)]
        )

    def get_portfolio_by_id(self, portfolio_id):
        """Get a specific portfolio by its ID."""
        if not self.is_connected():
            return None
        from bson import ObjectId
        return self.db.portfolios.find_one({"_id": ObjectId(portfolio_id)})

    # =========================================================================
    # MODEL VERSION TRACKING
    # =========================================================================
    def save_model_version(self, model_type, version_tag, metrics, file_path=""):
        """Track a trained model version."""
        if not self.is_connected():
            return None

        doc = {
            "model_type": model_type,  # "technical" or "fundamental"
            "version": version_tag,
            "trained_at": datetime.utcnow(),
            "metrics": metrics,
            "file_path": file_path,
        }
        result = self.db.model_versions.insert_one(doc)
        return result.inserted_id

    def get_model_versions(self, model_type=None, limit=10):
        """List model versions, optionally filtered by type."""
        if not self.is_connected():
            return []

        query = {}
        if model_type:
            query["model_type"] = model_type

        cursor = self.db.model_versions.find(query).sort("trained_at", -1).limit(limit)
        return list(cursor)

    def get_latest_model_version(self, model_type):
        """Get the most recent version of a model type."""
        if not self.is_connected():
            return None
        return self.db.model_versions.find_one(
            {"model_type": model_type},
            sort=[("trained_at", -1)]
        )

    # =========================================================================
    # REBALANCING
    # =========================================================================
    def mark_rebalanced(self, old_portfolio_id, new_portfolio_id):
        """Mark old portfolio as rebalanced, linking to new one."""
        if not self.is_connected():
            return
        from bson import ObjectId
        self.db.portfolios.update_one(
            {"_id": ObjectId(old_portfolio_id)},
            {"$set": {
                "status": "rebalanced",
                "rebalanced_to": ObjectId(new_portfolio_id),
                "rebalanced_at": datetime.utcnow()
            }}
        )

    def update_portfolio_in_place(self, portfolio_id, portfolio_data, allocations, backtest=None):
        """Overwrite an existing portfolio's metrics and allocations in-place."""
        if not self.is_connected():
            return None
        from bson import ObjectId
        
        doc_updates = {
            "updated_at": datetime.utcnow(),
            "portfolio_metrics": {
                "eq_pct": portfolio_data.get("eq_pct"),
                "cash_pct": portfolio_data.get("cash_pct"),
                "port_ret": portfolio_data.get("port_ret"),
                "sharpe": portfolio_data.get("sharpe"),
            },
            "allocations": [
                {
                    "ticker": a["ticker"],
                    "name": a.get("name", a["ticker"]),
                    "weight_pct": a["weight_pct"],
                    "capital": a["capital"],
                    "shares": a["shares"],
                    "entry_price": a.get("current_price", 0),
                    "combined_score": a.get("combined_score"),
                    "signal": a.get("combined_signal"),
                    "predicted_return": a.get("predicted_return"),
                }
                for a in allocations
            ]
        }
        
        if backtest is not None:
            doc_updates["backtest"] = backtest
            
        result = self.db.portfolios.update_one(
            {"_id": ObjectId(portfolio_id)},
            {"$set": doc_updates}
        )
        return result.modified_count


# =============================================================================
# CLI TEST
# =============================================================================
if __name__ == "__main__":
    db = MongoDB()
    if db.is_connected():
        print("\n✅ MongoDB connection test passed!")
        print(f"   Database: {db.db.name}")
        print(f"   Collections: {db.db.list_collection_names()}")

        # Test user creation
        db.save_user("test@example.com", "Test User")
        user = db.get_user("test@example.com")
        print(f"\n   Test user: {user['email']} (created: {user.get('created_at', 'N/A')})")

        # Cleanup test user
        db.db.users.delete_one({"email": "test@example.com"})
        print("   ✓ Test user cleaned up")
    else:
        print("\n❌ MongoDB not connected. Check your .env file:")
        print('   MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net/"')
        print('   MONGODB_DB="robo_advisory"')
