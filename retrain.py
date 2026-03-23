"""
==============================================================================
Live Model Retraining Engine
==============================================================================
Supports incremental fine-tuning and full retraining for both models:
  - Technical (LSTM+Attention Ensemble): Incremental or Full mode
  - Fundamental (RandomForest): Always full retrain (fast ~2 min)

Usage:
    python retrain.py --model technical --mode incremental
    python retrain.py --model technical --mode full --max_stocks 10
    python retrain.py --model fundamental
    python retrain.py --model all

Model versions are saved with timestamps and tracked in MongoDB.
==============================================================================
"""

import os
import sys
import json
import shutil
import argparse
import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TECH_DIR = os.path.join(BASE_DIR, "technical analysis")
FUND_DIR = os.path.join(BASE_DIR, "fundamental analysis")
VERSIONS_DIR = os.path.join(BASE_DIR, "model_versions")

# Ensure version directory exists
os.makedirs(VERSIONS_DIR, exist_ok=True)


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ==============================================================================
# TECHNICAL MODEL RETRAINING
# ==============================================================================

def retrain_technical(mode="incremental", max_stocks=None, epochs=None,
                      progress_callback=None):
    """
    Retrain the technical analysis LSTM+Attention ensemble.

    Args:
        mode: 'incremental' (fine-tune existing weights, fast) or 'full' (from scratch)
        max_stocks: limit number of stocks to process (for testing)
        epochs: override epoch count
        progress_callback: optional callable(stage, pct, msg) for progress updates

    Returns:
        dict with version_tag, metrics, and file paths
    """
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    # Import from existing training module
    sys.path.insert(0, TECH_DIR)
    from train_technical import (
        CONFIG, prepare_stock_features, calculate_targets,
        create_sequences, get_feature_columns, build_enhanced_model,
        ModelEnsemble, evaluate_model, DirectionAwareLoss,
        MarketContextFeatures
    )
    import yfinance as yf
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    version_tag = f"tech_v{_timestamp()}"
    _report = lambda stage, pct, msg: progress_callback(stage, pct, msg) if progress_callback else print(f"  [{stage}] {pct:.0%} — {msg}")

    _report("init", 0, f"Starting {mode} retraining for technical model...")

    # Override config for incremental mode
    if mode == "incremental":
        train_years = 2       # Only last 2 years for speed
        n_epochs = epochs or 10
    else:
        train_years = CONFIG['TRAIN_YEARS']  # Full 15 years
        n_epochs = epochs or CONFIG['EPOCHS']

    tickers = CONFIG['TICKERS'][:max_stocks] if max_stocks else CONFIG['TICKERS']

    # Phase 1: Download market data
    _report("data", 0.05, f"Downloading S&P 500 index data ({train_years}yr)...")
    market_df = None
    if CONFIG['ENABLE_MARKET_FEATURES']:
        market_df = yf.download(CONFIG['MARKET_INDEX'], period=f"{train_years}y",
                                interval='1d', progress=False)
        if isinstance(market_df.columns, pd.MultiIndex):
            market_df.columns = market_df.columns.get_level_values(0)

    # Phase 2: Process stocks
    _report("data", 0.10, f"Processing {len(tickers)} stocks...")
    stock_data = []
    for i, ticker in enumerate(tickers):
        pct = 0.10 + (i / len(tickers)) * 0.30
        _report("data", pct, f"Processing {ticker} ({i+1}/{len(tickers)})...")

        try:
            df = yf.download(ticker, period=f"{train_years}y", interval="1d", progress=False)
            if len(df) < 200:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Calculate features using the existing pipeline
            from train_technical import EnhancedIndicators
            df['RSI'] = EnhancedIndicators.get_rsi(df['Close'])
            df['ADX'] = EnhancedIndicators.get_adx(df['High'], df['Low'], df['Close'])
            df['NATR'] = EnhancedIndicators.get_natr(df['High'], df['Low'], df['Close'])
            df['OBV_Slope'] = EnhancedIndicators.get_obv_slope(df['Close'], df['Volume'])
            df['Dist_SMA'] = EnhancedIndicators.get_dist_sma(df['Close'])
            df['MACD'] = EnhancedIndicators.get_macd(df['Close'])
            df['ROC'] = EnhancedIndicators.get_roc(df['Close'])
            df['Vol_Ratio'] = EnhancedIndicators.get_volume_ratio(df['Volume'])
            df['BB_Position'] = EnhancedIndicators.get_bollinger_position(df['Close'])

            if CONFIG['ENABLE_MARKET_FEATURES'] and market_df is not None:
                market_aligned = market_df.reindex(df.index, method='ffill')
                market_features = MarketContextFeatures.calculate_market_features(market_aligned['Close'])
                df = pd.concat([df, market_features], axis=1)
                stock_returns = df['Close'].pct_change(20)
                market_returns = market_aligned['Close'].pct_change(20)
                df['relative_strength'] = stock_returns - market_returns
                stock_ret = df['Close'].pct_change()
                market_ret = market_aligned['Close'].pct_change()
                df['beta'] = stock_ret.rolling(60).cov(market_ret) / (market_ret.rolling(60).var() + 1e-8)

            df.dropna(inplace=True)
            if len(df) >= CONFIG['SEQ_LENGTH'] + 50:
                stock_data.append((ticker, df))
        except Exception as e:
            continue

    if len(stock_data) == 0:
        return {"error": "No valid stock data downloaded"}

    _report("data", 0.40, f"Processed {len(stock_data)}/{len(tickers)} stocks successfully")

    # Phase 3: Create sequences
    _report("sequences", 0.45, "Creating training sequences...")
    feature_cols = get_feature_columns()
    all_X, all_y = [], []

    for ticker, df in stock_data:
        targets_df = calculate_targets(df)
        df = pd.concat([df, targets_df], axis=1)
        df.dropna(inplace=True)

        features = df[feature_cols].values
        targets = df[[f'Target_Day_{i}' for i in range(1, CONFIG['MAX_HORIZON']+1)]].values
        X_stock, y_stock = create_sequences(features, targets, CONFIG['SEQ_LENGTH'])
        if len(X_stock) > 0:
            all_X.append(X_stock)
            all_y.append(y_stock)

    X_data = np.concatenate(all_X)
    y_data = np.concatenate(all_y)
    _report("sequences", 0.50, f"Created {len(X_data):,} sequences (shape: {X_data.shape})")

    # Phase 4: Scale features
    _report("scaling", 0.52, "Scaling features...")
    n_samples, seq_len, n_features = X_data.shape
    X_reshaped = X_data.reshape(-1, n_features)

    if mode == "incremental" and os.path.exists(CONFIG['SCALER_NAME']):
        # Reuse existing scaler for consistency
        scaler = joblib.load(CONFIG['SCALER_NAME'])
        _report("scaling", 0.54, "Reusing existing scaler")
    else:
        scaler = StandardScaler()
        split_idx = int(len(X_reshaped) * 0.8)
        scaler.fit(X_reshaped[:split_idx])

    X_scaled = scaler.transform(X_reshaped).reshape(n_samples, seq_len, n_features)

    # Phase 5: Train/Val split
    split_idx = int(len(X_scaled) * 0.8)
    gap = min(CONFIG['MAX_HORIZON'], len(X_scaled) - split_idx - 10)
    X_train = X_scaled[:split_idx]
    y_train = y_data[:split_idx]
    X_val = X_scaled[split_idx + gap:]
    y_val = y_data[split_idx + gap:]

    _report("training", 0.55, f"Training set: {len(X_train):,} | Val set: {len(X_val):,}")

    # Phase 6: Train models
    if mode == "incremental" and os.path.exists(CONFIG['ENSEMBLE_DIR']):
        # Load existing model weights & fine-tune
        _report("training", 0.58, "Loading existing ensemble for fine-tuning...")
        ensemble_models = []
        ensemble_info = joblib.load(CONFIG['ENSEMBLE_INFO'])
        old_weights = ensemble_info.get('weights', [])

        for i in range(CONFIG['N_ENSEMBLE']):
            model_path = os.path.join(CONFIG['ENSEMBLE_DIR'], f'model_{i+1}.keras')
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path, custom_objects={
                    'DirectionAwareLoss': DirectionAwareLoss
                })
                # Fine-tune with low learning rate
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss=DirectionAwareLoss(direction_weight=0.3),
                    metrics=['mae']
                )
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=0),
                ]

                pct = 0.60 + (i / CONFIG['N_ENSEMBLE']) * 0.25
                _report("training", pct, f"Fine-tuning model {i+1}/{CONFIG['N_ENSEMBLE']}...")

                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=n_epochs,
                    batch_size=CONFIG['BATCH_SIZE'],
                    callbacks=callbacks,
                    verbose=0
                )
                ensemble_models.append(model)
            else:
                _report("training", pct, f"Model {i+1} not found, building fresh...")
                model = build_enhanced_model(CONFIG['SEQ_LENGTH'], n_features, CONFIG['MAX_HORIZON'], seed=42+i)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=DirectionAwareLoss(direction_weight=0.3),
                    metrics=['mae']
                )
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
                ]
                model.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=n_epochs, batch_size=CONFIG['BATCH_SIZE'],
                          callbacks=callbacks, verbose=0)
                ensemble_models.append(model)

        # Recalculate weights
        weights = []
        for model in ensemble_models:
            val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
            weights.append(1.0 / (val_loss + 1e-6))
        weights = np.array(weights)
        weights = weights / weights.sum()

    else:
        # Full training from scratch
        _report("training", 0.58, "Training ensemble from scratch...")
        ensemble = ModelEnsemble(n_models=CONFIG['N_ENSEMBLE'])

        # Redirect ensemble training progress
        for i in range(CONFIG['N_ENSEMBLE']):
            pct = 0.58 + (i / CONFIG['N_ENSEMBLE']) * 0.27
            _report("training", pct, f"Training model {i+1}/{CONFIG['N_ENSEMBLE']}...")

        ensemble.train(X_train, y_train, X_val, y_val,
                       CONFIG['SEQ_LENGTH'], n_features, CONFIG['MAX_HORIZON'])
        ensemble_models = ensemble.models
        weights = ensemble.weights

    _report("saving", 0.87, "Saving retrained models...")

    # Save — overwrite current models
    os.makedirs(CONFIG['ENSEMBLE_DIR'], exist_ok=True)
    for i, model in enumerate(ensemble_models):
        model_path = os.path.join(CONFIG['ENSEMBLE_DIR'], f'model_{i+1}.keras')
        model.save(model_path)

    # Save scaler (only if full retrain)
    if mode == "full":
        joblib.dump(scaler, CONFIG['SCALER_NAME'])

    # Save ensemble info
    ensemble_info = {
        'n_models': CONFIG['N_ENSEMBLE'],
        'weights': weights.tolist() if isinstance(weights, np.ndarray) else weights,
        'model_dir': CONFIG['ENSEMBLE_DIR'],
        'market_index': CONFIG['MARKET_INDEX'],
        'tickers': [t for t, _ in stock_data],
        'feature_cols': feature_cols,
        'trained_at': datetime.now().isoformat(),
        'mode': mode,
        'version': version_tag,
    }
    joblib.dump(ensemble_info, CONFIG['ENSEMBLE_INFO'])

    # Also save to versioned directory
    version_dir = os.path.join(VERSIONS_DIR, version_tag)
    os.makedirs(version_dir, exist_ok=True)
    with open(os.path.join(version_dir, "info.json"), "w") as f:
        json.dump(ensemble_info, f, indent=2)

    # Phase 7: Evaluate
    _report("eval", 0.90, "Evaluating retrained model...")

    # Build a simple predictor wrapper
    class _EnsemblePredictor:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
        def predict(self, X):
            preds = np.array([m.predict(X, verbose=0) for m in self.models])
            return np.average(preds, axis=0, weights=self.weights)

    predictor = _EnsemblePredictor(ensemble_models, weights)
    metrics = evaluate_model(predictor, X_val, y_val)

    # Save metrics to versioned dir
    metrics_summary = {
        "version": version_tag,
        "mode": mode,
        "trained_at": datetime.now().isoformat(),
        "n_stocks": len(stock_data),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "epochs": n_epochs,
        "horizon_metrics": {
            k: {mk: round(mv, 4) for mk, mv in v.items()}
            for k, v in metrics.items()
        }
    }
    with open(os.path.join(version_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    _report("done", 1.0, f"✅ Technical model retrained → {version_tag}")

    return {
        "version": version_tag,
        "mode": mode,
        "metrics": metrics_summary,
        "path": version_dir,
    }


# ==============================================================================
# FUNDAMENTAL MODEL RETRAINING
# ==============================================================================

def retrain_fundamental(progress_callback=None):
    """
    Retrain the fundamental analysis RandomForest model.
    Always does a full retrain since RF is fast (~2 min).

    Returns:
        dict with version_tag, metrics, path
    """
    version_tag = f"fund_v{_timestamp()}"
    _report = lambda stage, pct, msg: progress_callback(stage, pct, msg) if progress_callback else print(f"  [{stage}] {pct:.0%} — {msg}")

    _report("init", 0, "Starting fundamental model retraining...")

    # Import from existing module
    sys.path.insert(0, FUND_DIR)
    from score import run_experiment, load_best_model, META_PATH, MODEL_PATH

    # Phase 1: Run the experiment (downloads fresh data + retrains)
    _report("data", 0.10, "Downloading fresh SEC EDGAR filings & retraining...")

    try:
        val_ic = run_experiment(tag=f"retrain_{version_tag}")
    except Exception as e:
        return {"error": f"Fundamental retraining failed: {e}"}

    _report("eval", 0.80, f"Retraining complete. Val IC = {val_ic:.4f}")

    # Phase 2: Load the saved model metadata
    meta = {}
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())

    # Phase 3: Save versioned copy
    version_dir = os.path.join(VERSIONS_DIR, version_tag)
    os.makedirs(version_dir, exist_ok=True)

    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, os.path.join(version_dir, "model.pkl"))
    if META_PATH.exists():
        shutil.copy2(META_PATH, os.path.join(version_dir, "meta.json"))

    metrics_summary = {
        "version": version_tag,
        "mode": "full",
        "trained_at": datetime.now().isoformat(),
        "val_ic": val_ic,
        "meta": meta,
    }
    with open(os.path.join(version_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2, default=str)

    _report("done", 1.0, f"✅ Fundamental model retrained → {version_tag}")

    return {
        "version": version_tag,
        "mode": "full",
        "val_ic": val_ic,
        "metrics": metrics_summary,
        "path": version_dir,
    }


# ==============================================================================
# LIST MODEL VERSIONS
# ==============================================================================

def list_model_versions():
    """List all saved model versions from the model_versions directory."""
    versions = []
    if not os.path.exists(VERSIONS_DIR):
        return versions

    for name in sorted(os.listdir(VERSIONS_DIR), reverse=True):
        version_dir = os.path.join(VERSIONS_DIR, name)
        if not os.path.isdir(version_dir):
            continue

        info_path = os.path.join(version_dir, "metrics.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            versions.append({
                "version": name,
                "type": "technical" if name.startswith("tech") else "fundamental",
                "mode": info.get("mode", "unknown"),
                "trained_at": info.get("trained_at", ""),
                "path": version_dir,
                "metrics": info,
            })
        else:
            versions.append({
                "version": name,
                "type": "technical" if name.startswith("tech") else "fundamental",
                "mode": "unknown",
                "trained_at": "",
                "path": version_dir,
                "metrics": {},
            })

    return versions


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain robo-advisory models on live data")
    parser.add_argument("--model", type=str, required=True,
                        choices=["technical", "fundamental", "all"],
                        help="Which model to retrain")
    parser.add_argument("--mode", type=str, default="incremental",
                        choices=["incremental", "full"],
                        help="Retraining mode (only for technical)")
    parser.add_argument("--max_stocks", type=int, default=None,
                        help="Limit stocks for testing")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f" ROBO-ADVISORY MODEL RETRAINING")
    print(f"{'='*70}")
    print(f"  Model:      {args.model}")
    print(f"  Mode:       {args.mode}")
    print(f"  Max stocks: {args.max_stocks or 'all'}")
    print(f"  Epochs:     {args.epochs or 'default'}")
    print(f"{'='*70}\n")

    results = {}

    if args.model in ("technical", "all"):
        print("\n--- TECHNICAL MODEL ---")
        results["technical"] = retrain_technical(
            mode=args.mode,
            max_stocks=args.max_stocks,
            epochs=args.epochs
        )

    if args.model in ("fundamental", "all"):
        print("\n--- FUNDAMENTAL MODEL ---")
        results["fundamental"] = retrain_fundamental()

    # Print summary
    print(f"\n{'='*70}")
    print(" RETRAINING COMPLETE")
    print(f"{'='*70}")
    for model_name, result in results.items():
        if "error" in result:
            print(f"  ❌ {model_name}: {result['error']}")
        else:
            print(f"  ✅ {model_name}: {result['version']}")
    print(f"{'='*70}\n")

    # Try to save to MongoDB
    try:
        from db import MongoDB
        db = MongoDB()
        if db.is_connected():
            for model_name, result in results.items():
                if "error" not in result:
                    db.save_model_version(
                        model_type=model_name,
                        version_tag=result["version"],
                        metrics=result.get("metrics", {}),
                        file_path=result.get("path", "")
                    )
                    print(f"  📦 {model_name} version saved to MongoDB")
    except Exception as e:
        print(f"  ⚠️  MongoDB save skipped: {e}")
