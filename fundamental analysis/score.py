"""
score.py — THE FILE THE AGENT MODIFIES.

This is the scoring model for fundamental analysis.
The agent edits feature engineering, model architecture, and scoring logic.
The human does NOT touch this file — only program.md.

Evaluation metric: val_ic (Spearman Information Coefficient on held-out data)
  - Range: -1 to +1. Higher is better.
  - IC > 0.03 → weak signal
  - IC > 0.05 → useful signal
  - IC > 0.10 → strong signal (rare, publication-worthy)

Output: Each company filing gets a score 0–100:
  - BUY  : score >= 65
  - HOLD : 35 <= score < 65
  - SELL : score < 35
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Import fixed utilities from prepare.py
from prepare import (
    prepare_dataset, get_feature_cols, train_val_split,
    evaluate_ic, evaluate_long_short_sharpe, evaluate_hit_rate, log_result,
    CACHE_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERSISTENCE PATHS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH    = CACHE_DIR / "best_model.pkl"         # trained sklearn pipeline
META_PATH     = CACHE_DIR / "best_model_meta.json"   # val_ic, tag, feature cols, thresholds

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  ← Agent can add / remove / transform features here
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw fundamentals into model-ready features.
    Agent can add new derived features, interaction terms, or transformations.

    Currently: identity pass-through (use raw ratio features as-is).
    Ideas to try:
      - Log transforms for skewed ratios
      - Interaction terms (e.g., rev_growth * gross_margin)
      - Momentum: change in margin YoY (need lagged features)
      - Piotroski F-score components
      - Beneish M-score (earnings manipulation)
      - Cross-sectional rank-normalization per year
    """
    out = df.copy()

    # --- Add log-transform of revenue growth and debt_to_equity ---
    out["log_rev_growth_1y"] = np.log1p(out["rev_growth_1y"].clip(-0.9, None))
    out["log_rev_growth_2y"] = np.log1p(out["rev_growth_2y"].clip(-0.9, None))
    out["log_debt_to_equity"] = np.log1p(out["debt_to_equity"].clip(0, None))

    # --- Example: Quality composite ---
    # out["quality"] = out[["roe", "roa", "gross_margin", "fcf_to_assets"]].mean(axis=1)

    # --- Example: Piotroski-style profitability signals (binary) ---
    # out["pos_roa"]     = (out["roa"] > 0).astype(float)
    # out["pos_ocf"]     = (out["ocf_to_assets"] > 0).astype(float)
    # out["accrual_ok"]  = (out["accruals"] < 0).astype(float)   # cash earnings > accruals

    return out


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION  ← Agent can swap model class and hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

def build_model() -> Pipeline:
    """
    Return an sklearn Pipeline: imputer → scaler → regressor.
    The pipeline predicts forward returns from features; we then rank-normalize
    predictions into a 0–100 score.

    Agent: try different regressors or hyperparameters:
      - GradientBoostingRegressor (current)
      - RandomForestRegressor
      - Ridge / Lasso (linear models are surprisingly competitive)
      - XGBRegressor (pip install xgboost)
      - LGBMRegressor (pip install lightgbm)
      - Custom weighted factor score (no ML, pure fundamental model)
    """
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
        ("model",   regressor),
    ])
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# SCORING  ← Agent can modify scoring thresholds or scoring formula
# ─────────────────────────────────────────────────────────────────────────────

# Score thresholds for BUY / HOLD / SELL signals
BUY_THRESHOLD  = 65   # score >= this → BUY
SELL_THRESHOLD = 35   # score <  this → SELL


def predictions_to_scores(raw_preds: np.ndarray) -> np.ndarray:
    """
    Convert raw model predictions (forward return estimates) to 0–100 scores
    via cross-sectional rank normalization.

    This ensures the score is relative (percentile rank), not absolute,
    which is standard practice in factor investing.
    Agent: can try percentile bins, sigmoid transforms, etc.
    """
    ranks = pd.Series(raw_preds).rank(pct=True)
    scores = (ranks * 100).clip(0, 100)
    return scores.values


def score_to_signal(score: float) -> str:
    """Map a 0–100 score to BUY / HOLD / SELL."""
    if score >= BUY_THRESHOLD:
        return "BUY"
    elif score < SELL_THRESHOLD:
        return "SELL"
    else:
        return "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION LOOP  ← Agent can modify training strategy
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, feature_cols: list, val_ic: float, tag: str) -> None:
    """Persist the model and its metadata to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {
        "tag":           tag,
        "val_ic":        val_ic,
        "feature_cols":  feature_cols,
        "buy_threshold": BUY_THRESHOLD,
        "sell_threshold": SELL_THRESHOLD,
        "saved_at":      pd.Timestamp.now().isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[score] ✓ Model saved → {MODEL_PATH}")
    print(f"[score] ✓ Metadata  → {META_PATH}")


def load_best_model():
    """Load the best saved model and its metadata. Raises if none saved yet."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}. Run score.py first."
        )
    model = joblib.load(MODEL_PATH)
    meta  = json.loads(META_PATH.read_text())
    return model, meta


def run_experiment(tag: str = "baseline") -> float:
    """
    Main experiment function.
    1. Load data
    2. Engineer features
    3. Train model on train split
    4. Score val split
    5. Compute IC and log result
    6. Save model if it beats the previous best
    Returns val_ic (the primary metric).
    """
    print(f"\n[score] ── Experiment: {tag} ──")

    # Load dataset
    df = prepare_dataset()
    df = engineer_features(df)

    feature_cols = get_feature_cols(df)
    print(f"[score] Features ({len(feature_cols)}): {feature_cols}")

    train_df, val_df = train_val_split(df)
    print(f"[score] Train: {len(train_df)} rows | Val: {len(val_df)} rows")

    X_train = train_df[feature_cols].values
    y_train = train_df["forward_return_12m"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["forward_return_12m"].values

    # Train on full training set
    model = build_model()
    model.fit(X_train, y_train)

    # Score
    train_raw    = model.predict(X_train)
    val_raw      = model.predict(X_val)
    train_scores = predictions_to_scores(train_raw)
    val_scores   = predictions_to_scores(val_raw)

    # Evaluate
    train_ic = evaluate_ic(train_scores, y_train)
    val_ic   = evaluate_ic(val_scores,   y_val)
    sharpe   = evaluate_long_short_sharpe(val_scores, y_val)
    hit_rate = evaluate_hit_rate(val_scores, y_val)

    log_result(tag, val_ic, train_ic, sharpe, hit_rate)

    # ── Save if this is the best model so far ──────────────────────────────
    prev_best_ic = -999.0
    if META_PATH.exists():
        prev_best_ic = json.loads(META_PATH.read_text()).get("val_ic", -999.0)

    if val_ic > prev_best_ic:
        print(f"[score] New best! val_ic {prev_best_ic:.4f} → {val_ic:.4f}. Saving model.")
        # Retrain on ALL data before saving (maximise signal for production use)
        full_model = build_model()
        full_model.fit(
            df[feature_cols].values,
            df["forward_return_12m"].values,
        )
        save_model(full_model, feature_cols, val_ic, tag)
    else:
        print(f"[score] val_ic {val_ic:.4f} did not beat best {prev_best_ic:.4f}. Model not saved.")

    # Print signal distribution
    signals = [score_to_signal(s) for s in val_scores]
    buys  = signals.count("BUY")
    holds = signals.count("HOLD")
    sells = signals.count("SELL")
    print(f"[score] Val signals — BUY: {buys} | HOLD: {holds} | SELL: {sells}")

    # Show top 10 BUY candidates from most recent val filings
    val_df = val_df.copy()
    val_df["score"]  = val_scores
    val_df["signal"] = signals
    recent  = val_df.sort_values("filed_date", ascending=False).head(200)
    top_buys = recent[recent["signal"] == "BUY"].nlargest(10, "score")[
        ["ticker", "filed_date", "score", "signal", "forward_return_12m"]
    ]
    if not top_buys.empty:
        print("\n[score] Top BUY candidates (most recent filings):")
        print(top_buys.to_string(index=False))

    return val_ic


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="baseline",
                        help="Experiment identifier (used in results.tsv)")
    args = parser.parse_args()

    val_ic = run_experiment(tag=args.tag)

    print(f"\n[score] ✓ val_ic = {val_ic:.6f}")
    # Exit code 0 if improved over naive baseline (IC=0), 1 otherwise
    sys.exit(0 if val_ic > 0 else 1)
