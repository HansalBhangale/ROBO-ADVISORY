"""
predict.py — Use the saved best model to score companies anytime.

Usage:
  uv run predict.py                        # score all companies
  uv run predict.py --ticker AAPL          # score a single company
  uv run predict.py --signal BUY           # filter to BUY signals only
  uv run predict.py --output signals.csv   # save results to CSV

This loads the model saved by score.py — no retraining, runs in seconds.
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from prepare import prepare_dataset, CACHE_DIR
from score import engineer_features, predictions_to_scores, score_to_signal, load_best_model

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_all(ticker_filter: str = None,
                signal_filter: str = None,
                output_path: str  = None) -> pd.DataFrame:
    """
    Load saved model, score every company on their latest 10-K filing,
    return a DataFrame with scores and BUY/SELL/HOLD signals.
    """

    # 1. Load saved model + metadata
    print("[predict] Loading saved model...")
    model, meta = load_best_model()

    print(f"[predict] Model: '{meta['tag']}'  |  val_ic: {meta['val_ic']:.4f}  |  saved: {meta['saved_at'][:10]}")
    feature_cols  = meta["feature_cols"]
    buy_threshold = meta["buy_threshold"]
    sell_threshold = meta["sell_threshold"]

    # 2. Load fundamental data (uses cache — instant if already downloaded)
    print("[predict] Loading fundamental data...")
    df = prepare_dataset()
    df = engineer_features(df)

    # 3. Keep only the MOST RECENT filing per company (latest 10-K)
    latest = (
        df.sort_values("filed_date")
          .groupby("ticker")
          .last()
          .reset_index()
    )

    # Optional: filter to a single ticker
    if ticker_filter:
        latest = latest[latest["ticker"].str.upper() == ticker_filter.upper()]
        if latest.empty:
            print(f"[predict] ✗ Ticker '{ticker_filter}' not found in dataset.")
            return pd.DataFrame()

    # 4. Score
    print(f"[predict] Scoring {len(latest)} companies...")
    X = latest[feature_cols].values
    raw_preds = model.predict(X)

    # Cross-sectional rank normalisation → 0-100
    latest = latest.copy()
    latest["score"]  = predictions_to_scores(raw_preds)
    latest["signal"] = latest["score"].apply(
        lambda s: "BUY" if s >= buy_threshold else ("SELL" if s < sell_threshold else "HOLD")
    )

    # 5. Build output table
    result = latest[[
        "ticker", "filed_date", "score", "signal",
        "rev_growth_1y", "gross_margin", "net_margin",
        "roe", "roa", "fcf_to_assets", "debt_to_equity",
    ]].copy()

    result = result.rename(columns={
        "rev_growth_1y":   "rev_growth",
        "gross_margin":    "gross_margin",
        "net_margin":      "net_margin",
        "fcf_to_assets":   "fcf_yield",
        "debt_to_equity":  "d_e_ratio",
    })

    result["filed_date"] = pd.to_datetime(result["filed_date"]).dt.date
    result["score"]      = result["score"].round(1)

    # Round ratio columns
    for col in ["rev_growth", "gross_margin", "net_margin", "roe", "roa", "fcf_yield", "d_e_ratio"]:
        result[col] = result[col].round(3)

    result = result.sort_values("score", ascending=False).reset_index(drop=True)

    # 6. Optional signal filter
    if signal_filter:
        result = result[result["signal"] == signal_filter.upper()]

    # 7. Print
    _print_results(result, signal_filter)

    # 8. Optional CSV export
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"\n[predict] ✓ Saved to {output_path}")

    return result


def _print_results(df: pd.DataFrame, signal_filter: str = None) -> None:
    """Pretty-print the signal table."""
    if df.empty:
        print("[predict] No results.")
        return

    total    = len(df)
    n_buy    = (df["signal"] == "BUY").sum()
    n_hold   = (df["signal"] == "HOLD").sum()
    n_sell   = (df["signal"] == "SELL").sum()

    print(f"\n{'─'*70}")
    if signal_filter:
        print(f"  {signal_filter} SIGNALS  ({total} companies)")
    else:
        print(f"  FULL UNIVERSE  ({total} companies)  —  BUY: {n_buy}  HOLD: {n_hold}  SELL: {n_sell}")
    print(f"{'─'*70}")

    # Always show top BUYs first
    show_df = df if signal_filter else df.head(30)
    print(show_df.to_string(index=False))

    if not signal_filter:
        print(f"\n  ... showing top 30 of {total}. Use --signal BUY/HOLD/SELL to filter.")

    print(f"{'─'*70}")
    print(f"  Score 0–100: BUY ≥ {65}  |  HOLD 35–64  |  SELL < 35")
    print(f"  Based on most recent 10-K filing per company.")
    print(f"{'─'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFO HELPER
# ─────────────────────────────────────────────────────────────────────────────

def show_model_info() -> None:
    """Print info about the currently saved model."""
    model_path = CACHE_DIR / "best_model.pkl"
    meta_path  = CACHE_DIR / "best_model_meta.json"

    if not meta_path.exists():
        print("[predict] No model saved yet. Run: uv run score.py --tag baseline")
        return

    meta = json.loads(meta_path.read_text())
    size_mb = model_path.stat().st_size / 1024 / 1024 if model_path.exists() else 0

    print(f"\n{'─'*50}")
    print(f"  Saved Model Info")
    print(f"{'─'*50}")
    print(f"  Tag        : {meta['tag']}")
    print(f"  val_ic     : {meta['val_ic']:.4f}")
    print(f"  Saved at   : {meta['saved_at'][:19]}")
    print(f"  Features   : {len(meta['feature_cols'])} features")
    print(f"  BUY  ≥     : {meta['buy_threshold']}")
    print(f"  SELL <     : {meta['sell_threshold']}")
    print(f"  File size  : {size_mb:.1f} MB")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score companies using the saved best model."
    )
    parser.add_argument("--ticker",  type=str, default=None,
                        help="Score a single ticker, e.g. --ticker AAPL")
    parser.add_argument("--signal",  type=str, default=None, choices=["BUY", "HOLD", "SELL"],
                        help="Filter output to one signal type")
    parser.add_argument("--output",  type=str, default=None,
                        help="Save results to CSV, e.g. --output signals.csv")
    parser.add_argument("--info",    action="store_true",
                        help="Show info about the currently saved model")
    args = parser.parse_args()

    if args.info:
        show_model_info()
    else:
        predict_all(
            ticker_filter=args.ticker,
            signal_filter=args.signal,
            output_path=args.output,
        )
