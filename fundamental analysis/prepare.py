"""
prepare.py — FIXED constants, SEC EDGAR data pipeline, and evaluation utilities.
DO NOT MODIFY THIS FILE. The agent only modifies score.py.

This file handles:
  1. Downloading fundamental data from SEC EDGAR XBRL API
  2. Downloading forward price returns from Yahoo Finance
  3. Building a point-in-time dataset (no lookahead bias)
  4. Defining evaluate_ic() — the single metric used to rank experiments
"""

import os
import json
import time
import pickle
import requests
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# FIXED CONSTANTS — do not change these
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR     = Path(os.environ.get("FINRESEARCH_CACHE", Path.home() / ".cache" / "finresearch"))
DATA_PATH     = CACHE_DIR / "dataset.parquet"
TICKERS_PATH  = CACHE_DIR / "tickers.json"
RESULTS_FILE  = "results.tsv"

# Evaluation window: score companies using filings up to DATE_CUTOFF,
# measure actual return over the FORWARD_MONTHS after each filing.
FORWARD_MONTHS   = 12       # how many months ahead to measure return
MIN_HISTORY_YRS  = 5        # minimum years of filings to include a company
MIN_COMPANIES    = 10       # abort if fewer companies survive filtering
EDGAR_SLEEP_SEC  = 0.12     # SEC rate limit: max ~10 req/s, we stay safe

# Train/val split — last VAL_RATIO of filings (by date) are held out
VAL_RATIO = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# EDGAR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "finresearch-autoresearch contact@finresearch.local",
    "Accept-Encoding": "gzip, deflate",
}

def get_sp500_tickers() -> dict[str, str]:
    """Return {ticker: CIK} for S&P 500 companies using SEC's company_tickers JSON."""
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    raw = r.json()
    # raw is {idx: {cik_str, ticker, title}}
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in raw.values()}


def get_company_facts(cik: str) -> dict | None:
    """Fetch all XBRL company facts for a given CIK."""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# XBRL concept aliases — we try multiple names per metric for robustness
XBRL_CONCEPTS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "operating_cashflow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "rd_expense": ["ResearchAndDevelopmentExpense"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "shares_outstanding": ["CommonStockSharesOutstanding"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "interest_expense": ["InterestExpense"],
    "eps_basic": ["EarningsPerShareBasic"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
}


def extract_annual_series(facts: dict, concept_aliases: list[str]) -> pd.Series:
    """
    Extract annual (10-K) time series for a concept from company facts.
    Returns a pd.Series indexed by filing date (the 'filed' date for PIT correctness).
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concept_aliases:
        if concept not in us_gaap:
            continue
        units = us_gaap[concept].get("units", {})
        # Prefer USD, fallback to shares
        for unit_key in ("USD", "shares"):
            if unit_key not in units:
                continue
            entries = units[unit_key]
            # Filter to annual (10-K) filings with valid data
            annual = [
                e for e in entries
                if e.get("form") in ("10-K", "10-K/A")
                and e.get("val") is not None
                and e.get("filed") is not None
                and e.get("end") is not None
            ]
            if not annual:
                continue
            # Use 'filed' date as the point-in-time key (when data became public)
            df = pd.DataFrame(annual)[["filed", "end", "val"]].copy()
            df["filed"] = pd.to_datetime(df["filed"])
            df["end"] = pd.to_datetime(df["end"])
            # Keep only the latest filing per fiscal year end
            df = df.sort_values("filed").drop_duplicates(subset="end", keep="last")
            return pd.Series(df["val"].values, index=df["filed"].values, name=concept)
    return pd.Series(dtype=float)


def build_fundamental_rows(ticker: str, cik: str) -> list[dict]:
    """
    For a single company, build one row per 10-K filing with:
      - Raw financials (point-in-time, as of filing date)
      - Computed ratios
    Returns list of dicts, one per annual filing.
    """
    facts = get_company_facts(cik)
    if facts is None:
        return []

    series = {}
    for metric, aliases in XBRL_CONCEPTS.items():
        series[metric] = extract_annual_series(facts, aliases)

    if series["revenue"].empty or series["total_assets"].empty:
        return []

    # Align all series to a common filing date index
    all_dates = sorted(set().union(*[s.index for s in series.values() if not s.empty]))
    if len(all_dates) < MIN_HISTORY_YRS:
        return []

    rows = []
    for i, filed_date in enumerate(all_dates):
        def last_val(s: pd.Series, n_back: int = 0):
            """Get value from n_back periods ago relative to current index."""
            past = [d for d in s.index if d <= filed_date]
            if len(past) <= n_back:
                return np.nan
            target_date = past[-(1 + n_back)]
            # Use positional index to avoid duplicate-label ambiguity
            positions = [j for j, d in enumerate(s.index) if d == target_date]
            val = s.iloc[positions[-1]]  # take last if duplicates
            if isinstance(val, (pd.Series, np.ndarray)):
                val = val.iloc[-1] if hasattr(val, 'iloc') else val[-1]
            return float(val)

        rev   = last_val(series["revenue"])
        rev1  = last_val(series["revenue"], 1)
        rev2  = last_val(series["revenue"], 2)
        ni    = last_val(series["net_income"])
        ta    = last_val(series["total_assets"])
        tl    = last_val(series["total_liabilities"])
        eq    = last_val(series["equity"])
        ocf   = last_val(series["operating_cashflow"])
        capex = last_val(series["capex"])
        gp    = last_val(series["gross_profit"])
        oi    = last_val(series["operating_income"])
        ltd   = last_val(series["long_term_debt"])
        ie    = last_val(series["interest_expense"])
        rd    = last_val(series["rd_expense"])

        if pd.isna(rev) or rev <= 0 or pd.isna(ta) or ta <= 0:
            continue

        row = {
            "ticker":         ticker,
            "filed_date":     filed_date,
            # Raw fundamentals (normalised by assets for scale-invariance)
            "rev_to_assets":  rev / ta,
            "ni_to_assets":   ni / ta if not pd.isna(ni) else np.nan,
            "eq_to_assets":   eq / ta if not pd.isna(eq) else np.nan,
            "ocf_to_assets":  ocf / ta if not pd.isna(ocf) else np.nan,
            "ltd_to_assets":  ltd / ta if not pd.isna(ltd) and not pd.isna(ltd) else np.nan,
            # Growth
            "rev_growth_1y":  (rev / rev1 - 1) if not pd.isna(rev1) and rev1 > 0 else np.nan,
            "rev_growth_2y":  (rev / rev2 - 1) if not pd.isna(rev2) and rev2 > 0 else np.nan,
            # Profitability
            "gross_margin":   gp / rev if not pd.isna(gp) else np.nan,
            "op_margin":      oi / rev if not pd.isna(oi) else np.nan,
            "net_margin":     ni / rev if not pd.isna(ni) else np.nan,
            "roe":            ni / eq if not pd.isna(ni) and not pd.isna(eq) and eq > 0 else np.nan,
            "roa":            ni / ta if not pd.isna(ni) else np.nan,
            # Cash quality
            "fcf_to_assets":  (ocf - (capex or 0)) / ta if not pd.isna(ocf) else np.nan,
            "accruals":       (ni - ocf) / ta if not pd.isna(ni) and not pd.isna(ocf) else np.nan,
            # Leverage
            "debt_to_equity": ltd / eq if not pd.isna(ltd) and not pd.isna(eq) and eq > 0 else np.nan,
            "interest_cover": oi / abs(ie) if not pd.isna(oi) and not pd.isna(ie) and ie != 0 else np.nan,
            # R&D intensity
            "rd_to_rev":      rd / rev if not pd.isna(rd) else np.nan,
        }
        rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# PRICE / FORWARD RETURN UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_forward_return(ticker: str, from_date: pd.Timestamp, months: int = 12) -> float | None:
    """
    Compute the total return for `ticker` from `from_date` to `from_date + months`.
    Uses Yahoo Finance via yfinance. Returns None if data unavailable.
    """
    try:
        import yfinance as yf
        start = from_date + timedelta(days=5)
        end   = start + timedelta(days=months * 31)

        # Skip if forward window hasn't completed yet (future dates)
        if end > pd.Timestamp.now():
            return None

        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True
        )

        # Flatten MultiIndex columns (yfinance >= 0.2.40)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        if hist.empty or len(hist) < 20:
            return None

        # Find Close column case-insensitively
        close_col = next((c for c in hist.columns if str(c).lower() == "close"), None)
        if close_col is None:
            return None

        p0 = float(hist[close_col].iloc[0])
        p1 = float(hist[close_col].iloc[-1])
        if p0 <= 0:
            return None
        return (p1 / p0) - 1.0
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PREPARATION  (run once, cached to disk)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(max_companies: int = 200, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build and cache the full point-in-time fundamental dataset.

    Columns: ticker, filed_date, [features...], forward_return_12m
    This is the ground truth; score.py should not modify this.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists() and not force_rebuild:
        print(f"[prepare] Loading cached dataset from {DATA_PATH}")
        return pd.read_parquet(DATA_PATH)

    print("[prepare] Building dataset from SEC EDGAR + Yahoo Finance...")
    print("[prepare] This takes ~30-60 min on first run. Grab a coffee ☕")

    # 1. Get ticker→CIK mapping
    print("[prepare] Fetching ticker→CIK map from SEC...")
    all_tickers = get_sp500_tickers()
    tickers = dict(list(all_tickers.items())[:max_companies])
    print(f"[prepare] Processing {len(tickers)} companies")

    # 2. Download fundamentals from EDGAR
    all_rows = []
    for i, (ticker, cik) in enumerate(tickers.items()):
        if i % 20 == 0:
            print(f"[prepare]   EDGAR fundamentals: {i}/{len(tickers)}")
        rows = build_fundamental_rows(ticker, cik)
        all_rows.extend(rows)
        time.sleep(EDGAR_SLEEP_SEC)

    if not all_rows:
        raise RuntimeError("[prepare] No rows collected — check network access to SEC EDGAR")

    df = pd.DataFrame(all_rows)
    df["filed_date"] = pd.to_datetime(df["filed_date"])
    print(f"[prepare] Collected {len(df)} filing rows across {df['ticker'].nunique()} companies")

    # 3. Fetch forward returns (this is slow — one yfinance call per row)
    print("[prepare] Fetching forward returns from Yahoo Finance...")
    returns = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"[prepare]   Forward returns: {i}/{len(df)}")
        ret = get_forward_return(row["ticker"], row["filed_date"], FORWARD_MONTHS)
        returns.append(ret)

    df["forward_return_12m"] = returns

    # 4. Drop rows with missing labels
    before = len(df)
    df = df.dropna(subset=["forward_return_12m"])
    print(f"[prepare] Dropped {before - len(df)} rows with missing forward returns")

    # Winsorize returns at 1st/99th percentile to reduce outlier influence
    lo, hi = df["forward_return_12m"].quantile([0.01, 0.99])
    df["forward_return_12m"] = df["forward_return_12m"].clip(lo, hi)

    n_companies = df["ticker"].nunique()
    if n_companies < MIN_COMPANIES:
        print(f"[prepare] WARNING: Only {n_companies} companies with valid forward returns.")
        print(f"[prepare] This usually means yfinance failed or all filings are too recent (< {FORWARD_MONTHS} months old).")
        print(f"[prepare] Try: pip install --upgrade yfinance")
        if n_companies == 0:
            raise RuntimeError("[prepare] Zero companies survived — check your internet connection and yfinance version.")

    df = df.sort_values("filed_date").reset_index(drop=True)
    df.to_parquet(DATA_PATH, index=False)
    print(f"[prepare] Dataset saved: {len(df)} rows, {df['ticker'].nunique()} companies → {DATA_PATH}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes metadata and label)."""
    exclude = {"ticker", "filed_date", "forward_return_12m"}
    return [c for c in df.columns if c not in exclude]


def train_val_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train/val split — val set is the most recent VAL_RATIO of dates."""
    cutoff_idx = int(len(df) * (1 - VAL_RATIO))
    return df.iloc[:cutoff_idx].copy(), df.iloc[cutoff_idx:].copy()


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  — THE SINGLE METRIC
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ic(scores: np.ndarray, returns: np.ndarray) -> float:
    """
    Compute the Rank Information Coefficient (Spearman correlation) between
    predicted scores and actual forward returns.

    IC = Spearman(scores, forward_returns)

    Range: -1 to +1. Higher is better. IC > 0.05 is considered meaningful in practice.
    This is the canonical metric for cross-sectional factor models.
    """
    mask = ~np.isnan(scores) & ~np.isnan(returns)
    if mask.sum() < 10:
        return -1.0
    ic, _ = spearmanr(scores[mask], returns[mask])
    return float(ic) if not np.isnan(ic) else -1.0


def evaluate_long_short_sharpe(scores: np.ndarray, returns: np.ndarray,
                                 top_pct: float = 0.2, bot_pct: float = 0.2) -> float:
    """
    Simulate a long/short portfolio: long top_pct, short bottom_pct of scores.
    Returns annualised Sharpe ratio (assumes monthly rebalancing, 12 periods/yr).
    Secondary metric — not used for keep/discard but logged for context.
    """
    mask = ~np.isnan(scores) & ~np.isnan(returns)
    s, r = scores[mask], returns[mask]
    if len(s) < 20:
        return 0.0
    n = len(s)
    top_n = max(1, int(n * top_pct))
    bot_n = max(1, int(n * bot_pct))
    top_idx = np.argsort(s)[-top_n:]
    bot_idx = np.argsort(s)[:bot_n]
    long_ret  = r[top_idx].mean()
    short_ret = r[bot_idx].mean()
    ls_ret = long_ret - short_ret
    # Very rough Sharpe: assume std of single observation ~30% annualised
    return float(ls_ret / 0.30) if ls_ret != 0 else 0.0


def evaluate_hit_rate(scores: np.ndarray, returns: np.ndarray, top_pct: float = 0.3) -> float:
    """Fraction of top-scored companies that beat median return."""
    mask = ~np.isnan(scores) & ~np.isnan(returns)
    s, r = scores[mask], returns[mask]
    if len(s) < 10:
        return 0.5
    top_n = max(1, int(len(s) * top_pct))
    top_idx = np.argsort(s)[-top_n:]
    median_ret = np.median(r)
    return float((r[top_idx] > median_ret).mean())


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_result(tag: str, val_ic: float, train_ic: float,
               sharpe: float, hit_rate: float, note: str = "") -> None:
    """Append one experiment result to results.tsv."""
    header = not Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a") as f:
        if header:
            f.write("timestamp\ttag\tval_ic\ttrain_ic\tsharpe\thit_rate\tnote\n")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts}\t{tag}\t{val_ic:.6f}\t{train_ic:.6f}\t{sharpe:.4f}\t{hit_rate:.4f}\t{note}\n")
    print(f"[result] tag={tag}  val_ic={val_ic:.4f}  train_ic={train_ic:.4f}  "
          f"sharpe={sharpe:.3f}  hit_rate={hit_rate:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run once to prepare data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_companies", type=int, default=200)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    df = prepare_dataset(max_companies=args.max_companies, force_rebuild=args.force_rebuild)
    print(f"\n[prepare] ✓ Dataset ready: {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"[prepare]   Date range: {df['filed_date'].min().date()} → {df['filed_date'].max().date()}")
    print(f"[prepare]   Features: {get_feature_cols(df)}")
    print(f"[prepare]   Forward return stats:\n{df['forward_return_12m'].describe()}")
    print(f"\nNow run:  python score.py")