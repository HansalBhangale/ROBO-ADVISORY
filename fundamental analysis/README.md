# finresearch

**Autoresearch-style autonomous ML loop for fundamental stock analysis.**

Inspired directly by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
Same loop, different domain: instead of improving an LLM's `val_bpb`,
we improve a fundamental scoring model's `val_ic` (Information Coefficient).

```
The human writes program.md.
The agent iterates on score.py.
You wake up to a better model and a BUY/SELL/HOLD signal for every company.
```

---

## Architecture (direct mapping from autoresearch)

| autoresearch          | finresearch                  |
| --------------------- | ---------------------------- |
| `prepare.py`        | `prepare.py` (fixed)       |
| `train.py`          | `score.py` (agent edits)   |
| `program.md`        | `program.md` (human edits) |
| metric:`val_bpb` ↓ | metric:`val_ic` ↑         |
| 5-min time budget     | ~2-min experiment budget     |
| GPU compute           | CPU + free APIs              |
| LLM quality           | Stock picking quality        |

---

## Data Sources

- **Fundamentals**: [SEC EDGAR XBRL API](https://www.sec.gov/developer) — free, no key needed, 30+ years
- **Forward Returns**: [Yahoo Finance](https://finance.yahoo.com) via `yfinance` — free
- **Coverage**: US public companies (S&P 500 focused)

---

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data from SEC EDGAR (one-time, ~30-60 min)
uv run prepare.py --max_companies 200

# 4. Run baseline experiment
uv run score.py --tag baseline

# 5. Start autonomous research mode
# Point Claude Code (or any agent) at program.md and let it go:
# "Read program.md and score.py. Run the baseline, then start the research loop."
```

---

## Running the Agent

Open Claude Code (or any AI coding agent) in this directory and prompt:

```
Read program.md and score.py. Let's kick off a new experiment session.
Run the baseline first, then start the autonomous research loop.
For each experiment: state your hypothesis, make one change, run it,
commit if improved, revert if worse.
```

The agent will run ~15-20 experiments per hour (2-min each), trying different
feature engineering, model architectures, and scoring strategies.

---

## Output

After each experiment, `results.tsv` grows:

```
timestamp            tag                 val_ic   train_ic  sharpe  hit_rate
2025-01-15 02:31:00  baseline            0.0312   0.0891    0.221   0.531
2025-01-15 02:33:00  log_transforms      0.0398   0.0912    0.287   0.548
2025-01-15 02:35:00  piotroski_features  0.0521   0.0967    0.341   0.572  ← saved
2025-01-15 02:37:00  xgb_model           0.0489   0.1102    0.312   0.561  ← reverted
...
```

The best model is **automatically saved** to `~/.cache/finresearch/best_model.pkl`.
Every time a new experiment beats the previous best `val_ic`, the model is retrained
on the full dataset and saved — so you always have the best model ready.

---

## Using the Saved Model (predict.py)

Once a model is saved you never need to retrain to get signals:

```bash
# Score all companies (shows top 30)
uv run predict.py

# Only show BUY signals
uv run predict.py --signal BUY

# Score a single stock
uv run predict.py --ticker AAPL

# Export full universe to CSV
uv run predict.py --output signals.csv

# Check saved model info
uv run predict.py --info
```

---

## File Structure

```
prepare.py     — FIXED. SEC EDGAR pipeline + evaluation functions. Do not modify.
score.py       — AGENT'S FILE. Feature engineering + model + scoring. Agent modifies.
predict.py     — YOUR FILE. Load saved model, get BUY/SELL/HOLD anytime. Never retrain.
program.md     — HUMAN'S FILE. Agent instructions. Human iterates.
pyproject.toml — Dependencies.
results.tsv    — Auto-generated experiment log.

~/.cache/finresearch/
  dataset.parquet        ← cached SEC EDGAR + forward returns data
  best_model.pkl         ← best trained model (auto-saved by score.py)
  best_model_meta.json   ← val_ic, features, thresholds, timestamp
```

---

## Evaluation Metric

**`val_ic`** = Spearman rank correlation between model scores and actual 12-month forward returns, on the held-out validation set (most recent 25% of filings).

| IC         | Interpretation              |
| ---------- | --------------------------- |
| < 0.0      | Worse than random           |
| 0.0–0.03  | Noise                       |
| 0.03–0.05 | Weak signal                 |
| 0.05–0.10 | Useful (keep this)          |
| > 0.10     | Strong (publication-worthy) |

---

## License

MIT
