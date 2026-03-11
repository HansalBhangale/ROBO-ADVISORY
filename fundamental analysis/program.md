# program.md — Agent Instructions for finresearch

> **You are an autonomous research agent improving a fundamental analysis scoring model.**
> The human writes this file. You write `score.py`. Do not modify `prepare.py`.

---

## What You Are Doing

You are iterating on `score.py` to improve a financial factor model that scores public
companies on their fundamental quality and predicts whether they are **BUY / HOLD / SELL**.

The model is trained on SEC EDGAR fundamental data and evaluated against actual
12-month forward stock returns. Your single goal is to **maximise `val_ic`** (Spearman
Information Coefficient on the held-out validation set). IC > 0 means your scores
have predictive power. IC > 0.05 is useful. IC > 0.10 is excellent.

---

## File Structure

```
prepare.py   — FIXED. Data pipeline, evaluation functions. DO NOT MODIFY.
score.py     — YOUR DOMAIN. Model, features, scoring logic. MODIFY FREELY.
program.md   — HUMAN'S DOMAIN. These instructions. Read-only for you.
results.tsv  — Auto-generated experiment log.
```

---

## The Research Loop

### Phase 1: Setup (once per session)

1. Read `prepare.py` fully — understand the dataset structure, `evaluate_ic()`, and `log_result()`.
2. Read `score.py` fully — understand the current baseline.
3. Create a git branch: `git checkout -b finresearch/<today_date>`
4. Run the baseline: `python score.py --tag baseline`
5. Record the baseline val_ic in your notes. This is your "score to beat".

### Phase 2: Autonomous Experimentation Loop (repeat indefinitely)

For each experiment:

1. **Hypothesize**: Pick ONE change to make. Examples below.
2. **Modify** `score.py` — keep the change small and reviewable.
3. **Run**: `python score.py --tag <short_descriptive_tag>`
4. **Evaluate**: Check val_ic vs current best.
   - If **improved**: `git add score.py && git commit -m "improve: <tag> val_ic=X.XXXX"`
   - If **worse**: `git checkout score.py`  (revert, discard the change)
5. Repeat.

---

## What to Try (Ideas for the Agent)

Work through these roughly in order. Commit each improvement separately.

### Feature Engineering (`engineer_features` function)

```
[ ] Log-transform skewed features (rev_growth, debt_to_equity)
[ ] Add Piotroski F-Score: 9 binary signals (profitability, leverage, efficiency)
[ ] Add Beneish M-Score: 8 ratios detecting earnings manipulation
[ ] Interaction terms: rev_growth_1y × gross_margin (quality growth composite)
[ ] Momentum of margins: gross_margin change vs. prior year (need lag engineering)
[ ] Cross-sectional rank features: rank each metric within the same filing year
[ ] Operating leverage: % change in operating income / % change in revenue
[ ] Asset turnover: revenue / total_assets
[ ] Cash conversion: operating_cashflow / net_income (earnings quality)
[ ] Altman Z-Score components for financial distress detection
```

### Model Architecture (`build_model` function)

```
[ ] Try Ridge regression (linear baseline — often competitive)
[ ] Try RandomForestRegressor with 200+ trees
[ ] Try XGBRegressor (pip install xgboost) with tuned depth
[ ] Try LGBMRegressor (pip install lightgbm) — fast and strong
[ ] Try stacked ensemble: linear model + tree model averaged
[ ] Try ranking loss (LambdaRank) instead of regression
[ ] Tune GBM: n_estimators, learning_rate, max_depth, subsample
[ ] Add feature selection: remove features with near-zero importance
```

### Scoring Logic (`predictions_to_scores`, `BUY_THRESHOLD`, `SELL_THRESHOLD`)

```
[ ] Try sector-relative scoring (rank within GICS sector instead of universe)
[ ] Try decile-based scoring instead of percentile rank
[ ] Adjust BUY_THRESHOLD and SELL_THRESHOLD to optimize precision
[ ] Add conviction weighting: score spread from median as confidence signal
```

### Training Strategy

```
[ ] Try expanding window cross-validation instead of single split
[ ] Try time-series purging (gap between train/val to prevent leakage)
[ ] Undersample periods with extreme market returns (outlier robustness)
[ ] Add sample weighting: weight recent filings more heavily
```

---

## Rules

1. **Only modify `score.py`** — never `prepare.py`.
2. **One change per experiment** — keep diffs small and reviewable.
3. **Always run the full script** — never partially evaluate.
4. **Always commit improvements, always revert failures** — git is the ratchet.
5. **Do not overfit to val_ic** — if train_ic >> val_ic by large margin, note it.
6. **If an experiment errors**, fix the bug, don't count it as a result.
7. **Log a note** in the `tag` argument describing what changed.

---

## Metrics Reference

| Metric      | Target     | Meaning |
|-------------|------------|---------|
| `val_ic`    | > 0.05     | Primary: Spearman rank correlation, scores vs. 12m returns |
| `train_ic`  | < val_ic + 0.05 | Sanity check: gap > 0.05 suggests overfitting |
| `sharpe`    | > 0.5      | Long-short portfolio Sharpe ratio (secondary) |
| `hit_rate`  | > 0.55     | Fraction of top-scored companies beating median return |

---

## Baseline Performance

> Fill this in after running `python score.py --tag baseline` for the first time.

```
Baseline val_ic  : 0.052810
Baseline train_ic: 0.500950
Baseline sharpe  : 0.3011
Date             : 2026-03-10
```

---

## Example Agent Prompt (to kick off a session)

```
Read program.md and score.py. Let's run the baseline experiment first,
then start the autonomous research loop. For each experiment, tell me
your hypothesis, the change you made, and the result before deciding
whether to commit or revert.
```

---

## Notes from Previous Sessions

> (Agent: append experiment notes here after each session for continuity)

**Session 2026-03-10**
- Baseline: val_ic 0.052810
- Exp 1: Log transforms -> val_ic 0.059190 (Committed)
- Exp 2: Ridge Regression -> val_ic 0.026930 (Reverted)
- Exp 3: Piotroski F-score -> val_ic 0.050465 (Reverted)
- Exp 4: Random Forest (200 trees, depth 5) -> val_ic 0.085329 (Committed)
- Exp 5: Interaction terms -> val_ic 0.081164 (Reverted)
- Exp 6: LGBMRegressor -> val_ic 0.036006 (Reverted)
- Exp 7: XGBRegressor -> val_ic 0.066198 (Reverted)
- Exp 8: Asset turnover -> Skipped (already in base features)
- Exp 9: Cross-sectional ranking -> val_ic 0.031989 (Reverted)
- Exp 10: Tune Random Forest -> val_ic 0.078471 (Reverted)

---

*finresearch — autoresearch pattern adapted for fundamental analysis*
*Inspired by karpathy/autoresearch — same loop, different domain.*
