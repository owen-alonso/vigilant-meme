# Using the equity return forecaster

This file is the operating manual for the **equity forecast** model in `forecast/`. It is not the language-model path (`main.py`, Shakespeare, Dynamic A LM). Those share a Mamba backbone; they do not share data, targets, or `generate.py`.

The **estimand** is last-bar **cross-sectional** Pearson/Spearman on **next-day residual** returns (sector ETF when the parquet exists, otherwise SPY), over a train-era-locked liquid **equity** book (50–200 names). Index/sector/macro ETFs load as hedges, not as names you rank. Train labels start in 1999 by default; val/test calendar cuts stay locked.

Success band (honest, locked test window): mean CS IC **0.04–0.08** with t-stat **> 3**, and long-short **net IR ~1** after costs **without** a ruinous path (report unlevered IR + causal-vol max DD; `vol_target=1.0` is a 100% vol book). A single-date CS IC print near 0.14 can happen; report the **mean and the CS IC time series**. Do not retarget after seeing test. Dynamic A / bigger Mamba / more IC-loss weight are **not** the path.

Run commands from the **repo root**.

```bash
python -m forecast.training -h
python -m forecast.generate -h
python -m forecast.backtest -h
```

---

## What the model actually does

Given **daily** OHLCV for a liquid universe (Yahoo/Stooq split-adjusted, plus SPY and sector ETFs), it predicts each **equity's** next-session residual log return vs a trailing-beta **sector or SPY** hedge, in vol units.

It does **not**:

- Pull the live tape unless you ran `forecast.download` first. `generate.py` reads a **local parquet**. `LATEST as of …` is the last timestamp **in that file**, not wall-clock now.
- Mix overnight gap IC into the close-to-close headline. Default `--label-return close` predicts the next **session close**. `--label-return overnight` is a **different** book: `log(open_{t+1}) - log(close_t)` residual (next open is a label, never a feature).
- Output a trade, size, or “buy/sell”. It outputs a number in **basis points** (and a residual score used by the backtest).
- See the future day it is predicting. Features are causal (bars `<= t` only). **Realized** is scored afterwards when that future already exists in the file.
- Use next-bar **open** as a feature. Do not disable the calendar embargo. Same-bar `open_t` is a candle feature known at the close.

The network predicts a **volatility-normalized residual**. `generate.py` multiplies by the vol known at bar `t` and reports basis points.

\[
y_t = \frac{r_{t+1} - \beta_t r^{\mathrm{hedge}}_{t+1}}{\sigma_t}
\]

Default \(r_{t+1}\) is close-to-close. Overnight is \(r^{on}_t = \log(\mathrm{open}_{t+1}) - \log(\mathrm{close}_t)\). \(\beta_t\) uses same-bar returns **through \(t\) only**. The hedge is the mapped sector ETF when `--sector-residual` (default) and that parquet exists, otherwise SPY. `--double-residual` fits causal betas vs SPY **and** sector; `--industry-residual` adds a mapped industry ETF when that parquet exists. `--residualize-features` subtracts the same betas times same-bar hedge `ret_*` (not a label leak). \(\sigma_t\) is EWM realized vol using only bars **up to \(t-1\)**. **1 bp = 0.01%**. Hedge *forward* return is a **label** term, never a feature.

The same weights apply to any ticker: features are scale-free (no per-symbol embedding). Trading names are equities on the train-era-locked list in `forecast/universe.py`; SPY and sector/macro ETFs are **hedges**, not book names.

---

## Quick start (CS residual protocol)

1. Wipe mixed weekly caches if you still have them, then pull **adjusted daily** history for the locked universe:

```bash
python -m forecast.diagnostics --data-dir data --interval weekly --delete-mixed
python -m forecast.download --universe liquid --source yahoo --replace --interval daily
```

2. Ridge-only baseline first (`best.pt` is selected by **mean CS IC**, not pooled Pearson):

```bash
python -m forecast.training --universe liquid --interval daily --skip-only --checkpoint-dir checkpoints/forecast_ridge
# optional: --no-sector-residual --no-equities-only --no-train-from --no-ridge-rank-target
```

The skip defaults to **within-date rank-target ridge** with ``ridge=10``, feature winsor 3, and ``--ridge-features no_long_ts``, plus same-bar CS product features (val-selected). Do **not** enable walk-forward / later ``train_from`` / ListNet-skip / crash-date drop / year-balance / year-stable mask / double residual / feature residualization / industry residual / ``liquid_wide`` / regime heads / ``no_vol_products`` / trailing readout windows / trailing skip-IC shrink as the default from test: those lost or were a dead heat on locked **close-to-close** val.

``--label-return overnight`` is a **different estimand** (close_t → open_{t+1} residual). Features stay at close t; next open is a label. Yahoo/Stooq adjclose rescales OHLC together. The overnight **trade** is MOC t → MOO t+1 (flat in the next session). Do **not** mix overnight IC into the close-to-close headline. Close-to-close 0.04–0.08 was not reached; that book stays levered net IR ~1.

Overnight live book (val-gate; do not retarget from test). Paper 10 bp flatten is **not** live P&L.

```bash
# skip-only overnight residual (promoted skip recipe, different y)
python -m forecast.training --universe liquid --interval daily --skip-only \
  --label-return overnight --checkpoint-dir checkpoints/forecast_ridge_overnight

# locked TEST direction % + next-open MAE (PR #8 baseline + train-only readouts)
# fit on TRAIN, promote on locked VAL, report locked TEST. Not live P&L.
# cond_dir_blend = high-|pred| mix of left_tail_l1 ⊕ confidence_blend (TRAIN q,λ)
# decile_reliability = keep residual*sigma only in TRAIN-reliable pred_r bins
# cs_left_veto = always-up except CS-bottom ∩ TS left tail (TRAIN q, τ)
# logistic_up = TRAIN logistic P(up|pred_r) with TRAIN-chosen τ
python scripts/overnight_accuracy.py --data-dir data --universe liquid \
    --json checkpoints/forecast_ridge_overnight/accuracy.json \
    --calibrate-json checkpoints/forecast_ridge_overnight/overnight_calibrate.json
# optional: apply VAL-gated overnight readout to generate.py prices
# (affine {a,b} or drift_veto / bin / weekday spec from overnight_calibrate.json)
python -m forecast.generate --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
    --calibrate-json checkpoints/forecast_ridge_overnight/overnight_calibrate.json

# year series + live auction/locate/long-only stress (locked test)
python scripts/cs_overnight.py --data-dir data --universe liquid \
  --out checkpoints/forecast_ridge_overnight/overnight.json
# optional: val-gate open+15m fill as a separate estimand (does not replace overnight y)
python scripts/cs_overnight.py --data-dir data --universe liquid --try-fill 15 --no-lastbar-residual
# optional weekly residual fallback if harsh MOO kills overnight
python scripts/cs_overnight.py --data-dir data --universe liquid --try-weekly --no-lastbar-residual

# VAL-gated overnight LS vs long-only (TEST report-only). Cloud VM: --synthetic.
python scripts/overnight_shorting.py --data-dir data --universe liquid \
    --json checkpoints/forecast_ridge_overnight/shorting.json
python scripts/overnight_shorting.py --synthetic

# honest LS live_locate (locate + borrow) — default --live-costs; auto-prints long-only
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs
# unconstrained shorts (old live; not the honest default)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --cost-bundle live --compare-long-only
# long-only, no locate, borrow=0 — default live book after LS failed VAL
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs --long-only
# VAL-promoted long-only spec on synthetic (rank vs q20); confirm on liquid VAL
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs --long-only --weighting rank
# long-only conviction / inv-vol resize (VAL-gated; default remains equal q20)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs --long-only --long-size inv_vol --conf-pctile 0.5
# optional liquid sleeve (top CS turnover tercile; same skip w; use a lower min-names)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs --long-only --adv-floor-pctile 0.67 --min-names 8
# LS haircut experiment (NOT default): HTB shorts at half size, short NAV 0.30
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs --ls-haircut-experiment
# causal trailing overnight CS-IC trade gate (TRAIN-fit W,τ; default off)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs --long-only --ic-gate-window 60 --ic-gate-tau 0.0
# causal Friday / weekend weekday mask (VAL-gated; default always-on)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs --long-only --weekday-mask flat_friday
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs --long-only --weekday-mask weekend_only
# sector-overnight residual is the default skip (--sector-residual).
# SPY-only overnight residual baseline (A) for the VAL compare:
python -m forecast.training --universe liquid --interval daily --skip-only \
  --label-return overnight --no-sector-residual \
  --checkpoint-dir checkpoints/forecast_ridge_overnight_spy
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight_spy/best.pt \
  --holding overnight --live-costs --long-only
# causal CS-dispersion stress gate (TRAIN-fit kind/W/τ; default off)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --live-costs --long-only --disp-gate-kind cc --disp-gate-window 1 --disp-gate-tau 0.02
# overnight ⊕ close-to-close rank ensemble is TRAIN-chosen α, VAL-gated (α=1 default)
# causal adaptive α_t (trailing CS IC of overnight vs c2c) is TRAIN W/rule, VAL-gated (default off)
# sticky long-only enter/exit hysteresis is TRAIN-chosen, VAL-gated (default always-rebuild q20)
# soft trailing CS-IC gross scale is TRAIN-chosen, VAL-gated (default off / full q20)
python -m forecast.training --universe liquid --interval daily --skip-only \
  --label-return close --checkpoint-dir checkpoints/forecast_ridge
# harsh auction stress
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --cost-bundle harsh
# old flat overlay (20bp RT + 10bp exit-half auction + 5 borrow + 10 hedge)
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_overnight/best.pt \
  --holding overnight --cost-bundle live_flat

# open+15m fill is a different label; only train it after it wins locked val
python -m forecast.training --universe liquid --interval daily --skip-only \
  --label-return open15 --checkpoint-dir checkpoints/forecast_ridge_fill15
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge_fill15/best.pt \
  --holding open_fill --cost-bundle fill_live
```

Overnight **live vs paper** (same flatten book, 15% causal vol):

| bundle | meaning |
|---|---|
| `paper` / `--cost-bps 10` | enter+exit 10 bp. Understates auction/locate. |
| `live_flat` | 20 bp RT + 10 bp on the *exit half-notional* + 5 borrow + 10 hedge. First live-ish overlay. |
| `live` | 20 bp RT + **5 bp MOC + 10 bp MOO on full \|w\|**, ×2 on the bottom 30% CS `turnover_z`, + `8 * max(vol_level,0)` bp impact, + 5 borrow + 10 hedge. Unconstrained shorts — not the honest default. |
| `live_locate` (`--live-costs`) | `live` plus no shorts in the bottom 30% turnover (HTB proxy). Honest LS default. Report IR vs long-only on the same window. |
| `live_long_only` | `live` with no shorts, borrow=0. Residual still assumes a liquid ETF hedge overlay. Long sleeve ADV participation is ~2× the 50/50 long sleeve. |
| `harsh` | ugly MOO (30 bp), HTB, higher impact. If net IR dies, stop; next estimand is open+N fill or weekly residual — not bigger Mamba. |
| `ex_post_gap` | sensitivity: extra `0.25 * \|overnight move\| * \|w\|`. Uses realized. Not the default. |
| `fill_live` | MOC + continuous open+N exit (no MOO). Only with `--label-return open15`. |

`--open-auction-bps` is the *legacy* extra on the exit half-notional. Prefer `--moc-bps` / `--moo-bps`. Open+N (`open15`) mixes `15/390` of next-session return into the **label**; do not silently train it as overnight `y`.

```bash
# regime heads / surgical vol+CS-product drop / trailing readout window (all lost on val)
python scripts/cs_regime_ablate.py --data-dir data --universe liquid
# causal trailing skip-IC shrink (lost on close-to-close val)
python scripts/cs_shrink_ablate.py --data-dir data --universe liquid --also-labels
# ~170-equity 2018-era book (Yahoo extras; --skip-existing reuses the 85-name cache)
python -m forecast.download --universe liquid_wide --source yahoo --interval daily --skip-existing
python -m forecast.training --universe liquid_wide --interval daily --skip-only
# year-balance / year-stable mask / expanding WF diagnostic
python scripts/cs_year_ablate.py --data-dir data --universe liquid
# two-factor market+sector residual (labels); optional feature residualization
python -m forecast.training --universe liquid --skip-only --double-residual
python -m forecast.training --universe liquid --skip-only --residualize-features
python -m forecast.training --universe liquid --skip-only --industry-residual
```

```bash
python scripts/cs_collapse_ablate.py --data-dir data --universe liquid
```

3. Optional tiny frozen-skip encoder (do **not** scale Mamba / Dynamic A / IC-loss weight to chase 0.14):

```bash
python -m forecast.training --universe liquid --interval daily --checkpoint-dir checkpoints/forecast --d-model 32 --n-layer 1
```

4. Locked-window book with costs. Default is **quantile tails**, 1-day hold smoothing, **causal** expanding vol at 15% annual (not a 100% vol toy). A 5-day hold kills the 1-day CS signal.

```bash
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge/best.pt --cost-bps 10 --json checkpoints/forecast_ridge/backtest.json --cs-csv checkpoints/forecast_ridge/cs_ic.csv
# Owen-comparable tails + 100% vol (will print huge max DD):
python -m forecast.backtest --checkpoint checkpoints/forecast_ridge/best.pt --weighting quantile --hold-halflife 0 --vol-target 1 --full-sample-vol --cost-bps 10
```

5. Ablations (synthetic CS universe on CPU, or your `data/` on GPU):

```bash
python scripts/ablate_cs.py
python scripts/ablate_cs.py --data-dir data --universe liquid --skip-only-only
```

6. Vendor / split sanity:

```bash
python scripts/split_report.py data/AAPL_daily.parquet
```

---

## Data contract

| Requirement | Detail |
|---|---|
| File | `data/<SYMBOL>_daily.parquet` from `python -m forecast.download` |
| Columns | `datetime` (US/Eastern), `open`, `high`, `low`, `close`, `volume`, `source=alphavantage`, `interval=daily` |
| Session | One bar per US trading day |
| Timezone | Tz-aware stamps are converted to America/New_York, then made naive |
| `Close <= 0` | Rejected (log price undefined) |
| Session | 09:30–15:59 Eastern, **390** minute slots |
| Timezone | Tz-aware stamps are converted to America/New_York, then made naive |
| `Close <= 0` | Rejected (log price undefined) |

Sparse vendor minutes are **forward-filled inside the session** and marked `traded=0`. A hole is a stale last print, not a new trade. Prices **do not** carry across dropped days. A return that would jump a gap of more than **4 calendar days** is treated as missing.

A training **label** is valid only if **all** of these hold:

- This bar is a **real print** (`traded`)
- `t+60` is still in the **same session**
- `t+60` is a **real print** (unless you passed `--allow-stale-horizon`)
- Features are finite and past the warmup (default **2 sessions**)

If you generate on the last hour of a file, **Realized** will be “not yet known”: the horizon is still open. That is expected.

Splits are **chronological by session**, not random rows: train, then val (`--val-fraction`), then test (`--test-fraction`). A label never crosses a split boundary because the target cannot leave its session.

---

## Training: `python -m forecast.training -h`

Default loss is **Huber on the mean only**. Read **`val_ic`**, not `val_loss`. One-hour returns are mostly noise; loss can improve by shrinking toward zero while the model learns nothing.

`best.pt` is the checkpoint with the **highest validation IC**, not the lowest loss. `last.pt` is whatever the loop had at the last eval. After training, the logged **TEST** line reloads **`best.pt`**.

`--heteroscedastic` and `--loss gaussian` must agree: gaussian needs the extra head; the extra head is only trained under gaussian on this CLI. Do not pass `--heteroscedastic` and `--no-heteroscedastic` together.

### data

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir` | `data` | Folder of symbol parquets. |
| `--horizon` | `1` | Label lookahead in **bars** (`1` = next session on daily data). Scoring horizon>1 on every overlapping bar fakes Pearson — leave this at 1 for the CS protocol. |
| `--seq-len` | `256` | Minutes of history in each training window. |
| `--stride` | `64` | How far the window slides. Smaller → more overlapping samples, more compute. |
| `--min-context` | `64` | First this many bars of every window are **unsupervised**. The SSM is still warming up. |
| `--min-session-bars` | `30` | Drop a day with fewer real prints than this. |
| `--val-fraction` | `0.15` | Fraction of **sessions** for validation (end of the train era, not shuffled bars). |
| `--test-fraction` | `0.15` | Fraction of sessions for test. Train gets the rest. |
| `--allow-stale-horizon` | off | Label bars whose `t+horizon` slot was a fill, not a print. **Not recommended** (stale zeros). |

Not on `-h` but fixed in code (and stored in the checkpoint): vol EWM half-life = 390 bars, vol floor `1e-5`, z-score window = 5 sessions, feature clip = 8, warmup = 780 bars, max session gap = 4 days. Changing those in `DataConfig` without retraining makes `generate.py` lie, because generate rebuilds features from the **checkpoint’s** `data_config`.

### model

| Flag | Default | Meaning |
|---|---|---|
| `--d-model` | `96` | Hidden width. Bigger → more capacity and memory. |
| `--n-layer` | `4` | Number of Mamba blocks. |
| `--d-state` | `16` | SSM state size per channel. |
| `--expand` | `2` | Inner width is `expand * d-model`. |
| `--dropout` | `0.1` | Dropout after the input projection. |
| `--dynamic-weights` | off | Turn on **Dynamic A** (bar-dependent timescale on \(A\)). Off = plain Mamba-1. |
| `--dynamic-strength` | `0.1` | How far Dynamic A may move \(A\). Only matters with `--dynamic-weights`. |
| `--heteroscedastic` | off | Extra **log-sigma** head. Requires `--loss gaussian`. |
| `--no-heteroscedastic` | off | Mean-only head even with `--loss gaussian`. |

### optim

| Flag | Default | Meaning |
|---|---|---|
| `--batch-size` | `16` | Windows per step. Lower if you OOM. |
| `--epochs` | `8` | Passes over training windows. Ignored if `--max-steps` is set. |
| `--max-steps` | none | Hard cap on optimizer steps (replaces `epochs × steps/epoch`). |
| `--lr` | `0.001` | Peak AdamW LR. Warmup is 5% of total steps, then cosine down to 10% of `--lr` (not CLI flags). |
| `--weight-decay` | `0.01` | L2 on most weights. Biases, norms, `A_log`, and `D` are excluded. Grad clip is **1.0** (not a flag). |
| `--loss` | `huber` | `huber` (default, robust), `mse` (squares outliers), `gaussian` (mean + log-sigma NLL; needs `--heteroscedastic`). |
| `--precision` | `bf16` | Autocast: `bf16`, `fp16` (GPU + GradScaler), `fp32`. CPU `fp16` falls back. |
| `--seed` | `42` | Python / NumPy / PyTorch seed. |
| `--eval-interval` | `250` | Validate (and maybe write `best.pt`) every N steps. |
| `--log-interval` | `25` | Print train loss / grad / LR every N steps. |
| `--num-workers` | `0` | DataLoader workers. Keep `0` on Windows unless you know you need more. |
| `--checkpoint-dir` | `checkpoints/forecast` | `best.pt`, `last.pt`, `summary.json`. Relative paths are under the **repo root**. |
| `--early-stop-evals` | `8` | Stop if val IC does not improve for this many evals. |
| `--cpu` | off | Force CPU even if CUDA is available. |

Example with uncertainty head (only then does generate show **uncert. (bp)**):

```bash
python -m forecast.training --loss gaussian --heteroscedastic
```

---

## How to read training output

Startup looks like:

```
device=... params=... features=18 seq_len=256 horizon=60 dynamic_weights=... loss=huber heteroscedastic=False
checkpoints -> ...
AAPL: N sessions, ... labelled | train<YYYY-MM-DD val<... test>= ...
WARNING ... train vendor is mostly X but test is mostly Y
train: N windows, ... labelled bars
```

Take a **vendor WARNING** seriously. A strong test IC can be “the vendor changed,” not “the model works.”

### Step line

```
step     25/2000  loss=0.51234  grad=0.83  lr=1.00e-03  12.40 it/s
```

| Field | Meaning |
|---|---|
| `step a/b` | Optimizer step / planned total. |
| `loss` | Masked train loss over the last `--log-interval` steps (Huber / MSE / gaussian NLL). |
| `grad` | Global grad norm after clip. Exploding or always ~0 is a red flag. |
| `lr` | Current learning rate (warmup then cosine). |
| `it/s` | Steps per second. |
| Dynamic A extras | If `--dynamic-weights`: scale stats. Collapse = stuck near 1; saturation = stuck at the tanh bounds. |

`non-finite gradients, skipping optimizer step` means that step was dropped. Occasional skips are a warning; a stream of them means the run is unhealthy.

### Eval / TEST line

```
eval step 250: loss=0.50123 ic=+0.0420 r2=+0.00180 dir=0.5123 pred_std=2.10bps n=18420
new best val_ic=+0.0420 -> checkpoints/forecast/best.pt
TEST (best step 250): loss=... ic=... r2=... dir=... pred_std=...bps n=...
```

| Field | How to read it |
|---|---|
| `loss` | Same objective as train, on val/test labelled bars. **Not** the selection metric. |
| `ic` / `ic_raw` | **Pooled** last-bar Pearson. Useful as a diagnostic. **Not** the Phase-3 estimand and **not** “test IC 0.14”. |
| `cs_ic` | **Mean cross-sectional Pearson** over dates with enough names. **This is the selection metric** when it is finite (`best.pt`). Target band 0.04–0.08 with `cs_t` > 3. |
| `cs_sp` | Mean CS Spearman. Report it; do **not** blend it with Pearson and call the blend test IC. |
| `cs_t` / `cs_dates` | t-stat of daily CS ICs, and how many dates went into the mean. |
| `r2` | Skill vs the honest baseline **predict zero**. Negative r2 = worse than predicting no move. Tiny positive r2 can still pair with useful IC. |
| `dir` | Fraction of **non-zero** outcomes where `sign(pred) == sign(realized)`. 0.50 = coin flip. |
| `pred_std` | Std of predicted moves in **bp**. If this collapses toward 0 while IC is ~0, the model is shrinking to the mean, not forecasting. |
| `n` | How many labelled bars went into the metrics. Tiny `n` → noisy IC. |

Early stop: no new best val IC for `--early-stop-evals` evals.

---

## Generate: flags

```bash
python -m forecast.generate --checkpoint checkpoints/forecast/best.pt
python -m forecast.generate --checkpoint checkpoints/forecast/best.pt --symbols AAPL,MSFT --last 10
python -m forecast.generate --checkpoint checkpoints/forecast/best.pt --data data/AAPL_1min.parquet
```

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `checkpoints/forecast/best.pt` | Weights **plus** `data_config`, feature mean/std, and symbol list. Generate rebuilds features the same way train did. |
| `--data` | every `*.parquet` in the checkpoint’s `data_dir` | One parquet, or a **directory** of parquets. Omitted = all files in `data_dir`. That is **not** the live market; it is whatever is on disk. |
| `--symbols` | all discovered files | Comma-separated tickers, e.g. `AAPL,MSFT`. Must match parquet names (`AAPL_*.parquet`). |
| `--context` | training `--seq-len` | History length per forecast. Only the **last** bar of each window is used as the printed prediction. Shorter than `--min-context` is off-distribution (you will get a Note). |
| `--last` | `1` | Newest N bars **per symbol**. `1` prints the LATEST snapshot only; `>1` also prints a time table of predicted moves. |
| `--asof` | none | Single bar at or before this timestamp, e.g. `2026-08-21 14:30`. |
| `--csv` | none | Wide predicted-move table: `datetime` plus **one column per ticker** (bp). |
| `--batch-size` | `32` | Inference batching only. |
| `--cpu` | off | Force CPU. |

The model sees **real historical bars up through time `t`**. It does **not** see `t+60`. If you score a name/period that was in training, that is **in-sample** (weights already saw that history). The forward pass still does not peek at the label.

stderr prints `forecasting AAPL ...` per file so a large `data/` folder does not look hung.

---

## How to read `generate.py` output

Each **stock is its own column**. The number you care about is **`pred (bp)`** — the predicted next-hour move for that ticker.

```
========================================================================
  NEXT-HOUR RETURN FORECAST
========================================================================
  Horizon     60 minutes ahead  (same session)
  Context     last 256 minute bars
  Stocks      AAPL, MSFT, NVDA
  Checkpoint  ...
  Trained on  AAPL, MSFT, ...
  Device      cpu

  Units: bp = basis points = 0.01%.  +10 bp means the model
  expects the price about 0.10% higher in one hour.
  Each ticker is its own column. pred (bp) is the predicted move.

------------------------------------------------------------------------
  LATEST  (one column per stock)
------------------------------------------------------------------------
                 AAPL          MSFT          NVDA
as of    2026-08-21 15:59  2026-08-21 15:59  2026-08-21 15:58
print                 yes             yes              no
last $            25.0950        410.1200         120.5000
pred (bp)            -0.4            +1.2             -2.1
pred $ in 1h      25.0940        410.1690         120.4748
realized (bp)           -               -                -
```

With `--last` greater than 1, a second table lists predicted moves over those bars (rows = time, columns = tickers, `-` if that symbol has no bar at that minute):

```
------------------------------------------------------------------------
  PREDICTED MOVE (bp)  |  one column per stock
------------------------------------------------------------------------
when                  AAPL   MSFT   NVDA
2026-08-21 15:58      -     +0.8    -2.0
2026-08-21 15:59     -0.4   +1.2    -2.1
```

### Header

| Line | Meaning |
|---|---|
| `NEXT-HOUR RETURN FORECAST` | Same-session next-hour call for every scored parquet. |
| `Horizon` | Lookahead from the **checkpoint** (usually 60 minutes, same session). |
| `Context` | Minutes of history fed in. |
| `Stocks` | Tickers that produced a forecast. |
| `Checkpoint` | Weight file. |
| `Trained on` | Symbols listed in the checkpoint. Missing tickers → out-of-sample Note. |
| `Device` | `cpu` or `cuda`. |
| `Note:` | Warnings; see below. |

### LATEST table (columns = tickers)

| Row | Meaning |
|---|---|
| `as of` | Timestamp of that symbol’s last requested minute (`YYYY-MM-DD HH:MM`), session-local Eastern. **Not** “now” unless the parquet was updated to now. Files can end at different minutes. |
| `print` | `yes` = real print that minute; `no` = forward-filled hole. |
| `last $` | `close` at that bar. |
| `pred (bp)` | **Predicted move**: expected 60-minute log return in bp (`pred_norm × scale × 10,000`). This is the column to read per stock. |
| `pred $ in 1h` | `last_price × exp(predicted log return)`. Point map of the mean log-return, not `E[future price]` under a lognormal (no `½σ²`). |
| `realized (bp)` | What actually happened over the next 60 minutes, in bp, when the label is valid; `-` if the horizon is still open. |

Huber/MSE checkpoints do **not** print an uncertainty row (untrained sigma would be noise). Train with `--loss gaussian --heteroscedastic` if you want residual 1σ.

### Notes you may see

| Note | What to do |
|---|---|
| No trained uncertainty … Only the predicted mean is shown | Normal for default Huber. Do not invent a confidence from a missing column. |
| `context=` shorter than training `min_context` | That last bar was never a training label. Don’t treat it as a standard call. |
| Out-of-sample symbol(s): … | Those tickers were not in the training set. Allowed (shared features), but not a train-set backtest. |
| Train vendor was mostly X, test mostly Y | Do not read a backtest IC as the same experiment as live/another vendor. |
| `skipped SYMBOL: ...` | That file had too little history or no bar at `--asof`. Others still printed. |

### `--csv`

Wide table: index/column `datetime`, then **one column per ticker**, values = predicted move in **bp**. Missing timestamps are empty.

---

## Using it correctly

1. **Always generate with `best.pt`** unless you are debugging the last step on purpose.
2. **Match the file to the question.** `--data` is the tape the model sees. An old parquet cannot “line up with the market right now.”
3. **Need a full minute bar.** High/low/close/volume for minute `t` are inputs. This is not a mid-bar forecast.
4. **Don’t change `--horizon` / `--seq-len` / grid rules at generate time** relative to the checkpoint. Generate already reloads `data_config` from the checkpoint; `--context` is the main inference override, and cutting it below `min_context` is a footgun.
5. **In-sample vs test vs new ticker** are different claims. Training IC is fit; val IC selected `best.pt`; test IC is the honest historical number **if** the vendor mix matches; a new parquet of a new name is OOS.
6. **Direction accuracy ~0.50 and IC ~0** means no usable linear edge on that split. Do not “trade the predicted move” anyway.
7. **`pred_std` near 0** with flat IC: the model is predicting almost nothing. A large predicted move on one quiet bar is not automatically more trustworthy.
8. **Filled slots** (`print = no`) are last-known price, not a new print. Prefer `real print` rows for decisions.
9. **Same-session only.** A 15:30 forecast’s “1 hour” is still 16:30 **on the session clock** (15:59 is the last slot — many late-day horizons are unlabeled on purpose).
10. **This is not execution advice.** Slippage, fees, short-sale rules, and that the target is a **log return**, not a fill, are all outside the model.

---

## Checkpoints

Each `*.pt` stores weights, `model_config`, `data_config`, `train_config`, feature names, feature mean/std, and per-symbol split metadata. Load the file you trained; mixing a new architecture flag with an old file will fail `load_state_dict` or silently change the head (`heteroscedastic`).

`summary.json` next to the checkpoints repeats best val IC, test metrics, and symbol meta.

---

## Related commands

| Command | Role |
|---|---|
| `python -m forecast.training -h` | Train-flag reference (this file explains each one). |
| `python -m forecast.generate -h` | Inference flags. |
| `python scripts/split_report.py` | Session counts and vendor mix per train/val/test. |
| `python main.py -h` | **Language model**, not this forecaster. |
