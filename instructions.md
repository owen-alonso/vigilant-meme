# Using the next-hour return forecaster

This file is the operating manual for the **equity forecast** model in `forecast/`. It is not the language-model path (`main.py`, Shakespeare, Dynamic A LM). Those share a Mamba backbone; they do not share data, targets, or `generate.py`.

Run commands from the **repo root**.

```bash
python -m forecast.training -h
python -m forecast.generate -h
```

---

## What the model actually does

Given 1-minute OHLCV for one equity, it predicts the **expected log return over the next 60 minutes of the same US cash session** (09:30–15:59 Eastern).

It does **not**:

- Pull the live tape. `generate.py` reads a **local parquet**. `LATEST as of …` is the last timestamp **in that file**, not wall-clock now.
- Forecast overnight, 24h, or the next calendar hour after 15:59. The horizon must land in the **same session**.
- Output a trade, size, or “buy/sell”. It outputs a number in **basis points** and an implied price.
- See the future hour it is predicting. Features are causal (bars `<= t` only). **Realized** is scored afterwards when that future already exists in the file.

The network does not predict dollars. It predicts a **volatility-normalized** return. `generate.py` multiplies by the vol known at bar `t` and reports basis points.

\[
y_t = \frac{\log C_{t+60} - \log C_t}{\sigma_t \sqrt{60}}
\]

\(\sigma_t\) is EWM realized vol using only bars **up to \(t-1\)**. **1 bp = 0.01%**. **+10 bp** means the model expects the price about **0.10% higher** in one hour.

The same weights apply to any ticker: features are scale-free (no per-symbol embedding). That does **not** mean every name is in-sample; see [Out of sample](#out-of-sample-and-vendors).

---

## Quick start

1. Put one parquet per symbol in `data/`, named `<SYMBOL>_*.parquet` (example: `AAPL_clean_1min.parquet`).
2. Train (use `best.pt`, not `last.pt`):

```bash
python -m forecast.training
```

3. Forecast every symbol parquet (one **column per ticker**, cells = predicted move in bp):

```bash
python -m forecast.generate --checkpoint checkpoints/forecast/best.pt
python -m forecast.generate --checkpoint checkpoints/forecast/best.pt --symbols AAPL,MSFT
python -m forecast.generate --checkpoint checkpoints/forecast/best.pt --data data/AAPL_clean_1min.parquet
```

4. Before trusting a test IC, check whether train and test used the same vendor:

```bash
python scripts/split_report.py
```

---

## Data contract

| Requirement | Detail |
|---|---|
| File | `data/<SYMBOL>_*.parquet` |
| Columns | `datetime`, `Open`, `High`, `Low`, `Close`, `Volume` (case is normalized) |
| Optional | `source` (vendor name; used for mix warnings) |
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
| `--horizon` | `60` | Label lookahead in **1-minute bars** (`60` = one hour, same session). Changing this changes what the model *is*. |
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
| `ic` | **Information coefficient**: Pearson correlation of prediction vs realized return (in vol units). **This is the number that matters.** `+1` lockstep, `0` no linear relationship, `-1` backwards. On noisy 1h returns, even a real edge is often a few hundredths. |
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
python -m forecast.generate --checkpoint checkpoints/forecast/best.pt --data data/AAPL_clean_1min.parquet
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
