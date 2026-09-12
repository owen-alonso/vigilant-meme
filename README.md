# Dynamic A Mamba (V1)

A small **Mamba-1** language model with an optional token-dependent **Dynamic A** controller.

The first milestone is not “full dynamic Mamba”. It is:

- original (fixed-weight) Mamba-1
- plus a small input-dependent controller
- plus stable, token-dependent modulation of the SSM matrix \(A\)
- plus tests, a speed benchmark, and a controlled training comparison

Later flags (`dynamic_B`, `dynamic_C`, `dynamic_dt`, low-rank updates, a learned gate) exist on the config object but are **not implemented** in V1 and raise `NotImplementedError` if enabled.

## Why this repo

The original tree was an empty PyCharm stub. This package therefore includes a faithful Mamba-1 baseline **and** Dynamic A, sharing one selective-scan implementation. `dynamic_weights=False` never constructs or runs the controller.

## Equations

Mamba-1 uses a diagonal state-space model (S6). After the input projection, causal conv, and SiLU, the SSM input \(x_t \in \mathbb{R}^{D}\) (here \(D =\) `d_inner`) is mixed with a latent state \(h_t \in \mathbb{R}^{D \times N}\):

\[
A = -\exp(A_{\log}) \in \mathbb{R}^{D \times N}
\]

\[
\Delta_t = \mathrm{softplus}(W_\Delta x_t + b_\Delta) \in \mathbb{R}^{D}
\]

\[
\overline{A}_t = \exp(\Delta_t \, A), \quad \overline{B}_t = \Delta_t \, B_t
\]

\[
h_t = \overline{A}_t \odot h_{t-1} + \overline{B}_t \odot x_t, \quad y_t = C_t \cdot h_t + D \odot x_t
\]

\(B_t, C_t\) are already input-dependent in Mamba-1. \(A\) is not. Dynamic A makes the **timescale** of \(A\) token-dependent without generating a dense \(N \times N\) matrix per token.

### Dynamic A (V1)

A small controller runs over the full sequence (not a Python loop over tokens):

\[
\delta A_t = W_2\,\mathrm{SiLU}(W_1 x_t + b_1) + b_2 \in \mathbb{R}^{N}
\]

\[
s_t = \mathrm{clamp}\big(1 + \sigma \tanh(\delta A_t),\; \varepsilon\big) \in \mathbb{R}^{N}
\]

\[
A_t = A \odot s_t \in \mathbb{R}^{D \times N}
\]

\(s_t\) is broadcast across channels \(D\). Default \(\sigma =\) `dynamic_strength` \(= 0.1\) keeps \(s_t \in [0.9, 1.1]\), so \(A_t\) stays negative whenever \(A\) is. The controller **never** writes into \(A_{\log}\).

\(W_2\) and \(b_2\) are **zero-initialized**, so at step 0 \(\delta A = 0\), \(s_t = 1\), and Dynamic A matches the baseline.

Shapes:

| tensor | shape |
|---|---|
| block in/out | `[B, L, d_model]` |
| SSM input \(x\) | `[B, L, D]` with `D = expand * d_model` |
| `A_log`, base \(A\) | `[D, N]` |
| `delta_A`, scale \(s\) | `[B, L, N]` |
| dynamic \(A_t\) | `[B, L, D, N]` |
| `B`, `C` | `[B, L, N]` |
| `dt` | `[B, L, D]` |

## Layout

```
mamba_lm/
  config.py          # MambaConfig / TrainConfig
  mamba.py           # MambaBlock, MambaLayer (RMSNorm + residual)
  scan.py            # shared selective scan (baseline and dynamic)
  model.py           # MambaLM
  dynamic/           # controller, modulator, DynamicParameters
  train.py           # AdamW, AMP, eval, checkpoints
  reporting.py       # parameter split + overhead %
scripts/benchmark.py
scripts/train_compare.py
tests/
```

## Configuration

```python
from mamba_lm import MambaConfig, MambaLM

baseline = MambaLM(MambaConfig(dynamic_weights=False))
dynamic = MambaLM(MambaConfig(
    dynamic_weights=True,
    dynamic_A=True,
    dynamic_strength=0.1,          # σ
    dynamic_controller_dim=None,   # default max(32, d_model // 4)
    dynamic_parameterization="elementwise",
))
```

V1 implemented fields: `dynamic_weights`, `dynamic_A`, `dynamic_controller_dim`, `dynamic_strength`, `dynamic_parameterization="elementwise"`.

Reserved (off): `dynamic_B`, `dynamic_C`, `dynamic_dt`, `dynamic_gate`, `dynamic_rank`.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pytest tests/
```

## Commands

```bash
# Parameter split
python main.py report --dynamic-weights

# Speed: baseline vs Dynamic A (same B, L, d_model, N, dtype)
python main.py benchmark
# or
python scripts/benchmark.py

# Tiny Shakespeare, identical hparams, baseline vs Dynamic A
python main.py compare --max-steps 50 --d-model 64 --n-layer 2 --seq-len 128
# or
python scripts/train_compare.py

# Train one mode
python main.py train
python main.py train --dynamic-weights --dynamic-strength 0.1
```

Checkpoints store `config` next to weights. A **baseline** checkpoint still loads into a Dynamic A model (`strict=False` when the flag disagrees); missing controller keys stay at zero init.

Training logs periodic Dynamic A scale stats (`mean/std/min/max`) so you can see collapse (\(s \approx 1\)), saturation (\(s\) stuck at the tanh bounds), or useful variation.

## Next-day residual CS forecasting

`forecast/` predicts **next-day residual** returns (sector hedge, else SPY) for
train-era-locked **equities**. The estimand is last-bar **mean CS IC**, not
weekly AAPL+MSFT time-series Pearson. Dynamic A is optional and off by default.

```
forecast/
  universe.py      # 2018-era liquid names, equity filter, sector hedge map
  data.py          # causal features, residual label, CS z/rank, CS ridge
  training.py      # skip-only ridge, CS IC / RankNet, checkpoints
  backtest.py      # rank/quantile book, hold smoothing, causal vol, costs
  diagnostics.py   # mixed adjusted/raw weekly gate
  synthetic.py     # planted CS-momentum universe for CPU ablations
```

```bash
python -m forecast.download --universe liquid --source yahoo --replace --interval daily
python -m forecast.training --universe liquid --skip-only
python -m forecast.backtest --checkpoint checkpoints/forecast/best.pt --cost-bps 10
python scripts/cs_collapse_ablate.py --data-dir data --universe liquid
python scripts/cs_year_ablate.py --data-dir data --universe liquid
python scripts/cs_regime_ablate.py --data-dir data --universe liquid
python scripts/cs_shrink_ablate.py --data-dir data --universe liquid --also-labels
python -m forecast.training --universe liquid --skip-only --label-return overnight
python scripts/pretrain_finetune.py --data-dir data --universe liquid
python -m forecast.panel_mmap --data-dir data --universe liquid --write
python -m forecast.training --universe liquid --skip-only --label-return overnight --pup-head --num-workers 0
python scripts/overnight_pup.py --data-dir data --universe liquid
python scripts/cs_overnight.py --data-dir data --universe liquid
python scripts/overnight_accuracy.py --data-dir data --universe liquid
python scripts/ablate_cs.py
python scripts/split_report.py data/AAPL_daily.parquet
```

Yahoo/Stooq daily caches are split-adjusted (`adjclose` is written into
`close`, and the same factor rescales open/high/low). Alpha Vantage compact
daily is unadjusted and too short for this protocol. Close-to-close target:

\[
y_t = \frac{r_{t+1} - \beta_t r^{\mathrm{hedge}}_{t+1}}{\sigma_t}
\]

Overnight (`--label-return overnight`) replaces \(r_{t+1}\) with
\(\log(\mathrm{open}_{t+1})-\log(\mathrm{close}_t)\). Next open is a **label**,
never a feature. `scripts/pretrain_finetune.py` runs the historic pretrain →
recent fine-tune schedule from `pretrain_finetune_cuts.json` (FT train gets
session-rank recency, half-life 126; labels unchanged). Cut (0) skip-only FT
did not beat live_locate +5.58 — do not promote that ckpt. Train CS batches
read the Data Manager mmap feed (`data/_panel_cache/mmap_manifest.json`)
when present (`load_manifest` / `load_symbol_mmap(..., mmap_mode="r")`);
never `cs_train_*.pt`. `--pup-head` is the (2a) direct overnight-up P(up)
classifier (VAL overnight-up ≥60% at cover ≥5%; TEST report-only). PIT
drops overnight labels where `data/_pit/factors/{SYM}_daily_factors.parquet`
has `next_split_days==1`. Optional membership as-of:
`data/_pit/liquid_membership.json`.
VAL-gate the *book* on cost-aware `live_locate` IR vs +5.58 (report-only
on the P(up) cut). Backtest
`--holding overnight` flattens every open (MOC→MOO).
`--live-costs` is the Owen-runnable pack (20 bp RT + name-level MOC/MOO +
thin/vol impact + 5 bp borrow + 10 bp hedge). `--long-only` drops shorts and
borrow (no locate). `--locate-adv-pctile 0.3` blocks shorts in the bottom
turnover tercile. `--adv-floor-pctile 0.67 --min-names 8` is the liquid
sleeve (same skip, drop thin names before weights). `--compare-long-only`
prints both books. Do not headline `vol_target=1`. Open+N fill
(`--label-return open15`) is a **separate** estimand; it does not replace
overnight `y` unless it wins the locked-val gate.

`best.pt` is selected by mean CS IC when a cross-section exists. `backtest.py`
builds a dollar-neutral (or `--long-only`) quantile book on the locked
test window, with round-trip costs, 1-day hold smoothing, and causal vol
targeting at 15% annual. Report mean CS IC + t-stat + **unlevered** net IR
and causal max DD. Do not call a Spearman/Pearson blend, a val number, or a
100% vol path "test IC 0.14" / "IR 1 with a survivable book".

## Tests

Shape, baseline equivalence, gradients, token/batch dependence, fp32 + AMP stability, checkpoint round-trip, parameter reporting, scan backward, and forecast causality/masks. CUDA is used automatically when present.
