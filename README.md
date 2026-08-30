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

## Next-hour return forecasting

`forecast/` reuses the Mamba backbone for a regression task: given 1-minute
OHLCV bars for an equity, predict its **log return over the next 60 minutes**.

```
forecast/
  config.py     # DataConfig / ForecastModelConfig / ForecastTrainConfig
  data.py       # session grid, causal features, next-hour target, windowing
  model.py      # ReturnForecaster: continuous features -> scalar per bar
  training.py   # masked loss, IC / directional metrics, checkpoints
  generate.py   # forecasts for any symbol's parquet
```

```bash
python -m forecast.training --epochs 3
python -m forecast.generate --checkpoint checkpoints/forecast/best.pt --last 10
python scripts/split_report.py          # bar density + vendor mix per split
```

### Data contract

One parquet per symbol in `data/`, named `<SYMBOL>_*.parquet`, with columns
`datetime, Open, High, Low, Close, Volume`. Bars are snapped onto a regular
390-slot session grid (09:30–15:59); untraded slots are forward-filled and
flagged, since vendor files for illiquid names skip minutes entirely.

### Target

\[
y_t = \frac{\log C_{t+60} - \log C_t}{\sigma_t \sqrt{60}}
\]

\(\sigma_t\) is an EWM realized-volatility estimate using only bars up to
\(t-1\). Predicting a volatility-normalized return keeps the objective
stationary across calm and stressed regimes; `generate.py` multiplies by
\(\sigma_t\sqrt{60}\) to report basis points. A bar is labelled only when it is
a real print, \(t+60\) falls inside the same session, **and** the horizon bar
itself is a real print (not a forward-filled hole).

Intraday missing minutes are forward-filled **within the session**. Prices do
not carry across dropped days; returns that would cross a gap of more than
four calendar days are treated as missing. Tz-aware timestamps are converted
to US/Eastern before the 09:30–15:59 filter.

Default training uses Huber on the mean only. Pass `--loss gaussian
--heteroscedastic` if you want a trained residual-uncertainty head;
`generate.py` will otherwise omit the uncertainty column.

Features are scale-free (vol-normalized returns, ranges, standardized volume,
staleness, time-of-day), so the model carries no per-symbol parameters and the
same weights apply to any equity.

### Reading the metrics

`val_ic` (correlation between prediction and realized return) is the metric
that matters, not `val_loss`. One-hour return noise dominates the loss, so a
model can reduce MSE by shrinking toward zero while learning nothing. `r2` is
measured against the honest baseline of predicting zero.

## Tests

Shape, baseline equivalence, gradients, token/batch dependence, fp32 + AMP stability, checkpoint round-trip, parameter reporting, scan backward, and forecast causality/masks. CUDA is used automatically when present.
