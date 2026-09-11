"""Locked-TEST overnight TRUE/FALSE accuracy helpers and synthetic protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from forecast.accuracy import (
    abs_error_block,
    direction_hits,
    evaluate_overnight_skip,
    hit_rate_inference,
    score_eval_frame,
    two_sided_normal_p,
)
from forecast.synthetic import write_cs_overnight_universe


def test_two_sided_normal_p_symmetric():
    assert two_sided_normal_p(0.0) == 1.0
    assert 0.04 < two_sided_normal_p(2.0) < 0.05
    assert two_sided_normal_p(10.0) < 1e-20


def test_hit_rate_inference_coin_flip_is_not_significant():
    rng = np.random.default_rng(0)
    hits = rng.integers(0, 2, size=40).astype(np.float64)
    stats = hit_rate_inference(hits)
    assert stats["n"] == 40
    assert 0.2 < stats["hit_rate"] < 0.8
    assert stats["p_vs_half"] > 0.05


def test_hit_rate_inference_detects_edge():
    hits = np.ones(400, dtype=np.float64)
    hits[:80] = 0.0
    stats = hit_rate_inference(hits)
    assert stats["hit_rate_pct"] == 80.0
    assert stats["p_vs_half"] < 1e-10
    assert stats["z_vs_half"] > 0


def test_direction_hits_drops_flat_realized():
    pred = np.array([1.0, -1.0, 1.0])
    realized = np.array([0.2, -0.1, 0.0])
    hits = direction_hits(pred, realized)
    assert hits.tolist() == [1.0, 1.0]


def test_abs_error_block_mae_median_rmse():
    err = np.array([1.0, 2.0, 3.0])
    block = abs_error_block(err)
    assert block["n"] == 3
    assert block["mae"] == 2.0
    assert block["median_ae"] == 2.0
    assert abs(block["rmse"] - np.sqrt(14.0 / 3.0)) < 1e-12


def test_score_eval_frame_perfect_overnight_prices():
    import pandas as pd

    close = np.array([100.0, 50.0, 25.0, 10.0])
    r_on = np.array([0.01, -0.02, 0.03, -0.01])
    nxt = close * np.exp(r_on)
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "date": [1, 1, 2, 2],
            "pred": r_on / 0.01,
            "y": r_on / 0.01,
            "scale": np.full(4, 0.01),
            "close": close,
            "next_open": nxt,
            "r_on": r_on,
            "pred_r": r_on,
            "implied_open": nxt,
            "implied_open_given_hedge": nxt,
        }
    )
    out = score_eval_frame(df, min_names=2)
    assert out["n_samples"] == 4
    assert out["n_dates"] == 2
    assert out["direction"]["overall"]["hit_rate"] == 1.0
    assert out["direction"]["realized_overnight_up_pct"] == 50.0
    assert out["price_error"]["dollars"]["mae"] < 1e-9


def test_synthetic_overnight_skip_beats_coin_flip_on_residual(tmp_path: Path):
    data_dir = tmp_path / "data"
    write_cs_overnight_universe(data_dir, n_names=12, n_days=220, seed=1, rho=0.65)
    payload = evaluate_overnight_skip(str(data_dir), "synthetic", log_fn=None)
    assert payload["n_samples"] >= 20
    assert payload["n_names"] >= 8
    resid = payload["direction"]["residual_vs_residual"]["hit_rate"]
    # Planted CS overnight residual should not be a coin flip on y.
    assert resid > 0.55
    assert payload["cs_ic"]["cs_ic"] > 0.05
    mae = payload["price_error"]["dollars"]["mae"]
    assert np.isfinite(mae) and mae > 0
    pct = payload["price_error"]["pct_of_prior_close"]["mae"]
    assert np.isfinite(pct) and pct > 0
