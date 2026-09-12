"""IDEA N overnight-up rank sleeve + Dynamic A unused-encoder autopsy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.overnight_dynamic import (
    resolve_dynamic_a_steps,
    skip_encoder_stats,
    train_step_budget,
)
from forecast.overnight_up_rank import (
    decide_overnight_up_rank_promote,
    fit_logit2_up,
    apply_logit2_up,
    fit_overnight_up_rank_on_train,
    format_overnight_up_rank_block,
    high_drift_weekdays,
    rank_score,
)


def _ascii_ok(text: str) -> None:
    text.encode("cp1252")
    text.encode("ascii")


def test_resolve_dynamic_a_steps_keeps_tiny_ci_budget():
    assert resolve_dynamic_a_steps(6, n_train_dates=40) == 6
    assert resolve_dynamic_a_steps(80, n_train_dates=110) == 80
    # Liquid-scale panel: 80 is a fraction of an epoch -- scale up.
    assert resolve_dynamic_a_steps(80, n_train_dates=4000) == 400
    assert resolve_dynamic_a_steps(0, n_train_dates=4000) == 400
    assert resolve_dynamic_a_steps(0, n_train_dates=110) == 80
    # CLI default requested=0 must not become a 1-step no-op
    # (max(1, 0) == 1 was the train-loop footgun).
    assert max(1, int(0)) == 1
    assert train_step_budget(0, n_train_dates=4000) == 400
    assert train_step_budget(0, n_train_dates=110) == 80
    assert train_step_budget(6, n_train_dates=40) == 6


def test_skip_encoder_stats_flags_unused_clone():
    skip = np.linspace(-1.0, 1.0, 50)
    clone = skip.copy()
    unused = skip_encoder_stats(skip, clone)
    assert unused["unused"] is True
    assert unused["resid_std"] < 1e-12
    _ascii_ok(unused["reason"])
    rng = np.random.default_rng(0)
    live = skip_encoder_stats(skip, skip + rng.normal(0.0, 0.4, size=50))
    assert live["unused"] is False
    assert live["resid_std"] > 0.1
    # High corr with a material residual is aligned, not a skip clone.
    aligned = skip_encoder_stats(skip, skip * 1.05)
    assert aligned["near_skip"] is True
    assert aligned["unused"] is False
    assert aligned["resid_std"] > 1e-3


def test_rank_score_pred_pos_gap_masks_down_forecasts():
    df = pd.DataFrame(
        {
            "pred": [1.0, 2.0, 3.0],
            "pred_r": [0.01, -0.02, 0.03],
            "date": [1, 1, 1],
        }
    )
    s = rank_score(df, "pred_pos_gap")
    assert np.isfinite(s[0]) and np.isfinite(s[2])
    assert not np.isfinite(s[1])


def test_logit2_is_train_only_and_finite():
    rng = np.random.default_rng(1)
    pred = rng.normal(size=200)
    pred_r = 0.01 * pred + rng.normal(scale=0.005, size=200)
    r_on = pred_r + rng.normal(scale=0.01, size=200)
    spec = fit_logit2_up(pred, pred_r, r_on)
    p = apply_logit2_up(pred, pred_r, spec)
    assert p.shape == pred.shape
    assert np.isfinite(p).all()
    assert (p > 0).all() and (p < 1).all()


def test_high_drift_weekdays_keeps_above_pool():
    # Mon/Tue high up-rate, Wed-Fri low. Monday=0.
    wd = np.array([0, 1, 2, 3, 4] * 20)
    r = np.where(wd <= 1, 0.01, -0.01).astype(np.float64)
    df = pd.DataFrame({"weekday": wd, "r_on": r})
    keep = high_drift_weekdays(df)
    assert keep == [0, 1]


def test_fit_up_rank_is_train_only_and_respects_cover():
    rng = np.random.default_rng(2)
    n_dates, n_names = 16, 10
    rows = []
    for d in range(n_dates):
        for i in range(n_names):
            pred = float(rng.normal())
            r = 0.02 * pred + float(rng.normal(scale=0.01))
            rows.append(
                {
                    "symbol": f"S{i:02d}",
                    "date": d,
                    "weekday": int(d % 5),
                    "pred": pred,
                    "pred_r": 0.01 * pred,
                    "r_on": r,
                    "close": 100.0,
                    "next_open": 100.0 * np.exp(r),
                    "implied_open": 100.0 * np.exp(0.01 * pred),
                    "y": pred,
                    "scale": 0.01,
                }
            )
    df = pd.DataFrame(rows)
    fit = fit_overnight_up_rank_on_train(df, min_names=3)
    assert fit["fit_split"] == "train"
    chosen = fit["chosen"]
    assert chosen
    assert float(chosen.get("cover_full") or 0.0) >= 0.05
    assert str(chosen.get("kind")) in {
        "pred",
        "pred_r",
        "pred_pos_gap",
        "logit2",
        "cs_blend",
    }


def test_up_rank_promote_requires_60_or_e_lift_and_cover():
    thin = decide_overnight_up_rank_promote(
        val_chosen={"up_pct": 90.0, "coverage": 0.02, "uncond_up_pct": 54.0},
        val_e={"up_pct": 58.0},
        chosen={"kind": "pred_r", "q": 0.95},
    )
    assert thin["promote_up_rank"] is False
    assert thin["reached_60"] is False
    _ascii_ok(thin["reason"])
    hit = decide_overnight_up_rank_promote(
        val_chosen={
            "up_pct": 61.0,
            "cover_full": 0.06,
            "coverage": 0.06,
            "uncond_up_pct": 54.3,
            "excess_pp": 6.7,
        },
        val_e={"up_pct": 58.14},
        chosen={"kind": "logit2", "q": 0.93, "abs_q": 0.80},
    )
    assert hit["promote_up_rank"] is True
    assert hit["reached_60"] is True
    assert hit["gated_on"] == "val"
    assert hit["live_book_unchanged"] is True
    _ascii_ok(hit["reason"])
    text = format_overnight_up_rank_block(
        {
            "overnight_up_rank": {
                "fit": {"chosen": {"kind": "logit2", "q": 0.93, "abs_q": 0.8, "blend_alpha": 0.5, "dow_mode": "all"}, "fit_split": "train", "n_candidates": 12},
                "compare": {
                    "val": {"up_pct": 61.0, "excess_pp": 6.7, "cover_full": 0.06, "n": 100},
                    "test": {"up_pct": 56.7, "cover_full": 0.07},
                    "val_e": {"up_pct": 58.14, "coverage": 0.069},
                    "test_e": {"up_pct": 56.72},
                },
                "promotion": hit,
            }
        }
    )
    assert "PROMOTE OVERNIGHT-UP RANK? YES" in text
    _ascii_ok(text)
