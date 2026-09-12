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
    evaluate_overnight_up_rank,
    fit_logit2_up,
    apply_logit2_up,
    fit_logit4_up,
    apply_logit4_up,
    fit_overnight_up_rank_on_train,
    format_overnight_up_rank_block,
    high_disp_threshold,
    high_drift_weekdays,
    pick_up_rank_on_val,
    rank_score,
    score_boolean_sleeve,
    spec_id,
    vote_mask,
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
        "pred_r_pos",
        "logit2",
        "logit4",
        "vote_z",
        "gap_turn",
        "cs_blend",
        "vote_k",
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


def test_rank_score_pred_r_pos_masks_down_gaps():
    df = pd.DataFrame(
        {
            "pred": [1.0, 2.0, 3.0],
            "pred_r": [0.01, -0.02, 0.03],
            "date": [1, 1, 1],
        }
    )
    s = rank_score(df, "pred_r_pos")
    assert s[0] == 0.01 and s[2] == 0.03
    assert not np.isfinite(s[1])


def test_pick_up_rank_on_val_ignores_train_greedy_and_test():
    """Liquid autopsy: TRAIN logit2 62% must not beat a milder VAL winner."""
    train_rows = [
        {
            "kind": "logit2",
            "q": 0.95,
            "abs_q": 0.70,
            "blend_alpha": 0.5,
            "dow_mode": "all",
            "abs_tau": 0.2,
            "up_pct": 62.15,
            "cover_full": 0.052,
            "excess_pp": 8.0,
        },
        {
            "kind": "pred_r",
            "q": 0.80,
            "abs_q": 0.0,
            "blend_alpha": 0.5,
            "dow_mode": "all",
            "abs_tau": 0.0,
            "up_pct": 58.50,
            "cover_full": 0.10,
            "excess_pp": 4.0,
        },
        {
            "kind": "pred",
            "q": 0.90,
            "abs_q": 0.60,
            "blend_alpha": 0.5,
            "dow_mode": "all",
            "abs_tau": 0.1,
            "up_pct": 59.00,
            "cover_full": 0.07,
            "excess_pp": 4.5,
        },
    ]
    val_rows = [
        {**train_rows[0], "up_pct": 57.52, "cover_full": 0.056, "excess_pp": 3.2},
        {**train_rows[1], "up_pct": 58.90, "cover_full": 0.11, "excess_pp": 4.6},
        {**train_rows[2], "up_pct": 61.20, "cover_full": 0.065, "excess_pp": 6.9},
    ]
    # A TEST-only winner must not be selectable (not in val_rows).
    picked = pick_up_rank_on_val(train_rows, val_rows)
    assert picked["select_split"] == "val"
    assert picked["n_hit_60"] == 1.0
    chosen = picked["chosen"]
    assert chosen["kind"] == "pred"
    assert chosen["q"] == 0.90
    assert chosen["val_up_pct"] == 61.20
    # Among 60% hits, higher VAL up wins; a fatter 60.4% must not beat 61.2%.
    fat = {
        "kind": "cs_blend",
        "q": 0.80,
        "abs_q": 0.0,
        "blend_alpha": 0.0,
        "dow_mode": "all",
        "abs_tau": 0.0,
        "up_pct": 60.10,
        "cover_full": 0.12,
        "excess_pp": 6.0,
    }
    train_rows.append(fat)
    val_rows.append({**fat, "up_pct": 60.40, "cover_full": 0.12, "excess_pp": 6.1})
    picked2 = pick_up_rank_on_val(train_rows, val_rows)
    assert picked2["chosen"]["kind"] == "pred"
    assert picked2["chosen"]["val_up_pct"] == 61.20
    # Equal VAL up: more cover wins.
    twin = {
        "kind": "pred_r_pos",
        "q": 0.90,
        "abs_q": 0.50,
        "blend_alpha": 0.5,
        "dow_mode": "all",
        "abs_tau": 0.05,
        "up_pct": 59.00,
        "cover_full": 0.09,
        "excess_pp": 4.5,
    }
    train_rows.append(twin)
    val_rows.append({**twin, "up_pct": 61.20, "cover_full": 0.09, "excess_pp": 6.9})
    picked3 = pick_up_rank_on_val(train_rows, val_rows)
    assert picked3["chosen"]["kind"] == "pred_r_pos"
    assert picked3["chosen"]["val_cover_full"] == 0.09
    assert spec_id(picked3["chosen"])[0] == "pred_r_pos"


def test_pick_up_rank_rejects_thin_val_cover_and_train_losers():
    train_rows = [
        {
            "kind": "logit2",
            "q": 0.93,
            "abs_q": 0.80,
            "blend_alpha": 0.5,
            "dow_mode": "all",
            "up_pct": 70.0,
            "cover_full": 0.03,
            "excess_pp": 15.0,
        },
        {
            "kind": "pred",
            "q": 0.80,
            "abs_q": 0.0,
            "blend_alpha": 0.5,
            "dow_mode": "all",
            "up_pct": 52.0,
            "cover_full": 0.10,
            "excess_pp": -2.0,
        },
        {
            "kind": "pred_r",
            "q": 0.80,
            "abs_q": 0.0,
            "blend_alpha": 0.5,
            "dow_mode": "all",
            "up_pct": 56.0,
            "cover_full": 0.09,
            "excess_pp": 1.5,
        },
    ]
    val_rows = [
        {**train_rows[0], "up_pct": 80.0, "cover_full": 0.08},
        {**train_rows[1], "up_pct": 59.0, "cover_full": 0.10},
        {**train_rows[2], "up_pct": 58.2, "cover_full": 0.09},
    ]
    picked = pick_up_rank_on_val(train_rows, val_rows)
    # Thin TRAIN cover and TRAIN excess < 0 are dropped; leftover is pred_r.
    assert picked["chosen"]["kind"] == "pred_r"
    assert picked["n_val_ok"] == 1.0


def test_evaluate_val_select_does_not_use_test_labels():
    """VAL-good / TEST-bad sleeve must still be chosen (no TEST peek)."""
    rng = np.random.default_rng(4)
    rows = []
    for d in range(30):
        split = "train" if d < 16 else ("val" if d < 23 else "test")
        for i in range(10):
            pred = float(rng.normal())
            pred_r = 0.01 * pred
            if split == "val":
                r = 0.03 * pred_r / 0.01 + float(rng.normal(scale=0.005))
            elif split == "test":
                r = -0.03 * pred_r / 0.01 + float(rng.normal(scale=0.005))
            else:
                r = 0.015 * pred + float(rng.normal(scale=0.01))
            rows.append(
                {
                    "symbol": f"S{i:02d}",
                    "date": d,
                    "weekday": int(d % 5),
                    "pred": pred,
                    "pred_r": pred_r,
                    "r_on": r,
                    "close": 100.0,
                    "next_open": 100.0 * np.exp(r),
                    "implied_open": 100.0 * np.exp(pred_r),
                    "y": pred,
                    "scale": 0.01,
                    "split": split,
                }
            )
    df = pd.DataFrame(rows)
    frames = {
        "train": df[df["split"] == "train"].drop(columns=["split"]),
        "val": df[df["split"] == "val"].drop(columns=["split"]),
        "test": df[df["split"] == "test"].drop(columns=["split"]),
    }
    payload = evaluate_overnight_up_rank(
        frames, min_names=3, e_val={"up_pct": 55.0}, e_test={"up_pct": 54.0}
    )
    assert payload["fit"]["select_split"] == "val"
    assert payload["autopsy"]["test_used"] is False
    assert payload["autopsy"]["select_split"] == "val"
    assert payload["promotion"]["gated_on"] == "val"
    text = format_overnight_up_rank_block({"overnight_up_rank": payload})
    assert "VAL-select" in text or "VAL pick" in text
    assert "not used" in text
    _ascii_ok(text)


def test_logit4_and_vote_z_and_gap_turn_are_finite():
    rng = np.random.default_rng(3)
    pred = rng.normal(size=200)
    pred_r = 0.01 * pred + rng.normal(scale=0.004, size=200)
    turn = rng.normal(size=200)
    r_on = pred_r + 0.002 * turn + rng.normal(scale=0.01, size=200)
    spec = fit_logit4_up(pred, pred_r, r_on, turn)
    p = apply_logit4_up(pred, pred_r, turn, spec)
    assert p.shape == pred.shape
    assert np.isfinite(p).all()
    df = pd.DataFrame(
        {
            "pred": pred[:12],
            "pred_r": pred_r[:12],
            "turnover_z": turn[:12],
            "date": np.repeat([1, 2, 3], 4),
        }
    )
    vz = rank_score(df, "vote_z")
    gt = rank_score(df, "gap_turn")
    assert vz.shape == (12,) and gt.shape == (12,)
    assert np.isfinite(vz).all()


def test_high_disp_threshold_keeps_wide_days():
    rows = []
    for d in range(10):
        sd = 0.02 if d >= 5 else 0.002
        for i in range(8):
            rows.append({"date": d, "pred_r": float((i - 3) * sd), "pred": 0.0})
    df = pd.DataFrame(rows)
    tau = high_disp_threshold(df, q=0.50)
    assert tau > 0.002
    from forecast.overnight_up_rank import _apply_disp_filter

    sub = _apply_disp_filter(df, tau)
    assert set(sub["date"].unique()) <= set(range(5, 10))


def test_vote_mask_requires_k_members():
    rng = np.random.default_rng(5)
    rows = []
    for d in range(8):
        for i in range(10):
            pred = float(rng.normal())
            rows.append(
                {
                    "symbol": f"S{i:02d}",
                    "date": d,
                    "pred": pred,
                    "pred_r": 0.01 * pred,
                    "r_on": 0.01 * pred,
                    "turnover_z": 0.0,
                }
            )
    df = pd.DataFrame(rows)
    a = {"kind": "pred", "q": 0.80, "abs_tau": 0.0, "blend_alpha": 0.5}
    b = {"kind": "pred_r", "q": 0.80, "abs_tau": 0.0, "blend_alpha": 0.5}
    m2 = vote_mask(df, [a, b], 2, min_names=3)
    m1 = vote_mask(df, [a, b], 1, min_names=3)
    assert int(m2.sum()) <= int(m1.sum())
    assert int(m1.sum()) > 0
    scored = score_boolean_sleeve(df, m2, n_full=len(df))
    assert scored["cover_full"] == float(m2.mean())
    if int(m2.sum()) > 0:
        assert np.isfinite(scored["up_pct"])


def test_stop_60_line_is_ascii_when_no_hit():
    miss = decide_overnight_up_rank_promote(
        val_chosen={
            "up_pct": 59.07,
            "cover_full": 0.067,
            "uncond_up_pct": 53.6,
            "excess_pp": 5.48,
        },
        val_e={"up_pct": 58.14},
        chosen={"kind": "pred_pos_gap", "q": 0.70, "abs_q": 0.80},
    )
    assert miss["reached_60"] is False
    assert miss["promote_up_rank"] is True
    text = format_overnight_up_rank_block(
        {
            "overnight_up_rank": {
                "fit": {
                    "chosen": {
                        "kind": "pred_pos_gap",
                        "q": 0.70,
                        "abs_q": 0.80,
                        "blend_alpha": 0.5,
                        "dow_mode": "all",
                        "disp_mode": "all",
                    },
                    "fit_split": "train",
                    "select_split": "val",
                    "n_candidates": 243,
                    "train_greedy": {"kind": "pred", "q": 0.90, "abs_q": 0.80, "up_pct": 61.98},
                },
                "compare": {
                    "val": {"up_pct": 59.07, "excess_pp": 5.48, "cover_full": 0.067, "n": 100},
                    "test": {"up_pct": 57.07, "cover_full": 0.078},
                    "val_e": {"up_pct": 58.14, "coverage": 0.069},
                    "test_e": {"up_pct": 56.72},
                    "train_greedy_val": {"up_pct": 58.60},
                },
                "autopsy": {
                    "n_val_ok": 223,
                    "n_hit_60": 0,
                    "best_val_up": 59.07,
                    "train_greedy_cliff_pp": 3.37,
                },
                "promotion": miss,
            }
        }
    )
    assert "REACHED 60%? NO" in text
    assert "STOP 60%" in text
    assert "best_val_up=59.07%" in text
    _ascii_ok(text)
