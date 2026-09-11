"""Overnight long-short sleeves, short constraints, and VAL-only LS gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.accuracy import sleeve_book_block
from forecast.backtest import book_pnl, conviction_long_weights, sleeve_direction_from_weights
from forecast.shorting import (
    IR_LIFT,
    SHORT_EXCESS_LIFT_PP,
    decide_conviction_live_promote,
    decide_ic_gate_promote,
    decide_ic_scale_promote,
    decide_lo_promote,
    decide_weekday_promote,
    decide_sector_promote,
    decide_disp_gate_promote,
    decide_ensemble_promote,
    decide_adaptive_ensemble_promote,
    blend_cs_scores,
    blend_cs_scores_adaptive,
    adaptive_alpha_series,
    map_ics_to_alpha,
    ENSEMBLE_ALPHAS,
    ADAPTIVE_WINDOWS,
    ADAPTIVE_RULES,
    FIXED_ENSEMBLE_ALPHA,
    fit_ensemble_on_train,
    fit_adaptive_ensemble_on_train,
    STICKY_ENTERS,
    STICKY_EXITS,
    decide_sticky_promote,
    fit_sticky_on_train,
    decide_lo_refine,
    decide_ls_promote,
    evaluate_overnight_shorting,
    format_shorting_report,
    frame_to_wide,
    split_shorting_metrics,
    val_knob_grid,
    val_long_only_grid,
)
from forecast.synthetic import write_cs_overnight_universe
from forecast.universe import hedge_symbol_for, is_equity_name


def test_sleeve_book_block_short_selects_bottom_pred_and_scores_down():
    df = pd.DataFrame(
        {
            "date": [1, 1, 1, 1, 2, 2, 2, 2],
            "pred": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            "r_on": [0.01, 0.01, 0.01, -0.02, -0.01, 0.01, 0.01, 0.01],
        }
    )
    block = sleeve_book_block(df, score_col="pred", side="short", q=0.25, min_names=3)
    # Bottom 25% of 4 names is the min on each date (pred=0).
    assert block["side"] == "short"
    assert block["n"] == 2.0
    # Date1 bottom is up, date2 bottom is down -> 50% down vs 25% uncond (2/8).
    assert abs(block["down_pct"] - 50.0) < 1e-9
    assert abs(block["uncond_down_pct"] - 25.0) < 1e-9
    assert block["excess_pp"] == pytest.approx(25.0)


def test_sleeve_direction_from_weights_splits_long_up_and_short_down():
    dates = pd.bdate_range("2022-01-03", periods=2)
    names = ["A", "B", "C", "D"]
    w = pd.DataFrame(
        [[0.5, 0.0, 0.0, -0.5], [0.5, 0.0, 0.0, -0.5]],
        index=dates,
        columns=names,
    )
    r = pd.DataFrame(
        [[0.01, 0.01, -0.01, -0.02], [0.01, -0.01, 0.01, 0.02]],
        index=dates,
        columns=names,
    )
    out = sleeve_direction_from_weights(w, r)
    # Longs: both dates A is up. Shorts: date1 D down, date2 D up -> 50% down.
    assert out["long_n"] == 2.0
    assert out["long_up_pct"] == pytest.approx(100.0)
    assert out["short_n"] == 2.0
    assert out["short_down_pct"] == pytest.approx(50.0)
    assert out["uncond_down_pct"] == pytest.approx(37.5)  # 3/8 downs


def test_book_pnl_overnight_r_prints_sleeve_after_locate():
    dates = pd.bdate_range("2022-01-03", periods=40)
    names = [f"S{i}" for i in range(10)]
    pred = pd.DataFrame(
        np.tile(np.linspace(-1, 1, 10), (40, 1)), index=dates, columns=names
    )
    realized = pred * 0.02
    r_on = pred * 0.01
    # High turnover on the short tail so locate does not wipe the whole leg.
    tz = pd.DataFrame(
        np.tile(np.linspace(2, -2, 10), (40, 1)), index=dates, columns=names
    )
    stats = book_pnl(
        pred,
        realized,
        holding="overnight",
        hold_halflife=0.0,
        vol_target=0.0,
        causal_vol=False,
        round_trip_bps=10.0,
        locate_pctile=0.3,
        min_names=8,
        turnover_z=tz,
        overnight_r=r_on,
    )
    sleeve = stats["sleeve"]
    assert sleeve["long_n"] > 0
    assert sleeve["short_n"] > 0
    assert np.isfinite(sleeve["long_up_pct"])
    assert np.isfinite(sleeve["short_down_pct"])
    assert stats["mean_cost_unlev_bp"] == pytest.approx(stats["mean_cost_unlev"] * 1e4)


def test_decide_ls_promote_ignores_test_and_requires_short_skill():
    juicy_test = {
        "sleeves": {"short_bottom20": {"excess_pp": 9.9}},
        "books": {
            "live_locate": {"unlevered_net_ir": 4.0},
            "live_long_only": {"unlevered_net_ir": 0.1},
        },
    }
    no_skill = decide_ls_promote(
        {
            "sleeves": {"short_bottom20": {"excess_pp": -1.0}},
            "books": {
                "live_locate": {"unlevered_net_ir": 2.0},
                "live_long_only": {"unlevered_net_ir": 1.0},
            },
        }
    )
    assert no_skill["promote_ls"] is False
    assert no_skill["default_book"] == "live_long_only"
    assert "no skill" in no_skill["reason"].lower()
    assert no_skill["gated_on"] == "val"
    # TEST numbers must not be consulted (function takes VAL only).
    assert juicy_test["sleeves"]["short_bottom20"]["excess_pp"] > SHORT_EXCESS_LIFT_PP

    costs_wipe = decide_ls_promote(
        {
            "sleeves": {"short_bottom20": {"excess_pp": 1.5}},
            "books": {
                "live_locate": {"unlevered_net_ir": 1.0},
                "live_long_only": {"unlevered_net_ir": 1.8},
            },
        }
    )
    assert costs_wipe["promote_ls"] is False
    assert costs_wipe["short_skill"] is True
    assert costs_wipe["ir_beats"] is False
    assert "borrow" in costs_wipe["reason"].lower() or "wipe" in costs_wipe["reason"].lower()

    yes = decide_ls_promote(
        {
            "sleeves": {"short_bottom20": {"excess_pp": 1.5}},
            "books": {
                "live_locate": {"unlevered_net_ir": 1.0 + IR_LIFT + 0.01},
                "live_long_only": {"unlevered_net_ir": 1.0},
            },
        }
    )
    assert yes["promote_ls"] is True
    assert yes["default_book"] == "live_locate"


def test_conviction_long_weights_matches_top_q_and_abs_floor():
    s = pd.Series([0.0, 0.5, 1.0, 2.0], index=list("abcd"))
    w = conviction_long_weights(s, q=0.75, abs_tau=0.0, min_names=3)
    assert w["d"] == pytest.approx(1.0)
    assert w[["a", "b", "c"]].sum() == pytest.approx(0.0)
    w2 = conviction_long_weights(s, q=0.75, abs_tau=2.5, min_names=3)
    assert w2.sum() == pytest.approx(0.0)
    s3 = pd.Series([-2.0, -1.0, 1.0, 2.0], index=list("abcd"))
    w3 = conviction_long_weights(
        s3, q=0.50, abs_tau=0.0, min_names=3, require_above_median=True
    )
    assert w3[["c", "d"]].sum() == pytest.approx(1.0)
    assert w3[["a", "b"]].sum() == pytest.approx(0.0)


def test_decide_conviction_live_promote_is_val_only():
    chosen = {"q": 0.90, "abs_tau": 0.3, "abs_q": 0.70}
    q20 = {
        "unlevered_net_ir": 1.00,
        "unlevered_max_dd": -0.20,
        "mean_turnover": 1.6,
        "coverage": 0.25,
    }
    ok = {
        "unlevered_net_ir": 1.10,
        "unlevered_max_dd": -0.22,
        "mean_turnover": 1.8,
        "coverage": 0.08,
    }
    yes = decide_conviction_live_promote(
        val_q20=q20, val_chosen=ok, chosen=chosen
    )
    assert yes["promote_conviction_live"] is True
    assert yes["gated_on"] == "val"
    assert yes["default_book_unchanged"] is True
    weak_ir = dict(ok, unlevered_net_ir=1.02)
    no_ir = decide_conviction_live_promote(
        val_q20=q20, val_chosen=weak_ir, chosen=chosen
    )
    assert no_ir["promote_conviction_live"] is False
    bad_dd = dict(ok, unlevered_max_dd=-0.30)
    no_dd = decide_conviction_live_promote(
        val_q20=q20, val_chosen=bad_dd, chosen=chosen
    )
    assert no_dd["promote_conviction_live"] is False
    thin = dict(ok, coverage=0.03)
    no_cover = decide_conviction_live_promote(
        val_q20=q20, val_chosen=thin, chosen=chosen
    )
    assert no_cover["promote_conviction_live"] is False
    same = decide_conviction_live_promote(
        val_q20=q20,
        val_chosen=dict(ok, unlevered_net_ir=9.9, coverage=0.25),
        chosen={"q": 0.80, "abs_tau": 0.0, "abs_q": 0.0},
    )
    assert same["promote_conviction_live"] is False
    juicy_test = {"unlevered_net_ir": 99.0, "unlevered_max_dd": 0.0, "coverage": 0.99}
    assert juicy_test["unlevered_net_ir"] > yes["val_ir"]


def test_decide_lo_promote_requires_ir_lift_not_worse_dd():
    base = {
        "name": "lo_q20_adv0",
        "unlevered_net_ir": 1.0,
        "unlevered_max_dd": -0.20,
        "weighting": "quantile",
        "quantile": 0.2,
        "adv_floor_pctile": 0.0,
    }
    no_lift = decide_lo_promote({"baseline": base, "best": dict(base)})
    assert no_lift["promote_lo"] is False
    assert no_lift["gated_on"] == "val"
    assert no_lift["spec"]["quantile"] == 0.2

    yes = decide_lo_promote(
        {
            "baseline": base,
            "best": {
                "name": "lo_q15_adv0",
                "unlevered_net_ir": 1.20,
                "unlevered_max_dd": -0.21,
                "weighting": "quantile",
                "quantile": 0.15,
                "adv_floor_pctile": 0.0,
            },
        }
    )
    assert yes["promote_lo"] is True
    assert yes["spec"]["quantile"] == 0.15

    dd_worse = decide_lo_promote(
        {
            "baseline": base,
            "best": {
                "name": "lo_q10_adv0",
                "unlevered_net_ir": 1.20,
                "unlevered_max_dd": -0.40,
                "weighting": "quantile",
                "quantile": 0.10,
                "adv_floor_pctile": 0.0,
            },
        }
    )
    assert dd_worse["promote_lo"] is False
    assert dd_worse["spec"]["quantile"] == 0.2


def test_decide_lo_refine_test_veto_does_not_pick_another_cell():
    grid = {
        "baseline": {
            "name": "lo_q20_equal_c0",
            "unlevered_net_ir": 1.0,
            "unlevered_max_dd": -0.20,
            "weighting": "quantile",
            "quantile": 0.2,
            "long_size": "equal",
            "conf_pctile": 0.0,
            "adv_floor_pctile": 0.0,
        },
        "best": {
            "name": "lo_q15_abs_pred_c50",
            "unlevered_net_ir": 1.30,
            "unlevered_max_dd": -0.18,
            "weighting": "quantile",
            "quantile": 0.15,
            "long_size": "abs_pred",
            "conf_pctile": 0.5,
            "adv_floor_pctile": 0.0,
        },
    }
    collapsed = decide_lo_refine(
        grid,
        test_baseline={"unlevered_net_ir": 2.0},
        test_best={"unlevered_net_ir": 1.0},
    )
    assert collapsed["promote_lo"] is False
    assert collapsed["test_veto"] is True
    assert collapsed["spec"]["quantile"] == 0.2
    assert collapsed["spec"]["long_size"] == "equal"

    ok = decide_lo_refine(
        grid,
        test_baseline={"unlevered_net_ir": 2.0},
        test_best={"unlevered_net_ir": 1.98},
    )
    assert ok["promote_lo"] is True
    assert ok["test_veto"] is False
    assert ok["spec"]["long_size"] == "abs_pred"


def test_decide_ic_gate_promote_is_val_only_and_needs_coverage():
    always = {"unlevered_net_ir": 1.0, "unlevered_max_dd": -0.20}
    gated = {
        "unlevered_net_ir": 1.20,
        "unlevered_max_dd": -0.18,
        "ic_gate_coverage": 0.55,
    }
    juicy_test = {"unlevered_net_ir": 9.9, "unlevered_max_dd": 0.0, "ic_gate_coverage": 0.99}
    yes = decide_ic_gate_promote(
        val_always=always,
        val_gated=gated,
        chosen={"window": 60, "tau": 0.0, "ic_gate_coverage": 0.7},
    )
    assert yes["promote_ic_gate"] is True
    assert yes["gated_on"] == "val"
    assert yes["fit_split"] == "train"
    # TEST numbers are not arguments — cannot flip the call.
    assert juicy_test["unlevered_net_ir"] > yes["ir_gated"]

    thin = decide_ic_gate_promote(
        val_always=always,
        val_gated={**gated, "ic_gate_coverage": 0.10},
        chosen={"window": 60, "tau": 0.04},
    )
    assert thin["promote_ic_gate"] is False
    assert thin["spec"]["window"] == 0

    no_fit = decide_ic_gate_promote(val_always=always, val_gated=gated, chosen={})
    assert no_fit["promote_ic_gate"] is False


def test_decide_ic_scale_promote_is_val_only():
    always = {"unlevered_net_ir": 1.0, "unlevered_max_dd": -0.20}
    scaled = {
        "unlevered_net_ir": 1.20,
        "unlevered_max_dd": -0.18,
        "mean_ic_scale": 0.7,
    }
    juicy_test = {"unlevered_net_ir": 9.9}
    yes = decide_ic_scale_promote(
        val_always=always,
        val_scaled=scaled,
        chosen={"window": 20, "tau": 0.04, "s_max": 1.25},
    )
    assert yes["promote_ic_scale"] is True
    assert yes["gated_on"] == "val"
    assert yes["spec"]["s_max"] == 1.25
    assert juicy_test["unlevered_net_ir"] > yes["ir_scaled"]

    no = decide_ic_scale_promote(
        val_always=always,
        val_scaled={**scaled, "unlevered_net_ir": 1.01},
        chosen={"window": 20, "tau": 0.04, "s_max": 1.25},
    )
    assert no["promote_ic_scale"] is False
    assert no["spec"]["window"] == 0


def test_decide_weekday_promote_is_val_only_and_needs_coverage():
    always = {
        "name": "wd_always",
        "weekday_mask": "always",
        "unlevered_net_ir": 1.0,
        "unlevered_max_dd": -0.20,
        "weekday_coverage": 1.0,
    }
    skip_fr = {
        "name": "wd_flat_friday",
        "weekday_mask": "flat_friday",
        "unlevered_net_ir": 1.20,
        "unlevered_max_dd": -0.18,
        "weekday_coverage": 0.80,
    }
    thin_we = {
        "name": "wd_weekend_only",
        "weekday_mask": "weekend_only",
        "unlevered_net_ir": 9.9,
        "unlevered_max_dd": 0.0,
        "weekday_coverage": 0.18,
    }
    juicy_test = {"unlevered_net_ir": 99.0}
    yes = decide_weekday_promote(
        {
            "baseline": always,
            "best": skip_fr,
            "rows": [skip_fr, always, thin_we],
            "eligible": [skip_fr, always],
        }
    )
    assert yes["promote_weekday"] is True
    assert yes["gated_on"] == "val"
    assert yes["spec"]["weekday_mask"] == "flat_friday"
    assert juicy_test["unlevered_net_ir"] > yes["ir_gated"]

    no_cover = decide_weekday_promote(
        {"baseline": always, "best": thin_we, "rows": [thin_we, always]}
    )
    # weekend_only juiciness is ineligible at 18% coverage; if it is still
    # passed as "best", coverage floor must block.
    assert no_cover["promote_weekday"] is False
    assert no_cover["spec"]["weekday_mask"] == "always"

    no_lift = decide_weekday_promote(
        {
            "baseline": always,
            "best": {**skip_fr, "unlevered_net_ir": 1.01},
            "rows": [always, skip_fr],
        }
    )
    assert no_lift["promote_weekday"] is False


def test_decide_sector_promote_is_val_only():
    spy = {"unlevered_net_ir": 1.0, "unlevered_max_dd": -0.20}
    sector = {"unlevered_net_ir": 1.20, "unlevered_max_dd": -0.18}
    juicy_test = {"unlevered_net_ir": 9.9}
    yes = decide_sector_promote(
        val_spy=spy, val_sector=sector, n_sector_hedges=8, n_names=12
    )
    assert yes["promote_sector"] is True
    assert yes["gated_on"] == "val"
    assert juicy_test["unlevered_net_ir"] > yes["ir_sector"]

    no_files = decide_sector_promote(
        val_spy=spy, val_sector=sector, n_sector_hedges=0, n_names=12
    )
    assert no_files["promote_sector"] is False

    no_lift = decide_sector_promote(
        val_spy=spy,
        val_sector={"unlevered_net_ir": 1.01, "unlevered_max_dd": -0.18},
        n_sector_hedges=8,
        n_names=12,
    )
    assert no_lift["promote_sector"] is False


def test_decide_disp_gate_promote_is_val_only_and_needs_coverage():
    always = {"unlevered_net_ir": 1.0, "unlevered_max_dd": -0.20}
    gated = {
        "unlevered_net_ir": 1.20,
        "unlevered_max_dd": -0.18,
        "disp_gate_coverage": 0.55,
    }
    juicy_test = {"unlevered_net_ir": 9.9}
    yes = decide_disp_gate_promote(
        val_always=always,
        val_gated=gated,
        chosen={"kind": "cc", "window": 1, "tau": 0.02, "q": 0.8},
    )
    assert yes["promote_disp_gate"] is True
    assert yes["gated_on"] == "val"
    assert yes["fit_split"] == "train"
    assert juicy_test["unlevered_net_ir"] > yes["ir_gated"]

    thin = decide_disp_gate_promote(
        val_always=always,
        val_gated={**gated, "disp_gate_coverage": 0.10},
        chosen={"kind": "cc", "window": 1, "tau": 0.02, "q": 0.9},
    )
    assert thin["promote_disp_gate"] is False
    assert thin["spec"]["kind"] == ""


def test_blend_cs_scores_w1_is_z_a_and_w0_is_z_b():
    dates = pd.bdate_range("2022-01-03", periods=4)
    names = ["A", "B", "C", "D"]
    pred_a = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]] * 4, index=dates, columns=names
    )
    pred_b = pd.DataFrame(
        [[4.0, 3.0, 2.0, 1.0]] * 4, index=dates, columns=names
    )
    z1 = blend_cs_scores(pred_a, pred_b, 1.0)
    z0 = blend_cs_scores(pred_a, pred_b, 0.0)
    mid = blend_cs_scores(pred_a, pred_b, 0.5)
    assert z1.loc[dates[0], "D"] > z1.loc[dates[0], "A"]
    assert z0.loc[dates[0], "A"] > z0.loc[dates[0], "D"]
    assert mid.loc[dates[0]].abs().max() < 1e-9
    # Mutating B must not change w=1.
    pred_b2 = pred_b * -1.0
    assert blend_cs_scores(pred_a, pred_b2, 1.0).equals(z1)


def test_decide_ensemble_promote_is_val_only():
    val_a = {
        "name": "ens_a1.00",
        "alpha": 1.0,
        "weight": 1.0,
        "unlevered_net_ir": 1.0,
        "unlevered_max_dd": -0.20,
        "coverage": 1.0,
        "net_ir": 0.8,
    }
    val_b = {
        "name": "ens_a0.50",
        "alpha": 0.5,
        "weight": 0.5,
        "unlevered_net_ir": 1.20,
        "unlevered_max_dd": -0.18,
        "coverage": 1.0,
        "net_ir": 0.95,
    }
    juicy_test = {"unlevered_net_ir": 9.9}
    yes = decide_ensemble_promote(
        val_overnight=val_a,
        val_chosen=val_b,
        chosen={"alpha": 0.5, "weight": 0.5},
    )
    assert yes["promote_ensemble"] is True
    assert yes["gated_on"] == "val"
    assert yes["spec"]["alpha"] == 0.5
    assert juicy_test["unlevered_net_ir"] > yes["ir_ensemble"]

    no = decide_ensemble_promote(
        val_overnight=val_a,
        val_chosen=val_a,
        chosen={"alpha": 1.0, "weight": 1.0},
    )
    assert no["promote_ensemble"] is False
    assert no["spec"]["alpha"] == 1.0

    thin = decide_ensemble_promote(
        val_overnight=val_a,
        val_chosen={**val_b, "coverage": 0.10},
        chosen={"alpha": 0.5, "weight": 0.5},
    )
    assert thin["promote_ensemble"] is False


def test_ensemble_alphas_are_train_grid_without_pure_c2c():
    assert ENSEMBLE_ALPHAS == (0.5, 0.6, 0.7, 0.8, 1.0)
    assert FIXED_ENSEMBLE_ALPHA == pytest.approx(0.70)
    assert ADAPTIVE_WINDOWS == (20, 60, 120)
    assert ADAPTIVE_RULES == ("relu_ratio", "signed_ratio", "softmax")


def test_map_ics_to_alpha_rules():
    assert map_ics_to_alpha(0.2, 0.2, "relu_ratio") == pytest.approx(0.5, abs=1e-6)
    assert map_ics_to_alpha(0.2, -0.1, "relu_ratio") == pytest.approx(1.0, abs=1e-6)
    assert map_ics_to_alpha(-0.2, 0.1, "relu_ratio") == pytest.approx(0.0, abs=1e-6)
    assert map_ics_to_alpha(0.2, 0.2, "signed_ratio") == pytest.approx(0.5, abs=1e-6)
    assert map_ics_to_alpha(-0.2, -0.1, "signed_ratio") == pytest.approx(1.0)
    assert map_ics_to_alpha(0.0, 0.0, "softmax") == pytest.approx(0.5, abs=1e-6)
    assert not np.isfinite(map_ics_to_alpha(float("nan"), 0.1, "relu_ratio"))


def test_adaptive_alpha_is_causal_dates_before_t():
    rng = np.random.default_rng(2)
    n_days, n_names = 40, 8
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    names = [f"S{i}" for i in range(n_names)]
    true = np.linspace(-1.0, 1.0, n_names)
    pred_a = pd.DataFrame(np.tile(true, (n_days, 1)), index=dates, columns=names)
    pred_b = pred_a * 0.35 + rng.normal(0.0, 0.25, size=pred_a.shape)
    pred_b = pd.DataFrame(pred_b, index=dates, columns=names)
    y = pd.DataFrame(np.tile(true, (n_days, 1)), index=dates, columns=names)
    alpha = adaptive_alpha_series(
        pred_a, pred_b, y, window=20, rule="relu_ratio", min_names=6
    )
    t = dates[25]
    t_next = dates[26]
    y_flip_t = y.copy()
    y_flip_t.loc[t] = -y_flip_t.loc[t]
    alpha_flip_t = adaptive_alpha_series(
        pred_a, pred_b, y_flip_t, window=20, rule="relu_ratio", min_names=6
    )
    assert np.isfinite(alpha.loc[t])
    assert alpha.loc[t] == pytest.approx(float(alpha_flip_t.loc[t]), abs=1e-12)
    # Date t's overnight *does* enter α_{t+1} (newest prior IC).
    assert abs(float(alpha.loc[t_next]) - float(alpha_flip_t.loc[t_next])) > 1e-12


def test_blend_cs_scores_adaptive_constant_alpha_matches_fixed():
    dates = pd.bdate_range("2022-01-03", periods=4)
    names = ["A", "B", "C", "D"]
    pred_a = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]] * 4, index=dates, columns=names)
    pred_b = pd.DataFrame([[4.0, 3.0, 2.0, 1.0]] * 4, index=dates, columns=names)
    alpha = pd.Series(1.0, index=dates)
    got = blend_cs_scores_adaptive(pred_a, pred_b, alpha)
    want = blend_cs_scores(pred_a, pred_b, 1.0)
    pd.testing.assert_frame_equal(got, want)
    mid = blend_cs_scores_adaptive(pred_a, pred_b, pd.Series(0.5, index=dates))
    assert mid.loc[dates[0]].abs().max() < 1e-9


def test_decide_adaptive_ensemble_promote_is_val_only():
    val_070 = {
        "name": "ens_a0.70",
        "alpha": 0.70,
        "unlevered_net_ir": 1.10,
        "unlevered_max_dd": -0.20,
        "coverage": 1.0,
        "net_ir": 0.90,
    }
    val_1 = {
        "name": "ens_a1.00",
        "alpha": 1.0,
        "unlevered_net_ir": 1.00,
        "unlevered_max_dd": -0.18,
        "coverage": 1.0,
        "net_ir": 0.80,
    }
    val_adp = {
        "name": "adp_W60_relu_ratio",
        "window": 60,
        "rule": "relu_ratio",
        "unlevered_net_ir": 1.20,
        "unlevered_max_dd": -0.19,
        "coverage": 1.0,
        "net_ir": 1.00,
        "mean_alpha": 0.55,
    }
    juicy_test = {"unlevered_net_ir": 9.9}
    yes = decide_adaptive_ensemble_promote(
        val_adaptive=val_adp,
        val_fixed_070=val_070,
        val_alpha1=val_1,
        chosen={"window": 60, "rule": "relu_ratio"},
    )
    assert yes["promote_adaptive_ensemble"] is True
    assert yes["gated_on"] == "val"
    assert yes["baseline_name"] == "fixed_a0.70"
    assert yes["ir_delta"] == pytest.approx(0.10)
    assert juicy_test["unlevered_net_ir"] > yes["ir_adaptive"]

    no = decide_adaptive_ensemble_promote(
        val_adaptive=val_070,
        val_fixed_070=val_070,
        val_alpha1=val_1,
        chosen={"window": 60, "rule": "relu_ratio"},
    )
    assert no["promote_adaptive_ensemble"] is False

    thin = decide_adaptive_ensemble_promote(
        val_adaptive={**val_adp, "coverage": 0.10},
        val_fixed_070=val_070,
        val_alpha1=val_1,
        chosen={"window": 60, "rule": "relu_ratio"},
    )
    assert thin["promote_adaptive_ensemble"] is False

    worse_dd = decide_adaptive_ensemble_promote(
        val_adaptive={**val_adp, "unlevered_max_dd": -0.30},
        val_fixed_070=val_070,
        val_alpha1=val_1,
        chosen={"window": 60, "rule": "relu_ratio"},
    )
    assert worse_dd["promote_adaptive_ensemble"] is False


def test_fit_adaptive_ensemble_on_train_stays_on_train_grid():
    rng = np.random.default_rng(0)
    n_days, n_names = 80, 8
    dates = np.repeat(np.arange(n_days, dtype=np.int64) + 18000, n_names)
    names = np.tile([f"S{i}" for i in range(n_names)], n_days)
    true = np.tile(np.linspace(-1.0, 1.0, n_names), n_days)
    day_shock = np.repeat(rng.normal(0.0, 0.008, n_days), n_names)
    amp = np.repeat(0.4 + np.abs(rng.normal(1.0, 0.35, n_days)), n_names)
    r_on = true * 0.012 + day_shock
    frame_on = pd.DataFrame(
        {
            "symbol": names,
            "date": dates,
            "pred": true,
            "y": true * amp,
            "r_on": r_on,
            "turnover_z": -true,
            "vol_level": np.full(len(dates), 0.2),
        }
    )
    frame_cc = frame_on.copy()
    frame_cc["pred"] = true * 0.4 + rng.normal(0.0, 0.3, size=len(true))
    fit = fit_adaptive_ensemble_on_train(
        frame_on, frame_cc, min_names=6, vol_target=0.15
    )
    assert fit["fit_split"] == "train"
    assert fit["chosen"]
    assert int(fit["chosen"]["window"]) in set(ADAPTIVE_WINDOWS)
    assert str(fit["chosen"]["rule"]) in set(ADAPTIVE_RULES)
    assert {int(r["window"]) for r in fit["rows"]} <= set(ADAPTIVE_WINDOWS)
    assert {str(r["rule"]) for r in fit["rows"]} <= set(ADAPTIVE_RULES)


def test_fit_ensemble_on_train_picks_overnight_when_c2c_is_anti():
    rng = np.random.default_rng(0)
    n_days, n_names = 40, 8
    dates = np.repeat(np.arange(n_days, dtype=np.int64) + 18000, n_names)
    names = np.tile([f"S{i}" for i in range(n_names)], n_days)
    true = np.tile(np.linspace(-1.0, 1.0, n_names), n_days)
    day_shock = np.repeat(rng.normal(0.0, 0.008, n_days), n_names)
    amp = np.repeat(0.4 + np.abs(rng.normal(1.0, 0.35, n_days)), n_names)
    r_on = true * 0.012 + day_shock
    frame_on = pd.DataFrame(
        {
            "symbol": names,
            "date": dates,
            "pred": true,
            "y": true * amp,
            "r_on": r_on,
            "turnover_z": -true,
            "vol_level": np.full(len(dates), 0.2),
        }
    )
    frame_cc = frame_on.copy()
    # Exact anti-rank: α>0.5 keeps A's order, so TRAIN IR ties and larger α wins.
    frame_cc["pred"] = -true
    fit = fit_ensemble_on_train(frame_on, frame_cc, min_names=6, vol_target=0.15)
    assert fit["fit_split"] == "train"
    assert fit["chosen"]
    assert float(fit["chosen"]["alpha"]) == 1.0
    assert {round(float(r["alpha"]), 2) for r in fit["rows"]} <= set(ENSEMBLE_ALPHAS)


def test_decide_sticky_promote_is_val_only():
    val_q20 = {
        "name": "q20_rebuild",
        "unlevered_net_ir": 1.0,
        "unlevered_max_dd": -0.20,
        "coverage": 1.0,
        "mean_name_churn": 0.30,
        "mean_turnover": 1.0,
    }
    val_st = {
        "name": "sticky_e15_x40",
        "q_enter": 0.15,
        "q_exit": 0.40,
        "unlevered_net_ir": 1.20,
        "unlevered_max_dd": -0.18,
        "coverage": 1.0,
        "mean_name_churn": 0.12,
        "mean_turnover": 1.0,
    }
    juicy_test = {"unlevered_net_ir": 9.9}
    yes = decide_sticky_promote(
        val_baseline=val_q20,
        val_chosen=val_st,
        chosen={"q_enter": 0.15, "q_exit": 0.40},
    )
    assert yes["promote_sticky"] is True
    assert yes["gated_on"] == "val"
    assert yes["spec"]["q_enter"] == 0.15
    assert juicy_test["unlevered_net_ir"] > yes["ir_sticky"]

    no = decide_sticky_promote(
        val_baseline=val_q20,
        val_chosen=val_q20,
        chosen={},
    )
    assert no["promote_sticky"] is False

    thin = decide_sticky_promote(
        val_baseline=val_q20,
        val_chosen={**val_st, "coverage": 0.10},
        chosen={"q_enter": 0.15, "q_exit": 0.40},
    )
    assert thin["promote_sticky"] is False


def test_fit_sticky_on_train_requires_exit_gt_enter():
    rng = np.random.default_rng(1)
    n_days, n_names = 50, 10
    dates = np.repeat(np.arange(n_days, dtype=np.int64) + 18000, n_names)
    names = np.tile([f"S{i}" for i in range(n_names)], n_days)
    true = np.tile(np.linspace(-1.0, 1.0, n_names), n_days)
    amp = np.repeat(0.4 + np.abs(rng.normal(1.0, 0.35, n_days)), n_names)
    day_shock = np.repeat(rng.normal(0.0, 0.008, n_days), n_names)
    frame = pd.DataFrame(
        {
            "symbol": names,
            "date": dates,
            "pred": true,
            "y": true * amp,
            "r_on": true * 0.012 + day_shock,
            "turnover_z": -true,
            "vol_level": np.full(len(dates), 0.2),
        }
    )
    fit = fit_sticky_on_train(frame, min_names=6, vol_target=0.15)
    assert fit["fit_split"] == "train"
    assert fit["chosen"]
    assert float(fit["chosen"]["q_exit"]) > float(fit["chosen"]["q_enter"])
    assert {round(float(r["q_enter"]), 2) for r in fit["rows"]} <= set(STICKY_ENTERS)


def test_synthetic_names_map_to_sector_etfs_and_etfs_are_not_book_names():
    assert hedge_symbol_for("S00", sector_residual=True) == "XLK"
    assert hedge_symbol_for("S05", sector_residual=True) == "XLK"
    assert hedge_symbol_for("S06", sector_residual=True) == "XLF"
    assert hedge_symbol_for("S00", sector_residual=False) == "SPY"
    assert is_equity_name("S00")
    assert not is_equity_name("XLK")
    assert not is_equity_name("XLF")


def test_frame_to_wide_uses_calendar_index():
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "date": [1, 1, 2, 2],
            "pred": [0.1, -0.2, 0.3, -0.4],
        }
    )
    wide = frame_to_wide(df, "pred")
    assert list(wide.columns) == ["A", "B"]
    assert wide.loc[pd.Timestamp("1970-01-02"), "A"] == pytest.approx(0.1)
    assert wide.loc[pd.Timestamp("1970-01-03"), "B"] == pytest.approx(-0.4)


def test_split_shorting_metrics_ls_has_borrow_and_short_nav():
    dates = np.repeat(np.arange(20, dtype=np.int64) + 18000, 8)
    names = np.tile([f"S{i}" for i in range(8)], 20)
    ranks = np.tile(np.linspace(-1.0, 1.0, 8), 20)
    r_on = ranks * 0.01
    df = pd.DataFrame(
        {
            "symbol": names,
            "date": dates,
            "pred": ranks,
            "y": ranks,
            "scale": np.full(len(dates), 0.01),
            "close": np.full(len(dates), 100.0),
            "next_open": 100.0 * np.exp(r_on),
            "r_on": r_on,
            "turnover_z": -ranks,
            "vol_level": np.full(len(dates), 0.2),
            "pred_r": ranks * 0.01,
            "implied_open": 100.0 * np.exp(ranks * 0.01),
            "implied_open_given_hedge": 100.0 * np.exp(r_on),
        }
    )
    out = split_shorting_metrics(df, min_names=6, vol_target=0.0)
    short = out["sleeves"]["short_bottom20"]
    long = out["sleeves"]["long_only_top20"]
    assert short["excess_pp"] > 0
    assert long["excess_pp"] > 0
    ls = out["books"]["live_locate"]
    lo = out["books"]["live_long_only"]
    assert ls["mean_short_nav"] > 0
    assert lo["mean_short_nav"] == pytest.approx(0.0, abs=1e-12)
    assert ls["borrow_bps"] > 0
    assert lo["borrow_bps"] == pytest.approx(0.0)
    assert float((ls.get("cost_parts") or {}).get("borrow") or 0.0) > 0
    assert float((lo.get("cost_parts") or {}).get("borrow") or 0.0) == pytest.approx(0.0)
    grid = val_knob_grid(df, min_names=6, vol_target=0.0)
    assert grid["rows"]
    kinds = {r["kind"] for r in grid["rows"]}
    assert "long_only" in kinds and "live_locate" in kinds
    assert grid["lo_best"]["kind"] == "long_only"
    lo_grid = val_long_only_grid(df, min_names=6, vol_target=0.0)
    assert lo_grid["baseline"]["name"] == "lo_q20_adv0"
    assert lo_grid["rows"]
    assert all(r["kind"] == "long_only" for r in lo_grid["rows"])


def test_synthetic_overnight_short_sleeve_has_skill(tmp_path: Path):
    data_dir = tmp_path / "data"
    write_cs_overnight_universe(
        data_dir, n_names=12, n_days=220, seed=1, rho=0.65, include_sectors=True
    )
    payload = evaluate_overnight_shorting(str(data_dir), "synthetic", log_fn=None)
    assert payload["promotion"]["gated_on"] == "val"
    test = payload["test"]
    val = payload["val"]
    assert test["sleeves"]["short_bottom20"]
    # Planted CS overnight residual: at least one locked split should show
    # short-side down excess (do not require VAL promote after live costs).
    val_xs = float(val["sleeves"]["short_bottom20"].get("excess_pp") or float("nan"))
    test_xs = float(test["sleeves"]["short_bottom20"].get("excess_pp") or float("nan"))
    assert np.isfinite(val_xs) and np.isfinite(test_xs)
    assert max(val_xs, test_xs) > 0.0
    assert "live_locate" in test["books"]
    assert "live_long_only" in test["books"]
    text = format_shorting_report(payload)
    assert "LOCKED VAL" in text
    assert "LOCKED TEST" in text
    assert "short bot20" in text
    assert "DESKTOP" in text
    # TEST juiciness must not flip a no-skill VAL.
    assert payload["promotion"]["short_excess_pp"] == pytest.approx(val_xs)
    assert payload["lo_promotion"]["gated_on"] == "val"
    assert payload["ls_experiment"]["promote_as_default"] is False
    assert payload["lo_refine_promotion"]["gated_on"] == "val"
    assert payload["ic_gate_fit"]["fit_split"] == "train"
    assert payload["ic_gate_promotion"]["gated_on"] == "val"
    assert payload["ic_scale_fit"]["fit_split"] == "train"
    assert payload["ic_scale_promotion"]["gated_on"] == "val"
    assert "PROMOTE IC-SCALE" in text
    assert payload["weekday_promotion"]["gated_on"] == "val"
    assert payload["sector_promotion"]["gated_on"] == "val"
    assert payload["sector_compare"]["n_sector_hedges"] > 0
    assert payload["disp_gate_fit"]["fit_split"] == "train"
    assert payload["disp_gate_promotion"]["gated_on"] == "val"
    assert "PROMOTE IC-GATE" in text
    assert "PROMOTE WEEKDAY MASK" in text
    assert "PROMOTE SECTOR-OVERNIGHT" in text
    assert "PROMOTE DISP-GATE" in text
    assert payload["ensemble_promotion"]["gated_on"] == "val"
    assert payload["ensemble_fit"]["fit_split"] == "train"
    train_a = float((payload["ensemble_fit"].get("chosen") or {}).get("alpha") or 1.0)
    assert train_a in set(ENSEMBLE_ALPHAS)
    assert payload["ensemble_compare"]["train_alpha"] == pytest.approx(train_a)
    assert "PROMOTE OVERNIGHT" in text and "ENSEMBLE" in text
    assert payload["adaptive_ensemble_fit"]["fit_split"] == "train"
    assert payload["adaptive_ensemble_promotion"]["gated_on"] == "val"
    adp_ch = payload["adaptive_ensemble_fit"].get("chosen") or {}
    if adp_ch:
        assert int(adp_ch["window"]) in set(ADAPTIVE_WINDOWS)
        assert str(adp_ch["rule"]) in set(ADAPTIVE_RULES)
    assert "PROMOTE ADAPTIVE OVERNIGHT" in text
    assert payload["sticky_fit"]["fit_split"] == "train"
    assert payload["sticky_promotion"]["gated_on"] == "val"
    chosen_st = payload["sticky_fit"].get("chosen") or {}
    if chosen_st:
        assert float(chosen_st["q_enter"]) in set(STICKY_ENTERS)
        assert float(chosen_st["q_exit"]) in set(STICKY_EXITS)
        assert float(chosen_st["q_exit"]) > float(chosen_st["q_enter"])
    assert "PROMOTE STICKY" in text
    assert "VAL LONG-ONLY GRID" in text
    assert "VAL LONG-ONLY REFINE" in text
    assert "LS HAIRCUT EXPERIMENT" in text
    assert payload["conviction_live"]["fit_split"] == "train"
    assert payload["conviction_live_promotion"]["gated_on"] == "val"
    assert payload["conviction_live_promotion"]["default_book_unchanged"] is True
    assert "PROMOTE CONVICTION LIVE" in text
    assert "chosen" in (payload["conviction_live"].get("val") or {})
    assert "q20" in (payload["conviction_live"].get("test") or {})
