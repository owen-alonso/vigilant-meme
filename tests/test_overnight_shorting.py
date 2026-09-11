"""Overnight long-short sleeves, short constraints, and VAL-only LS gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast.accuracy import sleeve_book_block
from forecast.backtest import book_pnl, sleeve_direction_from_weights
from forecast.shorting import (
    IR_LIFT,
    SHORT_EXCESS_LIFT_PP,
    decide_ls_promote,
    evaluate_overnight_shorting,
    format_shorting_report,
    frame_to_wide,
    split_shorting_metrics,
    val_knob_grid,
)
from forecast.synthetic import write_cs_overnight_universe


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


def test_synthetic_overnight_short_sleeve_has_skill(tmp_path: Path):
    data_dir = tmp_path / "data"
    write_cs_overnight_universe(data_dir, n_names=12, n_days=220, seed=1, rho=0.65)
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
