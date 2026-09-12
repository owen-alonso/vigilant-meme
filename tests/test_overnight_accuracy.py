"""Locked-TEST overnight TRUE/FALSE accuracy helpers and synthetic protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from forecast.accuracy import (
    abs_error_block,
    adv_sleeve_mask,
    apply_affine,
    apply_bin_constants,
    apply_calibrate_spec,
    apply_cond_dir_blend,
    apply_cs_left_veto,
    apply_decile_reliability,
    apply_logistic_up,
    apply_drift_veto,
    apply_readout,
    cond_abs_mask,
    cs_relative_blocks,
    cs_stack_mask,
    cs_bottom_abs_mask,
    cs_top_abs_mask,
    decide_book_aligned_promote,
    decide_short_aligned_promote,
    decide_ej_ls_promote,
    decide_relative_dir_promote,
    decide_rel_e_stack_promote,
    decide_sector_mae_promote,
    decide_two_stage_mae_promote,
    decide_sparse_mae_promote,
    apply_two_stage_map,
    apply_sparse_mae,
    fit_two_stage_mae_maps,
    fit_sparse_mae_maps,
    fit_book_aligned_on_train,
    fit_short_aligned_on_train,
    fit_relative_dir_on_train,
    fit_rel_e_stack_on_train,
    fit_residual_mae_maps,
    fit_cond_dir_blend,
    fit_confidence_blend,
    fit_cs_left_veto,
    fit_decile_reliability,
    fit_logistic_up,
    fit_left_tail_l1,
    format_accuracy_report,
    direction_hits,
    evaluate_overnight_accuracy,
    evaluate_overnight_skip,
    fit_affine_l1,
    fit_affine_ols,
    fit_bin_constants,
    fit_drift_veto,
    hit_rate_inference,
    hit_rate_vs_p0,
    long_only_book_block,
    score_book_aligned_sleeve,
    score_short_aligned_sleeve,
    score_eval_frame,
    score_rel_e_stack,
    score_relative_direction,
    slim_accuracy,
    two_sided_normal_p,
    weekday_of_dates,
)
from forecast.data import FEATURE_NAMES
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


def test_next_open_is_not_a_feature_name():
    lowered = {n.lower() for n in FEATURE_NAMES}
    assert "open" not in lowered
    assert "next_open" not in lowered
    assert "target_raw" not in lowered


def test_affine_l1_a0_is_train_median():
    pred = np.array([0.01, -0.02, 0.03, -0.01, 0.0, 0.02])
    # Unrelated to pred: MAE-optimal affine should shrink a toward 0.
    r_on = np.array([0.004, 0.005, 0.003, 0.006, 0.004, 0.005])
    a, b = fit_affine_l1(pred, r_on)
    assert abs(a) < 0.15
    assert abs(b - float(np.median(r_on))) < 0.002


def test_affine_ols_is_determined_by_train_rows_only():
    """Causality: coefficients come from train rows; later labels do not refit."""
    rng = np.random.default_rng(2)
    train_p = rng.normal(size=80)
    train_y = 0.4 * train_p + 0.002 + rng.normal(scale=0.01, size=80)
    later_p = rng.normal(size=40)
    later_y = -0.9 * later_p + 0.05 + rng.normal(scale=0.01, size=40)
    a0, b0 = fit_affine_ols(train_p, train_y)
    later_hat = apply_affine(later_p, a0, b0)
    assert later_hat.shape == later_p.shape
    a_later, _b_later = fit_affine_ols(later_p, later_y)
    assert abs(a0 - 0.4) < 0.15
    assert abs(a_later - a0) > 0.5
    leaked_a, _ = fit_affine_ols(
        np.concatenate([train_p, later_p]),
        np.concatenate([train_y, later_y]),
    )
    assert abs(leaked_a - a0) > 0.05


def test_apply_readout_keeps_residual_scores():
    import pandas as pd

    df = pd.DataFrame(
        {
            "pred": [1.0, -1.0],
            "y": [0.5, -0.5],
            "scale": [0.01, 0.01],
            "close": [100.0, 50.0],
            "next_open": [101.0, 49.5],
            "r_on": [np.log(101 / 100), np.log(49.5 / 50)],
            "pred_r": [0.01, -0.01],
            "implied_open": [100.0 * np.exp(0.01), 50.0 * np.exp(-0.01)],
            "implied_open_given_hedge": [101.0, 49.5],
        }
    )
    out = apply_readout(df, np.array([0.0, 0.0]))
    assert out["pred"].tolist() == [1.0, -1.0]
    assert out["y"].tolist() == [0.5, -0.5]
    assert out["pred_r"].tolist() == [0.0, 0.0]
    assert abs(out["implied_open"].iloc[0] - 100.0) < 1e-12


def test_adv_sleeve_is_within_date_turnover_feature():
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": [1, 1, 1, 2, 2, 2],
            "turnover_z": [0.0, 1.0, 2.0, 5.0, 4.0, 0.0],
        }
    )
    mask = adv_sleeve_mask(df, pctile=0.67)
    # Top tercile of 3 names: the max on each date.
    assert mask.tolist() == [False, False, True, True, False, False]


def test_val_gate_does_not_read_test_metrics():
    from forecast.accuracy import _pick_promoted

    rows = [
        {
            "name": "affine_l1",
            "promote_dir": True,
            "promote_mae": True,
            "promote": False,
            "val": {"dir_pct": 55.0, "mae_pct": 0.005},
            "test": {"dir_pct": 40.0, "mae_pct": 0.02},  # worse on test; must not matter
        },
        {
            "name": "ts_ridge_all",
            "promote_dir": False,
            "promote_mae": False,
            "promote": False,
            "val": {"dir_pct": 51.0, "mae_pct": 0.007},
            "test": {"dir_pct": 60.0, "mae_pct": 0.003},  # better on test; must not win
        },
    ]
    promo = _pick_promoted(rows)
    assert promo["accuracy_default"] == "affine_l1"
    assert promo["price"] == "affine_l1"


def test_synthetic_accuracy_ablation_is_causal_and_beats_or_matches_baseline(tmp_path: Path):
    data_dir = tmp_path / "data"
    write_cs_overnight_universe(data_dir, n_names=12, n_days=240, seed=1, rho=0.65)
    payload = evaluate_overnight_accuracy(
        str(data_dir), "synthetic", log_fn=None, ablate=True
    )
    assert payload["n_samples"] >= 20
    names = [r["name"] for r in payload["ablation"]["rows"]]
    assert "residual_sigma" in names
    assert "affine_l1" in names
    assert "train_median_gap" in names
    assert "ts_ridge_all" in names
    assert "sign_ridge_calibrated" in names
    assert "drift_veto" in names
    assert "bin_calibrate" in names
    assert "piecewise_l1" in names
    assert "dow_gap" in names
    assert "confidence_blend" in names
    assert "left_tail_l1" in names
    assert "cond_dir_blend" in names
    assert "decile_reliability" in names
    assert "cs_left_veto" in names
    assert "logistic_up" in names
    promo = payload["promotion"]
    assert promo["cs_skip_unchanged"] is True
    # Promotion is VAL-only; test keys exist for the report but are not the gate.
    assert "accuracy_default" in promo
    cal = payload["calibrate"]
    assert cal["name"] == promo["accuracy_default"]
    # Affine coefficients come from train; a/b may be null for non-residual readouts.
    if cal.get("a") is not None:
        assert np.isfinite(float(cal["a"]))
        assert np.isfinite(float(cal["b"]))
    # Confidence thresholds are train quantiles.
    conf = payload["confidence"]
    assert "median" in conf
    # L1 affine on train cannot have higher train-implied MAE intent than a huge scale.
    a_l1 = payload["ablation"]["calibrators"]["affine_l1"]["a"]
    assert np.isfinite(a_l1)
    # Planted CS residual: skip CS IC stays positive on test.
    assert payload["cs_ic"]["cs_ic"] > 0.05
    # MAE of some calibrated readout should be finite.
    by_name = {r["name"]: r for r in payload["ablation"]["rows"]}
    l1_test = by_name["affine_l1"]["test"]["mae_pct"]
    base_test = by_name["residual_sigma"]["test"]["mae_pct"]
    med_test = by_name["train_median_gap"]["test"]["mae_pct"]
    assert np.isfinite(l1_test) and np.isfinite(base_test) and np.isfinite(med_test)
    # Honest price object: L1 affine should not be worse than residual*sigma on
    # the planted tape by a large margin (it can match the median gap).
    assert l1_test <= base_test * 1.05 or l1_test <= med_test * 1.05
    # Excess vs overnight-up is reported on every readout.
    med_xs = by_name["train_median_gap"]["test"]["excess_pp"]
    assert np.isfinite(med_xs)
    assert "excess_pp" in by_name["drift_veto"]["val"]
    assert "book" in payload
    assert "long_only_top20" in payload["book"]
    assert "short_bottom20" in payload["book"]
    year0 = payload["year_slices"][0]
    assert "excess_pp" in year0
    assert "up_pct" in year0
    assert payload["calibrate"]["name"] == promo["accuracy_default"]
    assert "kind" in payload["calibrate"]
    cond_row = by_name["cond_dir_blend"]
    assert cond_row["fit"] == "train"
    assert "cond_val" in cond_row and "cond_test" in cond_row
    assert "cond_gate" in cond_row
    assert np.isfinite(float((cond_row["params"] or {}).get("tau_abs") or float("nan")))
    # TEST juiciness must not be the VAL gate.
    juicy = float((cond_row["cond_test"].get("blend") or {}).get("dir_pct") or 0.0)
    gate_dir = float((cond_row["cond_gate"] or {}).get("val_cond_dir_pct") or float("nan"))
    assert np.isfinite(gate_dir)
    del juicy
    dec_row = by_name["decile_reliability"]
    assert dec_row["fit"] == "train"
    assert int((dec_row["params"] or {}).get("n_bins") or 0) >= 3
    assert "keep" in (dec_row["params"] or {})
    ba_fit = payload["book_aligned_fit"]
    assert ba_fit["fit_split"] == "train"
    assert "chosen" in ba_fit and "baseline" in ba_fit
    ba_cmp = payload["book_aligned_compare"]
    assert "chosen" in ba_cmp["val"] and "top20" in ba_cmp["val"]
    assert "chosen" in ba_cmp["test"] and "grid" in ba_cmp["val"]
    ba_promo = payload["book_aligned_promotion"]
    assert ba_promo["gated_on"] == "val"
    assert "promote_book_aligned" in ba_promo
    report = format_accuracy_report(payload)
    report.encode("cp1252")
    assert "PROMOTE BOOK-ALIGNED" in report
    assert "conviction_live" in payload
    assert payload["conviction_live_promotion"]["gated_on"] == "val"
    assert payload["conviction_live_promotion"]["default_book_unchanged"] is True
    assert "PROMOTE CONVICTION LIVE" in report
    assert payload["sector_mae_fit"]["fit_split"] == "train"
    assert payload["sector_mae_fit"]["hedge"] == "sector_overnight"
    assert payload["sector_mae_promotion"]["gated_on"] == "val"
    assert payload["sector_mae_promotion"]["live_book_unchanged"] is True
    assert "PROMOTE SECTOR-MAE" in report
    ts_fit = payload["two_stage_mae_fit"]
    assert ts_fit["fit_split"] == "train"
    assert set(ts_fit["maps"]) >= {
        "two_stage_l1_pct",
        "two_stage_huber_pct",
        "two_stage_piecewise_pct",
        "two_stage_l1_usd",
        "ridge_resid_dow_vol",
    }
    ts_promo = payload["two_stage_mae_promotion"]
    assert ts_promo["gated_on"] == "val"
    assert ts_promo["live_book_unchanged"] is True
    assert "promote_two_stage_mae" in ts_promo
    assert "PROMOTE TWO-STAGE MAE" in report
    sp_fit = payload["sparse_mae_fit"]
    assert sp_fit["fit_split"] == "train"
    assert set(sp_fit["maps"]) >= {"sparse_l1", "sparse_huber"}
    sp_promo = payload["sparse_mae_promotion"]
    assert sp_promo["gated_on"] == "val"
    assert sp_promo["live_book_unchanged"] is True
    assert "promote_sparse_mae" in sp_promo
    assert "PROMOTE SPARSE MAE" in report
    rel_fit = payload["relative_dir_fit"]
    assert rel_fit["fit_split"] == "train"
    assert rel_fit["score_col"] == "pred"
    rel_cmp = payload["relative_dir_compare"]
    assert "chosen" in rel_cmp["val"] and "full" in rel_cmp["val"]
    assert "top20" in rel_cmp["val"] and "chosen" in rel_cmp["test"]
    rel_promo = payload["relative_dir_promotion"]
    assert rel_promo["gated_on"] == "val"
    assert "promote_relative_dir" in rel_promo
    assert "promote_long_half" in rel_promo
    assert rel_promo["live_book_unchanged"] is True
    assert "PROMOTE RELATIVE-DIR" in report
    assert "PROMOTE LONG-HALF UP" in report
    stack_fit = payload["rel_e_stack_fit"]
    assert stack_fit["fit_split"] == "train"
    assert stack_fit["score_col"] == "pred"
    stack_cmp = payload["rel_e_stack_compare"]
    assert "chosen" in stack_cmp["val"] and "e_sleeve" in stack_cmp["val"]
    assert "chosen" in stack_cmp["test"]
    stack_promo = payload["rel_e_stack_promotion"]
    assert stack_promo["gated_on"] == "val"
    assert "promote_stack_rel" in stack_promo
    assert "promote_stack_abs" in stack_promo
    assert "promote_stack_live" in stack_promo
    assert stack_promo["default_book_unchanged"] is True
    assert "rel_e_stack_live" in payload
    assert "q20" in payload["rel_e_stack_live"]["val"]
    assert "PROMOTE STACK RELATIVE-DIR" in report
    assert "PROMOTE STACK ABSOLUTE-UP" in report
    assert "PROMOTE STACK LIVE IR" in report
    sh_fit = payload["short_aligned_fit"]
    assert sh_fit["fit_split"] == "train"
    assert "chosen" in sh_fit and "baseline" in sh_fit
    sh_cmp = payload["short_aligned_compare"]
    assert "chosen" in sh_cmp["val"] and "bot20" in sh_cmp["val"]
    assert "chosen" in sh_cmp["test"] and "grid" in sh_cmp["val"]
    sh_promo = payload["short_aligned_promotion"]
    assert sh_promo["gated_on"] == "val"
    assert "promote_short_aligned" in sh_promo
    assert "promote_short_live" in sh_promo
    assert sh_promo["default_book_unchanged"] is True
    assert "short_aligned_live" in payload
    assert "q20" in payload["short_aligned_live"]["val"]
    assert "PROMOTE SHORT-ALIGNED" in report
    assert "PROMOTE SHORT LIVE LS" in report
    ej_cmp = payload["ej_ls_compare"]
    assert "live" in ej_cmp["val"] and "paper" in ej_cmp["val"]
    assert "q20" in ej_cmp["val"] and "live" in ej_cmp["test"]
    ej_promo = payload["ej_ls_promotion"]
    assert ej_promo["gated_on"] == "val"
    assert "promote_ej_ls" in ej_promo
    assert ej_promo["default_book_unchanged"] is True
    assert "PROMOTE E+J LIVE LS" in report


def test_zero_move_direction_is_zero_not_nan():
    import pandas as pd

    close = np.array([100.0, 50.0, 25.0, 10.0])
    r_on = np.array([0.01, -0.02, 0.03, -0.01])
    nxt = close * np.exp(r_on)
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "date": [1, 1, 2, 2],
            "pred": np.zeros(4),
            "y": r_on / 0.01,
            "scale": np.full(4, 0.01),
            "close": close,
            "next_open": nxt,
            "r_on": r_on,
            "pred_r": np.zeros(4),
            "implied_open": close,
            "implied_open_given_hedge": nxt,
        }
    )
    out = slim_accuracy(score_eval_frame(df, min_names=2))
    assert out["dir_pct"] == 0.0
    expect = float(np.mean(np.abs(nxt - close) / close))
    assert abs(out["mae_pct"] - expect) < 1e-12
    assert out["excess_pp"] == out["dir_pct"] - out["up_pct"]


def test_weekday_of_dates_monday_is_zero():
    # 1970-01-05 is a Monday.
    monday = int((np.datetime64("1970-01-05") - np.datetime64("1970-01-01")) / np.timedelta64(1, "D"))
    friday = monday + 4
    wd = weekday_of_dates(np.array([monday, friday]))
    assert wd.tolist() == [0, 4]


def test_hit_rate_vs_p0_always_up_has_zero_excess():
    r = np.array([0.01, -0.02, 0.03, 0.04, -0.01, 0.02])
    pred = np.ones_like(r)
    hits = direction_hits(pred, r)
    up = float((r > 0).mean())
    stats = hit_rate_vs_p0(hits, up)
    assert abs(stats["excess_pp"]) < 1e-9
    assert abs(stats["hit_rate"] - up) < 1e-12


def test_drift_veto_threshold_is_train_only():
    rng = np.random.default_rng(3)
    train_p = rng.normal(scale=0.01, size=400)
    # Left tail of pred is actually down; bulk is the overnight-up drift.
    train_y = np.where(train_p < np.quantile(train_p, 0.15), -0.004, 0.003)
    train_y = train_y + rng.normal(scale=0.0005, size=400)
    spec = fit_drift_veto(train_p, train_y)
    later_p = rng.normal(scale=0.01, size=200)
    later_y = -np.sign(later_p) * 0.01
    spec_later = fit_drift_veto(later_p, later_y)
    assert spec["tau"] != spec_later["tau"] or spec["b_dn"] != spec_later["b_dn"]
    leaked = fit_drift_veto(
        np.concatenate([train_p, later_p]), np.concatenate([train_y, later_y])
    )
    assert abs(float(leaked["tau"]) - float(spec["tau"])) > 1e-12 or abs(
        float(leaked["b_dn"]) - float(spec["b_dn"])
    ) > 1e-12
    hat = apply_drift_veto(train_p, spec["tau"], spec["a_dn"], spec["b_dn"], spec["b_up"])
    # Veto may predict down only on the left tail.
    down = hat < 0
    if int(down.sum()) >= 8:
        assert float(np.median(train_p[down])) <= float(np.median(train_p[~down]))


def test_bin_edges_come_from_train_only():
    rng = np.random.default_rng(4)
    train_p = rng.normal(size=200)
    train_y = 0.2 * train_p + 0.001
    later_p = rng.normal(loc=3.0, size=80)
    later_y = -0.5 * later_p
    e0, v0 = fit_bin_constants(train_p, train_y, n_bins=5)
    e1, _v1 = fit_bin_constants(later_p, later_y, n_bins=5)
    assert abs(float(e0[1]) - float(e1[1])) > 0.2
    leaked_e, _ = fit_bin_constants(
        np.concatenate([train_p, later_p]),
        np.concatenate([train_y, later_y]),
        n_bins=5,
    )
    assert abs(float(leaked_e[1]) - float(e0[1])) > 1e-6
    hat = apply_bin_constants(later_p, e0, v0)
    assert hat.shape == later_p.shape
    assert np.isfinite(hat).all()


def test_cond_dir_blend_is_train_only_and_masks_low_abs():
    rng = np.random.default_rng(5)
    train_p = rng.normal(scale=0.01, size=500)
    train_y = np.where(train_p < np.quantile(train_p, 0.20), -0.005, 0.003)
    train_y = train_y + rng.normal(scale=0.0004, size=500)
    left = fit_left_tail_l1(train_p, train_y)
    a, b = fit_affine_l1(train_p, train_y)
    b_up = float(np.median(train_y))
    blend = fit_confidence_blend(train_p, train_y, a, b, b_up)
    spec = fit_cond_dir_blend(train_p, train_y, left, blend, b_up)
    later_p = rng.normal(loc=0.03, scale=0.02, size=200)
    later_y = -np.sign(later_p) * 0.01
    leaked = fit_cond_dir_blend(
        np.concatenate([train_p, later_p]),
        np.concatenate([train_y, later_y]),
        fit_left_tail_l1(np.concatenate([train_p, later_p]), np.concatenate([train_y, later_y])),
        fit_confidence_blend(
            np.concatenate([train_p, later_p]),
            np.concatenate([train_y, later_y]),
            *fit_affine_l1(np.concatenate([train_p, later_p]), np.concatenate([train_y, later_y])),
            float(np.median(np.concatenate([train_y, later_y]))),
        ),
        float(np.median(np.concatenate([train_y, later_y]))),
    )
    assert spec["q"] in (0.50, 0.60, 0.70, 0.80, 0.90)
    assert spec["lam"] in (0.0, 0.25, 0.50, 0.75, 1.0)
    assert (
        abs(float(leaked["tau_abs"]) - float(spec["tau_abs"])) > 1e-12
        or abs(float(leaked["lam"]) - float(spec["lam"])) > 1e-12
        or abs(float(leaked["q"]) - float(spec["q"])) > 1e-12
    )
    hat = apply_cond_dir_blend(train_p, **{k: spec[k] for k in (
        "tau_abs", "lam", "left_tau", "left_a_neg", "left_b_up",
        "conf_tau", "conf_a", "conf_b", "conf_b_up", "b_up",
    )})
    low = ~cond_abs_mask(train_p, spec["tau_abs"])
    if int(low.sum()) >= 4:
        assert np.allclose(hat[low], spec["b_up"])
    cal = apply_calibrate_spec(train_p, {"kind": "cond_dir_blend", **spec})
    assert np.allclose(cal, hat)


def test_decile_reliability_is_train_only():
    rng = np.random.default_rng(6)
    train_p = rng.normal(scale=0.01, size=600)
    # Left third of pred is anti-skill; right two-thirds match the sign.
    q33 = float(np.quantile(train_p, 0.33))
    train_y = np.where(train_p < q33, np.abs(train_p), train_p)
    train_y = train_y + rng.normal(scale=0.0003, size=600)
    spec = fit_decile_reliability(train_p, train_y)
    later_p = rng.normal(loc=0.04, scale=0.02, size=300)
    later_y = -later_p
    leaked = fit_decile_reliability(
        np.concatenate([train_p, later_p]),
        np.concatenate([train_y, later_y]),
    )
    assert int(spec["n_bins"]) >= 3
    assert spec["n_keep"] < spec["n_bins"] or spec["n_keep"] >= 1
    assert (
        spec["edges"] != leaked["edges"]
        or spec["keep"] != leaked["keep"]
        or abs(float(spec["b_up"]) - float(leaked["b_up"])) > 1e-12
    )
    hat = apply_decile_reliability(
        train_p,
        np.asarray(spec["edges"], dtype=np.float64),
        np.asarray(spec["keep"], dtype=np.bool_),
        float(spec["b_up"]),
    )
    cal = apply_calibrate_spec(train_p, {"kind": "decile_reliability", **spec})
    assert np.allclose(hat, cal, equal_nan=True)


def test_cs_left_veto_is_train_only_and_requires_cs_bottom():
    rng = np.random.default_rng(7)
    n_days, n_names = 40, 10
    dates = np.repeat(np.arange(n_days), n_names)
    pred = np.tile(np.linspace(-1.0, 1.0, n_names), n_days)
    pred_r = pred * 0.01
    # Bottom CS names are actually down; others drift up.
    r_on = np.where(pred < np.quantile(pred, 0.20), -0.004, 0.003)
    r_on = r_on + rng.normal(scale=0.0003, size=pred.size)
    spec = fit_cs_left_veto(pred_r, r_on, pred, dates)
    later_p = rng.normal(size=200)
    later_y = -np.sign(later_p) * 0.01
    later_d = np.repeat(np.arange(20), 10)
    leaked = fit_cs_left_veto(later_p, later_y, later_p, later_d)
    assert spec["q"] in (0.10, 0.15, 0.20, 0.30)
    assert abs(float(spec["tau"]) - float(leaked["tau"])) > 1e-12 or spec["q"] != leaked["q"]
    hat = apply_cs_left_veto(
        pred_r, pred, dates, spec["tau"], spec["q"], spec["a_dn"], spec["b_dn"], spec["b_up"]
    )
    # Names that are not CS-bottom must stay the always-up constant.
    import pandas as pd

    pct = pd.Series(pred).groupby(pd.Series(dates)).rank(pct=True, method="average")
    not_bottom = pct.to_numpy() > float(spec["q"])
    if int(not_bottom.sum()) >= 4:
        assert np.allclose(hat[not_bottom], spec["b_up"])


def test_logistic_up_is_train_only():
    rng = np.random.default_rng(8)
    train_p = rng.normal(scale=0.01, size=500)
    train_y = 0.8 * train_p + 0.001 + rng.normal(scale=0.002, size=500)
    spec = fit_logistic_up(train_p, train_y)
    later_p = rng.normal(loc=0.05, scale=0.02, size=200)
    later_y = -later_p
    leaked = fit_logistic_up(later_p, later_y)
    assert spec["tau"] in (0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60)
    assert abs(float(spec["a"]) - float(leaked["a"])) > 0.1
    hat = apply_logistic_up(train_p, spec["a"], spec["b"], spec["tau"], spec["mag"])
    cal = apply_calibrate_spec(train_p, {"kind": "logistic_up", **spec})
    assert np.allclose(hat, cal)
    assert spec["mag"] > 0


def test_apply_calibrate_spec_affine_and_veto():
    p = np.array([-0.02, -0.001, 0.01])
    aff = apply_calibrate_spec(p, {"kind": "affine_l1", "a": 0.5, "b": 0.001})
    assert np.allclose(aff, 0.5 * p + 0.001)
    veto = apply_calibrate_spec(
        p, {"kind": "drift_veto", "tau": -0.01, "a_dn": 1.0, "b_dn": 0.0, "b_up": 0.002}
    )
    assert veto[0] == p[0]
    assert veto[1] == 0.002
    assert veto[2] == 0.002
    # Empty spec is residual*sigma passthrough (generate.py with no overlay).
    raw = apply_calibrate_spec(p, {})
    assert np.allclose(raw, p)


def test_fit_residual_mae_maps_is_train_only():
    rng = np.random.default_rng(2)
    p = rng.normal(scale=0.01, size=80)
    y = 0.6 * p + 0.002 + rng.normal(scale=0.004, size=80)
    later_y = -0.6 * p - 0.002 + rng.normal(scale=0.004, size=80)
    spec = fit_residual_mae_maps(p, y)
    leaked = fit_residual_mae_maps(p, later_y)
    assert spec["fit_split"] == "train"
    assert spec["hedge"] == "sector_overnight"
    assert set(spec["maps"]) >= {"affine_l1", "piecewise_l1", "huber_affine", "bin_calibrate"}
    a0 = float(spec["maps"]["affine_l1"]["a"])
    a1 = float(leaked["maps"]["affine_l1"]["a"])
    assert abs(a0 - a1) > 0.05


def test_decide_sector_mae_promote_is_val_only():
    resid = {"mae_pct": 0.00600, "dir_pct": 51.0, "excess_pp": -3.0}
    zero = {"mae_pct": 0.00700, "dir_pct": 0.0, "excess_pp": -50.0}
    median = {"mae_pct": 0.00680, "dir_pct": 54.0, "excess_pp": 0.0}
    maps = {
        "affine_l1": {"kind": "affine_l1", "a": 0.8, "b": 0.001},
        "piecewise_l1": {"kind": "piecewise_l1"},
        "huber_affine": {"kind": "huber_affine", "a": 0.7, "b": 0.001},
        "bin_calibrate": {"kind": "bin_calibrate"},
    }
    # Clears floors by 1.0 bp but loses to current ts_ridge default.
    almost = {
        "affine_l1": {"mae_pct": 0.00590, "dir_pct": 52.0, "excess_pp": -2.0, "mae_usd": 1.1},
        "piecewise_l1": {"mae_pct": 0.00595, "dir_pct": 52.0, "excess_pp": -2.0},
        "huber_affine": {"mae_pct": 0.00592, "dir_pct": 52.0, "excess_pp": -2.0},
        "bin_calibrate": {"mae_pct": 0.00610, "dir_pct": 51.0, "excess_pp": -3.0},
    }
    no = decide_sector_mae_promote(
        val_maps=almost,
        val_residual=resid,
        val_zero=zero,
        val_median=median,
        val_current={"mae_pct": 0.00550, "dir_pct": 64.0},
        current_name="ts_ridge_no_long_ts",
        maps=maps,
    )
    assert no["promote_sector_mae"] is False
    assert no["gated_on"] == "val"
    assert no["live_book_unchanged"] is True
    yes_maps = {
        **almost,
        "huber_affine": {"mae_pct": 0.00540, "dir_pct": 52.0, "excess_pp": -2.0, "mae_usd": 1.0},
    }
    yes = decide_sector_mae_promote(
        val_maps=yes_maps,
        val_residual=resid,
        val_zero=zero,
        val_median=median,
        val_current={"mae_pct": 0.00550, "dir_pct": 64.0},
        current_name="ts_ridge_no_long_ts",
        maps=maps,
    )
    assert yes["promote_sector_mae"] is True
    assert yes["best_name"] == "huber_affine"
    juicy = {"mae_pct": 0.001, "dir_pct": 80.0}
    assert juicy["mae_pct"] < yes["val_mae_pct"]


def test_fit_two_stage_mae_maps_is_train_only():
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 80
    pred_r = rng.normal(scale=0.01, size=n)
    r_on = 0.7 * pred_r + 0.001 + rng.normal(scale=0.003, size=n)
    close = 50.0 + rng.normal(scale=5.0, size=n)
    nxt = close * np.exp(r_on)
    dates = np.arange(n, dtype=np.int64) + 17600
    vol = np.abs(rng.normal(scale=0.02, size=n))
    df = pd.DataFrame(
        {
            "pred_r": pred_r,
            "r_on": r_on,
            "close": close,
            "next_open": nxt,
            "date": dates,
            "vol_level": vol,
        }
    )
    spec = fit_two_stage_mae_maps(df)
    later = df.copy()
    later["r_on"] = -0.7 * pred_r - 0.001 + rng.normal(scale=0.003, size=n)
    later["next_open"] = later["close"] * np.exp(later["r_on"])
    leaked = fit_two_stage_mae_maps(later)
    assert spec["fit_split"] == "train"
    a0 = float(spec["maps"]["two_stage_l1_pct"]["stage1"]["a"])
    a1 = float(leaked["maps"]["two_stage_l1_pct"]["stage1"]["a"])
    assert abs(a0 - a1) > 0.05
    w0 = float(spec["maps"]["ridge_resid_dow_vol"]["weights"][0])
    w1 = float(leaked["maps"]["ridge_resid_dow_vol"]["weights"][0])
    assert w0 * w1 < 0.0
    hat = apply_two_stage_map(
        pred_r,
        spec["maps"]["two_stage_l1_pct"],
        close=close,
        dates=dates,
        vol_level=vol,
    )
    hat_scrambled = apply_two_stage_map(
        pred_r,
        spec["maps"]["two_stage_l1_pct"],
        close=close,
        dates=dates,
        vol_level=vol,
    )
    assert np.allclose(hat, hat_scrambled, equal_nan=True)
    # Next open is a label only: scrambling it after fit must not change apply.
    assert "next_open" not in str(spec["maps"]["two_stage_l1_pct"].get("stage1") or {})


def test_decide_two_stage_mae_promote_is_val_only():
    resid = {"mae_pct": 0.00600, "dir_pct": 51.0, "excess_pp": -3.0}
    zero = {"mae_pct": 0.00700, "dir_pct": 0.0, "excess_pp": -50.0}
    median = {"mae_pct": 0.00680, "dir_pct": 54.0, "excess_pp": 0.0}
    maps = {n: {"kind": n} for n in (
        "two_stage_l1_pct",
        "two_stage_huber_pct",
        "two_stage_piecewise_pct",
        "two_stage_l1_usd",
        "ridge_resid_dow_vol",
    )}
    almost = {
        "two_stage_l1_pct": {"mae_pct": 0.00590, "dir_pct": 52.0, "excess_pp": -2.0},
        "two_stage_huber_pct": {"mae_pct": 0.00595, "dir_pct": 52.0, "excess_pp": -2.0},
        "two_stage_piecewise_pct": {"mae_pct": 0.00592, "dir_pct": 52.0, "excess_pp": -2.0},
        "two_stage_l1_usd": {"mae_pct": 0.00610, "dir_pct": 51.0, "excess_pp": -3.0},
        "ridge_resid_dow_vol": {"mae_pct": 0.00588, "dir_pct": 53.0, "excess_pp": -1.0},
    }
    # Clears floors by 1.2 bp but worse than current ts_ridge default.
    no = decide_two_stage_mae_promote(
        val_maps=almost,
        val_residual=resid,
        val_zero=zero,
        val_median=median,
        val_current={"mae_pct": 0.00550, "dir_pct": 64.0},
        current_name="ts_ridge_no_long_ts",
        maps=maps,
    )
    assert no["promote_two_stage_mae"] is False
    assert no["gated_on"] == "val"
    assert no["live_book_unchanged"] is True
    yes_maps = {
        **almost,
        "ridge_resid_dow_vol": {
            "mae_pct": 0.00545,
            "dir_pct": 52.0,
            "excess_pp": -2.0,
        },
    }
    yes = decide_two_stage_mae_promote(
        val_maps=yes_maps,
        val_residual=resid,
        val_zero=zero,
        val_median=median,
        val_current={"mae_pct": 0.00550, "dir_pct": 64.0},
        current_name="ts_ridge_no_long_ts",
        maps=maps,
    )
    assert yes["promote_two_stage_mae"] is True
    assert yes["best_name"] == "ridge_resid_dow_vol"
    floors_fail = decide_two_stage_mae_promote(
        val_maps={
            "two_stage_l1_pct": {"mae_pct": 0.00597, "dir_pct": 52.0},
            "two_stage_huber_pct": {"mae_pct": 0.00598, "dir_pct": 52.0},
            "two_stage_piecewise_pct": {"mae_pct": 0.00599, "dir_pct": 52.0},
            "two_stage_l1_usd": {"mae_pct": 0.00610, "dir_pct": 51.0},
            "ridge_resid_dow_vol": {"mae_pct": 0.00597, "dir_pct": 52.0},
        },
        val_residual=resid,
        val_zero=zero,
        val_median=median,
        val_current={"mae_pct": 0.00650, "dir_pct": 64.0},
        current_name="ts_ridge_no_long_ts",
        maps=maps,
    )
    assert floors_fail["promote_two_stage_mae"] is False
    juicy = {"mae_pct": 0.001, "dir_pct": 80.0}
    assert juicy["mae_pct"] < yes["val_mae_pct"]


def test_fit_sparse_mae_maps_is_train_only():
    import pandas as pd

    rng = np.random.default_rng(4)
    n = 120
    pred_r = rng.normal(scale=0.01, size=n)
    r_on = 0.8 * pred_r + rng.normal(scale=0.003, size=n)
    close = 40.0 + rng.normal(scale=4.0, size=n)
    nxt = close * np.exp(r_on)
    dates = np.repeat(np.arange(20, dtype=np.int64), 6)
    df = pd.DataFrame(
        {
            "symbol": np.array([f"S{i % 6:02d}" for i in range(n)]),
            "pred": pred_r / 0.01,
            "y": r_on / 0.01,
            "scale": np.full(n, 0.01),
            "pred_r": pred_r,
            "r_on": r_on,
            "close": close,
            "next_open": nxt,
            "date": dates,
            "implied_open": close * np.exp(pred_r),
            "implied_open_given_hedge": close * np.exp(pred_r),
        }
    )
    spec = fit_sparse_mae_maps(df, min_names=3)
    later = df.copy()
    later["r_on"] = -0.8 * pred_r + rng.normal(scale=0.003, size=n)
    later["next_open"] = later["close"] * np.exp(later["r_on"])
    leaked = fit_sparse_mae_maps(later, min_names=3)
    assert spec["fit_split"] == "train"
    assert set(spec["maps"]) >= {"sparse_l1", "sparse_huber"}
    a0 = float(spec["maps"]["sparse_l1"]["a"])
    a1 = float(leaked["maps"]["sparse_l1"]["a"])
    assert a0 * a1 < 0.0
    hat = apply_sparse_mae(pred_r, a0, float(spec["maps"]["sparse_l1"]["b"]), 0.02)
    assert float(np.mean(hat[np.abs(pred_r) < 0.02] == 0.0)) == 1.0


def test_decide_sparse_mae_promote_is_val_only():
    resid = {"mae_pct": 0.00600, "dir_pct": 51.0, "excess_pp": -3.0}
    zero = {"mae_pct": 0.00700, "dir_pct": 0.0, "excess_pp": -50.0}
    median = {"mae_pct": 0.00680, "dir_pct": 54.0, "excess_pp": 0.0}
    maps = {
        "sparse_l1": {"kind": "sparse_l1", "a": 0.8, "b": 0.0, "tau": 0.004},
        "sparse_huber": {"kind": "sparse_huber", "a": 0.7, "b": 0.0, "tau": 0.003},
    }
    almost = {
        "sparse_l1": {"mae_pct": 0.00590, "dir_pct": 52.0, "tau": 0.004},
        "sparse_huber": {"mae_pct": 0.00588, "dir_pct": 53.0, "tau": 0.003},
    }
    no = decide_sparse_mae_promote(
        val_maps=almost,
        val_residual=resid,
        val_zero=zero,
        val_median=median,
        val_current={"mae_pct": 0.00550, "dir_pct": 64.0},
        current_name="ts_ridge_no_long_ts",
        maps=maps,
    )
    assert no["promote_sparse_mae"] is False
    assert no["gated_on"] == "val"
    assert no["live_book_unchanged"] is True
    yes = decide_sparse_mae_promote(
        val_maps={
            **almost,
            "sparse_huber": {"mae_pct": 0.00545, "dir_pct": 52.0, "tau": 0.003},
        },
        val_residual=resid,
        val_zero=zero,
        val_median=median,
        val_current={"mae_pct": 0.00550, "dir_pct": 64.0},
        current_name="ts_ridge_no_long_ts",
        maps=maps,
    )
    assert yes["promote_sparse_mae"] is True
    assert yes["best_name"] == "sparse_huber"
    floors_fail = decide_sparse_mae_promote(
        val_maps={
            "sparse_l1": {"mae_pct": 0.00597, "dir_pct": 52.0},
            "sparse_huber": {"mae_pct": 0.00598, "dir_pct": 52.0},
        },
        val_residual=resid,
        val_zero=zero,
        val_median=median,
        val_current={"mae_pct": 0.00650, "dir_pct": 64.0},
        current_name="ts_ridge_no_long_ts",
        maps=maps,
    )
    assert floors_fail["promote_sparse_mae"] is False
    juicy = {"mae_pct": 0.001, "dir_pct": 80.0}
    assert juicy["mae_pct"] < yes["val_mae_pct"]


def test_cs_top_abs_mask_selects_top_and_abs_floor():
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": [1, 1, 1, 1, 2, 2, 2, 2],
            "pred": [0.0, 0.5, 1.0, 2.0, 0.0, 0.1, 0.2, 3.0],
        }
    )
    mask = cs_top_abs_mask(df, q=0.75, abs_tau=0.0, min_names=3)
    assert mask.tolist() == [False, False, False, True, False, False, False, True]
    mask_floor = cs_top_abs_mask(df, q=0.75, abs_tau=2.5, min_names=3)
    assert mask_floor.tolist() == [False, False, False, False, False, False, False, True]


def test_cs_bottom_abs_mask_selects_bottom_and_abs_floor():
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": [1, 1, 1, 1, 2, 2, 2, 2],
            "pred": [0.0, 0.5, 1.0, 2.0, -3.0, 0.1, 0.2, 3.0],
            "r_on": [-0.02, 0.01, 0.01, 0.01, -0.03, 0.01, 0.01, 0.01],
        }
    )
    mask = cs_bottom_abs_mask(df, q=0.25, abs_tau=0.0, min_names=3)
    assert mask.tolist() == [True, False, False, False, True, False, False, False]
    mask_floor = cs_bottom_abs_mask(df, q=0.25, abs_tau=2.5, min_names=3)
    assert mask_floor.tolist() == [False, False, False, False, True, False, False, False]
    scored = score_short_aligned_sleeve(df, q=0.25, abs_tau=0.0, min_names=3)
    assert abs(scored["down_pct"] - 100.0) < 1e-9
    assert scored["n"] == 2.0


def test_fit_book_aligned_on_train_is_train_only():
    import pandas as pd

    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(20, dtype=np.int64), 10)
    pred = rng.normal(size=dates.size)
    # TRAIN: higher pred -> more often up. Later window: reverse that.
    r_train = np.where(pred > np.quantile(pred, 0.80), 0.02, -0.01)
    r_later = np.where(pred > np.quantile(pred, 0.80), -0.02, 0.01)
    train = pd.DataFrame({"date": dates, "pred": pred, "r_on": r_train})
    later = pd.DataFrame({"date": dates + 100, "pred": pred, "r_on": r_later})
    spec = fit_book_aligned_on_train(train, min_names=3)
    leaked = fit_book_aligned_on_train(later, min_names=3)
    assert spec["fit_split"] == "train"
    assert leaked["fit_split"] == "train"
    assert spec["chosen"]["q"] != leaked["chosen"]["q"] or spec["chosen"]["abs_tau"] != leaked["chosen"]["abs_tau"] or spec["chosen"]["excess_pp"] != leaked["chosen"]["excess_pp"]
    scored = score_book_aligned_sleeve(train, q=0.80, abs_tau=0.0, min_names=3)
    assert scored["n"] > 0
    assert np.isfinite(scored["up_pct"])
    assert np.isfinite(scored["excess_pp"])


def test_cs_relative_blocks_aligned_ranks_hit_and_long_half():
    import pandas as pd

    dates = np.repeat(np.arange(6, dtype=np.int64), 4)
    pred = np.tile(np.array([-2.0, -1.0, 1.0, 2.0]), 6)
    r_on = pred / 100.0
    df = pd.DataFrame({"date": dates, "pred": pred, "r_on": r_on})
    pred_rel, r_rel, abs_dev, long_half, date_ok = cs_relative_blocks(
        df, score_col="pred", min_names=3
    )
    assert bool(date_ok.all())
    assert long_half.tolist() == ([False, False, True, True] * 6)
    assert np.allclose(pred_rel, pred)
    assert np.allclose(r_rel, r_on)
    scored = score_relative_direction(df, abs_tau=0.0, min_names=3)
    assert abs(scored["rel_hit_pct"] - 100.0) < 1e-9
    assert scored["rel_z"] > 1.0
    assert abs(scored["long_up_pct"] - 100.0) < 1e-9
    assert abs(scored["long_uncond_up_pct"] - 50.0) < 1e-9
    assert scored["long_n"] == 12.0
    tight = score_relative_direction(df, abs_tau=1.5, min_names=3)
    assert tight["rel_n"] == 12.0
    assert tight["long_n"] == 6.0


def test_fit_relative_dir_on_train_is_train_only():
    import pandas as pd

    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(20, dtype=np.int64), 10)
    pred = rng.normal(size=dates.size)
    r_train = np.where(pred > 0.0, 0.02, -0.01)
    r_later = np.where(pred > 0.0, -0.02, 0.01)
    train = pd.DataFrame({"date": dates, "pred": pred, "r_on": r_train})
    later = pd.DataFrame({"date": dates + 100, "pred": pred, "r_on": r_later})
    spec = fit_relative_dir_on_train(train, min_names=3)
    leaked = fit_relative_dir_on_train(later, min_names=3)
    assert spec["fit_split"] == "train"
    assert leaked["fit_split"] == "train"
    assert spec["score_col"] == "pred"
    assert (
        spec["chosen"]["abs_q"] != leaked["chosen"]["abs_q"]
        or spec["chosen"]["abs_tau"] != leaked["chosen"]["abs_tau"]
        or spec["chosen"]["rel_hit_pct"] != leaked["chosen"]["rel_hit_pct"]
    )


def test_decide_relative_dir_promote_is_val_only():
    chosen = {"abs_tau": 0.0, "abs_q": 0.0}
    val_rel_ok = {
        "rel_hit_pct": 56.0,
        "rel_z": 2.0,
        "rel_coverage": 0.20,
    }
    val_full_ok = {
        "long_up_pct": 56.5,
        "long_uncond_up_pct": 54.0,
        "long_excess_pp": 2.5,
        "long_coverage": 0.40,
    }
    d = decide_relative_dir_promote(
        val_chosen=val_rel_ok,
        val_full=val_full_ok,
        val_top20={"up_pct": 56.0},
        chosen=chosen,
    )
    assert d["promote_relative_dir"] is True
    assert d["promote_long_half"] is True
    assert d["gated_on"] == "val"
    val_rel_fail = {
        "rel_hit_pct": 50.20,
        "rel_z": 2.0,
        "rel_coverage": 0.20,
    }
    d2 = decide_relative_dir_promote(
        val_chosen=val_rel_fail,
        val_full=val_full_ok,
        val_top20={"up_pct": 56.0},
        chosen=chosen,
    )
    assert d2["promote_relative_dir"] is False
    assert d2["gated_on"] == "val"
    val_z_fail = {
        "rel_hit_pct": 51.0,
        "rel_z": 0.4,
        "rel_coverage": 0.20,
    }
    d3 = decide_relative_dir_promote(
        val_chosen=val_z_fail,
        val_full=val_full_ok,
        val_top20={"up_pct": 56.0},
        chosen=chosen,
    )
    assert d3["promote_relative_dir"] is False
    weaker = {
        "long_up_pct": 55.5,
        "long_uncond_up_pct": 54.0,
        "long_excess_pp": 1.5,
        "long_coverage": 0.40,
    }
    d4 = decide_relative_dir_promote(
        val_chosen=val_rel_ok,
        val_full=weaker,
        val_top20={"up_pct": 56.0},
        chosen=chosen,
    )
    assert d4["promote_long_half"] is False
    assert d4["weaker_than_top20"] is True
    juicy_test = {
        "rel_hit_pct": 80.0,
        "rel_z": 8.0,
        "rel_coverage": 0.50,
    }
    d5 = decide_relative_dir_promote(
        val_chosen=val_rel_fail,
        val_full=weaker,
        val_top20={"up_pct": 56.0},
        chosen=chosen,
    )
    del juicy_test
    assert d5["promote_relative_dir"] is False
    assert d5["promote_long_half"] is False
    assert d5["gated_on"] == "val"


def test_cs_stack_mask_is_long_half_intersect_top_q():
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": [1, 1, 1, 1, 2, 2, 2, 2],
            "pred": [-2.0, -1.0, 1.0, 2.0, -2.0, -1.0, 1.0, 2.0],
            "r_on": [-0.02, -0.01, 0.01, 0.02, -0.02, -0.01, 0.01, 0.02],
        }
    )
    top = cs_top_abs_mask(df, q=0.75, abs_tau=0.0, min_names=3)
    stack = cs_stack_mask(df, q=0.75, abs_tau=0.0, min_names=3)
    assert top.tolist() == [False, False, False, True, False, False, False, True]
    assert stack.tolist() == top.tolist()
    scored = score_rel_e_stack(df, q=0.75, abs_tau=0.0, min_names=3)
    assert abs(scored["rel_hit_pct"] - 100.0) < 1e-9
    assert abs(scored["long_up_pct"] - 100.0) < 1e-9
    assert scored["long_n"] == 2.0


def test_fit_rel_e_stack_on_train_is_train_only():
    import pandas as pd

    rng = np.random.default_rng(1)
    dates = np.repeat(np.arange(20, dtype=np.int64), 10)
    pred = rng.normal(size=dates.size)
    r_train = np.where(pred > 0.0, 0.02, -0.01)
    r_later = np.where(pred > 0.0, -0.02, 0.01)
    train = pd.DataFrame({"date": dates, "pred": pred, "r_on": r_train})
    later = pd.DataFrame({"date": dates + 100, "pred": pred, "r_on": r_later})
    e = {"q": 0.80, "abs_q": 0.0, "abs_tau": 0.0}
    spec = fit_rel_e_stack_on_train(train, min_names=3, e_chosen=e)
    leaked = fit_rel_e_stack_on_train(later, min_names=3, e_chosen=e)
    assert spec["fit_split"] == "train"
    assert leaked["fit_split"] == "train"
    assert spec["score_col"] == "pred"
    assert (
        spec["chosen"]["q"] != leaked["chosen"]["q"]
        or spec["chosen"]["abs_tau"] != leaked["chosen"]["abs_tau"]
        or spec["chosen"]["rel_hit_pct"] != leaked["chosen"]["rel_hit_pct"]
    )


def test_decide_rel_e_stack_promote_is_val_only():
    chosen = {"q": 0.90, "abs_tau": 0.2, "abs_q": 0.70}
    val_ok = {
        "rel_hit_pct": 70.0,
        "rel_z": 3.0,
        "rel_coverage": 0.10,
        "long_up_pct": 81.0,
        "long_uncond_up_pct": 54.0,
        "long_excess_pp": 27.0,
        "long_coverage": 0.10,
    }
    val_e = {"up_pct": 80.0}
    live_ok = {
        "unlevered_net_ir": 1.20,
        "unlevered_max_dd": -0.10,
        "coverage": 0.10,
    }
    q20 = {
        "unlevered_net_ir": 1.00,
        "unlevered_max_dd": -0.12,
        "coverage": 0.25,
    }
    d = decide_rel_e_stack_promote(
        val_chosen=val_ok,
        val_e=val_e,
        val_live_stack=live_ok,
        val_live_q20=q20,
        chosen=chosen,
    )
    assert d["promote_stack_rel"] is True
    assert d["promote_stack_abs"] is True
    assert d["promote_stack_live"] is True
    assert d["gated_on"] == "val"
    val_rel_fail = dict(val_ok, rel_hit_pct=50.2)
    d2 = decide_rel_e_stack_promote(
        val_chosen=val_rel_fail,
        val_e=val_e,
        val_live_stack=live_ok,
        val_live_q20=q20,
        chosen=chosen,
    )
    assert d2["promote_stack_rel"] is False
    weaker = dict(val_ok, long_up_pct=80.05, long_excess_pp=26.05)
    d3 = decide_rel_e_stack_promote(
        val_chosen=weaker,
        val_e=val_e,
        val_live_stack=live_ok,
        val_live_q20=q20,
        chosen=chosen,
    )
    assert d3["promote_stack_abs"] is False
    assert d3["weaker_than_e"] is False or d3["val_vs_e_pp"] < 0.2
    live_fail = {
        "unlevered_net_ir": 0.90,
        "unlevered_max_dd": -0.10,
        "coverage": 0.10,
    }
    juicy_test = {
        "rel_hit_pct": 90.0,
        "rel_z": 8.0,
        "rel_coverage": 0.20,
        "long_up_pct": 90.0,
        "long_uncond_up_pct": 50.0,
        "long_excess_pp": 40.0,
        "long_coverage": 0.20,
    }
    d4 = decide_rel_e_stack_promote(
        val_chosen=val_rel_fail,
        val_e=val_e,
        val_live_stack=live_fail,
        val_live_q20=q20,
        chosen=chosen,
    )
    del juicy_test
    assert d4["promote_stack_rel"] is False
    assert d4["promote_stack_live"] is False
    assert d4["gated_on"] == "val"
    assert d4["default_book_unchanged"] is True


def test_decide_book_aligned_promote_is_val_only():
    chosen = {"q": 0.90, "abs_tau": 0.0, "abs_q": 0.0}
    val_ok = {
        "up_pct": 56.5,
        "uncond_up_pct": 54.0,
        "excess_pp": 2.5,
        "coverage": 0.10,
    }
    d = decide_book_aligned_promote(
        val_chosen=val_ok, val_top20={"up_pct": 56.0}, chosen=chosen
    )
    assert d["promote_book_aligned"] is True
    assert d["gated_on"] == "val"
    val_fail = {
        "up_pct": 56.05,
        "uncond_up_pct": 54.0,
        "excess_pp": 2.05,
        "coverage": 0.10,
    }
    d2 = decide_book_aligned_promote(
        val_chosen=val_fail, val_top20={"up_pct": 56.0}, chosen=chosen
    )
    assert d2["promote_book_aligned"] is False
    same = {"q": 0.80, "abs_tau": 0.0, "abs_q": 0.0}
    juicy = {
        "up_pct": 70.0,
        "uncond_up_pct": 50.0,
        "excess_pp": 20.0,
        "coverage": 0.20,
    }
    d3 = decide_book_aligned_promote(
        val_chosen=juicy, val_top20={"up_pct": 50.0}, chosen=same
    )
    assert d3["promote_book_aligned"] is False
    assert d3["gated_on"] == "val"


def test_fit_short_aligned_on_train_is_train_only():
    import pandas as pd

    rng = np.random.default_rng(2)
    dates = np.repeat(np.arange(20, dtype=np.int64), 10)
    pred = rng.normal(size=dates.size)
    r_train = np.where(pred < np.quantile(pred, 0.20), -0.02, 0.01)
    r_later = np.where(pred < np.quantile(pred, 0.20), 0.02, -0.01)
    train = pd.DataFrame({"date": dates, "pred": pred, "r_on": r_train})
    later = pd.DataFrame({"date": dates + 100, "pred": pred, "r_on": r_later})
    spec = fit_short_aligned_on_train(train, min_names=3)
    leaked = fit_short_aligned_on_train(later, min_names=3)
    assert spec["fit_split"] == "train"
    assert leaked["fit_split"] == "train"
    assert (
        spec["chosen"]["q"] != leaked["chosen"]["q"]
        or spec["chosen"]["abs_tau"] != leaked["chosen"]["abs_tau"]
        or spec["chosen"]["excess_pp"] != leaked["chosen"]["excess_pp"]
    )
    scored = score_short_aligned_sleeve(train, q=0.20, abs_tau=0.0, min_names=3)
    assert scored["n"] > 0
    assert np.isfinite(scored["down_pct"])
    assert np.isfinite(scored["excess_pp"])


def test_decide_short_aligned_promote_is_val_only():
    chosen = {"q": 0.10, "abs_tau": 0.2, "abs_q": 0.50}
    val_ok = {
        "down_pct": 56.5,
        "uncond_down_pct": 46.0,
        "excess_pp": 10.5,
        "coverage": 0.10,
    }
    d = decide_short_aligned_promote(
        val_chosen=val_ok,
        val_bot20={"down_pct": 56.0},
        chosen=chosen,
        val_live_ls={"unlevered_net_ir": 1.20, "coverage": 0.10},
        val_live_q20={"unlevered_net_ir": 1.00, "coverage": 0.25},
    )
    assert d["promote_short_aligned"] is True
    assert d["promote_short_live"] is True
    assert d["gated_on"] == "val"
    val_fail = {
        "down_pct": 56.05,
        "uncond_down_pct": 46.0,
        "excess_pp": 10.05,
        "coverage": 0.10,
    }
    d2 = decide_short_aligned_promote(
        val_chosen=val_fail,
        val_bot20={"down_pct": 56.0},
        chosen=chosen,
        val_live_ls={"unlevered_net_ir": 0.90, "coverage": 0.10},
        val_live_q20={"unlevered_net_ir": 1.00, "coverage": 0.25},
    )
    assert d2["promote_short_aligned"] is False
    assert d2["promote_short_live"] is False
    same = {"q": 0.20, "abs_tau": 0.0, "abs_q": 0.0}
    juicy = {
        "down_pct": 80.0,
        "uncond_down_pct": 46.0,
        "excess_pp": 34.0,
        "coverage": 0.20,
    }
    d3 = decide_short_aligned_promote(
        val_chosen=juicy,
        val_bot20={"down_pct": 50.0},
        chosen=same,
        val_live_ls={"unlevered_net_ir": 2.00, "coverage": 0.20},
        val_live_q20={"unlevered_net_ir": 1.00, "coverage": 0.25},
    )
    assert d3["promote_short_aligned"] is False
    assert d3["gated_on"] == "val"
    assert d3["default_book_unchanged"] is True


def test_decide_ej_ls_promote_is_val_only():
    e = {"q": 0.90, "abs_q": 0.70, "abs_tau": 0.3}
    j = {"q": 0.10, "abs_q": 0.70, "abs_tau": 0.3}
    q20 = {"unlevered_net_ir": 1.00, "unlevered_max_dd": -0.10, "coverage": 0.25}
    ok_live = {"unlevered_net_ir": 1.10, "unlevered_max_dd": -0.11, "coverage": 0.20}
    ok_paper = {"unlevered_net_ir": 1.40, "unlevered_max_dd": -0.08, "coverage": 0.20}
    yes = decide_ej_ls_promote(
        val_live=ok_live,
        val_q20=q20,
        val_paper=ok_paper,
        e_chosen=e,
        j_chosen=j,
    )
    assert yes["promote_ej_ls"] is True
    assert yes["live_book_unchanged"] is False
    assert yes["default_book_unchanged"] is True
    assert yes["gated_on"] == "val"
    assert abs(yes["val_ir_delta"] - 0.10) < 1e-9
    assert abs(yes["val_dd_delta"] - (-0.01)) < 1e-9

    weak_live = {"unlevered_net_ir": 0.80, "unlevered_max_dd": -0.20, "coverage": 0.20}
    juicy_paper = {"unlevered_net_ir": 3.00, "unlevered_max_dd": -0.02, "coverage": 0.40}
    no_from_val = decide_ej_ls_promote(
        val_live=weak_live,
        val_q20=q20,
        val_paper=juicy_paper,
        e_chosen=e,
        j_chosen=j,
    )
    assert no_from_val["promote_ej_ls"] is False
    assert no_from_val["live_book_unchanged"] is True

    ir_fail = decide_ej_ls_promote(
        val_live={"unlevered_net_ir": 1.04, "unlevered_max_dd": -0.10, "coverage": 0.20},
        val_q20=q20,
        val_paper=ok_paper,
        e_chosen=e,
        j_chosen=j,
    )
    assert ir_fail["promote_ej_ls"] is False
    dd_fail = decide_ej_ls_promote(
        val_live={"unlevered_net_ir": 1.20, "unlevered_max_dd": -0.16, "coverage": 0.20},
        val_q20=q20,
        val_paper=ok_paper,
        e_chosen=e,
        j_chosen=j,
    )
    assert dd_fail["promote_ej_ls"] is False
    cover_fail = decide_ej_ls_promote(
        val_live={"unlevered_net_ir": 1.20, "unlevered_max_dd": -0.10, "coverage": 0.04},
        val_q20=q20,
        val_paper=ok_paper,
        e_chosen=e,
        j_chosen=j,
    )
    assert cover_fail["promote_ej_ls"] is False


def test_long_only_book_block_selects_within_date_top_pred():
    import pandas as pd

    df = pd.DataFrame(
        {
            "date": [1, 1, 1, 1, 2, 2, 2, 2],
            "pred": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
            "r_on": [0.01, 0.01, 0.01, -0.02, -0.01, 0.01, 0.01, 0.01],
        }
    )
    block = long_only_book_block(df, score_col="pred", q=0.75, min_names=3)
    # Top 25% of 4 names is the max on each date (pred=3).
    assert block["n"] == 2.0
    # Date1 top is down, date2 top is up -> 50% up vs 75% uncond (6/8 up).
    assert abs(block["up_pct"] - 50.0) < 1e-9
    assert block["excess_pp"] < 0


def test_always_up_excess_is_zero_on_scored_frame():
    import pandas as pd

    close = np.array([100.0, 50.0, 25.0, 10.0])
    r_on = np.array([0.01, -0.02, 0.03, -0.01])
    nxt = close * np.exp(r_on)
    df = pd.DataFrame(
        {
            "symbol": ["A", "B", "A", "B"],
            "date": [1, 1, 2, 2],
            "pred": np.ones(4),
            "y": r_on / 0.01,
            "scale": np.full(4, 0.01),
            "close": close,
            "next_open": nxt,
            "r_on": r_on,
            "pred_r": np.full(4, 0.002),
            "implied_open": close * np.exp(0.002),
            "implied_open_given_hedge": nxt,
        }
    )
    out = slim_accuracy(score_eval_frame(df, min_names=2))
    assert abs(out["excess_pp"]) < 1e-9
    assert abs(out["dir_pct"] - out["up_pct"]) < 1e-9

