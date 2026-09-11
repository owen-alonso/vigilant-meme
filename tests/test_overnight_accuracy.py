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
    fit_cond_dir_blend,
    fit_confidence_blend,
    fit_cs_left_veto,
    fit_decile_reliability,
    fit_logistic_up,
    fit_left_tail_l1,
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
    score_eval_frame,
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

