"""(2b) cost-aware RankNet / IR-proxy aligned to live_locate after costs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from forecast.config import ForecastTrainConfig
from forecast.cost_rank import (
    BASELINE_DD,
    BASELINE_IR,
    DD_TOL,
    IR_LIFT,
    apply_cost_adjusted_target,
    book_gate_payload,
    cost_adjusted_residual,
    cost_rank_terms,
    decide_book_gate,
    format_cost_rank_block,
    labelled_cost_aux,
    masked_cost_rank_loss,
    masked_ir_proxy_loss,
    name_live_locate_cost,
    thin_mask_for_dates,
)
from forecast.data import FEATURE_NAMES, SymbolArrays
from forecast.training import build_arg_parser, configs_from_cli, masked_loss


def _cfg(**kwargs) -> ForecastTrainConfig:
    base = dict(
        loss="huber",
        location_loss_weight=1.0,
        ic_loss_weight=0.0,
        sign_loss_weight=0.0,
        rank_loss_weight=0.0,
        listnet_loss_weight=0.0,
        pred_std_weight=0.0,
        cost_rank_loss=True,
        cost_rank_weight=1.0,
        ir_proxy_weight=0.0,
        cost_rank_replace_huber=False,
    )
    base.update(kwargs)
    return ForecastTrainConfig(**base)


def test_defaults_do_not_enable_cost_rank():
    args = build_arg_parser().parse_args([])
    _data, _model, train = configs_from_cli(args)
    assert train.cost_rank_loss is False
    assert train.cost_rank_replace_huber is False
    assert train.location_loss_weight == pytest.approx(0.4)


def test_cli_wires_cost_rank_flags():
    args = build_arg_parser().parse_args(
        [
            "--cost-rank-loss",
            "--cost-rank-replace-huber",
            "--cost-rank-weight",
            "1.5",
            "--ir-proxy-weight",
            "0.25",
            "--label-return",
            "overnight",
            "--num-workers",
            "0",
            "--skip-only",
            "--mmap-manifest",
            "data/_panel_cache/mmap_manifest.json",
        ]
    )
    data_cfg, _model, train = configs_from_cli(args)
    assert train.cost_rank_loss is True
    assert train.cost_rank_replace_huber is True
    assert train.cost_rank_weight == pytest.approx(1.5)
    assert train.ir_proxy_weight == pytest.approx(0.25)
    assert train.skip_only is True
    assert train.num_workers == 0
    assert data_cfg.label_return == "overnight"
    assert data_cfg.mmap_manifest.endswith("mmap_manifest.json")


def test_thin_names_pay_more_than_liquid():
    dates = np.zeros(10, dtype=np.int64)
    turn = np.linspace(-2.0, 2.0, 10)
    vol = np.zeros(10)
    cost = name_live_locate_cost(turn, vol, dates, short=np.zeros(10))
    assert float(cost[0]) > float(cost[-1])
    short = name_live_locate_cost(turn, vol, dates, short=np.ones(10))
    assert float(short[0]) > float(cost[0])


def test_cost_adjusted_residual_shrinks_expensive_names():
    y = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
    scale = np.full(4, 0.01)
    turn = np.array([-2.0, 2.0, -2.0, 2.0])
    vol = np.array([2.0, 0.0, 2.0, 0.0])
    dates = np.zeros(4, dtype=np.int64)
    net = cost_adjusted_residual(y, scale, turn, vol, dates)
    # Same gross residual: thin+high-vol long is worse net than liquid+quiet.
    assert float(net[0]) < float(net[1])
    # Shorts pay borrow; expensive short is less attractive (net less negative).
    assert float(net[2]) > float(net[3])


def test_ranknet_prefers_pred_aligned_with_net_y():
    mean_ok = torch.tensor([[0.2, 1.0, -0.2, -1.0]])
    mean_flip = torch.tensor([[1.0, 0.2, -1.0, -0.2]])
    y = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    mask = torch.ones_like(y)
    dates = torch.zeros_like(y, dtype=torch.long)
    scale = torch.full_like(y, 0.01)
    turn = torch.tensor([[-2.0, 2.0, -2.0, 2.0]])
    vol = torch.tensor([[2.0, 0.0, 2.0, 0.0]])
    ok = float(
        masked_cost_rank_loss(
            mean_ok, y, mask, date_ids=dates, scale=scale,
            turnover_z=turn.reshape(-1), vol_level=vol.reshape(-1),
        )
    )
    bad = float(
        masked_cost_rank_loss(
            mean_flip, y, mask, date_ids=dates, scale=scale,
            turnover_z=turn.reshape(-1), vol_level=vol.reshape(-1),
        )
    )
    assert ok < bad


def test_ir_proxy_lower_when_ranks_match_net():
    y = torch.tensor([[1.2, 0.8, -0.7, -1.1, 0.1, 0.0]])
    mask = torch.ones_like(y)
    dates = torch.zeros_like(y, dtype=torch.long)
    scale = torch.full_like(y, 0.02)
    turn = torch.linspace(-1.5, 1.5, 6).view(1, -1)
    vol = torch.zeros_like(y)
    aligned = y.clone()
    flipped = -y
    good = float(
        masked_ir_proxy_loss(
            aligned, y, mask, date_ids=dates, scale=scale,
            turnover_z=turn.reshape(-1), vol_level=vol.reshape(-1),
        )
    )
    bad = float(
        masked_ir_proxy_loss(
            flipped, y, mask, date_ids=dates, scale=scale,
            turnover_z=turn.reshape(-1), vol_level=vol.reshape(-1),
        )
    )
    assert good < bad


def test_masked_loss_replace_huber_drops_location_term():
    mean = torch.tensor([[4.0, -4.0, 0.0, 1.0]])
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.ones_like(mean)
    dates = torch.zeros_like(mean, dtype=torch.long)
    scale = torch.ones_like(mean)
    zeros = torch.zeros_like(mean)
    huber_only = _cfg(cost_rank_loss=False, location_loss_weight=1.0)
    replaced = _cfg(cost_rank_replace_huber=True, location_loss_weight=1.0)
    loss_h = float(masked_loss(mean, zeros, target, mask, huber_only, date_ids=dates))
    loss_r = float(
        masked_loss(
            mean, zeros, target, mask, replaced, date_ids=dates, scale=scale,
        )
    )
    # Huber on a large residual is > 0; replace-huber uses only ranking/IR.
    assert loss_h > 1.0
    assert loss_r < loss_h


def test_masked_loss_off_flag_ignores_weights():
    mean = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    target = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    mask = torch.ones_like(mean)
    cfg = _cfg(cost_rank_loss=False, cost_rank_weight=9.0, ir_proxy_weight=9.0)
    a = float(masked_loss(mean, torch.zeros_like(mean), target, mask, cfg))
    cfg2 = _cfg(cost_rank_loss=False, cost_rank_weight=0.0, ir_proxy_weight=0.0)
    b = float(masked_loss(mean, torch.zeros_like(mean), target, mask, cfg2))
    assert a == pytest.approx(b, abs=1e-6)


def test_cost_rank_terms_finite_and_grad():
    mean = torch.tensor([[0.4, 0.2, -0.1, -0.5]], requires_grad=True)
    y = torch.tensor([[0.5, 0.1, -0.2, -0.4]])
    mask = torch.ones_like(y)
    dates = torch.zeros_like(y, dtype=torch.long)
    scale = torch.full_like(y, 0.015)
    turn = torch.tensor([[-1.0, 0.0, 0.5, 1.2]])
    vol = torch.zeros_like(y)
    cfg = _cfg(ir_proxy_weight=0.5)
    loss = cost_rank_terms(
        mean, y, mask, cfg, date_ids=dates, scale=scale,
        turnover_z=turn.reshape(-1), vol_level=vol.reshape(-1),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()


def _symbol(name: str, n: int, turn: float, vol: float, y: float) -> SymbolArrays:
    f = len(FEATURE_NAMES)
    feat = np.zeros((n, f), dtype=np.float32)
    feat[:, FEATURE_NAMES.index("turnover_z")] = turn
    feat[:, FEATURE_NAMES.index("vol_level")] = vol
    valid = np.ones(n, dtype=bool)
    valid[:3] = False
    return SymbolArrays(
        symbol=name,
        features=feat,
        target=np.full(n, y, dtype=np.float32),
        scale=np.full(n, 0.01, dtype=np.float32),
        valid=valid,
        dates=np.arange(n, dtype=np.int64),
    )


def test_labelled_cost_aux_aligns_with_valid():
    syms = [_symbol("AAA", 8, -2.0, 1.5, 0.8), _symbol("BBB", 8, 2.0, 0.0, 0.8)]
    scale, turn, vol = labelled_cost_aux(syms)
    assert scale.shape == turn.shape == vol.shape
    assert scale.shape[0] == 10  # 5 valid each
    assert float(turn[0]) == pytest.approx(-2.0)
    assert float(turn[-1]) == pytest.approx(2.0)


def test_apply_cost_adjusted_target_changes_skip_y():
    n = 8
    dates = np.array([10] * 4 + [11] * 4, dtype=np.int64)
    y = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    feat_a = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    feat_b = feat_a.copy()
    feat_a[:, FEATURE_NAMES.index("turnover_z")] = -2.0
    feat_b[:, FEATURE_NAMES.index("turnover_z")] = 2.0
    feat_a[:, FEATURE_NAMES.index("vol_level")] = 2.0
    syms = [
        SymbolArrays(
            "THIN", feat_a, y.astype(np.float32),
            np.full(n, 0.01, dtype=np.float32), np.ones(n, dtype=bool), dates,
        ),
        SymbolArrays(
            "LIQ", feat_b, y.astype(np.float32),
            np.full(n, 0.01, dtype=np.float32), np.ones(n, dtype=bool), dates,
        ),
    ]
    # labelled_rows concatenates THIN then LIQ. Cost-adjust the stacked y.
    y_stack = np.concatenate([y, y])
    net = apply_cost_adjusted_target(y_stack, syms)
    assert net.shape == y_stack.shape
    assert not np.allclose(net, y_stack)
    # First name each date is thin; same +1 residual should rank below liquid.
    assert float(net[0]) < float(net[8])


def test_book_gate_keeps_knobs_unless_val_clears():
    payload = book_gate_payload()
    assert payload["baseline_ir"] == pytest.approx(BASELINE_IR)
    assert payload["baseline_dd"] == pytest.approx(BASELINE_DD)
    miss = decide_book_gate(5.58, -0.95)
    assert miss["promote"] is False
    assert "q20/h0.5/s0.50" in miss["keep_knobs"]
    lift_ir = decide_book_gate(5.58 + IR_LIFT + 0.01, -0.95)
    assert lift_ir["promote"] is True
    worse_dd = decide_book_gate(5.58 + IR_LIFT + 0.01, -0.95 - DD_TOL - 0.01)
    assert worse_dd["promote"] is False
    assert worse_dd["test_report_only"] is True


def test_format_block_mentions_gate():
    text = format_cost_rank_block(ForecastTrainConfig(cost_rank_loss=True))
    assert "5.58" in text
    assert "0.95" in text
    assert "q20" in text


def test_skip_fit_uses_cost_adjusted_y():
    from forecast.config import ForecastModelConfig
    from forecast.model import ReturnForecaster
    from forecast.training import apply_ridge_skip

    n, f = 24, len(FEATURE_NAMES)
    rng = np.random.default_rng(1)
    dates = np.repeat(np.arange(6, dtype=np.int64), 4)
    x = rng.normal(size=(n, f)).astype(np.float32)
    y = (x[:, 0] * 0.5).astype(np.float32)
    x[:, FEATURE_NAMES.index("turnover_z")] = np.linspace(-2, 2, n)
    x[:, FEATURE_NAMES.index("vol_level")] = 0.0
    sym = SymbolArrays(
        "AAA", x, y, np.full(n, 0.02, dtype=np.float32),
        np.ones(n, dtype=bool), dates,
    )
    bundle = {
        "train_symbols": [sym],
        "feature_mean": np.zeros(f, dtype=np.float32),
        "feature_std": np.ones(f, dtype=np.float32),
        "cs_min_names": 3,
    }
    model = ReturnForecaster(
        ForecastModelConfig(n_features=f, d_model=16, n_layer=1, d_state=8, linear_skip=True)
    )
    apply_ridge_skip(
        model, bundle,
        ForecastTrainConfig(ridge_skip=1.0, freeze_skip=True, cost_rank_loss=True),
        torch.device("cpu"),
    )
    assert float(model.skip.weight.abs().sum()) > 0.0


def test_thin_mask_is_within_date():
    turn = np.array([-2.0, -1.0, 1.0, 2.0, -2.0, -1.0, 1.0, 2.0])
    dates = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    mask = thin_mask_for_dates(turn, dates, pctile=0.5)
    assert mask[:4].tolist() == [True, True, False, False]
    assert mask[4:].tolist() == [True, True, False, False]


def test_synth_shorting_cli_is_documented():
    """Cloud CI entry: scripts/overnight_shorting.py --synthetic."""
    text = (Path(__file__).resolve().parents[1] / "scripts" / "overnight_shorting.py").read_text(
        encoding="utf-8"
    )
    assert "--synthetic" in text
    assert "live_locate" in text
