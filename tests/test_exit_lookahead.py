"""出场回测未来数据回归测试 — 2026-08-14 修复。"""
import pandas as pd
import pytest

from backtest.exit_engine import ExitEngine
from manual_backtest import historical


_COLS = [
    "direction", "high", "low", "sdt", "edt", "power", "fx_b_mark",
    "fx_b_low", "fx_a_mark", "pivot_dir", "pivot_gg", "pivot_zd", "pivot_zg",
]


def _struct_df(rows):
    return pd.DataFrame([{c: r.get(c, "") for c in _COLS} for r in rows])


def _pivot(edt, gg=15.0, sdt="2024-01-15"):
    return {
        "direction": "向上", "high": gg, "low": 9.0, "sdt": sdt, "edt": edt,
        "power": 1.0, "fx_b_mark": "", "fx_b_low": 0.0, "fx_a_mark": "",
        "pivot_dir": "上涨", "pivot_gg": gg, "pivot_zd": 10.0, "pivot_zg": 12.0,
    }


def test_snapshot_ignores_unfinished_pivot():
    struct = _struct_df([_pivot(edt="2024-01-15", gg=12.0)])
    eng = ExitEngine("000001", 10.0, "2024-01-10", "三买_标准", struct)
    assert eng.entry_pivot_gg == 0.0


def test_snapshot_uses_finished_pivot():
    struct = _struct_df([_pivot(edt="2024-01-05", gg=12.0, sdt="2023-12-20")])
    eng = ExitEngine("000001", 10.0, "2024-01-10", "三买_标准", struct)
    assert eng.entry_pivot_gg == pytest.approx(12.0)


def test_update_defense_ignores_unfinished_pivot():
    struct = _struct_df([_pivot(edt="2024-01-31")])
    eng = ExitEngine("000001", 10.0, "2024-01-10", "二买_标准", struct)
    before = eng.defense
    eng.update_defense("2024-01-20")
    assert eng.defense == before


def test_update_defense_uses_finished_pivot():
    struct = _struct_df([_pivot(edt="2024-01-25")])
    eng = ExitEngine("000001", 10.0, "2024-01-10", "二买_标准", struct)
    eng.update_defense("2024-01-26")
    assert eng.defense == pytest.approx(15.0)


def test_patch_paths_points_rps_to_section(tmp_path):
    import l3_filter
    from qlib_ml import signal_predictor
    import zone3_regime

    bt = tmp_path / "bt"
    orig_signal_daily = signal_predictor.DATA_DIR
    orig_signal_signals = signal_predictor.SIGNAL_DIR
    restore = historical._patch_paths(bt, pd.Timestamp("2024-01-01"))
    try:
        assert l3_filter.STOCK_RPS == bt / "reference" / "stock_rps.parquet"
        assert zone3_regime.INDUSTRY_RPS == bt / "reference" / "industry_rps.parquet"
        assert zone3_regime.INDUSTRY_MAP == bt / "reference" / "industry_classification.parquet"
        assert l3_filter.L3Filter._rps_stock is None
        assert signal_predictor.DATA_DIR == bt / "daily"
        assert signal_predictor.SIGNAL_DIR == bt / "signals"
    finally:
        restore()
    assert l3_filter.STOCK_RPS == l3_filter.WORKDIR / "data" / "reference" / "stock_rps.parquet"
    assert zone3_regime.INDUSTRY_RPS == zone3_regime.BASE / "data" / "reference" / "industry_rps.parquet"
    assert signal_predictor.DATA_DIR == orig_signal_daily
    assert signal_predictor.SIGNAL_DIR == orig_signal_signals


def test_snapshot_reference_data_filters_cutoff(tmp_path, monkeypatch):
    monkeypatch.setattr(historical, "BASE", tmp_path)
    ref = tmp_path / "data" / "reference"
    ref.mkdir(parents=True)
    sr = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        "code": ["a", "a"], "rps_20d": [1.0, 2.0],
    })
    sr.to_parquet(ref / "stock_rps.parquet")
    ir = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "industry": ["x"], "rps_20d": [3.0],
    })
    ir.to_parquet(ref / "industry_rps.parquet")
    (tmp_path / "data" / "industry_classification.parquet").write_bytes(b"")

    bt = tmp_path / "bt"
    historical._snapshot_reference_data(bt, pd.Timestamp("2024-01-15"))

    out = pd.read_parquet(bt / "reference" / "stock_rps.parquet")
    assert out["date"].max() == pd.Timestamp("2024-01-01")
    assert (bt / "reference" / "industry_classification.parquet").exists()


def _daily_up_bis(powers):
    rows = []
    for i, p in enumerate(powers):
        rows.append({
            "direction": "向上", "high": 11.0, "low": 9.0,
            "sdt": f"2023-12-{10 + i * 7:02d}", "edt": f"2023-12-{17 + i * 7:02d}",
            "power": float(p), "fx_b_mark": "", "fx_b_low": 0.0,
            "fx_a_mark": "", "pivot_dir": "", "pivot_gg": 0.0,
            "pivot_zd": 0.0, "pivot_zg": 0.0,
        })
    return _struct_df(rows)


def test_divergence_uses_30m_tier1(monkeypatch):
    import backtest.exit_engine as ee

    struct30 = pd.DataFrame([
        {"direction": "向上", "edt": pd.Timestamp("2024-01-08"), "power": 5.0, "fx_b_has_zs": False},
        {"direction": "向上", "edt": pd.Timestamp("2024-01-15"), "power": 3.0, "fx_b_has_zs": False},
    ])
    monkeypatch.setattr(ee, "_load_struct_30m", lambda code: struct30)
    eng = ExitEngine("000001", 10.0, "2024-01-01", "二买_标准", _daily_up_bis([1.0, 2.0]))
    assert eng.check_divergence("2024-01-20") is True


def test_divergence_tier1_ignores_zs_bis(monkeypatch):
    import backtest.exit_engine as ee

    struct30 = pd.DataFrame([
        {"direction": "向上", "edt": pd.Timestamp("2024-01-08"), "power": 5.0, "fx_b_has_zs": False},
        {"direction": "向上", "edt": pd.Timestamp("2024-01-12"), "power": 100.0, "fx_b_has_zs": True},
        {"direction": "向上", "edt": pd.Timestamp("2024-01-15"), "power": 8.0, "fx_b_has_zs": False},
    ])
    monkeypatch.setattr(ee, "_load_struct_30m", lambda code: struct30)
    eng = ExitEngine("000001", 10.0, "2024-01-01", "二买_标准", _daily_up_bis([1.0, 2.0]))
    assert eng.check_divergence("2024-01-20") is False


def test_divergence_falls_back_without_has_zs(monkeypatch):
    import backtest.exit_engine as ee

    struct30 = pd.DataFrame([
        {"direction": "向上", "edt": pd.Timestamp("2024-01-08"), "power": 5.0},
        {"direction": "向上", "edt": pd.Timestamp("2024-01-15"), "power": 3.0},
    ])
    monkeypatch.setattr(ee, "_load_struct_30m", lambda code: struct30)
    eng = ExitEngine("000001", 10.0, "2024-01-01", "二买_标准", _daily_up_bis([2.0, 1.0]))
    assert eng.check_divergence("2024-01-20") is True


def test_l4_ranker_passes_signal_date():
    from unittest.mock import Mock

    from l4_ranker import L4Ranker

    df = pd.DataFrame({
        "code": ["000001", "000002"],
        "buy_type": ["二买", "二买"],
        "passed": True,
        "total_score": [80.0, 60.0],
        "sector_rps": [60.0, 70.0],
        "stock_rps": [50.0, 90.0],
    })
    pred = Mock()
    pred.score.return_value = pd.Series([0.6, 0.8], index=["000001", "000002"])
    out = L4Ranker(qlib_predictor=pred).rank(df, signal_date="2021-03-01")
    pred.score.assert_called_once_with(["000001", "000002"], signal_date="2021-03-01")
    assert out["qlib_score"].notna().all()


def test_zone4_passes_signal_date_to_ranker(tmp_path, monkeypatch):
    from unittest.mock import Mock

    import zone4_regime

    fake = Mock()
    fake.rank.return_value = pd.DataFrame({
        "code": ["000001"],
        "买点类型": ["二买_标准"],
        "L2_Regime": ["CHOP"],
        "L2_综合得分": [80.0],
        "total_score": [80.0],
        "composite": [0.8],
        "global_rank": [1],
    })
    monkeypatch.setattr(zone4_regime, "L4Ranker", lambda **kw: fake)
    monkeypatch.setattr(zone4_regime, "SignalQlibPredictor", lambda: None)
    monkeypatch.setattr(zone4_regime, "ZONES", tmp_path)
    monkeypatch.setattr(zone4_regime, "NAME_DB", tmp_path / "missing.parquet")
    input_df = pd.DataFrame({
        "代码": ["000001"],
        "买点类型": ["二买_标准"],
        "L2_综合得分": [80.0],
        "sector_rps": [60.0],
        "stock_rps": [50.0],
        "现价": [10.0],
        "最新日期": ["2021-03-01"],
        "L2_Regime": ["CHOP"],
    })
    zone4_regime.run(input_df, top_n=100, skip_fundamental=True, signal_date="2021-03-01")
    assert fake.rank.call_args.kwargs["signal_date"] == "2021-03-01"
