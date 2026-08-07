"""人工回测模块集成测试."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from manual_backtest.engine import ManualBacktester
from manual_backtest.analyzer import ManualAnalyzer


class TestManualBacktester:
    def test_run_pipeline_known_date(self):
        bt = ManualBacktester()
        l4 = bt.run_pipeline("2024-02-05")
        assert isinstance(l4, pd.DataFrame)
        if not l4.empty:
            for col in ["code", "buy_type", "composite", "global_rank", "zone_rank"]:
                assert col in l4.columns, f"missing column {col}"

    def test_export_and_load_roundtrip(self):
        bt = ManualBacktester()
        l4 = bt.run_pipeline("2024-08-02")
        if l4.empty:
            pytest.skip("no candidates")
        out = bt.export_for_marking()
        assert out.exists()
        df = pd.read_csv(out, encoding="utf-8-sig")
        df.at[0, "selected"] = 1
        marked_path = out.parent / out.name.replace(".csv", "_marked.csv")
        df.to_csv(marked_path, index=False, encoding="utf-8-sig")
        marked = bt.load_marked(marked_path)
        assert len(marked) == 1
        assert marked.iloc[0]["code"] == df.at[0, "code"]

    def test_backtest_selected(self):
        bt = ManualBacktester()
        l4 = bt.run_pipeline("2024-08-02")
        if l4.empty:
            pytest.skip("no candidates")
        bt.marked = l4.head(1).copy()
        bt.marked["selected"] = 1
        trades = bt.backtest_selected()
        assert isinstance(trades, pd.DataFrame)
        if not trades.empty:
            for col in ["code", "buy_type", "entry_date", "exit_date",
                         "return_pct", "hold_days", "exit_reason", "trajectory"]:
                assert col in trades.columns, f"missing {col}"
            traj = trades.iloc[0]["trajectory"]
            assert isinstance(traj, list) and len(traj) > 0

    def test_analyzer(self):
        a = ManualAnalyzer(pd.DataFrame())
        assert a.analyze()["summary"]["total_trades"] == 0
        df = pd.DataFrame([
            {"code": "000001", "buy_type": "一买", "return_pct": 5.0,
             "hold_days": 10, "exit_reason": "卖点", "regime": "BULL"},
            {"code": "000002", "buy_type": "二买", "return_pct": -3.0,
             "hold_days": 5, "exit_reason": "结构止损", "regime": "BEAR"},
        ])
        a = ManualAnalyzer(df)
        s = a.analyze()
        assert s["summary"]["total_trades"] == 2
        assert s["summary"]["win_rate"] == 0.5
