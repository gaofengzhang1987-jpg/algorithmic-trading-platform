"""人工回测模块 — L4 报告输出 → 人工标记 → 逐只回测 → 统计分析."""

from manual_backtest.engine import ManualBacktester
from manual_backtest.report import export_l4_csv, export_trades_csv, print_summary

__all__ = ["ManualBacktester", "export_l4_csv", "export_trades_csv", "print_summary"]
