"""CLI entry: python3 -m manual_backtest date|batch|backtest|analyze."""
import sys
from pathlib import Path

import pandas as pd

from manual_backtest.engine import ManualBacktester
from manual_backtest.analyzer import ManualAnalyzer
from manual_backtest.report import export_trades_csv, print_summary

_OUT_BASE = Path(__file__).parent.parent / "tmp_out" / "manual_backtest"


def _parse_max(args: list) -> int:
    """从 args 列表中提取 --max N."""
    for i, a in enumerate(args):
        if a == '--max' and i + 1 < len(args):
            return int(args[i + 1])
    return 0


def _usage() -> None:
    print("Usage: python3 -m manual_backtest {date|batch|backtest|analyze} [args...]")
    print("  date YYYY-MM-DD [--max N]")
    print("  batch YYYY-MM-DD YYYY-MM-DD [--max N]")
    print("  backtest MARKED_CSV [--out OUT]")
    print("  analyze TRADES_CSV [--auto-csv AUTO] [--compare-top N]")


def cmd_date(date: str, max_candidates: int = 0):
    config = {} if max_candidates <= 0 else {"max_candidates": max_candidates}
    bt = ManualBacktester(config)
    l4 = bt.run_pipeline(date)
    if l4.empty:
        print(f"  {date}: 无候选")
        return
    out = bt.export_for_marking()
    print(f"  L4: {out} ({len(l4)} candidates)")
    print(f"  标记后运行: python3 -m manual_backtest backtest {out}")


def cmd_batch(start: str, end: str, max_candidates: int = 0):
    config = {} if max_candidates <= 0 else {"max_candidates": max_candidates}
    bt = ManualBacktester(config)
    dates = pd.date_range(start, end, freq="B")
    print(f"  {start} -> {end}, {len(dates)} trading days")
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        try:
            l4 = bt.run_pipeline(ds)
            if not l4.empty:
                bt.export_for_marking()
            print(f"  {ds}: {len(l4)} candidates")
        except Exception as e:
            print(f"  {ds}: error — {e}")


def cmd_backtest(marked_csv: str, out: str | None = None):
    bt = ManualBacktester()
    marked = bt.load_marked(marked_csv)
    print(f"  Marked: {len(marked)} stocks")
    trades = bt.backtest_selected()
    if trades.empty:
        print("  No trades")
        return
    out_path = Path(out or (Path(marked_csv).parent / "trades.csv"))
    export_trades_csv(trades, out_path)
    print(f"  Trades: {out_path} ({len(trades)} executions)")
    analyzer = ManualAnalyzer(trades)
    print_summary(analyzer.analyze())


def cmd_analyze(trades_csv: str, auto_csv: str | None = None, compare_top: int = 50):
    trades_df = pd.read_csv(trades_csv, encoding="utf-8-sig")
    print(f"  Loaded: {len(trades_df)} trades")
    auto_df = None
    if auto_csv:
        auto_df = pd.read_csv(auto_csv, encoding="utf-8-sig")
        print(f"  Auto: {len(auto_df)} trades")
    analyzer = ManualAnalyzer(trades_df, auto_df)
    stats = analyzer.analyze()
    print_summary(stats)
    if auto_df is not None and not auto_df.empty:
        cmp = analyzer.compare(top_n=compare_top)
        print("\n  人工 vs 自动对比:")
        print(cmp.to_string(index=False))


def main():
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd in ("--help", "-h"):
        _usage()
        sys.exit(0)
    if cmd == "date":
        if len(sys.argv) < 3:
            print("Usage: python3 -m manual_backtest date YYYY-MM-DD [--max N]"); sys.exit(1)
        max_n = _parse_max(sys.argv[3:])
        cmd_date(sys.argv[2], max_n)
    elif cmd == "batch":
        if len(sys.argv) < 4:
            print("Usage: python3 -m manual_backtest batch START END [--max N]"); sys.exit(1)
        max_n = _parse_max(sys.argv[4:])
        cmd_batch(sys.argv[2], sys.argv[3], max_n)
    elif cmd == "backtest":
        if len(sys.argv) < 3:
            print("Usage: python3 -m manual_backtest backtest MARKED_CSV [--out OUT]"); sys.exit(1)
        out = None
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "--out" and i + 1 < len(args):
                out = args[i + 1]; i += 2
            else:
                i += 1
        cmd_backtest(sys.argv[2], out)
    elif cmd == "analyze":
        if len(sys.argv) < 3:
            print("Usage: python3 -m manual_backtest analyze TRADES_CSV [--auto-csv AUTO] [--compare-top N]"); sys.exit(1)
        auto_csv = None; compare_top = 50
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            if args[i] == "--auto-csv" and i + 1 < len(args):
                auto_csv = args[i + 1]; i += 2
            elif args[i] == "--compare-top" and i + 1 < len(args):
                compare_top = int(args[i + 1]); i += 2
            else:
                i += 1
        cmd_analyze(sys.argv[2], auto_csv, compare_top)
    else:
        print(f"Unknown command: {cmd}"); sys.exit(1)


if __name__ == "__main__":
    main()
