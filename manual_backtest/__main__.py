"""CLI entry: python3 -m manual_backtest date|batch|backtest|analyze."""
import sys
from pathlib import Path

import pandas as pd

from manual_backtest.engine import ManualBacktester
from manual_backtest.analyzer import ManualAnalyzer
from manual_backtest.report import export_trades_csv, print_summary

_OUT_BASE = Path(__file__).parent.parent / "tmp_out" / "manual_backtest"



def _usage() -> None:
    print("Usage: python3 -m manual_backtest {historical|backtest|analyze} [args...]")
    
    print("  historical YYYY-MM-DD [--sample N] [--workers N]")
    print("  backtest MARKED_CSV [--out OUT]")
    print("  analyze TRADES_CSV [--auto-csv AUTO] [--compare-top N]")


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
    elif cmd == "historical":
        if len(sys.argv) < 3:
            print("Usage: python3 -m manual_backtest historical YYYY-MM-DD [--sample N] [--workers N]"); sys.exit(1)
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("date")
        ap.add_argument("--sample", type=int, default=0)
        ap.add_argument("--workers", type=int, default=4)
        a, _ = ap.parse_known_args(sys.argv[2:])
        from manual_backtest.historical import run
        run(a.date, sample=a.sample, workers=a.workers)
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
