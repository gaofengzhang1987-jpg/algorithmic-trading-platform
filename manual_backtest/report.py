"""CSV 输出 + 控制台摘要."""
import json
from pathlib import Path

import pandas as pd


L4_COLUMNS = [
    "selected", "global_rank", "zone_rank", "code", "name", "buy_type",
    "signal_date", "composite", "n_l2", "stock_rps", "sector_rps",
    "qlib_score", "regime", "sector", "total_score", "passed",
]

TRADE_COLUMNS = [
    "code", "buy_type", "signal_date", "entry_date", "exit_date",
    "entry_price", "exit_price", "return_pct", "hold_days",
    "exit_reason", "l4_rank", "composite", "regime", "trajectory_json",
]


def export_l4_csv(l4_df: pd.DataFrame, out_path: Path) -> Path:
    """导出 L4 报告 CSV，selected 默认为 0."""
    df = l4_df.copy()
    if "selected" not in df.columns:
        df["selected"] = 0
    for col in L4_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[L4_COLUMNS].to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def export_trades_csv(trades_df: pd.DataFrame, out_path: Path) -> Path:
    """导出回测结果 CSV，trajectory 序列化为 JSON 字符串."""
    df = trades_df.copy()
    if "trajectory" in df.columns:
        df["trajectory_json"] = df["trajectory"].apply(
            lambda t: json.dumps(t, ensure_ascii=False) if isinstance(t, list) else ""
        )
        df = df.drop(columns=["trajectory"])
    for col in TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[TRADE_COLUMNS].to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def print_summary(stats: dict):
    """控制台打印分层统计摘要."""
    s = stats.get("summary", {})
    print(f"\n{'='*60}")
    print(f"  回测统计摘要")
    print(f"{'='*60}")
    print(f"  总交易笔数:    {s.get('total_trades', 0)}")
    print(f"  胜率:          {s.get('win_rate', 0):.1%}")
    print(f"  平均盈亏:      {s.get('avg_return', 0):.2%}")
    print(f"  盈亏比:        {s.get('win_loss_ratio', 0):.2f}")
    print(f"  平均持仓天数:  {s.get('avg_hold_days', 0):.0f}")
    print(f"  最大单笔盈利:  {s.get('max_win', 0):.2%}")
    print(f"  最大单笔亏损:  {s.get('max_loss', 0):.2%}")
    print(f"  总复合收益:    {s.get('total_return', 0):.2%}")

    for section, title in [
        ("by_buy_type", "按买点类型"),
        ("by_regime", "按市场状态"),
        ("by_exit_reason", "按出场原因"),
    ]:
        data = stats.get(section, {})
        if not data:
            continue
        print(f"\n  {title}:")
        print(f"  {'类型':<12} {'笔数':<6} {'胜率':<8} {'平均收益':<10}")
        print(f"  {'-'*36}")
        for k, v in data.items():
            print(f"  {k:<12} {v.get('count',0):<6} {v.get('win_rate',0):.1%}  {v.get('avg_return',0):.2%}".rstrip())
    print(f"{'='*60}\n")
