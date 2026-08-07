"""Backtrader 交叉验证执行器。"""
import pandas as pd, numpy as np
from pathlib import Path
import backtrader as bt
from backtrader_xv.adapter import parquet_to_bt_cerebro
from backtrader_xv.strategy import CZSCSignalStrategy
from core.constants import COMMISSION, CAPITAL_PER_TRADE


def run_single_stock(code, cash=100000, commission=0.001):
    cerebro = parquet_to_bt_cerebro(code, cash=cash, commission=commission)
    cerebro.addstrategy(CZSCSignalStrategy, code=code)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.02, annualize=True)
    try:
        results = cerebro.run()
    except Exception as e:
        print(f"  {code} Backtrader error: {e}", flush=True)
        return None

    strat = results[0]
    ta = strat.analyzers.trades.get_analysis()
    total = ta.get("total", {}).get("total", 0)
    if total == 0:
        return None
    won = ta.get("won", {}).get("total", 0)
    win_rate = won / total if total > 0 else 0.0
    sr = strat.analyzers.sharpe.get_analysis()
    sharpe = sr.get("sharperatio", 0) or 0.0
    trade_details = []
    for t in (ta.get("closed", []) or []):
        pnl = float(t.get("pnl", 0))
        val = float(t.get("value", 1))
        trade_details.append({
            "entry_date": str(t.get("dt", "")).split(" ")[0],
            "return_pct": round(pnl / val * 100, 2) if val else 0,
            "hold_days": int(t.get("len", 0)),
        })
    avg_return = float(np.mean([t["return_pct"] for t in trade_details])) if trade_details else 0.0
    return {
        "code": code, "total_trades": int(total),
        "win_rate": round(float(win_rate) * 100, 1),
        "avg_return": round(avg_return, 2),
        "sharpe": round(float(sharpe), 2),
        "trade_details": trade_details,
    }


def run_comparison(codes=None, max_stocks=50):
    bt_file = Path("tmp_out/backtest_1500.parquet")
    if codes is None and bt_file.exists():
        df_custom = pd.read_parquet(bt_file)
        codes = sorted(df_custom["code"].unique())[:max_stocks]
    elif codes is None:
        from core.constants import SIGNALS_DIR
        codes = sorted(p.stem for p in SIGNALS_DIR.glob("*.parquet"))[:max_stocks]

    print(f"Backtrader 交叉验证: {len(codes)} 只")
    bt_results = []
    for i, code in enumerate(codes):
        r = run_single_stock(code, cash=CAPITAL_PER_TRADE, commission=COMMISSION)
        if r:
            bt_results.append(r)
        if (i + 1) % 10 == 0:
            done = sum(r["total_trades"] for r in bt_results) if bt_results else 0
            print(f"  进度: {i+1}/{len(codes)}, 有效: {len(bt_results)}, 交易: {done}", flush=True)

    if not bt_results:
        print("Backtrader 无交易")
        return {}

    df_bt = pd.DataFrame(bt_results)
    bt_metrics = {
        "total_trades": int(df_bt["total_trades"].sum()),
        "win_rate": round(float(np.average(df_bt["win_rate"].values, weights=df_bt["total_trades"].values)), 1),
        "avg_return": round(float(df_bt["avg_return"].mean()), 2),
        "sharpe": round(float(df_bt["sharpe"].mean()), 2),
    }

    custom_metrics = {}
    if bt_file.exists():
        df_custom = pd.read_parquet(bt_file)
        subset = df_custom[df_custom["code"].isin(codes)]
        from core.metrics import compute_metrics
        custom_metrics = compute_metrics(subset.to_dict("records"), weighted=False)

    comparison = {}
    if custom_metrics:
        comparison = {
            "trade_count_ratio": round(bt_metrics["total_trades"] / max(custom_metrics.get("total_trades", 1), 1), 2),
            "win_rate_diff": round(bt_metrics["win_rate"] - custom_metrics.get("win_rate", 0), 1),
            "avg_return_diff": round(bt_metrics["avg_return"] - custom_metrics.get("avg_return", 0), 2),
        }

    print(f"\nBacktrader: {bt_metrics}")
    print(f"Custom:     {custom_metrics}")
    print(f"差异:       {comparison}")
    return {"bt_metrics": bt_metrics, "custom_metrics": custom_metrics, "comparison": comparison}
