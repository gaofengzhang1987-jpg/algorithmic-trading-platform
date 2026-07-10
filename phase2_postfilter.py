#!/usr/bin/env python3
"""Phase 2: L2 单规则贡献 — 从基线交易中按规则后过滤，避免重新模拟。"""

import json, re, time
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
SIG = BASE / "data" / "signals"
DAILY = BASE / "data" / "daily"
OUT = BASE / "data" / "backtest"

# 信号列
B1="日线_D1B_BUY1"; S1="日线_D1_三买辅助V230228"; S2="日线_D1#SMA#34_BS3辅助V230319"
MC="日线_D1MACD12#26#9_BS1辅助V230313"; M5="日线_D1SMA#5_分类V221101"; M20="日线_D1SMA#20_分类V221101"


def _parse(val):
    if pd.isna(val) or str(val) == "0": return {"v1":"","v2":"","v3":"","score":"0"}
    parts = str(val).rsplit("_",3)
    return {"v1":parts[0] if len(parts)>=4 else "","v2":parts[1] if len(parts)>=4 else "","v3":parts[2] if len(parts)>=4 else "","score":parts[3] if len(parts)>=4 else "0"}


def compute_rule_map(stocks):
    """为每只股票的每个买点预计算规则通过状态。
    返回: {(code, signal_date): {rule_name: bool}}
    """
    rule_map = {}
    for i, code in enumerate(stocks):
        sp = SIG / f"{code}.parquet"
        if not sp.exists(): continue
        sd = pd.read_parquet(sp)
        if len(sd) < 2: continue

        # 预计算 per-signal-row 的规则状态
        for ri in range(len(sd)):
            row = sd.iloc[ri]
            dt = str(row["dt"].date()) if hasattr(row["dt"],'date') else ""

            # L2-01: 笔数 ≥ 9
            stroke_ok = False
            for col in [B1,S1,S2]:
                if col not in sd.columns: continue
                p = _parse(str(sd.iloc[ri][col]))
                if ("一买" in p["v1"] or "三买" in p["v1"]) and (m := re.search(r'(\d+)', p["v2"])):
                    if int(m.group(1)) >= 9: stroke_ok = True; break

            # L2-09: MACD 金叉 (近5天)
            macd_ok = False
            if MC in sd.columns:
                start = max(0, ri-4)
                for j in range(start, ri+1):
                    if "金叉" in _parse(str(sd.iloc[j][MC]))["v2"]:
                        macd_ok = True; break

            # L2-10: MA 多头
            ma5_u = ma20_u = False
            if M5 in sd.columns:
                p=_parse(str(sd.iloc[ri][M5])); ma5_u="多头" in p["v1"] and "向上" in p["v2"]
            if M20 in sd.columns:
                p=_parse(str(sd.iloc[ri][M20])); ma20_u="多头" in p["v1"] and "向上" in p["v2"]
            ma_ok = ma5_u and ma20_u

            rule_map[(code, dt)] = {
                "L2-01_笔数": stroke_ok,
                "L2-09_MACD金叉": macd_ok,
                "L2-10_MA多头": ma_ok,
            }

        if (i+1) % 200 == 0: print(f"  预计算规则: {i+1}/{len(stocks)}", flush=True)

    return rule_map


def compute_volume_map(stocks):
    """L2-06: 成交量 — 每天都会变, 需要 per-date 检查。"""
    vol_map = {}
    for i, code in enumerate(stocks):
        dp = DAILY / f"{code}.parquet"
        if not dp.exists(): continue
        dd = pd.read_parquet(dp)
        if len(dd) < 20: continue
        # 对每个交易日计算当天 MA20 和量比
        for ri in range(20, len(dd)):
            dt = str(dd.iloc[ri]["date"].date())
            last_vol = float(dd.iloc[ri]["volume"])
            ma_vol = float(dd.iloc[ri-19:ri+1]["volume"].mean())
            vol_map[(code, dt)] = (ma_vol > 0 and last_vol / ma_vol >= 1.5)

        if (i+1) % 200 == 0: print(f"  成交量缓存: {i+1}/{len(stocks)}", flush=True)

    return vol_map


def compute_metrics(trades_list):
    import numpy as np
    if not trades_list: return {"total_trades":0,"win_rate":0,"avg_return":0,"sharpe":0}
    returns = [t["return_pct"]/100 for t in trades_list]
    hold_days = [t["hold_days"] for t in trades_list]
    wins = sum(1 for r in returns if r > 0)
    n = len(trades_list)
    wr = wins/n if n>0 else 0
    avg_ret = np.mean(returns) if returns else 0
    avg_hold = np.mean(hold_days) if hold_days else 0
    if len(returns)>1 and avg_hold>0:
        sharpe = (avg_ret/(np.std(returns)+1e-8))*np.sqrt(252/avg_hold)
    else:
        sharpe = 0
    return {"total_trades":n, "win_rate":round(wr*100,1), "avg_return":round(avg_ret*100,2), "sharpe":round(sharpe,2)}


def main():
    print("=== Phase 2: 后过滤单规则回测 ===")
    # 加载基线
    baseline = pd.read_parquet(OUT / "baseline_all.parquet")
    bl_metrics = compute_metrics(baseline.to_dict("records"))
    bl_metrics["rule"] = "BASELINE"
    print(f"Baseline: {bl_metrics['total_trades']} trades, WR={bl_metrics['win_rate']}%, Ret={bl_metrics['avg_return']}%, Sharpe={bl_metrics['sharpe']}")

    # 获取所有股票
    stocks = baseline["code"].unique().tolist()
    print(f"Stocks: {len(stocks)}")

    # 预计算规则映射
    t0 = time.time()
    rule_map = compute_rule_map(stocks)
    vol_map = compute_volume_map(stocks)
    print(f"预计算完成: {time.time()-t0:.0f}s")

    # 为 L2-06 添加成交量检查到 rule_map
    for (code, dt), rules in rule_map.items():
        rules["L2-06_成交量"] = vol_map.get((code, dt), False)

    # 过滤基线交易
    results = [bl_metrics]
    rule_names = ["L2-01_笔数", "L2-06_成交量", "L2-09_MACD金叉", "L2-10_MA多头"]

    for rn in rule_names:
        t0 = time.time()
        filtered = []
        for _, t in baseline.iterrows():
            code, sd = t["code"], t["signal_date"]
            rules = rule_map.get((code, sd), {})
            if rules.get(rn, False):
                filtered.append(t.to_dict())

        m = compute_metrics(filtered)
        m["rule"] = rn
        dt = time.time() - t0
        print(f"  {rn}: WR={m['win_rate']}% Ret={m['avg_return']}% Sharpe={m['sharpe']} Trades={m['total_trades']} ({dt:.1f}s)")
        results.append(m)

    # Summary
    print(f"\n{'Rule':<20} {'Trades':>8} {'WR%':>8} {'Ret%':>8} {'Sharpe':>8} {'ΔWR':>8}")
    print("-"*80)
    for r in results:
        wr = r.get("win_rate",0)
        dw = wr - bl_metrics["win_rate"]
        print(f'{r["rule"]:<20} {r.get("total_trades",0):>8} {wr:>8.1f} {r.get("avg_return",0):>8.2f} {r.get("sharpe",0):>8.2f} {dw:>+8.1f}')

    json.dump(results, open(OUT/"phase2_results.json","w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT/'phase2_results.json'}")


if __name__ == "__main__":
    main()
