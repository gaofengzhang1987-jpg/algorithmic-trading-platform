#!/usr/bin/env python3
"""Phase 2: L2-12 力度背驰 — 后过滤方式，每只股票只创建一次 CZSC"""

import json, re, time
from pathlib import Path
import pandas as pd
from czsc import CZSC, RawBar, Freq
from czsc.objects import Direction

BASE = Path(__file__).parent
SIG = BASE / "data" / "signals"
DAILY = BASE / "data" / "daily"
OUT = BASE / "data" / "backtest"

def compute_divergence_map(stocks):
    """每只股票计算一次 CZSC, 判断力度背驰: B.power < A.power"""
    div_map = {}
    for i, code in enumerate(stocks):
        dp = DAILY / f"{code}.parquet"
        if not dp.exists(): continue
        df = pd.read_parquet(dp)
        if len(df) < 30: continue
        bars = [RawBar(symbol=code, id=j+1, dt=r["date"].to_pydatetime(), freq=Freq.D,
                open=r["open"],close=r["close"],high=r["high"],low=r["low"],
                vol=r["volume"],amount=r["amount"]) for j,(_,r) in enumerate(df.iterrows())]
        try:
            c = CZSC(bars)
        except: continue
        dbs = [bi for bi in c.bi_list if bi.direction == Direction.Down]
        div_ok = False
        if len(dbs) >= 2 and dbs[-2].power > 0:
            div_ok = dbs[-1].power < dbs[-2].power
        div_map[code] = div_ok

        if (i+1)%200==0: print(f"  背驰预计算: {i+1}/{len(stocks)}", flush=True)
    return div_map


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
    print("=== Phase 2: L2-12 力度背驰 (后过滤) ===")
    baseline = pd.read_parquet(OUT / "baseline_all.parquet")
    bl = compute_metrics(baseline.to_dict("records"))
    bl["rule"] = "BASELINE"
    print(f"Baseline: {bl['total_trades']} trades, WR={bl['win_rate']}%")

    stocks = baseline["code"].unique().tolist()

    t0 = time.time()
    div_map = compute_divergence_map(stocks)
    print(f"背驰预计算完成: {time.time()-t0:.0f}s, {sum(div_map.values())} stocks with divergence")

    t0 = time.time()
    filtered = []
    for _, t in baseline.iterrows():
        if div_map.get(t["code"], False):
            filtered.append(t.to_dict())

    m = compute_metrics(filtered)
    m["rule"] = "L2-12_力度背驰"
    dt = time.time()-t0
    print(f"  L2-12_力度背驰: WR={m['win_rate']}% Ret={m['avg_return']}% Sharpe={m['sharpe']} Trades={m['total_trades']} ({dt:.1f}s)")

    # Load existing phase2 results for comparison
    p2_path = OUT / "phase2_results.json"
    results = json.loads(p2_path.read_text()) if p2_path.exists() else [bl]
    results.append(m)

    print(f"\n{'Rule':<20} {'Trades':>8} {'WR%':>8} {'Ret%':>8} {'Sharpe':>8} {'ΔWR':>8}")
    print("-"*80)
    for r in results:
        wr = r.get("win_rate",0); dw = wr - bl["win_rate"]
        print(f'{r["rule"]:<20} {r.get("total_trades",0):>8} {wr:>8.1f} {r.get("avg_return",0):>8.2f} {r.get("sharpe",0):>8.2f} {dw:>+8.1f}')

    json.dump(results, open(p2_path,"w"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
