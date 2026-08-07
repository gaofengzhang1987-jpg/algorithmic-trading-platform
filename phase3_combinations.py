#!/usr/bin/env python3
"""Phase 3: L2 规则组合回测 + Ablation 移除实验"""

import json, re, time
from pathlib import Path
import pandas as pd
import numpy as np
from core.signal_parser import parse_signal
import logging
logger = logging.getLogger(__name__)  # L3组合回测

BASE = Path(__file__).parent
SIG = BASE / "data" / "signals"
DAILY = BASE / "data" / "daily"
OUT = BASE / "data" / "backtest"

B1="日线_D1B_BUY1"; S1="日线_D1_三买辅助V230228"; S2="日线_D1#SMA#34_BS3辅助V230319"
BS2="日线_D1#SMA#21_BS2辅助V230320"; BS3="日线_D1#SMA#34_BS3辅助V230318"
MC="日线_D1MACD12#26#9_BS1辅助V230313"; M5="日线_D1SMA#5_分类V221101"; M20="日线_D1SMA#20_分类V221101"


def compute_full_rule_map(stocks):
    """Precompute ALL rules per (code, signal_date) pair."""
    rule_map = {}
    for i, code in enumerate(stocks):
        sp = SIG / f"{code}.parquet"
        if not sp.exists(): continue
        sd = pd.read_parquet(sp)
        if len(sd) < 2: continue

        for ri in range(len(sd)):
            row = sd.iloc[ri]
            dt = str(row["dt"].date()) if hasattr(row["dt"],'date') else ""

            # L2-02: 买点类型分级
            bp1 = parse_signal(str(sd.iloc[ri][B1]) if B1 in sd.columns else "")
            bp2 = parse_signal(str(sd.iloc[ri][BS2]) if BS2 in sd.columns else "")
            sp  = parse_signal(str(sd.iloc[ri][S1]) if S1 in sd.columns else "")
            sp2 = parse_signal(str(sd.iloc[ri][S2]) if S2 in sd.columns else "")
            b3p = parse_signal(str(sd.iloc[ri][BS3]) if BS3 in sd.columns else "")
            has_ym = "一买" in bp1["v1"]; has_em = "二买" in bp2["v1"]
            has_sm = "三买" in (sp["v1"]+sp2["v1"]+b3p["v1"])
            macd_p = parse_signal(str(sd.iloc[ri][MC]) if MC in sd.columns else "")
            macd_golden = "金叉" in macd_p["v2"]
            buytype_ok = has_ym or has_sm or (has_em and macd_golden)

            # L2-03: 信号新鲜度 (近30天有变化)
            fresh_ok = False
            if ri >= 30:
                lookback = sd.iloc[max(0,ri-30):ri+1]
                buy_cols = [c for c in [B1,BS2,BS3,S1,S2] if c in sd.columns]
                for col in buy_cols:
                    for j in range(1, len(lookback)):
                        o = str(lookback.iloc[j-1][col]); n = str(lookback.iloc[j][col])
                        if o != n and any(k in parse_signal(n)["v1"] for k in ["一买","二买","三买"]):
                            fresh_ok = True; break
                    if fresh_ok: break
            else:
                fresh_ok = True  # 数据不足, 默认通过

            # L2-01: 笔数
            stroke_ok = False
            for col in [B1,S1,S2]:
                if col not in sd.columns: continue
                p = parse_signal(str(sd.iloc[ri][col]))
                if ("一买" in p["v1"] or "三买" in p["v1"]) and (m := re.search(r'(\d+)', p["v2"])):
                    if int(m.group(1)) >= 9: stroke_ok = True; break

            # L2-09: MACD金叉
            macd_ok = False
            if MC in sd.columns:
                for j in range(max(0,ri-4), ri+1):
                    if "金叉" in parse_signal(str(sd.iloc[j][MC]))["v2"]:
                        macd_ok = True; break

            # L2-10: MA多头
            ma5_u = ma20_u = False
            if M5 in sd.columns:
                p=parse_signal(str(sd.iloc[ri][M5])); ma5_u="多头" in p["v1"] and "向上" in p["v2"]
            if M20 in sd.columns:
                p=parse_signal(str(sd.iloc[ri][M20])); ma20_u="多头" in p["v1"] and "向上" in p["v2"]
            ma_ok = ma5_u and ma20_u

            rule_map[(code, dt)] = {
                "L2-02_买点类型": buytype_ok,
                "L2-03_新鲜度": fresh_ok,
                "L2-01_笔数": stroke_ok,
                "L2-09_MACD金叉": macd_ok,
                "L2-10_MA多头": ma_ok,
            }

        if (i+1) % 200 == 0: print(f"  预计算规则: {i+1}/{len(stocks)}", flush=True)

    return rule_map


def compute_volume_map(stocks):
    vol_map = {}
    for i, code in enumerate(stocks):
        dp = DAILY / f"{code}.parquet"
        if not dp.exists(): continue
        dd = pd.read_parquet(dp)
        if len(dd) < 20: continue
        for ri in range(20, len(dd)):
            dt = str(dd.iloc[ri]["date"].date())
            last_vol = float(dd.iloc[ri]["volume"])
            ma_vol = float(dd.iloc[ri-19:ri+1]["volume"].mean())
            vol_map[(code, dt)] = (ma_vol > 0 and last_vol / ma_vol >= 1.5)
        if (i+1) % 200 == 0: print(f"  成交量缓存: {i+1}/{len(stocks)}", flush=True)
    return vol_map


def compute_divergence_map(stocks):
    from czsc import CZSC, RawBar, Freq
    from czsc.objects import Direction
    div_map = {}
    for i, code in enumerate(stocks):
        dp = DAILY / f"{code}.parquet"
        if not dp.exists(): continue
        df = pd.read_parquet(dp)
        if len(df) < 30: continue
        bars = [RawBar(symbol=code, id=j+1, dt=r["date"].to_pydatetime(), freq=Freq.D,
                open=r["open"],close=r["close"],high=r["high"],low=r["low"],
                vol=r["volume"],amount=r["amount"]) for j,(_,r) in enumerate(df.iterrows())]
        try: c = CZSC(bars)
        except: continue
        dbs = [bi for bi in c.bi_list if bi.direction == Direction.Down]
        div_ok = len(dbs)>=2 and dbs[-2].power>0 and dbs[-1].power < dbs[-2].power
        div_map[code] = div_ok
        if (i+1)%200==0: print(f"  背驰预计算: {i+1}/{len(stocks)}", flush=True)
    return div_map


def compute_metrics(trades_list):
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
    print("=== Phase 3: L2 规则组合回测 + Ablation ===")
    baseline = pd.read_parquet(OUT / "baseline_all.parquet")
    bl = compute_metrics(baseline.to_dict("records"))
    bl["rule"] = "BASELINE"
    print(f"Baseline: {bl['total_trades']} trades, WR={bl['win_rate']}%")

    stocks = baseline["code"].unique().tolist()

    # Precompute all rules
    t0 = time.time()
    rule_map = compute_full_rule_map(stocks)
    vol_map = compute_volume_map(stocks)
    div_map = compute_divergence_map(stocks)
    print(f"预计算完成: {time.time()-t0:.0f}s")

    # Merge volume and divergence into rule_map
    for (code, dt), rules in rule_map.items():
        rules["L2-06_成交量"] = vol_map.get((code, dt), False)
        rules["L2-12_背驰"] = div_map.get(code, False)

    # Build per-trade rule matrix
    print("Building trade rule matrix...", flush=True)
    rule_names = ["L2-02_买点类型","L2-03_新鲜度","L2-01_笔数","L2-06_成交量","L2-09_MACD金叉","L2-10_MA多头","L2-12_背驰"]
    # For each trade, store which rules pass
    trade_rules = {}  # trade_index -> set of rule names that pass
    for idx, t in baseline.iterrows():
        code, sd = t["code"], t["signal_date"]
        rules = rule_map.get((code, sd), {})
        passing = {rn for rn in rule_names if rules.get(rn, False)}
        if passing:
            trade_rules[idx] = passing
    print(f"  {len(trade_rules)} trades with rule data", flush=True)

    # Define combinations
    P0 = ["L2-02_买点类型","L2-03_新鲜度"]
    P1_all = ["L2-01_笔数","L2-06_成交量","L2-09_MACD金叉","L2-10_MA多头","L2-12_背驰"]

    combos = [("BASELINE", [])]
    combos.append(("P0 Only", P0))
    for rn in P1_all:
        combos.append((f"P0+{rn.split('_')[0]}", P0 + [rn]))
    combos.append(("ALL", P0 + P1_all))
    # Ablation: remove one at a time from ALL
    for rn in P1_all:
        removed = [r for r in P1_all if r != rn]
        combos.append((f"ALL-{rn.split('_')[0]}", P0 + removed))

    results = []
    for label, required in combos:
        t0 = time.time()
        req_set = set(required)
        if not req_set:
            # BASELINE
            m = bl.copy(); m["rule"] = label
        else:
            filtered = []
            for idx, passing in trade_rules.items():
                if req_set.issubset(passing):
                    filtered.append(baseline.iloc[idx].to_dict())
            m = compute_metrics(filtered)
            m["rule"] = label
        dt = time.time()-t0
        print(f"  {label:<25} Trades={m['total_trades']:>6} WR={m['win_rate']:>5.1f}% ΔWR={m['win_rate']-bl['win_rate']:>+5.1f}% ({dt:.1f}s)")
        results.append(m)

    # Summary
    print(f"\n{'Rule':<25} {'Trades':>6} {'WR%':>6} {'Ret%':>7} {'Sharpe':>7} {'ΔWR':>6}")
    print("-"*80)
    for r in results:
        wr=r.get("win_rate",0); dw=wr-bl["win_rate"]
        print(f'{r["rule"]:<25} {r.get("total_trades",0):>6} {wr:>6.1f} {r.get("avg_return",0):>7.2f} {r.get("sharpe",0):>7.2f} {dw:>+6.1f}')

    json.dump(results, open(OUT/"phase3_results.json","w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT/'phase3_results.json'}")


if __name__ == "__main__":
    main()
