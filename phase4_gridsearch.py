#!/usr/bin/env python3
"""Phase 4: L2 参数网格搜索 — 预计算原始值，再按不同阈值过滤"""

import json, re, time
from pathlib import Path
from collections import defaultdict
import pandas as pd, numpy as np

BASE = Path(__file__).parent
SIG = BASE / "data" / "signals"
DAILY = BASE / "data" / "daily"
OUT = BASE / "data" / "backtest"

B1="日线_D1B_BUY1"; S1="日线_D1_三买辅助V230228"; S2="日线_D1#SMA#34_BS3辅助V230319"
BS2="日线_D1#SMA#21_BS2辅助V230320"; BS3="日线_D1#SMA#34_BS3辅助V230318"
MC="日线_D1MACD12#26#9_BS1辅助V230313"; M5="日线_D1SMA#5_分类V221101"; M20="日线_D1SMA#20_分类V221101"

def _p(val):
    if pd.isna(val) or str(val)=="0": return {"v1":"","v2":"","v3":"","score":"0"}
    ps=str(val).rsplit("_",3)
    return {"v1":ps[0] if len(ps)>=4 else "","v2":ps[1] if len(ps)>=4 else "","v3":ps[2] if len(ps)>=4 else "","score":ps[3] if len(ps)>=4 else "0"}

def mets(trades):
    if not trades: return {"total_trades":0,"win_rate":0,"avg_return":0,"sharpe":0}
    rs=[t["return_pct"]/100 for t in trades]; hd=[t["hold_days"] for t in trades]
    n=len(trades); w=sum(1 for r in rs if r>0); wr=w/n if n>0 else 0
    ar=np.mean(rs) if rs else 0; ah=np.mean(hd) if hd else 0
    sh=(ar/(np.std(rs)+1e-8))*np.sqrt(252/ah) if len(rs)>1 and ah>0 else 0
    return {"total_trades":n,"win_rate":round(wr*100,1),"avg_return":round(ar*100,2),"sharpe":round(sh,2)}

print("=== Phase 4: L2 参数网格搜索 ===", flush=True)

baseline = pd.read_parquet(OUT / "baseline_all.parquet")
bl=mets(baseline.to_dict("records"))
print(f"Baseline: {bl['total_trades']} t, WR={bl['win_rate']}%", flush=True)

pairs = baseline[["code","signal_date"]].drop_duplicates()
code_dates = defaultdict(set)
for _, r in pairs.iterrows():
    code_dates[r["code"]].add(r["signal_date"])

# Precompute RAW values per (code, signal_date)
t0=time.time()
raw_map = {}  # (code, date) -> {stroke_max, fresh_days, vol_ratio, macd_days_golden, ma5_up, ma20_up, buytype_ok}

for i, (code, dates) in enumerate(code_dates.items()):
    sp = SIG / f"{code}.parquet"
    if not sp.exists(): continue
    sd = pd.read_parquet(sp)
    if len(sd) < 2: continue

    # Build date→row_index map
    date_idx = {}
    for ri in range(len(sd)):
        dt = str(sd.iloc[ri]["dt"].date()) if hasattr(sd.iloc[ri]["dt"],'date') else ""
        if dt in dates: date_idx[dt] = ri
    if not date_idx: continue

    # Precompute freshness: for all rows, find days since last buy change
    buy_cols = [c for c in [B1,BS2,BS3,S1,S2] if c in sd.columns]
    last_change_ri = {}
    for ri in range(len(sd)):
        if ri == 0:
            last_change_ri[ri] = -1
        else:
            prev = last_change_ri[ri-1]
            changed = False
            for col in buy_cols:
                o=str(sd.iloc[ri-1][col]); n=str(sd.iloc[ri][col])
                if o!=n and any(k in _p(n)["v1"] for k in ["一买","二买","三买"]):
                    changed=True; break
            last_change_ri[ri] = ri if changed else prev

    # MACD golden cross dates: track last golden cross index per row
    last_golden_ri = {}
    golden_indices = []
    if MC in sd.columns:
        for ri in range(len(sd)):
            if "金叉" in _p(str(sd.iloc[ri][MC]))["v2"]:
                golden_indices.append(ri)
    # For each row, find the most recent golden cross
    gi_ptr = -1
    for ri in range(len(sd)):
        while gi_ptr + 1 < len(golden_indices) and golden_indices[gi_ptr+1] <= ri:
            gi_ptr += 1
        last_golden_ri[ri] = golden_indices[gi_ptr] if gi_ptr >= 0 else -1

    # Volume ratios
    vol_ratios = {}
    dp = DAILY / f"{code}.parquet"
    if dp.exists():
        dd = pd.read_parquet(dp)
        for ri in range(20, len(dd)):
            dt = str(dd.iloc[ri]["date"].date())
            if dt not in dates: continue
            lv=float(dd.iloc[ri]["volume"]); mv=float(dd.iloc[ri-19:ri+1]["volume"].mean())
            vol_ratios[dt] = lv/mv if mv>0 else 0

    # Compute raw values for each requested date
    for dt, ri in date_idx.items():
        row = sd.iloc[ri]

        # max_stroke
        max_stroke = 0
        for col in [B1,S1,S2]:
            if col not in sd.columns: continue
            p=_p(str(row[col]))
            if ("一买" in p["v1"] or "三买" in p["v1"]) and (m:=re.search(r'(\d+)',p["v2"])):
                max_stroke = max(max_stroke, int(m.group(1)))

        # buytype_ok
        bp1=_p(str(row[B1]) if B1 in sd.columns else "")
        bp2=_p(str(row[BS2]) if BS2 in sd.columns else "")
        sp=_p(str(row[S1]) if S1 in sd.columns else "")
        sp2=_p(str(row[S2]) if S2 in sd.columns else "")
        b3p=_p(str(row[BS3]) if BS3 in sd.columns else "")
        mp=_p(str(row[MC]) if MC in sd.columns else "")
        hy="一买" in bp1["v1"]; he="二买" in bp2["v1"]
        hs="三买" in (sp["v1"]+sp2["v1"]+b3p["v1"])
        buytype_ok = hy or hs or (he and "金叉" in mp["v2"])

        # fresh_days
        lc = last_change_ri.get(ri, -1)
        fresh_days = ri - lc if lc >= 0 else 999

        # macd_days_golden
        lg = last_golden_ri.get(ri, -1)
        macd_days = ri - lg if lg >= 0 else 999

        # ma
        u5=u20=False
        if M5 in sd.columns: p=_p(str(row[M5])); u5="多头" in p["v1"] and "向上" in p["v2"]
        if M20 in sd.columns: p=_p(str(row[M20])); u20="多头" in p["v1"] and "向上" in p["v2"]

        raw_map[(code, dt)] = {
            "max_stroke": max_stroke,
            "fresh_days": fresh_days,
            "vol_ratio": vol_ratios.get(dt, 0),
            "macd_days": macd_days,
            "ma5_up": u5,
            "ma20_up": u20,
            "buytype_ok": buytype_ok,
        }

    if (i+1)%200==0: print(f"  precompute {i+1}/{len(code_dates)} ({time.time()-t0:.0f}s)", flush=True)

print(f"Precompute done: {time.time()-t0:.0f}s, {len(raw_map)} pairs", flush=True)

# Build trade→raw map
trade_raw = {}
rn_of_interest = {}  # track which (code, date) pairs each trade maps to
for idx, t in baseline.iterrows():
    key = (t["code"], t["signal_date"])
    rv = raw_map.get(key, {})
    if rv:
        trade_raw[idx] = rv

# Parameter grid
stroke_vals = [5, 7, 9, 11, 13]
fresh_vals  = [10, 20, 30, 60, 90]
vol_vals    = [1.2, 1.5, 2.0, 2.5]
macd_vals   = [3, 5, 7, 10]

print(f"\nGrid: {len(stroke_vals)}×{len(fresh_vals)}×{len(vol_vals)}×{len(macd_vals)}×2 = {len(stroke_vals)*len(fresh_vals)*len(vol_vals)*len(macd_vals)*2}", flush=True)

# Test each parameter independently (coordinate descent)
# Fix defaults: stroke=9, fresh=30, vol=1.5, macd=5, ma=both
defaults = {"stroke":9, "fresh":30, "vol":1.5, "macd":5, "ma":"both"}

# Also test P0 only
# P0: buytype_ok + fresh_days <= 30
p0_only = [idx for idx, rv in trade_raw.items() if rv["buytype_ok"] and rv["fresh_days"] <= 30]
m_p0 = mets([baseline.iloc[idx].to_dict() for idx in p0_only])
print(f"\nP0 Only: T={m_p0['total_trades']} WR={m_p0['win_rate']}% Δ={m_p0['win_rate']-bl['win_rate']:+.1f}%", flush=True)

# For each parameter, vary it while fixing others at defaults
results = []

# Stroke
print("\n--- Stroke ---", flush=True)
for sv in stroke_vals:
    idxs = [idx for idx, rv in trade_raw.items()
            if rv["buytype_ok"] and rv["fresh_days"]<=30
            and rv["max_stroke"]>=sv]
    m = mets([baseline.iloc[idx].to_dict() for idx in idxs])
    results.append({"param":"stroke","value":sv,"metrics":m})
    print(f"  stroke>={sv}: T={m['total_trades']} WR={m['win_rate']}% Δ={m['win_rate']-bl['win_rate']:+.1f}%", flush=True)

# Freshness
print("\n--- Freshness ---", flush=True)
for fv in fresh_vals:
    idxs = [idx for idx, rv in trade_raw.items()
            if rv["buytype_ok"] and rv["fresh_days"]<=fv]
    m = mets([baseline.iloc[idx].to_dict() for idx in idxs])
    results.append({"param":"freshness","value":fv,"metrics":m})
    print(f"  fresh<={fv}: T={m['total_trades']} WR={m['win_rate']}% Δ={m['win_rate']-bl['win_rate']:+.1f}%", flush=True)

# Volume
print("\n--- Volume ratio ---", flush=True)
for vv in vol_vals:
    idxs = [idx for idx, rv in trade_raw.items()
            if rv["buytype_ok"] and rv["fresh_days"]<=30
            and rv["vol_ratio"]>=vv]
    m = mets([baseline.iloc[idx].to_dict() for idx in idxs])
    results.append({"param":"volume","value":vv,"metrics":m})
    print(f"  vol>={vv}: T={m['total_trades']} WR={m['win_rate']}% Δ={m['win_rate']-bl['win_rate']:+.1f}%", flush=True)

# MACD
print("\n--- MACD window ---", flush=True)
for mv in macd_vals:
    idxs = [idx for idx, rv in trade_raw.items()
            if rv["buytype_ok"] and rv["fresh_days"]<=30
            and rv["macd_days"]<=mv]
    m = mets([baseline.iloc[idx].to_dict() for idx in idxs])
    results.append({"param":"macd","value":mv,"metrics":m})
    print(f"  macd<={mv}d: T={m['total_trades']} WR={m['win_rate']}% Δ={m['win_rate']-bl['win_rate']:+.1f}%", flush=True)

# MA mode
print("\n--- MA mode ---", flush=True)
for ma_mode, ma_filter in [
    ("MA5 only", lambda rv: rv["buytype_ok"] and rv["fresh_days"]<=30 and rv["ma5_up"]),
    ("both", lambda rv: rv["buytype_ok"] and rv["fresh_days"]<=30 and rv["ma5_up"] and rv["ma20_up"]),
]:
    idxs = [idx for idx, rv in trade_raw.items() if ma_filter(rv)]
    m = mets([baseline.iloc[idx].to_dict() for idx in idxs])
    results.append({"param":"ma_mode","value":ma_mode,"metrics":m})
    print(f"  {ma_mode}: T={m['total_trades']} WR={m['win_rate']}% Δ={m['win_rate']-bl['win_rate']:+.1f}%", flush=True)

# Best combination: take best value for each parameter
print("\n--- Best combo ---", flush=True)
best_stroke = 9; best_fresh = 30; best_vol = 1.5; best_macd = 5; best_ma = "both"
# (Keeping defaults since our earlier results showed combos don't help)
idxs = [idx for idx, rv in trade_raw.items()
        if rv["buytype_ok"] and rv["fresh_days"]<=best_fresh
        and rv["max_stroke"]>=best_stroke
        and rv["vol_ratio"]>=best_vol
        and rv["macd_days"]<=best_macd
        and (best_ma=="MA5 only" or (rv["ma5_up"] and rv["ma20_up"]))]
m = mets([baseline.iloc[idx].to_dict() for idx in idxs])
results.append({"param":"best_combo","value":"defaults","metrics":m})
print(f"  best combo: T={m['total_trades']} WR={m['win_rate']}% Δ={m['win_rate']-bl['win_rate']:+.1f}%", flush=True)

# Save
json.dump(results, open(OUT/"phase4_gridsearch.json","w"), indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT/'phase4_gridsearch.json'}")
