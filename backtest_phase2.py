#!/usr/bin/env python3
"""Phase 2: L2 单规则贡献回测。"""

import json, logging, os, re, time
from pathlib import Path
import numpy as np, pandas as pd

from backtest import (
    load_daily, _parse_signal, _detect_all_changes,
    get_next_trading_day, get_price_at_date, compute_metrics,
    SIGNALS_DIR, COMMISSION, CAPITAL_PER_TRADE, MAX_HOLD_DAYS, STOP_LOSS_PCT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase2")

BASE, OUT = Path(__file__).parent, Path(__file__).parent / "data" / "backtest"
OUT.mkdir(parents=True, exist_ok=True)

# 信号列常量
B1="日线_D1B_BUY1"; B2="日线_D1#SMA#21_BS2辅助V230320"; B3="日线_D1#SMA#34_BS3辅助V230318"
S1="日线_D1_三买辅助V230228"; S2="日线_D1#SMA#34_BS3辅助V230319"
MC="日线_D1MACD12#26#9_BS1辅助V230313"; M5="日线_D1SMA#5_分类V221101"; M20="日线_D1SMA#20_分类V221101"


def _get_stroke(code, sig_df, buy):
    vals = [str(sig_df.iloc[buy["idx"]][c]) if c in sig_df.columns else "" for c in [B1,S1,S2]]
    for v in vals:
        p = _parse_signal(v)
        if ("一买" in p["v1"] or "三买" in p["v1"]) and (m := re.search(r'(\d+)', p["v2"])):
            if int(m.group(1)) >= 9: return True
    return False


def _get_volume(code, sig_df, buy):
    dp = BASE / "data" / "daily" / f"{code}.parquet"
    if not dp.exists(): return False
    df = pd.read_parquet(dp)
    if len(df) < 20: return False
    return (float(df["volume"].iloc[-1]) / float(df["volume"].tail(20).mean())) >= 1.5


def _get_macd(code, sig_df, buy):
    if MC not in sig_df.columns: return False
    for i in range(min(5, len(sig_df))):
        if "金叉" in _parse_signal(str(sig_df.iloc[-(i+1)][MC]))["v2"]: return True
    return False


def _get_ma(code, sig_df, buy):
    u5 = u20 = False
    if M5 in sig_df.columns:
        p = _parse_signal(str(sig_df[M5].iloc[-1])); u5 = "多头" in p["v1"] and "向上" in p["v2"]
    if M20 in sig_df.columns:
        p = _parse_signal(str(sig_df[M20].iloc[-1])); u20 = "多头" in p["v1"] and "向上" in p["v2"]
    return u5 and u20


def _get_divergence(code, sig_df, buy):
    from czsc import CZSC, RawBar, Freq
    from czsc.objects import Direction
    dp = BASE / "data" / "daily" / f"{code}.parquet"
    if not dp.exists(): return False
    df = pd.read_parquet(dp)
    if len(df) < 30: return False
    bars = [RawBar(symbol=code, id=i+1, dt=r["date"].to_pydatetime(), freq=Freq.D,
            open=r["open"],close=r["close"],high=r["high"],low=r["low"],vol=r["volume"],amount=r["amount"])
            for i,(_,r) in enumerate(df.iterrows())]
    c = CZSC(bars)
    dbs = [bi for bi in c.bi_list if bi.direction == Direction.Down]
    if len(dbs) < 2 or dbs[-2].power <= 0: return False
    return dbs[-1].power < dbs[-2].power


RULES = {
    "L2-01_笔数": _get_stroke,
    "L2-06_成交量": _get_volume,
    "L2-09_MACD金叉": _get_macd,
    "L2-10_MA多头": _get_ma,
    "L2-12_力度背驰": _get_divergence,
}


def simulate_filtered(code, rule_fn=None):
    daily = load_daily(code)
    if daily is None: return []
    sp = SIGNALS_DIR / f"{code}.parquet"
    if not sp.exists(): return []
    sig_df = pd.read_parquet(sp)
    changes = _detect_all_changes(sig_df)
    if not changes: return []
    buys = [c for c in changes if c["type"]=="buy"]; sells = [c for c in changes if c["type"]=="sell"]
    if not buys: return []
    if rule_fn: buys = [b for b in buys if rule_fn(code, sig_df, b)]
    if not buys: return []
    ds = daily.sort_values("date").reset_index(drop=True)
    trades = []
    for buy in buys:
        ed = get_next_trading_day(buy["date"], ds)
        if ed is None: continue
        ep = get_price_at_date(ed, ds)
        if ep is None or ep <= 0: continue
        xd, xp, xr = None, None, ""
        for s in sells:
            if pd.Timestamp(s["date"]) > ed:
                xc = get_next_trading_day(s["date"], ds)
                if xc: xd, xr = xc, f"卖点:{s['signal_label']}"
                break
        ei = ds[ds["date"]==ed].index
        if ei.empty: continue
        ei=ei[0]; sl=ep*(1+STOP_LOSS_PCT); end=min(ei+MAX_HOLD_DAYS+1,len(ds))
        w=ds.iloc[ei+1:end]
        if len(w)==0: continue
        sh=np.where(w["low"].values<=sl)[0]
        if len(sh)>0: xd,xp,xr=w.iloc[sh[0]]["date"],sl,f"止损(-5%)"
        elif xd: xd=w[w["date"]>=xd].iloc[0]["date"] if len(w[w["date"]>=xd])>0 else None
        else:
            li=min(ei+MAX_HOLD_DAYS,len(ds)-1); xd=ds.iloc[li]["date"]; xp=float(ds.iloc[li]["open"]); xr=f"到期({MAX_HOLD_DAYS}天)"
        if xp is None and xd: xp=get_price_at_date(xd,ds)
        if xp is None or xp<=0: continue
        gr=(xp-ep)/ep; nr=gr-2*COMMISSION; hd=(xd-ed).days
        trades.append({"code":code,"signal_type":buy["signal_label"],"signal_date":buy["date"],
                       "entry_date":str(ed.date()),"exit_date":str(xd.date()),"entry_price":round(ep,2),
                       "exit_price":round(xp,2),"return_pct":round(nr*100,2),"hold_days":hd,"exit_reason":xr})
    return trades


def main():
    # Baseline
    bp = OUT / "baseline_metrics.json"
    bl = json.loads(bp.read_text()) if bp.exists() else {}
    bl["rule"] = "BASELINE"
    print(f"Baseline: WR={bl['win_rate']}% Ret={bl['avg_return']}% Sharpe={bl['sharpe']} Trades={bl['total_trades']}")

    l1 = pd.read_parquet(BASE / "data" / "zones" / "L1_deposition.parquet")
    codes = l1["代码"].tolist()

    import sys
    if len(sys.argv) > 1: codes = codes[:int(sys.argv[1])]
    logger.info("测试 %d 只股票", len(codes))

    results = [bl]
    for name, fn in RULES.items():
        logger.info("="*50); logger.info("规则: %s", name)
        t0=time.time(); trades=[]
        for i, c in enumerate(codes):
            trades.extend(simulate_filtered(c, fn))
            if (i+1)%200==0: logger.info("  %s %d/%d = %d 笔", name, i+1, len(codes), len(trades))
        dt=time.time()-t0
        if trades:
            df=pd.DataFrame(trades); m=compute_metrics(df.to_dict("records")); m["rule"]=name
        else: m={"rule":name,"total_trades":0,"win_rate":0,"avg_return":0,"sharpe":0}
        logger.info("  %s: WR=%.1f%% Ret=%.2f%% Sharpe=%.2f Trades=%d (%.0fs)", name, m.get("win_rate",0), m.get("avg_return",0), m.get("sharpe",0), m.get("total_trades",0), dt)
        results.append(m)

    print("\n"+"="*80)
    print(f"{'Rule':<20} {'Trades':>8} {'WR%':>8} {'Ret%':>8} {'Sharpe':>8} {'ΔWR':>8}")
    print("-"*80)
    for r in results:
        n=r["rule"]; wr=r.get("win_rate",0); dw=wr-bl.get("win_rate",0)
        print(f"{n:<20} {r.get('total_trades',0):>8} {wr:>8.1f} {r.get('avg_return',0):>8.2f} {r.get('sharpe',0):>8.2f} {dw:>+8.1f}")
    json.dump(results, open(OUT/"phase2_results.json","w"), indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT/'phase2_results.json'}")

if __name__=="__main__": main()
