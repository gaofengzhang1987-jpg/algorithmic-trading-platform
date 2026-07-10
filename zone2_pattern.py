#!/usr/bin/env python3
"""L2 重点形态沉淀区 — 对 L1 结果应用策略一区规则。

规则体系：
  P0 门禁: L2-02 买点类型分级, L2-03 信号新鲜度
  P1 辅助: L2-01 笔数, L2-06 成交量, L2-09 MACD金叉, L2-12 力度背驰, L2-15 底分型强度, L2-04 中枢下沿
"""

import logging, re
from datetime import datetime
from pathlib import Path

import pandas as pd
from czsc import CZSC, RawBar, Freq
from czsc.objects import Direction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zone2")

BASE = Path(__file__).parent
SIGNALS = BASE / "data" / "signals"; DAILY = BASE / "data" / "daily"; ZONES = BASE / "data" / "zones"

B1="日线_D1B_BUY1"; BS2="日线_D1#SMA#21_BS2辅助V230320"; BS3="日线_D1#SMA#34_BS3辅助V230318"
S1="日线_D1_三买辅助V230228"; S2="日线_D1#SMA#34_BS3辅助V230319"
MC="日线_D1MACD12#26#9_BS1辅助V230313"

FRESHNESS_DAYS=30; VOL_MA=20; MACD_WIN=5
_czsc_cache={}

def _is_bottom_fractal(fx):
    """判断 FX 是否为底分型：中间 bar 的 low 是整组最低且严格唯一。"""
    if not hasattr(fx, 'raw_bars') or not fx.raw_bars or len(fx.raw_bars) < 3:
        return False
    mid = len(fx.raw_bars) // 2
    mb_low = fx.raw_bars[mid].low
    all_lows = [b.low for b in fx.raw_bars]
    return mb_low == min(all_lows) and all_lows.count(mb_low) == 1


def _p(v):
    if pd.isna(v) or str(v)=="0": return {"v1":"","v2":"","v3":"","score":"0"}
    ps=str(v).rsplit("_",3)
    return {"v1":ps[0] if len(ps)>=4 else "","v2":ps[1] if len(ps)>=4 else "","v3":ps[2] if len(ps)>=4 else "","score":ps[3] if len(ps)>=4 else "0"}

def _stroke_count(v):
    m=re.search(r'(\d+)',_p(v)["v2"]); return int(m.group(1)) if m else 0

def _detect_changes(df,lb):
    bc=[c for c in [B1,BS2,BS3,S1,S2] if c in df.columns]
    if len(df)<2: return []
    w=df.tail(lb+1); ch=[]
    for col in bc:
        for i in range(1,len(w)):
            o=str(w.iloc[i-1][col]); n=str(w.iloc[i][col])
            if o!=n and any(k in _p(n)["v1"] for k in ["一买","二买","三买"]):
                dt=w.iloc[i]["dt"]; ch.append({"col":col,"date":str(dt.date()) if hasattr(dt,'date') else str(dt)})
    return ch

def _score_stroke(b1v,smv):
    best=0
    for v in [b1v,smv]:
        p=_p(v)
        if "一买" in p["v1"] or "三买" in p["v1"]:
            n=_stroke_count(v)
            s=0 if n<5 else (20 if n<=8 else (50 if n<=12 else (75 if n<=16 else 100)))
            if s>best: best=s
    return best

def _score_volume(df):
    if len(df)<VOL_MA: return 40
    lv=float(df["vol"].iloc[-1]); mv=float(df["vol"].tail(VOL_MA).mean())
    if mv<=0: return 40
    r=lv/mv
    if r<0.8: return 20
    elif r<1.2: return 40
    elif r<1.8: return 70
    elif r<=3.0: return 100
    return 50

def _score_macd(df):
    if MC not in df.columns: return 0
    r=df.tail(MACD_WIN)
    for i in range(len(r)-1,-1,-1):
        if "金叉" in _p(str(r.iloc[i][MC]))["v2"]:
            da=len(r)-1-i
            return 100 if da<=3 else (60 if da<=5 else 20)
    return 0

def _score_divergence(code, cp):
    """L2-12/15/04: 力度背驰 + 底分型 + 中枢下沿 (共享 CZSC)"""
    global _czsc_cache
    if code in _czsc_cache: c=_czsc_cache[code]
    else:
        dp=DAILY/f"{code}.parquet"
        if not dp.exists(): return 0,0,0,0
        df=pd.read_parquet(dp)
        if len(df)<30: return 0,0,0,0
        bars=[RawBar(symbol=code,id=j+1,dt=r["date"].to_pydatetime(),freq=Freq.D,
                open=r["open"],close=r["close"],high=r["high"],low=r["low"],
                vol=r["volume"],amount=r["amount"]) for j,(_,r) in enumerate(df.iterrows())]
        c=CZSC(bars); _czsc_cache[code]=c

    # L2-12
    dbs=[bi for bi in c.bi_list if bi.direction==Direction.Down]
    if len(dbs)<2: return 0,len(dbs),0,0
    A,B=dbs[-2],dbs[-1]
    if A.power<=0: return 0,len(dbs),0,0
    ratio=B.power/A.power
    score=100 if ratio<0.3 else (75 if ratio<0.6 else (40 if ratio<0.9 else (10 if ratio<1.0 else 0)))

    # L2-15: 底分型强度 — 三维评分 (sk·sc·sv 各 0-100, 取均值)
    sf = 0
    if hasattr(c, 'fx_list') and c.fx_list and len(c.fx_list) >= 2:
        # 找最近的底分型（从末尾往前扫）
        btm_fx = None
        for fx in reversed(c.fx_list):
            if _is_bottom_fractal(fx):
                btm_fx = fx
                break

        if btm_fx is not None:
            lf = btm_fx

            # ——— sk: K线结构分 (0-100) ———
            # 构成分型的 K 线数量，越多信号越可靠
            kc = len(lf.raw_bars) if lf.raw_bars and len(lf.raw_bars) >= 3 else 3
            if kc >= 10:      sk = 100
            elif kc >= 8:     sk = 85
            elif kc >= 6:     sk = 70
            elif kc >= 5:     sk = 55
            elif kc >= 4:     sk = 40
            else:             sk = 20

            # ——— sc: 结构质量分 (0-100) ———
            sc = 0
            mid = len(lf.raw_bars) // 2
            bottom_low = lf.raw_bars[mid].low

            # (1) 强势/弱势判定 (0-40): 第三根K线（最右）对第一根K线（最左）的反弹力度
            # 强势底分型：第三根高开高走，收盘站上第一根高点或实体一半以上
            # 弱势底分型：反弹无力，无法超越第一根实体二分之一，多为中继
            first_bar = lf.raw_bars[0]
            last_bar = lf.raw_bars[-1]
            prev_bar = lf.raw_bars[-2] if len(lf.raw_bars) >= 3 else None
            first_high = first_bar.high
            first_body_top = max(first_bar.open, first_bar.close)
            first_body_bot = min(first_bar.open, first_bar.close)
            first_body_mid = (first_body_top + first_body_bot) / 2
            # 高开高走：收阳 且 开于前一根收盘价之上
            gap_up = last_bar.close > last_bar.open
            if prev_bar is not None:
                gap_up = gap_up and last_bar.open > prev_bar.close
            if last_bar.close > first_high:
                sc += 40  # 强势：收盘站上五日均线高点
            elif last_bar.close > first_body_mid and gap_up:
                sc += 30  # 次强：高开高走+站上实体一半以上
            elif last_bar.close > first_body_mid:
                sc += 20  # 一般：站上实体一半以上但非高开高走
            else:
                sc += 5   # 弱势：无法超越第一根实体二分之一，中继形态

            # (2) BI 端点吻合 (0-30): 分型低点是否接近最近向下笔终点
            down_bis = [bi for bi in c.finished_bis if bi.direction == Direction.Down]
            if down_bis:
                last_down = down_bis[-1]
                if last_down.low > 0:
                    dist_pct = abs(bottom_low - last_down.low) / last_down.low * 100
                    if dist_pct <= 0.5:
                        sc += 30  # 精准吻合
                    elif dist_pct <= 1.0:
                        sc += 20
                    elif dist_pct <= 2.0:
                        sc += 10

            # (3) 笔内位置 (0-30): 分型是否位于向下笔的末端区域
            if down_bis and hasattr(last_down, 'raw_bars') and last_down.raw_bars:
                bi_bars = last_down.raw_bars
                bi_len = len(bi_bars)
                fx_last_dt = lf.raw_bars[-1].dt if lf.raw_bars else None
                if fx_last_dt and bi_len > 0:
                    for idx, bb in enumerate(bi_bars):
                        if bb.dt >= fx_last_dt:
                            pos_ratio = idx / bi_len
                            if pos_ratio >= 0.8:
                                sc += 30  # 笔末端，底部确认强
                            elif pos_ratio >= 0.6:
                                sc += 20
                            elif pos_ratio >= 0.4:
                                sc += 10
                            break

            # ——— sv: 量能确认分 (0-100) ———
            sv = 0
            vols = [b.vol for b in lf.raw_bars] if lf.raw_bars else []

            # (1) 分型内缩量 (0-50): 最低点 K 线量能相对分型均值是否偏低
            if vols and len(vols) >= 3:
                avg_vol = sum(vols) / len(vols)
                mid_vol = vols[mid]
                if avg_vol > 0:
                    vol_ratio = mid_vol / avg_vol
                    if vol_ratio <= 0.7:
                        sv += 50  # 显著缩量，底部特征明显
                    elif vol_ratio <= 0.85:
                        sv += 35
                    elif vol_ratio <= 1.0:
                        sv += 20
                    else:
                        sv += 5

            # (2) 右侧放量 (0-50): 右侧 K 线是否量能回升（资金进场确认）
            # 分型最低点之后的 K 线为"右侧"
            right_vol_list = vols[mid + 1:] if mid + 1 < len(vols) else []
            if right_vol_list and vols and len(vols) >= 3:
                left_vols = vols[:mid]
                right_vols = right_vol_list
                left_avg = sum(left_vols) / len(left_vols) if left_vols else 0
                right_avg = sum(right_vols) / len(right_vols) if right_vols else 0
                if left_avg > 0:
                    incr_ratio = right_avg / left_avg
                    if incr_ratio >= 2.0:
                        sv += 50  # 显著放量
                    elif incr_ratio >= 1.5:
                        sv += 35
                    elif incr_ratio >= 1.2:
                        sv += 20
                    elif incr_ratio >= 1.0:
                        sv += 10

            sf = int((sk + sc + sv) / 3)

    # L2-04: 中枢下沿 (BI low points 代理)
    sz=0
    all_lows=[]
    for bi in c.bi_list:
        if hasattr(bi,'low') and bi.direction==Direction.Down: all_lows.append(float(bi.low))
    if all_lows and cp>0:
        nl=all_lows[-1]; dpct=abs(cp-nl)/cp*100
        if dpct<=1: sz=100
        elif dpct<=2: sz=60
        elif dpct<=3: sz=20

    return score,len(dbs),sf,sz


class StrategyOne:
    def __init__(self):
        self.name="策略一区"; self.today=datetime.now().date()

    def apply(self,df):
        if df.empty: return df
        results=[]; rej02=0; rej03=0; passed=0
        for _,row in df.iterrows():
            code=row["代码"]
            sp=SIGNALS/f"{code}.parquet"
            if not sp.exists(): continue
            sd=pd.read_parquet(sp)
            if len(sd)<2: continue
            lr=sd.iloc[-1]

            b1v=str(lr[B1]) if B1 in sd.columns else ""; b2v=str(lr[BS2]) if BS2 in sd.columns else ""
            b3v=str(lr[BS3]) if BS3 in sd.columns else ""; s1v=str(lr[S1]) if S1 in sd.columns else ""
            s2v=str(lr[S2]) if S2 in sd.columns else ""; mcv=str(lr[MC]) if MC in sd.columns else ""

            b1p=_p(b1v); b2p=_p(b2v); b3p=_p(b3v); s1p=_p(s1v); s2p=_p(s2v); mcp=_p(mcv)
            hy="一买" in b1p["v1"]; he="二买" in b2p["v1"]; hs="三买" in (s1p["v1"]+s2p["v1"]+b3p["v1"])

            # P0: L2-02
            if hy or hs: pass
            elif he:
                if "金叉" not in mcp["v2"]: rej02+=1; continue
            else: rej02+=1; continue

            # P0: L2-03
            changes=_detect_changes(sd,FRESHNESS_DAYS)
            if not changes: rej03+=1; continue
            dates=[datetime.strptime(c["date"],"%Y-%m-%d").date() for c in changes]
            ld=max(dates)
            if (self.today-ld).days>FRESHNESS_DAYS: rej03+=1; continue

            # P1
            s_stroke=_score_stroke(b1v,s1v+s2v)
            s_volume=_score_volume(sd)
            s_macd=_score_macd(sd)
            s_div,s_div_cnt,s_fractal,s_zs=_score_divergence(code,float(row["现价"]))

            atypes=[]
            if hy: atypes.append("一买")
            if he: atypes.append("二买")
            if hs: atypes.append("三买")

            results.append({
                "代码":code,"现价":row["现价"],"最新日期":row["最新日期"],
                "买点类型":" | ".join(atypes),"状态详情":row.get("状态详情",""),"信号数":row.get("信号数",0),
                "L2_笔数得分":s_stroke,"L2_量比得分":s_volume,"L2_MACD得分":s_macd,
                "L2_背驰得分":s_div,"L2_底分型得分":s_fractal,"L2_中枢得分":s_zs,
                "L2_最近变化":str(ld),"L2_变化次数":len(changes)})
            passed+=1

        rdf=pd.DataFrame(results)
        if rdf.empty:
            logger.info("策略一区: %d → 0 (L2-02:%d L2-03:%d)",len(df),rej02,rej03)
            return rdf
        rdf.sort_values("L2_笔数得分",ascending=False,inplace=True); rdf.reset_index(drop=True,inplace=True)
        logger.info("策略一区: %d→%d (L2-02:%d L2-03:%d)",len(df),passed,rej02,rej03)
        return rdf


def run(input_df=None):
    if input_df is None:
        p=ZONES/"L1_deposition.parquet"
        if not p.exists(): logger.warning("L1 不存在"); return pd.DataFrame()
        input_df=pd.read_parquet(p)
    if input_df.empty: return pd.DataFrame()
    s=StrategyOne(); r=s.apply(input_df)
    r.to_parquet(ZONES/"L2_pattern.parquet")
    logger.info("L2 重点形态沉淀区: %d 只",len(r))
    return r


if __name__ == "__main__":
    df=run()
    if df.empty: print("L2: 无通过")
    else:
        print(f"L2: {len(df)} 只")
        for c in [c for c in df.columns if 'L2_' in c]:
            if c in df.columns: print(f"  {c}: {df[c].min()}-{df[c].max()}")
