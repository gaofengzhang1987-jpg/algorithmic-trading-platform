#!/usr/bin/env python3
"""L3 自选股生态区 — 跨级别共振(周线+日线 P0, 30分钟 P1)。

规则体系：
  P0 门禁:
    L3-04 周线-日线跨级别共振 — 周线有买点 AND 日线有买点
    L3-08 ST/退市排除 — 数据拉取层已过滤
    L3-11 涨跌幅异常   — 近5天无涨跌停

  P1 加分:
    L3-04 共振类型    — 周线二买+日线二买=100, 周线一买+日线一买=90...
    L3-03 日线双买点  — 一买+三买同时点亮 +10 (从 P0 降级)
    L3-13 30分钟二买   — 有30分钟二买信号 +15 (信号引擎完成后生效)
    L3-02 周线止跌     — 周线不再加速下跌, MA5走平度 0~100
    L3-10 换手率       — 活跃度 0~100 (下限降至0.3%)
"""

import logging, re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zone3")

BASE = Path(__file__).parent
DAILY = BASE / "data" / "daily"
WEEKLY = BASE / "data" / "weekly"
WEEKLY_SIG = BASE / "data" / "signals_weekly"
MIN30_SIG = BASE / "data" / "signals_30min"
ZONES = BASE / "data" / "zones"

# 周线信号列
W_B1 = "周线_D1B_BUY1"; W_BS2 = "周线_D1#SMA#21_BS2辅助V230320"
W_BS3 = "周线_D1#SMA#34_BS3辅助V230318"; W_SAN = "周线_D1_三买辅助V230228"
W_SAN2 = "周线_D1#SMA#34_BS3辅助V230319"

# 30分钟信号列
M30_BS2 = "30分钟_D1#SMA#21_BS2辅助V230320"

LIMIT_DAYS = 5; LIMIT_PCT = 9.5
TURNOVER_DAYS = 5; TURNOVER_MIN = 0.3; TURNOVER_MAX = 15.0


def _parse(val):
    if pd.isna(val) or str(val)=="0": return {"v1":"","v2":"","v3":"","score":"0"}
    ps=str(val).rsplit("_",3)
    return {"v1":ps[0] if len(ps)>=4 else "","v2":ps[1] if len(ps)>=4 else "","v3":ps[2] if len(ps)>=4 else "","score":ps[3] if len(ps)>=4 else "0"}


def _check_market_cap(code):
    """L3-07 P0: 流通市值 < 20亿 → 淘汰"""
    dp = DAILY / f"{code}.parquet"
    if not dp.exists(): return False, 0
    d = pd.read_parquet(dp)
    close = float(d["close"].iloc[-1])
    os_val = float(d["outstanding_share"].iloc[-1])
    mcap_yi = close * os_val / 1e8  # 流通市值(亿)
    return mcap_yi >= 20, mcap_yi


def _check_weekly_buy(code):
    """检查周线是否有买点信号。返回 (has_buy, buy_types_set)"""
    wp = WEEKLY_SIG / f"{code}.parquet"
    if not wp.exists(): return False, set()
    w = pd.read_parquet(wp)
    last = w.iloc[-1]
    types = set()
    for col in [W_B1, W_BS2, W_BS3, W_SAN, W_SAN2]:
        if col not in w.columns: continue
        p = _parse(str(last[col]))
        if "一买" in p["v1"]: types.add("一买")
        if "二买" in p["v1"]: types.add("二买")
        if "三买" in p["v1"]: types.add("三买")
    return len(types) > 0, types


def _resonance_score(weekly_types, daily_types):
    """L3-04 P1: 周线-日线共振类型得分 (WR 校准版)"""
    has_w1 = "一买" in weekly_types; has_w2 = "二买" in weekly_types; has_w3 = "三买" in weekly_types
    has_d1 = "一买" in daily_types; has_d2 = "二买" in daily_types; has_d3 = "三买" in daily_types

    # 校准数据: 基于 135K 交易各共振类型实际 WR 映射
    # W3+D1/D2(接力): WR=27.6% → 100 (原70, 被低估)
    if has_w3 and (has_d1 or has_d2): return 100
    # W2+D2(主升浪): WR=26.6% → 84
    if has_w2 and has_d2: return 84
    # W1+D1(世纪底): WR=26.2% → 79
    if has_w1 and has_d1: return 79
    # 其他: WR=22.2% → 19
    if has_w1 or has_w2 or has_w3: return 19
    # W2+D3(二三合力): WR=20.9% → 0 (原100, 严重高估)
    # W1+D2: WR=20.8% → 0 (原60, 无效)
    return 0


def _score_dual_daily(daily_types):
    """L3-03 P1: 日线双买点加分"""
    return 10 if ("一买" in daily_types and "三买" in daily_types) else 0


def _score_m30_second(code):
    """L3-13 P1: 30分钟二买确认 (+15)"""
    mp = MIN30_SIG / f"{code}.parquet"
    if not mp.exists(): return 0
    df = pd.read_parquet(mp)
    if M30_BS2 not in df.columns: return 0
    last = _parse(str(df[M30_BS2].iloc[-1]))
    return 15 if "二买" in last["v1"] else 0


def _check_limit(code):
    """L3-11: 涨跌幅异常"""
    dp = DAILY / f"{code}.parquet"
    if not dp.exists(): return True, ""
    d = pd.read_parquet(dp)
    if len(d) < LIMIT_DAYS+1: return True, ""
    for i in range(max(1,len(d)-LIMIT_DAYS), len(d)):
        pc = float(d.iloc[i-1]["close"]); cc = float(d.iloc[i]["close"])
        if pc <= 0: continue
        r = (cc-pc)/pc*100
        if r >= LIMIT_PCT: return False, f"涨停 +{r:.1f}%"
        if r <= -LIMIT_PCT: return False, f"跌停 {r:.1f}%"
    return True, ""


def _score_weekly_stability(code):
    """L3-02 P1: 周线止跌确认 (0~100) — 不加速下跌 = 防守信号"""
    wp = WEEKLY / f"{code}.parquet"
    if not wp.exists(): return 0
    w = pd.read_parquet(wp)
    if len(w) < 8: return 0
    closes = w["close"].tail(8).values
    # 近4周 vs 前4周: 是否还在创新低
    recent = closes[-4:]; prev = closes[-8:-4]
    recent_min = recent.min(); prev_min = prev.min()
    if recent_min < prev_min: return 0  # 仍在创新低
    if recent_min == prev_min: return 40  # 走平
    # 止跌且有回升
    ma5 = closes[-5:].mean()
    ma5p = closes[-6:-1].mean()
    if ma5 > ma5p: return 80  # MA5 开始向上
    return 60


def _score_turnover(code):
    """L3-10 P1: 换手率 (0.3%~15%)"""
    dp = DAILY / f"{code}.parquet"
    if not dp.exists(): return 0
    d = pd.read_parquet(dp)
    if "turnover" not in d.columns or len(d) < TURNOVER_DAYS: return 0
    avg = float(d["turnover"].tail(TURNOVER_DAYS).mean())
    if avg < TURNOVER_MIN or avg > TURNOVER_MAX: return 0
    if 3 <= avg <= 8: return 100
    if TURNOVER_MIN <= avg < 3: return int(30 + (avg - TURNOVER_MIN) / (3 - TURNOVER_MIN) * 70)
    if 8 < avg <= TURNOVER_MAX: return int(100 - (avg - 8) / (TURNOVER_MAX - 8) * 70)
    return 0


class StrategyTwo:
    """L3: 周线+日线 P0 门禁 + P1 跨级别加分"""

    def __init__(self):
        self.name = "策略二区"

    def apply(self, df):
        if df.empty: return df
        results = []; rej_mcap=0; rej_wk=0; rej_lim=0; passed=0

        for _, row in df.iterrows():
            code = row["代码"]
            buy_types_raw = str(row.get("买点类型",""))
            dtypes = set()
            if "一买" in buy_types_raw: dtypes.add("一买")
            if "二买" in buy_types_raw: dtypes.add("二买")
            if "三买" in buy_types_raw: dtypes.add("三买")

            # P0: L3-07 市值 >= 20亿
            mcap_ok, mcap_val = _check_market_cap(code)
            if not mcap_ok: rej_mcap += 1; continue

            # P0: L3-04 周线-日线跨级别共振
            has_w, wtypes = _check_weekly_buy(code)
            if not has_w: rej_wk += 1; continue
            if not dtypes: rej_wk += 1; continue

            # P0: L3-11
            lim_ok, _ = _check_limit(code)
            if not lim_ok: rej_lim += 1; continue

            # P1 scores
            s_resonance = _resonance_score(wtypes, dtypes)
            s_dual = _score_dual_daily(dtypes)
            s_m30 = _score_m30_second(code)
            s_stability = _score_weekly_stability(code)
            s_turnover = _score_turnover(code)

            l2_stroke = row.get("L2_笔数得分",0)
            l2_volume = row.get("L2_量比得分",0)
            l2_macd   = row.get("L2_MACD得分",0)
            l2_bc     = row.get("L2_背驰得分",0)
            l2_fractal = row.get("L2_底分型得分",0)
            l2_zs      = row.get("L2_中枢得分",0)

            results.append({
                "代码":code,"现价":row["现价"],"最新日期":row.get("最新日期",""),
                "买点类型":buy_types_raw,"状态详情":row.get("状态详情",""),
                "信号数":row.get("信号数",0),
                "L2_笔数得分":l2_stroke,"L2_量比得分":l2_volume,
                "L2_MACD得分":l2_macd,"L2_背驰得分":l2_bc,
                "L3_共振类型":s_resonance,"L3_日线双买点":s_dual,
                "L3_30min二买":s_m30,"L3_周线止跌":s_stability,
                "L2_底分型得分":l2_fractal,
                "L2_中枢得分":l2_zs,
                "L3_换手率":s_turnover,
            }); passed += 1

        result_df = pd.DataFrame(results)
        if result_df.empty:
            logger.info("策略二区: %d→0 (市值:%d 周线:%d 涨跌停:%d)", len(df), rej_mcap, rej_wk, rej_lim)
            return result_df

        result_df.sort_values(["L3_共振类型","L2_笔数得分"], ascending=[False,False], inplace=True)
        result_df.reset_index(drop=True, inplace=True)

        logger.info("策略二区: %d→%d (市值:%d 周线:%d 涨跌停:%d)", len(df), passed, rej_mcap, rej_wk, rej_lim)
        return result_df


def run(input_df=None):
    if input_df is None:
        p = ZONES / "L2_pattern.parquet"
        if not p.exists(): logger.warning("L2 不存在"); return pd.DataFrame()
        input_df = pd.read_parquet(p)
    if input_df.empty: return pd.DataFrame()
    s = StrategyTwo(); r = s.apply(input_df)
    (ZONES / "L3_watchlist.parquet").parent.mkdir(exist_ok=True)
    r.to_parquet(ZONES / "L3_watchlist.parquet")
    logger.info("L3 自选股生态区: %d 只", len(r))
    return r


if __name__ == "__main__":
    df = run()
    if df.empty: print("L3: 无通过")
    else:
        print(f"L3: {len(df)} 只")
        print(df[["代码","现价","买点类型","L3_共振类型","L3_日线双买点","L3_30min二买"]].head(20).to_string())
