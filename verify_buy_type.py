"""B+ 结构验证器 + 买点细分标签。

verify_buy_type(code, buy_type) -> bool     B+ 验证（原接口不变）
get_buy_type_tag(code, buy_type) -> str     返回细分标签
"""
import pandas as pd
from pathlib import Path

LOOKBACK_DAYS = 120
ZG_UPPER_RATIO = 1.2
BUY1_COL = "日线_D1B_BUY1"
BS2_COL = "日线_D1#SMA#21_BS2辅助V230320"  # 二买信号列


WORKDIR = Path(__file__).resolve().parent
SIG_DIR = WORKDIR / "data" / "signals"
STRUCT_DIR = WORKDIR / "data" / "struct_cache"
DAILY_DIR = WORKDIR / "data" / "daily"


# ========== 验证入口（原接口不变）==========

def verify_buy_type(code: str, buy_type: str) -> bool:
    """B+ 结构验证。一买免检，二买/三买需结构链确认。"""
    if buy_type == "一买":
        return True
    try:
        if buy_type == "二买":
            passed, _ = _verify_second_buy(code)
            return passed
        elif buy_type == "三买":
            passed, _ = _verify_third_buy(code)
            return passed
    except Exception:
        pass
    return False


# ========== 标签入口（新接口）==========

def _get_trend(code: str) -> str:
    """检测个股趋势方向。"""
    try:
        dp = DAILY_DIR / f"{code}.parquet"
        if not dp.exists():
            return "震荡"
        daily = pd.read_parquet(dp).sort_values("date")
        if len(daily) < 60:
            return "震荡"
        close = float(daily.iloc[-1]["close"])
        ma20 = float(daily["close"].rolling(20).mean().iloc[-1])
        ma60 = float(daily["close"].rolling(60).mean().iloc[-1])
        ma20_prev = float(daily["close"].rolling(20).mean().iloc[-6]) if len(daily) >= 26 else ma20
        slope = (ma20 - ma20_prev) / ma20_prev * 100 if ma20_prev > 0 else 0
        if close > ma20 > ma60 and slope > 0:
            return "多头"
        elif close < ma20 < ma60 and slope < 0:
            return "空头"
        return "震荡"
    except:
        return "震荡"


def get_buy_label(code: str, buy_type: str = None) -> str:
    """获取股票的完整买点标签（验证通过和不通过都有）。

    格式: {CZSC来源}_{CZSC规则}_{结构状态}

    示例:
      "一买_D1B"                    — 标准一买，免检
      "二买_BS2_标准"                — B+通过的结构完整二买
      "二买_BS2_无基础"              — 从未有过一买，辅助信号不可靠
      "二买_BS2_创新低"              — 回踩已破一买底，结构破坏
      "二买_BS2_未企稳"              — 价格偏离回踩区域
      "三买_V228_12笔_无中枢基础"     — 12笔规则标记三买，但无上涨中枢
      "三买_无中枢基础"               — 无上涨中枢
      "三买_未突破中枢"               — 有中枢但价格未突破
      "三买_回踩进中枢"               — 突破后回踩进入了中枢
      "三买_跌破中枢"               — 价格已跌破中枢
    """
    BUY_COLS = ["日线_D1B_BUY1","日线_D1#SMA#21_BS2辅助V230320","日线_D1#SMA#34_BS3辅助V230318",
                "日线_D1_三买辅助V230228","日线_D1#SMA#34_BS3辅助V230319"]
    try:
        sf = pd.read_parquet(SIG_DIR / f"{code}.parquet")
        last = sf.iloc[-1]

        # 显式指定 buy_type → 直接走标签生成
        if buy_type is not None:
            if buy_type == "一买":
                return "一买"
            try:
                if buy_type == "二买":
                    passed, tag = _verify_second_buy(code)
                    trend = _get_trend(code)
                    return tag if passed else tag + "_" + trend
                elif buy_type == "三买":
                    passed, tag = _verify_third_buy(code)
                    trend = _get_trend(code)
                    return tag if passed else tag + "_" + trend
            except Exception:
                pass
            return buy_type  # fallback"

        for c in BUY_COLS:
            if c not in sf.columns: continue
            v = str(last.get(c, ''))
            if v.startswith("一买_"):
                return "一买_D1B"
            elif v.startswith("二买_"):
                passed, tag = _verify_second_buy(code)
                if passed:
                    return tag  # B+通过，不加趋势
                return tag + "_" + _get_trend(code)  # B+未通过，加趋势
            elif v.startswith("三买_"):
                # Determine CZSC source from column
                if "三买辅助V230228" in c:
                    rule = v.split("_")[1] if "_" in v else ""
                    src = f"V228_{rule}" if rule else "V228"
                elif "BS3辅助V230318" in c:
                    src = "BS318"
                elif "BS3辅助V230319" in c:
                    src = "BS319"
                else:
                    src = "?"
                passed, tag = _verify_third_buy(code)
                if not passed:
                    return tag + "_" + _get_trend(code)  # B+未通过，加趋势
                return f"三买_{tag}"
            break
    except Exception:
        pass
    return "无买点信号"


def get_buy_type_tag(code: str, buy_type: str) -> str:
    """获取买点细分标签。
    
    二买: "二买_标准" / "二买_浅回踩" / "二买_类"
    三买: "三买_标准" / "三买_弱突破" / "三买_类"
    一买: "一买"（无细分）
    验证不通过: ""
    """
    if buy_type == "一买":
        return "一买"
    try:
        if buy_type == "二买":
            passed, tag = _verify_second_buy(code)
            return tag if passed else ""
        elif buy_type == "三买":
            passed, tag = _verify_third_buy(code)
            return tag if passed else ""
    except Exception:
        pass
    return ""


# ========== 二买验证 ==========

def _verify_second_buy(code: str):
    """二买验证 + 细分标签。

    Returns: (passed: bool, tag: str)
    """
    # 1. 一买历史
    sf = pd.read_parquet(SIG_DIR / f"{code}.parquet")
    if BUY1_COL not in sf.columns:
        return False, "二买_无基础"
    recent = sf[BUY1_COL].tail(LOOKBACK_DAYS).dropna()
    if not any(str(v).startswith("一买_") for v in recent):
        return False, "二买_无基础"

    # 2. 结构链：根据信号日期截断笔序列
    # 找到二买信号首次触发日期
    signal_date = None
    if BS2_COL in sf.columns:
        bs2_vals = sf[BS2_COL].tail(LOOKBACK_DAYS).astype(str)
        for i in range(len(bs2_vals)-1, 0, -1):  # 从最近往前找最后一次变化
            prev = bs2_vals.iloc[i-1]; curr = bs2_vals.iloc[i]
            if "二买" not in prev and "二买" in curr:
                signal_date = pd.to_datetime(sf["dt"].iloc[len(sf)-len(bs2_vals)+i])
                break

    sp = STRUCT_DIR / f"{code}.parquet"
    if not sp.exists():
        return False, "二买_无基础"
    struct = pd.read_parquet(sp)
    bis = struct[struct["direction"].isin(["向上", "向下"])].sort_values("sdt")

    # 截断到信号日期：仅当信号在 30 天内触发时才截断（长期保持的信号用全量笔）
    if signal_date is not None:
        days_since = (pd.Timestamp.now() - signal_date).days
        if days_since <= 30:
            bis = bis[pd.to_datetime(bis["edt"]) <= signal_date]

    if len(bis) < 3:
        return False, "二买_未企稳"

    last = bis.iloc[-1]; second = bis.iloc[-2]; third = bis.iloc[-3]

    if last["direction"] != "向下" or second["direction"] != "向上" or third["direction"] != "向下":
        return False, "二买_未企稳"
    if last["low"] <= third["low"]:
        return False, "二买_创新低"

    dp = DAILY_DIR / f"{code}.parquet"
    if not dp.exists():
        return False, "二买_未企稳"
    price = float(pd.read_parquet(dp).sort_values("date").iloc[-1]["close"])
    if price < float(last["low"]) * 0.95:
        return False, "二买_未企稳"

    # 3. 细分标签
    denom = float(second["high"]) - float(third["low"])
    if denom <= 0:
        return True, "二买_类"
    retrace = (float(second["high"]) - float(last["low"])) / denom
    has_fx = "底分型" in str(last.get("fx_b_mark", ""))

    if retrace < 0.3 and has_fx:
        return True, "二买_浅回踩"
    elif retrace <= 0.7 and has_fx:
        return True, "二买_标准"
    else:
        return True, "二买_类"


# ========== 三买验证 ==========

def _verify_third_buy(code: str):
    """三买验证 + 细分标签。

    Returns: (passed: bool, tag: str)
    """
    sp = STRUCT_DIR / f"{code}.parquet"
    if not sp.exists():
        return False, ""
    struct = pd.read_parquet(sp)
    pivots = struct[struct["direction"] == "pivot"]
    up_pivots = pivots[pivots["pivot_dir"] == "上涨"]
    if up_pivots.empty:
        return False, "三买_无中枢基础"

    last_pivot = up_pivots.iloc[-1]
    zg = float(last_pivot["pivot_zg"])
    if zg <= 0:
        return False, "三买_无中枢基础"

    bis = struct[struct["direction"].isin(["向上", "向下"])].sort_values("sdt")  # sdt: chronological order, edt can misorder short bis
    pivot_end = pd.to_datetime(last_pivot["edt"])
    after_pivot = bis[pd.to_datetime(bis["sdt"]) >= pivot_end]
    up_after = after_pivot[after_pivot["direction"] == "向上"]
    if up_after.empty:
        return False, "三买_未突破中枢"

    # 检查所有突破笔（不只看第一根）
    breakout_mask = up_after["high"] > zg
    if not breakout_mask.any():
        return False, "三买_未突破中枢"
    breakout = up_after[breakout_mask].iloc[0]  # 取第一次有效突破

    bis_after = after_pivot.sort_values("sdt")  # sdt: chronological order, edt can misorder short bis
    tag = "三买_类"
    if len(bis_after) >= 2 and bis_after.iloc[-1]["direction"] == "向下":
        retrace = bis_after.iloc[-1]
        if retrace["low"] <= zg:
            return False, "三买_回踩进中枢"
        has_fx = "底分型" in str(retrace.get("fx_b_mark", ""))
        breakout_pct = (float(breakout["high"]) - zg) / zg
        if breakout_pct > 0.05 and has_fx:
            tag = "三买_标准"
        elif has_fx:
            tag = "三买_弱突破"  # 突破幅度 < 5% ZG
        else:
            tag = "三买_类"

    dp = DAILY_DIR / f"{code}.parquet"
    if not dp.exists():
        return False, "三买_跌破中枢"
    price = float(pd.read_parquet(dp).sort_values("date").iloc[-1]["close"])
    if price < zg:
        return False, "三买_跌破中枢"

    return True, tag
