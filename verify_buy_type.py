"""B+ 结构验证器 + 买点细分标签。
logger = logging.getLogger(__name__)  # 验证逻辑

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
SIG_WEEKLY_DIR = WORKDIR / "data" / "signals_weekly"
SIG_30MIN_DIR = WORKDIR / "data" / "signals_30min"
STRUCT_DIR = WORKDIR / "data" / "struct_cache"
STRUCT_30M_DIR = WORKDIR / "data" / "struct_cache_30m"
STRUCT_WEEKLY_DIR = WORKDIR / "data" / "struct_cache_weekly"
DAILY_DIR = WORKDIR / "data" / "daily"
from core.date_utils import get_effective_date
from core.date_utils import get_effective_date


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

    # 截断到信号日期：已移除 —— 用全量笔（struct_cache max_bi_num=50），
    # 避免截断后落入与当前信号无关的远古结构
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

def _find_reference_pivot(code: str, signal_date, struct, daily):
    """找三买参照中枢。

    信号日前最近的在多头趋势（c > MA20 > MA60 且 MA20 斜率为正）
    中完成的中枢。忽略 CZSC 的 pivot_dir（上涨/下跌）标签，
    只用 MA 趋势确认。

    Returns: (pivot_row | None, zg: float)
    """
    pivots = struct[struct["direction"] == "pivot"]
    if pivots.empty:
        return None, 0
    completed = pivots[pd.to_datetime(pivots["edt"]) <= signal_date]
    if completed.empty:
        return None, 0  # 信号日之前没有完成的中枢

    for _, p in completed.iloc[::-1].iterrows():
        p_edt = pd.Timestamp(p["edt"])
        daily_at = daily[daily["date"] <= p_edt]
        if len(daily_at) < 60:
            continue
        c = float(daily_at["close"].iloc[-1])
        m20 = float(daily_at["close"].rolling(20).mean().iloc[-1])
        m60 = float(daily_at["close"].rolling(60).mean().iloc[-1])
        if len(daily_at) >= 26:
            m20_prev = float(daily_at["close"].rolling(20).mean().iloc[-6])
            slope = (m20 - m20_prev) / m20_prev * 100 if m20_prev > 0 else 0
        else:
            slope = 0
        if c > m20 > m60 and slope > 0:
            return p, float(p["pivot_zg"])
    return None, 0


def _verify_third_buy(code: str, signal_date=None):
    """三买验证 + 细分标签。

    signal_date: 信号触发日期（用于确定参照中枢的截止时间），
                 默认从信号文件自动推断。

    Returns: (passed: bool, tag: str)
    """
    sp = STRUCT_DIR / f"{code}.parquet"
    if not sp.exists():
        return False, ""
    struct = pd.read_parquet(sp)

    dp = DAILY_DIR / f"{code}.parquet"
    if not dp.exists():
        return False, "三买_跌破中枢"
    daily = pd.read_parquet(dp).sort_values("date")

    # 信号日期：优先传入值，否则从信号文件找三买首次触发日，兜底日线末日期
    if signal_date is None:
        try:
            sf = pd.read_parquet(SIG_DIR / f"{code}.parquet")
            for col in sf.columns:
                if "三买" not in col:
                    continue
                vals = sf[col].astype(str)
                for i in range(len(vals) - 1, 0, -1):
                    if "三买" in vals.iloc[i] and "三买" not in vals.iloc[i-1]:
                        signal_date = pd.to_datetime(sf["dt"].iloc[i])
                        break
                if signal_date is not None:
                    break
        except Exception:
            pass
    if signal_date is None:
        signal_date = daily["date"].iloc[-1]

    pivots = struct[struct["direction"] == "pivot"]
    if pivots.empty:
        return False, ""
    ref, zg = _find_reference_pivot(code, signal_date, struct, daily)
    if ref is None:
        return False, "三买_无中枢基础"
    if zg <= 0:
        return False, "三买_无中枢基础"

    bis = struct[struct["direction"].isin(["向上", "向下"])].sort_values("sdt")
    pivot_end = pd.to_datetime(ref["edt"])
    after_pivot = bis[pd.to_datetime(bis["sdt"]) >= pivot_end]
    up_after = after_pivot[after_pivot["direction"] == "向上"]
    if up_after.empty:
        return False, "三买_未突破中枢"

    breakout_mask = up_after["high"] > zg
    if not breakout_mask.any():
        return False, "三买_未突破中枢"
    breakout = up_after[breakout_mask].iloc[0]

    bis_after = after_pivot.sort_values("sdt")
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
            tag = "三买_弱突破"
        else:
            tag = "三买_类"

    price = float(daily.iloc[-1]["close"])
    if price < zg:
        return False, "三买_跌破中枢"

    if price > zg * 1.5:
        return False, "三买_远离入场区"
    return True, tag

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
SIG_WEEKLY_DIR = WORKDIR / "data" / "signals_weekly"
SIG_30MIN_DIR = WORKDIR / "data" / "signals_30min"
STRUCT_DIR = WORKDIR / "data" / "struct_cache"
STRUCT_30M_DIR = WORKDIR / "data" / "struct_cache_30m"
STRUCT_WEEKLY_DIR = WORKDIR / "data" / "struct_cache_weekly"
DAILY_DIR = WORKDIR / "data" / "daily"
from core.date_utils import get_effective_date
from core.date_utils import get_effective_date


# ========== 验证入口（原接口不变）==========


# ——— 三级联立共振检测 ———


# ========== 周日共振预筛 ==========

def check_resonance(code: str, buy_type: str, signal_date=None) -> bool:
    """三级联立：日线+周线+30min 是否同时存在同一买点类型（严格时间对齐）。

    读取 data/signals_weekly/ 和 data/signals_30min/ 的信号文件，
    检查末行是否存在与日线相同的买点类型，且满足时间新鲜度。
    
    时间对齐规则（与 zone2_regime.py 一致）：
    - 数据新鲜度：最后 bar 距今天 ≤ 7 自然日
    - 周线 ↔ 日线：≤ 5 自然日
    - 30分钟 ↔ 日线：≤ 2 自然日
    
    Args:
        code: 股票代码
        buy_type: '一买' | '二买' | '三买'
        signal_date: 日线 L1 信号日期 (pd.Timestamp 或 str)，用于时间对齐。
                     不传则退化到旧行为（仅检查末行买点类型）。
    Returns:
        True 当且仅当周线+30min 末行均有该买点类型信号，且满足时间对齐
    """
    import pandas as pd
    from datetime import datetime
    
    freq_map = {
        "周线":  (SIG_WEEKLY_DIR / f"{code}.parquet", 5),
        "30分钟": (SIG_30MIN_DIR / f"{code}.parquet", 2),
    }
    now = get_effective_date()
    if signal_date is not None:
        signal_date = pd.Timestamp(signal_date)
    
    for label, (path, max_day_gap) in freq_map.items():
        if not path.exists():
            return False
        try:
            df = pd.read_parquet(path)
            df["dt"] = pd.to_datetime(df["dt"])
            last_dt = df["dt"].max()
            
            # 数据新鲜度：最后 bar 距现在 ≤ 7 自然日
            if (now - last_dt).days > 7:
                return False
            
            # 时间对齐：找到离信号日最近的 bar，检查日历差
            if signal_date is not None:
                idx = (df["dt"] - signal_date).abs().idxmin()
                aligned_dt = df.loc[idx, "dt"]
                if abs((aligned_dt - signal_date).days) > max_day_gap:
                    return False
                last = df.iloc[idx]
            else:
                last = df.iloc[-1]
            
            # 买点类型匹配
            if not any(str(last[col]).startswith(f"{buy_type}_") for col in df.columns):
                return False
        except Exception:
            return False
    return True


# ——— 结构级共振检测（2026-08-07 新增）———

def check_structural_resonance(code: str, buy_type: str, signal_date=None, m30_signal_dt=None, weekly_signal_dt=None) -> bool:
    """结构级三级联立：三级别底分型同价区 + 中枢重叠 + 信号时间对齐。

    6 条件全过才标记 [周_日_30m_结构联立]：
    1. 日线 → 取最后向下笔 low + 中枢区间
    2. 30min → 向下笔 low 与日线偏差 < 5%
    3. 日线+30min → 中枢重叠（任一级别缺中枢则跳过）
    4. 周线 → 最后向下笔 low 与日线偏差 < 2%
    5. 日线+周线 → 中枢重叠（任一级别缺中枢则跳过）
    6. 三级信号时间对齐 ≤ 5 天（缺信号文件 → 不标记）

    Args:
        code: 股票代码
        buy_type: '一买' | '二买' | '三买'（一买跳过，直接 False）
        signal_date: 日线信号日期
        m30_signal_dt: 30min 信号文件最新日期（可选，用于时间对齐）
        weekly_signal_dt: 周线信号文件最新日期（可选，用于时间对齐）
    Returns:
        True 当且仅当底分型同价区 + 中枢重叠 + 时间对齐
    """
    if buy_type == "一买":
        return False

    if signal_date is not None:
        signal_date = pd.Timestamp(signal_date)
    else:
        signal_date = get_effective_date()

    PRICE_TOLERANCE = 0.02

    sp = STRUCT_DIR / f"{code}.parquet"
    if not sp.exists():
        return False
    try:
        sd = pd.read_parquet(sp)
        sd["sdt_dt"] = pd.to_datetime(sd["sdt"])
        sd["edt_dt"] = pd.to_datetime(sd["edt"])
        pre_d = sd[sd["sdt_dt"].dt.date <= signal_date.date()]
        d_down = pre_d[pre_d["direction"] == "向下"]
        if len(d_down) == 0:
            return False
        d_bottom = d_down.iloc[-1]
        d_low = float(d_bottom["low"])
        d_bottom_sdt = pd.to_datetime(d_bottom["sdt"])
        d_bottom_edt = pd.to_datetime(d_bottom["edt"])

        d_pivots = pre_d[pre_d["direction"] == "pivot"]
        if len(d_pivots) == 0:
            d_zg, d_zd = 0.0, 0.0
        else:
            lp = d_pivots.iloc[-1]
            d_zg = float(lp.get("pivot_zg", 0) or 0)
            d_zd = float(lp.get("pivot_zd", 0) or 0)
    except Exception:
        return False

    # --- 30min 价格 + 中枢 + 包含 ---
    sp30 = STRUCT_30M_DIR / f"{code}.parquet"
    if not sp30.exists():
        return False
    try:
        s30 = pd.read_parquet(sp30)
        s30["sdt_dt"] = pd.to_datetime(s30["sdt"])
        s30["edt_dt"] = pd.to_datetime(s30["edt"])
        pre_30 = s30[s30["sdt_dt"].dt.date <= signal_date.date()]
        m30_down = pre_30[pre_30["direction"] == "向下"]
        if len(m30_down) == 0:
            return False

        # 条件 2: 30min 价格对齐（只取最后向下笔，容差 3%，2026-08-10）
        m30_last = m30_down.iloc[-1]
        if abs(m30_last["low"] - d_low) / d_low >= 0.03:
            return False

        # 条件 3: 日线+30min 中枢重叠
        m30_pivots = pre_30[pre_30["direction"] == "pivot"]
        has_m30_pivot = len(m30_pivots) > 0
        has_d_pivot = not (d_zg == 0 and d_zd == 0)
        if has_m30_pivot and has_d_pivot:
            lp30 = m30_pivots.iloc[-1]
            m30_zg = float(lp30.get("pivot_zg", 0) or 0)
            m30_zd = float(lp30.get("pivot_zd", 0) or 0)
            if not (min(m30_zg, d_zg) >= max(m30_zd, d_zd)):
                return False
    except Exception:
        return False

    # --- 周线 价格 + 中枢 + 重叠 ---
    wp = STRUCT_WEEKLY_DIR / f"{code}.parquet"
    if not wp.exists():
        return False
    try:
        sw = pd.read_parquet(wp)
        sw["sdt_dt"] = pd.to_datetime(sw["sdt"])
        sw["edt_dt"] = pd.to_datetime(sw["edt"])
        pre_w = sw[sw["sdt_dt"] <= signal_date]
        w_down = pre_w[pre_w["direction"] == "向下"]
        if len(w_down) == 0:
            return False

        # 条件 4: 周线价格对齐（只取最后向下笔，容差 2%，2026-08-10）
        w_last = w_down.iloc[-1]
        if abs(w_last["low"] - d_low) / d_low >= 0.02:
            return False

        # 条件 5: 日线+周线 中枢重叠
        w_pivots = pre_w[pre_w["direction"] == "pivot"]
        has_w_pivot = len(w_pivots) > 0
        has_d_pivot = not (d_zg == 0 and d_zd == 0)
        if has_w_pivot and has_d_pivot:
            lpw = w_pivots.iloc[-1]
            w_zg = float(lpw.get("pivot_zg", 0) or 0)
            w_zd = float(lpw.get("pivot_zd", 0) or 0)
            if not (min(w_zg, d_zg) >= max(w_zd, d_zd)):
                return False
    except Exception:
        return False

    # 条件 6: 三级信号时间对齐（缺任一信号文件 → 直接不标记）
    if m30_signal_dt is None or weekly_signal_dt is None:
        return False
    d_dt = signal_date
    if not (abs((m30_signal_dt - d_dt).days) <= 5 and
            abs((weekly_signal_dt - d_dt).days) <= 5 and
            abs((weekly_signal_dt - m30_signal_dt).days) <= 5):
        return False

    return True

# --- 周-日结构共振检测 (2026-08-07) ---

def check_weekly_structural_resonance(code: str, signal_date=None) -> bool:
    """周日结构共振: 日线底分型与周线底分型是否在同一价格区。

    读取 data/struct_cache/ 和 data/struct_cache_weekly/，
    检查信号日之前的日线最后向下笔 low 与周线最后向下笔 low 偏差 < 2%，且日线/周线最后中枢区间重叠（借鉴三级联立，2026-08-09）。
    """
    if signal_date is not None:
        signal_date = pd.Timestamp(signal_date)
    else:
        signal_date = get_effective_date()

    PRICE_TOLERANCE = 0.02

    sp = STRUCT_DIR / f"{code}.parquet"
    if not sp.exists():
        return False
    try:
        sd = pd.read_parquet(sp)
        sd["sdt_dt"] = pd.to_datetime(sd["sdt"])
        sd["edt_dt"] = pd.to_datetime(sd["edt"])
        pre_d = sd[sd["sdt_dt"].dt.date <= signal_date.date()]
        d_down = pre_d[pre_d["direction"] == "向下"]
        if len(d_down) == 0:
            return False
        d_low = float(d_down.iloc[-1]["low"])
    except Exception:
        return False

    wp = STRUCT_WEEKLY_DIR / f"{code}.parquet"
    if not wp.exists():
        return False
    try:
        sw = pd.read_parquet(wp)
        sw["sdt_dt"] = pd.to_datetime(sw["sdt"])
        pre_w = sw[sw["sdt_dt"] <= signal_date]
        w_down = pre_w[pre_w["direction"] == "向下"]
        if len(w_down) == 0:
            return False

        # 只取周线最后向下笔（与日线最后向下笔对齐，不做全历史扫描）
        w_last_low = float(w_down.iloc[-1]["low"])
        if not (abs(w_last_low - d_low) / d_low < PRICE_TOLERANCE):
            return False

        # 中枢重叠检查（借鉴三级联立）：日线最后中枢与周线最后中枢区间重叠
        d_pivots = pre_d[pre_d["direction"] == "pivot"]
        w_pivots = pre_w[pre_w["direction"] == "pivot"]
        has_d_pivot = len(d_pivots) > 0
        has_w_pivot = len(w_pivots) > 0
        if has_d_pivot and has_w_pivot:
            lp_d = d_pivots.iloc[-1]
            d_zg = float(lp_d.get("pivot_zg", 0) or 0)
            d_zd = float(lp_d.get("pivot_zd", 0) or 0)
            lp_w = w_pivots.iloc[-1]
            w_zg = float(lp_w.get("pivot_zg", 0) or 0)
            w_zd = float(lp_w.get("pivot_zd", 0) or 0)
            if d_zg == 0 and d_zd == 0:
                pass  # 日线无有效中枢 → 跳过
            elif w_zg == 0 and w_zd == 0:
                pass  # 周线无有效中枢 → 跳过
            elif not (min(d_zg, w_zg) >= max(d_zd, w_zd)):
                return False  # 中枢不重叠
        return True
    except Exception:
        return False

