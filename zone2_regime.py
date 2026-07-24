#!/usr/bin/env python3
"""L2 Regime-Routed Entry Filter -- EntryFilter pipeline adapter.

Replaces zone2_pattern.py simple scoring with regime-weighted + threshold filtering.
Default regime="CHOP" until regime detection is integrated.
"""

import logging
from pathlib import Path

import pandas as pd

from entry_filter import EntryFilter, REGIME_THRESHOLDS
from zone2_pattern import _detect_changes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zone2_regime")

BASE = Path(__file__).parent
SIGNALS = BASE / "data" / "signals"
DAILY = BASE / "data" / "daily"
ZONES = BASE / "data" / "zones"

# Per-run resonance caches
_weekly_res_cache = {}
_30min_res_cache = {}

WEEKLY_SIG = BASE / "data" / "signals_weekly"
W_B1 = "周线_D1B_BUY1"; W_BS2 = "周线_D1#SMA#21_BS2辅助V230320"
W_BS3 = "周线_D1#SMA#34_BS3辅助V230318"; W_SAN = "周线_D1_三买辅助V230228"
W_SAN2 = "周线_D1#SMA#34_BS3辅助V230319"

def _check_weekly_resonance(code):
    if code in _weekly_res_cache:
        return _weekly_res_cache[code]
    wp = WEEKLY_SIG / f"{code}.parquet"
    if not wp.exists(): return None
    try:
        w = pd.read_parquet(wp); last = w.iloc[-1]
        wtypes = set()
        for col, bt in [(W_B1,"一买"),(W_BS2,"二买"),(W_BS3,"三买"),(W_SAN,"三买"),(W_SAN2,"三买")]:
            if col in w.columns:
                v = str(last[col])
                if bt in v and v != "0": wtypes.add(bt)
        r = wtypes if wtypes else set()
        _weekly_res_cache[code] = r
        return r
    except:
        _weekly_res_cache[code] = None
        return None

M30_SIG = BASE / "data" / "signals_30min"
M_B1 = "30分钟_D1B_BUY1"; M_BS2 = "30分钟_D1#SMA#21_BS2辅助V230320"
M_BS3 = "30分钟_D1#SMA#34_BS3辅助V230318"; M_SAN = "30分钟_D1_三买辅助V230228"
M_SAN2 = "30分钟_D1#SMA#34_BS3辅助V230319"

def _check_30min_resonance(code):
    if code in _30min_res_cache:
        return _30min_res_cache[code]
    mp = M30_SIG / f"{code}.parquet"
    if not mp.exists(): return None
    try:
        m = pd.read_parquet(mp); last = m.iloc[-1]
        mtypes = set()
        for col, bt in [(M_B1,"一买"),(M_BS2,"二买"),(M_BS3,"三买"),(M_SAN,"三买"),(M_SAN2,"三买")]:
            if col in m.columns:
                v = str(last[col])
                if bt in v and v != "0": mtypes.add(bt)
        r = mtypes if mtypes else set()
        _30min_res_cache[code] = r
        return r
    except:
        _30min_res_cache[code] = None
        return None

# Map EntryFilter dimension_scores to legacy L2/L3/L4 column names
DIM_MAP = {
    "笔数": "L2_笔数得分",
    "量比": "L2_量比得分",
    "MACD": "L2_MACD得分",
    "核心验证": "L2_背驰得分",
    "底分型": "L2_底分型得分",
    "中枢": "L2_中枢得分",
    "距离": "L2_中枢得分",
}


def _get_signal_value_at_date(sig_df, col, date_str):
    ts = pd.Timestamp(date_str)
    row = sig_df[sig_df["dt"] == ts]
    if len(row) == 0:
        return ""
    return str(row.iloc[0].get(col, ""))


def _determine_buy_type(signal_val):
    if "一买" in signal_val: return "一买"
    if "二买" in signal_val: return "二买"
    if "三买" in signal_val: return "三买"
    return "一买"


def run(input_df=None, regime="CHOP", bplus_codes=None):
    if input_df is None:
        p = ZONES / "L1_deposition.parquet"
        if not p.exists():
            logger.warning("L1 not found")
            return pd.DataFrame()
        input_df = pd.read_parquet(p)
    if input_df.empty:
        return pd.DataFrame()

    global _weekly_res_cache, _30min_res_cache
    _weekly_res_cache = {}
    _30min_res_cache = {}
    logger.info("L2 Regime: regime=%s candidates=%d", regime, len(input_df))

    results = []
    rej = {"no_changes": 0, "below_thr": 0, "data_miss": 0, "czsc_err": 0}
    passed = 0

    for idx, (_, row) in enumerate(input_df.iterrows()):
        code = row["代码"]
        price = row["现价"]

        sp = SIGNALS / f"{code}.parquet"
        dp = DAILY / f"{code}.parquet"
        if not sp.exists() or not dp.exists():
            rej["data_miss"] += 1; continue

        try:
            sig_df = pd.read_parquet(sp)
            daily_df = pd.read_parquet(dp)
        except Exception:
            rej["data_miss"] += 1; continue

        if len(sig_df) < 2 or len(daily_df) < 30:
            rej["data_miss"] += 1; continue

        changes = _detect_changes(sig_df, 20)
        if not changes:
            rej["no_changes"] += 1; continue

        try:
            ef = EntryFilter(code, daily_df, sig_df, regime=regime)
        except Exception:
            rej["czsc_err"] += 1; continue

        for ch in changes:
            col_name = ch["col"]
            date_str = ch["date"]
            sig_val = _get_signal_value_at_date(sig_df, col_name, date_str)
            buy_type = _determine_buy_type(sig_val)

            # 多级别共振标识：二级(周×日×共振) / 三级联立(周×日×30m×共振)
            # 附加到买点类型标签后面，三级联立直通（加入 bplus_codes）
            wtypes = _check_weekly_resonance(code)
            r30 = _check_30min_resonance(code)
            resonance_suffix = ""
            is_three_level = False
            if wtypes is not None and buy_type in wtypes:
                if r30 is not None and buy_type in r30:
                    resonance_suffix = " [周_日_30m_联立]"
                    is_three_level = True
                else:
                    resonance_suffix = " [周_日_共振]"

            buy_event = {"date": date_str, "signal_label": sig_val, "col": col_name}

            try:
                result = ef.filter(buy_event, regime)
            except Exception:
                rej["czsc_err"] += 1; continue

            # 三级联立加入 bplus 直通（必须在阈值检查之前）
            if is_three_level and bplus_codes is not None:
                bplus_codes.add((code, buy_type))

            # B+ / 三级联立 通过的买点不设阈值，照常打分但不拦截
            if not result.passed:
                if bplus_codes and (code, buy_type) in bplus_codes:
                    pass  # B+ / 联立 保留，正常计入
                else:
                    rej["below_thr"] += 1; continue

            threshold = REGIME_THRESHOLDS.get(buy_type, {}).get(regime, 350)

            buy_type_label = str(row.get("买点类型", buy_type)) + resonance_suffix
            out_row = {
                "代码": code, "现价": price,
                "最新日期": row.get("最新日期", ""),
                "买点类型": buy_type_label,
                "状态详情": row.get("状态详情", ""),
                "信号数": row.get("信号数", 0),
                "L2_综合得分": round(result.total_score, 1),
                "L2_阈值": threshold,
                "L2_Regime": regime,
                "L2_维度得分": result.dimension_scores,
            }

            dims = result.dimension_scores
            for dim_key, col_name_map in DIM_MAP.items():
                out_row[col_name_map] = dims.get(dim_key, 0)

            for legacy_col in ["L2_笔数得分", "L2_量比得分", "L2_MACD得分",
                                "L2_背驰得分", "L2_底分型得分", "L2_中枢得分"]:
                if legacy_col not in out_row:
                    out_row[legacy_col] = 0

            results.append(out_row)
            passed += 1

        if (idx + 1) % 100 == 0:
            logger.info("  progress: %d/%d (passed:%d no_ch:%d thr:%d)",
                        idx + 1, len(input_df), passed,
                        rej["no_changes"], rej["below_thr"])

    result_df = pd.DataFrame(results)
    if result_df.empty:
        logger.info("L2 Regime: %d->0 (no_changes:%d below_thr:%d data_miss:%d czsc:%d)",
                    len(input_df), *[rej[k] for k in ["no_changes","below_thr","data_miss","czsc_err"]])
        return result_df

    result_df.sort_values("L2_综合得分", ascending=False, inplace=True)
    result_df = result_df.drop_duplicates(subset=["代码", "买点类型"], keep="first")
    result_df.reset_index(drop=True, inplace=True)

    out_path = ZONES / "L2_regime.parquet"
    result_df.to_parquet(out_path)
    logger.info("L2 Regime: %d candidates -> %d passed (no_ch:%d thr:%d miss:%d czsc:%d)",
                len(input_df), len(result_df),
                *[rej[k] for k in ["no_changes","below_thr","data_miss","czsc_err"]])
    return result_df


if __name__ == "__main__":
    df = run()
    if df.empty:
        print("L2 Regime: no passes")
    else:
        print(f"L2 Regime: {len(df)} passed")
        print(df[["代码", "现价", "买点类型", "L2_综合得分", "L2_Regime"]].head(15).to_string())
