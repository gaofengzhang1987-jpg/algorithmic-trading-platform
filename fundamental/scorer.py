"""基本面评分引擎 — 三框架（tech / finance / standard），白盒规则。

依赖：fundamental/puller.py 拉取的 stock_financial_abstract 数据。
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("fundamental.scorer")


def _get_val(df: pd.DataFrame, indicator: str, period: str) -> float | None:
    """从 financial_abstract DataFrame 提取指定指标在指定报告期的值。"""
    mask = df["指标"] == indicator
    if not mask.any():
        return None
    row = df[mask].iloc[0]
    val = row.get(period, None)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return float(val)


def _get_latest_period(df: pd.DataFrame) -> str:
    """找最新有数据的报告期列。"""
    period_cols = [c for c in df.columns if c not in ("选项", "指标") and c[0].isdigit()]
    for col in period_cols:
        vals = df[col].dropna()
        if len(vals) > 5:  # 至少 5 个有效值才认为有数据
            return col
    return period_cols[0] if period_cols else ""


def _score_linear(val: float, thresholds: list[tuple[float, float]]) -> float:
    """分段线性评分。

    thresholds: [(上限, 得分), ...], 从低到高排列。
    超过最高上限则拿最高分，低于最低下限则拿最低分。
    例: [(0, 0), (5, 10), (10, 15), (15, 25)] 表示
        ROE<0→0, 0-5→线性, 5-10→线性, 10-15→线性, ≥15→25
    """
    if val is None:
        return 0.0

    # Below lowest threshold
    low_thr, low_score = thresholds[0]
    if val <= low_thr:
        return low_score

    # Above highest threshold
    high_thr, high_score = thresholds[-1]
    if val >= high_thr:
        return high_score

    # Linear interpolation between thresholds
    for i in range(len(thresholds) - 1):
        t1, s1 = thresholds[i]
        t2, s2 = thresholds[i + 1]
        if t1 <= val <= t2:
            ratio = (val - t1) / (t2 - t1) if t2 != t1 else 0
            return s1 + ratio * (s2 - s1)

    return 0.0


# ─── 三个评分框架 ────────────────────────────────────────────

FRAMEWORKS = {
    "standard": {
        "name": "标准型",
        "dims": [
            {
                "label": "盈利能力",
                "weight": 25,
                "indicator": "净资产收益率(ROE)",
                "thresholds": [(0, 5), (5, 12), (10, 18), (15, 25)],
            },
            {
                "label": "成长性",
                "weight": 20,
                "indicator": "营业总收入增长率",
                "thresholds": [(0, 5), (10, 12), (20, 20)],
            },
            {
                "label": "现金流质量",
                "weight": 20,
                "indicator": "经营活动净现金/归属母公司的净利润",
                "thresholds": [(0, 5), (0.5, 12), (1.0, 20)],
            },
            {
                "label": "财务安全",
                "weight": 20,
                "indicator": "资产负债率",
                "thresholds": [(80, 5), (60, 10), (40, 15), (20, 20)],
                "reverse": True,  # 越低越好
            },
            {
                "label": "盈利质量",
                "weight": 15,
                "indicator": "毛利率",
                "thresholds": [(20, 5), (30, 8), (50, 15)],
            },
        ],
    },
    "tech": {
        "name": "科技成长型",
        "dims": [
            {
                "label": "增长质量",
                "weight": 30,
                "indicator": "营业总收入增长率",
                "thresholds": [(0, 5), (15, 15), (30, 22), (50, 30)],
            },
            {
                "label": "盈利质量",
                "weight": 25,
                "indicator": "毛利率",
                "thresholds": [(20, 5), (40, 12), (60, 18), (80, 25)],
            },
            {
                "label": "现金流安全",
                "weight": 25,
                "indicator": "经营活动净现金/销售收入",
                "thresholds": [(0, 5), (0.1, 12), (0.3, 18), (0.5, 25)],
            },
            {
                "label": "财务安全",
                "weight": 10,
                "indicator": "资产负债率",
                "thresholds": [(70, 2), (50, 5), (30, 7), (15, 10)],
                "reverse": True,
            },
            {
                "label": "增长持续性",
                "weight": 10,
                "indicator": "归属母公司净利润增长率",
                "thresholds": [(0, 3), (10, 6), (30, 8), (50, 10)],
            },
        ],
    },
    "finance": {
        "name": "金融型",
        "dims": [
            {
                "label": "盈利能力",
                "weight": 30,
                "indicator": "净资产收益率(ROE)",
                "thresholds": [(0, 5), (5, 12), (10, 20), (15, 30)],
            },
            {
                "label": "成长性",
                "weight": 20,
                "indicator": "营业总收入增长率",
                "thresholds": [(0, 5), (5, 12), (15, 20)],
            },
            {
                "label": "现金流质量",
                "weight": 20,
                "indicator": "经营活动净现金/归属母公司的净利润",
                "thresholds": [(0, 5), (0.5, 12), (1.0, 20)],
            },
            {
                "label": "资产效率",
                "weight": 15,
                "indicator": "总资产净利率_平均",
                "thresholds": [(0.2, 3), (0.5, 6), (1.0, 10), (1.5, 15)],
            },
            {
                "label": "盈利质量",
                "weight": 15,
                "indicator": "销售净利率",
                "thresholds": [(10, 3), (20, 7), (30, 11), (40, 15)],
            },
        ],
    },
}


def score_one(df: pd.DataFrame, framework: str = "standard") -> dict:
    """对单只股票按指定框架评分。

    Args:
        df: stock_financial_abstract 返回的 DataFrame
        framework: "standard" | "tech" | "finance"

    Returns:
        dict: {score (0-100), dims: [{label, score, weight}], indicators: {name: value}}
    """
    fw = FRAMEWORKS.get(framework, FRAMEWORKS["standard"])
    period = _get_latest_period(df)

    if not period:
        return {"score": -1.0, "dims": [], "indicators": {}, "error": "无有效报告期"}

    total = 0.0
    dim_results = []
    indicators = {}

    for dim in fw["dims"]:
        val = _get_val(df, dim["indicator"], period)
        indicators[dim["indicator"]] = val

        if val is None:
            dim_results.append({"label": dim["label"], "score": 0, "weight": dim["weight"], "raw": None})
            continue

        if dim.get("reverse"):
            # Reverse scoring: higher raw value → lower score
            # Flip the thresholds
            rev_thresholds = []
            for t, s in dim["thresholds"]:
                rev_thresholds.append((t, s))
            # For reverse scoring: the thresholds define "below X → score Y"
            # Since lower is better, we invert: find the highest threshold that val exceeds
            score = 0.0
            for t, s in sorted(dim["thresholds"], key=lambda x: x[0]):
                if val <= t:
                    score = s
                    break
            else:
                score = dim["thresholds"][-1][1]  # above all thresholds → max score
        else:
            score = _score_linear(val, dim["thresholds"])

        dim_results.append({"label": dim["label"], "score": round(score, 1), "weight": dim["weight"], "raw": val})
        total += score * dim["weight"] / 100.0

    return {
        "score": round(total, 1),
        "dims": dim_results,
        "indicators": indicators,
        "framework": framework,
        "framework_name": fw["name"],
        "period": period,
    }


def score_batch(data_map: dict[str, pd.DataFrame], framework_map: dict[str, str]) -> pd.DataFrame:
    """批量评分。

    Args:
        data_map: code → financial_abstract DataFrame
        framework_map: code → framework name ("standard"|"tech"|"finance")

    Returns:
        DataFrame: code + fundamental_score + 各维子分 + 原始指标
    """
    rows = []
    failed = []

    for code, df in data_map.items():
        fw = framework_map.get(code, "standard")
        result = score_one(df, fw)

        if result["score"] < 0:
            failed.append(code)
            continue

        row = {
            "code": code,
            "fundamental_score": result["score"],
            "fundamental_framework": result["framework"],
            "fundamental_period": result["period"],
        }
        for dim in result["dims"]:
            row[f"f_{dim['label']}"] = dim["score"]
            row[f"f_{dim['label']}_raw"] = dim["raw"]

        rows.append(row)

    if failed:
        logger.warning("评分失败: %d 只 — %s", len(failed), ", ".join(failed[:10]))

    return pd.DataFrame(rows)


# ─── 风险检测（11 项） ─────────────────────────────────────────

def detect_risks(fa_df: pd.DataFrame, bs_df: "pd.DataFrame | None",
                 stock_name: str = "") -> dict:
    """检测单只股票的风险信号。

    Args:
        fa_df: stock_financial_abstract 返回的 DataFrame
        bs_df: stock_balance_sheet_by_yearly_em 返回的 DataFrame，可为 None
        stock_name: 股票名称（用于 ST 标记检测）

    Returns:
        dict: {risk_score (0-10), risk_flags (str), risks: [{item, level, detail}]}
    """
    flags = []  # (item_name, level: 0=clean, 1=yellow, 2=red)
    period = _get_latest_period(fa_df)

    # 辅助：获取多期值
    period_cols = [c for c in fa_df.columns if c not in ("选项", "指标") and c[0].isdigit()]
    period_cols_sorted = sorted(period_cols, reverse=True)  # 最新在前

    def _fa_val(indicator: str, p: str = None) -> float | None:
        return _get_val(fa_df, indicator, p or period)

    def _fa_vals(indicator: str, n: int = 8) -> list[float | None]:
        """获取最近 n 期值。"""
        return [_fa_val(indicator, p) for p in period_cols_sorted[:n]]

    def _bs_val(col: str) -> float | None:
        if bs_df is None or bs_df.empty:
            return None
        row = bs_df.iloc[0]  # 最新年度
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)

    # ─── 第 0 项：审计意见（最高优先级） ───
    opinion = None
    if bs_df is not None and not bs_df.empty:
        opinion = str(bs_df.iloc[0].get("OPINION_TYPE", ""))
    if opinion and opinion != "nan" and "无保留" not in str(opinion):
        flags.append(("审计意见", 2, f"非标: {opinion}"))
        return {"risk_score": 10.0, "risk_flags": "审计非标", "risks": flags}
    if opinion and "强调事项" in str(opinion):
        flags.append(("审计意见", 1, f"带强调事项段: {opinion}"))

    # ─── ST 名称兜底 ───
    if stock_name and ("*ST" in stock_name or stock_name.startswith("ST")):
        flags.append(("已ST", 2, stock_name))
        return {"risk_score": 10.0, "risk_flags": "已ST", "risks": flags}

    if not period:
        return {"risk_score": 0.0, "risk_flags": "", "risks": []}

    # ─── 1. 连续亏损 ───
    profits = _fa_vals("归母净利润", 8)
    profits_clean = [p for p in profits if p is not None]
    if profits_clean:
        neg_count = sum(1 for p in profits_clean if p < 0)
        if neg_count >= 8:
            flags.append(("连续亏损", 2, f"近8季全负"))
        elif neg_count >= 3 and len(profits_clean) >= 4:
            recent4 = profits_clean[:4]
            if sum(1 for p in recent4 if p < 0) >= 3:
                flags.append(("连续亏损", 1, f"近4季{sum(1 for p in recent4 if p<0)}季负"))

    # ─── 2. 净资产转负 ───
    equity = _fa_val("股东权益合计(净资产)")
    if equity is not None and equity < 0:
        flags.append(("净资产转负", 2, f"净资产={equity/1e8:.1f}亿"))

    # ─── 3. 营收陷阱 ───
    revenue = _fa_val("营业总收入")
    net_profit = _fa_val("净利润")
    if revenue is not None and net_profit is not None:
        rev_yi = revenue / 1e8  # 转为亿
        if rev_yi < 1 and net_profit < 0:
            flags.append(("营收陷阱", 2, f"营收{rev_yi:.2f}亿,净利<0"))
        elif rev_yi < 3 and net_profit < 0:
            flags.append(("营收陷阱", 1, f"营收{rev_yi:.1f}亿,净利<0"))

    # ─── 4. 扣非背离 ───
    deducted = _fa_val("扣非净利润")
    if deducted is not None and net_profit is not None and net_profit > 0:
        ratio = deducted / net_profit
        if ratio < 0.3:
            flags.append(("扣非背离", 2, f"扣非/归母={ratio:.2f}"))
        elif ratio < 0.5:
            flags.append(("扣非背离", 1, f"扣非/归母={ratio:.2f}"))

    # ─── 5. 利润断崖 ───
    profit_growths = _fa_vals("归属母公司净利润增长率", 4)
    profit_growths_clean = [(i, g) for i, g in enumerate(profit_growths) if g is not None]
    if len(profit_growths_clean) >= 2:
        curr_g = profit_growths_clean[0][1]
        for idx, prev_g in profit_growths_clean[1:]:
            if prev_g > 0 and curr_g < 0:
                drop = prev_g - curr_g
                if drop > 50:
                    flags.append(("利润断崖", 2, f"增速{prev_g:.0f}%→{curr_g:.0f}%"))
                elif drop > 30:
                    flags.append(("利润断崖", 1, f"增速{prev_g:.0f}%→{curr_g:.0f}%"))
                break

    # ─── 6. 毛利率塌方 ───
    margins = _fa_vals("毛利率", 8)
    margins_clean = [(i, m) for i, m in enumerate(margins) if m is not None]
    if len(margins_clean) >= 4:
        recent_m = [m for _, m in margins_clean[:4]]
        # 检查是否连续下降
        consec_drop = 0
        for i in range(1, len(recent_m)):
            if recent_m[i] < recent_m[i-1]:
                consec_drop += 1
            else:
                break
        cumul_drop = recent_m[-1] - recent_m[0] if consec_drop > 0 else 0  # negative
        if consec_drop >= 4 and cumul_drop < -10:
            flags.append(("毛利率塌方", 2, f"连续4季降{cumul_drop:.0f}pp"))
        elif consec_drop >= 3 and cumul_drop < -5:
            flags.append(("毛利率塌方", 1, f"连续3季降{cumul_drop:.0f}pp"))

    # ─── 7. 增收不增利 ───
    rev_growth = _fa_val("营业总收入增长率")
    if rev_growth is not None and profit_growths_clean:
        pg = profit_growths_clean[0][1]
        if rev_growth > 0 and pg < -30:
            flags.append(("增收不增利", 2, f"营收+{rev_growth:.0f}%,净利{pg:.0f}%"))
        elif rev_growth > 0 and pg < -10:
            flags.append(("增收不增利", 1, f"营收+{rev_growth:.0f}%,净利{pg:.0f}%"))

    # ─── 8. 现金流枯竭 ───
    ocfs = _fa_vals("经营现金流量净额", 4)
    ocfs_clean = [(i, o) for i, o in enumerate(ocfs) if o is not None]
    if len(ocfs_clean) >= 2:
        neg_streak = 0
        worsening = False
        for i in range(len(ocfs_clean)):
            if ocfs_clean[i][1] < 0:
                neg_streak += 1
            else:
                break
        if neg_streak >= 2:
            # Check if worsening
            if neg_streak >= 2:
                v1 = ocfs_clean[0][1]
                v2 = ocfs_clean[1][1]
                worsening = v1 < v2  # more negative
            if worsening:
                flags.append(("现金流枯竭", 2, f"连续{neg_streak}季负且恶化"))
            else:
                flags.append(("现金流枯竭", 1, f"连续{neg_streak}季负"))

    # ─── 9. 应收异常 (balance sheet) ───
    rece = _bs_val("ACCOUNTS_RECE")
    rece_yoy = _bs_val("ACCOUNTS_RECE_YOY")
    total_assets = _bs_val("TOTAL_ASSETS")
    if rece_yoy is not None and rev_growth is not None:
        gap = rece_yoy - rev_growth
        if gap > 20:
            rece_ratio = rece / total_assets if rece and total_assets else 0
            if rece_ratio > 0.3:
                flags.append(("应收异常", 2, f"应收+{rece_yoy:.0f}%vs营收+{rev_growth:.0f}%,gap{gap:.0f}pp"))
            else:
                flags.append(("应收异常", 1, f"应收+{rece_yoy:.0f}%vs营收+{rev_growth:.0f}%,gap{gap:.0f}pp"))
        elif gap > 10:
            flags.append(("应收异常", 1, f"应收+{rece_yoy:.0f}%vs营收+{rev_growth:.0f}%,gap{gap:.0f}pp"))

    # ─── 10. 商誉暴雷 (balance sheet) ───
    goodwill = _bs_val("GOODWILL")
    if goodwill is not None and equity is not None and equity > 0:
        gw_ratio = goodwill / equity
        if gw_ratio > 0.5:
            flags.append(("商誉暴雷", 2, f"商誉/净资产={gw_ratio*100:.0f}%"))
        elif gw_ratio > 0.3:
            flags.append(("商誉暴雷", 1, f"商誉/净资产={gw_ratio*100:.0f}%"))

    # ─── 计分 ───
    score = 0.0
    desc_parts = []
    for item, level, detail in flags:
       if level == 2:
           score += 1.5
           desc_parts.append(f"🔴{item}")
       elif level == 1:
           score += 0.5
           desc_parts.append(f"🟡{item}")

    score = min(score, 10.0)

    return {
        "risk_score": round(score, 1),
        "risk_flags": " | ".join(desc_parts) if desc_parts else "",
        "risks": flags,
    }
