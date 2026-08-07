"""基本面分析模块 — 行业路由 + 三框架评分 + 风险检测 + 批量拉取。

使用方式：
    from fundamental import run_overlay

    l4_df = run()  # zone4_regime
    l4_df = run_overlay(l4_df)  # 叠加基本面评分 + 风险检测
"""

import json
import logging
from pathlib import Path

import pandas as pd

from .puller import pull_batch, pull_balance_sheet_batch
from .scorer import score_batch, detect_risks

logger = logging.getLogger("fundamental")

BASE = Path(__file__).parent.parent
MAP_PATH = Path(__file__).parent / "framework_map.json"
INDUSTRY_DB = BASE / "data" / "industry_classification.parquet"


def _load_framework_map() -> dict[str, str]:
    """加载行业 → 框架路由表。"""
    with open(MAP_PATH, "r") as f:
        config = json.load(f)

    industry_map = {}
    for fw in ["tech", "finance"]:
        for ind in config.get(fw, []):
            industry_map[ind] = fw
    return industry_map


def run_overlay(l4_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """在 L4 输出上叠加基本面评分 + 风险检测。

    Args:
        l4_df: L4Ranker 输出（至少含 code 列）
        force_refresh: 是否强制重新拉取数据

    Returns:
        原 DataFrame + fundamental_score / risk_score / risk_flags 等
    """
    if l4_df.empty:
        return l4_df

    codes = l4_df["code"].tolist()
    logger.info("基本面分析: %d 只股票", len(codes))

    # 1. 查行业分类 → 确定框架 + 股票名称
    name_map = {}
    if INDUSTRY_DB.exists():
        ind_df = pd.read_parquet(INDUSTRY_DB)
        ind_map = ind_df.drop_duplicates(subset=["code"]).set_index("code")["industry"].to_dict()
        name_map = ind_df.drop_duplicates(subset=["code"]).set_index("code")["name"].to_dict()
    else:
        ind_map = {}

    fw_map = _load_framework_map()
    code_fw = {}
    ambiguous = []

    for code in codes:
        industry = ind_map.get(code, "")
        if industry and industry in fw_map:
            code_fw[code] = fw_map[industry]
        else:
            code_fw[code] = "standard"
            if industry:
                ambiguous.append(code)

    tech_n = sum(1 for f in code_fw.values() if f == "tech")
    fin_n = sum(1 for f in code_fw.values() if f == "finance")
    std_n = sum(1 for f in code_fw.values() if f == "standard")
    logger.info("框架分布: tech=%d finance=%d standard=%d", tech_n, fin_n, std_n)
    if ambiguous:
        logger.info("不在 tech/finance 列表 (归入 standard): %d 只 — %s",
                    len(ambiguous), ", ".join(ambiguous[:10]))

    # 2. 批量拉取财务摘要数据
    data_map = pull_batch(codes, force_refresh=force_refresh)

    if not data_map:
        logger.warning("基本面分析: 无有效财务数据")
        return l4_df

    # 3. 基本面评分
    score_df = score_batch(data_map, code_fw)

    if score_df.empty:
        logger.warning("基本面分析: 评分全部失败")
        return l4_df

    # 4. 批量拉取资产负债表（用于风险检测）
    bs_map = pull_balance_sheet_batch(codes, force_refresh=force_refresh)
    logger.info("资产负债表: %d/%d 只有效", len(bs_map), len(codes))

    # 5. 风险检测
    risk_rows = []
    for code in codes:
        fa_df = data_map.get(code)
        bs_df = bs_map.get(code)
        sname = name_map.get(code, "")

        if fa_df is None:
            risk_rows.append({"code": code, "risk_score": -1.0, "risk_flags": "数据缺失"})
        else:
            result = detect_risks(fa_df, bs_df, sname)
            risk_rows.append({
                "code": code,
                "risk_score": result["risk_score"],
                "risk_flags": result["risk_flags"],
            })

    risk_df = pd.DataFrame(risk_rows)

    # 6. 合并回 L4
    merge_cols = ["code"] + [c for c in score_df.columns if c != "code"]
    l4_df = l4_df.merge(score_df[merge_cols], on="code", how="left")
    l4_df = l4_df.merge(risk_df, on="code", how="left")

    l4_df["fundamental_score"] = l4_df["fundamental_score"].fillna(-1)
    l4_df["risk_score"] = l4_df["risk_score"].fillna(-1)

    n_missing_score = (l4_df["fundamental_score"] == -1).sum()
    n_missing_risk = (l4_df["risk_score"] == -1).sum()
    if n_missing_score > 0:
        missing_codes = l4_df[l4_df["fundamental_score"] == -1]["code"].tolist()
        logger.warning("基本面评分缺失: %d 只 — %s", n_missing_score, ", ".join(missing_codes))
    if n_missing_risk > 0:
        missing_risk_codes = l4_df[l4_df["risk_score"] == -1]["code"].tolist()
        logger.warning("风险检测缺失: %d 只 — %s", n_missing_risk, ", ".join(missing_risk_codes))
    n_risky = (l4_df["risk_score"] >= 5).sum()
    n_safe = (l4_df["risk_score"] <= 1.0).sum()
    logger.info("基本面完成: %d评分 %d风险检测 — 缺评分:%d 缺风险:%d 高危≥5:%d 安全≤1:%d",
                len(score_df), len(risk_df), n_missing_score, n_missing_risk, n_risky, n_safe)

    return l4_df
