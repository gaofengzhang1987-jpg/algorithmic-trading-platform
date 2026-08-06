"""基本面数据拉取模块 — akshare.stock_financial_abstract，带本地缓存。

缓存目录：data/fundamental/
缓存时效：7 天（财务数据每季度更新，7 天足够覆盖财报发布窗口）
"""

import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import akshare as ak

logger = logging.getLogger("fundamental.puller")

BASE = Path(__file__).parent.parent
CACHE_DIR = BASE / "data" / "fundamental"
CACHE_TTL_DAYS = 30
BS_CACHE_DIR = BASE / "data" / "fundamental" / "balance_sheet"


def _cache_path(code: str) -> Path:
    return CACHE_DIR / f"{code}.parquet"


def _bs_cache_path(code: str) -> Path:
    return BS_CACHE_DIR / f"{code}.parquet"


def _code_to_ts_code(code: str) -> str:
    """将 6 位代码转为 ts_code 格式 (000001 → 000001.SZ)。"""
    if "." in code:
        return code
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{code}.SZ"
    if code.startswith(("600", "601", "603", "605", "688")):
        return f"{code}.SH"
    if code.startswith(("8", "9")):
        return f"{code}.BJ"
    return f"{code}.SZ"  # fallback


def _is_cache_valid(code: str) -> bool:
    p = _cache_path(code)
    if not p.exists():
        return False
    mtime = datetime.fromtimestamp(p.stat().st_mtime)
    return (datetime.now() - mtime).days < CACHE_TTL_DAYS


def pull_single(code: str, force_refresh: bool = False) -> pd.DataFrame | None:
    """拉取单只股票的财务摘要数据。

    Returns:
        DataFrame: 80 行 × N 列（选项, 指标, 各报告期）, 或 None（拉取失败）
    """
    p = _cache_path(code)
    if not force_refresh and _is_cache_valid(code):
        return pd.read_parquet(p)

    try:
        df = ak.stock_financial_abstract(symbol=code)
        time.sleep(0.3)  # akshare 请求间隔
    except Exception as e:
        logger.warning("拉取 %s 失败: %s", code, e)
        return None

    if df is None or df.empty:
        logger.warning("拉取 %s 返回空数据", code)
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def pull_batch(codes: list[str], force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """批量拉取财务数据。

    Returns:
        dict: code → DataFrame（拉取失败的不在返回中）
    """
    results = {}
    failed = []

    for i, code in enumerate(codes):
        df = pull_single(code, force_refresh=force_refresh)
        if df is not None:
            results[code] = df
        else:
            failed.append(code)

        if (i + 1) % 20 == 0:
            logger.info("拉取进度: %d/%d (失败: %d)", i + 1, len(codes), len(failed))

    if failed:
        logger.warning("拉取失败: %d 只 — %s", len(failed), ", ".join(failed[:10]))

    return results


def pull_balance_sheet(code: str, force_refresh: bool = False):
    """拉取单只股票的年度资产负债表（东方财富）。

    Returns:
        DataFrame: 含 OPINION_TYPE / ACCOUNTS_RECE / GOODWILL / INVENTORY / TOTAL_ASSETS 等，或 None
    """
    p = _bs_cache_path(code)
    BS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_refresh and p.exists():
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        if (datetime.now() - mtime).days < CACHE_TTL_DAYS:
            return pd.read_parquet(p)

    ts_code = _code_to_ts_code(code)
    try:
        import akshare as ak
        import time as _time
        df = ak.stock_balance_sheet_by_yearly_em(symbol=ts_code)
        _time.sleep(0.3)
    except Exception as e:
        logger.warning("资产负债表 %s 拉取失败: %s", code, e)
        return None

    if df is None or df.empty:
        return None

    df.to_parquet(p, index=False)
    return df


def pull_balance_sheet_batch(codes: list[str], force_refresh: bool = False) -> dict[str, "pd.DataFrame"]:
    """批量拉取资产负债表。"""
    results = {}
    failed = []
    for i, code in enumerate(codes):
        df = pull_balance_sheet(code, force_refresh=force_refresh)
        if df is not None:
            results[code] = df
        else:
            failed.append(code)
        if (i + 1) % 20 == 0:
            logger.info("资产负债表进度: %d/%d (失败: %d)", i + 1, len(codes), len(failed))
    if failed:
        logger.warning("资产负债表拉取失败: %d 只 — %s", len(failed), ", ".join(failed[:10]))
    return results
