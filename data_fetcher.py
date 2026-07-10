#!/usr/bin/env python3
"""全市场 A 股日线数据拉取模块。

数据源：Sina（akshare.stock_zh_a_daily），带前复权。
缓存：Parquet 格式存于 data/daily/，按股票代码分文件。
支持增量更新，默认拉取近 2 年数据。
"""

import os
import sys
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import akshare as ak

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_fetcher")

# ——— 配置 ———
DATA_DIR = Path(__file__).parent / "data" / "daily"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "20200101"          # 最早数据起始日
LOOKBACK_DAYS = 300              # 最少需要的交易日数
MAX_WORKERS = 8                  # 并发请求数
REQUEST_DELAY = 0.15             # 单股请求间隔（秒）
RETRY_COUNT = 3                  # 失败重试次数


def get_stock_list() -> pd.DataFrame:
    """获取全市场 A 股代码与名称列表。"""
    df = ak.stock_info_a_code_name()
    df.columns = ["code", "name"]
    # 只保留 6 位数字代码（过滤退市/特殊代码）
    df = df[df["code"].str.match(r"^\d{6}$")]
    # 过滤 ST、退市、北交所
    mask_st = df["name"].str.contains("ST|退", na=False)
    mask_bj = df["code"].str.startswith(("8", "9"))
    df = df[~(mask_st | mask_bj)].copy()
    logger.info("获取到 %d 只有效 A 股", len(df))
    return df


def stock_code_to_sina(code: str) -> str:
    """将纯数字代码转为 Sina 格式 (sz000001 / sh600000)。"""
    if code.startswith(("0", "3")):
        return f"sz{code}"
    else:
        return f"sh{code}"


def fetch_single_stock(code: str, start_date: str = START_DATE) -> pd.DataFrame | None:
    """拉取单只股票日线数据，失败返回 None。"""
    sina_code = stock_code_to_sina(code)
    for attempt in range(RETRY_COUNT):
        try:
            df = ak.stock_zh_a_daily(
                symbol=sina_code,
                start_date=start_date,
                end_date=datetime.now().strftime("%Y%m%d"),
                adjust="qfq",
            )
            time.sleep(REQUEST_DELAY)
            if df is None or df.empty:
                return None
            # 标准化列名
            df.rename(columns={
                "date": "date", "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume",
                "amount": "amount",
            }, inplace=True)
            df["code"] = code
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception as e:
            logger.debug("拉取 %s 第 %d 次失败: %s", code, attempt + 1, str(e)[:80])
            time.sleep(REQUEST_DELAY * (attempt + 1))
    logger.warning("拉取 %s 失败（已重试 %d 次），跳过", code, RETRY_COUNT)
    return None


def cache_path(code: str) -> Path:
    return DATA_DIR / f"{code}.parquet"


def load_cached(code: str) -> pd.DataFrame | None:
    """从缓存读取股票数据。"""
    p = cache_path(code)
    if p.exists():
        return pd.read_parquet(p)
    return None


def save_cache(code: str, df: pd.DataFrame):
    """保存股票数据到 Parquet 缓存。"""
    p = cache_path(code)
    df.to_parquet(p, index=False)


def is_stale(code: str) -> bool:
    """判断缓存是否过期（最新日 < 昨天 或 不足 LOOKBACK_DAYS 天）。"""
    df = load_cached(code)
    if df is None:
        return True
    today = pd.Timestamp.now().normalize()
    yesterday = today - pd.Timedelta(days=1)
    if df["date"].max() < yesterday:
        return True
    if len(df) < LOOKBACK_DAYS:
        return True
    return False


def fetch_all(codes: list[str], force: bool = False) -> dict[str, pd.DataFrame]:
    """并发拉取全量数据。

    Args:
        codes: 股票代码列表
        force: True 时忽略缓存强制刷新

    Returns:
        {code: DataFrame} 成功拉取的股票数据
    """
    # 确定需要拉取的股票
    to_fetch = []
    cached_count = 0
    for code in codes:
        if force or is_stale(code):
            to_fetch.append(code)
        else:
            cached_count += 1

    logger.info("缓存有效 %d 只，需拉取 %d 只", cached_count, len(to_fetch))

    results: dict[str, pd.DataFrame] = {}

    # 先加载有效缓存
    if not force:
        for code in codes:
            if code not in to_fetch:
                df = load_cached(code)
                if df is not None:
                    results[code] = df

    # 并发拉取
    fetched = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_single_stock, code): code for code in to_fetch}
        for future in as_completed(futures):
            code = futures[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    save_cache(code, df)
                    results[code] = df
                    fetched += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.debug("异常 %s: %s", code, str(e)[:80])

            if (fetched + failed) % 50 == 0:
                logger.info("进度: 已成功 %d, 失败 %d, 总计 %d", fetched, failed, len(to_fetch))

    logger.info("拉取完成: 成功 %d, 失败 %d, 缓存 %d, 最终 %d 只",
                fetched, failed, cached_count, len(results))
    return results


def main():
    """命令行入口：拉取全市场日线数据。

    Usage:
        python3 data_fetcher.py              # 增量更新
        python3 data_fetcher.py --force      # 强制全量刷新
    """
    force = "--force" in sys.argv

    logger.info("=== 全市场数据拉取开始 ===")
    logger.info("数据源: Sina (akshare), 起始日: %s", START_DATE)

    t0 = time.time()
    stocks = get_stock_list()
    codes = stocks["code"].tolist()

    results = fetch_all(codes, force=force)

    elapsed = time.time() - t0
    logger.info("=== 完成 === 耗时 %.1f 秒, 覆盖 %d/%d 只股票", elapsed, len(results), len(codes))

    # 输出摘要
    cache_files = list(DATA_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
    logger.info("缓存文件: %d 个, 总大小: %.1f MB", len(cache_files), total_size)


if __name__ == "__main__":
    main()
