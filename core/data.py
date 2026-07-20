"""日线和信号数据加载。"""
from typing import Optional

import pandas as pd
from core.constants import DATA_DIR, SIGNALS_DIR


def load_daily(code: str) -> pd.DataFrame | None:
    """从 data/daily/{code}.parquet 加载日线数据。"""
    p = DATA_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def load_signals(code: str) -> pd.DataFrame | None:
    """从 data/signals/{code}.parquet 加载信号数据。"""
    p = SIGNALS_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def get_next_trading_day(date_str: str, daily: pd.DataFrame) -> Optional[pd.Timestamp]:
    """给定 date_str 后 daily 中第一个 > date_str 的交易日。"""
    sig_date = pd.Timestamp(date_str)
    mask = daily["date"] > sig_date
    future = daily.loc[mask, "date"]
    if future.empty:
        return None
    return future.iloc[0]


def get_price_at_date(date_ts: pd.Timestamp, daily: pd.DataFrame) -> Optional[float]:
    """获取指定日期的开盘价。"""
    row = daily[daily["date"] == date_ts]
    if row.empty:
        return None
    return float(row["open"].iloc[0])
