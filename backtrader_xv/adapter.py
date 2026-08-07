"""parquet 日线数据 → Backtrader PandasData feed 适配器。"""
import pandas as pd
import backtrader as bt
from pathlib import Path


def parquet_to_feed(code, data_dir=None):
    """将单只股票的日线 parquet 转为 Backtrader PandasData feed。

    Args:
        code: 股票代码（如 "000001"）
        data_dir: 日线目录，默认 core.constants.DATA_DIR

    Returns:
        bt.feeds.PandasData 实例
    """
    if data_dir is None:
        from core.constants import DATA_DIR
        data_dir = DATA_DIR

    df = pd.read_parquet(Path(data_dir) / f"{code}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["openinterest"] = 0

    # Backtrader 要求 datetime 为 index
    df.set_index("date", inplace=True)
    # 列名映射: Backtrader 期望 open/high/low/close/volume/openinterest
    df.rename(columns={"vol": "volume"}, inplace=True)

    return bt.feeds.PandasData(dataname=df, timeframe=bt.TimeFrame.Days)


def parquet_to_bt_cerebro(code, data_dir=None, cash=100000, commission=0.001):
    """为单只股票创建 Backtrader Cerebro 引擎。

    Args:
        code: 股票代码
        data_dir: 日线目录
        cash: 初始资金
        commission: 佣金率

    Returns:
        bt.Cerebro 实例
    """
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    data = parquet_to_feed(code, data_dir)
    cerebro.adddata(data)
    return cerebro
