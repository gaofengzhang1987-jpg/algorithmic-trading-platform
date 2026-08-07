"""基于 CZSC 信号的 Backtrader 策略。"""
import backtrader as bt
import pandas as pd
from pathlib import Path


class CZSCSignalStrategy(bt.Strategy):
    params = (
        ("stop_pct", 0.08),
        ("max_hold_days", 99999),
        ("code", ""),
    )

    def __init__(self):
        self.buy_dates = set()
        self.sell_dates = set()
        self._load_signals()
        self.entry_bar = -1
        self.entry_price = 0.0

    def _load_signals(self):
        code = self.p.code or getattr(self.datas[0], '_name', 'unknown')
        if code == 'unknown':
            return

        sig_path = Path('/Users/hz/Desktop/Algorithmic Trading Platform/data/signals') / f"{code}.parquet"
        if not sig_path.exists():
            return

        sig = pd.read_parquet(sig_path)
        if "dt" not in sig.columns:
            return

        sig["dt"] = pd.to_datetime(sig["dt"])
        for col in sig.columns:
            if "BUY" in col.upper():
                for _, row in sig.iterrows():
                    val = str(row[col])
                    if val != "其他_任意_任意_0" and any(k in val for k in ["一买", "二买", "三买"]):
                        self.buy_dates.add(row["dt"].date())
            if "SELL" in col.upper():
                for _, row in sig.iterrows():
                    val = str(row[col])
                    if val != "其他_任意_任意_0" and "卖" in val:
                        self.sell_dates.add(row["dt"].date())

    def next(self):
        bar_dt = bt.num2date(self.datas[0].datetime[0])
        bar_date = bar_dt.date()
        close = self.data.close[0]

        if not self.position:
            if bar_date in self.buy_dates:
                size = int(self.broker.getcash() * 0.95 / close / 100) * 100
                if size > 0:
                    self.buy(size=size)
                    self.entry_bar = len(self)
                    self.entry_price = close
        else:
            # 止损
            if close < self.entry_price * (1 - self.p.stop_pct):
                self.close()
            # 卖点
            elif bar_date in self.sell_dates:
                self.close()
            # 超时
            elif len(self) - self.entry_bar > self.p.max_hold_days:
                self.close()
