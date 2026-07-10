#!/usr/bin/env python3
"""czsc 策略定义 — 继承 CzscStrategyBase，用于选股和回测。

策略逻辑：一买信号开多 / 一卖信号或 MA 转空头平多。
"""

import pandas as pd
from typing import List
from czsc import CzscStrategyBase
from czsc.objects import Position, Event, Factor, Operate, Signal


class ChanStrategy(CzscStrategyBase):
    """基于缠论一买的择时策略。

    开仓条件（任意满足其一）：
      1. 一买信号触发（5笔/7笔/9笔）
      2. MACD 金叉买点

    平仓条件（任意满足其一）：
      1. 一卖信号触发
      2. MA5 转空头
      3. 止损（-10%）
    """

    @property
    def positions(self) -> List[Position]:
        pos_name = f"{self.symbol}_日线_一买策略"

        return [
            Position(
                symbol=self.symbol,
                name=pos_name,
                opens=[
                    Event(
                        name="一买开仓",
                        operate=Operate.LE,
                        factors=[
                            Factor(
                                name="一买信号组",
                                signals_all=[],  # Signal 条件在主列表匹配
                            )
                        ],
                        # 匹配任意一个一买信号列
                        signals_any=[Signal(k3="BUY1")],
                    ),
                ],
                exits=[
                    Event(
                        name="一卖平仓",
                        operate=Operate.LX,
                        factors=[Factor(name="一卖信号", signals_all=[])],
                        signals_any=[Signal(k3="SELL1")],
                    ),
                    Event(
                        name="MA5空头平仓",
                        operate=Operate.LX,
                        factors=[Factor(name="MA转空", signals_all=[])],
                        signals_any=[Signal(k3="分类V221101", v1="空头")],
                    ),
                ],
                interval=3600 * 24,     # 日线级别
                timeout=20,             # 最大持仓 20 个交易日
                stop_loss=100,          # 止损 -10%（100 基点 = 10%）
                T0=False,
            )
        ]


def strategy_example():
    """展示策略定义 - 打印关键属性。"""
    strategy = ChanStrategy(symbol="000001")
    print(f"标的: {strategy.symbol}")
    print(f"K线周期: {strategy.freqs}")
    print(f"信号数: {len(strategy.unique_signals)}")
    print(f"持仓策略数: {len(strategy.positions)}")
    for pos in strategy.positions:
        print(f"  {pos.name}: 开仓 {len(pos.opens)} 个事件, 平仓 {len(pos.exits)} 个事件")


if __name__ == "__main__":
    strategy_example()
