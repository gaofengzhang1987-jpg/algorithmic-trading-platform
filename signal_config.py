#!/usr/bin/env python3
"""czsc 信号配置 — 支持多级别 (日线/周线/15分钟)。

信号格式: {freq}_k2_k3_v1_v2_v3_score
"""

from czsc.traders.sig_parse import get_signals_config


def _make_signals(freq: str) -> dict:
    """为指定频率生成全套信号序列。freq: '日线' | '周线' | '15分钟'"""
    f = freq

    first_buy = [f'{f}_D1B_BUY1_一买_{n}笔_任意_0' for n in [5,7,9,11,13,15,17,19,21]]
    second_buy = [f'{f}_D1#SMA#21_BS2辅助V230320_二买_任意_任意_0']
    third_buy = [
        f'{f}_D1#SMA#34_BS3辅助V230318_三买_任意_任意_0',
        *[f'{f}_D1_三买辅助V230228_三买_{n}笔_任意_0' for n in [6,8,10,12,14]],
        f'{f}_D1#SMA#34_BS3辅助V230319_三买_均线底分_任意_0',
    ]
    first_sell = [f'{f}_D1B_SELL1_一卖_{n}笔_任意_0' for n in [5,7,9,11,13,15,17,19,21]]
    second_sell = [f'{f}_D1#SMA#21_BS2辅助V230320_二卖_任意_任意_0']
    third_sell = [
        f'{f}_D1#SMA#34_BS3辅助V230318_三卖_任意_任意_0',
        f'{f}_D1#SMA#34_BS3辅助V230319_三卖_均线新低_任意_0',
    ]
    macd = [
        f'{f}_D1MACD12#26#9_BS1辅助V230313_一买_金叉_任意_0',
        f'{f}_D1MACD12#26#9_BS1辅助V230313_一卖_死叉_任意_0',
    ]
    ma = [
        f'{f}_D1SMA#5_分类V221101_多头_向上_任意_0',
        f'{f}_D1SMA#20_分类V221101_多头_向上_任意_0',
    ]
    bar_sig = [f'{f}_D2N5T300_绝对动量V230227_强势_任意_任意_0']
    vol_sig = [f'{f}_D1K_量柱V221218_低量柱_6K_任意_0']

    buy = first_buy + second_buy + third_buy
    sell = first_sell + second_sell + third_sell
    all_sigs = buy + sell + macd + ma + bar_sig + vol_sig

    return {
        "FIRST_BUY": first_buy, "SECOND_BUY": second_buy, "THIRD_BUY": third_buy,
        "FIRST_SELL": first_sell, "SECOND_SELL": second_sell, "THIRD_SELL": third_sell,
        "MACD": macd, "MA": ma, "BAR": bar_sig, "VOL": vol_sig,
        "BUY": buy, "SELL": sell, "ALL": all_sigs,
    }


# ——— 日线 (保持向后兼容) ———
_d = _make_signals('日线')
FIRST_BUY_SIGNALS = _d["FIRST_BUY"]
SECOND_BUY_SIGNALS = _d["SECOND_BUY"]
THIRD_BUY_SIGNALS = _d["THIRD_BUY"]
FIRST_SELL_SIGNALS = _d["FIRST_SELL"]
SECOND_SELL_SIGNALS = _d["SECOND_SELL"]
THIRD_SELL_SIGNALS = _d["THIRD_SELL"]
MACD_SIGNALS = _d["MACD"]
MA_SIGNALS = _d["MA"]
BAR_SIGNALS = _d["BAR"]
VOL_SIGNALS = _d["VOL"]
BUY_SIGNALS = _d["BUY"]
SELL_SIGNALS = _d["SELL"]
ALL_SIGNALS = _d["ALL"]

# ——— 多级别信号字典 ———
SIGNALS_BY_FREQ = {
    '日线': _d,
    '周线': _make_signals('周线'),
    '15分钟': _make_signals('15分钟'),
    '30分钟': _make_signals('30分钟'),
}


def get_config(signals: list[str] | None = None, freq: str = '日线') -> list[dict]:
    """获取 czsc 信号配置列表。freq: '日线'|'周线'|'15分钟'"""
    seq = signals or SIGNALS_BY_FREQ[freq]["ALL"]
    return get_signals_config(seq)


if __name__ == '__main__':
    for f in ['日线', '周线', '15分钟']:
        cfgs = get_config(freq=f)
        print(f'{f}: {len(cfgs)} 个信号函数')
