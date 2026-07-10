#!/usr/bin/env python3
"""预计算每只股票的结构上下文 — 按 BI 维度存储。
存到 data/reference/{code}.parquet。

输出列（每行 = 一根 BI）:
  sdt, edt, direction, high, low, power,
  fx_a_mark, fx_a_high, fx_a_low,
  fx_b_mark, fx_b_high, fx_b_low,
  pivot_gg, pivot_dd, pivot_dir, pivot_id
"""

import logging, time
from pathlib import Path
import pandas as pd
from czsc import CZSC, RawBar, Freq
from czsc.objects import Direction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("precompute")

BASE = Path(__file__).parent
DAILY = BASE / "data" / "daily"
OUT = BASE / "data" / "reference"
OUT.mkdir(parents=True, exist_ok=True)


def _extract_pivots(bi_list):
    """从 bi_list 提取中枢。返回 {bi_index: {pivot_id, gg, dd, dir}}。

    算法: 滑动窗口, 每 3 笔检查重叠区间。离开段方向判定中枢方向。
    """
    bi_to_pivot = {}
    i = 0
    pivot_id = 0

    while i < len(bi_list) - 2:
        bi_a, bi_b, bi_c = bi_list[i], bi_list[i+1], bi_list[i+2]
        overlap_high = min(bi_a.high, bi_b.high, bi_c.high)
        overlap_low  = max(bi_a.low, bi_b.low, bi_c.low)

        if overlap_high > overlap_low:
            pivot_bis = [bi_a, bi_b, bi_c]
            zg, zd = overlap_high, overlap_low
            gg, dd = overlap_high, overlap_low

            j = i + 3
            while j < len(bi_list):
                bi_next = bi_list[j]
                if bi_next.high >= zd and bi_next.low <= zg:
                    pivot_bis.append(bi_next)
                    gg = max(gg, bi_next.high)
                    dd = min(dd, bi_next.low)
                    j += 1
                else:
                    break

            leave_bi = bi_list[j] if j < len(bi_list) else None
            if leave_bi and leave_bi.direction == Direction.Up:
                direction = "上涨"
            elif leave_bi and leave_bi.direction == Direction.Down:
                direction = "下跌"
            else:
                direction = "上涨" if pivot_bis[0].direction == Direction.Up else "下跌"

            for bi in pivot_bis:
                idx = bi_list.index(bi)
                bi_to_pivot[idx] = {
                    "pivot_id": pivot_id,
                    "gg": float(gg),
                    "dd": float(dd),
                    "dir": direction,
                }

            pivot_id += 1
            i = j
        else:
            i += 1

    return bi_to_pivot


def precompute_structure(code):
    """按 BI 维度存储结构上下文, 每行一根 BI."""
    dp = DAILY / f"{code}.parquet"
    if not dp.exists():
        return None
    df = pd.read_parquet(dp)
    if len(df) < 500:
        return None

    bars = [RawBar(symbol=code, id=j+1, dt=r["date"].to_pydatetime(),
                   freq=Freq.D, open=r["open"], close=r["close"],
                   high=r["high"], low=r["low"],
                   vol=r.get("volume", 0), amount=r.get("amount", 0))
            for j, (_, r) in enumerate(df.iterrows())]

    try:
        c = CZSC(bars, max_bi_num=100)
    except Exception as e:
        logger.warning("%s: CZSC 失败: %s", code, e)
        return None

    if not c.bi_list:
        return None

    bi_to_pivot = _extract_pivots(c.bi_list)

    rows = []
    for bi in c.bi_list:
        idx = c.bi_list.index(bi)
        pv = bi_to_pivot.get(idx, {})

        row = {
            "sdt": bi.sdt,
            "edt": bi.edt,
            "direction": str(bi.direction),
            "high": float(bi.high),
            "low": float(bi.low),
            "power": float(bi.power),
            "fx_a_mark": str(bi.fx_a.mark) if hasattr(bi.fx_a, 'mark') else "",
            "fx_a_high": float(bi.fx_a.high) if hasattr(bi.fx_a, 'high') else 0.0,
            "fx_a_low": float(bi.fx_a.low) if hasattr(bi.fx_a, 'low') else 0.0,
            "fx_b_mark": str(bi.fx_b.mark) if hasattr(bi.fx_b, 'mark') else "",
            "fx_b_high": float(bi.fx_b.high) if hasattr(bi.fx_b, 'high') else 0.0,
            "fx_b_low": float(bi.fx_b.low) if hasattr(bi.fx_b, 'low') else 0.0,
            "pivot_gg": pv.get("gg", 0.0),
            "pivot_dd": pv.get("dd", 0.0),
            "pivot_dir": pv.get("dir", ""),
            "pivot_id": pv.get("pivot_id", -1),
            "code": code,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    codes = sorted(p.stem for p in DAILY.glob("*.parquet"))
    logger.info("预计算 %d 只股票的结构数据", len(codes))
    t0 = time.time()

    for i, code in enumerate(codes):
        r = precompute_structure(code)
        if r is not None:
            out_path = OUT / f"{code}.parquet"
            r.to_parquet(out_path, index=False)
        if (i+1) % 200 == 0:
            logger.info("进度: %d/%d (%.0fs)", i+1, len(codes), time.time()-t0)

    elapsed = time.time() - t0
    count = len(list(OUT.glob("*.parquet")))
    logger.info("完成: %d 只股票 (%.0fs)", count, elapsed)


if __name__ == "__main__":
    main()
