#!/usr/bin/env python3
"""L1 → 30min 信号桥接：只对 L1 候选代码补缺 30min CZSC 信号。

逻辑：
  1. 读 data/zones/L1_deposition.parquet → 获取候选代码集
  2. 遍历 data/signals_30min/ → 找出缺失或陈旧（< 日线最新信号日）的代码
  3. 增量模式（--max-bars N）生成 CZSC 信号

用法:
    python3 l1_30min_bridge.py                     # 默认 800 bar 增量
    python3 l1_30min_bridge.py --max-bars 500      # 自定义增量条数
    python3 l1_30min_bridge.py --full               # 全量（禁用增量）
"""

import argparse, sys, time
from pathlib import Path
import pandas as pd
from czsc import RawBar, Freq, generate_czsc_signals
from signal_config import get_config
import logging
logger = logging.getLogger(__name__)  # L1 30分钟桥接

BASE = Path(__file__).parent
L1_PATH = BASE / "data" / "zones" / "L1_deposition.parquet"
MIN30_BAR = BASE / "data" / "min30"
SIG_DIR = BASE / "data" / "signals_30min"


def main():
    parser = argparse.ArgumentParser(description="L1 → 30min 信号桥接")
    parser.add_argument("--max-bars", type=int, default=800, help="增量 bar 数 (默认 800, 0=全量)")
    parser.add_argument("--full", action="store_true", help="全量模式（等同于 --max-bars 0）")
    args = parser.parse_args()

    max_bars = 0 if args.full else args.max_bars

    # 1. 读取 L1 候选代码
    if not L1_PATH.exists():
        print("[bridge] L1_deposition.parquet 不存在")
        return
    l1 = pd.read_parquet(L1_PATH)
    l1_codes = set(l1["代码"].unique())
    print(f"[bridge] L1 候选: {len(l1_codes)} 代码")

    # 2. 找出需要补缺的代码
    SIG_DIR.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in SIG_DIR.glob("*.parquet")}
    missing = l1_codes - existing
    stale = set()
    # 动态确定参考日期：取日线信号目录中最新的 dt 作为陈旧判断基准
    daily_sig_dir = BASE / "data" / "signals"
    try:
        latest_dates = []
        for sig_f in sorted(daily_sig_dir.glob("*.parquet"))[-200:]:  # 取样 200 只即可
            df = pd.read_parquet(sig_f, columns=["dt"])
            latest_dates.append(pd.Timestamp(df["dt"].max()))
        ref_date = max(latest_dates) if latest_dates else pd.Timestamp.now()
    except Exception:
        ref_date = pd.Timestamp.now()
    logger.info("参考日期: %s", ref_date.date())

    for c in l1_codes & existing:
        try:
            df = pd.read_parquet(SIG_DIR / f"{c}.parquet")
            if pd.Timestamp(df["dt"].max()) < ref_date:
                stale.add(c)
        except Exception:
            stale.add(c)

    todo = sorted(missing | stale)
    print(f"[bridge] 缺信号: {len(missing)}  陈旧(<{ref_date.date()}): {len(stale)}  合计需处理: {len(todo)}")

    if not todo:
        print("[bridge] 无需处理，退出")
        return

    # 3. CZSC 配置
    sig_config = get_config(freq="30分钟")

    # 4. 逐只生成
    mode = "增量" if max_bars > 0 else "全量"
    print(f"[bridge] {mode}模式 (max_bars={max_bars or '∞'}), 开始处理 {len(todo)} 只")

    success, failed, skipped = 0, 0, 0
    t0 = time.time()

    for i, code in enumerate(todo):
        bar_path = MIN30_BAR / f"{code}.parquet"
        if not bar_path.exists():
            print(f"  [{i+1}/{len(todo)}] {code} — 无30min bar 数据")
            skipped += 1
            continue

        try:
            df = pd.read_parquet(bar_path).sort_values("date")
            if len(df) < 200:
                print(f"  [{i+1}/{len(todo)}] {code} — bar 不足 ({len(df)}<200)")
                skipped += 1
                continue

            if max_bars > 0 and len(df) > max_bars:
                df = df.tail(max_bars)

            bars = [
                RawBar(symbol=code, id=j + 1, dt=r["date"].to_pydatetime(),
                       freq=Freq.F30, open=r["open"], close=r["close"],
                       high=r["high"], low=r["low"],
                       vol=float(r.get("volume", 0)), amount=float(r.get("amount", 0)))
                for j, (_, r) in enumerate(df.iterrows())
            ]

            sdt = str(df["date"].iloc[0].date()).replace("-", "")
            sigs = generate_czsc_signals(
                bars, signals_config=sig_config, sdt=sdt,
                init_n=min(500, len(bars)), df=True,
                tqdm_kwargs={"disable": True},
            )

            # 删除旧文件（如有）
            old_sig = SIG_DIR / f"{code}.parquet"
            if old_sig.exists():
                old_sig.unlink()

            if sigs is not None and not sigs.empty:
                sigs = sigs.drop(columns=[c for c in ["freq", "cache"] if c in sigs.columns])
                sigs.to_parquet(old_sig, index=False)
                success += 1
                dur = time.time() - t0
                eta = (dur / (i + 1)) * (len(todo) - i - 1) if i < len(todo) - 1 else 0
                print(f"  [{i+1}/{len(todo)}] {code} ✓ ({len(sigs)} 行, {len(df)} bar, ETA {eta:.0f}s)")
            else:
                failed += 1
                print(f"  [{i+1}/{len(todo)}] {code} ✗ CZSC 返回空")
        except Exception as e:
            failed += 1
            print(f"  [{i+1}/{len(todo)}] {code} ✗ {type(e).__name__}: {e}")

    elapsed = time.time() - t0
    print(f"\n[bridge] 完成: success={success} failed={failed} skipped={skipped}, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
