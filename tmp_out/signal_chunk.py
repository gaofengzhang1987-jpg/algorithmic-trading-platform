#!/usr/bin/env python3
"""CZSC 信号分片处理。被 l1_update.py 以 nohup 方式调用。

用法: python3 tmp_out/signal_chunk.py --chunk 0/4
"""
import sys, time, json, argparse
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from czsc import RawBar, Freq, generate_czsc_signals
import signal_config

DAILY = BASE / "data" / "daily"
SIG_DIR = BASE / "data" / "signals"
PROGRESS_DIR = BASE / "tmp_out"
MIN_BARS = 120
INIT_N = 300


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", required=True, help="格式: 当前/总数，如 0/4")
    args = parser.parse_args()

    parts = args.chunk.split("/")
    chunk_idx = int(parts[0])
    total_chunks = int(parts[1])

    codes = sorted(p.stem for p in DAILY.glob("*.parquet"))
    my_codes = codes[chunk_idx::total_chunks]

    sig_config = signal_config.get_config(freq="日线")
    progress_file = PROGRESS_DIR / f"signal_progress_chunk_{chunk_idx}.txt"
    result_file = PROGRESS_DIR / f"chunk_{chunk_idx}_result.json"

    success, failed, skipped = 0, 0, 0
    t0 = time.time()

    for i, code in enumerate(my_codes):
        try:
            df = pd.read_parquet(DAILY / f"{code}.parquet")
            if len(df) < MIN_BARS:
                skipped += 1
                continue

            df = df.sort_values("date")
            bars = [
                RawBar(
                    symbol=code, id=j + 1, dt=r["date"].to_pydatetime(),
                    freq=Freq.D, open=r["open"], close=r["close"],
                    high=r["high"], low=r["low"],
                    vol=r.get("volume", 0), amount=r.get("amount", 0),
                )
                for j, (_, r) in enumerate(df.iterrows())
            ]

            sigs = generate_czsc_signals(
                bars, signals_config=sig_config,
                sdt="20200101", init_n=min(INIT_N, len(bars)), df=True,
            )
            if sigs is not None and not sigs.empty:
                sigs = sigs.drop(columns=[c for c in ["freq", "cache"] if c in sigs.columns])
                SIG_DIR.mkdir(parents=True, exist_ok=True)
                sigs.to_parquet(SIG_DIR / f"{code}.parquet", index=False)
                success += 1
        except Exception:
            failed += 1

        if (i + 1) % 100 == 0 or i == len(my_codes) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = len(my_codes) - i - 1
            eta = remaining / rate if rate > 0 else 0
            msg = (
                f"[chunk {chunk_idx}/{total_chunks}] {i + 1}/{len(my_codes)} "
                f"成功{success} 失败{failed} 跳过{skipped} "
                f"{rate:.1f}只/s ETA {eta:.0f}s"
            )
            progress_file.write_text(msg)
            print(msg, flush=True)

    result = {
        "success": success, "failed": failed, "skipped": skipped,
        "elapsed": time.time() - t0, "chunk": chunk_idx, "total": total_chunks,
    }
    result_file.write_text(json.dumps(result))
    print(f"[chunk {chunk_idx}/{total_chunks}] 完成: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
