#!/usr/bin/env python3
"""历史截面回测 — 按 cutoff 日期截断数据 → 重跑 CZSC → 跑全漏斗。

用法:
    python3 -m manual_backtest historical 2025-09-11
    python3 -m manual_backtest historical 2025-09-11 --sample 50

策略（对齐 l1_update.py 全量更新）:
    ③  日线 CZSC: 全量并行重修
    ③b 周线 CZSC: 全量并行重修
    ③c 周日共振预筛: 用 struct_cache 过滤出周日共振股（三级联立必要条件）
    ③d 30min CZSC: 仅对周日共振候选按需补缺
"""
import argparse, subprocess, sys, time, json, shutil
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
TMP = BASE / "tmp_out"
WORKER = TMP / "hist_worker.py"
MIN_BARS = {"日线": 120, "周线": 60, "30分钟": 200}


def filter_kline(src: Path, dst: Path, cutoff: str, freq: str):
    """截断 K 线到 cutoff 日期（无未来bar, 匹配实盘）。"""
    dst.mkdir(parents=True, exist_ok=True)
    cutoff_dt = pd.Timestamp(cutoff)
    min_b = MIN_BARS[freq]
    copied = 0
    for f in sorted(src.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
            dc = "date" if "date" in df.columns else "dt"
            df[dc] = pd.to_datetime(df[dc])
            filtered = df[df[dc] <= cutoff_dt]
            if len(filtered) < min_b:
                continue
            filtered.to_parquet(dst / f.name, index=False)
            copied += 1
        except Exception:
            pass
    return copied


def generate_signals(codes: list, data_dir: Path, sig_dir: Path,
                     freq: str, cutoff: str, n_workers: int = 4,
                     max_bars: int = 0):
    """subprocess.Popen 并行 CZSC 信号生成。max_bars=0 全量, >0 增量(tail N bar)。"""
    if not codes:
        return 0
    chunks = [[] for _ in range(n_workers)]
    for i, code in enumerate(codes):
        chunks[i % n_workers].append(code)

    print(f"  [{freq}] {len(codes)} 只, {n_workers} 核")
    t0 = time.time()
    procs, lfs, pfs, cfs = [], [], [], []

    for i in range(n_workers):
        if not chunks[i]:
            continue
        cfg = TMP / f"hist_cfg_{freq}_{i}.json"
        pf = TMP / f"hist_prog_{freq}_{i}.txt"
        lf = open(str(TMP / f"hist_log_{freq}_{i}.log"), "w")
        with open(cfg, "w") as f:
            json.dump({
                "codes": chunks[i], "data_dir": str(data_dir),
                "sig_dir": str(sig_dir), "freq": freq,
                "cutoff": cutoff, "progress_file": str(pf),
                "max_bars": max_bars,
            }, f)
        p = subprocess.Popen(
            [sys.executable, "-u", str(WORKER), str(cfg)],
            stdout=lf, stderr=subprocess.STDOUT)
        procs.append(p); lfs.append(lf); pfs.append(pf); cfs.append(cfg)

    while any(p.poll() is None for p in procs):
        time.sleep(30)
        for pf in pfs:
            if pf.exists():
                t = pf.read_text().strip()
                if t: print(f"    {t}")
        d = sum(1 for p in procs if p.poll() is not None)
        print(f"    [{freq}] {d}/{len(procs)} done {time.time()-t0:.0f}s")

    for lf in lfs: lf.close()
    for cf in cfs: cf.unlink(missing_ok=True)
    for pf in pfs: pf.unlink(missing_ok=True)

    n = len(list(sig_dir.glob("*.parquet"))) if sig_dir.exists() else 0
    print(f"  [{freq}] 完成 {time.time()-t0:.0f}s, {n} 信号文件")
    return n


def _patch_paths(bt_data: Path, cutoff: pd.Timestamp):
    """Monkey-patch 数据目录指向历史回测目录。返回 restore 函数。"""
    import verify_buy_type, zone1_deposition, zone2_regime, zone3_regime, l3_filter
    import core.date_utils
    _orig_get_effective = core.date_utils.get_effective_date
    core.date_utils.get_effective_date = lambda: cutoff
    # 同时 patch 使用 from-import 的模块本地绑定（Python 不会自动更新）
    zone2_regime.get_effective_date = lambda: cutoff
    verify_buy_type.get_effective_date = lambda: cutoff

    _orig = {}
    def _p(mod, attr, val):
        if hasattr(mod, attr):
            _orig[(mod, attr)] = getattr(mod, attr)
            setattr(mod, attr, val)

    _p(verify_buy_type, "SIG_DIR", bt_data / "signals")
    _p(verify_buy_type, "SIG_WEEKLY_DIR", bt_data / "signals_weekly")
    _p(verify_buy_type, "SIG_30MIN_DIR", bt_data / "signals_30min")
    _p(verify_buy_type, "STRUCT_DIR", bt_data / "struct_cache")
    _p(verify_buy_type, "STRUCT_30M_DIR", bt_data / "struct_cache_30m")
    _p(verify_buy_type, "STRUCT_WEEKLY_DIR", bt_data / "struct_cache_weekly")
    _p(verify_buy_type, "DAILY_DIR", bt_data / "daily")
    _p(zone1_deposition, "SIGNALS_DIR", bt_data / "signals")
    _p(zone1_deposition, "ZONES_DIR", bt_data / "zones")
    _p(zone2_regime, "SIGNALS", bt_data / "signals")
    _p(zone2_regime, "DAILY", bt_data / "daily")
    _p(zone2_regime, "WEEKLY_SIG", bt_data / "signals_weekly")
    _p(zone2_regime, "M30_SIG", bt_data / "signals_30min")
    _p(zone2_regime, "STRUCT_DIR", bt_data / "struct_cache")
    _p(zone2_regime, "STRUCT_30M_DIR", bt_data / "struct_cache_30m")
    _p(zone2_regime, "STRUCT_WEEKLY_DIR", bt_data / "struct_cache_weekly")
    (bt_data / "zones").mkdir(parents=True, exist_ok=True)
    _p(zone2_regime, "ZONES", bt_data / "zones")
    if hasattr(zone3_regime, "ZONES"):
        _p(zone3_regime, "ZONES", bt_data / "zones")
    _p(l3_filter, "DAILY", bt_data / "daily")

    def restore():
        core.date_utils.get_effective_date = _orig_get_effective
        zone2_regime.get_effective_date = _orig_get_effective
        verify_buy_type.get_effective_date = _orig_get_effective
        for (mod, attr), val in _orig.items():
            setattr(mod, attr, val)
    return restore


def run(date: str, sample: int = 0, workers: int = 4):
    cutoff = pd.Timestamp(date)
    bt_dir = TMP / f"backtest_{date}"
    bt_data = bt_dir / "data"

    # 每次回测前清理旧数据，避免残留文件干扰（幂等设计）
    if bt_dir.exists(): shutil.rmtree(bt_dir)
    # 同时清理 tmp_out 下上次跑的临时文件（hist_log/prog/cfg），防止 stdout 缓冲时误读旧日志
    for pattern in ["hist_log_*.log", "hist_prog_*.txt", "hist_cfg_*.json"]:
        for f in TMP.glob(pattern):
            try: f.unlink()
            except OSError: pass
    bt_data.mkdir(parents=True, exist_ok=True)

    print(f"=== 历史截面回测: {date} ===\n")

    # ── Step 1: 过滤 K 线 ──
    print("[1/4] 过滤 K 线 (含 CZSC 缓冲)...")
    specs = [
        ("日线", BASE / "data" / "daily", bt_data / "daily"),
        ("周线", BASE / "data" / "weekly", bt_data / "weekly"),
        ("30分钟", BASE / "data" / "min30", bt_data / "min30"),
    ]
    all_codes = {}
    for freq, src, dst in specs:
        n = filter_kline(src, dst, date, freq)
        all_codes[freq] = sorted(p.stem for p in dst.glob("*.parquet"))
        print(f"  {freq}: {n} 只")

    if sample > 0:
        import random; random.seed(42)
        ds = random.sample(all_codes["日线"], min(sample, len(all_codes["日线"])))
        all_codes["日线"] = ds
        all_codes["周线"] = [c for c in all_codes["周线"] if c in ds]
        all_codes["30分钟"] = [c for c in all_codes["30分钟"] if c in ds]
        print(f"\n  [样本] 日{len(ds)} 周{len(all_codes['周线'])} 30m{len(all_codes['30分钟'])}")

    # ── ③ 日线 CZSC（全量） ──
    print("\n[2/4] ③ 日线 CZSC（最后600 bar, 4核并行）...")
    generate_signals(all_codes["日线"], bt_data / "daily",
                     bt_data / "signals", "日线", date, workers, max_bars=600)

    # ── ③b 周线 CZSC（全量） ──
    print("\n[2b/4] ③b 周线 CZSC（全量并行）...")
    generate_signals(all_codes["周线"], bt_data / "weekly",
                     bt_data / "signals_weekly", "周线", date, workers)

    # ── struct_cache 复制（周日共振预筛需要） ──
    print("\n[2b+/4] 复制 struct_cache (日线+周线) ...")
    import shutil as _shutil
    for _sn, _dn in [("struct_cache", "struct_cache"),
                      ("struct_cache_weekly", "struct_cache_weekly")]:
        _src = BASE / "data" / _sn
        if _src.exists():
            _shutil.copytree(str(_src), str(bt_data / _dn))
            print(f"  {_sn} → {bt_data/_dn}")

    # ── ③c 30min CZSC（按 L1 + 周日共振候选按需） ──
    print("\n[2c/4] ③c L1 → 周日共振预筛 → 30min CZSC 按需...")

    # 先跑 L1 获取候选代码
    restore_temp = _patch_paths(bt_data, cutoff)
    l1_codes = set()
    try:
        import zone1_deposition
        df1 = zone1_deposition.run(lookback=20)
        if df1 is not None and len(df1) > 0:
            l1_codes = set(df1["代码"].astype(str).str.zfill(6))
        print(f"  L1 候选: {len(l1_codes)} 只")

        # 周日共振预筛：三级联立 ⊆ 周日共振，不满足周日共振的不可能通过三级联立
        from verify_buy_type import check_weekly_structural_resonance
        _rc = set()
        for _cd in sorted(l1_codes):
            try:
                if check_weekly_structural_resonance(_cd, signal_date=cutoff):
                    _rc.add(_cd)
            except Exception:
                pass
        print(f"  周日共振预筛: {len(_rc)}/{len(l1_codes)} 只")
        resonance_codes = _rc
    finally:
        restore_temp()

    # 只对满足周日共振的 L1 候选跑 30min CZSC
    m30_codes = [c for c in all_codes["30分钟"] if c in resonance_codes]
    print(f"  30min 按需: {len(m30_codes)}/{len(all_codes['30分钟'])} 只")
    generate_signals(m30_codes, bt_data / "min30",
                     bt_data / "signals_30min", "30分钟", date, workers,
                     max_bars=800)

    # ── Step 4: 全管道 L1→L4 ──
    print("\n[3/4] 全管道 L1→L4...")

    restore = _patch_paths(bt_data, cutoff)
    try:
        from manual_backtest.engine import ManualBacktester
        bt = ManualBacktester()
        l4 = bt.run_pipeline(date=date)
        if l4 is not None and len(l4) > 0:
            out_path = bt.export_for_marking(str(bt_dir))
            print(f"\n  L4: {out_path} ({len(l4)} candidates)")
        else:
            print("\n  [无候选] 当日无符合条件股票")
    finally:
        restore()

    print(f"\n产物: {bt_dir}/")
    return l4 if 'l4' in dir() else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("date")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.date, sample=args.sample, workers=args.workers)
