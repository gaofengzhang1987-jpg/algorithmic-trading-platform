#!/usr/bin/env python3
"""一键执行四级漏斗全流程 L1 -> L2 -> L3 -> L4（含 B+ 直通分叉）。"""

import sys
import time
import logging

import pandas as pd
from pathlib import Path

from zone1_deposition import run as l1_run
from verify_buy_type import verify_buy_type, check_resonance

from zone2_regime import run as l2_run
from pathlib import Path

# ── 30分钟信号按需生成 ─────────────────────────────────────────
SIG_30MIN_PATH = Path("data/signals_30min")
MIN30_DATA = Path("data/min30")


def ensure_30min(codes):
    """按需生成 30min 信号（仅对需要共振检查的候选）。"""
    import pandas as pd
    from czsc import RawBar, Freq, generate_czsc_signals
    import signal_config
    if not codes:
        return

    # 确定参考日期
    daily_sig_dir = Path("data/signals")
    ref_dates = []
    for sig_f in sorted(daily_sig_dir.glob("*.parquet"))[-200:]:
        df = pd.read_parquet(sig_f, columns=["dt"])
        ref_dates.append(pd.Timestamp(df["dt"].max()))
    ref_date = max(ref_dates) if ref_dates else pd.Timestamp.now()

    # 找出需要生成的代码
    todo = []
    SIG_30MIN_PATH.mkdir(parents=True, exist_ok=True)
    for c in codes:
        sig_f = SIG_30MIN_PATH / f"{c}.parquet"
        if not sig_f.exists():
            todo.append(c)
        else:
            try:
                df = pd.read_parquet(sig_f)
                if pd.Timestamp(df["dt"].max()) < ref_date:
                    todo.append(c)
            except Exception:
                todo.append(c)

    if not todo:
        logger.info("30min 信号无缺失，跳过")
        return

    t0 = time.time()
    logger.info("30min 按需生成: %d 只", len(todo))
    sig_config = signal_config.get_config(freq="30分钟")
    success = 0
    for i, code in enumerate(todo):
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            logger.info("  30min: %d/%d (%.0fs)", i + 1, len(todo), time.time() - t0)
        try:
            df = pd.read_parquet(MIN30_DATA / f"{code}.parquet")
            if len(df) < 200:
                continue
            df = df.sort_values("date").tail(800)
            bars = [
                RawBar(symbol=code, id=j+1, dt=r["date"].to_pydatetime(),
                       freq=Freq.F30, open=r["open"], close=r["close"],
                       high=r["high"], low=r["low"],
                       vol=float(r.get("volume", 0)), amount=float(r.get("amount", 0)))
                for j, (_, r) in enumerate(df.iterrows())
            ]
            sdt = str(df["date"].iloc[0].date()).replace("-", "")
            sigs = generate_czsc_signals(
                bars, signals_config=sig_config, tqdm_kwargs={"disable": True},
                sdt=sdt, init_n=min(100, len(bars)), df=True)
            if sigs is not None and not sigs.empty:
                sigs = sigs.drop(columns=[c for c in ["freq", "cache"] if c in sigs.columns])
                sigs.to_parquet(SIG_30MIN_PATH / f"{code}.parquet", index=False)
                success += 1
        except Exception:
            pass
    logger.info("30min 完成: %d/%d (%.0fs)", success, len(todo), time.time() - t0)

import regime_detector
from zone3_regime import run as l3_run
from zone4_regime import run as l4_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_zones")




def main():
    t0 = time.time()

    # 清理旧输出，防止管道中断后残留文件被下游独立脚本误读
    ZONES = Path(__file__).parent / "data" / "zones"
    for stale in ["L2_regime.parquet", "L3_regime.parquet", "L4_recommend.parquet"]:
        sp = ZONES / stale
        if sp.exists():
            sp.unlink()
            logger.info("已清理旧文件: %s", stale)
    # Regime detection（五维牛熊评分制，支持手动覆盖）

    if len(sys.argv) > 2:
        regime = sys.argv[2]
        logger.info("Manual regime: %s (sys.argv override)", regime)
    else:
        regime, score, dims, breadth_pct = regime_detector.detect()
        logger.info("Auto regime: %s (Bull Score: %.1f/100)", regime, score)
        logger.info("  均线:%d/20 位置:%d/20 ADX:%.1f/20 量价:%d/15 协同:%d/5 宽度:%d/20 (%.1f%%)",
                    dims["均线排列"], dims["价格位置"],
                    dims["ADX趋势强度"], dims["量价关系"], dims["指数协同"],
                    dims["市场宽度"], breadth_pct)
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    # L1: 全量沉淀区（已含拆行 + 标签）
    logger.info("=" * 40)
    logger.info("L1 全量沉淀区: 扫描全市场买点信号...")
    df1 = l1_run()
    if df1.empty:
        logger.info("L1 无候选，流程终止")
        return
    logger.info("L1 完成: %d 只 (%.1fs)", len(df1), time.time() - t0)

    # B+ 统计（对齐回测设计：verify=加分不豁免, resonance=仅计数, bplus 由 zone2_regime 内部三级联立填充）
    from verify_buy_type import check_weekly_structural_resonance

    bplus_codes = set()        # 由 zone2_regime 内部三级联立填充，B+ 循环不写入
    bplus_verify_count = 0     # 结构合规（加分不豁免）
    resonance_count = 0        # 三级标签共振（仅计数）
    resonance_candidates = []  # 需要共振检查的候选（二/三买且结构验证失败）

    for _, row in df1.iterrows():
        code = row["代码"]
        label = row["买点类型"]
        bt = "一买" if "一买" in label else ("二买" if "二买" in label else "三买")
        if bt == "一买":
            continue
        # 独立检查，不互斥：一只股票可以同时结构合规+周日共振
        if verify_buy_type(code, bt):
            bplus_verify_count += 1  # 加分不豁免
        resonance_candidates.append((code, bt, row.get("最新日期")))  # 所有非一买都检查共振

    # 周日共振预筛：只对日线+周线结构匹配的候选生成 30min
    if resonance_candidates:
        rc_codes = list(set(c for c, _, _ in resonance_candidates))
        prescreened = set()
        for c in rc_codes:
            try:
                if check_weekly_structural_resonance(c):
                    prescreened.add(c)
            except Exception:
                pass
        logger.info("周日共振预筛: %d/%d 只通过", len(prescreened), len(rc_codes))

        if prescreened:
            ensure_30min(list(prescreened))
            for code, bt, sig_date in resonance_candidates:
                if code in prescreened and check_resonance(code, bt, signal_date=sig_date):
                    resonance_count += 1  # 仅计数（对齐回测）
    logger.info("B+: %d只(加分) %d只(共振)",
                bplus_verify_count, resonance_count)

    # L2: Regime 路由
    logger.info("=" * 40)
    logger.info("L2 Regime Filter: regime=%s, candidates=%d", regime, len(df1))
    df2 = l2_run(df1, regime=regime, bplus_codes=bplus_codes)
    logger.info("L2 完成: %d 只 (%.1fs)", len(df2), time.time() - t0)

    # L3: 策略二区
    logger.info("=" * 40)
    logger.info("L3 策略二区: 应用策略二区规则...")
    df3 = l3_run(df2, regime=regime)
    logger.info("L3 完成: %d 只 (%.1fs)", len(df3), time.time() - t0)

    # L4: 购买建议区
    logger.info("=" * 40)
    logger.info("L4 购买建议区: 评分排名...")
    df4 = l4_run(df3, top_n=top_n)
    logger.info("L4 完成: %d 只 (%.1fs)", len(df4), time.time() - t0)

    # 漏斗摘要
    logger.info("=" * 40)
    logger.info("漏斗摘要: %d -> %d -> %d -> %d (%.0fs)",
                len(df1), len(df2), len(df3), len(df4), time.time() - t0)
    logger.info("Regime: %s", regime)

    if not df4.empty:
        print("\n=== L4 TOP 10 ===")
        cols = [c for c in ["代码", "买点类型", "L2_综合得分", "L2_Regime"] if c in df4.columns]
        print(df4[cols].head(10).to_string())


if __name__ == "__main__":
    main()
