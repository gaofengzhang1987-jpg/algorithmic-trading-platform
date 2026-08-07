#!/usr/bin/env python3
"""L1 全量更新流水线：数据拉取 → 并行信号重生 → RPS → L1-L4 打分。
	
流程（对齐 docs/全量更新流水线.md）：
   ① 拉日线 (tushare) + 上证指数
   ② 拉 30 分钟数据 (akshare, 最近 30 天)
   ②b 周线 K 线生成 (从日线本地重采样)
   ③ CZSC 日线信号并行重生 (run_signal_parallel.py, 4 核)
   ③b CZSC 周线信号并行重生 (run_signal_parallel.py --freq 周线)
   ③c CZSC 30分钟信号 bridge (l1_30min_bridge.py, 按 L1 候选按需补缺)
   ③d 信号质量模型重训练 (qlib_ml/signal_trainer.py)
   ④ RPS 刷新 (rps_calc.py full)
   ⑤⑥⑦ L1→L4 正式管线 (run_zones.py 100 ×2)
	
用法：
  python3 l1_update.py                # 完整流水线 (默认截止今天)
  python3 l1_update.py --dry-run      # 仅检查、不执行
  python3 l1_update.py --end 2026-07-15       # 指定截止日期
  python3 l1_update.py --skip-pull    # 跳过外部 API 拉取
  python3 l1_update.py --skip-rps     # 跳过 RPS 刷新
  python3 l1_update.py --skip-l1l4    # 跳过 L1-L4 打分
  python3 l1_update.py --skip-weekly  # 跳过周线信号
  python3 l1_update.py --skip-30min   # 跳过 30 分钟信号
  python3 l1_update.py --skip-signal-train  # 跳过信号模型重训练
"""

import logging, os, sys, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import json
import subprocess

import pandas as pd
import tushare as ts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("l1_update")

BASE_DIR = Path(__file__).parent
DAILY_DIR = BASE_DIR / "data" / "daily"
SIG_DIR = BASE_DIR / "data" / "signals"
STATE_FILE = BASE_DIR / "data" / "last_update.txt"
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "a201cb89dff50044936fc4554d0751939a56bc9afab8726424ac234e")

MIN_BARS = 120
ST_PREFIXES = ('ST', '*ST', 'SST', 'S*ST', 'NST')
BJ_PREFIX = ('8', '9')
MIN30_DIR = BASE_DIR / "data/min30"
MIN30_DIR.mkdir(parents=True, exist_ok=True)
WEEKLY_DIR = BASE_DIR / "data" / "weekly"
WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
SIG_WEEKLY_DIR = BASE_DIR / "data" / "signals_weekly"
SIG_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
SIG_30MIN_DIR = BASE_DIR / "data" / "signals_30min"
SIG_30MIN_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
#  步骤 ①：拉取增量日线
# ================================================================

def _filter_stock(ts_code: str) -> bool:
    """返回 True 表示该股票应拉取。跳过 ST/北交所/退市。"""
    code = ts_code.split('.')[0]
    if code.startswith(BJ_PREFIX):
        return False
    # tushare 的 daily 接口不直接返回 ST 标记，后续通过 stock_basic 判断
    return True


def _filter_stock_basic(daily_df, pro):
    """根据 stock_basic 过滤 ST 股。"""
    try:
        basic = pro.stock_basic(exchange='', list_status='L',
                                fields='ts_code,name,list_date')
        st_codes = set()
        for _, r in basic.iterrows():
            name = str(r.get('name', ''))
            if any(name.startswith(p) for p in ST_PREFIXES):
                st_codes.add(r['ts_code'].split('.')[0])
        return daily_df[~daily_df['ts_code'].apply(
            lambda x: str(x).split('.')[0]).isin(st_codes)]
    except:
        return daily_df  # 过滤失败时直接返回原始数据


def _is_new_stock(code: str) -> bool:
    """日线少于 120 根 bar 的新股。"""
    fpath = DAILY_DIR / f"{code}.parquet"
    if not fpath.exists():
        return True
    df = pd.read_parquet(fpath, columns=['date'])
    return len(df) < MIN_BARS


def get_last_update_date():
    if STATE_FILE.exists():
        return datetime.strptime(STATE_FILE.read_text().strip(), "%Y-%m-%d")
    # 默认从 2020-01-01 开始
    return datetime(2020, 1, 1)

def pull_daily(end_date: str, dry_run=False):
    """拉取增量日线。返回 (新数据股票数, 最新日期)。"""
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    last_date = get_last_update_date()
    target_end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

    start = last_date + timedelta(days=1)
    if start > target_end:
        logger.info("日线已是最新 (%s)，跳过拉取", last_date.strftime("%Y-%m-%d"))
        return 0, target_end

    start_str = start.strftime("%Y%m%d")
    end_str = target_end.strftime("%Y%m%d")
    logger.info("拉取日线: %s → %s", start_str, end_str)
    if dry_run:
        logger.info("[DRY-RUN] 跳过实际拉取")
        return 0, target_end

    existing_codes = set(p.stem for p in DAILY_DIR.glob("*.parquet"))
    updated = set()

    try:
        df = pro.daily(start_date=start_str, end_date=end_str)
        if df is not None and not df.empty:
            df = df[df.apply(lambda r: _filter_stock(str(r['ts_code'])), axis=1)]  # 过滤 ST/北交
            df = _filter_stock_basic(df, pro)  # 过滤 ST 股
            for code, group in df.groupby(df['ts_code'].apply(lambda x: str(x).split('.')[0])):
                if dry_run:
                    updated.add(code)
                    continue
                fpath = DAILY_DIR / f"{code}.parquet"
                bars = []
                for _, r in group.iterrows():
                    bars.append({
                        'date': pd.Timestamp(str(r['trade_date'])),
                        'code': code,
                        'open': float(r['open']), 'high': float(r['high']),
                        'low': float(r['low']), 'close': float(r['close']),
                        'volume': float(r['vol']), 'amount': float(r['amount']),
                    })
                new_df = pd.DataFrame(bars)
                if fpath.exists():
                    old_df = pd.read_parquet(fpath)
                    combined = pd.concat([old_df, new_df], ignore_index=True)
                else:
                    combined = new_df
                combined = combined.drop_duplicates('date', keep='last').sort_values('date')
                combined.to_parquet(fpath, index=False)
                updated.add(code)
    except Exception as e:
        logger.warning("日线拉取失败: %s", str(e)[:100])

    # 拉取上证指数
    try:
        idx_df = pro.index_daily(ts_code='000001.SH',
                                 start_date=start_str, end_date=end_str)
        if idx_df is not None and not idx_df.empty:
            idx_path = BASE_DIR / "data/index/000001.parquet"
            idx_bars = []
            for _, r in idx_df.iterrows():
                idx_bars.append({
                    'date': pd.Timestamp(str(r['trade_date'])),
                    'open': float(r['open']), 'high': float(r['high']),
                    'low': float(r['low']), 'close': float(r['close']),
                    'volume': float(r['vol']),
                    'code': '000001',
                })
            new_idx = pd.DataFrame(idx_bars)
            if idx_path.exists():
                old_idx = pd.read_parquet(idx_path)
                combined = pd.concat([old_idx, new_idx], ignore_index=True)
            else:
                combined = new_idx
            combined = combined.drop_duplicates('date', keep='last').sort_values('date')
            combined.to_parquet(idx_path, index=False)
            logger.info("上证指数更新: %d 条", len(idx_bars))
    except Exception as e:
        logger.warning("上证指数拉取失败: %s", str(e)[:80])

    # 拉取中证1000（regime 协同验证）
    try:
        idx2_df = pro.index_daily(ts_code='000852.SH',
                                  start_date=start_str, end_date=end_str)
        if idx2_df is not None and not idx2_df.empty:
            idx2_path = BASE_DIR / "data/index/000852.parquet"
            idx2_bars = []
            for _, r in idx2_df.iterrows():
                idx2_bars.append({
                    'date': pd.Timestamp(str(r['trade_date'])),
                    'open': float(r['open']), 'high': float(r['high']),
                    'low': float(r['low']), 'close': float(r['close']),
                    'volume': float(r['vol']),
                    'code': '000852',
                })
            new_idx2 = pd.DataFrame(idx2_bars)
            if idx2_path.exists():
                old_idx2 = pd.read_parquet(idx2_path)
                combined = pd.concat([old_idx2, new_idx2], ignore_index=True)
            else:
                combined = new_idx2
            combined = combined.drop_duplicates('date', keep='last').sort_values('date')
            combined.to_parquet(idx2_path, index=False)
            logger.info("中证1000更新: %d 条", len(idx2_bars))
    except Exception as e:
        logger.warning("中证1000拉取失败: %s", str(e)[:80])

    STATE_FILE.write_text(target_end.strftime("%Y-%m-%d"))
    logger.info("日线更新: %d 只股票", len(updated))
    return len(updated), target_end


def pull_min30(dry_run=False, last_days=30) -> int:
    """拉取最近 last_days 天的 30 分钟数据。返回更新股票数。"""
    import akshare as ak

    codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
    logger.info("拉取 30 分钟数据: %d 只, 最近 %d 天", len(codes), last_days)
    if dry_run:
        return 0

    def _to_sina(code: str) -> str:
        if code.startswith(("0", "3")):
            return f"sz{code}"
        return f"sh{code}"

    updated = 0
    for i, code in enumerate(codes):
        try:
            time.sleep(0.25)  # 避免 akshare 限流
            df_5min = ak.stock_zh_a_minute(symbol=_to_sina(code), period="5", adjust="qfq")
            if df_5min is None or df_5min.empty:
                continue
            df_5min = df_5min.rename(columns={'day': 'date'})
            df_5min['date'] = pd.to_datetime(df_5min['date'])
            for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df_5min[col] = pd.to_numeric(df_5min[col], errors='coerce').fillna(0.0)

            # 只保留最近 last_days 天
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=last_days)
            df_5min = df_5min[df_5min['date'] >= cutoff]

            # 重采样为 30 分钟
            df_5min = df_5min.set_index('date')
            ohlc = df_5min.resample("30min").agg({
                'open': 'first', 'high': 'max',
                'low': 'min', 'close': 'last',
                'volume': 'sum', 'amount': 'sum',
            })
            
            df_30min = ohlc.dropna(subset=['open']).reset_index()
            df_30min["code"] = code
            df_30min = df_30min[["date", "open", "high", "low", "close", "volume", "amount", "code"]]

            fpath = MIN30_DIR / f"{code}.parquet"
            if fpath.exists():
                old = pd.read_parquet(fpath)
                combined = pd.concat([old, df_30min], ignore_index=True)
                combined = combined.drop_duplicates('date', keep='last').sort_values('date')
            else:
                combined = df_30min
            combined.to_parquet(fpath, index=False)
            updated += 1
        except Exception as e:
            logger.debug("30 分钟数据拉取失败 %s: %s", code, str(e)[:60])

        if (i + 1) % 200 == 0:
            logger.info("30 分钟: %d/%d", i + 1, len(codes))

    logger.info("30 分钟数据完成: %d 只更新", updated)
    return updated


# ================================================================
#  周线 K 线生成：从日线数据本地重采样
# ================================================================

def generate_weekly_kline():
    """从 data/daily/ 重采样生成 data/weekly/ 周线 K 线。"""
    codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
    logger.info("周线 K 线生成: %d 只", len(codes))
    t0 = time.time()
    updated = 0
    for code in codes:
        try:
            df = pd.read_parquet(DAILY_DIR / f"{code}.parquet").sort_values("date")
            df = df.set_index("date")
            weekly = df.resample("W").agg({
                "open": "first", "high": "max",
                "low": "min", "close": "last",
                "volume": "sum", "amount": "sum",
            }).dropna(subset=["open"]).reset_index()
            weekly["code"] = code
            weekly = weekly[["date", "open", "high", "low", "close", "volume", "amount", "code"]]
            weekly.to_parquet(WEEKLY_DIR / f"{code}.parquet", index=False)
            updated += 1
        except Exception as e:
            logger.debug("周线K线 %s: %s", code, str(e)[:60])
    logger.info("周线 K 线完成: %d/%d 只, %.0fs", updated, len(codes), time.time() - t0)


# ================================================================


def run_l1_l4():
    """运行 L1→L4 正式管线：run_zones.py 100 ×2（对齐 docs/全量更新流水线.md）。"""
    for round_num in [1, 2]:
        logger.info("=== L1→L4 管线 第 %d 轮 ===", round_num)
        ret = subprocess.run(
            [sys.executable, "-u", str(BASE_DIR / "run_zones.py"), "100"],
            cwd=str(BASE_DIR), check=False, capture_output=False
        ).returncode
        if ret != 0:
            logger.warning("run_zones.py 第 %d 轮退出码: %d", round_num, ret)


# ================================================================
#  步骤 ④：RPS 增量刷新
# ================================================================

def refresh_rps(dry_run=False):
    """RPS 全量刷新 (rps_calc.py full → 重建 close_matrix + daily_close + RPS)。"""
    logger.info("=== RPS 增量刷新 ===")
    if dry_run:
        logger.info("[DRY-RUN] 跳过")
        return
    ret = subprocess.run(['python3', str(BASE_DIR / 'rps_calc.py'), 'full'],
                         capture_output=True, text=True)
    if ret.returncode != 0:
        logger.warning("RPS 刷新返回码 %d: %s", ret.returncode, ret.stderr[:200])
    else:
        for line in ret.stdout.strip().split('\n'):
            logger.info("  %s", line)
# ================================================================
#  步骤 ③d：信号质量模型重训练
# ================================================================

def train_signal_model():
    logger.info("=== 信号质量模型重训练 ===")
    trainer = str(BASE_DIR / "qlib_ml" / "signal_trainer.py")
    ret = subprocess.run(
        [sys.executable, "-u", trainer],
        cwd=str(BASE_DIR), check=False,
        capture_output=True, text=True, timeout=600,
    )
    if ret.returncode != 0:
        logger.warning("信号模型训练失败 (exit=%d): %s", ret.returncode, ret.stderr[-300:])
    else:
        for line in ret.stdout.strip().split('\n')[-5:]:
            logger.info("  %s", line)
        logger.info("步骤③d完成")


# ================================================================
# ================================================================
#  主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="L1 增量更新流水线")
    parser.add_argument("--dry-run", action="store_true", help="仅检查不执行")
    parser.add_argument("--end", type=str, help="截止日期 YYYY-MM-DD")
    parser.add_argument("--skip-pull", action="store_true", help="跳过外部 API 拉取 (①+②)")
    parser.add_argument("--skip-signals", action="store_true", help="跳过信号重生 (③+③b+③c)")
    parser.add_argument("--skip-weekly", action="store_true", help="跳过周线信号重生 (③b)")
    parser.add_argument("--skip-weekly-kline", action="store_true", help="仅跳过周线K线生成 (②b)")
    parser.add_argument("--skip-weekly-signals", action="store_true", help="仅跳过周线CZSC信号 (③b)")
    parser.add_argument("--skip-30min", action="store_true", help="跳过30分钟信号重生 (③c)")
    parser.add_argument("--skip-signal-train", action="store_true", help="跳过信号质量模型重训练 (③d)")
    parser.add_argument("--skip-rps", action="store_true", help="跳过 RPS 刷新 (④)")
    parser.add_argument("--skip-l1l4", action="store_true", help="跳过 L1-L4 打分 (⑤⑥⑦)")
    args = parser.parse_args()
    
    start_ts = time.time()
    
    # Step ① 拉日线 + 上证指数
    if not args.skip_pull and not args.dry_run:
        n, end_date = pull_daily(args.end, args.dry_run)
        logger.info("步骤①完成: %d 只更新, 截止 %s", n, end_date.strftime("%Y-%m-%d"))
    elif args.skip_pull:
        logger.info("步骤①跳过 (--skip-pull)")
   
    # Step ② 拉 30 分钟数据（与日线同开关）
    if not args.skip_pull and not args.dry_run:
        n30 = pull_min30(dry_run=args.dry_run)
        logger.info("步骤②完成: %d 只更新", n30)
    elif args.skip_pull:
        logger.info("步骤②跳过 (--skip-pull)")
    
    # 周线 K 线生成（从日线本地重采样，--skip-pull 时同样执行）
    if not (args.skip_weekly or args.skip_weekly_kline) and not args.dry_run:
        generate_weekly_kline()
    else:
        logger.info("周线 K 线生成跳过 (%s)",
                    "--skip-weekly" if args.skip_weekly else "--skip-weekly-kline")
    
    # Step ③ CZSC 日线信号并行重生 (run_signal_parallel.py, 4 核)
    if not args.skip_signals and not args.dry_run:
        logger.info("步骤③: 日线信号并行重生 (run_signal_parallel.py)")
        ret = subprocess.run(
            [sys.executable, str(BASE_DIR / "tmp_out" / "run_signal_parallel.py")],
            cwd=str(BASE_DIR), check=False
        )
        if ret.returncode != 0:
            logger.error("步骤③失败: run_signal_parallel.py exit=%d", ret.returncode)
        else:
            logger.info("步骤③完成: 日线信号重生")
    elif args.skip_signals:
        logger.info("步骤③跳过 (--skip-signals)")
    
    # Step ③b CZSC 周线信号并行重生 (run_signal_parallel.py --freq 周线)
    if not args.skip_signals and not (args.skip_weekly or args.skip_weekly_signals) and not args.dry_run:
        logger.info("步骤③b: 周线信号并行重生 (run_signal_parallel.py --freq 周线)")
        ret = subprocess.run(
            [sys.executable, str(BASE_DIR / "tmp_out" / "run_signal_parallel.py"), "--freq", "周线"],
            cwd=str(BASE_DIR), check=False
        )
        if ret.returncode != 0:
            logger.error("步骤③b失败: run_signal_parallel.py exit=%d", ret.returncode)
        else:
            logger.info("步骤③b完成: 周线信号重生")
    elif args.skip_signals:
        logger.info("步骤③b跳过 (--skip-signals)")
    else:
        logger.info("步骤③b跳过 (%s)",
                    "--skip-weekly" if args.skip_weekly else "--skip-weekly-signals")
    
    # Step ③c CZSC 30分钟信号 bridge (l1_30min_bridge.py, 按 L1 候选按需补缺)
    if not args.skip_signals and not args.skip_30min and not args.dry_run:
        logger.info("步骤③c: 30分钟信号 bridge (l1_30min_bridge.py)")
        ret = subprocess.run(
            [sys.executable, str(BASE_DIR / "l1_30min_bridge.py")],
            cwd=str(BASE_DIR), check=False
        )
        if ret.returncode != 0:
            logger.error("步骤③c失败: l1_30min_bridge.py exit=%d", ret.returncode)
        else:
            logger.info("步骤③c完成: 30分钟信号 bridge")
    elif args.skip_signals:
        logger.info("步骤③c跳过 (--skip-signals)")
    elif args.skip_30min:
        logger.info("步骤③c跳过 (--skip-30min)")
    
    # Step ③d 信号质量模型重训练
    if not args.skip_signals and not args.skip_signal_train and not args.dry_run:
        train_signal_model()
    elif args.skip_signals:
        logger.info("步骤③d跳过 (--skip-signals)")
    elif args.skip_signal_train:
        logger.info("步骤③d跳过 (--skip-signal-train)")

    # Step ④ RPS 刷新
    if not args.skip_rps and not args.dry_run:
        refresh_rps()
        logger.info("步骤④完成")
    
    # Step ⑤⑥⑦ L1→L4
    if not args.skip_l1l4 and not args.dry_run:
        run_l1_l4()
        logger.info("步骤⑤⑥⑦完成")
    
    elapsed = int(time.time() - start_ts)
    logger.info("流水线完成, 总耗时 %ds", elapsed)


if __name__ == "__main__":
    main()
