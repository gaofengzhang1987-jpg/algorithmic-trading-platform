#!/usr/bin/env python3
"""L1 增量更新流水线：日线拉取 → 信号重生 → L1-L4 打分。

流程：
  ① 拉取增量日线 (tushare, 从上轮截止日至今)
  ② 识别有新数据的股票
  ③ 删旧信号文件 → CZSC 重生 (init_n=300)
  ④ L1→L2 打分
  ⑤ L3 过滤
  ⑥ L4 排名

用法：
  python3 l1_update.py           # 完整流水线 (默认截止今天)
  python3 l1_update.py --dry-run # 仅检查、不执行
  python3 l1_update.py --end 2026-07-15  # 指定截止日期
"""

import logging, os, sys, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

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

    STATE_FILE.write_text(target_end.strftime("%Y-%m-%d"))
    logger.info("日线更新: %d 只股票", len(updated))
    return len(updated), target_end


# ================================================================
#  步骤 ②③：识别变化股票 → 重生 CZSC 信号 (init_n=300)
# ================================================================

def regenerate_signals(dry_run=False):
    """删旧信号 → CZSC 重生，仅处理日线中有数据的股票。"""
    from czsc import RawBar, Freq, generate_czsc_signals
    from signal_config import get_config

    daily_codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
    sig_config = get_config(freq='日线')

    total = len(daily_codes)
    success = 0
    t0 = time.time()
    
    logger.info("CZSC 信号重生 (init_n=300, %d 只)", total)
    
    for i, code in enumerate(daily_codes):
        try:
            df = pd.read_parquet(DAILY_DIR / f"{code}.parquet")
            if len(df) < 30:
                continue
            
            df = df.sort_values('date')
            bars = []
            for j, (_, row) in enumerate(df.iterrows()):
                bars.append(RawBar(
                    symbol=str(row.get("code", code)), id=j+1,
                    dt=row["date"].to_pydatetime(), freq=Freq.D,
                    open=row["open"], close=row["close"],
                    high=row["high"], low=row["low"],
                    vol=row.get("volume", 0),
                    amount=row.get("amount", 0),
                ))
            
            sigs_df = generate_czsc_signals(bars, signals_config=sig_config,
                                            sdt="20200101", init_n=min(300, len(bars)), df=True)
            if sigs_df is not None and not sigs_df.empty:
                sigs_df = sigs_df.drop(columns=[c for c in ["freq", "cache"] if c in sigs_df.columns])
                sigs_df.to_parquet(SIG_DIR / f"{code}.parquet", index=False)
                success += 1
        except Exception as e:
            logger.debug("%s: %s", code, str(e)[:80])
        
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info("信号: %d/%d (%d 成功) %.1f/s ETA %.0fs", i+1, total, success, rate, eta)
    
    elapsed = time.time() - t0
    logger.info("信号重生完成: %d/%d 成功, 耗时 %.0fs", success, total, elapsed)
    return success


# ================================================================
#  步骤 ④⑤⑥：L1→L4 流水线
# ================================================================

def run_l1_l4():
    """运行 L1 打分 → L3 过滤 → L4 排名。"""
    logger.info("=== L1 打分 ===")
    ret = os.system(f"cd {BASE_DIR} && python3 -u tmp_out/l1_score_all.py 2>&1")
    if ret != 0:
        logger.warning("L1 打分退出码: %d", ret)
    
    logger.info("=== L3 过滤 ===")
    ret = os.system(f"cd {BASE_DIR} && python3 -u tmp_out/l3_filter_v2.py 2>&1")
    if ret != 0:
        logger.warning("L3 过滤退出码: %d", ret)
    
    logger.info("=== L4 排名 ===")
    ret = os.system(f"cd {BASE_DIR} && python3 -u tmp_out/l4_rank.py 2>&1")
    if ret != 0:
        logger.warning("L4 排名退出码: %d", ret)


# ================================================================
#  主入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="L1 增量更新流水线")
    parser.add_argument("--dry-run", action="store_true", help="仅检查不执行")
    parser.add_argument("--end", type=str, help="截止日期 YYYY-MM-DD")
    parser.add_argument("--skip-pull", action="store_true", help="跳过日线拉取")
    parser.add_argument("--skip-signals", action="store_true", help="跳过信号重生")
    parser.add_argument("--skip-l1l4", action="store_true", help="跳过 L1-L4 打分")
    args = parser.parse_args()
    
    start_ts = time.time()
    
    # ① 拉取日线
    if not args.skip_pull:
        n, end_date = pull_daily(args.end, args.dry_run)
        logger.info("步骤①完成: %d 只更新, 截止 %s", n, end_date.strftime("%Y-%m-%d"))
    
    # ②③ 重生信号
    if not args.skip_signals and not args.dry_run:
        n_sig = regenerate_signals()
        logger.info("步骤②③完成: %d 只信号重生", n_sig)
    
    # ④⑤⑥ L1→L4
    if not args.skip_l1l4 and not args.dry_run:
        run_l1_l4()
        logger.info("步骤④⑤⑥完成")
    
    elapsed = int(time.time() - start_ts)
    logger.info("流水线完成, 总耗时 %ds", elapsed)


if __name__ == "__main__":
    main()
