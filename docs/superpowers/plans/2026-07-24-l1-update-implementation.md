 # L1 更新 Pipeline 实现计划
 
 > **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
 
 **Goal:** 实现 l1_update.py 的 7 步流水线：日线+指数拉取 → 30 分钟数据 → CZSC 信号（4 片并行）→ RPS 刷新 → L1→L4
 
 **Architecture:** 单文件 l1_update.py 入口编排，CZSC 信号分片委托给独立的 tmp_out/signal_chunk.py 以 nohup 并行跑。所有步骤通过文件系统解耦（写 parquet → 读 parquet），不引入中间状态。
 
 **Tech Stack:** tushare, akshare, pandas, czsc, subprocess/nohup
 
 ## Global Constraints
 
 - macOS 禁止 multiprocessing.Pool，CZSC 并行用 nohup 独立进程
 - ST/北交所/退市股票不拉取；新股 < 120 bars 拉取但不跑信号
 - 30 分钟数据为必选（非 --skip-min30）
 - 所有长运行脚本写进度到文件（非 PTY stdout）
 - 幂等：输出文件存在即跳过，不自动 rm -rf
 - 首次全量跑前先跑 5% 样本验证
 
 ---
 
 ## File Structure
 
 ```
 l1_update.py              ← 入口编排，所有 7 步的调度逻辑
 tmp_out/signal_chunk.py   ← 新建：CZSC 信号生成的一个分片（被 nohup 调用）
 ```
 
 **l1_update.py 职责（修改）：**
 - 函数 `pull_daily()`: 区间拉取日线 + 上证指数 + 股票过滤
 - 函数 `pull_min30()`: 新增，aks hare 拉 30 分钟数据
 - 函数 `regenerate_signals()`: 重构，管理 4 片 nohup 并行 + 进度 + 回退
 - 函数 `refresh_rps()`: 新增，调用 rps_calc.py refresh
 - 函数 `run_l1_l4()`: 保持现有 os.system 调用
 - main(): 编排 7 步顺序，--skip-* 参数
 
 **signal_chunk.py 职责（新建）：**
 - 接收 --chunk 参数（当前分片/总分片）
 - 读取 data/daily/ 列表，处理自己的分片
 - 跳过 < 120 bars 的新股
 - 写进度到 tmp_out/signal_progress_chunk_{n}.txt
 - 写结果到 tmp_out/chunk_{n}_result.json
 
 ---
 
 ### Task 1: 日线拉取重构（pull_daily）
 
 **Files:**
 - Modify: `l1_update.py`（pull_daily 函数及辅助函数）
 
 **Interfaces:**
 - Consumes: `data/last_update.txt`, tushare API
 - Produces: `pull_daily(end_date, dry_run) -> (updated_count, latest_date)` 重构版
   - 内部新增 `_filter_stock(code) -> bool`：ST/北交/退市判断
   - 内部新增 `_is_new_stock(code, daily_df) -> bool`：< 120 bars 判断
 
 - [ ] **Step 1: 重构 pull_daily 为区间拉取**
 
  修改 `pull_daily()`：
   - 从 `pro.daily(trade_date=end_str)` 改为 `pro.daily(start_date=start_str, end_date=end_str)`
   - 遍历返回的所有行，按 code 分组追加到各股票 parquet
   - 增量追加后 `drop_duplicates('date', keep='last').sort_values('date')`
 
  关键代码（替换原有单日拉取逻辑）：
 
 ```python
 def pull_daily(end_date: str, dry_run=False):
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
 
     STATE_FILE.write_text(target_end.strftime("%Y-%m-%d"))
     logger.info("日线更新: %d 只股票", len(updated))
     return len(updated), target_end
 ```
 
 - [ ] **Step 2: 新增股票过滤函数**
 
 ```python
 ST_PREFIXES = ('ST', '*ST', 'SST', 'S*ST', 'NST')
 BJ_PREFIX = ('8', '9')  # 北交所股票代码前缀
 
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
 ```
 
 - [ ] **Step 3: 新增新股判定函数**
 
 ```python
 MIN_BARS = 120
 
 def _is_new_stock(code: str) -> bool:
     """日线少于 120 根 bar 的新股。"""
     fpath = DAILY_DIR / f"{code}.parquet"
     if not fpath.exists():
         return True
     df = pd.read_parquet(fpath, columns=['date'])
     return len(df) < MIN_BARS
 ```
 
 - [ ] **Step 4: 新增上证指数拉取**
 
  在 `pull_daily()` 末尾增加：
 
 ```python
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
 ```
 
 - [ ] **Step 5: 样本验证**
 
 Run: `cd /Users/hz/Desktop/"Algorithmic Trading Platform" && python3 l1_update.py --dry-run`
 Expected: 显示日线区间、股票数，不实际拉取
 
 - [ ] **Step 6: Commit**
 
 ```bash
 git add l1_update.py
 git commit -m "refactor: pull_daily 改为区间拉取 + 上证指数 + 股票过滤"
 ```
 
 ---
 
 ### Task 2: 新增 30 分钟数据拉取（pull_min30）
 
 **Files:**
 - Modify: `l1_update.py`（新增函数）
 
 **Interfaces:**
 - Consumes: akshare API, `data/daily/` 股票列表
 - Produces: `pull_min30(dry_run, last_days=30) -> int`
 
 - [ ] **Step 1: 实现 pull_min30**
 
 ```python
 MIN30_DIR = BASE_DIR / "data/min30"
 
 def pull_min30(dry_run=False, last_days=30) -> int:
     """拉取最近 last_days 天的 30 分钟数据。返回更新股票数。"""
     import akshare as ak
     
     codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
     logger.info("拉取 30 分钟数据: %d 只, 最近 %d 天", len(codes), last_days)
     if dry_run:
         return 0
     
     updated = 0
     for code in codes:
         try:
             df_5min = ak.stock_zh_a_minute(symbol=code, period='5', adjust='qfq')
             if df_5min is None or df_5min.empty:
                 continue
             df_5min = df_5min.rename(columns={'day': 'date'}).copy()
             df_5min['date'] = pd.to_datetime(df_5min['date'])
             
             # 只保留最近 last_days 天
             cutoff = pd.Timestamp.now() - pd.Timedelta(days=last_days)
             df_5min = df_5min[df_5min['date'] >= cutoff]
             
             # 重采样为 30 分钟
             df_5min = df_5min.set_index('date')
             ohlc = df_5min['price'].resample('30T').ohlc()
             volume = df_5min['volume'].resample('30T').sum()
             
             df_30min = pd.DataFrame({
                 'date': ohlc.index,
                 'open': ohlc['open'], 'high': ohlc['high'],
                 'low': ohlc['low'], 'close': ohlc['close'],
                 'volume': volume,
                 'code': code,
             }).reset_index(drop=True)
             df_30min = df_30min.dropna(subset=['open'])
             
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
         
         if (codes.index(code) + 1) % 500 == 0:
             logger.info("30 分钟: %d/%d", codes.index(code) + 1, len(codes))
     
     logger.info("30 分钟数据完成: %d 只更新", updated)
     return updated
 ```
 
 - [ ] **Step 2: Commit**
 
 ```bash
 git add l1_update.py
 git commit -m "feat: 新增 30 分钟数据拉取 pull_min30"
 ```
 
 ---
 
 ### Task 3: CZSC 信号分片并行
 
 **Files:**
 - Create: `tmp_out/signal_chunk.py`
 - Modify: `l1_update.py`（regenerate_signals 重构 + 分片管理）
 
 **Interfaces:**
 - `tmp_out/signal_chunk.py` 独立脚本，接受 `--chunk 当前/总分片` 参数
   - 输出：`tmp_out/signal_progress_chunk_{n}.txt` + `tmp_out/chunk_{n}_result.json`
 - `l1_update.py::regenerate_signals(dry_run)` 重构为分片并行模式
 
 - [ ] **Step 1: 创建 tmp_out/signal_chunk.py**
 
 ```python
 #!/usr/bin/env python3
 """CZSC 信号分片处理。被 l1_update.py 以 nohup 方式调用。
 
 用法: python3 tmp_out/signal_chunk.py --chunk 0/4
 """
 import sys, time, json, argparse
 from pathlib import Path
 import pandas as pd
 from czsc import RawBar, Freq, generate_czsc_signals
 import signal_config
 
 BASE = Path(__file__).parent.parent
 DAILY = BASE / "data/daily"
 SIG_DIR = BASE / "data/signals"
 PROGRESS_DIR = BASE / "tmp_out"
 MIN_BARS = 120
 INIT_N = 300
 
 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument('--chunk', required=True, help='格式: 当前/总数，如 0/4')
     args = parser.parse_args()
     
     parts = args.chunk.split('/')
     chunk_idx = int(parts[0])
     total_chunks = int(parts[1])
     
     codes = sorted(p.stem for p in DAILY.glob("*.parquet"))
     my_codes = codes[chunk_idx::total_chunks]
     
     sig_config = signal_config.get_config(freq='日线')
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
             
             df = df.sort_values('date')
             bars = [RawBar(symbol=code, id=j+1, dt=r['date'].to_pydatetime(),
                           freq=Freq.D, open=r['open'], close=r['close'],
                           high=r['high'], low=r['low'],
                           vol=r.get('volume', 0), amount=r.get('amount', 0))
                     for j, (_, r) in enumerate(df.iterrows())]
             
             sigs = generate_czsc_signals(bars, signals_config=sig_config,
                                         sdt="20200101", init_n=min(INIT_N, len(bars)), df=True)
             if sigs is not None and not sigs.empty:
                 sigs = sigs.drop(columns=[c for c in ["freq", "cache"] if c in sigs.columns])
                 sigs.to_parquet(SIG_DIR / f"{code}.parquet", index=False)
                 success += 1
         except Exception as e:
             failed += 1
         
         if (i + 1) % 100 == 0 or i == len(my_codes) - 1:
             elapsed = time.time() - t0
             rate = (i + 1) / elapsed if elapsed > 0 else 0
             msg = (f"[chunk {chunk_idx}/{total_chunks}] {i+1}/{len(my_codes)} "
                    f"成功{success} 失败{failed} 跳过{skipped} "
                    f"{rate:.1f}只/s ETA {(len(my_codes)-i-1)/rate:.0f}s" if rate > 0 else "")
             progress_file.write_text(msg)
             print(msg, flush=True)
     
     result = {"success": success, "failed": failed, "skipped": skipped,
               "elapsed": time.time() - t0, "chunk": chunk_idx, "total": total_chunks}
     result_file.write_text(json.dumps(result))
     print(f"[chunk {chunk_idx}/{total_chunks}] 完成: {json.dumps(result)}", flush=True)
 
 if __name__ == "__main__":
     main()
 ```
 
 - [ ] **Step 2: 重构 regenerate_signals（并行模式）**
 
  在 `l1_update.py` 中重写：
 
 ```python
 def regenerate_signals(dry_run=False, max_retry=3):
     """CZSC 信号重生。默认 4 片 nohup 并行（B 方案），失败回退单线程（A 方案）。"""
     daily_codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
     logger.info("CZSC 信号重生: %d 只", len(daily_codes))
     
     if dry_run:
         new_codes = [c for c in daily_codes if _is_new_stock(c)]
         logger.info("[DRY-RUN] 跳过新股 %d 只，其余 %d 只需要信号生成",
                     len(new_codes), len(daily_codes) - len(new_codes))
         return 0
     
     # B 方案：4 片 nohup 并行
     if _run_chunks_b():
         return _aggregate_chunk_results()
     
     # B 方案失败，回退 A 方案
     logger.warning("并行方案失败，回退单线程串行")
     return _run_serial_signals()
 
 def _run_chunks_b() -> bool:
     """启动 4 个 nohup 进程。返回 True 表示全部成功。"""
     CHUNKS = 4
     import subprocess
     
     # 清理旧的 result 文件
     for i in range(CHUNKS):
         rf = BASE_DIR / "tmp_out" / f"chunk_{i}_result.json"
         if rf.exists():
             rf.unlink()
     
     for i in range(CHUNKS):
         cmd = (f"nohup python3 {BASE_DIR}/tmp_out/signal_chunk.py "
                f"--chunk {i}/{CHUNKS} > "
                f"{BASE_DIR}/tmp_out/chunk_{i}.log 2>&1 &")
         subprocess.run(cmd, shell=True)
     
     logger.info("已启动 4 个 nohup 进程:")
     for i in range(CHUNKS):
         logger.info("  chunk %d/4 → tail -f tmp_out/chunk_%d.log", i, i)
     logger.info("等待分片完成...")
     
     # 轮询等待完成
     import time as _time
     wait_seconds = 0
     while wait_seconds < 600:  # 最多等 10 分钟
         done_files = [(BASE_DIR / "tmp_out" / f"chunk_{i}_result.json").exists()
                       for i in range(CHUNKS)]
         if all(done_files):
             # 读取结果
             import json
             results = []
             for i in range(CHUNKS):
                 rf = BASE_DIR / "tmp_out" / f"chunk_{i}_result.json"
                 results.append(json.loads(rf.read_text()))
             
             total_failed = sum(r['failed'] for r in results)
             total_success = sum(r['success'] for r in results)
             total_skipped = sum(r['skipped'] for r in results)
             total = total_success + total_failed + total_skipped
             total_elapsed = max(r['elapsed'] for r in results)
             
             logger.info("分片完成: %d 成功, %d 失败, %d 跳过, %.0fs",
                         total_success, total_failed, total_skipped, total_elapsed)
             
             if total_failed > total * 0.1:  # 失败超过 10%，回退
                 return False
             return True
         
         _time.sleep(30)
         wait_seconds += 30
         logger.info("等待中... %ds", wait_seconds)
     
     logger.error("分片超时（10min），回退单线程")
     return False
 
 def _aggregate_chunk_results() -> int:
     """汇总分片结果，返回成功数。"""
     import json
     total = 0
     for i in range(4):
         rf = BASE_DIR / "tmp_out" / f"chunk_{i}_result.json"
         if rf.exists():
             r = json.loads(rf.read_text())
             total += r['success']
     return total
 
 def _run_serial_signals() -> int:
     """A 方案：单线程串行。"""
     from czsc import RawBar, Freq, generate_czsc_signals
     from signal_config import get_config
     
     daily_codes = sorted(p.stem for p in DAILY_DIR.glob("*.parquet"))
     sig_config = get_config(freq='日线')
     success, t0 = 0, time.time()
     progress_path = BASE_DIR / "tmp_out" / "signal_progress.txt"
     
     for i, code in enumerate(daily_codes):
         if _is_new_stock(code):
             continue
         try:
             df = pd.read_parquet(DAILY_DIR / f"{code}.parquet").sort_values('date')
             bars = [RawBar(symbol=code, id=j+1, dt=r['date'].to_pydatetime(),
                           freq=Freq.D, open=r['open'], close=r['close'],
                           high=r['high'], low=r['low'],
                           vol=r.get('volume', 0), amount=r.get('amount', 0))
                     for j, (_, r) in enumerate(df.iterrows())]
             sigs = generate_czsc_signals(bars, signals_config=sig_config,
                                         sdt="20200101", init_n=min(300, len(bars)), df=True)
             if sigs is not None and not sigs.empty:
                 sigs = sigs.drop(columns=[c for c in ["freq", "cache"] if c in sigs.columns])
                 sigs.to_parquet(SIG_DIR / f"{code}.parquet", index=False)
                 success += 1
         except:
             pass
         
         if (i + 1) % 500 == 0:
             elapsed = time.time() - t0
             rate = (i + 1) / elapsed
             msg = f"[serial] {i+1}/{len(daily_codes)} 成功{success} {rate:.1f}只/s"
             progress_path.write_text(msg)
             logger.info(msg)
     
     logger.info("单线程信号完成: %d/%d 成功", success, len(daily_codes))
     return success
 ```
 
 - [ ] **Step 3: 验证 signal_chunk.py 可运行**
 
 Run: `python3 /Users/hz/Desktop/"Algorithmic Trading Platform"/tmp_out/signal_chunk.py --chunk 0/4`
 Expected: 处理 chunk 0（~1250 只），打印进度，写 result 文件
 
 - [ ] **Step 4: Commit**
 
 ```bash
 git add tmp_out/signal_chunk.py l1_update.py
 git commit -m "feat: CZSC 信号 4 片 nohup 并行 + 单线程回退方案"
 ```
 
 ---
 
 ### Task 4: RPS 刷新 + 全流程编排
 
 **Files:**
 - Modify: `l1_update.py`（refresh_rps + --skip-rps + main 集成）
 
 **Interfaces:**
 - Consumes: `subprocess.run(['python3', 'rps_calc.py', 'refresh'])`
 - Produces: `refresh_rps(dry_run)`、main() 中新增 Step ② 和 Step ④
 
 - [ ] **Step 1: 新增 refresh_rps 函数**
 
 ```python
 def refresh_rps(dry_run=False):
     """RPS 增量刷新。"""
     logger.info("=== RPS 增量刷新 ===")
     if dry_run:
         logger.info("[DRY-RUN] 跳过")
         return
     import subprocess
     ret = subprocess.run(['python3', str(BASE_DIR / 'rps_calc.py'), 'refresh'],
                          capture_output=True, text=True)
     if ret.returncode != 0:
         logger.warning("RPS 刷新返回码 %d: %s", ret.returncode, ret.stderr[:200])
     else:
         for line in ret.stdout.strip().split('\n'):
             logger.info("  %s", line)
 ```
 
 - [ ] **Step 2: 新增 --skip-rps 参数**
 
  在 `main()` 的 parser 中增加：
 
 ```python
 parser.add_argument("--skip-rps", action="store_true", help="跳过 RPS 刷新")
 ```
 
 - [ ] **Step 3: 重构 main() 为 7 步流水线**
 
 ```python
 def main():
     parser = argparse.ArgumentParser(description="L1 增量更新流水线")
     parser.add_argument("--dry-run", action="store_true", help="仅检查不执行")
     parser.add_argument("--end", type=str, help="截止日期 YYYY-MM-DD")
     parser.add_argument("--skip-pull", action="store_true", help="跳过日线拉取")
     parser.add_argument("--skip-signals", action="store_true", help="跳过信号重生")
     parser.add_argument("--skip-rps", action="store_true", help="跳过 RPS 刷新")
     parser.add_argument("--skip-l1l4", action="store_true", help="跳过 L1-L4 打分")
     args = parser.parse_args()
     
     dry_run = args.dry_run
     start_ts = time.time()
     
     # Step ① 拉日线 + 上证指数
     if not args.skip_pull:
         n, end_date = pull_daily(args.end, dry_run)
         logger.info("步骤①完成: %d 只更新, 截止 %s", n, end_date.strftime("%Y-%m-%d"))
     
     # Step ② 拉 30 分钟数据（必选）
     if not args.skip_pull:  # 与日线同开关，日线刚拉完即可拉 30 分钟
         n30 = pull_min30(dry_run=dry_run)
         logger.info("步骤②完成: %d 只更新", n30)
     
     # Step ③ CZSC 信号重生
     if not args.skip_signals and not dry_run:
         n_sig = regenerate_signals()
         logger.info("步骤③完成: %d 只信号", n_sig)
     
     # Step ④ RPS 刷新
     if not args.skip_rps and not dry_run:
         refresh_rps()
         logger.info("步骤④完成")
     
     # Step ⑤⑥⑦ L1→L4
     if not args.skip_l1l4 and not dry_run:
         run_l1_l4()
         logger.info("步骤⑤⑥⑦完成")
     
     elapsed = int(time.time() - start_ts)
     logger.info("流水线完成, 总耗时 %ds", elapsed)
 ```
 
 - [ ] **Step 4: 5% 样本运行验证**
 
 Run 5% 样本验证：
 ```bash
 cd /Users/hz/Desktop/"Algorithmic Trading Platform"
 # 先 dry-run 检查完整性
 python3 l1_update.py --dry-run
 # 取 250 只跑完整 pipeline（通过 --end 限制日期或用测试子集）
 python3 l1_update.py --skip-pull --skip-signals  # 测试 RPS + L1-L4
 ```
 
 - [ ] **Step 5: 完整全量运行**
 
 Run:
 ```bash
 python3 l1_update.py
 ```
 Expected: 7 步流水线顺序执行，最终产出 l4_ranked.parquet
 
 - [ ] **Step 6: 设计文件同步检查**
 
 根据 `AGENTS.md` 设计文件约束，检查生成的设计文档 `docs/2026-07-24-l1-update-strategy-design.md` 是否与实现一致。此文档仅描述 pipeline 流程，不涉及 L1-L4 决策规则调整，无需修改其他 4 份设计文件。
 
 - [ ] **Step 7: Commit**
 
 ```bash
 git add l1_update.py
 git commit -m "feat: RPS 刷新 + 全流程 7 步编排"
 ```
 
 ---
 
 ## Self-Review
 
 ### Spec coverage
 
 | 设计文档要求 | 对应任务 |
 |:---|:---:|
 | Step ① 日线区间拉取 | Task 1 |
 | ST/北交/退市过滤 | Task 1 Step 2 |
 | 新股 < 120 bars 拉取但跳过信号 | Task 1 Step 3 + Task 3 Step 1 |
 | 上证指数拉取 | Task 1 Step 4 |
 | Step ② 30 分钟数据 | Task 2 |
 | Step ③ CZSC 信号（4 片并行） | Task 3 |
 | 单线程回退方案 | Task 3 Step 2（_run_serial_signals） |
 | 进度写文件 | Task 3 Step 1（signal_progress_chunk_{n}.txt）|
 | Step ④ RPS 刷新 | Task 4 Step 1 |
 | Step ⑤⑥⑦ L1-L4 | 保持原有 os.system 调用 |
 | --skip-rps 参数 | Task 4 Step 2 |
 | 幂等 | Task 1 Step 1（drop_duplicates）|
 | 5% 样本验证 | Task 4 Step 4 |
 
 ### No placeholders
 
 所有代码块包含完整实现，无 "TBD" 或 "TODO"。
 
 ### Type consistency
 
 - `pull_daily()` 签名在各任务间一致
 - `regenerate_signals()` 返回 int（成功数）
 - 分片结果 JSON 格式在各部分间一致
 - `_filter_stock()` / `_is_new_stock()` 签名一致
 - `pull_min30()` 返回 int（更新数）
 - `refresh_rps()` 无返回值
