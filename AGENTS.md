 
## 大模型自动化开发与回测智能体规范（2026-07-20 新增）

### 🛑 一、行为边界与硬熔断 (FinOps & Boundaries)

**1. 额度与流量控制**
- **硬限额（Hard Limit）：** Agent 持有的 API Key 必须在服务商后台配置日/月最大消费硬上限，严禁使用无额度限制的 Master Key。
- **并发限速（RPM Restriction）：** 自动化优化脚本循环体中调用 LLM 时，必须硬编码 `time.sleep(3)`（单次调用间隔 ≥ 3 秒），脚本级最大 RPM ≤ 20。

**2. 死循环拦截 (Max Retries Break)**
- **重试阈值：** Agent 针对同一逻辑错误的连续自动修复尝试不得超过 **3 次**。
- **强制退出：** `retry_count >= 3` 且仍未通过时，必须执行 `sys.exit(1)` 强行中断，保留现场并向开发者发出警报。

### 🔄 二、架构与流程约束 (Architectural Constraints)

采用**回测（本地计算）与生成（LLM 调用）解耦**的异步架构，消除 Token 放大效应：

| 阶段 | 行为 | Token 消耗 |
|------|------|-----------|
| **1. 本地离线回测** | Agent 启动本地回测框架运行测试集，属于 CPU/GPU 密集型任务 | 必须为 0 |
| **2. 错误与日志收集** | 自动收集失败 Case、指标数据及崩溃日志，汇总为本地 `error_manifest.json` | 必须为 0 |
| **3. 批处理优化投递** | 将错误清单打包，**单次**投递给 LLM 请求优化方案 | 单次 Batch 调用 |

严禁在回测循环的迭代内部高频嵌套调用 API。


### 有效信号日期规则（2026-08-12）

- 全量更新 + 回测中所有新鲜度检查（L1 信号日 <=10 天、周线/30min 末 bar <=7 天）统一使用 core/date_utils.get_effective_date()
- 16:00 前：返回昨天（当日 K 线数据未就绪）
- 16:00 后：返回今天
- 禁止在任何新鲜度闸门中使用裸 pd.Timestamp.now()

## 进程管理规则（2026-07-15 新增）

**暂停/续跑原则**：任何长时间运行脚本（标注、打分、Optuna）暂停后恢复时，必须先杀掉旧进程，再启动新进程。

```
❌ 错误：VM 里新旧两个 python3 同时写同一个 parquet → 数据竞争 + CPU 争抢
✅ 正确：ssh VM "killall -9 python3" → 确认已杀 → python3 resume.py
```

**操作流程**：
1. `killall -9 python3` 杀旧进程
2. `ps aux | grep python3` 确认清理完毕
3. 启动续跑脚本
4. 定期验证输出文件 mtime 和数据量是否增长

## CZSC 运行环境规则（2026-07-17 新增）

**优先级**：本地 macOS > colima VM/Linux。

- CZSC 在 Intel Mac 上已验证稳定（100 只/0 崩溃，220ms/只）
- 所有 CZSC 相关脚本（打分、struct_df 预计算、信号生成）默认在 macOS 本地跑
- **仅在 macOS 上出现 segfault 或崩溃时**，先报告用户确认，再切换到 colima VM 执行
- 禁止未经确认自行切到 VM



---

## CZSC 信号并行重生规则（2026-07-29 新增）

**原则：Codex 沙箱内 `&` 后台进程会随 `exec_command` 退出被 SIGHUP 清理，`nohup`/`disown` 均无效。**

### 全量更新前置动作（2026-08-11）

- **启动全量更新流水线前，必须先执行 `killall -9 Python`**——清除之前失败的 nohup/disown/& 尝试遗留的僵尸 signal_chunk 进程
- 僵尸进程会抢 CPU（Intel Mac 4 核跑 10 个 worker → 单核效率降 60%），导致耗时膨胀 3 倍以上
- 执行后 `ps aux | grep python | grep -v grep` 确认清零，再启动流水线

### 沙箱边界（2026-08-11 踩坑补充）

- `exec_command` session 退出时，沙箱会杀死该 session 内创建的所有进程（包括 `&` 后台、`nohup`、`disown`），无一例外
- **任何跨 session 的后台任务都是不可行的**——每当你开始新 exec_command，上一次的后台进程已死
- 唯一存活方案：在**单个 exec_command 前台 session** 内完成所有工作，用 `write_stdin` 定时空轮询（≤30s 间隔）维持 session

### subprocess.run 陷阱（2026-08-11）

- **禁止用 `subprocess.run` 包装 `run_signal_parallel.py`**（如 `l1_update.py`、`full_pipeline.py` 中的用法）
- `subprocess.run` 创建中间层进程，导致 signal_chunk workers（孙进程）在 session 退出时被连带清理
- 直接运行 `python3 -u tmp_out/run_signal_parallel.py`：workers 通过 `subprocess.Popen` 从直属父进程 spawn，存活正常
- 流水线脚本如需串联多步骤，用 `subprocess.Popen` + `p.wait()` 而非 `subprocess.run`
  [chunk 1/4] 0/1302 就绪
  [chunk 2/4] 0/1302 就绪
  [chunk 3/4] 0/1302 就绪
  [chunk 1/4] 0/1302 就绪
  [chunk 2/4] 0/1302 就绪
  [chunk 3/4] 0/1302 就绪
  [chunk 1/4] 0/1302 就绪
  [chunk 2/4] 0/1302 就绪
  [chunk 3/4] 0/1302 就绪
  [chunk 1/4] 0/1302 就绪
  [chunk 2/4] 10/1302 成功7 失败0 跳过3 0.2只/s ETA 8498s
  [chunk 3/4] 0/1302 就绪
  [chunk 0/4] 10/1302 成功10 失败0 跳过0 0.1只/s ETA 12493s
  [chunk 1/4] 10/1302 成功10 失败0 跳过0 0.1只/s ETA 12506s
  [chunk 2/4] 10/1302 成功7 失败0 跳过3 0.2只/s ETA 8498s
  [chunk 3/4] 10/1302 成功10 失败0 跳过0 0.1只/s ETA 12296s
  [chunk 0/4] 10/1302 成功10 失败0 跳过0 0.1只/s ETA 12493s
  [chunk 1/4] 10/1302 成功10 失败0 跳过0 0.1只/s ETA 12506s
  [chunk 2/4] 30/1302 成功24 失败1 跳过5 0.2只/s ETA 8313s
  [chunk 3/4] 10/1302 成功10 失败0 跳过0 0.1只/s ETA 12296s
  [chunk 0/4] 10/1302 成功10 失败0 跳过0 0.1只/s ETA 12493s
  [chunk 1/4] 10/1302 成功10 失败0 跳过0 0.1只/s ETA 12506s
  [chunk 2/4] 20/1302 成功16 失败0 跳过4 0.1只/s ETA 10026s
  [chunk 3/4] 10/1302 成功10 失败0 跳过0 0.1只/s ETA 12296s
  [chunk 0/4] 20/1302 成功20 失败0 跳过0 0.1只/s ETA 12137s
  [chunk 1/4] 20/1302 成功19 失败1 跳过0 0.1只/s ETA 11678s
  [chunk 2/4] 20/1302 成功16 失败0 跳过4 0.1只/s ETA 10026s
  [chunk 3/4] 20/1302 成功20 失败0 跳过0 0.1只/s ETA 12057s
  [chunk 0/4] 20/1302 成功20 失败0 跳过0 0.1只/s ETA 12137s
  [chunk 1/4] 20/1302 成功19 失败1 跳过0 0.1只/s ETA 11678s
  [chunk 2/4] 30/1302 成功24 失败1 跳过5 0.1只/s ETA 9388s
  [chunk 3/4] 20/1302 成功20 失败0 跳过0 0.1只/s ETA 12057s
  [chunk 0/4] 20/1302 成功20 失败0 跳过0 0.1只/s ETA 12137s
  [chunk 1/4] 20/1302 成功19 失败1 跳过0 0.1只/s ETA 11678s
  [chunk 2/4] 30/1302 成功24 失败1 跳过5 0.1只/s ETA 9388s
  [chunk 3/4] 20/1302 成功20 失败0 跳过0 0.1只/s ETA 12057s
  [chunk 0/4] 30/1302 成功30 失败0 跳过0 0.1只/s ETA 11319s
  [chunk 1/4] 30/1302 成功29 失败1 跳过0 0.1只/s ETA 10996s
  [chunk 2/4] 50/1302 成功41 失败1 跳过8 0.1只/s ETA 8375s
  [chunk 3/4] 30/1302 成功30 失败0 跳过0 0.1只/s ETA 11217s
  [chunk 0/4] 30/1302 成功30 失败0 跳过0 0.1只/s ETA 11319s
  [chunk 1/4] 30/1302 成功29 失败1 跳过0 0.1只/s ETA 10996s
  [chunk 2/4] 40/1302 成功34 失败1 跳过5 0.1只/s ETA 9413s
  [chunk 3/4] 30/1302 成功30 失败0 跳过0 0.1只/s ETA 11217s
  [chunk 0/4] 40/1302 成功40 失败0 跳过0 0.1只/s ETA 10849s
  [chunk 1/4] 40/1302 成功39 失败1 跳过0 0.1只/s ETA 10637s
  [chunk 2/4] 60/1302 成功50 失败1 跳过9 0.1只/s ETA 8369s
  [chunk 3/4] 40/1302 成功40 失败0 跳过0 0.1只/s ETA 10781s
  [chunk 0/4] 40/1302 成功40 失败0 跳过0 0.1只/s ETA 10849s
  [chunk 1/4] 40/1302 成功39 失败1 跳过0 0.1只/s ETA 10637s
  [chunk 2/4] 50/1302 成功41 失败1 跳过8 0.1只/s ETA 8829s
  [chunk 3/4] 40/1302 成功40 失败0 跳过0 0.1只/s ETA 10781s
  [chunk 0/4] 40/1302 成功40 失败0 跳过0 0.1只/s ETA 10849s
  [chunk 1/4] 60/1302 成功56 失败2 跳过2 0.1只/s ETA 9340s
  [chunk 2/4] 50/1302 成功41 失败1 跳过8 0.1只/s ETA 8829s
  [chunk 3/4] 60/1302 成功57 失败1 跳过2 0.1只/s ETA 9423s
  [chunk 0/4] 50/1302 成功49 失败0 跳过1 0.1只/s ETA 10332s
  [chunk 1/4] 50/1302 成功48 失败1 跳过1 0.1只/s ETA 10155s
  [chunk 2/4] 60/1302 成功50 失败1 跳过9 0.1只/s ETA 8725s
  [chunk 3/4] 50/1302 成功49 失败0 跳过1 0.1只/s ETA 10163s
  [chunk 0/4] 50/1302 成功49 失败0 跳过1 0.1只/s ETA 10332s
  [chunk 1/4] 70/1302 成功65 失败2 跳过3 0.1只/s ETA 9129s
  [chunk 2/4] 80/1302 成功65 失败3 跳过12 0.2只/s ETA 7930s
  [chunk 3/4] 50/1302 成功49 失败0 跳过1 0.1只/s ETA 10163s
  [chunk 0/4] 60/1302 成功59 失败0 跳过1 0.1只/s ETA 10178s
  [chunk 1/4] 60/1302 成功56 失败2 跳过2 0.1只/s ETA 9720s
  [chunk 2/4] 70/1302 成功59 失败1 跳过10 0.1只/s ETA 8677s
  [chunk 3/4] 70/1302 成功67 失败1 跳过2 0.1只/s ETA 9418s
  [chunk 0/4] 60/1302 成功59 失败0 跳过1 0.1只/s ETA 10178s
  [chunk 1/4] 60/1302 成功56 失败2 跳过2 0.1只/s ETA 9720s
  [chunk 2/4] 90/1302 成功73 失败3 跳过14 0.2只/s ETA 7855s
  [chunk 3/4] 80/1302 成功73 失败3 跳过4 0.1只/s ETA 8895s
  [chunk 0/4] 60/1302 成功59 失败0 跳过1 0.1只/s ETA 10178s
  [chunk 1/4] 70/1302 成功65 失败2 跳过3 0.1只/s ETA 9450s
  [chunk 2/4] 80/1302 成功65 失败3 跳过12 0.1只/s ETA 8245s
  [chunk 3/4] 70/1302 成功67 失败1 跳过2 0.1只/s ETA 9609s
  [chunk 0/4] 70/1302 成功69 失败0 跳过1 0.1只/s ETA 10070s
  [chunk 1/4] 70/1302 成功65 失败2 跳过3 0.1只/s ETA 9450s
  [chunk 2/4] 80/1302 成功65 失败3 跳过12 0.1只/s ETA 8245s
  [chunk 3/4] 70/1302 成功67 失败1 跳过2 0.1只/s ETA 9609s
  [chunk 0/4] 70/1302 成功69 失败0 跳过1 0.1只/s ETA 10070s
  [chunk 1/4] 90/1302 成功83 失败4 跳过3 0.1只/s ETA 9030s
  [chunk 2/4] 90/1302 成功73 失败3 跳过14 0.1只/s ETA 8267s
  [chunk 3/4] 80/1302 成功73 失败3 跳过4 0.1只/s ETA 9210s
  [chunk 0/4] 70/1302 成功69 失败0 跳过1 0.1只/s ETA 10070s
  [chunk 1/4] 80/1302 成功75 失败2 跳过3 0.1只/s ETA 9586s
  [chunk 2/4] 90/1302 成功73 失败3 跳过14 0.1只/s ETA 8267s
  [chunk 3/4] 80/1302 成功73 失败3 跳过4 0.1只/s ETA 9210s
  [chunk 0/4] 80/1302 成功79 失败0 跳过1 0.1只/s ETA 10161s
  [chunk 1/4] 80/1302 成功75 失败2 跳过3 0.1只/s ETA 9586s
  [chunk 2/4] 110/1302 成功93 失败3 跳过14 0.1只/s ETA 7978s
  [chunk 3/4] 80/1302 成功73 失败3 跳过4 0.1只/s ETA 9210s
  [chunk 0/4] 80/1302 成功79 失败0 跳过1 0.1只/s ETA 10161s
  [chunk 1/4] 90/1302 成功83 失败4 跳过3 0.1只/s ETA 9413s
  [chunk 2/4] 120/1302 成功100 失败5 跳过15 0.2只/s ETA 7568s
  [chunk 3/4] 100/1302 成功92 失败4 跳过4 0.1只/s ETA 9055s
  [chunk 0/4] 100/1302 成功97 失败1 跳过2 0.1只/s ETA 9514s
  [chunk 1/4] 90/1302 成功83 失败4 跳过3 0.1只/s ETA 9413s
  [chunk 2/4] 120/1302 成功100 失败5 跳过15 0.2只/s ETA 7568s
  [chunk 3/4] 100/1302 成功92 失败4 跳过4 0.1只/s ETA 9055s
  [chunk 0/4] 90/1302 成功88 失败0 跳过2 0.1只/s ETA 10053s
  [chunk 1/4] 110/1302 成功101 失败5 跳过4 0.1只/s ETA 8872s
  [chunk 2/4] 130/1302 成功109 失败6 跳过15 0.2只/s ETA 7258s
  [chunk 3/4] 110/1302 成功100 失败6 跳过4 0.1只/s ETA 8803s
  [chunk 0/4] 90/1302 成功88 失败0 跳过2 0.1只/s ETA 10053s
  [chunk 1/4] 120/1302 成功110 失败6 跳过4 0.1只/s ETA 8418s
  [chunk 2/4] 110/1302 成功93 失败3 跳过14 0.1只/s ETA 8359s
  [chunk 3/4] 120/1302 成功107 失败7 跳过6 0.1只/s ETA 8327s
  [chunk 0/4] 110/1302 成功107 失败1 跳过2 0.1只/s ETA 9326s
  [chunk 1/4] 130/1302 成功118 失败7 跳过5 0.1只/s ETA 7889s
  [chunk 2/4] 140/1302 成功118 失败6 跳过16 0.2只/s ETA 7270s
  [chunk 3/4] 130/1302 成功116 失败8 跳过6 0.1只/s ETA 7902s
  [chunk 0/4] 120/1302 成功115 失败1 跳过4 0.1只/s ETA 8898s
  [chunk 1/4] 110/1302 成功101 失败5 跳过4 0.1只/s ETA 9163s
  [chunk 2/4] 130/1302 成功109 失败6 跳过15 0.2只/s ETA 7569s
  [chunk 3/4] 110/1302 成功100 失败6 跳过4 0.1只/s ETA 9009s
  [chunk 0/4] 130/1302 成功125 失败1 跳过4 0.1只/s ETA 8452s
  [chunk 1/4] 120/1302 成功110 失败6 跳过4 0.1只/s ETA 8647s
  [chunk 2/4] 130/1302 成功109 失败6 跳过15 0.2只/s ETA 7569s
  [chunk 3/4] 120/1302 成功107 失败7 跳过6 0.1只/s ETA 8470s
- 流水线脚本如需串联多步骤，用  +  而非
### 沙箱边界（2026-08-11 踩坑补充）

- exec_command session 退出时，沙箱会杀死该 session 内创建的所有进程（包括 & 后台、nohup、disown），无一例外
- **任何跨 session 的后台任务都是不可行的**——每当你开始新 exec_command，上一次的后台进程已死
- 唯一存活方案：在**单个 exec_command 前台 session** 内完成所有工作，用 write_stdin 定时空轮询（≤30s 间隔）维持 session

### subprocess.run 陷阱（2026-08-11）

- **禁止用 subprocess.run 包装 run_signal_parallel.py**（如 l1_update.py、full_pipeline.py 中的用法）
- subprocess.run 创建中间层进程，导致 signal_chunk workers（孙进程）在 session 退出时被连带清理
- 直接运行 python3 -u tmp_out/run_signal_parallel.py：workers 通过 subprocess.Popen 从直属父进程 spawn，存活正常
- 流水线脚本如需串联多步骤，用 subprocess.Popen + p.wait() 而非 subprocess.run
### 启动方式

- 必须用 `subprocess.Popen` 在**单个 `exec_command`** 内启动 N 个 `signal_chunk.py` 并行
- 使用 `tmp_out/run_signal_parallel.py` 启动器，默认 **6 核**
- 禁止 `&` / `nohup` 跨 `exec_command` 后台进程
- 启动后 Agent 必须用 write_stdin 定期空轮询（≤30s 间隔）维持 session，否则 session 超时退出 => 全部子进程被杀

### 命令

```bash
python3 tmp_out/run_signal_parallel.py                 # 日线信号, 4 核
python3 tmp_out/run_signal_parallel.py --workers 4     # 4 核
python3 tmp_out/run_signal_parallel.py --freq 周线     # 周线信号
python3 tmp_out/run_signal_parallel.py --dry-run       # 仅预览
```

### 核数选择

| 核数 | 全量 ~4998 只（全量 bar） | 增量（最后 350 bar） |
|------|--------------------------|---------------------|
| 4 (默认) | ~75 分钟 | ~7 分钟 |
| 6 | ~55 分钟 | ~5 分钟 |


### signal_chunk.py 开发规范

1. **`--freq` 参数必支持**：launcher 传入的 `--freq` 必须在 chunk 脚本中声明并映射到正确的数据/信号目录和 `Freq` 枚举。

   ```python
   FREQ_MAP = {
       "日线":   {"data": "daily",  "signals": "signals",         "freq": Freq.D,   "min_bars": 120},
       "周线":   {"data": "weekly", "signals": "signals_weekly",  "freq": Freq.W,   "min_bars": 60},
       "30分钟": {"data": "min30",  "signals": "signals_30min",   "freq": Freq.F30, "min_bars": 200},
   }
   ```

2. **tqdm 必须关闭**：`generate_czsc_signals(..., tqdm_kwargs={"disable": True})`。不关 tqdm 导致 log 膨胀（197KB/chunk/4min → ~16MB 全量）。

3. **进度每 10 只写一次**（不是 100）：每核 ~833 只，100 只一次 = 首个进度点 ~8 分钟，launcher 无法感知进程存活。

   ```python
   if (i + 1) % 10 == 0 or i == len(my_codes) - 1:
       progress_file.write_text(msg)
   ```

4. **启动即写进度**：循环前写 `0/{total} 就绪`，确认进程已进入主循环。

5. **异常必须写 log**：`except Exception as e: print(f"FAIL {code}: {e}", file=sys.stderr, flush=True)`。静默 `failed += 1` 导致全量跑完后无法排查。

### run_signal_parallel.py 开发规范

1. **文件句柄显式管理**：`subprocess.Popen` 的 stdout/stderr 重定向文件必须在退出时 close。

   ```python
   log_file = open(str(log_path), "w")
   p = subprocess.Popen(..., stdout=log_file, stderr=subprocess.STDOUT)
   procs.append((p, log_file))  # 结束后统一 close
   ```

2. **dry-run 按 `--freq` 查对应目录**：不能硬编码 `data/daily`。


## 设计文件约束（2026-07-24 新增）

以下 5 份设计文件定义管道的决策逻辑与数据更新流程，代码变更必须同步更新对应文件：

| 文件 | 对应代码 |
|------|----------|
| `docs/L1买点标签规则.md` | `verify_buy_type.py`, `zone1_deposition.py` |
| `docs/L2 Regime路由规则.md` | `entry_filter.py`, `zone2_regime.py` |
| `docs/L3 质量过滤规则.md` | `l3_filter.py`, `zone3_regime.py` |
| `docs/L4 排名规则.md` | `l4_ranker.py`, `zone4_regime.py` |
| `docs/全量更新流水线.md` | `l1_update.py`, `tmp_out/run_signal_parallel.py`, `tmp_out/signal_chunk.py` |

### 同步规则

1. **修改阈值/权重/条件** → 必须同步更新设计文件中的对应数字和逻辑描述
2. **新增维度/标签** → 设计文件加新行 + 注释生效日期
3. **删除维度/标签** → 设计文件中标注 `[已废弃 YYYY-MM-DD]`，不删除行
4. **每次管道代码修改完成后** → 检查设计文件是否需要更新


## 线程上下文管理规则（2026-07-20 新增）

**Thread hygiene**：当单线程累计消耗超过 ~500K tokens 时（逐级递增的趋势说明上下文已饱和），在响应末尾提示当前线程已接近长上下文边界，建议存档后开新线程，避免进入单次调用数百万 tokens 的指数增长区。

## CodeGraph 健康检查规则（2026-07-29 新增）

> 2026-07-28 踩坑：CodeGraph daemon 被 watchdog 杀死 4 次（主线程 60s 无响应），
> 崩溃期间 Codex 回退到 grep/read 模式，单次线程消耗 87.6M tokens。

**原则：每次 Codex 会话开始时，必须先确认 CodeGraph 可用。**

### 启动检查流程

1. **会话首轮**：Agent 必须调用一次 `codegraph_explore` 验证 daemon 响应
2. **daemon 不可用时**：
   - 检查 `.codegraph/daemon.log` 是否有 `unresponsive` 或 `killing` 关键字
   - 如连续崩溃 ≥ 2 次：设置 `CODEGRAPH_NO_WATCHDOG=1` 临时禁用 watchdog
   - 如仍不可用：重建索引 `codegraph index --force`
3. **无需 CodeGraph 的纯执行类任务**（跑脚本、看结果）可跳过检查

### 崩溃恢复

```
✅ PID 文件存在且 daemon 响应 → 正常使用 CodeGraph
✅ daemon 崩溃 → 先排查 log，再决定重启/重建索引
❌ daemon 崩溃后 Agent 默默回退 grep → 上下文指数膨胀
```

---

## 大规模脚本运行规范（2026-07-23 新增）

> 2026-07-22 EntryFilter 标定 + ExitEngine 回测踩坑记录。单次会话消耗超 500K tokens，70% 浪费在 PTY 输出轮询、内联代码重发、macOS spawn 排查。

### 一、输出规范

**原则：不依赖 PTY stdout 捕获进度。**

- 所有长运行脚本（>30s）必须写进度到文件（`open(path, "w") + f.write(msg)`）
- Agent 通过读文件获取进度，禁止反复 `write_stdin` 轮询
- 终输出写入 `.parquet` 或 `.json`，不通过 stdout 传数据

```
✅ python3 script.py → agent 读 progress.txt + output.parquet
❌ python3 -u script.py → agent 反复 write_stdin 等 flush
```

### 二、脚本载体规范

**原则：只用文件，不用内联代码。**

- 禁止 `python3 -c "大段内联代码..."`（token 浪费 + shell 转义 bug）
- 所有 ≥5 行的代码写入独立 `.py` 文件再执行
- 修改后直接 `apply_patch` 改文件，不重发整个命令

```
✅ cat > tmp_out/script.py << 'PYEOF'\n...\nPYEOF\npython3 tmp_out/script.py
❌ python3 -c "import sys; ...; sys.exit(0)"
```

### 三、macOS 并行规范

**原则：macOS 上不用 `multiprocessing.Pool`。**

- macOS 默认 `spawn` 模式，子进程不继承父进程 import，导致静默失败
- 需并行时用 `nohup python3 script.py &` 独立 Python 进程，互不干扰
- 优先单线程 + 进度写盘（2-4 小时内可接受的首选方案）

```
✅ nohup python3 script.py --chunk 0/8 & （8 个独立进程）
✅ python3 script.py（单线程，进度写文件）
❌ Pool(8) on macOS（spawn + czsc import fail → 全部静默返回）
```

### 四、样本验证规范

**原则：任何大规模脚本（>1000 个文件/任务）必须先跑全量数据的 5% 样本。**

- 样本验证通过后再全量运行
- 验证项至少包括：产出文件数量、数据字段正确性、耗时/只 估算
- 全量启动前向用户确认

```
✅ validate: 5745 × 5% = 287 stocks × 2.3s = 11min → 确认产出正确 → 启动全量
❌ 5745 stocks 直接跑 → 跑完发现全部失败 → 重跑消耗 token
```

### 五、幂等与续跑

**原则：所有数据构建脚本必须支持续跑。**

- 输出文件存在即跳过（`if out_path.exists(): continue`）
- 删除旧输出必须显式确认（不自动 `rm -rf`）
- 中断后用同一命令续跑，不丢数据

```
✅ for f in files:
       if out_path.exists(): continue  # 续跑
       process(f)
❌ rm -rf output_dir && process_all()  # 中断后从头跑
```


### 六、批量任务 PTY 输出规范（2026-07-28 新增）

> 2026-07-28 30m 信号生成踩坑：`generate_czsc_signals` 内部 `tqdm` 进度条
> 逐 bar 打印更新，67 只股票 x 6190 bar x ~50 字符/更新 -> ~100K token
> 被 PTY 捕获后视为 token 消耗。
>
> 不限于此——**任何批量任务产生的大规模 stdout 都会被 PTY 捕获为 token**。

**原则：批量脚本的三类输出必须去 stdout 化。**

| 输出类型 | 目标 | 示例 |
|----------|------|------|
| 进度 | 写磁盘文件 | `open("progress.txt","w") + f.write(msg)` |
| 最终结果 | `.parquet` / `.json` / `.csv` | `df.to_parquet(out_path)` |
| 调试/错误 | 重定向到 `.log` 文件 | `python3 script.py > script.log 2>&1` |

**批量脚本模板**：

```python
# 进度写文件，不写 stdout
with open(progress_file, "w") as pf:
    pf.write(f"{done}/{total}
")

# 结果写 parquet，不打印
df.to_parquet(out_path, index=False)
```

**三类调用必须静默**：

1. **CZSC / tqdm 类**：
   ```python
   # ✅ 关 tqdm
   generate_czsc_signals(bars, signals_config=config, df=True,
                         tqdm_kwargs={"disable": True})
   # ✅ 或 stdout 重定向
   python3 script.py > script.log 2>&1
   ```

2. **pandas / dataframe 类**：
   ```python
   # ❌ print(df) / df.to_string()
   # ✅ df.to_parquet("output.parquet")
   # ✅ df.to_csv("output.csv", index=False)
   ```

3. **子进程类**：
   ```python
   # ❌ subprocess.Popen(cmd)  # stdout 继承 PTY
   # ✅ subprocess.Popen(cmd, stdout=open("cmd.log","w"), stderr=subprocess.STDOUT)
   ```

**禁止项**：

- 禁止批量运行时不关 tqdm
- 禁止 `python3 -u`（无缓冲模式）在批量脚本中使用
- 禁止批量脚本 print 大数据到 stdout
- 禁止 Agent 通过 `write_stdin` 轮询 PTY 输出获取进度

```
✅ python3 script.py > script.log 2>&1  # stdout 不进 PTY
✅ 进度写文件，Agent 读文件
❌ python3 -u script.py                # 无缓冲 -> tqdm 全量刷 PTY
❌ 反复 write_stdin 等 stdout 输出
```
✅ generate_czsc_signals(..., tqdm_kwargs={"disable": True})
❌ generate_czsc_signals(...)   # 批量跑不关 tqdm -> token 灾难
```
