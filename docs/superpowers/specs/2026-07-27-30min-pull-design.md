# 30 分钟 K 线增量拉取设计

## 目标
将 `data/min30/` 中的 30 分钟 K 线数据更新至 2026-07-24。

## 使用的修正
1. **`_to_sina()` 前缀转换**：SZ→`sz{code}`，SH→`sh{code}`
2. **`resample("30min")`**：新版 pandas 废弃 `"30T"`
3. **`pd.to_numeric()`**：akshare 返回的 `volume`/`amount` 为字符串类型
4. **`time.sleep(0.25)`**：避免 akshare 限流返回空数据

## 执行方式
进程运行于 exec_command 前台会话 PID 2666。
- 处理：4998 只股票，每只 ~3.25s（0.25s 延迟 + 3s API）
- 成功率：~98%（基于实际命中率）
- ETA：~4.6 小时
- 日志：`tmp_out/pull_min30_wrapper.log`
- 数据：`data/min30/`

## 监控方式
```
tail -f tmp_out/pull_min30_wrapper.log
ls data/min30/*.parquet | wc -l
python3 -c "from pathlib import Path; import pandas as pd; d=Path('data/min30'); ok=sum(1 for f in d.glob('*.parquet') if pd.read_parquet(f,columns=['date'])['date'].max()>=pd.Timestamp('2026-07-24')); print(f'{ok}/{len(list(d.glob(\"*.parquet\")))} up to date')"
```
