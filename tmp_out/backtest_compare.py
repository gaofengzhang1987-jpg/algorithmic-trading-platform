#!/usr/bin/env python3
"""三组 EntryFilter 配置对比回测：用预计算 dims + struct_cache，跳 CZSC。"""
import pandas as pd, numpy as np, time, json
from pathlib import Path

WORKDIR = Path("/Users/hz/Desktop/Algorithmic Trading Platform")
DAILY = WORKDIR / "data/daily"
STRUCT_CACHE_DIR = WORKDIR / "data" / "struct_cache"
COST_STOP_C = {"BULL": 0.92, "BEAR": 0.92, "CHOP": 0.88}

# ====== 三套配置 ======
OLD_WEIGHTS = {
    "一买": {"BEAR": {"核心验证": 1.34, "底分型": 0.43, "MACD": 0.89, "量比": 0.98, "中枢": 0.94, "距离": 0.89},
             "BULL": {"核心验证": 1.21, "底分型": 0.89, "MACD": 0.05, "量比": 0.83, "中枢": 0.94, "距离": 1.87}},
    "二买": {"BEAR": {"核心验证": 1.68, "量比": 0.00, "MACD": 1.57, "笔数": 1.00, "底分型": 0.78},
             "BULL": {"核心验证": 2.11, "量比": 0.16, "MACD": 1.38, "笔数": 1.28, "底分型": 1.61}},
    "三买": {"BEAR": {"核心验证": 1.12, "量比": 0.05, "MACD": 1.08, "笔数": 1.68, "底分型": 1.96},
             "BULL": {"核心验证": 1.07, "量比": 0.26, "MACD": 1.99, "笔数": 1.45, "底分型": 0.48}},
}
OLD_THRESHOLDS = {"一买": {"BEAR": 255, "BULL": 999}, "二买": {"BEAR": 273, "BULL": 327}, "三买": {"BEAR": 999, "BULL": 366}}

NEW_WEIGHTS = {
    "一买": {"BEAR": {"中枢": 0.57, "量比": 0.44, "MACD": 0.29, "底分型": 0.13, "核心验证": 0.10, "底部放量": 0.10},
             "BULL": {"量比": 0.70, "中枢": 0.45, "MACD": 0.39, "核心验证": 0.34, "距离": 0.26}},
    "二买": {"BEAR": {"MACD": 1.52, "量比": 0.90, "底分型": 0.44, "核心验证": 0.40},
             "BULL": {"MA防守": 2.96, "MACD": 0.99, "笔数": 0.98, "底分型": 0.08}},
    "三买": {"BEAR": {"笔数": 1.95, "量比": 0.96, "MACD": 0.59, "核心验证": 0.20, "底分型": 0.10},
             "BULL": {"MA排列强度": 1.09, "笔数": 0.90, "MACD": 0.72}},
}

CONFIGS = [
    ("baseline_A", OLD_WEIGHTS, {"一买": {"BEAR": 255, "BULL": 999}, "二买": {"BEAR": 273, "BULL": 327}, "三买": {"BEAR": 999, "BULL": 366}}),
    ("baseline_B", NEW_WEIGHTS, {"一买": {"BEAR": 70, "BULL": 999}, "二买": {"BEAR": 164, "BULL": 412}, "三买": {"BEAR": 999, "BULL": 190}}),
    ("experiment", NEW_WEIGHTS, {"一买": {"BEAR": 70, "BULL": 144}, "二买": {"BEAR": 164, "BULL": 412}, "三买": {"BEAR": 90, "BULL": 190}}),
]

# ====== ExitEngine (from backtest_full.py) ======
def simulate_exit(code, entry_price, entry_date, buy_type, struct_df, future_df, regime="CHOP"):
    sdf = struct_df
    entry_d = pd.Timestamp(entry_date).date()
    down_bis = sdf[(sdf["direction"].str.contains("向下", na=False)) & (pd.to_datetime(sdf["edt"]).dt.date <= entry_d)] if sdf is not None and not sdf.empty else pd.DataFrame()
    entry_bi_low_1buy = float(down_bis.iloc[-1]["low"]) if len(down_bis) > 0 else 0
    if entry_bi_low_1buy <= 0: entry_bi_low_1buy = entry_price * 0.92
    fx_before = []
    if sdf is not None and not sdf.empty:
        for _, row in sdf.iterrows():
            if pd.to_datetime(row["edt"]).date() <= entry_d and "底分型" in str(row.get("fx_b_mark", "")):
                fx_before.append(float(row.get("fx_b_low", 0)))
    entry_bi_low_2buy = fx_before[-1] if fx_before else entry_bi_low_1buy
    up_pivots = sdf[(sdf["pivot_dir"] == "上涨") & (pd.to_datetime(sdf["sdt"]).dt.date <= entry_d)] if sdf is not None and not sdf.empty else pd.DataFrame()
    entry_pivot_gg = float(up_pivots.iloc[-1]["pivot_gg"]) if len(up_pivots) > 0 else 0
    if entry_pivot_gg <= 0: entry_pivot_gg = entry_price * 0.95
    if "一买" in str(buy_type):
        low_val = entry_bi_low_1buy
        if low_val <= 0: low_val = entry_bi_low_2buy or entry_price * 0.96
        init_stop = low_val * 0.96
    elif "二买" in str(buy_type):
        base = max(entry_bi_low_1buy, entry_bi_low_2buy)
        if base <= 0: base = entry_price * 0.92
        init_stop = base * 0.92
    elif "三买" in str(buy_type):
        base = max(entry_pivot_gg, entry_bi_low_2buy)
        if base <= 0: base = entry_price * 0.95
        init_stop = base * 0.95
    else:
        init_stop = entry_price * 0.92
    defense = init_stop; exit_date = None; exit_price = None; exit_reason = ''
    for _, bar in future_df.iterrows():
        bar_d = bar['date']; bar_l = bar['low']; bar_c = bar['close']
        days = (bar_d - pd.Timestamp(entry_date)).days
        if sdf is not None and not sdf.empty and days > 0:
            entry_dt = pd.Timestamp(entry_date)
            fx_after = []
            for _, r in sdf.iterrows():
                ed = pd.to_datetime(r["edt"]).date()
                if ed <= bar_d.date() and ed > entry_dt.date() and "底分型" in str(r.get("fx_b_mark", "")):
                    fx_after.append(float(r.get("fx_b_low", 0)))
            if fx_after:
                candidate = fx_after[-1]
                if candidate > defense: defense = candidate
        if bar_l <= defense:
            exit_date = bar_d; exit_price = defense; exit_reason = '止损'; break
        if bar_c <= entry_price * COST_STOP_C.get(regime, 0.92):
            exit_date = bar_d; exit_price = bar_c; exit_reason = '收盘-8%'; break
        if days > 120:
            exit_date = bar_d; exit_price = bar_c; exit_reason = '超时120天'; break
    if exit_date is None:
        last_bar = future_df.iloc[-1]
        exit_date = last_bar['date']; exit_price = last_bar['close']; exit_reason = '期末平仓'
    days_held = (exit_date - pd.Timestamp(entry_date)).days
    return exit_date, exit_price, exit_reason, days_held

# ====== 主流程 ======
print("加载预计算维度...", flush=True)
df = pd.read_parquet(WORKDIR / "tmp_out/optuna_bullbear_full.parquet")
print(f"  {len(df):,} signals, {df.code.nunique()} stocks", flush=True)

results = {}
for config_name, weights_dict, thresholds_dict in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Running {config_name}...", flush=True)
    t_start = time.time()

    # 计算总分 + 过滤
    df["total"] = 0.0
    for bt in ["一买", "二买", "三买"]:
        for regime in ["BEAR", "BULL"]:
            w = weights_dict.get(bt, {}).get(regime, {})
            mask = (df["buy_type"] == bt) & (df["regime"] == regime)
            for dim, weight in w.items():
                if dim in df.columns:
                    df.loc[mask, "total"] += df.loc[mask, dim].fillna(0) * weight

    # 阈值过滤
    df["passed"] = False
    for bt in ["一买", "二买", "三买"]:
        for regime in ["BEAR", "BULL"]:
            th = thresholds_dict.get(bt, {}).get(regime, 999)
            mask = (df["buy_type"] == bt) & (df["regime"] == regime)
            df.loc[mask, "passed"] = df.loc[mask, "total"] >= th

    passed = df[df["passed"]].sort_values("signal_date").copy()
    print(f"  L2 passed: {len(passed):,} / {len(df):,} ({len(passed)/len(df)*100:.1f}%)", flush=True)

    # ExitEngine 模拟
    trades = []
    n = 0
    for _, row in passed.iterrows():
        code = row["code"]; T0 = row["signal_date"]
        dp = DAILY / f"{code}.parquet"
        if not dp.exists(): continue
        dd = pd.read_parquet(dp)
        dd["date"] = pd.to_datetime(dd["date"]); dd = dd.sort_values("date")
        future = dd[dd["date"] > T0]
        if len(future) < 2: continue
        entry_price = future.iloc[0]["open"]
        if entry_price <= 0: continue

        cache_path = STRUCT_CACHE_DIR / f"{code}.parquet"
        if not cache_path.exists(): continue
        sdf = pd.read_parquet(cache_path)
        if sdf is None or sdf.empty: continue

        try:
            exit_date, exit_price, exit_reason, days = simulate_exit(
                code, entry_price, future.iloc[0]["date"], row["buy_type"], sdf, future.iloc[1:], row["regime"]
            )
        except Exception:
            continue
        ret = (exit_price - entry_price) / entry_price
        trades.append({"code": code, "buy_type": row["buy_type"], "regime": row["regime"],
                       "signal_date": T0, "entry_price": entry_price, "exit_date": exit_date,
                       "exit_price": exit_price, "return": ret, "days": days,
                       "exit_reason": exit_reason, "total_score": row["total"]})
        n += 1
        if n % 10000 == 0:
            elapsed = time.time() - t_start
            print(f"    [{n:,}/{len(passed):,}] {elapsed/60:.0f}min {len(trades):,} trades", flush=True)

    out = pd.DataFrame(trades)
    wr = (out["return"] > 0).sum() / len(out) if len(out) > 0 else 0
    avg_ret = out["return"].mean() if len(out) > 0 else 0
    wins = out[out["return"] > 0]; losses = out[out["return"] <= 0]
    plr = wins["return"].mean() / abs(losses["return"].mean()) if len(wins) > 0 and len(losses) > 0 else 0
    avg_days = out["days"].mean() if len(out) > 0 else 0

    results[config_name] = {"trades": len(out), "win_rate": wr, "avg_return": avg_ret,
                             "profit_loss_ratio": plr, "avg_hold_days": avg_days,
                             "elapsed_min": (time.time() - t_start) / 60}
    print(f"  Done: {len(out):,} trades, WR={wr*100:.1f}%, avg_ret={avg_ret*100:.2f}%, elapsed={results[config_name]['elapsed_min']:.0f}min", flush=True)

# ====== 输出对比 ======
print("\n" + "=" * 70)
print(f"{'指标':<20} {'基线A':>10} {'基线B':>10} {'实验组':>10} {'A→C':>8} {'B→C':>8}")
print("-" * 70)
for key, label in [("trades", "交易笔数"), ("win_rate", "胜率"), ("avg_return", "平均收益"),
                    ("profit_loss_ratio", "盈亏比"), ("avg_hold_days", "平均持仓天")]:
    a = results["baseline_A"][key]
    b = results["baseline_B"][key]
    c = results["experiment"][key]
    if isinstance(a, float):
        d1 = f"{c-a:+.3f}" if key != "profit_loss_ratio" else f"{c-a:+.1f}"
        d2 = f"{c-b:+.3f}" if key != "profit_loss_ratio" else f"{c-b:+.1f}"
        fmt_a = f"{a:.3f}" if key not in ["avg_return", "win_rate"] else f"{a*100:.1f}%"
        fmt_b = f"{b:.3f}" if key not in ["avg_return", "win_rate"] else f"{b*100:.1f}%"
        fmt_c = f"{c:.3f}" if key not in ["avg_return", "win_rate"] else f"{c*100:.1f}%"
        if key == "avg_return": fmt_a = f"{a*100:+.2f}%"; fmt_b = f"{b*100:+.2f}%"; fmt_c = f"{c*100:+.2f}%"
    else:
        d1 = f"{c-a:+d}"; d2 = f"{c-b:+d}"
        fmt_a = f"{a:,d}"; fmt_b = f"{b:,d}"; fmt_c = f"{c:,d}"
    print(f"{label:<20} {fmt_a:>10} {fmt_b:>10} {fmt_c:>10} {d1:>8} {d2:>8}")

# 保存
with open(str(WORKDIR / "tmp_out/backtest_compare_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved: tmp_out/backtest_compare_results.json")
