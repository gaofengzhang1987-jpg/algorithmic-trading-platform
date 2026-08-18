#!/usr/bin/env python3
"""L4 选股引擎：从截面 L4 报告中输出 2-3 只推荐股票。

规则配置见 selection_rules.json；输出 tmp_out/selection/selection_{date}.csv。
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
TMP = BASE / "tmp_out"
RULES_PATH = BASE / "selection_rules.json"


def load_rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def select(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    k = int(rules.get("k", 3))
    pool = df.copy()
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    if "composite" not in pool.columns:
        pool["composite"] = 0.0
    for c in ["stock_rps", "sector_rps"]:
        if c not in pool.columns:
            pool[c] = 0.0
    if "qlib_score" not in pool.columns:
        pool["qlib_score"] = 0.5
    if "l4_rank" not in pool.columns and "global_rank" in pool.columns:
        pool["l4_rank"] = pool["global_rank"]

    exclude_top = int(rules.get("exclude_rank_top", 0))
    if exclude_top > 0 and "l4_rank" in pool.columns:
        pool = pool[pool["l4_rank"] > exclude_top]

    # SOP 重构分数：排名靠后加分，RPS/QLIB 减分
    for c in ["l4_rank", "stock_rps", "qlib_score"]:
        pool[c] = pd.to_numeric(pool[c], errors="coerce").fillna(pool[c].median())
        pool[c + "_pct"] = pool.groupby("signal_date")[c].rank(pct=True)
    pool["score"] = (
        pool.get("l4_rank_pct", 0.5)
        - pool.get("stock_rps_pct", 0.5)
        - pool.get("qlib_score_pct", 0.5)
    )

    prefers = rules.get("prefer_buy_types", [])
    if prefers:
        pool["_prefer"] = pool["buy_type"].astype(str).apply(
            lambda x: 0 if any(p in x for p in prefers) else 1
        )
    else:
        pool["_prefer"] = 0

    if rules.get("prefer_zone_21_50"):
        pool["_zone_prefer"] = pool["l4_rank"].apply(lambda r: 0 if 21 <= r <= 50 else 1)
    else:
        pool["_zone_prefer"] = 0

    pool = pool.sort_values(
        ["_prefer", "_zone_prefer", "score"], ascending=[True, True, False]
    ).head(k).copy()

    out = pd.DataFrame({
        "date": pool.get("signal_date", pd.Series([""] * len(pool))),
        "l4_rank": pool.get("global_rank", pd.Series(range(1, len(pool) + 1))),
        "rank": pool.get("global_rank", pd.Series(range(1, len(pool) + 1))),
        "code": pool["code"],
        "name": pool.get("name", ""),
        "buy_type": pool.get("buy_type", ""),
        "composite": pool.get("composite", 0.0),
        "stock_rps": pool.get("stock_rps", 0.0),
        "sector_rps": pool.get("sector_rps", 0.0),
        "qlib_score": pool.get("qlib_score", 0.5),
        "regime": pool.get("regime", ""),
        "score": pool.get("score", 0.0).round(4),
        "rule": "SOP_v0.1",
        "reason": [
            f"rank{r} {'中段' if 21 <= r <= 50 else '中后段'}; {b}; score {s:.3f}"
            for r, b, s in zip(
                pool.get("l4_rank", range(1, len(pool) + 1)),
                pool.get("buy_type", ""),
                pool.get("score", 0.0),
            )
        ],
        "weight": rules.get("weight", "equal"),
    })
    out.insert(1, "sop_recommend", range(1, len(out) + 1))
    out["date"] = out["date"].astype(str)
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="截面日期 YYYY-MM-DD")
    ap.add_argument("--input", help="自定义 L4 CSV 路径（默认 tmp_out/backtest_{date}/l4_{date}.csv）")
    args = ap.parse_args()

    date = args.date
    path = Path(args.input) if args.input else TMP / f"backtest_{date}" / f"l4_{date}.csv"
    if not path.exists():
        print(f"L4 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        print(f"{date}: L4 候选为空", file=sys.stderr)
        sys.exit(1)

    rules = load_rules()
    out = select(df, rules)
    out_dir = TMP / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"selection_{date}.csv"
    json_path = out_dir / f"selection_{date}.json"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(out.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"选股输出: {csv_path} ({len(out)} 只)")
    print(out[["rank", "code", "name", "buy_type", "composite", "regime"]].to_string(index=False))


if __name__ == "__main__":
    main()
