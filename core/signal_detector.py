"""全局信号变化检测 — 检测历史中所有买点/卖点信号变化。"""
from core.constants import BUY1_COL, BS2_COL, BS3_COL, SANMAI_COL, SANMAI2_COL, BUY_COLS, SELL_COLS
from core.signal_parser import parse_signal


def detect_all_changes(sig_df) -> list[dict]:
    """检测全部历史中的买点/卖点信号变化。"""
    buy_cols = [c for c in [BUY1_COL, BS2_COL, BS3_COL, SANMAI_COL, SANMAI2_COL] if c in sig_df.columns]
    sell_cols = [c for c in SELL_COLS if c in sig_df.columns]
    changes = []

    for col in buy_cols:
        for i in range(1, len(sig_df)):
            old_r = str(sig_df.iloc[i - 1][col])
            new_r = str(sig_df.iloc[i][col])
            if old_r == new_r:
                continue
            new_p = parse_signal(new_r)
            if any(k in new_p["v1"] for k in ["一买", "二买", "三买"]):
                dt = sig_df.iloc[i]["dt"]
                sig_label = f"{new_p['v1']}({col.split('_')[2][:8] if '_' in col else col[:8]})"
                changes.append({
                    "idx": i, "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                    "type": "buy", "signal_label": sig_label,
                })

    for col in sell_cols:
        for i in range(1, len(sig_df)):
            old_r = str(sig_df.iloc[i - 1][col])
            new_r = str(sig_df.iloc[i][col])
            if old_r == new_r:
                continue
            new_p = parse_signal(new_r)
            if any(k in new_p["v1"] for k in ["一卖", "二卖", "三卖"]):
                dt = sig_df.iloc[i]["dt"]
                sig_label = new_p["v1"]
                changes.append({
                    "idx": i, "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                    "type": "sell", "signal_label": sig_label,
                })

    changes.sort(key=lambda x: x["date"])
    return changes
