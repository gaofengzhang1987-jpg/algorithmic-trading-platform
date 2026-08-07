"""回测绩效指标计算 — 唯一版本的 compute_metrics。"""
import numpy as np


def compute_metrics(trades: list[dict], weighted: bool = False) -> dict:
    """计算回测绩效指标。

    Args:
        trades: 回测交易记录列表
        weighted: True 时用 "weight" 字段做加权（配合 PortfolioOptimizer）
    """
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "avg_return": 0,
                "total_return": 0, "max_return": 0, "min_return": 0,
                "avg_hold_days": 0, "sharpe": 0}

    returns = [t["return_pct"] / 100 for t in trades]
    hold_days = [t["hold_days"] for t in trades]
    wins = sum(1 for r in returns if r > 0)

    total_trades = len(trades)
    win_rate = wins / total_trades if total_trades > 0 else 0
    max_r = max(returns) if returns else 0
    min_r = min(returns) if returns else 0

    # ── 加权计算（对接 PortfolioOptimizer） ──
    if weighted and any("weight" in t for t in trades):
        weights = np.array([t.get("weight", 1.0 / total_trades) for t in trades])
        weights = weights / weights.sum()  # 归一化
        returns_arr = np.array(returns)

        avg_return = float(np.average(returns_arr, weights=weights))
        total_return = float(np.sum(returns_arr * weights))

        hold_arr = np.array(hold_days, dtype=float)
        avg_hold = float(np.average(hold_arr, weights=weights))

        if len(returns) > 1 and avg_hold > 0:
            weighted_mean = avg_return
            weighted_var = np.average((returns_arr - weighted_mean) ** 2, weights=weights)
            daily_std = np.sqrt(weighted_var)
            sharpe = (avg_return / (daily_std + 1e-8)) * np.sqrt(252 / avg_hold)
        else:
            sharpe = 0
    else:
        avg_return = np.mean(returns) if returns else 0
        total_return = sum(returns)
        avg_hold = np.mean(hold_days) if hold_days else 0
        if len(returns) > 1 and avg_hold > 0:
            daily_std = np.std(returns)
            sharpe = (avg_return / (daily_std + 1e-8)) * np.sqrt(252 / avg_hold)
        else:
            sharpe = 0

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate * 100, 1),
        "avg_return": round(avg_return * 100, 2),
        "total_return": round(total_return * 100, 2),
        "max_return": round(max_r * 100, 2),
        "min_return": round(min_r * 100, 2),
        "avg_hold_days": round(avg_hold, 1),
        "sharpe": round(sharpe, 2),
    }
