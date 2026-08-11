"""统计分析 + 人工 vs 自动对比."""
import pandas as pd
import numpy as np


class ManualAnalyzer:
    """回测结果统计分析器.

    stats 输出契约:
      summary: {total_trades, win_rate, avg_return, win_loss_ratio, avg_hold_days, max_win, max_loss, total_return}
      所有比率和收益字段均为小数 (0.05 = 5%), 供 print_summary 的 .1%/.2% 格式化使用.
      by_buy_type/by_regime/by_exit_reason: {key: {count, win_rate, avg_return, avg_hold_days}}
    """

    def __init__(self, trades_df: pd.DataFrame, auto_trades_df: pd.DataFrame | None = None):
        self.trades = trades_df
        self.auto_trades = auto_trades_df

    @staticmethod
    def _compute_summary(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"total_trades": 0, "win_rate": 0, "avg_return": 0,
                    "win_loss_ratio": 0, "avg_hold_days": 0,
                    "max_win": 0, "max_loss": 0, "total_return": 0}
        returns = df["return_pct"].values / 100.0
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        total = len(df)
        win_rate = len(wins) / total if total > 0 else 0
        avg_return = returns.mean()
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        avg_hold = df["hold_days"].mean() if "hold_days" in df.columns else 0
        max_win = returns.max() if total > 0 else 0
        max_loss = returns.min() if total > 0 else 0
        total_return = np.prod(1 + returns) - 1 if total > 0 else 0
        return {
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "avg_return": round(avg_return, 4),
            "win_loss_ratio": round(win_loss_ratio, 2),
            "avg_hold_days": round(avg_hold, 0),
            "max_win": round(max_win, 4),
            "max_loss": round(max_loss, 4),
            "total_return": round(total_return, 4),
        }

    @staticmethod
    def _group_stats(df: pd.DataFrame, col: str) -> dict:
        if df.empty or col not in df.columns:
            return {}
        result = {}
        for name, grp in df.groupby(col):
            s = ManualAnalyzer._compute_summary(grp)
            result[str(name)] = {
                "count": s["total_trades"],
                "win_rate": s["win_rate"],
                "avg_return": s["avg_return"],
                "avg_hold_days": s["avg_hold_days"],
            }
        return result

    def analyze(self) -> dict:
        """生成分层统计."""
        return {
            "summary": self._compute_summary(self.trades),
            "by_buy_type": self._group_stats(self.trades, "buy_type"),
            "by_regime": self._group_stats(self.trades, "regime"),
            "by_exit_reason": self._group_stats(self.trades, "exit_reason"),
            "hold_distribution": self._hold_distribution(),
        }

    def _hold_distribution(self) -> dict:
        if self.trades.empty or "hold_days" not in self.trades.columns:
            return {}
        days = self.trades["hold_days"]
        buckets = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 99999)]
        dist = {}
        for lo, hi in buckets:
            cnt = ((days >= lo) & (days < hi)).sum()
            if cnt > 0:
                dist[f"{lo}-{hi}天"] = cnt
        return dist

    def compare(self, top_n: int = 50) -> pd.DataFrame:
        """人工 vs 自动 top-N 对比."""
        if self.auto_trades is None or self.auto_trades.empty:
            raise ValueError("auto_trades_df 未提供，请先运行 backtest_auto_top_n")
        manual_s = self._compute_summary(self.trades)
        auto_s = self._compute_summary(self.auto_trades)
        rows = []
        metrics = [
            ("总交易笔数", "total_trades", "d"),
            ("胜率", "win_rate", ".1%"),
            ("平均盈亏", "avg_return", ".2%"),
            ("盈亏比", "win_loss_ratio", ".2f"),
            ("平均持仓天数", "avg_hold_days", ".0f"),
            ("最大单笔盈利", "max_win", ".2%"),
            ("最大单笔亏损", "max_loss", ".2%"),
            ("总复合收益", "total_return", ".2%"),
        ]
        for label, key, fmt in metrics:
            mv = manual_s.get(key, 0)
            av = auto_s.get(key, 0)
            if fmt == "d":
                rows.append({"指标": label, "人工选取": int(mv), f"自动 Top-{top_n}": int(av)})
            elif fmt == ".1%":
                rows.append({"指标": label, "人工选取": f"{mv:.1%}", f"自动 Top-{top_n}": f"{av:.1%}"})
            elif fmt == ".2%":
                rows.append({"指标": label, "人工选取": f"{mv:.2%}", f"自动 Top-{top_n}": f"{av:.2%}"})
            else:
                rows.append({"指标": label, "人工选取": f"{mv:{fmt}}", f"自动 Top-{top_n}": f"{av:{fmt}}"})
        return pd.DataFrame(rows)
