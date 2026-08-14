"""日期工具：全量更新/回测共用的有效信号日期逻辑。

规则：16:00 前 → 返回昨天（当日数据未就绪），16:00 后 → 返回今天。
"""
from datetime import datetime, timedelta
import pandas as pd


def get_effective_date() -> pd.Timestamp:
    """返回新鲜度检查用的有效参考日期。

    16:00 前：当日 K 线可能还没更新完毕，以昨天为基准
    16:00 后：当日数据已就绪，以今天为基准
    """
    now = datetime.now()
    if now.hour < 16:
        now = now - timedelta(days=1)
    return pd.Timestamp(now.date())
