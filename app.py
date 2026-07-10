#!/usr/bin/env python3
"""缠论选股仪表盘 — 四级漏斗 + 监控塔 (czsc 原生版)。"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))

from signal_engine import load_signals

st.set_page_config(page_title="缠论选股 · 四级漏斗", page_icon="📈", layout="wide")

BASE_DIR = Path(__file__).parent
ZONES_DIR = BASE_DIR / "data" / "zones"

# ——— 缓存数据读取 ———
@st.cache_data(ttl=30)
def read_zone(name: str) -> pd.DataFrame:
    p = ZONES_DIR / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


# ——— 侧边栏 ———
with st.sidebar:
    st.title("导航")
    page = st.radio("", [
        "📊 漏斗概览",
        "L1 全量沉淀区",
        "L2 重点形态沉淀区",
        "L3 自选股生态区",
        "L4 购买建议区",
        "📡 监控塔",
    ], label_visibility="collapsed")

    st.divider()
    signals_count = len(list((BASE_DIR / "data" / "signals").glob("*.parquet")))
    daily_count = len(list((BASE_DIR / "data" / "daily").glob("*.parquet")))
    st.metric("日线", f"{daily_count}")
    st.metric("信号", f"{signals_count}")
    st.caption("czsc · generate_czsc_signals\nSina akshare 数据源")

# ——— 快捷键 ———
if st.sidebar.button("🔄 重新运行漏斗", use_container_width=True):
    from run_zones import main as run_funnel
    run_funnel()
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("🔄 刷新监控", use_container_width=True):
    from monitor import run as run_monitor
    run_monitor()
    st.cache_data.clear()
    st.rerun()

# ================================================================
# 漏斗概览
# ================================================================
if page == "📊 漏斗概览":
    st.title("四级选股漏斗")
    st.caption("全市场 → L1 全量沉淀 → L2 策略一区 → L3 策略二区 → L4 购买建议")

    df1 = read_zone("L1_deposition")
    df2 = read_zone("L2_pattern")
    df3 = read_zone("L3_watchlist")
    df4 = read_zone("L4_recommend")
    sell = read_zone("monitor_sell")
    watch = read_zone("monitor_watchlist")

    counts = {
        "全市场": daily_count,
        "L1 全量沉淀区": len(df1),
        "L2 策略一区": len(df2),
        "L3 策略二区": len(df3),
        "L4 购买建议区": len(df4),
    }

    # 柱状图
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure(data=[
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                text=list(counts.values()),
                textposition="outside",
                marker_color=["#94a3b8", "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd"],
            )
        ])
        fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            title="选股漏斗 (数量递减)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.metric("L1 全量沉淀区", len(df1), "含买点股票")
        st.metric("L2 策略一区", len(df2), f"{'框架全通过' if len(df1)==len(df2) else '已过滤'}")
        st.metric("L3 策略二区", len(df3))
        st.metric("L4 购买建议区", len(df4), "最高评分")

    st.divider()

    # 监控状态
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📡 卖点监控")
        st.metric("卖点预警", len(sell))
        if not sell.empty:
            st.dataframe(sell[["代码","现价","卖点类型","MACD死叉"]],
                         use_container_width=True, hide_index=True)
    with col_b:
        st.subheader("🔍 细颗粒度监控")
        st.metric("细监清单", len(watch))
        if not watch.empty:
            st.dataframe(watch[["代码","现价","监控级别","状态"]],
                         use_container_width=True, hide_index=True)

# ================================================================
# L1 全量沉淀区
# ================================================================
elif page == "L1 全量沉淀区":
    st.title("L1 全量沉淀区")
    st.caption("含一买/二买/三买信号的所有股票")
    df = read_zone("L1_deposition")
    if df.empty:
        st.info("暂无数据。点击侧边栏「重新运行漏斗」。")
    else:
        st.metric("候选数", len(df))
        st.dataframe(df, use_container_width=True, hide_index=True)

# ================================================================
# L2 重点形态沉淀区
# ================================================================
elif page == "L2 重点形态沉淀区":
    st.title("L2 重点形态沉淀区")
    st.caption("策略一区过滤结果（当前框架模式：全通过）")
    df = read_zone("L2_pattern")
    if df.empty:
        st.info("暂无数据。")
    else:
        st.metric("候选数", len(df))
        st.dataframe(df, use_container_width=True, hide_index=True)

# ================================================================
# L3 自选股生态区
# ================================================================
elif page == "L3 自选股生态区":
    st.title("L3 自选股生态区")
    st.caption("策略二区过滤结果（当前框架模式：全通过）")
    df = read_zone("L3_watchlist")
    if df.empty:
        st.info("暂无数据。")
    else:
        st.metric("候选数", len(df))
        st.dataframe(df, use_container_width=True, hide_index=True)

# ================================================================
# L4 购买建议区
# ================================================================
elif page == "L4 购买建议区":
    st.title("L4 购买建议区")
    st.caption("按信号质量评分排名 — 重点关注股")

    df = read_zone("L4_recommend")
    if df.empty:
        st.info("暂无数据。")
    else:
        st.metric("推荐数", len(df))
        st.dataframe(df, use_container_width=True, hide_index=True)

        # TOP10 高亮
        st.divider()
        st.subheader("TOP10")
        top10 = df.head(10)
        for _, row in top10.iterrows():
            buy = row.get("买点类型", "")
            macd = "✅" if row.get("MACD确认") else "—"
            ma = "✅" if row.get("MA确认") else "—"
            st.markdown(
                f"**{row['代码']}**  ¥{row['现价']}  |  "
                f"{buy}  |  MACD {macd}  MA {ma}  |  "
                f"评分: {int(row['评分'])}"
            )

# ================================================================
# 监控塔
# ================================================================
elif page == "📡 监控塔":
    st.title("监控塔")
    st.caption("区域二：卖点预警 + L3/L4 细颗粒度监控")

    sell = read_zone("monitor_sell")
    watch = read_zone("monitor_watchlist")

    tab1, tab2 = st.tabs(["卖点预警", "细颗粒度监控"])

    with tab1:
        st.subheader("非 L1 股票卖点预警")
        if sell.empty:
            st.info("暂无非L1卖点预警")
        else:
            st.metric("预警数", len(sell))
            st.dataframe(sell, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("L3/L4 细颗粒度监控清单")
        st.caption("后续开发：多级别联立 · 实时推送 · 持仓状态追踪")
        if watch.empty:
            st.info("暂无细监清单")
        else:
            st.metric("监控数", len(watch))
            st.dataframe(watch, use_container_width=True, hide_index=True)
