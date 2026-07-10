#!/bin/bash
# 缠论选股仪表盘 — 一键启动脚本
cd "$(dirname "$0")"

# 检查数据
DAILY_COUNT=$(ls data/daily/*.parquet 2>/dev/null | wc -l | tr -d ' ')
SIGNAL_COUNT=$(ls data/signals/*.json 2>/dev/null | wc -l | tr -d ' ')

echo "=== 缠论选股仪表盘 ==="
echo "日线缓存: ${DAILY_COUNT} 只"
echo "信号缓存: ${SIGNAL_COUNT} 只"
echo ""

# 如果没有数据，先拉取
if [ "$DAILY_COUNT" -eq 0 ]; then
    echo "⚠️  无数据缓存，开始拉取全市场日线..."
    NO_PROXY="*" python3 data_fetcher.py
fi

# 如果没有信号，先计算
if [ "$SIGNAL_COUNT" -eq 0 ]; then
    echo "⚠️  无信号缓存，开始计算缠论信号..."
    NO_PROXY="*" python3 signal_engine.py
fi

echo ""
echo "🚀 启动 Streamlit..."
echo "   打开浏览器访问: http://localhost:8502"
echo "   按 Ctrl+C 停止"
echo ""

NO_PROXY="*" streamlit run app.py --server.port=8502 --server.headless=true
