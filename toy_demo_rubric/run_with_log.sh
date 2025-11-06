#!/bin/bash
set -x

echo "🚀 运行 Toy Demo 并保存日志"
LOG_FILE="toy_demo_rubric/training_$(date +%Y%m%d_%H%M%S).log"

echo "📝 日志将保存到: $LOG_FILE"
echo ""

# 运行训练并保存日志
bash toy_demo_rubric/run_simple_demo.sh 2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ 训练完成！日志已保存到: $LOG_FILE"
echo ""
echo "🔍 查看关键信息："
echo "   grep -A 2 '\[prompt\]' $LOG_FILE"
echo "   grep -A 2 '\[response\]' $LOG_FILE"
echo "   grep 'reward/correctness' $LOG_FILE"
