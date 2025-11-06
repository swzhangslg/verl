#!/bin/bash
# ============================================
# 快速启动脚本 - 一键运行所有步骤
# ============================================

set -e  # 遇到错误立即退出

echo "🚀 开始 Rubric Reward Toy Demo"
echo "================================"
echo ""

# 步骤1：创建数据集
echo "📊 步骤 1/4: 创建toy数据集..."
python3 toy_demo_rubric/create_toy_dataset.py
echo ""

# 步骤2：测试reward函数
echo "🧪 步骤 2/4: 测试reward函数..."
python3 toy_demo_rubric/rubric_reward.py
echo ""

# 步骤3：检查模型
echo "🔍 步骤 3/4: 检查模型..."
MODEL_ID="Qwen/Qwen2.5-0.5B"
MODEL_PATH="${HOME}/models/${MODEL_ID}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  模型不存在，需要下载（约1GB，可能需要几分钟）"
    read -p "是否现在下载？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
    else
        echo "❌ 需要模型才能继续，退出..."
        exit 1
    fi
else
    echo "✅ 模型已存在: $MODEL_PATH"
fi
echo ""

# 步骤4：运行训练
echo "🎯 步骤 4/4: 运行toy训练（只运行3步，约2-3分钟）..."
echo "请注意观察输出中的 'train/reward/correctness/mean' 等指标"
echo ""
bash toy_demo_rubric/run_toy_demo.sh

echo ""
echo "================================"
echo "✅ Demo 完成！"
echo ""
echo "📌 要点总结："
echo "  1. 每一步训练都会打印rubric各维度"
echo "  2. 查找 'train/reward/correctness/mean' 等指标"
echo "  3. 这些指标会显示模型在各维度上的表现"
echo ""
echo "💡 下一步："
echo "  - 修改 rubric_reward.py 来自定义评分逻辑"
echo "  - 调整权重以改变各维度的重要性"
echo "  - 将这个方法应用到你的实际项目中"
