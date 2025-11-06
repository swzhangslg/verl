#!/bin/bash
set -x

# ============================================
# 简化版 Toy Demo - 使用现有的GSM8K数据集
# ============================================

echo "🚀 运行简化版 Rubric Reward Demo"
echo "使用现有的GSM8K数据集（如果有的话）"
echo ""

# 配置
MODEL_ID="Qwen/Qwen2.5-0.5B"
MODEL_PATH="${HOME}/models/${MODEL_ID}"

# 检查GSM8K数据集
if [ -f "$HOME/data/gsm8k/train.parquet" ]; then
    echo "✅ 找到GSM8K数据集"
    TRAIN_FILES="$HOME/data/gsm8k/train.parquet"
    VAL_FILES="$HOME/data/gsm8k/test.parquet"
else
    echo "⚠️ 未找到GSM8K数据集，使用toy数据集"
    # 如果没有GSM8K，使用toy数据
    TRAIN_FILES="toy_demo_rubric/data/train.parquet"
    VAL_FILES="toy_demo_rubric/data/test.parquet"

    if [ ! -f "$TRAIN_FILES" ]; then
        echo "❌ 也没有toy数据集，请先运行: python3 toy_demo_rubric/create_toy_dataset.py"
        exit 1
    fi
fi

# 检查模型
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️ 模型不存在: $MODEL_PATH"
    echo "请手动下载或修改 MODEL_PATH"
    exit 1
fi

# 训练参数（最小配置）
NUM_GPUS=1
TRAIN_BATCH_SIZE=8
N_RESPONSES=2
TOTAL_STEPS=2  # 只运行2步

echo ""
echo "📊 配置:"
echo "  模型: $MODEL_PATH"
echo "  训练集: $TRAIN_FILES"
echo "  总步数: $TOTAL_STEPS"
echo ""

# 创建临时的入口脚本，自动导入patch
cat > /tmp/run_with_patch.py << 'EOF'
import sys
import os

# 添加toy_demo_rubric到path
sys.path.insert(0, os.path.join(os.getcwd(), 'toy_demo_rubric'))

# 导入patch（这会自动patch trainer）
try:
    import patch_ray_trainer
    print("✅ Rubric tracking patch loaded successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not load patch: {e}")
    print("   Training will continue without rubric dimension tracking")

# 运行main_ppo
from verl.trainer import main_ppo
main_ppo.main()
EOF

# 运行训练
python3 /tmp/run_with_patch.py \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=${N_RESPONSES} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=toy_demo_rubric/rubric_reward.py \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='toy_demo_rubric' \
    trainer.experiment_name='test_rubric' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.val_before_train=False

echo ""
echo "✅ Demo 完成！"
echo ""
echo "🔍 验证要点："
echo "  在上面的输出中搜索这些关键字："
echo "    - train/reward/correctness/mean"
echo "    - train/reward/completeness/mean"
echo "    - train/reward/practicality/mean"
echo ""
echo "  如果看到这些指标，说明方法验证成功！"
