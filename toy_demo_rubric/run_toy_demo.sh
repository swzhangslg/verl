#!/bin/bash
set -x

# ============================================
# Toy Demo: 验证Rubric Reward多维度打印
# ============================================

# 配置
MODEL_ID="Qwen/Qwen2.5-0.5B"
MODEL_PATH="${HOME}/models/${MODEL_ID}"
TRAIN_FILES="toy_demo_rubric/data/train.parquet"
VAL_FILES="toy_demo_rubric/data/test.parquet"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  模型不存在，正在下载..."
    huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
fi

# 训练参数（最小配置用于快速测试）
NUM_GPUS=1
TRAIN_BATCH_SIZE=8
N_RESPONSES=2
TOTAL_STEPS=3  # 只运行3步用于测试

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练（源代码已包含rubric维度提取逻辑）
python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=${N_RESPONSES} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=toy_demo_rubric/rubric_reward.py \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='toy_demo_rubric' \
    trainer.experiment_name='test_rubric_reward' \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.val_before_train=False

echo ""
echo "✅ Toy demo 完成！"
echo "请检查上面的输出，应该能看到类似这样的metrics:"
echo "  train/reward/correctness/mean: 0.65"
echo "  train/reward/completeness/mean: 0.58"
echo "  train/reward/practicality/mean: 0.42"
