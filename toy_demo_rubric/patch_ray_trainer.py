"""
Patch ray_trainer.py 以在每步训练中打印rubric各维度
"""
import numpy as np


def add_reward_dimension_metrics(batch, metrics):
    """
    从 batch.non_tensor_batch 中提取reward维度并添加到metrics

    Args:
        batch: DataProto对象
        metrics: 要更新的metrics字典
    """
    # 定义我们关心的reward维度
    reward_dimensions = ['correctness', 'completeness', 'practicality', 'response_length']

    for dim_name in reward_dimensions:
        if dim_name in batch.non_tensor_batch:
            values = batch.non_tensor_batch[dim_name]

            # 确保是数值类型
            if isinstance(values[0], (int, float, bool, np.number)):
                # 计算统计信息
                metrics[f'train/reward/{dim_name}/mean'] = float(np.mean(values))
                metrics[f'train/reward/{dim_name}/max'] = float(np.max(values))
                metrics[f'train/reward/{dim_name}/min'] = float(np.min(values))
                metrics[f'train/reward/{dim_name}/std'] = float(np.std(values))

    return metrics


def patch_trainer():
    """
    动态patch PPOTrainer以添加reward维度打印
    """
    from verl.trainer.ppo.ray_trainer import PPOTrainer

    # 保存原始的fit方法
    original_fit = PPOTrainer.fit

    def patched_fit(self):
        """Patched fit method with reward dimension tracking"""
        import verl.trainer.ppo.metric_utils as metric_utils

        # 保存原始的compute_data_metrics
        original_compute_data_metrics = metric_utils.compute_data_metrics

        def patched_compute_data_metrics(batch, use_critic=True):
            """Enhanced compute_data_metrics that includes reward dimensions"""
            # 调用原始函数
            metrics = original_compute_data_metrics(batch, use_critic)

            # 添加reward维度metrics
            metrics = add_reward_dimension_metrics(batch, metrics)

            return metrics

        # 替换函数
        metric_utils.compute_data_metrics = patched_compute_data_metrics

        try:
            # 调用原始fit
            print("\n" + "="*60)
            print("🎯 Rubric Reward Tracking ENABLED")
            print("   将打印以下维度: correctness, completeness, practicality")
            print("="*60 + "\n")

            return original_fit(self)
        finally:
            # 恢复原始函数
            metric_utils.compute_data_metrics = original_compute_data_metrics

    # 替换fit方法
    PPOTrainer.fit = patched_fit
    print("✅ PPOTrainer has been patched to track rubric dimensions")


# 自动执行patch
patch_trainer()
