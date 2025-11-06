# Toy Demo: Rubric Reward 多维度打印验证

这个 toy demo 用于验证在 GRPO 训练中打印多维度 rubric reward 的方法。

## 📁 文件结构

```
toy_demo_rubric/
├── create_toy_dataset.py      # 创建简单的数学问题数据集
├── rubric_reward.py           # 自定义的多维度reward函数
├── patch_ray_trainer.py       # Patch trainer以打印各维度
├── run_toy_demo.sh            # 运行脚本
├── README.md                  # 本文件
└── data/                      # 生成的数据集（自动创建）
    ├── train.parquet
    └── test.parquet
```

## 🚀 快速开始

### 步骤1：创建数据集

```bash
cd /home/user/verl
python3 toy_demo_rubric/create_toy_dataset.py
```

### 步骤2：测试reward函数

```bash
python3 toy_demo_rubric/rubric_reward.py
```

你应该看到三个测试案例的输出，展示不同回答的各维度分数。

### 步骤3：运行toy demo

```bash
chmod +x toy_demo_rubric/run_toy_demo.sh
bash toy_demo_rubric/run_toy_demo.sh
```

## 📊 预期输出

训练过程中，每一步都应该打印类似这样的metrics：

```
Step 1:
  training/global_step: 1
  critic/score/mean: 0.68
  critic/rewards/mean: 0.68
  train/reward/correctness/mean: 0.75      ← 正确性维度
  train/reward/correctness/max: 1.00
  train/reward/correctness/min: 0.50
  train/reward/correctness/std: 0.18
  train/reward/completeness/mean: 0.65     ← 全面性维度
  train/reward/completeness/max: 0.70
  train/reward/completeness/min: 0.50
  train/reward/practicality/mean: 0.55     ← 实用性维度
  train/reward/practicality/max: 0.70
  train/reward/practicality/min: 0.30
  train/reward/response_length/mean: 42.3  ← 额外信息
  actor/entropy: 2.45
  ...

Step 2:
  train/reward/correctness/mean: 0.78      ← 可以看到进步
  train/reward/completeness/mean: 0.68
  train/reward/practicality/mean: 0.58
  ...
```

## 🔍 验证要点

1. ✅ **各维度都被打印出来**：
   - `train/reward/correctness/mean`
   - `train/reward/completeness/mean`
   - `train/reward/practicality/mean`

2. ✅ **每一步训练都有输出**：
   - 不需要等到validation
   - 每个step都能实时看到

3. ✅ **包含统计信息**：
   - mean (平均值)
   - max (最大值)
   - min (最小值)
   - std (标准差)

## 🛠️ 技术细节

### Reward函数的返回格式

```python
def compute_score(...):
    return {
        'score': 0.68,              # 必须有，用于训练
        'correctness': 0.75,        # 自定义维度1
        'completeness': 0.65,       # 自定义维度2
        'practicality': 0.55,       # 自定义维度3
        'response_length': 42,      # 额外信息
    }
```

### 数据流转

```
compute_score()
    ↓ 返回字典
NaiveRewardManager
    ↓ reward_extra_info['correctness'].append(0.75)
compute_reward()
    ↓ reward_extra_infos_dict
batch.non_tensor_batch.update(...)
    ↓ batch.non_tensor_batch['correctness'] = np.array([...])
patched_compute_data_metrics()
    ↓ 提取并计算统计
metrics['train/reward/correctness/mean'] = np.mean(...)
    ↓
logger.log()
    ↓ 打印到终端/WandB
```

## 🎯 关键代码

### 1. Reward函数返回字典

```python
# rubric_reward.py
def compute_score(...):
    return {
        'score': final_score,        # 总分
        'correctness': correctness,  # 维度1
        'completeness': completeness,# 维度2
        'practicality': practicality,# 维度3
    }
```

### 2. Patch提取维度信息

```python
# patch_ray_trainer.py
def add_reward_dimension_metrics(batch, metrics):
    for dim_name in ['correctness', 'completeness', 'practicality']:
        if dim_name in batch.non_tensor_batch:
            values = batch.non_tensor_batch[dim_name]
            metrics[f'train/reward/{dim_name}/mean'] = float(np.mean(values))
```

## 🔧 调整和扩展

### 添加更多维度

在 `rubric_reward.py` 的返回字典中添加：

```python
return {
    'score': final_score,
    'correctness': correctness,
    'completeness': completeness,
    'practicality': practicality,
    'fluency': fluency,           # 新维度
    'creativity': creativity,     # 新维度
}
```

然后在 `patch_ray_trainer.py` 中添加到列表：

```python
reward_dimensions = [
    'correctness', 'completeness', 'practicality',
    'fluency', 'creativity'  # 新维度
]
```

### 更改权重

在 `rubric_reward.py` 中修改：

```python
weights = {
    'correctness': 0.6,    # 增加权重
    'completeness': 0.2,   # 减少权重
    'practicality': 0.2
}
```

## ❓ 故障排除

### 问题1：看不到rubric维度

检查：
- ✅ reward函数是否返回字典
- ✅ 字典中是否包含 `'score'` 键
- ✅ patch是否正确导入

### 问题2：数值都是0

检查：
- ✅ reward函数的逻辑是否正确
- ✅ 数据格式是否匹配

### 问题3：训练很慢

这是正常的，可以：
- 减少 `TOTAL_STEPS`
- 减少 `TRAIN_BATCH_SIZE`
- 减少 `N_RESPONSES`

## 📚 下一步

验证成功后，你可以：

1. **应用到实际项目**：
   - 修改你的reward函数返回多维度字典
   - 将patch代码集成到你的训练脚本

2. **持久化修改**：
   - 直接修改 `verl/trainer/ppo/ray_trainer.py`
   - 在 line 1316 后添加维度提取代码

3. **可视化**：
   - 使用 WandB 查看各维度的变化趋势
   - 对比不同维度的学习曲线

## 📝 许可

本demo基于 VERL 项目，遵循 Apache 2.0 许可证。
