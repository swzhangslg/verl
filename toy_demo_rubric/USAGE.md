# 🎯 Toy Demo 使用指南

完整的分步教程，帮你验证 Rubric Reward 多维度打印方法。

## 📋 前置要求

- ✅ VERL 已安装
- ✅ 至少1个GPU
- ✅ 已下载模型（推荐 `Qwen/Qwen2.5-0.5B`）

## 🚀 快速开始（3种方法）

### 方法1：测试 Reward 函数（最快，30秒）

这个方法只测试reward函数本身，不需要GPU或模型。

```bash
cd /home/user/verl
python3 toy_demo_rubric/rubric_reward.py
```

**预期输出：**
```
Test 1 - 完整回答:
  Response: Let me solve this step by step...
  Results: {'score': 0.78, 'correctness': 1.0, 'completeness': 1.0, 'practicality': 0.7, 'response_length': 78}

Test 2 - 简短回答:
  Response: The answer is 4.
  Results: {'score': 0.63, 'correctness': 1.0, 'completeness': 0.5, 'practicality': 0.5, 'response_length': 17}

Test 3 - 错误回答:
  Response: I think it's 5.
  Results: {'score': 0.26, 'correctness': 0.0, 'completeness': 0.2, 'practicality': 0.5, 'response_length': 16}
```

✅ **验证要点**：能看到各维度的分数

---

### 方法2：使用现有数据集（推荐，5-10分钟）

如果你已经有 GSM8K 数据集和模型。

```bash
cd /home/user/verl
bash toy_demo_rubric/run_simple_demo.sh
```

这个脚本会：
1. 自动检测GSM8K数据集
2. 加载patch
3. 运行2步训练
4. 打印各维度metrics

**预期输出：**
```
Step 1:
  training/global_step: 1
  critic/score/mean: 0.68
  train/reward/correctness/mean: 0.75       ← 看到这个就成功了！
  train/reward/correctness/max: 1.00
  train/reward/correctness/min: 0.50
  train/reward/completeness/mean: 0.65      ← 全面性维度
  train/reward/practicality/mean: 0.55      ← 实用性维度
  train/reward/response_length/mean: 42.3
  actor/entropy: 2.45
  ...
```

✅ **验证要点**：每一步都能看到 `train/reward/correctness/mean` 等指标

---

### 方法3：完整流程（需要下载数据，15-20分钟）

创建toy数据集并完整运行。

**步骤1：安装依赖（如果需要）**
```bash
pip install pandas  # 或 pyarrow
```

**步骤2：创建数据集**
```bash
cd /home/user/verl
python3 toy_demo_rubric/create_toy_dataset.py
```

**步骤3：运行完整demo**
```bash
bash toy_demo_rubric/run_toy_demo.sh
```

或者使用一键脚本：
```bash
bash toy_demo_rubric/quickstart.sh
```

---

## 📁 文件说明

```
toy_demo_rubric/
├── rubric_reward.py           # ⭐ 核心：自定义reward函数
├── patch_ray_trainer.py       # ⭐ 核心：提取并打印各维度
├── run_simple_demo.sh         # 推荐：使用现有数据的简化运行脚本
├── create_toy_dataset.py      # 可选：创建toy数据集
├── run_toy_demo.sh            # 可选：完整运行脚本
├── quickstart.sh              # 可选：一键运行所有步骤
├── README.md                  # 详细技术文档
└── USAGE.md                   # 本文件：使用指南
```

### ⭐ 核心文件详解

#### 1. `rubric_reward.py` - Reward函数

这是关键！定义了如何评估三个维度：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # 评估三个维度
    correctness = evaluate_correctness(solution_str, ground_truth)
    completeness = evaluate_completeness(solution_str)
    practicality = evaluate_practicality(solution_str)

    # 线性加权
    final_score = (
        0.5 * correctness +
        0.3 * completeness +
        0.2 * practicality
    )

    # 返回字典 - 这是关键！
    return {
        'score': final_score,         # 必须有，用于训练
        'correctness': correctness,   # 维度1
        'completeness': completeness, # 维度2
        'practicality': practicality, # 维度3
    }
```

**关键点：**
- ✅ 必须返回字典
- ✅ 字典必须包含 `'score'` 键
- ✅ 其他键就是你的自定义维度

#### 2. `patch_ray_trainer.py` - 提取维度

这个文件让trainer能打印各维度：

```python
def add_reward_dimension_metrics(batch, metrics):
    """从 batch.non_tensor_batch 提取reward维度"""
    for dim_name in ['correctness', 'completeness', 'practicality']:
        if dim_name in batch.non_tensor_batch:
            values = batch.non_tensor_batch[dim_name]
            metrics[f'train/reward/{dim_name}/mean'] = float(np.mean(values))
```

**工作原理：**
```
你的reward函数返回字典
    ↓
RewardManager存储到 reward_extra_info
    ↓
batch.non_tensor_batch['correctness'] = [0.8, 0.9, ...]
    ↓
patch提取并计算统计
    ↓
metrics['train/reward/correctness/mean'] = 0.85
    ↓
logger打印
```

---

## 🔧 自定义和扩展

### 修改权重

编辑 `rubric_reward.py` 中的权重：

```python
weights = {
    'correctness': 0.6,    # 增加正确性权重
    'completeness': 0.2,   # 减少全面性权重
    'practicality': 0.2
}
```

### 添加新维度

**步骤1：** 在 `rubric_reward.py` 中添加评估函数：

```python
def evaluate_fluency(solution_str):
    """评估流畅性"""
    # 你的评估逻辑
    return score

def compute_score(...):
    fluency = evaluate_fluency(solution_str)

    return {
        'score': final_score,
        'correctness': correctness,
        'completeness': completeness,
        'practicality': practicality,
        'fluency': fluency,  # 新维度
    }
```

**步骤2：** 在 `patch_ray_trainer.py` 中添加到列表：

```python
reward_dimensions = [
    'correctness', 'completeness', 'practicality',
    'fluency'  # 新维度
]
```

### 使用LLM作为评判

```python
from openai import OpenAI

client = OpenAI(api_key="your-key")

def evaluate_with_llm(solution_str, ground_truth):
    """使用LLM评估"""
    prompt = f"""
    评估以下回答的质量（0-1分）：
    正确答案：{ground_truth}
    学生答案：{solution_str}

    返回JSON: {{"correctness": 0.8, "completeness": 0.7, "practicality": 0.6}}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    return json.loads(response.choices[0].message.content)
```

---

## ❓ 常见问题

### Q1: 看不到 rubric 维度？

**检查清单：**
```bash
# 1. reward函数是否正确？
python3 toy_demo_rubric/rubric_reward.py

# 2. patch是否加载？
# 查看日志，应该有：
# "✅ PPOTrainer has been patched to track rubric dimensions"

# 3. 是否使用了自定义reward函数？
# 检查命令行参数：
# custom_reward_function.path=toy_demo_rubric/rubric_reward.py
# custom_reward_function.name=compute_score
```

### Q2: 数值都是0或NaN？

**可能原因：**
- reward函数逻辑有误
- 数据格式不匹配
- ground_truth为空

**调试方法：**
```python
# 在 rubric_reward.py 中添加打印
def compute_score(...):
    print(f"DEBUG: solution_str = {solution_str}")
    print(f"DEBUG: ground_truth = {ground_truth}")
    # ... 评估逻辑
```

### Q3: GPU内存不足？

**调整参数：**
```bash
# 减少batch size
data.train_batch_size=4

# 减少response数量
actor_rollout_ref.rollout.n=1

# 减少GPU内存使用
actor_rollout_ref.rollout.gpu_memory_utilization=0.3

# 开启offload
actor_rollout_ref.actor.fsdp_config.param_offload=True
```

### Q4: 训练很慢？

这是正常的！GRPO需要多次rollout。加速方法：

```bash
# 1. 减少训练步数
trainer.total_training_steps=1

# 2. 使用更小的模型
MODEL_ID="Qwen/Qwen2.5-0.5B"  # 而不是7B

# 3. 减少序列长度
data.max_prompt_length=64
data.max_response_length=64
```

---

## 📊 验证成功的标准

运行demo后，在输出中搜索这些关键字：

```bash
# 在终端中搜索
grep "train/reward/correctness" output.log
grep "train/reward/completeness" output.log
grep "train/reward/practicality" output.log
```

**应该看到类似：**
```
train/reward/correctness/mean: 0.75
train/reward/correctness/max: 1.00
train/reward/correctness/min: 0.50
train/reward/correctness/std: 0.18
train/reward/completeness/mean: 0.65
train/reward/completeness/max: 0.70
train/reward/completeness/min: 0.50
train/reward/practicality/mean: 0.55
```

✅ **如果看到这些，恭喜！方法验证成功！**

---

## 🎓 下一步

### 1. 应用到实际项目

将这两个核心文件应用到你的项目：

```bash
# 复制reward函数（修改成你的评估逻辑）
cp toy_demo_rubric/rubric_reward.py your_project/

# 在训练时使用
python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=your_project/rubric_reward.py \
    custom_reward_function.name=compute_score \
    # ... 其他参数
```

### 2. 持久化修改（可选）

如果你想永久添加这个功能：

**方法A：修改源代码**

在 `/home/user/verl/verl/trainer/ppo/ray_trainer.py` 的 line 1316 后添加：

```python
# Line 1316 后添加
metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))

# 添加这段 ↓
for key in ['correctness', 'completeness', 'practicality']:
    if key in batch.non_tensor_batch:
        values = batch.non_tensor_batch[key]
        metrics[f'train/reward/{key}/mean'] = float(np.mean(values))
        metrics[f'train/reward/{key}/max'] = float(np.max(values))
        metrics[f'train/reward/{key}/min'] = float(np.min(values))
# 添加结束 ↑

metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
```

**方法B：使用patch（推荐）**

在你的训练脚本开头添加：

```python
import sys
sys.path.insert(0, 'path/to/toy_demo_rubric')
import patch_ray_trainer  # 自动patch
```

### 3. 可视化

使用 WandB 可视化各维度的变化：

```bash
# 启用WandB
trainer.logger='["console","wandb"]'
trainer.project_name='my_project'
```

然后在WandB Dashboard中可以看到：
- `train/reward/correctness/mean` 曲线
- `train/reward/completeness/mean` 曲线
- `train/reward/practicality/mean` 曲线

对比不同维度的学习速度！

---

## 💡 最佳实践

1. **先测试reward函数**
   ```bash
   python3 your_rubric_reward.py
   ```

2. **使用少量数据快速迭代**
   ```bash
   trainer.total_training_steps=1
   data.train_batch_size=4
   ```

3. **保存每步的详细数据**
   ```bash
   trainer.rollout_data_dir=./logs/rollout
   ```

4. **定期验证**
   ```bash
   trainer.test_freq=5  # 每5步验证一次
   ```

5. **监控各维度的平衡**
   - 如果某个维度总是1.0或0.0，说明太简单或太难
   - 调整评估函数或权重

---

## 📚 相关资源

- **VERL文档**: https://docs.verl.com
- **示例reward函数**: `verl/utils/reward_score/`
- **配置说明**: `docs/examples/config.rst`
- **完整教程**: `toy_demo_rubric/README.md`

---

## 🆘 获取帮助

如果遇到问题：

1. 查看详细日志：`trainer.logger=console`
2. 添加调试打印：在reward函数中添加 `print()`
3. 检查数据格式：`print(batch.non_tensor_batch.keys())`
4. 简化问题：从最小demo开始，逐步添加复杂度

祝你成功！🎉
