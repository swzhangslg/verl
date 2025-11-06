# 🎯 开始验证 Rubric Reward 方法

## ⚡ 3分钟快速验证（最推荐）

**只需要3步，无需GPU或模型！**

```bash
cd /home/user/verl

# 步骤1: 测试reward函数
python3 toy_demo_rubric/rubric_reward.py

# 步骤2: 查看输出
# 你应该看到三个测试案例，每个都显示了不同维度的分数
```

**预期输出：**
```
Test 1 - 完整回答:
  Response: Let me solve this step by step. 2 + 2 equals 4...
  Results: {'score': 0.78, 'correctness': 1.0, 'completeness': 1.0, ...}
                              ↑ 看到这些维度就成功了！

Test 2 - 简短回答:
  Results: {'score': 0.63, 'correctness': 1.0, 'completeness': 0.5, ...}

Test 3 - 错误回答:
  Results: {'score': 0.26, 'correctness': 0.0, 'completeness': 0.2, ...}
```

✅ **如果看到上面的输出，说明reward函数工作正常！**

---

## 🚀 完整训练验证（需要GPU，10-15分钟）

验证在实际训练中是否能打印各维度。

### 前置条件

- [x] 有GPU
- [x] 已下载模型（推荐 `Qwen/Qwen2.5-0.5B`）
- [x] 有训练数据（GSM8K或自己的数据）

### 运行方式

**方式1：使用现有GSM8K数据（最简单）**

```bash
cd /home/user/verl
bash toy_demo_rubric/run_simple_demo.sh
```

脚本会自动：
1. 检测GSM8K数据集
2. 加载rubric tracking patch
3. 运行2步训练
4. 打印各维度指标

**方式2：创建toy数据集**

如果没有GSM8K数据：

```bash
# 步骤1: 安装pandas（如果需要）
pip install pandas

# 步骤2: 创建toy数据集
python3 toy_demo_rubric/create_toy_dataset.py

# 步骤3: 运行训练
bash toy_demo_rubric/run_toy_demo.sh
```

**方式3：一键运行（包含所有步骤）**

```bash
bash toy_demo_rubric/quickstart.sh
```

### 验证成功的标志

在训练输出中搜索这些关键字：

```bash
# 应该看到类似这样的输出：
train/reward/correctness/mean: 0.75    ← 正确性维度
train/reward/correctness/max: 1.00
train/reward/correctness/min: 0.50
train/reward/completeness/mean: 0.65   ← 全面性维度
train/reward/practicality/mean: 0.55   ← 实用性维度
```

✅ **如果每一步训练都能看到这些指标，说明方法验证成功！**

---

## 📁 核心文件说明

### 1. `rubric_reward.py` ⭐ 最重要

这是你需要自定义的文件，定义如何评估各维度。

**关键代码：**
```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # 评估三个维度
    correctness = evaluate_correctness(...)
    completeness = evaluate_completeness(...)
    practicality = evaluate_practicality(...)

    # 返回字典 - 这是核心！
    return {
        'score': 0.5*correctness + 0.3*completeness + 0.2*practicality,
        'correctness': correctness,
        'completeness': completeness,
        'practicality': practicality,
    }
```

### 2. `patch_ray_trainer.py` ⭐ 核心机制

让trainer能够提取并打印各维度。

**工作原理：**
- 从 `batch.non_tensor_batch` 中提取各维度
- 计算统计信息（mean, max, min, std）
- 添加到 metrics 中
- trainer自动打印

---

## 🎓 理解数据流

```
1. 你的reward函数返回字典
   ↓
   {'score': 0.75, 'correctness': 0.8, 'completeness': 0.7, ...}

2. RewardManager处理
   ↓
   reward_extra_info['correctness'].append(0.8)
   reward_extra_info['completeness'].append(0.7)

3. 存储到batch
   ↓
   batch.non_tensor_batch['correctness'] = np.array([0.8, 0.9, ...])

4. Patch提取
   ↓
   metrics['train/reward/correctness/mean'] = np.mean([0.8, 0.9, ...])

5. Logger打印
   ↓
   终端显示: train/reward/correctness/mean: 0.85
```

---

## 🔧 自定义你的Rubric

### 修改评分维度

编辑 `rubric_reward.py`：

```python
# 添加新维度
def evaluate_fluency(solution_str):
    # 你的评估逻辑
    return score

# 在compute_score中返回
return {
    'score': final_score,
    'correctness': ...,
    'completeness': ...,
    'practicality': ...,
    'fluency': evaluate_fluency(...),  # 新维度
}
```

### 修改权重

```python
weights = {
    'correctness': 0.6,   # 从0.5改为0.6
    'completeness': 0.2,  # 从0.3改为0.2
    'practicality': 0.2
}
```

---

## 📚 文档导航

- **START_HERE.md** ← 你在这里（快速开始）
- **USAGE.md** - 详细使用指南，包含所有使用方式
- **README.md** - 技术文档，解释原理和实现细节

---

## ❓ 常见问题

### Q: 必须用GPU吗？
A: 测试reward函数不需要GPU，但完整训练需要。

### Q: 需要什么模型？
A: 推荐 `Qwen/Qwen2.5-0.5B`（小模型，快速测试）

### Q: 怎么应用到我的项目？
A:
1. 复制 `rubric_reward.py` 并修改评估逻辑
2. 在训练时添加参数：
   ```bash
   custom_reward_function.path=your_rubric_reward.py
   custom_reward_function.name=compute_score
   ```
3. 使用 `patch_ray_trainer.py` 或直接修改源代码

### Q: 验证失败怎么办？
A:
1. 先测试reward函数：`python3 rubric_reward.py`
2. 检查patch是否加载
3. 查看日志中是否有错误
4. 参考 USAGE.md 中的故障排除章节

---

## 🎯 下一步行动

- [ ] 运行快速验证（3分钟）
- [ ] 理解代码工作原理
- [ ] 修改成你自己的评分逻辑
- [ ] 运行完整训练验证（可选）
- [ ] 应用到实际项目

---

## 💡 小贴士

1. **从简单开始**：先用3行代码测试reward函数
2. **逐步验证**：先测函数，再测训练，最后上规模
3. **保存日志**：`bash run_simple_demo.sh 2>&1 | tee output.log`
4. **可视化**：使用WandB查看各维度的学习曲线

祝你验证顺利！如有问题，查看 USAGE.md 的常见问题章节。🎉
