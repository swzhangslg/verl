#!/usr/bin/env python3
"""
创建一个简单的toy数据集用于测试rubric reward
"""
import pyarrow as pa
import pyarrow.parquet as pq
import os

# 创建简单的数学问题数据集
prompts = [
    'What is 2 + 2?',
    'What is 5 * 3?',
    'What is 10 - 4?',
    'What is 8 / 2?',
    'What is 3 + 5?',
    'What is 6 * 2?',
    'What is 9 - 3?',
    'What is 12 / 3?',
] * 4  # 重复4次，得到32个样本

ground_truths = [
    '4', '15', '6', '4', '8', '12', '6', '4'
] * 4

data_sources = ['toy_math'] * 32

# 创建reward_model字典列表
reward_models = [{'ground_truth': gt} for gt in ground_truths]

# 创建Arrow Table
table = pa.table({
    'prompt': prompts,
    'data_source': data_sources,
    'reward_model': reward_models,
})

# 创建目录
os.makedirs('toy_demo_rubric/data', exist_ok=True)

# 分割训练集和测试集
train_table = table.slice(0, 24)  # 前24个作为训练集
test_table = table.slice(24, 8)   # 后8个作为测试集

train_path = 'toy_demo_rubric/data/train.parquet'
test_path = 'toy_demo_rubric/data/test.parquet'

# 保存parquet文件
pq.write_table(train_table, train_path)
pq.write_table(test_table, test_path)

print(f"✅ 训练集已创建: {train_path} ({len(train_table)} 样本)")
print(f"✅ 测试集已创建: {test_path} ({len(test_table)} 样本)")
print("\n数据示例（前5个）:")
print(train_table.slice(0, 5).to_pandas())
