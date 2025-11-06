"""
自定义的Rubric Reward函数
评估三个维度：correctness（正确性）、completeness（全面性）、practicality（实用性）
"""
import re


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    多维度Rubric评分函数

    Args:
        data_source: 数据源名称
        solution_str: 模型生成的回答
        ground_truth: 正确答案
        extra_info: 额外信息

    Returns:
        dict: 包含各维度分数和总分
    """

    # 1. 评估正确性 (Correctness)
    correctness = evaluate_correctness(solution_str, ground_truth)

    # 2. 评估全面性 (Completeness)
    completeness = evaluate_completeness(solution_str)

    # 3. 评估实用性 (Practicality)
    practicality = evaluate_practicality(solution_str)

    # 线性加权计算总分
    weights = {
        'correctness': 0.5,
        'completeness': 0.3,
        'practicality': 0.2
    }

    final_score = (
        weights['correctness'] * correctness +
        weights['completeness'] * completeness +
        weights['practicality'] * practicality
    )

    # 返回字典，包含所有维度
    return {
        'score': final_score,           # 必须有这个key，用于训练
        'correctness': correctness,     # 正确性分数
        'completeness': completeness,   # 全面性分数
        'practicality': practicality,   # 实用性分数
        'response_length': len(solution_str),  # 额外信息：回答长度
    }


def evaluate_correctness(solution_str, ground_truth):
    """评估答案的正确性"""
    # 提取数字
    solution_numbers = re.findall(r'\d+', solution_str)
    ground_truth_numbers = re.findall(r'\d+', ground_truth)

    if not solution_numbers:
        return 0.0

    # 检查是否包含正确答案
    if ground_truth in solution_numbers:
        return 1.0
    elif any(num in ground_truth_numbers for num in solution_numbers):
        return 0.5  # 部分正确
    else:
        return 0.0


def evaluate_completeness(solution_str):
    """评估答案的全面性"""
    # 基于回答的详细程度
    length = len(solution_str)

    if length >= 50:
        return 1.0  # 非常详细
    elif length >= 30:
        return 0.7  # 中等详细
    elif length >= 15:
        return 0.5  # 基本完整
    else:
        return 0.2  # 过于简短


def evaluate_practicality(solution_str):
    """评估答案的实用性"""
    # 检查是否包含解释性词汇
    practical_keywords = [
        'because', 'so', 'therefore', 'equals', 'is',
        '因为', '所以', '等于', '答案是'
    ]

    lower_solution = solution_str.lower()
    keyword_count = sum(1 for keyword in practical_keywords if keyword in lower_solution)

    # 根据包含的关键词数量评分
    if keyword_count >= 3:
        return 1.0
    elif keyword_count == 2:
        return 0.7
    elif keyword_count == 1:
        return 0.5
    else:
        return 0.3


# 测试函数
if __name__ == "__main__":
    # 测试案例1：完整准确的回答
    test_response1 = "Let me solve this step by step. 2 + 2 equals 4 because we add two and two together."
    result1 = compute_score("toy_math", test_response1, "4")
    print("Test 1 - 完整回答:")
    print(f"  Response: {test_response1}")
    print(f"  Results: {result1}")
    print()

    # 测试案例2：简短但正确的回答
    test_response2 = "The answer is 4."
    result2 = compute_score("toy_math", test_response2, "4")
    print("Test 2 - 简短回答:")
    print(f"  Response: {test_response2}")
    print(f"  Results: {result2}")
    print()

    # 测试案例3：错误的回答
    test_response3 = "I think it's 5."
    result3 = compute_score("toy_math", test_response3, "4")
    print("Test 3 - 错误回答:")
    print(f"  Response: {test_response3}")
    print(f"  Results: {result3}")
