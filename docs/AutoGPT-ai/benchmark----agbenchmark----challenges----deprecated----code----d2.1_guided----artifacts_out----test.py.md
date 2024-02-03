# `.\AutoGPT\benchmark\agbenchmark\challenges\deprecated\code\d2.1_guided\artifacts_out\test.py`

```py
# 从 typing 模块中导入 List 类型
from typing import List

# 从 sample_code 模块中导入 two_sum 函数
from sample_code import two_sum

# 定义一个测试函数，用于测试 two_sum 函数的输出是否符合预期
def test_two_sum(nums: List, target: int, expected_result: List[int]) -> None:
    # 调用 two_sum 函数计算结果
    result = two_sum(nums, target)
    # 打印结果
    print(result)
    # 使用断言检查结果是否符合预期
    assert (
        result == expected_result
    ), f"AssertionError: Expected the output to be {expected_result}"

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 测试简单情况，使用前两个数字
    nums = [2, 7, 11, 15]
    target = 9
    expected_result = [0, 1]
    test_two_sum(nums, target, expected_result)

    # 测试使用零和相同数字两次的情况
    nums = [2, 7, 0, 15, 12, 0]
    target = 0
    expected_result = [2, 5]
    test_two_sum(nums, target, expected_result)

    # 测试使用第一个和最后一个索引以及负数的情况
    nums = [-6, 7, 11, 4]
    target = -2
    expected_result = [0, 3]
    test_two_sum(nums, target, expected_result)
```