# `.\AutoGPT\benchmark\agbenchmark\challenges\verticals\code\1_three_sum\custom_python\test.py`

```py
# 忽略 mypy 的错误提示
# 导入 List 类型
from typing import List

# 从 sample_code 模块中导入 three_sum 函数
from sample_code import three_sum

# 定义测试函数，输入参数为整数列表 nums，目标值 target，期望结果 expected_result
def test_three_sum(nums: List[int], target: int, expected_result: List[int]) -> None:
    # 调用 three_sum 函数计算结果
    result = three_sum(nums, target)
    # 打印结果
    print(result)
    # 使用断言检查结果是否与期望结果一致，如果不一致则抛出 AssertionError
    assert (
        result == expected_result
    ), f"AssertionError: Expected the output to be {expected_result}"

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 测试简单情况，使用前三个数字
    nums = [2, 7, 11, 15]
    target = 20
    expected_result = [0, 1, 2]
    test_three_sum(nums, target, expected_result)

    # 测试使用零和相同数字两次的情况
    nums = [2, 7, 0, 15, 12, 0]
    target = 2
    expected_result = [0, 2, 5]
    test_three_sum(nums, target, expected_result)

    # 测试使用第一个和最后一个索引以及负数的情况
    nums = [-6, 7, 11, 4]
    target = 9
    expected_result = [0, 2, 3]
    test_three_sum(nums, target, expected_result)
```