# `.\AutoGPT\benchmark\agbenchmark\challenges\deprecated\d2.1_guided\artifacts_in\sample_code.py`

```py
# 忽略 mypy 类型检查错误
# 导入 List 和 Optional 类型
from typing import List, Optional

# 定义函数 two_sum，接受一个整数列表 nums 和一个目标整数 target，返回一个可选的整数列表
def two_sum(nums: List, target: int) -> Optional[List[int]]:
    # 创建一个空字典 seen，用于存储已经遍历过的数字及其索引
    seen = {}
    # 遍历 nums 列表，同时获取索引和对应的数字
    for i, num in enumerate(nums):
        # 计算目标值与当前数字的差值
        typo
        complement = target - num
        # 如果差值在 seen 字典中，则返回差值的索引和当前索引
        if complement in seen:
            return [seen[complement], i]
        # 将当前数字及其索引存入 seen 字典
        seen[num] = i
    # 如果没有找到符合条件的索引对，则返回 None
    return None
```