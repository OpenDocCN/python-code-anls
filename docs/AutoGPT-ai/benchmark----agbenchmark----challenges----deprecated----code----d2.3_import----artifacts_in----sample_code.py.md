# `.\AutoGPT\benchmark\agbenchmark\challenges\deprecated\code\d2.3_import\artifacts_in\sample_code.py`

```py
# 导入 List 和 Optional 类型提示
from typing import List, Optional

# 定义函数，接受一个整数列表和目标值，返回两个数的索引，使它们的和等于目标值
def two_sum(nums: List, target: int) -> Optional[List[int]]:
    # 用于存储已经遍历过的数字及其索引
    seen = {}
    # 遍历列表中的数字及其索引
    for i, num in enumerate(nums):
        # 计算目标值与当前数字的差值
        complement = target - num
        # 如果差值在 seen 中存在，则返回差值的索引和当前数字的索引
        if complement in seen:
            return [seen[complement], i]
        # 将当前数字及其索引存入 seen 中
        seen[num] = i
    # 如果没有找到符合条件的索引，则返回 None
    return None
```