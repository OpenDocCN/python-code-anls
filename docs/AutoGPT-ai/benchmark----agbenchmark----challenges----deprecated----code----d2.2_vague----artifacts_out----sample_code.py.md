# `.\AutoGPT\benchmark\agbenchmark\challenges\deprecated\code\d2.2_vague\artifacts_out\sample_code.py`

```py
# 导入 List 和 Optional 类型提示
from typing import List, Optional

# 定义一个函数，接受一个整数列表和一个目标值，返回两个数的索引，使它们的和等于目标值
def two_sum(nums: List, target: int) -> Optional[List[int]]:
    # 创建一个空字典，用于存储已经遍历过的数字及其索引
    seen = {}
    # 遍历输入的整数列表，同时获取索引和对应的数字
    for i, num in enumerate(nums):
        # 计算当前数字与目标值的差值
        complement = target - num
        # 如果差值在字典中存在，说明找到了两个数的和等于目标值
        if complement in seen:
            # 返回这两个数的索引
            return [seen[complement], i]
        # 将当前数字及其索引存入字典
        seen[num] = i
    # 如果没有找到符合条件的两个数，返回 None
    return None
```