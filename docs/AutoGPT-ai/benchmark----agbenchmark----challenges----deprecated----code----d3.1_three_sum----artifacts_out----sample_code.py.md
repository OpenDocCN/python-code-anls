# `.\AutoGPT\benchmark\agbenchmark\challenges\deprecated\code\d3.1_three_sum\artifacts_out\sample_code.py`

```py
# 导入 List 和 Optional 类型提示
from typing import List, Optional

# 寻找列表中三个数的和等于目标值的索引，如果不存在则返回 None
def three_sum(nums: List[int], target: int) -> Optional[List[int]]:
    # 将数字和对应的索引组成元组，存储在列表中
    nums_indices = [(num, index) for index, num in enumerate(nums)]
    # 对元组列表按数字大小进行排序
    nums_indices.sort()
    # 遍历元组列表，找到三个数的和等于目标值的索引
    for i in range(len(nums_indices) - 2):
        # 如果当前数字和前一个数字相同，则跳过
        if i > 0 and nums_indices[i] == nums_indices[i - 1]:
            continue
        # 初始化左右指针
        l, r = i + 1, len(nums_indices) - 1
        # 在左右指针范围内查找三个数的和
        while l < r:
            # 计算三个数的和
            three_sum = nums_indices[i][0] + nums_indices[l][0] + nums_indices[r][0]
            # 如果和小于目标值，左指针右移
            if three_sum < target:
                l += 1
            # 如果和大于目标值，右指针左移
            elif three_sum > target:
                r -= 1
            # 如果和等于目标值，返回三个数的索引
            else:
                # 对索引进行排序，返回结果
                indices = sorted(
                    [nums_indices[i][1], nums_indices[l][1], nums_indices[r][1]]
                )
                return indices
    # 如果没有找到符合条件的索引，返回 None
    return None
```