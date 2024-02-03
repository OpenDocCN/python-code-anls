# `.\AutoGPT\benchmark\agbenchmark\challenges\verticals\code\1_three_sum\artifacts_out\sample_code.py`

```py
# 忽略 mypy 类型检查错误
# 导入 List 和 Optional 类型
from typing import List, Optional

# 定义函数 three_sum，接受一个整数列表 nums 和目标值 target，返回一个可选的整数列表
def three_sum(nums: List[int], target: int) -> Optional[List[int]]:
    # 将 nums 中的元素和索引组成元组，存储在 nums_indices 列表中
    nums_indices = [(num, index) for index, num in enumerate(nums)]
    # 对 nums_indices 列表进行排序
    nums_indices.sort()
    # 遍历 nums_indices 列表，i 从 0 到倒数第三个元素
    for i in range(len(nums_indices) - 2):
        # 如果 i 大于 0 并且当前元素与前一个元素相同，则跳过当前循环
        if i > 0 and nums_indices[i] == nums_indices[i - 1]:
            continue
        # 初始化左右指针 l 和 r
        l, r = i + 1, len(nums_indices) - 1
        # 当 l 小于 r 时循环
        while l < r:
            # 计算当前三个数的和
            three_sum = nums_indices[i][0] + nums_indices[l][0] + nums_indices[r][0]
            # 如果和小于目标值，则左指针右移
            if three_sum < target:
                l += 1
            # 如果和大于目标值，则右指针左移
            elif three_sum > target:
                r -= 1
            # 如果和等于目标值
            else:
                # 将三个数的索引排序后返回
                indices = sorted(
                    [nums_indices[i][1], nums_indices[l][1], nums_indices[r][1]]
                )
                return indices
    # 如果没有找到符合条件的三个数，则返回 None
    return None
```