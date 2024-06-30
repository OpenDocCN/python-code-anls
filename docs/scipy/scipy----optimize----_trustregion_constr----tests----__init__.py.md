# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\tests\__init__.py`

```
# 定义一个名为 `average` 的函数，用于计算列表中所有元素的平均值
def average(nums):
    # 如果列表为空，则直接返回 0
    if not nums:
        return 0
    # 否则，计算列表中所有元素的总和
    total = sum(nums)
    # 返回总和除以列表中元素的个数，即平均值
    return total / len(nums)
```