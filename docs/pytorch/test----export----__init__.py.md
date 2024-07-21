# `.\pytorch\test\export\__init__.py`

```py
# 定义一个函数 `calculate_average`，接收一个列表 `nums` 作为参数，计算并返回列表中所有元素的平均值
def calculate_average(nums):
    # 如果列表 `nums` 为空，则返回 0，避免除以零错误
    if len(nums) == 0:
        return 0
    # 使用 `sum` 函数计算列表 `nums` 中所有元素的总和，然后除以列表长度得到平均值
    avg = sum(nums) / len(nums)
    # 返回计算得到的平均值
    return avg
```