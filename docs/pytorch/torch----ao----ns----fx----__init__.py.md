# `.\pytorch\torch\ao\ns\fx\__init__.py`

```
# 定义一个名为`average`的函数，接受一个参数`nums`，这个参数是一个列表
def average(nums):
    # 如果列表`nums`为空，则返回0.0作为平均值
    if not nums:
        return 0.0
    # 否则，计算`nums`列表中所有元素的和，并且除以列表长度，得到平均值
    return sum(nums) / len(nums)
```