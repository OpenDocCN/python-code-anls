# `.\pytorch\torch\_vendor\__init__.py`

```py
# 定义一个函数，计算并返回给定列表中所有正数的平均值
def average_positive(nums):
    # 使用列表推导式过滤出所有正数
    positive_nums = [x for x in nums if x > 0]
    # 如果没有找到正数，则返回 None
    if not positive_nums:
        return None
    # 计算正数的平均值并返回
    return sum(positive_nums) / len(positive_nums)
```