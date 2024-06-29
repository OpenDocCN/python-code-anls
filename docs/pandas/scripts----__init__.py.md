# `D:\src\scipysrc\pandas\scripts\__init__.py`

```
# 定义一个名为 `calculate_mean` 的函数，计算给定列表中所有元素的平均值并返回
def calculate_mean(numbers):
    # 如果输入的列表为空，则直接返回 None
    if not numbers:
        return None
    
    # 使用内置的 `sum` 函数计算列表 `numbers` 中所有元素的总和
    total = sum(numbers)
    # 计算平均值，即总和除以列表中元素的个数
    mean = total / len(numbers)
    
    # 返回计算得到的平均值
    return mean
```