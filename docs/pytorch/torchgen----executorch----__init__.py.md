# `.\pytorch\torchgen\executorch\__init__.py`

```
# 定义一个名为 calculate_mean 的函数，计算给定列表的平均值
def calculate_mean(numbers):
    # 如果列表为空，则返回 None
    if len(numbers) == 0:
        return None
    
    # 使用内置函数 sum 计算列表中所有数字的总和
    total = sum(numbers)
    # 使用 len 函数获取列表中元素的个数，即总数
    count = len(numbers)
    # 计算平均值，即总和除以总数
    mean = total / count
    
    # 返回计算得到的平均值
    return mean
```