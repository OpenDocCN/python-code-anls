# `D:\src\scipysrc\seaborn\seaborn\_stats\__init__.py`

```
# 定义一个名为 `calculate_average` 的函数，接收一个参数 `numbers`，这个参数是一个列表，包含要计算平均值的数字
def calculate_average(numbers):
    # 如果 `numbers` 列表为空，则返回 None
    if not numbers:
        return None
    # 使用内置函数 `sum()` 对列表 `numbers` 中的所有数字求和，并除以 `numbers` 列表的长度，得到平均值
    avg = sum(numbers) / len(numbers)
    # 返回计算得到的平均值
    return avg
```