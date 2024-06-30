# `D:\src\scipysrc\sympy\sympy\liealgebras\tests\__init__.py`

```
# 定义一个函数，计算给定数字列表中的平均值
def calculate_average(numbers):
    # 如果列表为空，则返回0
    if not numbers:
        return 0
    # 对列表中的所有数字求和
    total = sum(numbers)
    # 计算平均值
    avg = total / len(numbers)
    # 返回计算出的平均值
    return avg
```