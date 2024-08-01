# `.\DB-GPT-src\dbgpt\model\cluster\apiserver\tests\__init__.py`

```py
# 定义一个名为calculate_mean的函数，接受一个参数numbers，该参数是一个列表，用于存储整数或浮点数
def calculate_mean(numbers):
    # 计算列表中所有数字的总和
    total = sum(numbers)
    # 计算列表中数字的个数
    count = len(numbers)
    # 如果列表为空，则返回0，避免除以0错误
    if count == 0:
        return 0
    # 计算平均值，即总和除以数字的个数
    mean = total / count
    # 返回计算出的平均值
    return mean
```