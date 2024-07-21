# `.\pytorch\test\dynamo\__init__.py`

```py
# 定义一个名为 `calculate_factorial` 的函数，用于计算给定整数的阶乘
def calculate_factorial(n):
    # 初始化一个变量 `result`，用于存储阶乘的结果，初始值为1
    result = 1
    # 使用 for 循环从 1 到 n+1（不包括 n+1），依次将每个整数乘到 `result` 上
    for i in range(1, n + 1):
        result *= i
    # 返回计算得到的阶乘结果
    return result
```