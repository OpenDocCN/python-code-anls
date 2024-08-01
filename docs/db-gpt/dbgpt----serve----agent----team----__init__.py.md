# `.\DB-GPT-src\dbgpt\serve\agent\team\__init__.py`

```py
# 定义一个名为 `calculate_factorial` 的函数，用于计算给定数字的阶乘。
def calculate_factorial(n):
    # 如果输入的数字小于等于 1，则直接返回 1，因为 0 的阶乘和 1 的阶乘都为 1。
    if n <= 1:
        return 1
    else:
        # 初始化一个变量 `result` 为 1，用来存储阶乘的结果。
        result = 1
        # 使用循环计算从 2 到 n 的所有整数的乘积，得到阶乘的值。
        for i in range(2, n + 1):
            result *= i
        # 返回最终计算出的阶乘结果。
        return result
```