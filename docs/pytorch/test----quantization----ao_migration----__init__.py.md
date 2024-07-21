# `.\pytorch\test\quantization\ao_migration\__init__.py`

```
# 定义一个名为 calculate_factorial 的函数，接收一个整数参数 n
def calculate_factorial(n):
    # 如果 n 小于等于 1，则直接返回 1，因为 0! = 1! = 1
    if n <= 1:
        return 1
    # 否则，初始化变量 result 为 1，用于存储阶乘的结果
    result = 1
    # 循环计算从 2 到 n 的所有整数的乘积，即计算 n!
    for i in range(2, n + 1):
        result *= i
    # 返回计算结果
    return result
```