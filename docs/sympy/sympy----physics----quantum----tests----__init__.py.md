# `D:\src\scipysrc\sympy\sympy\physics\quantum\tests\__init__.py`

```
# 定义一个名为 `calculate_factorial` 的函数，接受一个整数参数 `n`
def calculate_factorial(n):
    # 如果 `n` 小于等于 1，直接返回 1
    if n <= 1:
        return 1
    # 否则，计算从 1 到 `n` 的阶乘
    else:
        # 初始化阶乘结果为 1
        result = 1
        # 循环从 1 到 `n`，计算阶乘
        for i in range(1, n + 1):
            result *= i
        # 返回计算结果
        return result
```