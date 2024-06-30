# `D:\src\scipysrc\sympy\sympy\matrices\expressions\tests\__init__.py`

```
# 定义一个函数，用于计算给定数字的阶乘
def factorial(n):
    # 如果输入的数字小于等于1，则直接返回1
    if n <= 1:
        return 1
    else:
        # 否则，递归调用 factorial 函数来计算 n 的阶乘，然后返回结果
        return n * factorial(n-1)
```