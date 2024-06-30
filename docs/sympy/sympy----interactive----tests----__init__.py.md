# `D:\src\scipysrc\sympy\sympy\interactive\tests\__init__.py`

```
# 定义一个名为 calc_factorial 的函数，用来计算给定数字的阶乘
def calc_factorial(n):
    # 如果 n 小于等于 1，直接返回 1，因为 0! 和 1! 都等于 1
    if n <= 1:
        return 1
    else:
        # 否则，使用递归调用自身来计算 n 的阶乘，即 n * (n-1) 的阶乘
        return n * calc_factorial(n-1)
```