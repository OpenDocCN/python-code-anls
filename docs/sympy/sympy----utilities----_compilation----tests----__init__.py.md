# `D:\src\scipysrc\sympy\sympy\utilities\_compilation\tests\__init__.py`

```
# 定义一个函数，计算指定整数的阶乘
def factorial(n):
    # 如果输入的数字是0或1，直接返回1，因为0的阶乘和1的阶乘都是1
    if n == 0 or n == 1:
        return 1
    else:
        # 否则，递归计算n的阶乘，即n乘以(n-1)的阶乘
        return n * factorial(n-1)
```