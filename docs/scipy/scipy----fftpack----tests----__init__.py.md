# `D:\src\scipysrc\scipy\scipy\fftpack\tests\__init__.py`

```
# 定义一个名为 `calculate_factorial` 的函数，用于计算给定数字的阶乘
def calculate_factorial(n):
    # 如果输入的数字是0或1，直接返回1，因为0的阶乘和1的阶乘均为1
    if n == 0 or n == 1:
        return 1
    else:
        # 否则，使用递归方式计算阶乘，即 n * (n-1) 的阶乘
        return n * calculate_factorial(n - 1)
```