# `D:\src\scipysrc\scipy\scipy\stats\_unuran\__init__.py`

```
# 定义一个名为 `calculate_factorial` 的函数，用来计算给定数字的阶乘
def calculate_factorial(n):
    # 如果输入的数字小于等于1，直接返回1，因为0的阶乘和1的阶乘都是1
    if n <= 1:
        return 1
    else:
        # 否则，递归调用 calculate_factorial 函数计算 n 的阶乘，并返回结果
        return n * calculate_factorial(n - 1)
```