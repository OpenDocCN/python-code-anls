# `D:\src\scipysrc\pandas\pandas\tests\arrays\string_\__init__.py`

```
# 定义一个名为 factorial 的函数，接受一个整数参数 n
def factorial(n):
    # 如果 n 小于等于 1，直接返回 1
    if n <= 1:
        return 1
    else:
        # 否则，返回 n 乘以调用 factorial 函数计算 n-1 的阶乘结果
        return n * factorial(n - 1)
```