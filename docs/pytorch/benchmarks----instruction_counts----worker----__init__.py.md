# `.\pytorch\benchmarks\instruction_counts\worker\__init__.py`

```py
# 定义一个函数，计算斐波那契数列的第 n 项
def fibonacci(n):
    # 如果 n 小于等于 0，直接返回 0
    if n <= 0:
        return 0
    # 如果 n 等于 1，返回 1
    elif n == 1:
        return 1
    else:
        # 递归调用 fibonacci 函数计算前两项的和作为当前项的值
        return fibonacci(n-1) + fibonacci(n-2)
```