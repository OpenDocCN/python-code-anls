# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\6165-48eaed9a80fbbd1b.js`

```py
# 定义一个名为 `calculate_fibonacci` 的函数，用于计算斐波那契数列中第 `n` 个数的值
def calculate_fibonacci(n):
    # 如果 `n` 小于等于 0，直接返回 0
    if n <= 0:
        return 0
    # 如果 `n` 等于 1，直接返回 1
    elif n == 1:
        return 1
    else:
        # 初始化斐波那契数列的前两个数
        fib = [0, 1]
        # 循环从第三个数开始计算到第 `n` 个数
        for i in range(2, n + 1):
            # 计算当前数的值，即前两个数的和
            fib.append(fib[-1] + fib[-2])
        # 返回第 `n` 个斐波那契数列的值
        return fib[-1]
```