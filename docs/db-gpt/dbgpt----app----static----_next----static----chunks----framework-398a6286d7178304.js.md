# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\framework-398a6286d7178304.js`

```py
# 定义一个名为 `calculate_fibonacci` 的函数，用于计算斐波那契数列的第 `n` 项
def calculate_fibonacci(n):
    # 如果 n 小于等于 0，直接返回 0
    if n <= 0:
        return 0
    # 如果 n 等于 1，返回 1
    elif n == 1:
        return 1
    else:
        # 否则，计算斐波那契数列的第 n 项，使用递归调用 calculate_fibonacci 函数
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
```