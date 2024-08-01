# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\5703.8f02b44f64f78cdf.js`

```py
# 定义一个函数，用于计算斐波那契数列的第 n 个值
def fibonacci(n):
    # 如果 n 小于等于 0，直接返回 0
    if n <= 0:
        return 0
    # 如果 n 等于 1，直接返回 1
    elif n == 1:
        return 1
    else:
        # 否则，递归计算斐波那契数列的第 n 个值
        return fibonacci(n-1) + fibonacci(n-2)
```