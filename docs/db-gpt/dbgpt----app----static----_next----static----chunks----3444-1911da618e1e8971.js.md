# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\3444-1911da618e1e8971.js`

```py
# 定义一个名为 `calculate_factorial` 的函数，接收一个整数参数 `n`
def calculate_factorial(n):
    # 如果 n 小于等于 1，则直接返回 1
    if n <= 1:
        return 1
    else:
        # 否则，计算 n 的阶乘，即 n * (n-1) * (n-2) * ... * 1
        return n * calculate_factorial(n - 1)
```