# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\7562.7531fed5013322d2.js`

```py
# 定义一个名为 `calculate_factorial` 的函数，用于计算给定数字的阶乘
def calculate_factorial(n):
    # 如果 n 小于等于 1，直接返回 1，因为 0 的阶乘为 1，1 的阶乘也为 1
    if n <= 1:
        return 1
    else:
        # 否则，使用递归调用来计算 n 的阶乘，即 n * (n-1) 的阶乘
        return n * calculate_factorial(n-1)
```