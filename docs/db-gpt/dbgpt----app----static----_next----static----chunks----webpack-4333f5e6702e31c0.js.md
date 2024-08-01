# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\webpack-4333f5e6702e31c0.js`

```py
# 定义一个名为calculate_factorial的函数，接收一个整数n作为参数，计算并返回n的阶乘
def calculate_factorial(n):
    # 如果n小于等于1，直接返回1，因为0的阶乘和1的阶乘都是1
    if n <= 1:
        return 1
    else:
        # 否则，计算n的阶乘，使用递归调用自身来计算n-1的阶乘，并将两者相乘
        return n * calculate_factorial(n - 1)
```