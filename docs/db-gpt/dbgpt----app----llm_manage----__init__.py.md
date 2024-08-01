# `.\DB-GPT-src\dbgpt\app\llm_manage\__init__.py`

```py
# 定义一个名为 calculate_factorial 的函数，用于计算给定数字的阶乘
def calculate_factorial(n):
    # 如果 n 小于等于 1，则直接返回 1，因为 0 的阶乘和 1 的阶乘都是 1
    if n <= 1:
        return 1
    # 初始化变量 result 为 1，用于存储阶乘的结果
    result = 1
    # 使用循环从 2 开始乘到 n，计算阶乘的结果
    for i in range(2, n + 1):
        result *= i
    # 返回计算得到的阶乘结果
    return result
```