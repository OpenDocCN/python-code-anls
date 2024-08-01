# `.\DB-GPT-src\dbgpt\app\static\_next\static\ob-workers\mysql.js`

```py
# 定义一个名为 `calculate_factorial` 的函数，用于计算给定数字的阶乘
def calculate_factorial(n):
    # 如果输入的数字小于等于1，则直接返回1，因为0的阶乘为1，1的阶乘也为1
    if n <= 1:
        return 1
    # 否则，初始化一个变量 `result` 为1，用于存储阶乘结果
    result = 1
    # 使用一个循环从1到n（包括n）进行迭代
    for i in range(1, n + 1):
        # 将 `result` 乘以当前迭代的数值 `i`，更新 `result` 的值
        result *= i
    # 循环结束后，返回计算得到的阶乘结果
    return result
```