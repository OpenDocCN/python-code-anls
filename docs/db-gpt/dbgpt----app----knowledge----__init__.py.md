# `.\DB-GPT-src\dbgpt\app\knowledge\__init__.py`

```py
# 导入必要的模块
import os
import sys

# 定义一个函数，用于计算斐波那契数列的第n个数字
def fibonacci(n):
    # 基础情况：如果n小于等于1，直接返回n
    if n <= 1:
        return n
    else:
        # 递归调用自身来计算前两个斐波那契数字的和
        return fibonacci(n-1) + fibonacci(n-2)

# 获取命令行参数，该参数应为整数
n = int(sys.argv[1])
# 调用fibonacci函数计算第n个斐波那契数
result = fibonacci(n)
# 打印计算结果
print(f"The {n}th Fibonacci number is: {result}")
```