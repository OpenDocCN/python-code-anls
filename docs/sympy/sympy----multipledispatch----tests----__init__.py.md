# `D:\src\scipysrc\sympy\sympy\multipledispatch\tests\__init__.py`

```
# 定义一个名为 calculate_factorial 的函数，接受一个整数参数 n
def calculate_factorial(n):
    # 初始化一个变量 result 为 1，用于存储阶乘的结果
    result = 1
    # 使用 for 循环从 1 到 n (包含 n) 进行迭代
    for i in range(1, n + 1):
        # 将 result 与当前迭代值 i 相乘，更新 result 的值
        result *= i
    # 返回计算得到的阶乘结果
    return result
```