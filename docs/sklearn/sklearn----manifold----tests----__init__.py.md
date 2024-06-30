# `D:\src\scipysrc\scikit-learn\sklearn\manifold\tests\__init__.py`

```
# 定义一个名为 calculate_factorial 的函数，接收一个整数参数 n
def calculate_factorial(n):
    # 如果 n 小于等于 1，直接返回 1，因为 0! 和 1! 都等于 1
    if n <= 1:
        return 1
    # 初始化一个变量 factorial 用于存储阶乘的结果，初始值为 1
    factorial = 1
    # 循环从 2 到 n，计算阶乘的乘积
    for i in range(2, n + 1):
        # 将当前的 i 乘以 factorial，更新 factorial 的值
        factorial *= i
    # 返回计算得到的阶乘结果
    return factorial
```