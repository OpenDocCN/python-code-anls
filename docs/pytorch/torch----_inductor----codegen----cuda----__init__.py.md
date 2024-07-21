# `.\pytorch\torch\_inductor\codegen\cuda\__init__.py`

```
# 定义一个名为 "calculate_factorial" 的函数，接收一个整数参数 "n"
def calculate_factorial(n):
    # 如果 n 小于等于 1，则直接返回 1，因为 0! 和 1! 都等于 1
    if n <= 1:
        return 1
    # 初始化一个变量 "result"，用来保存阶乘的结果，初始值为 1
    result = 1
    # 使用循环计算 n 的阶乘，从 2 到 n，依次累乘到 "result" 中
    for i in range(2, n + 1):
        result *= i
    # 返回计算得到的阶乘结果
    return result
```