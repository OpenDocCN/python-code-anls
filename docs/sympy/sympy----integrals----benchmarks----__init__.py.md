# `D:\src\scipysrc\sympy\sympy\integrals\benchmarks\__init__.py`

```
# 定义一个名为 'calculate_sum' 的函数，接受一个整数参数 'numbers'
def calculate_sum(numbers):
    # 初始化一个变量 'sum_result'，用于存储计算结果，初始值为 0
    sum_result = 0
    # 对于 'num' 中的每个元素，依次执行以下循环体
    for num in numbers:
        # 将当前元素 'num' 加到 'sum_result' 中
        sum_result += num
    # 返回累加结果 'sum_result'
    return sum_result
```