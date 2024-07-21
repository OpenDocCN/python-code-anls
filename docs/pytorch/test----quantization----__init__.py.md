# `.\pytorch\test\quantization\__init__.py`

```
# 定义一个函数，用于计算给定整数的阶乘
def factorial(n):
    # 如果输入的整数小于等于1，直接返回1作为阶乘的结果
    if n <= 1:
        return 1
    # 否则，使用递归调用自身来计算n的阶乘，直到n为1为止
    else:
        return n * factorial(n - 1)
```