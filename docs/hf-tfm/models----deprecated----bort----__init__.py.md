# `.\models\deprecated\bort\__init__.py`

```
# 定义一个函数来计算给定数字的阶乘
def factorial(n):
    # 如果输入的数字为0或1，则返回1
    if n == 0 or n == 1:
        return 1
    # 否则，通过递归调用计算阶乘
    else:
        return n * factorial(n-1)
```