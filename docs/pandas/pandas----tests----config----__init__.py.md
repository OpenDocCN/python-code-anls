# `D:\src\scipysrc\pandas\pandas\tests\config\__init__.py`

```
# 定义一个名为 factorial 的函数，接受一个整数参数 n
def factorial(n):
    # 如果 n 小于等于 1，直接返回 1，这是阶乘的基本规则
    if n <= 1:
        return 1
    else:
        # 如果 n 大于 1，计算 n 的阶乘，使用递归调用自身来实现
        return n * factorial(n - 1)
```