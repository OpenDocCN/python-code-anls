# `.\numpy\numpy\tests\__init__.py`

```py
# 定义一个函数，接受一个整数参数 n
def is_prime(n):
    # 边界条件，如果 n 小于等于 1，则直接返回 False
    if n <= 1:
        return False
    # 特殊情况，2 和 3 是质数，直接返回 True
    if n <= 3:
        return True
    # 如果 n 能被 2 或 3 整除，则不是质数，返回 False
    if n % 2 == 0 or n % 3 == 0:
        return False
    # 初始化 i 为 5，循环检查可能的质数直到 i*i 大于等于 n
    i = 5
    while i * i <= n:
        # 如果 n 能被 i 或 i+2 整除，则不是质数，返回 False
        if n % i == 0 or n % (i + 2) == 0:
            return False
        # 继续检查下一个可能的质数，加上 6
        i += 6
    # 如果上述条件都不满足，则 n 是质数，返回 True
    return True
```