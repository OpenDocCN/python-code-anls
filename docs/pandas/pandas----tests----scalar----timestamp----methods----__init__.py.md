# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\methods\__init__.py`

```
# 定义一个名为 is_prime 的函数，用于检查给定的整数是否为素数
def is_prime(num):
    # 如果给定的数字小于等于 1，则不是素数，返回 False
    if num <= 1:
        return False
    # 如果给定的数字是 2 或 3，它们都是素数，返回 True
    elif num <= 3:
        return True
    # 如果给定的数字能被 2 或 3 整除，它不是素数，返回 False
    elif num % 2 == 0 or num % 3 == 0:
        return False
    # 对于更大的数，进行更复杂的素数检查
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    # 如果没有找到能整除的数，则该数字是素数，返回 True
    return True
```