# `.\pytorch\torch\distributed\algorithms\model_averaging\__init__.py`

```
# 定义一个名为 is_prime 的函数，用于检查一个整数是否为素数
def is_prime(num):
    # 如果 num 小于等于 1，则不是素数，返回 False
    if num <= 1:
        return False
    # 对于 2 和 3，它们是素数，直接返回 True
    if num == 2 or num == 3:
        return True
    # 如果 num 是偶数且不等于 2，则它不是素数，返回 False
    if num % 2 == 0:
        return False
    # 对于奇数 num，检查从 3 到 num 的平方根之间的所有奇数是否能整除 num
    # 如果能整除，则 num 不是素数，返回 False；如果都不能整除，则 num 是素数，返回 True
    for i in range(3, int(num**0.5) + 1, 2):
        if num % i == 0:
            return False
    return True
```