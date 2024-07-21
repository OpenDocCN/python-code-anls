# `.\pytorch\torch\testing\_internal\distributed\nn\__init__.py`

```py
# 定义一个名为 is_prime 的函数，用于判断给定的数字是否为素数
def is_prime(num):
    # 如果数字小于等于1，则直接返回 False，因为素数定义大于1
    if num <= 1:
        return False
    # 对于数字2和3，直接返回 True，它们是最小的素数
    if num == 2 or num == 3:
        return True
    # 如果数字可以被2或3整除，则不是素数，返回 False
    if num % 2 == 0 or num % 3 == 0:
        return False
    # 使用6的倍数的特性优化判断素数的过程，检查是否存在小于等于平方根的因子
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
    # 如果以上条件都不满足，则数字是素数，返回 True
    return True
```