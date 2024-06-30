# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\tests\__init__.py`

```
# 定义一个名为 is_prime 的函数，用于检查一个数是否为素数
def is_prime(num):
    # 如果输入的数小于等于1，则不是素数，直接返回 False
    if num <= 1:
        return False
    # 对于数字2和3，它们是素数，直接返回 True
    if num == 2 or num == 3:
        return True
    # 如果输入的数能被2整除，那么它不是素数，返回 False
    if num % 2 == 0:
        return False
    # 对于大于3的奇数，检查从3到平方根(num) + 1的所有奇数，看是否能整除num
    for i in range(3, int(num**0.5) + 1, 2):
        if num % i == 0:
            return False
    # 如果都不能整除，则num是素数，返回 True
    return True
```