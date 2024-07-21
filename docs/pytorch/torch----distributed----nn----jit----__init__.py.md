# `.\pytorch\torch\distributed\nn\jit\__init__.py`

```py
# 定义一个名为 is_prime 的函数，用来判断一个整数是否为素数
def is_prime(num):
    # 如果数字小于等于1，直接返回 False，因为素数必须大于1
    if num <= 1:
        return False
    # 如果数字是2或者3，它们是素数，直接返回 True
    if num == 2 or num == 3:
        return True
    # 如果数字是偶数并且不是2，那它不是素数，直接返回 False
    if num % 2 == 0:
        return False
    # 计算从3到数字平方根的范围，步长为2，来检查是否有其他因子
    for i in range(3, int(num**0.5) + 1, 2):
        if num % i == 0:
            return False
    # 如果都不满足上述条件，那么数字是素数，返回 True
    return True
```