# `.\DB-GPT-src\dbgpt\model\cluster\controller\__init__.py`

```py
# 定义一个名为 is_prime 的函数，用于检查一个整数是否为素数
def is_prime(num):
    # 如果数字小于等于 1，则直接返回 False，因为素数定义大于 1
    if num <= 1:
        return False
    # 如果数字为 2 或 3，则直接返回 True，因为它们是素数
    if num == 2 or num == 3:
        return True
    # 如果数字能被 2 整除，直接返回 False，因为除了 2 以外，偶数不可能是素数
    if num % 2 == 0:
        return False
    # 循环检查从 3 到 num 的平方根范围内的奇数是否能整除 num，如果能，返回 False
    for i in range(3, int(num**0.5) + 1, 2):
        if num % i == 0:
            return False
    # 如果以上条件都不满足，则认定 num 是素数，返回 True
    return True
```