# `.\pytorch\torch\testing\_internal\__init__.py`

```
# 定义一个函数，接收一个整数作为参数
def is_prime(num):
    # 如果输入的数字小于等于1，则不是质数，直接返回 False
    if num <= 1:
        return False
    # 对于输入的数字，从 2 开始到其平方根（向上取整），检查是否有能整除它的数字
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            # 如果能整除，说明不是质数，返回 False
            return False
    # 如果上述条件都不满足，则说明是质数，返回 True
    return True
```