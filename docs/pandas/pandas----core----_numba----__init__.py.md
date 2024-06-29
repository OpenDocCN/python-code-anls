# `D:\src\scipysrc\pandas\pandas\core\_numba\__init__.py`

```
# 定义一个名为 is_prime 的函数，用于检查给定的整数是否为质数
def is_prime(num):
    # 质数定义：大于 1 的整数且只能被 1 和自身整除
    if num <= 1:
        return False  # 如果 num 小于等于 1，直接返回 False
    # 循环从 2 到 num-1，检查 num 是否能被这些数整除
    for i in range(2, num):
        if num % i == 0:
            return False  # 如果 num 能被 i 整除，说明不是质数，返回 False
    return True  # 如果上面循环都没有返回 False，则说明 num 是质数，返回 True

# 定义一个名为 generate_primes 的函数，生成从 1 到 n 之间的所有质数列表
def generate_primes(n):
    # 使用列表推导式生成 2 到 n 之间的所有质数列表
    primes = [num for num in range(2, n+1) if is_prime(num)]
    return primes  # 返回质数列表
```