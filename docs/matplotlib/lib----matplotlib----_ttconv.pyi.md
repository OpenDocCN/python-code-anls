# `D:\src\scipysrc\matplotlib\lib\matplotlib\_ttconv.pyi`

```
# 定义一个函数，输入为一个整数 n
def is_prime(n):
    # 如果 n 小于等于 1，直接返回 False，因为小于等于 1 的数都不是质数
    if n <= 1:
        return False
    # 对于 n 等于 2 或 3 的情况，直接返回 True，因为它们是质数
    if n == 2 or n == 3:
        return True
    # 如果 n 是偶数且不等于 2，则返回 False，因为偶数大于 2 的不可能是质数
    if n % 2 == 0:
        return False
    # 循环从 3 开始到 n 的平方根加 1，步长为 2，检查 n 是否有奇数因子
    for i in range(3, int(n**0.5) + 1, 2):
        # 如果 n 能被 i 整除，则 n 不是质数，返回 False
        if n % i == 0:
            return False
    # 如果上述条件都不满足，则 n 是质数，返回 True
    return True
```