# `.\pytorch\torch\_inductor\fx_passes\__init__.py`

```
# 定义一个名为 is_prime 的函数，接收一个整数参数 n
def is_prime(n):
    # 如果 n 小于等于 1，直接返回 False
    if n <= 1:
        return False
    # 对于 2 和 3，直接返回 True，它们是质数
    if n <= 3:
        return True
    # 如果 n 能被 2 或 3 整除，返回 False
    if n % 2 == 0 or n % 3 == 0:
        return False
    # 初始化 i 为 5，循环检查 i*i 是否小于等于 n
    i = 5
    while i * i <= n:
        # 如果 n 能被 i 或 i+2 整除，返回 False（质数不可能被比自身更大的数整除）
        if n % i == 0 or n % (i + 2) == 0:
            return False
        # 递增 i，每次增加 6，因为质数一定是 6 的倍数加或减 1
        i += 6
    # 如果以上条件都不符合，则 n 是质数，返回 True
    return True
```