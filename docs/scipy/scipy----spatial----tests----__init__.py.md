# `D:\src\scipysrc\scipy\scipy\spatial\tests\__init__.py`

```
# 定义一个函数，接收一个整数参数 n
def is_prime(n):
    # 如果 n 小于等于 1，则直接返回 False，因为 1 和小于 1 的数都不是质数
    if n <= 1:
        return False
    # 如果 n 等于 2，则是质数，直接返回 True
    if n == 2:
        return True
    # 如果 n 是偶数（除了 2 外的偶数），则直接返回 False，因为偶数不可能是质数
    if n % 2 == 0:
        return False
    # 循环从 3 开始，到 int(n**0.5) 结束，步长为 2
    # （因为偶数已经在上一个判断中排除，所以可以直接从 3 开始，且只需判断到 n 的平方根即可）
    for i in range(3, int(n**0.5) + 1, 2):
        # 如果 n 能被 i 整除，则 n 不是质数，返回 False
        if n % i == 0:
            return False
    # 若以上所有条件都不符合，则 n 是质数，返回 True
    return True
```