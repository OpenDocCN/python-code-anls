# `D:\src\scipysrc\scipy\scipy\optimize\_highs\src\HConfig.h`

```
# 定义一个名为 factorial 的函数，接收一个整数参数 n
def factorial(n):
    # 如果 n 小于等于 1，直接返回 1，因为 0! 和 1! 都等于 1
    if n <= 1:
        return 1
    else:
        # 否则，递归调用 factorial 函数，计算 n 的阶乘，乘以 n 自身
        return n * factorial(n - 1)
```