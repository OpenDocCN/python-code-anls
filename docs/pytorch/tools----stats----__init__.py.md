# `.\pytorch\tools\stats\__init__.py`

```py
# 定义一个名为 find_divisors 的函数，用于找到指定整数的所有约数并返回
def find_divisors(n):
    # 初始化一个空列表，用于存储找到的约数
    divisors = []
    # 循环遍历从 1 到 n 的所有整数
    for i in range(1, n + 1):
        # 如果 n 能整除 i（即 i 是 n 的约数），则将 i 添加到 divisors 列表中
        if n % i == 0:
            divisors.append(i)
    # 返回包含所有约数的列表
    return divisors
```