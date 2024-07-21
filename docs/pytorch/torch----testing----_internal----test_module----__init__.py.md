# `.\pytorch\torch\testing\_internal\test_module\__init__.py`

```py
# 定义一个名为 find_factors 的函数，接受一个整数参数 num
def find_factors(num):
    # 创建一个空列表 factors 用于存储 num 的因子
    factors = []
    # 使用 for 循环遍历从 1 到 num 的所有整数
    for i in range(1, num + 1):
        # 如果 num 能被 i 整除，即 i 是 num 的因子
        if num % i == 0:
            # 将 i 添加到 factors 列表中
            factors.append(i)
    # 返回存储 num 所有因子的列表 factors
    return factors
```