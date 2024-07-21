# `.\pytorch\test\package\package_d\__init__.py`

```py
# 定义一个名为 calculate_factors 的函数，接收一个整数参数 num
def calculate_factors(num):
    # 创建一个空列表 factors 用来存放 num 的因子
    factors = []
    # 从 1 到 num（包括 num）遍历每个数
    for i in range(1, num + 1):
        # 如果 num 能被 i 整除
        if num % i == 0:
            # 将 i 添加到 factors 列表中作为 num 的一个因子
            factors.append(i)
    # 返回 factors 列表，包含了 num 的所有因子
    return factors
```