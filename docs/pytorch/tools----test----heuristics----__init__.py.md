# `.\pytorch\tools\test\heuristics\__init__.py`

```
# 定义一个函数，名称为 `calculate_factors`，接收一个整数 `n` 作为参数
def calculate_factors(n):
    # 初始化一个空列表，用于存放因子
    factors = []
    # 循环从 1 到 n+1（不包括 n+1），逐个检查每个数是否是 n 的因子
    for i in range(1, n+1):
        # 如果 n 能被 i 整除（即 i 是 n 的因子），则将 i 添加到 factors 列表中
        if n % i == 0:
            factors.append(i)
    # 返回存放因子的列表
    return factors
```