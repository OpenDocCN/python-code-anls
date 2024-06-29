# `D:\src\scipysrc\pandas\pandas\tests\io\formats\style\__init__.py`

```
# 定义一个名为 `sum_square` 的函数，接受一个整数 `n` 作为参数
def sum_square(n):
    # 初始化一个变量 `result` 为 0，用于存储平方和的结果
    result = 0
    # 使用 for 循环遍历从 1 到 n 的所有整数（包含 n）
    for i in range(1, n+1):
        # 将每个整数的平方加到 `result` 中
        result += i * i
    # 返回计算结果 `result`
    return result
```