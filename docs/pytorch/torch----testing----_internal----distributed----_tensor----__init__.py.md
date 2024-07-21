# `.\pytorch\torch\testing\_internal\distributed\_tensor\__init__.py`

```py
# 定义一个名为 `calculate_sum` 的函数，接收一个参数 `numbers`
def calculate_sum(numbers):
    # 初始化变量 `sum` 为 0，用于存储计算后的总和
    sum = 0
    # 使用 `for` 循环遍历参数 `numbers` 中的每一个元素
    for num in numbers:
        # 将当前元素 `num` 加到 `sum` 中
        sum += num
    # 返回累加结果 `sum`
    return sum
```