# `.\DB-GPT-src\dbgpt\util\benchmarks\llm\__init__.py`

```py
# 定义一个名为 calculate_average 的函数，接收一个参数 numbers
def calculate_average(numbers):
    # 判断列表 numbers 是否为空
    if not numbers:
        # 如果为空，返回 None
        return None
    
    # 计算列表 numbers 中所有元素的和
    total = sum(numbers)
    # 计算列表 numbers 中元素的个数
    count = len(numbers)
    # 返回计算出的平均值
    return total / count
```