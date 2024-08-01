# `.\DB-GPT-src\dbgpt\util\dbgpts\__init__.py`

```py
# 定义一个名为`average`的函数，接受一个参数`numbers`，这个参数是一个列表
def average(numbers):
    # 检查`numbers`是否为空列表，如果是，则返回None
    if not numbers:
        return None
    # 使用内置函数`sum()`计算`numbers`列表中所有元素的总和，并将结果除以`len(numbers)`得到平均值
    avg = sum(numbers) / len(numbers)
    # 返回计算得到的平均值
    return avg
```