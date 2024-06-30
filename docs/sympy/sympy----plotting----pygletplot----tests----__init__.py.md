# `D:\src\scipysrc\sympy\sympy\plotting\pygletplot\tests\__init__.py`

```
# 定义一个名为 `flatten` 的函数，用于将嵌套列表压平成一维列表
def flatten(lst):
    # 利用列表推导式和递归调用自身，处理嵌套列表，返回一维列表
    return [item for sublist in lst for item in (flatten(sublist) if isinstance(sublist, list) else [sublist])]
```