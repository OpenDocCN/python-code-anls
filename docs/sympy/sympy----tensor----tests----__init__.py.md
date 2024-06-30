# `D:\src\scipysrc\sympy\sympy\tensor\tests\__init__.py`

```
# 定义一个名为 reverse_dict 的函数，接收一个字典作为参数
def reverse_dict(d):
    # 使用字典推导式创建一个新的字典，将原始字典的键值对反转
    return {v: k for k, v in d.items()}
```