# `.\pytorch\test\package\package_a\long_name.py`

```py
# 定义一个函数，用于向给定列表 `d` 中添加一个元素
# noqa: B950 是用来告诉 linter 忽略对此行代码长度的检查

def add_function(d):
    d.append(
        # 调用具有长名称的函数，该函数返回整数 1337
        function_with_a_long_name_256charsplus_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    )


# 定义一个名称非常长的函数，它返回整数 1337
# noqa: B950 是用来告诉 linter 忽略对此行代码长度的检查
def function_with_a_long_name_256charsplus_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx():
    return 1337
```