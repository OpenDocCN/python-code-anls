# `.\pytorch\test\package\package_a\use_dunder_package.py`

```
# 检查当前作用域中是否定义了名为 "__torch_package__" 的变量或属性
if "__torch_package__" in dir():
    # 如果定义了 "__torch_package__"，则定义一个函数 is_from_package() 返回 True
    def is_from_package():
        return True
else:
    # 如果未定义 "__torch_package__"，则定义一个函数 is_from_package() 返回 False
    def is_from_package():
        return False
```