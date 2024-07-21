# `.\pytorch\torch\utils\_exposed_in.py`

```py
# mypy: allow-untyped-defs
# 允许未经类型定义的函数声明
# 
# 定义一个装饰器函数 exposed_in，用于将函数暴露在指定模块中
def exposed_in(module):
    # 内部函数 wrapper，接受一个函数作为参数
    def wrapper(fn):
        # 将函数 fn 的 __module__ 属性设置为传入的 module 参数值
        fn.__module__ = module
        # 返回被装饰的函数 fn
        return fn

    # 返回内部函数 wrapper 作为装饰器
    return wrapper
```