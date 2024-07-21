# `.\pytorch\torch\ao\nn\__init__.py`

```
# 声明一个类型检查器的特殊注解，允许未标记类型的函数定义
# 我们向最终用户暴露所有的子包。
# 由于可能存在互相依赖的情况，我们希望避免循环导入，
# 因此按照 https://peps.python.org/pep-0562/ 实现延迟加载的版本。

# 导入内置模块 importlib，用于动态导入其他模块

# 定义了一个列表 __all__，包含需要对外暴露的子模块名称
__all__ = [
    "intrinsic",
    "qat",
    "quantizable",
    "quantized",
    "sparse",
]

# 定义了一个特殊的函数 __getattr__，用于在当前模块中动态获取属性
def __getattr__(name):
    # 如果请求获取的属性名在 __all__ 列表中
    if name in __all__:
        # 则使用 importlib 动态导入该属性对应的子模块，并返回
        return importlib.import_module("." + name, __name__)
    # 如果请求的属性名不在 __all__ 列表中，则抛出 AttributeError 异常
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```