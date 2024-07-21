# `.\pytorch\torch\ao\__init__.py`

```
# mypy: allow-untyped-defs
# torch.ao is a package with a lot of interdependencies.
# We will use lazy import to avoid cyclic dependencies here.

# 定义模块中公开的所有属性列表
__all__ = [
    "nn",             # 神经网络模块
    "ns",             # 命名空间模块
    "quantization",   # 量化模块
    "pruning",        # 剪枝模块
]

# 定义一个特殊的函数 __getattr__，用于按需动态加载模块
def __getattr__(name):
    # 如果请求的模块名在 __all__ 列表中
    if name in __all__:
        # 动态导入模块并返回
        import importlib
        return importlib.import_module("." + name, __name__)
    # 如果请求的模块名不在 __all__ 列表中，抛出 AttributeError 异常
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```