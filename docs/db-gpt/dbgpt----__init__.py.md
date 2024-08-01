# `.\DB-GPT-src\dbgpt\__init__.py`

```py
"""DB-GPT: Next Generation Data Interaction Solution with LLMs.
"""
# 引入版本号信息，忽略导入时的错误码 E402
from dbgpt import _version  # noqa: E402
# 从 dbgpt.component 模块导入 BaseComponent 和 SystemApp，忽略未使用的名字 F401
from dbgpt.component import BaseComponent, SystemApp  # noqa: F401

# 核心库列表
_CORE_LIBS = ["core", "rag", "model", "agent", "datasource", "vis", "storage", "train"]
# 服务库列表
_SERVE_LIBS = ["serve"]
# 所有库的集合
_LIBS = _CORE_LIBS + _SERVE_LIBS

# 设置模块的版本号
__version__ = _version.version

# 模块中可以导出的所有公共接口
__ALL__ = ["__version__", "SystemApp", "BaseComponent"]


# 当访问不存在的属性时的处理函数
def __getattr__(name: str):
    # 惰性加载模块
    import importlib

    # 如果请求的模块名在 _LIBS 中
    if name in _LIBS:
        # 动态导入该模块
        return importlib.import_module("." + name, __name__)
    # 如果请求的模块名不在 _LIBS 中，抛出属性错误异常
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```