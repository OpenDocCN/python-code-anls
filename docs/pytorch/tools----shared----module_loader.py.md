# `.\pytorch\tools\shared\module_loader.py`

```
# 从 importlib.abc 模块中导入 Loader 类
from importlib.abc import Loader
# 从 types 模块中导入 ModuleType 类
from types import ModuleType
# 从 typing 模块中导入 cast 函数，用于类型转换
from typing import cast

# 定义一个函数 import_module，用于动态导入指定路径的模块并返回该模块对象
def import_module(name: str, path: str) -> ModuleType:
    # 导入 importlib.util 模块
    import importlib.util
    # 根据模块名和文件路径创建一个模块规范
    spec = importlib.util.spec_from_file_location(name, path)
    # 确保模块规范不为 None
    assert spec is not None
    # 根据模块规范创建一个模块对象
    module = importlib.util.module_from_spec(spec)
    # 强制类型转换 spec.loader 为 Loader 类型，并执行模块对象的加载
    cast(Loader, spec.loader).exec_module(module)
    # 返回加载后的模块对象
    return module
```