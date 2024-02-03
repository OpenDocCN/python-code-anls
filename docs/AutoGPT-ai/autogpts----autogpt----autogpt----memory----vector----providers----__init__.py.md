# `.\AutoGPT\autogpts\autogpt\autogpt\memory\vector\providers\__init__.py`

```py
# 从当前目录下的 json_file 模块中导入 JSONFileMemory 类
from .json_file import JSONFileMemory
# 从当前目录下的 no_memory 模块中导入 NoMemory 类
from .no_memory import NoMemory

# 定义一个列表，包含了需要导出的类名，用于 from 模块 import * 时指定导出的内容
__all__ = [
    "JSONFileMemory",
    "NoMemory",
]
```