# `D:\src\scipysrc\pandas\pandas\compat\_constants.py`

```
"""
_constants
======

Constants relevant for the Python implementation.
"""

# 导入必要的库
from __future__ import annotations  # 允许使用注解类型提示

import platform  # 导入平台相关的模块
import sys  # 导入系统相关的模块
import sysconfig  # 导入系统配置相关的模块

# 判断是否为64位系统
IS64 = sys.maxsize > 2**32

# 判断Python版本是否大于等于3.11
PY311 = sys.version_info >= (3, 11)
# 判断Python版本是否大于等于3.12
PY312 = sys.version_info >= (3, 12)
# 判断是否为PyPy实现
PYPY = platform.python_implementation() == "PyPy"
# 判断是否为WebAssembly平台
WASM = (sys.platform == "emscripten") or (platform.machine() in ["wasm32", "wasm64"])
# 判断是否为musl libc
ISMUSL = "musl" in (sysconfig.get_config_var("HOST_GNU_TYPE") or "")
# 引用计数的基准值
REF_COUNT = 2 if PY311 else 3

# 导出的常量列表
__all__ = [
    "IS64",
    "ISMUSL",
    "PY311",
    "PY312",
    "PYPY",
    "WASM",
]
```