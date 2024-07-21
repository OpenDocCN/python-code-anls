# `.\pytorch\torch\_C\_dynamo\eval_frame.pyi`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和类型定义
import types  # 导入 Python 的 types 模块
from typing import NewType  # 导入 NewType 类型定义

from torch._dynamo.types import DynamoCallback, DynamoGuardHook  # 导入 DynamoCallback 和 DynamoGuardHook 类型定义

# 我们为 Python >= 3.11 实现了自己的 FrameType 类型，因此它实际上不是 FrameType 的别名，但仍然暴露相同的接口。
_PyInterpreterFrame = NewType("_PyInterpreterFrame", types.FrameType)

# 定义函数签名，用 ... 表示未提供具体实现
def set_eval_frame(callback: DynamoCallback) -> DynamoCallback: ...
def reset_code(code: types.CodeType) -> None: ...
def unsupported(obj1: object, obj2: object) -> object: ...
def skip_code(code: types.CodeType) -> None: ...
def set_guard_error_hook(hook: DynamoGuardHook) -> None: ...

# 定义一个私有类 _CacheEntry
class _CacheEntry:
    # 定义检查函数 check_fn，使用 ... 表示未提供具体实现
    def check_fn(self, *args, **kwargs): ...
    code: types.CodeType  # 定义属性 code，类型为 types.CodeType
    next: _CacheEntry | None  # 定义属性 next，可以是 _CacheEntry 类型或 None

# 定义一个私有类 _ExtraState
class _ExtraState:
    # 定义方法 invalidate，用于使缓存条目无效
    def invalidate(self, cache_entry: _CacheEntry): ...

# 定义函数 _debug_get_cache_entry_list，返回一个类型为 _CacheEntry 的列表
def _debug_get_cache_entry_list(code: types.CodeType) -> list[_CacheEntry]: ...

# 定义变量 py_opcode_caches，类型为 int 列表
py_opcode_caches: list[int]
```