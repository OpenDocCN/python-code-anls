# `.\pytorch\torch\_weights_only_unpickler.py`

```
# mypy: allow-untyped-defs
# 指定不强制类型定义的配置，用于类型检查工具mypy

# Unpickler restricted to loading only state dicts
# 限制Unpickler只能加载状态字典

# Restrict constructing types to a list defined in _get_allowed_globals()
# 限制类型构造仅限于_get_allowed_globals()定义的列表中的类型

# Restrict BUILD operation to `Tensor`, `Parameter` and `OrderedDict` types only
# 限制BUILD操作仅限于`Tensor`、`Parameter`和`OrderedDict`类型

# Restrict APPEND/APPENDS to `list`
# 限制APPEND/APPENDS操作仅限于`list`

# In `GLOBALS` operation do not do class lookup by name, but rather rely on dictionary
# defined by `_get_allowed_globals()` method, that contains:
# 在GLOBALS操作中，不要根据名称进行类查找，而是依赖于_get_allowed_globals()方法定义的字典，该字典包含以下内容：
# - torch types (Storage, dtypes, Tensor, `torch.Size`),
# - `torch._utils._rebuild` functions.
# - `torch.nn.Parameter`
# - `collections.Counter`
# - `collections.OrderedDict`

# Additionally, users can use an allowlist for adding classes they have deemed as safe using
# `_add_safe_globals()` (`torch.serialization.add_safe_globals`)
# `_clear_safe_globals()` (`torch.serialization.clear_safe_globals`)
# `_get_safe_globals()` (`torch.serialization.get_safe_globals`)

# Based of https://github.com/python/cpython/blob/main/Lib/pickle.py
# 基于 https://github.com/python/cpython/blob/main/Lib/pickle.py
# 预期用于加载PyTorch模型权重

# For example:
# 例如：
# data = urllib.request.urlopen('https://download.pytorch.org/models/resnet50-0676ba61.pth').read()
# buf = io.BytesIO(data)
# weights = torch.load(buf, weights_only = True)

import functools as _functools
import warnings
from collections import Counter, OrderedDict
from pickle import (
    APPEND,
    APPENDS,
    BINFLOAT,
    BINGET,
    BININT,
    BININT1,
    BININT2,
    BINPERSID,
    BINPUT,
    BINUNICODE,
    BUILD,
    bytes_types,
    decode_long,
    EMPTY_DICT,
    EMPTY_LIST,
    EMPTY_SET,
    EMPTY_TUPLE,
    GLOBAL,
    LONG1,
    LONG_BINGET,
    LONG_BINPUT,
    MARK,
    NEWFALSE,
    NEWOBJ,
    NEWTRUE,
    NONE,
    PROTO,
    REDUCE,
    SETITEM,
    SETITEMS,
    SHORT_BINSTRING,
    STOP,
    TUPLE,
    TUPLE1,
    TUPLE2,
    TUPLE3,
    UnpicklingError,
)
from struct import unpack
from sys import maxsize
from typing import Any, Dict, List

import torch
from torch._utils import IMPORT_MAPPING, NAME_MAPPING


_marked_safe_globals_list: List[Any] = []
# 初始化一个空列表，用于存储安全全局变量

def _add_safe_globals(safe_globals: List[Any]):
    global _marked_safe_globals_list
    _marked_safe_globals_list += safe_globals
    # 将指定的安全全局变量列表添加到全局变量列表中

def _get_safe_globals() -> List[Any]:
    global _marked_safe_globals_list
    return _marked_safe_globals_list
    # 返回当前已标记为安全的全局变量列表

def _clear_safe_globals():
    global _marked_safe_globals_list
    _marked_safe_globals_list = []
    # 清空已标记为安全的全局变量列表

# Separate from _get_allowed_globals because of the lru_cache on _get_allowed_globals
# 由于_get_allowed_globals上的lru_cache，此处与_get_allowed_globals分离

def _get_user_allowed_globals():
    rc: Dict[str, Any] = {}
    for f in _marked_safe_globals_list:
        module, name = f.__module__, f.__name__
        rc[f"{module}.{name}"] = f
    return rc
    # 返回用户允许的全局变量字典，根据标记为安全的全局变量列表生成

def _tensor_rebuild_functions():
    # Placeholder for defining tensor rebuild functions, if needed
    # 定义张量重建函数的占位符，如果需要的话
    # 返回一个包含多个函数名的集合，这些函数用于在 PyTorch 中重建不同类型的对象。
    return {
        torch._utils._rebuild_parameter,  # 重建参数对象
        torch._utils._rebuild_parameter_with_state,  # 带状态的参数对象重建
        torch._utils._rebuild_qtensor,  # 重建量化张量对象
        torch._utils._rebuild_tensor,  # 重建张量对象
        torch._utils._rebuild_tensor_v2,  # 第二版本的张量对象重建
        torch._utils._rebuild_tensor_v3,  # 第三版本的张量对象重建
        torch._utils._rebuild_sparse_tensor,  # 重建稀疏张量对象
        torch._utils._rebuild_meta_tensor_no_storage,  # 没有存储的元数据张量对象重建
        torch._utils._rebuild_nested_tensor,  # 重建嵌套张量对象
        torch._utils._rebuild_wrapper_subclass,  # 重建包装子类对象
    }
# Unpickling machinery
# 使用 functools 模块中的 lru_cache 装饰器，缓存函数结果，最多缓存一个结果
@_functools.lru_cache(maxsize=1)
def _get_allowed_globals():
    # 定义允许使用的全局变量字典 rc
    rc: Dict[str, Any] = {
        "collections.OrderedDict": OrderedDict,  # 添加 collections.OrderedDict 到 rc 中
        "collections.Counter": Counter,  # 添加 collections.Counter 到 rc 中
        "torch.nn.parameter.Parameter": torch.nn.Parameter,  # 添加 torch.nn.parameter.Parameter 到 rc 中
        "torch.serialization._get_layout": torch.serialization._get_layout,  # 添加 torch.serialization._get_layout 到 rc 中
        "torch.Size": torch.Size,  # 添加 torch.Size 到 rc 中
        "torch.Tensor": torch.Tensor,  # 添加 torch.Tensor 到 rc 中
        "torch.device": torch.device,  # 添加 torch.device 到 rc 中
    }
    
    # dtype
    # 遍历 torch.storage._dtype_to_storage_type_map() 的键，将其添加到 rc 中
    for t in torch.storage._dtype_to_storage_type_map().keys():
        rc[str(t)] = t
    
    # 遍历 torch.storage._new_dtypes() 的元素，将其添加到 rc 中
    for t in torch.storage._new_dtypes():
        rc[str(t)] = t
    
    # Tensor classes
    # 遍历 torch._tensor_classes 中的元素 tt，将其完整的类名作为键，类本身作为值添加到 rc 中
    for tt in torch._tensor_classes:
        rc[f"{tt.__module__}.{tt.__name__}"] = tt
    
    # Storage classes
    # 遍历 torch._storage_classes 中的元素 ts
    for ts in torch._storage_classes:
        # 如果 ts 不是 torch.storage.TypedStorage 或 torch.storage.UntypedStorage，则将其包装在一个虚拟类中，添加到 rc 中
        if ts not in (torch.storage.TypedStorage, torch.storage.UntypedStorage):
            rc[f"{ts.__module__}.{ts.__name__}"] = torch.serialization.StorageType(ts.__name__)
        else:
            rc[f"{ts.__module__}.{ts.__name__}"] = ts
    
    # Quantization specific
    # 将 torch.quantization 模块中的量化相关对象添加到 rc 中
    for qt in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
        torch.per_channel_affine,
        torch.per_channel_symmetric,
        torch.per_channel_affine_float_qparams,
    ]:
        rc[str(qt)] = qt
    
    # Rebuild functions
    # 遍历 _tensor_rebuild_functions() 返回的函数对象，将其添加到 rc 中
    for f in _tensor_rebuild_functions():
        rc[f"torch._utils.{f.__name__}"] = f

    # Handles Tensor Subclasses, Tensor's with attributes.
    # 将 torch._tensor._rebuild_from_type_v2 函数添加到 rc 中
    # 注意：此函数处理 Tensor 子类和带属性的 Tensor，还调用上述重建函数处理常规 Tensor 类型。
    rc["torch._tensor._rebuild_from_type_v2"] = torch._tensor._rebuild_from_type_v2
    
    # 返回允许使用的全局变量字典 rc
    return rc


class Unpickler:
    def __init__(self, file, *, encoding: str = "bytes"):
        # 初始化 Unpickler 类
        self.encoding = encoding  # 设置编码方式
        self.readline = file.readline  # 设置读取一行数据的方法
        self.read = file.read  # 设置读取所有数据的方法
        self.memo: Dict[int, Any] = {}  # 初始化一个空的备忘录字典
        self.proto: int = -1  # 初始化协议版本为 -1

    # Return a list of items pushed in the stack after last MARK instruction.
    # 返回最后一个 MARK 指令之后压入堆栈的项目列表
    def pop_mark(self):
        items = self.stack  # 将堆栈中的项目保存到 items 变量中
        self.stack = self.metastack.pop()  # 弹出元数据堆栈的栈顶元素，更新堆栈
        self.append = self.stack.append  # 更新 append 方法为当前堆栈的 append 方法
        return items  # 返回保存的项目列表

    # Raises an UnpicklingError indicating unsupported persistent id encountered.
    # 抛出 UnpicklingError 表示遇到了不支持的持久化 id
    def persistent_load(self, pid):
        raise UnpicklingError("unsupported persistent id encountered")


def load(file, *, encoding: str = "ASCII"):
    # 返回一个 Unpickler 实例化对象的 load 方法
    return Unpickler(file, encoding=encoding).load()
```