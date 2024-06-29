# `D:\src\scipysrc\pandas\pandas\compat\pickle_compat.py`

```
"""
Pickle compatibility to pandas version 1.0
"""

# 导入未来的注解功能，确保与未来版本的兼容性
from __future__ import annotations

# 导入上下文管理器、IO操作和pickle模块
import contextlib
import io
import pickle

# 导入类型提示
from typing import (
    TYPE_CHECKING,
    Any,
)

# 导入NumPy库
import numpy as np

# 导入Pandas的数组和时间序列相关模块
from pandas._libs.arrays import NDArrayBacked
from pandas._libs.tslibs import BaseOffset

# 导入Pandas核心数组和内部管理模块
from pandas.core.arrays import (
    DatetimeArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.internals import BlockManager

# 如果在类型检查模式下，则导入生成器类
if TYPE_CHECKING:
    from collections.abc import Generator


# 如果类被移动，提供兼容性映射
_class_locations_map = {
    # 重定向反序列化块逻辑到 _unpickle_block，适用于 pandas <= 1.3.5
    ("pandas.core.internals.blocks", "new_block"): (
        "pandas._libs.internals",
        "_unpickle_block",
    ),
    # 避免Cython的警告“与Python的'class private name'规则相矛盾”
    ("pandas._libs.tslibs.nattype", "__nat_unpickle"): (
        "pandas._libs.tslibs.nattype",
        "_nat_unpickle",
    ),
    # 移除 Int64Index、UInt64Index 和 Float64Index 从代码库中
    ("pandas.core.indexes.numeric", "Int64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.indexes.numeric", "UInt64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    ("pandas.core.indexes.numeric", "Float64Index"): (
        "pandas.core.indexes.base",
        "Index",
    ),
    # 密集型数据类型 SparseDtype 的兼容处理
    ("pandas.core.arrays.sparse.dtype", "SparseDtype"): (
        "pandas.core.dtypes.dtypes",
        "SparseDtype",
    ),
}


# 自定义的Unpickler子类，重写方法和一些调度函数以实现兼容性，并使用pickle模块的非公开类
class Unpickler(pickle._Unpickler):
    # 重写find_class方法，查找类的位置映射，以确保兼容性
    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        module, name = _class_locations_map.get(key, key)
        return super().find_class(module, name)

    # 复制pickle模块的调度表
    dispatch = pickle._Unpickler.dispatch.copy()

    # 重写load_reduce方法，处理特定的反序列化情况以确保兼容性
    def load_reduce(self) -> None:
        stack = self.stack  # type: ignore[attr-defined]
        args = stack.pop()
        func = stack[-1]

        try:
            stack[-1] = func(*args)
        except TypeError:
            # 处理已弃用的函数的情况，尝试替换并重试
            if args and isinstance(args[0], type) and issubclass(args[0], BaseOffset):
                # 如果是BaseOffset子类的对象，尝试使用特定的类方法
                cls = args[0]
                stack[-1] = cls.__new__(*args)
                return
            elif args and issubclass(args[0], PeriodArray):
                # 如果是PeriodArray的子类对象，使用NDArrayBacked类的构造函数
                cls = args[0]
                stack[-1] = NDArrayBacked.__new__(*args)
                return
            raise

    # 重写pickle调度表中的REDUCE条目，指向自定义的load_reduce方法
    dispatch[pickle.REDUCE[0]] = load_reduce  # type: ignore[assignment]
    # 定义一个方法用于加载新对象，不返回任何结果
    def load_newobj(self) -> None:
        # 从栈中弹出参数（类型注释忽略），作为构造新对象的参数
        args = self.stack.pop()  # type: ignore[attr-defined]
        # 从栈中弹出类（类型注释忽略），表示要实例化的对象的类

        cls = self.stack.pop()  # type: ignore[attr-defined]

        # 兼容处理
        # 如果类是 DatetimeArray 的子类且没有参数
        if issubclass(cls, DatetimeArray) and not args:
            # 创建一个空的 np.array，dtype 为 "M8[ns]"，即 datetime64[ns] 类型的数组
            arr = np.array([], dtype="M8[ns]")
            # 使用类的 __new__ 方法创建新对象 obj，传入 arr 和其 dtype
            obj = cls.__new__(cls, arr, arr.dtype)
        # 如果类是 TimedeltaArray 的子类且没有参数
        elif issubclass(cls, TimedeltaArray) and not args:
            # 创建一个空的 np.array，dtype 为 "m8[ns]"，即 timedelta64[ns] 类型的数组
            arr = np.array([], dtype="m8[ns]")
            # 使用类的 __new__ 方法创建新对象 obj，传入 arr 和其 dtype
            obj = cls.__new__(cls, arr, arr.dtype)
        # 如果类是 BlockManager 且没有参数
        elif cls is BlockManager and not args:
            # 使用类的 __new__ 方法创建新对象 obj，传入空元组、空列表和 False
            obj = cls.__new__(cls, (), [], False)
        else:
            # 否则，使用类的 __new__ 方法创建新对象 obj，传入 args 中的参数
            obj = cls.__new__(cls, *args)
        
        # 将创建的对象 obj 添加到当前对象的末尾
        self.append(obj)  # type: ignore[attr-defined]

    # 将 load_newobj 方法注册到 pickle.NEWOBJ[0] 对应的分发表中
    dispatch[pickle.NEWOBJ[0]] = load_newobj  # type: ignore[assignment]
# 将字节对象加载为 Python 对象，类似于 pickle._loads 函数的功能
def loads(
    bytes_object: bytes,
    *,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
) -> Any:
    """
    Analogous to pickle._loads.
    """
    # 使用 io.BytesIO 封装字节对象，创建一个字节流对象
    fd = io.BytesIO(bytes_object)
    # 使用自定义的 Unpickler 对象从字节流中加载数据
    return Unpickler(
        fd, fix_imports=fix_imports, encoding=encoding, errors=errors
    ).load()


@contextlib.contextmanager
def patch_pickle() -> Generator[None, None, None]:
    """
    Temporarily patch pickle to use our unpickler.
    """
    # 保存原始的 pickle.loads 函数引用
    orig_loads = pickle.loads
    try:
        # 将 pickle.loads 替换为自定义的 loads 函数
        setattr(pickle, "loads", loads)
        # 执行 yield，使得此函数可以作为上下文管理器使用
        yield
    finally:
        # 恢复原始的 pickle.loads 函数
        setattr(pickle, "loads", orig_loads)
```