# `D:\src\scipysrc\pandas\pandas\core\arrays\arrow\extension_types.py`

```
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow

from pandas.compat import pa_version_under14p1

from pandas.core.dtypes.dtypes import (
    IntervalDtype,
    PeriodDtype,
)

from pandas.core.arrays.interval import VALID_CLOSED

if TYPE_CHECKING:
    from pandas._typing import IntervalClosedType


class ArrowPeriodType(pyarrow.ExtensionType):
    def __init__(self, freq) -> None:
        # 设置属性在调用 super init 之前是必须的
        # 因为 super init 调用了 serialize 方法
        self._freq = freq
        # 调用父类的初始化方法来设置扩展类型的基本信息
        pyarrow.ExtensionType.__init__(self, pyarrow.int64(), "pandas.period")

    @property
    def freq(self):
        # 返回频率属性
        return self._freq

    def __arrow_ext_serialize__(self) -> bytes:
        # 序列化为字节流的方法，将频率信息转换为 JSON 格式的元数据
        metadata = {"freq": self.freq}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowPeriodType:
        # 反序列化方法，从序列化的字节流中恢复对象
        metadata = json.loads(serialized.decode())
        return ArrowPeriodType(metadata["freq"])

    def __eq__(self, other):
        # 定义相等性比较操作符，用于扩展类型对象的比较
        if isinstance(other, pyarrow.BaseExtensionType):
            return type(self) == type(other) and self.freq == other.freq
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        # 定义不等操作符，与相等操作符相对应
        return not self == other

    def __hash__(self) -> int:
        # 定义哈希方法，用于扩展类型对象的哈希化
        return hash((str(self), self.freq))

    def to_pandas_dtype(self) -> PeriodDtype:
        # 将扩展类型转换为 Pandas 的 PeriodDtype 类型
        return PeriodDtype(freq=self.freq)


# register the type with a dummy instance
# 使用一个虚拟实例注册扩展类型
_period_type = ArrowPeriodType("D")
pyarrow.register_extension_type(_period_type)


class ArrowIntervalType(pyarrow.ExtensionType):
    def __init__(self, subtype, closed: IntervalClosedType) -> None:
        # 设置属性在调用 super init 之前是必须的
        # 因为 super init 调用了 serialize 方法
        assert closed in VALID_CLOSED
        self._closed: IntervalClosedType = closed
        if not isinstance(subtype, pyarrow.DataType):
            subtype = pyarrow.type_for_alias(str(subtype))
        self._subtype = subtype

        # 构建存储类型为 struct 的 PyArrow 类型
        storage_type = pyarrow.struct([("left", subtype), ("right", subtype)])
        pyarrow.ExtensionType.__init__(self, storage_type, "pandas.interval")

    @property
    def subtype(self):
        # 返回子类型属性
        return self._subtype

    @property
    def closed(self) -> IntervalClosedType:
        # 返回闭合类型属性
        return self._closed

    def __arrow_ext_serialize__(self) -> bytes:
        # 序列化为字节流的方法，将子类型和闭合类型信息转换为 JSON 格式的元数据
        metadata = {"subtype": str(self.subtype), "closed": self.closed}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowIntervalType:
        # 反序列化方法，从序列化的字节流中恢复对象
        metadata = json.loads(serialized.decode())
        subtype = pyarrow.type_for_alias(metadata["subtype"])
        closed = metadata["closed"]
        return ArrowIntervalType(subtype, closed)
    # 比较当前对象和另一个对象是否相等
    def __eq__(self, other):
        # 检查另一个对象是否是 pyarrow.BaseExtensionType 的实例
        if isinstance(other, pyarrow.BaseExtensionType):
            # 返回比较结果，包括类型、子类型和是否关闭状态
            return (
                type(self) == type(other)
                and self.subtype == other.subtype
                and self.closed == other.closed
            )
        else:
            # 如果不是同类型对象，则返回 NotImplemented 表示不支持该操作
            return NotImplemented

    # 实现不等于操作符
    def __ne__(self, other) -> bool:
        # 返回当前对象和另一个对象的相等性的相反结果
        return not self == other

    # 实现哈希函数
    def __hash__(self) -> int:
        # 返回基于对象自身、子类型和关闭状态的哈希值
        return hash((str(self), str(self.subtype), self.closed))

    # 将对象转换为 Pandas 的 IntervalDtype 类型
    def to_pandas_dtype(self) -> IntervalDtype:
        # 使用子类型和关闭状态创建 IntervalDtype 对象并返回
        return IntervalDtype(self.subtype.to_pandas_dtype(), self.closed)
# 创建一个 ArrowIntervalType 实例，作为自定义的扩展类型
_interval_type = ArrowIntervalType(pyarrow.int64(), "left")
# 使用 pyarrow.register_extension_type 注册这个自定义类型
pyarrow.register_extension_type(_interval_type)

# 定义错误信息字符串，用于提示不允许反序列化 'arrow.py_extension_type' 的操作
_ERROR_MSG = """\
Disallowed deserialization of 'arrow.py_extension_type':
storage_type = {storage_type}
serialized = {serialized}
pickle disassembly:\n{pickle_disassembly}

Reading of untrusted Parquet or Feather files with a PyExtensionType column
allows arbitrary code execution.
If you trust this file, you can enable reading the extension type by one of:

- upgrading to pyarrow >= 14.0.1, and call `pa.PyExtensionType.set_auto_load(True)`
- install pyarrow-hotfix (`pip install pyarrow-hotfix`) and disable it by running
  `import pyarrow_hotfix; pyarrow_hotfix.uninstall()`

We strongly recommend updating your Parquet/Feather files to use extension types
derived from `pyarrow.ExtensionType` instead, and register this type explicitly.
"""

# 定义函数 patch_pyarrow，用于修补 pyarrow 库的行为
def patch_pyarrow() -> None:
    # 如果 pyarrow 版本不低于 14.0.1，则直接返回，因为从这个版本开始有了自己的机制
    if not pa_version_under14p1:
        return

    # 如果安装并启用了 pyarrow-hotfix（https://github.com/pitrou/pyarrow-hotfix）
    if getattr(pyarrow, "_hotfix_installed", False):
        return

    # 定义一个 ForbiddenExtensionType 类，继承自 pyarrow.ExtensionType
    class ForbiddenExtensionType(pyarrow.ExtensionType):
        # 定义一个空的序列化方法，用于禁止序列化
        def __arrow_ext_serialize__(self) -> bytes:
            return b""

        # 定义反序列化方法，抛出运行时异常，包含详细错误信息
        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            import io
            import pickletools

            out = io.StringIO()
            # 使用 pickletools 分析反序列化的序列化数据，将分析结果写入 out
            pickletools.dis(serialized, out)
            # 抛出运行时异常，包含详细错误信息，使用预定义的 _ERROR_MSG 格式化字符串
            raise RuntimeError(
                _ERROR_MSG.format(
                    storage_type=storage_type,
                    serialized=serialized,
                    pickle_disassembly=out.getvalue(),
                )
            )

    # 注销 'arrow.py_extension_type' 扩展类型
    pyarrow.unregister_extension_type("arrow.py_extension_type")
    # 使用 ForbiddenExtensionType 类注册 'arrow.py_extension_type' 扩展类型
    pyarrow.register_extension_type(
        ForbiddenExtensionType(pyarrow.null(), "arrow.py_extension_type")
    )

    # 设置标志 _hotfix_installed 为 True，表示已经安装了 hotfix
    pyarrow._hotfix_installed = True

# 调用 patch_pyarrow 函数，执行 pyarrow 库的修补操作
patch_pyarrow()
```