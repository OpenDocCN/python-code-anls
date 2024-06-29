# `D:\src\scipysrc\pandas\pandas\core\interchange\buffer.py`

```
from __future__ import annotations
# 导入未来版本的类型注解特性

from typing import (
    TYPE_CHECKING,
    Any,
)
# 导入类型检查相关的模块和类型

from pandas.core.interchange.dataframe_protocol import (
    Buffer,
    DlpackDeviceType,
)
# 从 pandas 库中导入数据交换协议相关的类和类型

if TYPE_CHECKING:
    import numpy as np
    import pyarrow as pa
# 如果处于类型检查模式，导入 numpy 和 pyarrow 库

class PandasBuffer(Buffer):
    """
    数据在缓冲区中保证是内存连续的。
    """

    def __init__(self, x: np.ndarray, allow_copy: bool = True) -> None:
        """
        目前只处理常规列（即 numpy 数组）。
        """
        if x.strides[0] and not x.strides == (x.dtype.itemsize,):
            # 协议不支持步进缓冲区，因此需要进行复制。如果不允许复制，则抛出异常。
            if allow_copy:
                x = x.copy()
            else:
                raise RuntimeError(
                    "在非连续缓冲区的情况下，导出不能是零拷贝的"
                )

        # 将数据存储在私有属性中，以便通过它获取公共属性
        self._x = x

    @property
    def bufsize(self) -> int:
        """
        缓冲区大小，以字节为单位。
        """
        return self._x.size * self._x.dtype.itemsize

    @property
    def ptr(self) -> int:
        """
        缓冲区起始指针，返回整数。
        """
        return self._x.__array_interface__["data"][0]

    def __dlpack__(self) -> Any:
        """
        将该结构表示为 DLPack 接口。
        """
        return self._x.__dlpack__()

    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        缓冲区数据所在的设备类型和设备 ID。
        """
        return (DlpackDeviceType.CPU, None)

    def __repr__(self) -> str:
        return (
            "PandasBuffer("
            + str(
                {
                    "bufsize": self.bufsize,
                    "ptr": self.ptr,
                    "device": self.__dlpack_device__()[0].name,
                }
            )
            + ")"
        )


class PandasBufferPyarrow(Buffer):
    """
    数据在缓冲区中保证是内存连续的。
    """

    def __init__(
        self,
        buffer: pa.Buffer,
        *,
        length: int,
    ) -> None:
        """
        处理 pyarrow 的分块数组。
        """
        self._buffer = buffer
        self._length = length

    @property
    def bufsize(self) -> int:
        """
        缓冲区大小，以字节为单位。
        """
        return self._buffer.size

    @property
    def ptr(self) -> int:
        """
        缓冲区起始指针，返回整数。
        """
        return self._buffer.address

    def __dlpack__(self) -> Any:
        """
        表示该结构为 DLPack 接口。
        """
        raise NotImplementedError
    # 返回元组，表示数据缓冲区所在的设备类型和设备ID（如果适用）
    def __dlpack_device__(self) -> tuple[DlpackDeviceType, int | None]:
        """
        Device type and device ID for where the data in the buffer resides.
        """
        # 返回固定的设备类型为 CPU，设备 ID 为 None
        return (DlpackDeviceType.CPU, None)

    # 返回一个描述对象的字符串表示形式
    def __repr__(self) -> str:
        # 返回一个包含 PandasBuffer[pyarrow] 的字符串，以及一个包含对象属性的字典的字符串表示
        return (
            "PandasBuffer[pyarrow]("  # 返回包含类名和库名的字符串
            + str(  # 将对象属性字典转换为字符串并添加到主字符串中
                {
                    "bufsize": self.bufsize,  # 对象的缓冲区大小属性
                    "ptr": self.ptr,  # 对象的指针属性
                    "device": "CPU",  # 固定指示设备为 CPU
                }
            )
            + ")"  # 结束字符串表示
        )
```