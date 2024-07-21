# `.\pytorch\torch\utils\data\datapipes\iter\fileopener.py`

```py
# mypy: allow-untyped-defs
# 导入需要的模块和类
from io import IOBase  # IOBase类用于处理IO操作
from typing import Iterable, Optional, Tuple  # 导入类型提示相关的模块和类

from torch.utils.data.datapipes._decorator import functional_datapipe  # 导入功能性datapipe装饰器
from torch.utils.data.datapipes.datapipe import IterDataPipe  # 导入IterDataPipe类
from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames  # 导入文件路径转二进制数据的函数


__all__ = [
    "FileOpenerIterDataPipe",  # 公开的类名列表
]


@functional_datapipe("open_files")
# 标记为functional_datapipe，并指定功能名称为"open_files"
class FileOpenerIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    r"""
    给定路径名，打开文件并以元组形式（路径名，文件流）生成（功能名称: ``open_files``）。

    Args:
        datapipe: 提供路径名的可迭代datapipe
        mode: 可选的字符串，指定文件打开模式， 默认为 ``r``，其他选项有
            ``b`` 表示以二进制模式读取，``t`` 表示以文本模式读取。
        encoding: 可选的字符串，指定底层文件的编码方式，默认为 ``None`` 以匹配 ``open`` 的默认编码。
        length: datapipe的名义长度

    Note:
        打开的文件句柄将由Python的GC周期性关闭。用户可以选择显式关闭它们。

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
        >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith('.txt'))
        >>> dp = FileOpener(dp)
        >>> dp = StreamReader(dp)
        >>> list(dp)
        [('./abc.txt', 'abc')]
    """

    def __init__(
        self,
        datapipe: Iterable[str],
        mode: str = "r",
        encoding: Optional[str] = None,
        length: int = -1,
    ):
        super().__init__()
        self.datapipe: Iterable = datapipe  # 初始化datapipe属性
        self.mode: str = mode  # 初始化mode属性
        self.encoding: Optional[str] = encoding  # 初始化encoding属性

        if self.mode not in ("b", "t", "rb", "rt", "r"):
            raise ValueError(f"Invalid mode {mode}")  # 如果mode不合法，抛出错误

        if "b" in mode and encoding is not None:
            raise ValueError("binary mode doesn't take an encoding argument")  # 如果以二进制模式打开且有编码参数，抛出错误

        self.length: int = length  # 初始化length属性

    # 根据模式从路径名获取文件二进制数据，生成器函数
    def __iter__(self):
        yield from get_file_binaries_from_pathnames(
            self.datapipe, self.mode, self.encoding
        )

    # 返回datapipe的长度，如果长度为-1，则抛出类型错误
    def __len__(self):
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
```