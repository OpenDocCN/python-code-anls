# `.\pytorch\torch\utils\data\datapipes\iter\routeddecoder.py`

```py
# 导入必要的模块和类
from io import BufferedIOBase
from typing import Any, Callable, Iterable, Iterator, Sized, Tuple

# 导入特定的函数和类
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import _deprecation_warning
from torch.utils.data.datapipes.utils.decoder import (
    basichandlers as decoder_basichandlers,
    Decoder,
    extension_extract_fn,
    imagehandler as decoder_imagehandler,
)

# 导出该模块中的特定类
__all__ = ["RoutedDecoderIterDataPipe"]


# 使用 functional_datapipe 装饰器将类标记为数据管道函数
@functional_datapipe("routed_decode")
class RoutedDecoderIterDataPipe(IterDataPipe[Tuple[str, Any]]):
    """
    从输入的数据管道解码二进制流，以元组形式生成路径名和解码后的数据。

    (函数名: ``routed_decode``)

    Args:
        datapipe: 提供路径名和二进制流的可迭代数据管道
        handlers: 可选的用户自定义解码处理程序。如果为 ``None``，将设置基本和图像解码处理程序作为默认值。
                  如果提供多个处理程序，则按照处理程序的顺序设置优先级（第一个处理程序具有最高优先级）。
        key_fn: 用于从路径名中提取键以分派处理程序的函数，默认设置为从路径名中提取文件扩展名

    Note:
        当指定 ``key_fn`` 返回除扩展名之外的任何内容时，默认处理程序将无法正常工作，
        用户需要指定自定义处理程序。自定义处理程序可以使用正则表达式确定是否符合处理数据的条件。
    """

    def __init__(
        self,
        datapipe: Iterable[Tuple[str, BufferedIOBase]],
        *handlers: Callable,
        key_fn: Callable = extension_extract_fn,
    ) -> None:
        super().__init__()
        # 设置类属性以存储传入的数据管道和解码器实例
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        if not handlers:
            handlers = (decoder_basichandlers, decoder_imagehandler("torch"))
        # 初始化解码器对象，传入用户定义的处理程序和键提取函数
        self.decoder = Decoder(*handlers, key_fn=key_fn)
        # 发出弃用警告，提示功能名称的更新版本信息
        _deprecation_warning(
            type(self).__name__,
            deprecation_version="1.12",
            removal_version="1.13",
            old_functional_name="routed_decode",
        )

    # 添加新的解码处理程序
    def add_handler(self, *handler: Callable) -> None:
        self.decoder.add_handler(*handler)

    # 实现迭代器方法，遍历数据管道中的每个元素
    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for data in self.datapipe:
            pathname = data[0]
            # 使用解码器处理数据，返回解码后的结果字典，并生成路径名和对应的解码数据
            result = self.decoder(data)
            yield (pathname, result[pathname])

    # 实现长度方法，返回数据管道的长度（如果数据管道可计算长度）
    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)
        # 如果数据管道不可计算长度，则引发类型错误异常
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
```