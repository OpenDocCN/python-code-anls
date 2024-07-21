# `.\pytorch\torch\utils\data\datapipes\iter\streamreader.py`

```
# 引入允许未类型化的定义，用于类型检查
# mypy: allow-untyped-defs

# 引入必要的类型
from typing import Tuple

# 从torch.utils.data.datapipes._decorator模块中引入functional_datapipe装饰器
from torch.utils.data.datapipes._decorator import functional_datapipe
# 从torch.utils.data.datapipes.datapipe模块中引入IterDataPipe类
from torch.utils.data.datapipes.datapipe import IterDataPipe

# 定义可以被外部引用的类列表
__all__ = ["StreamReaderIterDataPipe"]

# 使用functional_datapipe装饰器将类标记为一个函数式的数据管道，其函数名为"read_from_stream"
@functional_datapipe("read_from_stream")
class StreamReaderIterDataPipe(IterDataPipe[Tuple[str, bytes]]):
    """
    给定IO流和它们的标签名称，以元组形式产生带标签名称的字节流。

    (函数名: ``read_from_stream``).

    Args:
        datapipe: 可迭代数据管道，提供标签/URL和字节流
        chunk: 每次迭代从流中读取的字节数。
            如果为 ``None``，则读取直到文件末尾。

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader
        >>> from io import StringIO
        >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])
        >>> list(StreamReader(dp, chunk=1))
        [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]
    """

    # 初始化函数，接受datapipe和chunk两个参数
    def __init__(self, datapipe, chunk=None):
        self.datapipe = datapipe  # 将datapipe参数赋值给实例变量self.datapipe
        self.chunk = chunk  # 将chunk参数赋值给实例变量self.chunk

    # 迭代器方法，用于生成器的迭代过程
    def __iter__(self):
        # 遍历self.datapipe中的每一个元素，每个元素为(furl, stream)元组
        for furl, stream in self.datapipe:
            # 进入无限循环，直到break被触发
            while True:
                # 从stream中读取self.chunk指定数量的字节数据
                d = stream.read(self.chunk)
                # 如果d为空（即已经读到文件末尾）
                if not d:
                    stream.close()  # 关闭当前的stream
                    break  # 跳出当前循环
                yield (furl, d)  # 生成(furl, d)的元组作为迭代器的下一个值
```