# `.\pytorch\torch\utils\data\datapipes\map\grouping.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型引用
from typing import List, Sized, TypeVar

# 从torch.utils.data.datapipes._decorator模块中导入functional_datapipe装饰器
from torch.utils.data.datapipes._decorator import functional_datapipe
# 从torch.utils.data.datapipes.datapipe模块中导入DataChunk和MapDataPipe类
from torch.utils.data.datapipes.datapipe import DataChunk, MapDataPipe

# 定义公开的模块接口
__all__ = ["BatcherMapDataPipe"]

# 定义泛型类型T
T = TypeVar("T")

# 使用functional_datapipe装饰器，标记BatcherMapDataPipe类为"batch"功能的数据管道
@functional_datapipe("batch")
class BatcherMapDataPipe(MapDataPipe[DataChunk]):
    """
    Create mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,
    or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> batch_dp = dp.batch(batch_size=2)
        >>> list(batch_dp)
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """

    # 声明datapipe属性为MapDataPipe类型
    datapipe: MapDataPipe
    # 声明batch_size属性为整数类型
    batch_size: int
    # 声明drop_last属性为布尔类型，默认为False
    drop_last: bool

    def __init__(
        self,
        datapipe: MapDataPipe[T],
        batch_size: int,
        drop_last: bool = False,
        wrapper_class=DataChunk,
    ) -> None:
        # 断言批处理大小大于0
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        # 调用父类的初始化方法
        super().__init__()
        # 初始化实例属性
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.wrapper_class = wrapper_class

    def __getitem__(self, index) -> DataChunk:
        # 创建空列表batch，用于存储批次数据
        batch: List = []
        # 计算当前批次的索引范围
        indices = range(index * self.batch_size, (index + 1) * self.batch_size)
        try:
            # 遍历索引范围，从datapipe中获取数据并添加到batch中
            for i in indices:
                batch.append(self.datapipe[i])
            # 使用wrapper_class将batch封装成DataChunk对象并返回
            return self.wrapper_class(batch)
        except IndexError as e:
            # 如果索引超出范围且不允许丢弃最后一个不完整的批次，则抛出异常
            if not self.drop_last and len(batch) > 0:
                return self.wrapper_class(batch)
            else:
                raise IndexError(f"Index {index} is out of bound.") from e

    def __len__(self) -> int:
        # 如果datapipe具有长度属性
        if isinstance(self.datapipe, Sized):
            # 如果允许丢弃最后一个不完整的批次，则返回整除的结果
            if self.drop_last:
                return len(self.datapipe) // self.batch_size
            # 否则返回向上取整的结果
            else:
                return (len(self.datapipe) + self.batch_size - 1) // self.batch_size
        else:
            # 如果datapipe没有有效的长度属性，则抛出类型错误异常
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
```