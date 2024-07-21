# `.\pytorch\torch\utils\data\datapipes\iter\grouping.py`

```
# mypy: allow-untyped-defs
# 引入警告模块，用于显示未来可能会移除的警告信息
import warnings
# 引入默认字典模块，用于创建默认值为列表的字典
from collections import defaultdict
# 引入类型相关模块
from typing import Any, Callable, DefaultDict, Iterator, List, Optional, Sized, TypeVar

# 引入sharding模块中的功能
import torch.utils.data.datapipes.iter.sharding
# 引入功能型数据管道装饰器
from torch.utils.data.datapipes._decorator import functional_datapipe
# 引入数据块和迭代数据管道类
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe
# 引入通用工具函数模块中的函数
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn

# 导出的模块名列表
__all__ = [
    "BatcherIterDataPipe",
    "GrouperIterDataPipe",
    "UnBatcherIterDataPipe",
]

# 定义一个协变类型变量
T_co = TypeVar("T_co", covariant=True)


# 自定义的__getattr__函数，用于动态获取属性
def __getattr__(name: str):
    # 如果属性名在以下列表中
    if name in ["SHARDING_PRIORITIES", "ShardingFilterIterDataPipe"]:
        # 发出警告，提醒用户这些属性将在未来版本中移除
        warnings.warn(
            f"`{name}` from `torch.utils.data.datapipes.iter.grouping` is going to be removed in PyTorch 2.1"
            f"Please use `{name}` from the `torch.utils.data.datapipes.iter.sharding`",
            category=FutureWarning,
            stacklevel=2,
        )
        # 返回sharding模块中对应的属性对象
        return getattr(torch.utils.data.datapipes.iter.sharding, name)

    # 如果属性名不在以上列表中，抛出属性错误异常
    raise AttributeError(f"module {__name__} has no attribute {name}")


# 使用功能型数据管道装饰器标记的类，用于批处理数据
@functional_datapipe("batch")
class BatcherIterDataPipe(IterDataPipe[DataChunk]):
    r"""
    Creates mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
    last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
        wrapper_class: wrapper to apply onto each batch (type ``List``) before yielding,
            defaults to ``DataChunk``

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> dp = dp.batch(batch_size=3, drop_last=True)
        >>> list(dp)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    # 数据管道对象
    datapipe: IterDataPipe
    # 每个批次的大小
    batch_size: int
    # 是否丢弃最后不足一批的数据
    drop_last: bool

    # 初始化方法，设置批处理相关参数
    def __init__(
        self,
        datapipe: IterDataPipe,
        batch_size: int,
        drop_last: bool = False,
        wrapper_class=DataChunk,
    ) -> None:
        # 断言批次大小必须大于0
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        # 调用父类初始化方法
        super().__init__()
        # 设置数据管道对象
        self.datapipe = datapipe
        # 设置批次大小
        self.batch_size = batch_size
        # 设置是否丢弃最后不足一批的数据
        self.drop_last = drop_last
        # 设置包装器类，默认为DataChunk
        self.wrapper_class = wrapper_class

    # 迭代器方法，用于生成批次数据
    def __iter__(self) -> Iterator[DataChunk]:
        # 初始化空批次列表
        batch: List = []
        # 遍历数据管道中的每个元素
        for x in self.datapipe:
            # 将元素添加到批次列表中
            batch.append(x)
            # 如果批次列表长度达到了设定的批次大小
            if len(batch) == self.batch_size:
                # 使用包装器类包装批次列表，并生成批次数据
                yield self.wrapper_class(batch)
                # 清空批次列表，准备下一个批次数据的收集
                batch = []
        # 如果还有剩余数据未形成完整的批次
        if len(batch) > 0:
            # 如果不丢弃最后不足一批的数据，则生成最后一个批次数据
            if not self.drop_last:
                yield self.wrapper_class(batch)
    # 定义一个特殊方法 __len__，用于返回对象的长度（元素个数），返回类型为整数
    def __len__(self) -> int:
        # 检查 self.datapipe 是否属于 Sized 类型（即具有 __len__ 方法）
        if isinstance(self.datapipe, Sized):
            # 如果指定了 drop_last 参数
            if self.drop_last:
                # 返回 datapipe 的长度除以 batch_size 的整数部分作为批次数
                return len(self.datapipe) // self.batch_size
            else:
                # 返回 datapipe 的长度加上 batch_size 再减一，再除以 batch_size 的整数部分作为批次数
                return (len(self.datapipe) + self.batch_size - 1) // self.batch_size
        else:
            # 如果 self.datapipe 不是 Sized 类型，则抛出 TypeError 异常
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
@functional_datapipe("unbatch")
class UnBatcherIterDataPipe(IterDataPipe):
    r"""
    Undos batching of data (functional name: ``unbatch``).

    In other words, it flattens the data up to the specified level within a batched DataPipe.

    Args:
        datapipe: Iterable DataPipe being un-batched
        unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
            it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
        >>> dp1 = source_dp.unbatch()
        >>> list(dp1)
        [[0, 1], [2], [3, 4], [5], [6]]
        >>> dp2 = source_dp.unbatch(unbatch_level=2)
        >>> list(dp2)
        [0, 1, 2, 3, 4, 5, 6]
    """

    def __init__(self, datapipe: IterDataPipe, unbatch_level: int = 1):
        self.datapipe = datapipe  # 接收传入的 Iterable DataPipe 对象
        self.unbatch_level = unbatch_level  # 设置解批次化的级别，默认为1

    def __iter__(self):
        for element in self.datapipe:  # 遍历传入的 Iterable DataPipe 对象的每一个元素
            yield from self._dive(element, unbatch_level=self.unbatch_level)  # 调用内部方法递归地解批次化每个元素

    def _dive(self, element, unbatch_level):
        if unbatch_level < -1:
            raise ValueError("unbatch_level must be -1 or >= 0")  # 如果解批次化级别小于-1，抛出异常
        if unbatch_level == -1:
            if isinstance(element, (list, DataChunk)):  # 如果解批次化级别为-1且元素是列表或数据块类型
                for item in element:
                    yield from self._dive(item, unbatch_level=-1)  # 递归地解批次化每个子元素
            else:
                yield element  # 否则直接输出元素
        elif unbatch_level == 0:
            yield element  # 如果解批次化级别为0，直接输出元素
        else:
            if isinstance(element, (list, DataChunk)):  # 如果元素是列表或数据块类型
                for item in element:
                    yield from self._dive(item, unbatch_level=unbatch_level - 1)  # 递归地解批次化每个子元素
            else:
                raise IndexError(
                    f"unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe"
                )  # 否则抛出索引错误，指示解批次化级别超出数据管道的深度


@functional_datapipe("groupby")
class GrouperIterDataPipe(IterDataPipe[DataChunk]):
    r"""
    Groups data from IterDataPipe by keys from ``group_key_fn``, yielding a ``DataChunk`` with batch size up to ``group_size``.

    (functional name: ``groupby``).

    The samples are read sequentially from the source ``datapipe``, and a batch of samples belonging to the same group
    will be yielded as soon as the size of the batch reaches ``group_size``. When the buffer is full,
    the DataPipe will yield the largest batch with the same key, provided that its size is larger
    than ``guaranteed_group_size``. If its size is smaller, it will be dropped if ``drop_remaining=True``.

    After iterating through the entirety of source ``datapipe``, everything not dropped due to the buffer capacity
    will be yielded from the buffer, even if the group sizes are smaller than ``guaranteed_group_size``.
    """

    def __init__(self, datapipe: IterDataPipe[DataChunk]):
        self.datapipe = datapipe  # 接收传入的 Iterable DataPipe[DataChunk] 对象

    def __iter__(self):
        # 实现数据分组逻辑的迭代器方法
        raise NotImplementedError("To be implemented in subclass")
    Args:
        datapipe: Iterable datapipe to be grouped
        group_key_fn: Function used to generate group key from the data of the source datapipe
        keep_key: Option to yield the matching key along with the items in a tuple,
            resulting in `(key, [items])` otherwise returning [items]
        buffer_size: The size of buffer for ungrouped data
        group_size: The max size of each group, a batch is yielded as soon as it reaches this size
        guaranteed_group_size: The guaranteed minimum group size to be yielded in case the buffer is full
        drop_remaining: Specifies if the group smaller than ``guaranteed_group_size`` will be dropped from buffer
            when the buffer is full

    Example:
        >>> import os
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def group_fn(file):
        ...     return os.path.basename(file).split(".")[0]
        >>> source_dp = IterableWrapper(["a.png", "b.png", "a.json", "b.json", "a.jpg", "c.json"])
        >>> dp0 = source_dp.groupby(group_key_fn=group_fn)
        >>> list(dp0)
        [['a.png', 'a.json', 'a.jpg'], ['b.png', 'b.json'], ['c.json']]
        >>> # A group is yielded as soon as its size equals to `group_size`
        >>> dp1 = source_dp.groupby(group_key_fn=group_fn, group_size=2)
        >>> list(dp1)
        [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
        >>> # Scenario where `buffer` is full, and group 'a' needs to be yielded since its size > `guaranteed_group_size`
        >>> dp2 = source_dp.groupby(group_key_fn=group_fn, buffer_size=3, group_size=3, guaranteed_group_size=2)
        >>> list(dp2)
        [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
    """

    # 初始化函数，用于初始化 GroupByIterator 对象
    def __init__(
        self,
        datapipe: IterDataPipe[T_co],
        group_key_fn: Callable[[T_co], Any],
        *,
        keep_key: bool = False,
        buffer_size: int = 10000,
        group_size: Optional[int] = None,
        guaranteed_group_size: Optional[int] = None,
        drop_remaining: bool = False,
    ):
        # 检查 group_key_fn 是否可以被序列化
        _check_unpickable_fn(group_key_fn)
        # 设置实例变量
        self.datapipe = datapipe  # 存储传入的可迭代数据管道
        self.group_key_fn = group_key_fn  # 存储用于生成分组键的函数

        self.keep_key = keep_key  # 是否保留分组键
        self.max_buffer_size = buffer_size  # 设置最大缓冲区大小
        self.buffer_elements: DefaultDict[Any, List] = defaultdict(list)  # 用于存储分组数据的字典，默认为空列表
        self.curr_buffer_size = 0  # 当前缓冲区的大小
        self.group_size = group_size  # 设置每个分组的最大大小，达到此大小时即返回一个批次
        self.guaranteed_group_size = None  # 保证的最小分组大小，默认为 None
        # 如果设置了 group_size 和 buffer_size，则设置保证的分组大小为 group_size
        if group_size is not None and buffer_size is not None:
            assert 0 < group_size <= buffer_size
            self.guaranteed_group_size = group_size
        # 如果设置了 guaranteed_group_size，则需确保 group_size 也被设置，并且 guaranteed_group_size 小于等于 group_size
        if guaranteed_group_size is not None:
            assert group_size is not None and 0 < guaranteed_group_size <= group_size
            self.guaranteed_group_size = guaranteed_group_size
        self.drop_remaining = drop_remaining  # 当缓冲区已满时，是否丢弃小于 guaranteed_group_size 的分组
        self.wrapper_class = DataChunk  # 设置用于包装数据的类
    # 返回当前缓冲区中大小最大的键对应的数据，并从缓冲区中移除该键
    def _remove_biggest_key(self):
        biggest_key = None  # 初始化变量，用于存储当前找到的大小最大的键
        biggest_size = 0  # 初始化变量，用于存储当前找到的最大的缓冲区大小
        result_to_yield = None  # 初始化变量，用于存储将要返回的数据

        # 遍历缓冲区中的所有键
        for findkey in self.buffer_elements.keys():
            # 如果当前键对应的缓冲区大小大于记录的最大大小，则更新最大大小和对应的键
            if len(self.buffer_elements[findkey]) > biggest_size:
                biggest_size = len(self.buffer_elements[findkey])
                biggest_key = findkey

        # 如果设置了保证的组大小，并且最大的缓冲区大小小于保证的组大小，并且不允许丢弃剩余数据
        if (
            self.guaranteed_group_size is not None
            and biggest_size < self.guaranteed_group_size
            and not self.drop_remaining
        ):
            # 抛出运行时错误，指示无法组合项目
            raise RuntimeError(
                "Failed to group items", str(self.buffer_elements[biggest_key])
            )

        # 如果未设置保证的组大小，或者最大的缓冲区大小大于等于保证的组大小
        if (
            self.guaranteed_group_size is None
            or biggest_size >= self.guaranteed_group_size
        ):
            # 将最大键对应的数据设置为将要返回的结果
            result_to_yield = self.buffer_elements[biggest_key]

        # 减少当前缓冲区大小，并从缓冲区中删除最大的键及其对应的数据
        self.curr_buffer_size -= biggest_size
        del self.buffer_elements[biggest_key]

        # 返回准备好的结果数据
        return result_to_yield

    # 迭代器方法，用于遍历数据管道中的数据
    def __iter__(self):
        for x in self.datapipe:  # 遍历数据管道中的每一个数据元素
            key = self.group_key_fn(x)  # 根据数据元素获取分组键

            self.buffer_elements[key].append(x)  # 将数据元素添加到对应分组键的缓冲区
            self.curr_buffer_size += 1  # 增加当前缓冲区大小

            # 如果设置了分组大小，并且缓冲区中对应分组键的元素个数达到分组大小
            if self.group_size is not None and self.group_size == len(
                self.buffer_elements[key]
            ):
                # 创建数据块对象，并根据需要保留键名，然后生成结果
                result: DataChunk[Any] = self.wrapper_class(self.buffer_elements[key])
                yield (key, result) if self.keep_key else result
                self.curr_buffer_size -= len(self.buffer_elements[key])  # 减少当前缓冲区大小
                del self.buffer_elements[key]  # 从缓冲区中删除对应分组键的数据

            # 如果当前缓冲区大小达到最大缓冲区大小
            if self.curr_buffer_size == self.max_buffer_size:
                # 移除当前缓冲区中大小最大的键对应的数据，并生成结果
                result_to_yield = self._remove_biggest_key()
                if result_to_yield is not None:
                    result = self.wrapper_class(result_to_yield)
                    yield (key, result) if self.keep_key else result

        # 遍历剩余未处理的所有分组键，并生成结果
        for key in tuple(self.buffer_elements.keys()):
            result = self.wrapper_class(self.buffer_elements.pop(key))
            self.curr_buffer_size -= len(result)
            yield (key, result) if self.keep_key else result

    # 重置方法，用于重置当前对象的状态
    def reset(self) -> None:
        self.curr_buffer_size = 0  # 重置当前缓冲区大小为0
        self.buffer_elements = defaultdict(list)  # 重置缓冲区元素为一个空的默认字典列表

    # 序列化方法，用于获取对象的状态信息
    def __getstate__(self):
        # 获取当前对象的所有状态信息
        state = (
            self.datapipe,
            self.group_key_fn,
            self.keep_key,
            self.max_buffer_size,
            self.group_size,
            self.guaranteed_group_size,
            self.drop_remaining,
            self.wrapper_class,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        # 如果存在状态获取钩子，则调用该钩子函数处理状态信息
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state
    # 定义特殊方法 __setstate__，用于从状态中恢复对象的属性
    def __setstate__(self, state):
        (
            self.datapipe,  # 将状态中的 datapipe 属性赋值给当前对象的 datapipe 属性
            self.group_key_fn,  # 将状态中的 group_key_fn 属性赋值给当前对象的 group_key_fn 属性
            self.keep_key,  # 将状态中的 keep_key 属性赋值给当前对象的 keep_key 属性
            self.max_buffer_size,  # 将状态中的 max_buffer_size 属性赋值给当前对象的 max_buffer_size 属性
            self.group_size,  # 将状态中的 group_size 属性赋值给当前对象的 group_size 属性
            self.guaranteed_group_size,  # 将状态中的 guaranteed_group_size 属性赋值给当前对象的 guaranteed_group_size 属性
            self.drop_remaining,  # 将状态中的 drop_remaining 属性赋值给当前对象的 drop_remaining 属性
            self.wrapper_class,  # 将状态中的 wrapper_class 属性赋值给当前对象的 wrapper_class 属性
            self._valid_iterator_id,  # 将状态中的 _valid_iterator_id 属性赋值给当前对象的 _valid_iterator_id 属性
            self._number_of_samples_yielded,  # 将状态中的 _number_of_samples_yielded 属性赋值给当前对象的 _number_of_samples_yielded 属性
        ) = state  # 将传入的状态元组 unpack 到对应的属性中
        self.curr_buffer_size = 0  # 初始化当前对象的 curr_buffer_size 属性为 0
        self.buffer_elements = defaultdict(list)  # 初始化当前对象的 buffer_elements 属性为一个 defaultdict，值为列表

    # 定义特殊方法 __del__，用于在对象被删除前执行清理操作
    def __del__(self):
        self.buffer_elements.clear()  # 清空当前对象的 buffer_elements 属性
```