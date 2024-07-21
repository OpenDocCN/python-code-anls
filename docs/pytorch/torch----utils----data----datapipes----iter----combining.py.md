# `.\pytorch\torch\utils\data\datapipes\iter\combining.py`

```
# mypy: allow-untyped-defs
# 引入 copy 模块并命名为 copymodule
import copy as copymodule
# 引入警告模块
import warnings
# 引入抽象基类 ABC 和抽象方法装饰器 abstractmethod
from abc import ABC, abstractmethod
# 引入双端队列数据结构
from collections import deque
# 引入类型提示相关模块
from typing import (
    Any,
    Callable,
    Deque,
    Iterator,
    List,
    Literal,
    Optional,
    Sized,
    Tuple,
    TypeVar,
)

# 引入 Torch 数据管道相关模块和函数
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn, StreamWrapper

# __all__ 列表，定义模块公开的内容
__all__ = [
    "ConcaterIterDataPipe",
    "DemultiplexerIterDataPipe",
    "ForkerIterDataPipe",
    "MultiplexerIterDataPipe",
    "ZipperIterDataPipe",
]

# 定义协变类型变量 T_co
T_co = TypeVar("T_co", covariant=True)


# functional_datapipe 装饰器，标记为 "concat" 函数式数据管道
@functional_datapipe("concat")
# ConcaterIterDataPipe 类，继承自 IterDataPipe 基类
class ConcaterIterDataPipe(IterDataPipe):
    r"""
    Concatenates multiple Iterable DataPipes (functional name: ``concat``).

    The resulting DataPipe will yield all the elements from the first input DataPipe, before yielding from the subsequent ones.

    Args:
        datapipes: Iterable DataPipes being concatenated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> import random
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1 = IterableWrapper(range(3))
        >>> dp2 = IterableWrapper(range(5))
        >>> list(dp1.concat(dp2))
        [0, 1, 2, 0, 1, 2, 3, 4]
    """

    # datapipes 属性，包含多个 Iterable DataPipe 对象的元组
    datapipes: Tuple[IterDataPipe]

    # 构造函数，接受多个 IterDataPipe 对象作为参数
    def __init__(self, *datapipes: IterDataPipe):
        # 如果未传入任何 DataPipe，则引发 ValueError
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        # 检查所有输入是否为 IterDataPipe 类型
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `IterDataPipe`")
        # 将输入的 DataPipe 参数保存到 datapipes 属性
        self.datapipes = datapipes  # type: ignore[assignment]

    # 迭代器方法，遍历所有内部的 DataPipe 对象
    def __iter__(self) -> Iterator:
        for dp in self.datapipes:
            yield from dp

    # 长度方法，返回所有内部 DataPipe 对象的总长度
    def __len__(self) -> int:
        # 如果所有的 DataPipe 都具有长度属性，则返回它们的总和
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return sum(len(dp) for dp in self.datapipes)
        else:
            # 否则，引发类型错误
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


# functional_datapipe 装饰器，标记为 "fork" 函数式数据管道
@functional_datapipe("fork")
# ForkerIterDataPipe 类，继承自 IterDataPipe 基类
class ForkerIterDataPipe(IterDataPipe):
    r"""
    Creates multiple instances of the same Iterable DataPipe (functional name: ``fork``).

    Args:
        datapipe: Iterable DataPipe being copied
        num_instances: number of instances of the datapipe to create
        buffer_size: this restricts how far ahead the leading child DataPipe
           can read relative to the slowest child DataPipe.
           Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
        copy: copy strategy to use for items yielded by each branch. Supported
            options are ``None`` for no copying, ``"shallow"`` for shallow object
            copies, and ``"deep"`` for deep object copies. Defaults to ``None``.
    """
    
    # 构造函数，接受多个参数，包括被复制的 Iterable DataPipe 实例
    def __init__(
        self,
        datapipe: IterDataPipe,
        num_instances: int,
        buffer_size: int = 1000,
        copy: Optional[Literal["shallow", "deep"]] = None,
    ):
        pass  # 此处省略具体实现，未提供详细代码
    """
    创建一个新的类方法 __new__，用于实例化对象
    cls: 类本身，通常作为第一个参数传递
    datapipe: IterDataPipe 类型的参数，表示数据管道
    num_instances: 整数参数，指定要创建的实例数目
    buffer_size: 整数参数，默认为 1000，表示缓冲区大小
    copy: 可选参数，指定复制方式，可以是 "shallow" 或 "deep"
    """

        if num_instances < 1:
            # 如果 num_instances 小于 1，则抛出值错误异常
            raise ValueError(
                f"Expected `num_instances` larger than 0, but {num_instances} is found"
            )
        if num_instances == 1:
            # 如果 num_instances 等于 1，则直接返回原始的 datapipe 对象
            return datapipe
        # 使用 _ForkerIterDataPipe 类创建一个容器对象，用于分叉数据管道
        container = _ForkerIterDataPipe(datapipe, num_instances, buffer_size, copy)  # type: ignore[abstract]
        # 返回一个列表，其中包含 num_instances 个 _ChildDataPipe 对象，每个对象分别对应一个分叉的数据管道
        return [_ChildDataPipe(container, i) for i in range(num_instances)]
class _ContainerTemplate(ABC):
    r"""Abstract class for container ``DataPipes``. The followings are three required methods."""

    @abstractmethod
    def get_next_element_by_instance(self, instance_id: int):
        # 抽象方法：根据实例ID获取下一个元素
        ...

    @abstractmethod
    def is_every_instance_exhausted(self) -> bool:
        # 抽象方法：检查所有实例是否已经耗尽
        ...

    @abstractmethod
    def reset(self) -> None:
        # 抽象方法：重置容器状态
        ...

    @abstractmethod
    def get_length_by_instance(self, instance_id: int):
        r"""Raise TypeError if it's not supposed to be implemented to support `list(datapipe)`."""
        # 抽象方法：根据实例ID获取长度，若不支持则抛出TypeError


def _no_op(x):
    return x


class _ForkerIterDataPipe(IterDataPipe, _ContainerTemplate):
    r"""
    Container to hold instance-specific information on behalf of ForkerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, and yields the next value
    as requested by the child DataPipes.
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        num_instances: int,
        buffer_size: int = 1000,
        copy: Optional[Literal["shallow", "deep"]] = None,
    ):
        self.main_datapipe = datapipe
        self._datapipe_iterator: Optional[Iterator[Any]] = None
        self.num_instances = num_instances
        self.buffer: Deque = deque()  # 用于缓存数据的双端队列
        self.buffer_size = buffer_size  # 缓存的大小限制
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `fork`, "
                "please be aware of OOM at random places",
                UserWarning,
            )
        if copy is None:
            self.copy_fn = _no_op  # 如果未指定复制方式，默认为无操作函数
        elif copy == "shallow":
            self.copy_fn = copymodule.copy  # 如果指定为"shallow"，使用浅拷贝函数
        elif copy == "deep":
            self.copy_fn = copymodule.deepcopy  # 如果指定为"deep"，使用深拷贝函数
        else:
            raise ValueError(
                f"Unknown copy method `{copy}` requested, choose one of None, `shallow` or `deep`."
            )

        self.child_pointers: List[int] = [
            0
        ] * num_instances  # 指示每个实例下一个要获取的元素的索引
        self.slowest_ptr = 0  # 最慢子实例读取的索引
        self.leading_ptr = 0  # 最快子实例读取的索引
        self.end_ptr: Optional[int] = None  # 停止子实例读取的索引
        self._child_stop: List[bool] = [True for _ in range(num_instances)]  # 标记每个子实例是否停止
    # 根据实例ID获取下一个元素的生成器函数
    def get_next_element_by_instance(self, instance_id: int):
        # 如果数据管道迭代器为空且指定实例的停止标志为真，则重新初始化数据管道迭代器
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            self._datapipe_iterator = iter(self.main_datapipe)
            self._snapshot_state = _SnapshotState.Iterating
            # 将所有实例的停止标志重置为假
            for i in range(self.num_instances):
                self._child_stop[i] = False
        try:
            # 当指定实例未停止时循环执行
            while not self._child_stop[instance_id]:
                # 指定实例的指针向前移动一位
                self.child_pointers[instance_id] += 1
                # 如果定义了结束指针，并且指定实例的指针达到结束指针，则设置停止标志为真并中断循环
                if (
                    self.end_ptr is not None
                    and self.child_pointers[instance_id] == self.end_ptr
                ):
                    self._child_stop[instance_id] = True
                    break
                # 使用缓冲区
                if self.buffer and self.child_pointers[instance_id] <= self.leading_ptr:
                    idx = self.child_pointers[instance_id] - self.slowest_ptr - 1
                    return_val = self.buffer[idx]
                else:  # 从主数据管道中获取一个元素
                    self.leading_ptr = self.child_pointers[instance_id]
                    try:
                        return_val = next(self._datapipe_iterator)  # type: ignore[arg-type]
                        self.buffer.append(return_val)
                    except StopIteration:
                        # 如果迭代器结束，设置指定实例的停止标志为真，并重置相关状态
                        self._child_stop[instance_id] = True
                        self._datapipe_iterator = None
                        self.end_ptr = self.leading_ptr
                        continue
                # 如果指定实例的指针为最慢指针+1，更新最慢指针并从缓冲区删除最旧的元素
                if self.child_pointers[instance_id] == self.slowest_ptr + 1:
                    new_min = min(
                        self.child_pointers
                    )  # 可以通过避免调用min()来进行优化
                    if self.slowest_ptr < new_min:
                        self.slowest_ptr = new_min
                        self.buffer.popleft()
                # 如果定义了缓冲区大小且超过了指定阈值，抛出缓冲区溢出错误
                if (
                    self.buffer_size >= 0
                    and self.leading_ptr > self.buffer_size + self.slowest_ptr
                ):
                    raise BufferError(
                        "ForkerIterDataPipe buffer overflow,"
                        + f"buffer size {self.buffer_size} is insufficient."
                    )
                # 返回复制处理后的元素值
                yield self.copy_fn(return_val)  # type: ignore[possibly-undefined]
        finally:
            # 设置指定实例的停止标志为真
            self._child_stop[instance_id] = True
            # 如果所有实例的停止标志均为真，则清理数据管道迭代器
            if all(self._child_stop):
                self._datapipe_iterator = None
                self._cleanup()

    # 检查所有实例是否已经耗尽数据
    def is_every_instance_exhausted(self) -> bool:
        return self.end_ptr is not None and all(self._child_stop)

    # 根据实例ID获取主数据管道的长度
    def get_length_by_instance(self, instance_id: int) -> int:
        return len(self.main_datapipe)
    # 重置对象的状态，将所有相关变量重置为初始状态
    def reset(self) -> None:
        self._datapipe_iterator = None  # 将迭代器设为None，表示迭代器未初始化
        self.buffer = deque()  # 创建一个空的双端队列，用于存储数据
        self.child_pointers = [0] * self.num_instances  # 创建一个长度为 num_instances 的列表，用0填充，表示子指针的初始位置
        self.slowest_ptr = 0  # 指向最慢子指针的索引，初始化为0
        self.leading_ptr = 0  # 指向最快子指针的索引，初始化为0
        self.end_ptr = None  # 结束指针，初始化为None
        self._child_stop = [True for _ in range(self.num_instances)]  # 创建一个长度为 num_instances 的列表，所有元素为True，表示子数据管道是否停止的状态

    # 将对象的状态序列化为一个元组，并返回该元组
    def __getstate__(self):
        state = (
            self.main_datapipe,  # 主数据管道对象的状态
            self.num_instances,  # 实例数量
            self.buffer_size,  # 缓冲区大小
            self.copy_fn,  # 复制函数
            self._valid_iterator_id,  # 有效迭代器ID
            self._number_of_samples_yielded,  # 已产生的样本数量
        )
        # 如果存在自定义的序列化钩子函数，则使用该钩子函数处理状态
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    # 根据给定的状态元组来设置对象的状态
    def __setstate__(self, state):
        (
            self.main_datapipe,  # 主数据管道对象
            self.num_instances,  # 实例数量
            self.buffer_size,  # 缓冲区大小
            self.copy_fn,  # 复制函数
            self._valid_iterator_id,  # 有效迭代器ID
            self._number_of_samples_yielded,  # 已产生的样本数量
        ) = state
        self._datapipe_iterator = None  # 将迭代器设为None，表示迭代器未初始化
        self.buffer = deque()  # 创建一个空的双端队列，用于存储数据
        self.child_pointers = [0] * self.num_instances  # 创建一个长度为 num_instances 的列表，用0填充，表示子指针的初始位置
        self.slowest_ptr = 0  # 指向最慢子指针的索引，初始化为0
        self.leading_ptr = 0  # 指向最快子指针的索引，初始化为0
        self.end_ptr = None  # 结束指针，初始化为None
        self._child_stop = [True for _ in range(self.num_instances)]  # 创建一个长度为 num_instances 的列表，所有元素为True，表示子数据管道是否停止的状态

    # 清理缓冲区中的所有数据，并关闭相关的流对象
    def _cleanup(self):
        while self.buffer:  # 当缓冲区不为空时
            d = self.buffer.popleft()  # 从缓冲区左侧取出一个数据
            StreamWrapper.close_streams(d)  # 关闭该数据对应的流对象

    # 对象被销毁时自动调用，用于清理缓冲区中的所有数据和相关的流对象
    def __del__(self):
        self._cleanup()  # 调用 _cleanup 方法，清理缓冲区中的数据和流对象
class _ChildDataPipe(IterDataPipe):
    r"""
    Iterable Datapipe that is a child of a main DataPipe.

    The instance of this class will pass its instance_id to get the next value from its main DataPipe.

    Note:
        ChildDataPipe, like all other IterDataPipe, follows the single iterator per IterDataPipe constraint.
        Since ChildDataPipes share a common buffer, when an iterator is created for one of the ChildDataPipes,
        the previous iterators for all ChildDataPipes must be invalidated, with the exception when a ChildDataPipe
        hasn't had an iterator created from it since the last invalidation. See the example below.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> # Singler Iterator per IteraDataPipe Invalidation
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> cdp1, cdp2 = source_dp.fork(num_instances=2)
        >>> it1, it2 = iter(cdp1), iter(cdp2)
        >>> it3 = iter(cdp1)
        >>> # The line above invalidates `it1` and `it2`, and resets `ForkerIterDataPipe`.
        >>> it4 = iter(cdp2)
        >>> # The line above doesn't invalidate `it3`, because an iterator for `cdp2` hasn't been created since
        >>> # the last invalidation.

    Args:
        main_datapipe: Main DataPipe with a method 'get_next_element_by_instance(instance_id)'
        instance_id: integer identifier of this instance
    """

    _is_child_datapipe: bool = True  # 标识这是一个子数据管道类

    def __init__(self, main_datapipe: IterDataPipe, instance_id: int):
        assert isinstance(main_datapipe, _ContainerTemplate)

        self.main_datapipe: IterDataPipe = main_datapipe  # 主数据管道对象
        self.instance_id = instance_id  # 当前实例的唯一标识

    def __iter__(self):
        # Note that the logic behind setting iterator ID and `reset` are handled within `hook_iterator`
        # We want to separate the code for reset and yield, so that 'reset' executes before __next__ is called
        return self.main_datapipe.get_next_element_by_instance(self.instance_id)  # 返回由主数据管道提供的迭代器对象

    def __len__(self):
        return self.main_datapipe.get_length_by_instance(self.instance_id)  # 返回由主数据管道提供的当前实例的长度信息

    # This method is called by `hook_iterator` in `_typing.py`.
    # 更新当前 DataPipe 对象和 main_datapipe 的有效迭代器 ID
    def _set_main_datapipe_valid_iterator_id(self) -> int:
        r"""
        Update the valid iterator ID for both this DataPipe object and `main_datapipe`.

        `main_datapipe.reset()` is called when the ID is incremented to a new generation.
        """
        # 1. 第一次创建任何子迭代器时
        if self.main_datapipe._valid_iterator_id is None:
            self.main_datapipe._valid_iterator_id = 0  # type: ignore[attr-defined]
        # 2. 此实例已经处于与 `main_datapipe` 相同的迭代器生成中，
        #    我们需要进一步将 ID 增加 1
        elif self.main_datapipe._valid_iterator_id == self._valid_iterator_id:  # type: ignore[has-type]
            self.main_datapipe._valid_iterator_id += 1  # type: ignore[attr-defined]
            # 每当创建一个新的迭代器生成时，`main_datapipe` 必须重置
            if not self.main_datapipe.is_every_instance_exhausted():
                warnings.warn(
                    "Some child DataPipes are not exhausted when __iter__ is called. We are resetting "
                    "the buffer and each child DataPipe will read from the start again.",
                    UserWarning,
                )
            self.main_datapipe.reset()
        # 3. 否则，迭代器落后于其他迭代器，因此它只需通过设置实例的迭代器来与 `main_datapipe` 匹配
        self._valid_iterator_id = self.main_datapipe._valid_iterator_id
        return self._valid_iterator_id

    # 此方法由 `_typing.py` 中的 `hook_iterator` 调用。
    def _check_valid_iterator_id(self, iterator_id) -> bool:
        r"""Check the valid iterator ID against that of DataPipe object and that of `main_datapipe`."""
        return (
            iterator_id == self._valid_iterator_id
            and iterator_id == self.main_datapipe._valid_iterator_id
        )
# 使用装饰器 @functional_datapipe("demux") 将类注册为数据管道，其功能为数据解复用
@functional_datapipe("demux")
# 定义 DemultiplexerIterDataPipe 类，继承自 IterDataPipe
class DemultiplexerIterDataPipe(IterDataPipe):
    # 类的文档字符串，描述该类的作用和使用方式
    r"""
    Splits the input DataPipe into multiple child DataPipes, using the given classification function (functional name: ``demux``).

    A list of the child DataPipes is returned from this operation.

    Args:
        datapipe: Iterable DataPipe being filtered
        num_instances: number of instances of the DataPipe to create
        classifier_fn: a function that maps values to an integer within the range ``[0, num_instances - 1]`` or ``None``
        drop_none: defaults to ``False``, if ``True``, the function will skip over elements classified as ``None``
        buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
            DataPipes while waiting for their values to be yielded.
            Defaults to ``1000``. Use ``-1`` for the unlimited buffer.

    Examples:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def odd_or_even(n):
        ...     return n % 2
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even)
        >>> list(dp1)
        [0, 2, 4]
        >>> list(dp2)
        [1, 3]
        >>> # It can also filter out any element that gets `None` from the `classifier_fn`
        >>> def odd_or_even_no_zero(n):
        ...     return n % 2 if n != 0 else None
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even_no_zero, drop_none=True)
        >>> list(dp1)
        [2, 4]
        >>> list(dp2)
        [1, 3]
    """

    # __new__ 方法，用于创建实例，接受一些参数来配置实例的创建
    def __new__(
        cls,
        datapipe: IterDataPipe,
        num_instances: int,
        classifier_fn: Callable[[T_co], Optional[int]],
        drop_none: bool = False,
        buffer_size: int = 1000,
    ):
        # 检查 num_instances 参数，确保大于等于 1，否则抛出 ValueError
        if num_instances < 1:
            raise ValueError(
                f"Expected `num_instances` larger than 0, but {num_instances} is found"
            )

        # 检查 classifier_fn 函数是否可序列化（不可 pickable），如果不可序列化则抛出异常
        _check_unpickable_fn(classifier_fn)

        # 创建 _DemultiplexerIterDataPipe 实例来处理数据的分类和缓冲管理
        container = _DemultiplexerIterDataPipe(datapipe, num_instances, classifier_fn, drop_none, buffer_size)  # type: ignore[abstract]
        
        # 返回一个包含 num_instances 个 _ChildDataPipe 实例的列表，每个实例代表一个数据解复用后的子数据管道
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


# _DemultiplexerIterDataPipe 类，继承自 IterDataPipe 和 _ContainerTemplate，用于管理解复用的具体实现
class _DemultiplexerIterDataPipe(IterDataPipe, _ContainerTemplate):
    # 类的文档字符串，描述该类的作用和管理子数据管道的状态
    r"""
    Container to hold instance-specific information on behalf of DemultiplexerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, classifies and yields the next correct value
    as requested by the child DataPipes.
    """
    # 初始化方法，接受数据管道、实例数、分类器函数、是否丢弃空值、缓冲区大小作为参数
    def __init__(
        self,
        datapipe: IterDataPipe[T_co],
        num_instances: int,
        classifier_fn: Callable[[T_co], Optional[int]],
        drop_none: bool,
        buffer_size: int,
    ):
        # 主数据管道
        self.main_datapipe = datapipe
        # 数据管道迭代器，初始为 None
        self._datapipe_iterator: Optional[Iterator[Any]] = None
        # 实例数
        self.num_instances = num_instances
        # 缓冲区大小
        self.buffer_size = buffer_size
        # 如果缓冲区大小小于 0，发出警告
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `demux`, "
                "please be aware of OOM at random places",
                UserWarning,
            )
        # 当前缓冲区使用量
        self.current_buffer_usage = 0
        # 子缓冲区列表，每个实例一个双端队列
        self.child_buffers: List[Deque[T_co]] = [deque() for _ in range(num_instances)]
        # 分类器函数
        self.classifier_fn = classifier_fn
        # 是否丢弃空值
        self.drop_none = drop_none
        # 主数据管道是否已耗尽
        self.main_datapipe_exhausted = False
        # 每个子实例是否停止的标志列表，初始为 True
        self._child_stop: List[bool] = [True for _ in range(num_instances)]

    # 私有方法，用于找到下一个符合实例 ID 的元素
    def _find_next(self, instance_id: int) -> T_co:  # type: ignore[type-var]
        while True:
            # 如果主数据管道已耗尽或当前实例已停止，则抛出 StopIteration 异常
            if self.main_datapipe_exhausted or self._child_stop[instance_id]:
                raise StopIteration
            # 如果数据管道迭代器为 None，则抛出 ValueError
            if self._datapipe_iterator is None:
                raise ValueError(
                    "_datapipe_iterator has not been set, likely because this private method is called directly "
                    "without invoking get_next_element_by_instance() first."
                )
            # 获取下一个元素
            value = next(self._datapipe_iterator)
            # 使用分类器函数对元素进行分类
            classification = self.classifier_fn(value)
            # 如果分类结果为 None 并且设置了丢弃空值，则关闭流并继续下一个循环
            if classification is None and self.drop_none:
                StreamWrapper.close_streams(value)
                continue
            # 如果分类结果超出范围或小于 0，则抛出 ValueError
            if (
                classification is None
                or classification >= self.num_instances
                or classification < 0
            ):
                raise ValueError(
                    f"Output of the classification fn should be between 0 and {self.num_instances - 1}. "
                    + f"{classification} is returned."
                )
            # 如果分类结果与实例 ID 相等，则返回该元素
            if classification == instance_id:
                return value
            # 将元素添加到相应分类的子缓冲区中
            self.child_buffers[classification].append(value)
            # 更新当前缓冲区使用量
            self.current_buffer_usage += 1
            # 如果缓冲区大小大于等于 0 且当前使用量超过缓冲区大小，则抛出 BufferError
            if self.buffer_size >= 0 and self.current_buffer_usage > self.buffer_size:
                raise BufferError(
                    f"DemultiplexerIterDataPipe buffer overflow, buffer size {self.buffer_size} is insufficient."
                )
    def get_next_element_by_instance(self, instance_id: int):
        # 如果迭代器为空且特定实例已停止，则重新初始化迭代器和状态
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            # 使用主数据管道创建迭代器
            self._datapipe_iterator = iter(self.main_datapipe)
            # 设置快照状态为迭代中，用于数据管道的正确重置
            self._snapshot_state = (
                _SnapshotState.Iterating
            )  # 这对于数据管道的正确重置是必要的
            self.main_datapipe_exhausted = False
            # 将所有实例的停止状态重置为 False
            for i in range(self.num_instances):
                self._child_stop[i] = False

        try:
            # 循环直到特定实例被停止
            while not self._child_stop[instance_id]:
                if self.child_buffers[instance_id]:
                    # 如果实例的缓冲区非空，减少当前缓冲区使用量并返回缓冲区中的下一个元素
                    self.current_buffer_usage -= 1
                    yield self.child_buffers[instance_id].popleft()
                else:
                    try:
                        # 否则寻找特定实例的下一个元素并返回
                        yield self._find_next(instance_id)
                    except StopIteration:
                        # 如果找不到下一个元素，将特定实例的停止状态设置为 True
                        self._child_stop[instance_id] = True
                        # 设置主数据管道已耗尽的标志为 True
                        self.main_datapipe_exhausted = True
                        # 重置迭代器为空
                        self._datapipe_iterator = None
        finally:
            # 最终始终将特定实例的停止状态设置为 True
            self._child_stop[instance_id] = True
            # 如果所有实例都已停止，将迭代器设置为空
            if all(self._child_stop):
                self._datapipe_iterator = None
            # 清理特定实例的缓冲区
            if self.child_buffers[instance_id]:
                self._cleanup(instance_id)

    def is_every_instance_exhausted(self) -> bool:
        # 检查主数据管道是否已耗尽且所有实例均已停止
        return self.main_datapipe_exhausted and all(self._child_stop)

    def get_length_by_instance(self, instance_id: int) -> int:
        # 抛出类型错误，该方法未实现
        raise TypeError

    def reset(self) -> None:
        # 重置所有状态和属性
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def __getstate__(self):
        # 返回对象的状态，用于序列化
        state = (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        # 恢复对象的状态，用于反序列化
        (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        # 重置所有状态和属性
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False
    # 定义私有方法 `_cleanup`，用于清理实例化的对象缓冲区
    def _cleanup(self, instance_id: Optional[int] = None):
        # 如果未指定特定实例 ID，处理所有实例
        ids = (
            range(self.num_instances)
            if instance_id is None
            else [
                instance_id,
            ]
        )
        # 遍历所有指定的实例 ID
        for i in ids:
            # 获取第 i 个子缓冲区
            q = self.child_buffers[i]
            # 循环处理缓冲区中的每个数据对象
            while q:
                # 从缓冲区中弹出一个数据对象
                d = q.popleft()
                # 关闭数据对象中的流对象
                StreamWrapper.close_streams(d)

    # 定义析构方法 `__del__`，在对象被销毁时自动调用
    def __del__(self):
        # 调用 `_cleanup` 方法，清理所有实例的缓冲区
        self._cleanup()
@functional_datapipe("zip")
class ZipperIterDataPipe(IterDataPipe[Tuple[T_co]]):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated
    """

    def __init__(self, *datapipes):
        self.datapipes = datapipes  # 存储传入的数据管道列表

    def __iter__(self):
        iterators = [iter(x) for x in self.datapipes]  # 初始化每个数据管道的迭代器列表
        while True:
            try:
                yield tuple(next(it) for it in iterators)  # 从每个迭代器中取出一个元素组成元组并产生
            except StopIteration:
                return  # 当任何一个数据管道迭代完毕时结束生成器
    # 定义一个数据流水线的类，用于将多个数据流水线并行处理
    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """

    # 类型提示：声明一个元组，其中每个元素都是 IterDataPipe 类型的数据流水线对象
    datapipes: Tuple[IterDataPipe]

    # 初始化方法，接收多个 IterDataPipe 对象作为参数
    def __init__(self, *datapipes: IterDataPipe):
        # 检查所有参数是否都是 IterDataPipe 类型的对象，如果有不是的，抛出 TypeError 异常
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError(
                "All inputs are required to be `IterDataPipe` " "for `ZipIterDataPipe`."
            )
        # 调用父类的初始化方法
        super().__init__()
        # 将接收到的数据流水线对象存储在实例变量 datapipes 中
        self.datapipes = datapipes  # type: ignore[assignment]

    # 迭代器方法，返回一个迭代器，迭代过程中将多个数据流水线对象并行处理
    def __iter__(self) -> Iterator[Tuple[T_co]]:
        # 创建一个迭代器列表，每个元素都是数据流水线对象的迭代器
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        # 使用 zip 函数将多个迭代器并行处理，生成一个元组序列
        yield from zip(*iterators)

    # 返回数据流水线对象的长度
    def __len__(self) -> int:
        # 检查所有数据流水线对象是否都具有长度属性
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            # 如果都有长度属性，则返回最小长度
            return min(len(dp) for dp in self.datapipes)
        else:
            # 如果有数据流水线对象没有长度属性，则抛出 TypeError 异常
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
```