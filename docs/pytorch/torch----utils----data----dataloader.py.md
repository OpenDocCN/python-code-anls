# `.\pytorch\torch\utils\data\dataloader.py`

```
# 设置 mypy 参数，允许未类型化的定义
r"""定义 DataLoader 和与 _BaseDataLoaderIter 子类相关的迭代器。

为了支持这两个类，在 `./_utils` 中我们定义了许多实用方法和函数，以便在多进程中运行。
例如，数据加载的工作循环在 `./_utils/worker.py` 中。
"""

# 导入所需的库和模块
import functools  # 用于创建部分函数
import itertools  # 用于生成迭代器的函数
import logging  # 用于日志记录
import multiprocessing as python_multiprocessing  # Python 多进程处理模块
import os  # 提供与操作系统交互的功能
import queue  # 提供队列数据结构，支持多线程编程
import threading  # 提供线程支持的模块
import warnings  # 提供警告处理功能
from typing import Any, Callable, Generic, Iterable, List, Optional, TypeVar, Union  # 类型提示相关模块

import torch  # PyTorch 深度学习框架
import torch.distributed as dist  # PyTorch 分布式模块
import torch.utils.data.graph_settings  # PyTorch 数据相关设置
from torch._utils import ExceptionWrapper  # 异常封装类
from torch.utils.data import _utils  # PyTorch 数据相关工具模块
from torch.utils.data.datapipes.datapipe import (  # PyTorch 数据管道模块
    _IterDataPipeSerializationWrapper,
    _MapDataPipeSerializationWrapper,
    IterDataPipe,
    MapDataPipe,
)
from torch.utils.data.dataset import Dataset, IterableDataset  # PyTorch 数据集类
from torch.utils.data.sampler import (  # PyTorch 数据采样器类
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

# 导出的模块和函数列表
__all__ = [
    "DataLoader",
    "get_worker_info",
    "default_collate",
    "default_convert",
]

T_co = TypeVar("T_co", covariant=True)  # 协变型类型变量 T_co
T = TypeVar("T")  # 类型变量 T
_worker_init_fn_t = Callable[[int], None]  # 定义工作线程初始化函数类型

# 理想情况下，我们希望按照 `collate_fn` 的返回类型对 `DataLoader` 进行参数化，
# 但是目前没有办法在用户未传递自定义 'collate_fn' 时设置默认值。
# 参考 https://github.com/python/mypy/issues/3737。
_collate_fn_t = Callable[[List[T]], Any]  # 定义用于集合处理的函数类型

# 这些函数曾经在本文件中定义。然而，它们已经移动到 _utils/collate.py 中。
# 尽管用户很难从用户空间直接访问它（必须显式直接 `import torch.utils.data.dataloader`），
# 但这种别名保持了在这方面的 BC 兼容性。
default_collate: _collate_fn_t = _utils.collate.default_collate  # 默认的集合处理函数
default_convert = _utils.collate.default_convert  # 默认的转换函数

get_worker_info = _utils.worker.get_worker_info  # 获取工作线程信息的函数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class _DatasetKind:
    Map = 0  # 表示数据集类型为 Map
    Iterable = 1  # 表示数据集类型为 Iterable

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(  # 创建 Map 类型数据集的数据获取器
                dataset, auto_collation, collate_fn, drop_last
            )
        else:
            return _utils.fetch._IterableDatasetFetcher(  # 创建 Iterable 类型数据集的数据获取器
                dataset, auto_collation, collate_fn, drop_last
            )


class _InfiniteConstantSampler(Sampler):
    r"""类似于 ``itertools.repeat(None, None)`` 的采样器。

    用作 :class:`~torch.utils.data.IterableDataset` 的采样器。
    """

    def __iter__(self):
        while True:
            yield None  # 无限产生 None 的迭代器


def _get_distributed_settings():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()  # 返回分布式设置中的世界大小和排名
    else:
        return 1, 0  # 如果分布式未初始化，则返回默认设置
# 定义一个用于初始化分片工作器的函数，用于设置全局工作器ID和初始化数据管道。
def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id):
    # 设置全局工作器ID为当前工作器ID
    global_worker_id = worker_id
    # 获取当前工作器信息
    info = torch.utils.data.get_worker_info()
    # 断言确保工作器信息不为None
    assert info is not None
    # 获取数据集中的总工作器数
    total_workers = info.num_workers
    # 获取数据管道对象
    datapipe = info.dataset
    # 断言确保数据管道是IterDataPipe或MapDataPipe的实例
    assert isinstance(datapipe, (IterDataPipe, MapDataPipe))
    # 为了在分布式进程中均匀分布元素，首先在分布式进程上进行分片，然后在工作器进程上进行分片
    total_workers *= world_size
    # 计算全局工作器ID
    global_worker_id = global_worker_id * world_size + rank_id
    # 使用默认的SHARDING_PRIORITIES进行数据分片
    torch.utils.data.graph_settings.apply_sharding(
        datapipe, total_workers, global_worker_id
    )
    # 如果提供了工作器初始化函数，则调用它
    if worker_init_fn is not None:
        worker_init_fn(worker_id)


# 定义一个用于共享分布式种子的函数，用于生成并广播一个共享种子
def _share_dist_seed(generator, pg):
    # 生成一个随机整数种子并存储在_shared_seed中
    _shared_seed = torch.empty((), dtype=torch.int64).random_(generator=generator)
    # 如果pg是dist.ProcessGroup的实例，则在进程组pg中广播_shared_seed
    if isinstance(pg, dist.ProcessGroup):
        dist.broadcast(_shared_seed, src=0, group=pg)
    # 返回共享的种子值
    return _shared_seed.item()


class DataLoader(Generic[T_co]):
    r"""
    Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.
    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.

    .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                 :ref:`data-loading-randomness` notes for random seed related questions.

    .. _multiprocessing context:
        https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    """

    # 数据加载器类的定义，继承自 Python 的 Dataset 类型
    dataset: Dataset[T_co]
    # 每个批次的大小，可选参数
    batch_size: Optional[int]
    # 使用的工作进程数量
    num_workers: int
    # 是否将数据存储在固定内存中
    pin_memory: bool
    # 是否丢弃最后一个不完整的批次
    drop_last: bool
    # 超时时间
    timeout: float
    # 数据采样器，可以是 Sampler 或 Iterable 类型
    sampler: Union[Sampler, Iterable]
    # 固定内存的设备
    pin_memory_device: str
    # 预取因子，可选参数
    prefetch_factor: Optional[int]
    # 数据加载器迭代器
    _iterator: Optional["_BaseDataLoaderIter"]
    # 初始化标志
    __initialized = False

    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Union[Sampler, Iterable, None] = None,
        batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        # 初始化函数，设置数据集、批次大小、是否混洗等参数
        ...

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        # 获取数据加载器迭代器
        if self.num_workers == 0:
            # 如果没有工作进程，则使用单进程数据加载器迭代器
            return _SingleProcessDataLoaderIter(self)
        else:
            # 否则检查工作进程数的合理性，并使用多进程数据加载器迭代器
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        # 多进程上下文的属性访问方法
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        # 检查是否传入了 multiprocessing_context 参数
        if multiprocessing_context is not None:
            # 检查是否设置了 num_workers 大于 0
            if self.num_workers > 0:
                # 如果 multiprocessing_context 是字符串类型，验证其是否为有效的启动方法
                if isinstance(multiprocessing_context, str):
                    # 获取所有有效的启动方法
                    valid_start_methods = torch.multiprocessing.get_all_start_methods()
                    # 如果 multiprocessing_context 不在有效的启动方法列表中，抛出 ValueError
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            "multiprocessing_context option "
                            f"should specify a valid start method in {valid_start_methods!r}, but got "
                            f"multiprocessing_context={multiprocessing_context!r}"
                        )
                    # 根据给定的 multiprocessing_context 创建 Torch multiprocessing 上下文
                    multiprocessing_context = torch.multiprocessing.get_context(
                        multiprocessing_context
                    )

                # 检查 multiprocessing_context 是否为有效的 multiprocessing 上下文对象
                if not isinstance(
                    multiprocessing_context, python_multiprocessing.context.BaseContext
                ):
                    raise TypeError(
                        "multiprocessing_context option should be a valid context "
                        "object or a string specifying the start method, but got "
                        f"multiprocessing_context={multiprocessing_context}"
                    )
            else:
                # 如果 num_workers 不大于 0，抛出 ValueError
                raise ValueError(
                    "multiprocessing_context can only be used with "
                    "multi-process loading (num_workers > 0), but got "
                    f"num_workers={self.num_workers}"
                )

        # 将 multiprocessing_context 参数赋值给实例变量 __multiprocessing_context
        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        # 如果已经初始化并且试图设置禁止设置的属性，则抛出 ValueError
        if self.__initialized and attr in (
            "batch_size",
            "batch_sampler",
            "sampler",
            "drop_last",
            "dataset",
            "persistent_workers",
        ):
            raise ValueError(
                f"{attr} attribute should not be set after {self.__class__.__name__} is initialized"
            )

        # 调用父类的 __setattr__ 方法设置属性
        super().__setattr__(attr, val)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> "_BaseDataLoaderIter":
        # 当使用单个 worker 时，每次都创建新的迭代器以避免重置其状态
        # 对于多个 worker 的迭代器，在 DataLoader 对象的生命周期内仅创建一次迭代器以便重复使用 worker
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                # 如果 _iterator 为 None，首次创建迭代器
                self._iterator = self._get_iterator()
            else:
                # 如果 _iterator 已存在，重置其状态
                self._iterator._reset(self)
            return self._iterator
        else:
            # 返回通过 _get_iterator 方法获取的迭代器
            return self._get_iterator()

    @property
    def _auto_collation(self):
        # 检查是否启用了自动批次合并
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # 用于生成 `_DatasetFetcher` 的索引的实际采样器
        # (参见 _utils/fetch.py)，用于每次读取数据。如果处于自动拼装模式，
        # 这将是 `.batch_sampler`；否则将是 `.sampler`。
        # 由于向后兼容性原因，我们不能更改 `.sampler` 和 `.batch_sampler` 属性。
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            # 注意 [ IterableDataset 和 __len__ ]
            #
            # 对于 `IterableDataset`，当使用多进程数据加载时，`__len__` 可能不准确，
            # 因为样本会被复制。但是，实际上不应该有使用该行为的真实用例，所以这应该
            # 视为用户错误。我们通常应该信任用户代码来执行正确的操作（例如，在 `__iter__`
            # 中为每个副本配置不同的设置），如果他们选择实现 `__len__`，我们应该给出正确
            # 的 `__len__`（如果数据集没有实现 `__len__`，这仍将抛出异常）。
            #
            # 为了提供进一步的警告，我们跟踪是否在 `DataLoader` 上调用了 `__len__`，
            # 将返回值保存在 `self._len_called` 中，如果迭代器产生的样本数超过这个数字，
            # 则发出警告。

            # 无法静态验证数据集是否具有固定大小
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore[assignment, arg-type]
            if (
                self.batch_size is not None
            ):  # IterableDataset 不允许自定义采样器或批量采样器
                from math import ceil

                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)
# 定义一个基础数据加载器迭代器的类，用于迭代 DataLoader 的数据集
class _BaseDataLoaderIter:
    # 初始化方法，接受一个 DataLoader 实例作为参数
    def __init__(self, loader: DataLoader) -> None:
        # 将 loader 的数据集对象保存到 _dataset 属性中
        self._dataset = loader.dataset
        # _shared_seed 和 _pg 属性初始化为 None
        self._shared_seed = None
        self._pg = None
        # 如果数据集是 IterDataPipe 的实例
        if isinstance(self._dataset, IterDataPipe):
            # 如果支持分布式训练且已经初始化
            if dist.is_available() and dist.is_initialized():
                # 使用 gloo 后端创建新的分组
                self._pg = dist.new_group(backend="gloo")
            # 获取共享的分布式随机种子
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            # 创建共享的随机数生成器
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            # 应用共享随机种子到数据集对象上
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(
                self._dataset, shared_rng
            )
        # 保存 loader 的一些属性到对应的属性中
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        # 获取分布式设置中的 world_size 和 rank
        ws, rank = _get_distributed_settings()
        self._world_size = ws
        self._rank = rank
        # 如果 loader 的 pin_memory_device 列表长度为 0
        if len(loader.pin_memory_device) == 0:
            # 设置 _pin_memory 属性为 loader 的 pin_memory 属性与 CUDA 是否可用的与运算结果
            self._pin_memory = loader.pin_memory and torch.cuda.is_available()
            self._pin_memory_device = None
        else:
            # 如果 pin_memory 未启用但 pin_memory_device 设置了，发出警告信息
            if not loader.pin_memory:
                warn_msg = (
                    "pin memory device is set and pin_memory flag is not used then device pinned memory won't be used"
                    "please set pin_memory to true, if you need to use the device pin memory"
                )
                warnings.warn(warn_msg)
            # 设置 _pin_memory 属性为 loader 的 pin_memory 属性
            self._pin_memory = loader.pin_memory
            # 设置 _pin_memory_device 属性为 loader 的 pin_memory_device 属性
            self._pin_memory_device = loader.pin_memory_device
        # 设置 _timeout 属性为 loader 的 timeout 属性
        self._timeout = loader.timeout
        # 设置 _collate_fn 属性为 loader 的 collate_fn 属性
        self._collate_fn = loader.collate_fn
        # 通过迭代器获取 _index_sampler 的迭代器对象并保存到 _sampler_iter 属性中
        self._sampler_iter = iter(self._index_sampler)
        # 生成基础种子并保存到 _base_seed 属性中
        self._base_seed = (
            torch.empty((), dtype=torch.int64)
            .random_(generator=loader.generator)
            .item()
        )
        # 设置 _persistent_workers 属性为 loader 的 persistent_workers 属性
        self._persistent_workers = loader.persistent_workers
        # 设置 _num_yielded 属性初始值为 0
        self._num_yielded = 0
        # 设置 _profile_name 属性为一个格式化的字符串
        self._profile_name = f"enumerate(DataLoader)#{self.__class__.__name__}.__next__"

    # 定义 __iter__ 方法，返回当前对象自身，以支持迭代器协议
    def __iter__(self) -> "_BaseDataLoaderIter":
        return self
    # 重置迭代器状态，初始化或重新加载迭代器对象
    def _reset(self, loader, first_iter=False):
        # 使用索引采样器创建迭代器对象
        self._sampler_iter = iter(self._index_sampler)
        # 记录已经生成的样本数，初始化为0
        self._num_yielded = 0
        # 记录是否调用过IterableDataset的长度方法
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        # 如果数据集是IterDataPipe的实例
        if isinstance(self._dataset, IterDataPipe):
            # 共享分布式随机数种子
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            # 创建共享的随机数生成器
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            # 设置数据集的随机种子
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(
                self._dataset, shared_rng
            )

    # 获取下一个索引的方法
    def _next_index(self):
        return next(self._sampler_iter)  # 可能会引发StopIteration异常

    # 获取下一个数据的方法，由子类实现
    def _next_data(self):
        raise NotImplementedError

    # 迭代器的下一个元素方法，返回任意类型的数据
    def __next__(self) -> Any:
        # 使用PyTorch的profiler记录函数执行时间
        with torch.autograd.profiler.record_function(self._profile_name):
            # 如果采样器迭代器为None，则重置迭代器状态
            if self._sampler_iter is None:
                # TODO(https://github.com/pytorch/pytorch/issues/76750)
                self._reset()  # type: ignore[call-arg]
            # 获取下一个数据
            data = self._next_data()
            # 记录已经生成的样本数加一
            self._num_yielded += 1
            # 如果数据集类型是Iterable，且已经调用过长度方法，且生成的样本数超过长度方法报告的数量
            if (
                self._dataset_kind == _DatasetKind.Iterable
                and self._IterableDataset_len_called is not None
                and self._num_yielded > self._IterableDataset_len_called
            ):
                # 发出警告消息，说明IterableDataset的长度与实际访问长度不一致的情况
                warn_msg = (
                    f"Length of IterableDataset {self._dataset} was reported to be {self._IterableDataset_len_called}"
                    f"(when accessing len(dataloader)), but {self._num_yielded} samples have been fetched. "
                )
                # 如果使用多进程数据加载，可能是由于未正确配置每个worker的IterableDataset副本所致
                if self._num_workers > 0:
                    warn_msg += (
                        "For multiprocessing data-loading, this could be caused by not properly configuring the "
                        "IterableDataset replica at each worker. Please see "
                        "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples."
                    )
                # 发出警告
                warnings.warn(warn_msg)
            # 返回获取的数据
            return data

    # 返回索引采样器的长度
    def __len__(self) -> int:
        return len(self._index_sampler)

    # 序列化对象时的方法，目前未实现，抛出NotImplementedError异常
    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)
class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Taking care of distributed sharding
        # 如果数据集是IterDataPipe或MapDataPipe类型的，处理分布式分片
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            # 对于向后兼容性，使用默认的SHARDING_PRIORITIES
            torch.utils.data.graph_settings.apply_sharding(
                self._dataset, self._world_size, self._rank
            )

        # 创建数据集获取器，用于获取数据集中的数据
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def _next_data(self):
        # 获取下一个数据索引，可能会引发StopIteration异常
        index = self._next_index()
        # 使用数据集获取器获取对应索引的数据，可能会引发StopIteration异常
        data = self._dataset_fetcher.fetch(index)
        # 如果设置了pin_memory选项，则将数据放入固定内存
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler."""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed (e.g., releasing GPU memory).
    #      Naturally, we implement the shutdown logic in `__del__` of
    #   DataLoaderIterator.
    #
    #   We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers and `pin_memory_thread` to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when the
    #      iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may already be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children.
    #
    #      Thus, we register an `atexit` hook that sets a global flag
    #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
    #      reverse order of registration, we are guaranteed that this flag is
    #      set before library resources we use are freed (which, at least in
    #      CPython, is done via an `atexit` handler defined in
    #      `multiprocessing/util.py`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/util.py#L320-L362
    #      registered when an object requiring this mechanism is first
    #      created, e.g., `mp.Queue`
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/context.py#L100-L103
    #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/queues.py#L29
    #      )
    #
    #      So in `__del__`, we check if `_utils.python_exit_status` is set or
    #      `None` (freed), and perform no-op if so.
    #
    #      However, simply letting library clean-up codes run can also be bad,
    #      because such codes (i.e., `multiprocessing.util._exit_function()`)
    #      include join putting threads for `mp.Queue`, which can be blocking.
    #      Hence, the main process putting threads are called with
    #      `cancel_join_thread` at creation.  See later section
    #      [ 3b. A process won't hang when putting into a queue; ]
    #      for more details.
    #
    #      Here are two example cases where library clean-up codes can run
    #      before `__del__` is called:
    #
    #        1. If we hold onto a reference to the iterator, it more often
    #           than not tries to do `multiprocessing` library cleaning before
    #           clearing the alive referenced objects (https://github.com/pytorch/pytorch/issues/48666)
    #           and thus prevents our cleaning-up code to run first.
    #
    #        2. A similar issue araises when a `DataLoader` is used in a subprocess.
    #           When a process ends, it shuts the all its daemonic children
    #           down with a SIGTERM (instead of joining them without a timeout).
    #           Simiarly for threads, but by a different mechanism. This fact,
    #           together with a few implementation details of multiprocessing, forces
    #           us to make workers daemonic. All of our problems arise when a
    #           DataLoader is used in a subprocess, and are caused by multiprocessing
    #           code which looks more or less like this:
    #
    #               try:
    #                   your_function_using_a_dataloader()
    #               finally:
    #                   multiprocessing.util._exit_function()
    #
    #           The joining/termination mentioned above happens inside
    #           `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #           throws, the stack trace stored in the exception will prevent the
    #           frame which uses `DataLoaderIter` to be freed. If the frame has any
    #           reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #           its  `__del__`, which starts the shutdown procedure, will not be
    #           called. That, in turn, means that workers aren't notified. Attempting
    #           to join in `_exit_function` will then result in a hang.
    #
    #           For context, `_exit_function` is also registered as an `atexit` call.
    #           So it is unclear to me (@ssnl) why this is needed in a finally block.
    #           The code dates back to 2008 and there is no comment on the original
    #           PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #           the finally block and the `atexit` registration) that explains this.
    #
    #
    #      Finally, another choice is to just shutdown workers with logic in 1
    #      above whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #        always corresponding to a `get()`), hanging on `get()` can still
    #        happen when data in queue is corrupted (e.g., due to
    #        `cancel_join_thread` or unexpected exit).
    #
    #        For child exit, we set a timeout whenever we try to get data
    #        from `data_queue`, and check the workers' status on each timeout
    #        and error.
    #        See `_DataLoaderiter._get_batch()` and
    #        `_DataLoaderiter._try_get_data()` for details.
    #
    #        Additionally, for child exit on non-Windows platforms, we also
    #        register a SIGCHLD handler (which is supported on Windows) on
    #        the main process, which checks if any of the workers fail in the
    #        (Python) handler. This is more efficient and faster in detecting
    #        worker failures, compared to only using the above mechanism.
    #        See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
    #
    #        For `.get()` calls where the sender(s) is not the workers, we
    #        guard them with timeouts, and check the status of the sender
    #        when timeout happens:
    #          + in the workers, the `_utils.worker.ManagerWatchdog` class
    #            checks the status of the main process.
    #          + if `pin_memory=True`, when getting from `pin_memory_thread`,
    #            check `pin_memory_thread` status periodically until `.get()`
    #            returns or see that `pin_memory_thread` died.
    #
    #     b. A process won't hang when putting into a queue;
    #
    #        We use `mp.Queue` which has a separate background thread to put
    #        objects from an unbounded buffer array. The background thread is
    #        daemonic and usually automatically joined when the process
    #        *exits*.
    #
    #        In case that the receiver has ended abruptly while
    #        reading from the pipe, the join will hang forever.  The usual
    #        solution for this in Python is calling  `q.cancel_join_thread`,
    #        which prevents automatically joining it when finalizing
    #        (exiting).
    #
    #        Nonetheless, `cancel_join_thread` must only be called when the
    #        queue is **not** going to be read from or write into by another
    #        process, because it may hold onto a lock or leave corrupted data
    #        in the queue, leading other readers/writers to hang.
    #
    #        Hence,
    #          + For worker processes, we only do so (for their output
    #            queues, i.e., `worker_result_queue`) before exiting.
    #          + For `pin_memory_thread`, its output queue `data_queue` is a
    #            `queue.Queue` that does blocking `put` if the queue is full.
    #            So there is no above problem, but as a result, in
    #               `_pin_memory_loop`, we do need to  wrap the `put` in a loop
    #               that breaks not only upon success, but also when the main
    #               process stops reading, i.e., is shutting down.
    #             + For loader process, we `cancel_join_thread()` for all
    #               `_index_queues` because the whole purpose of workers and
    #               `pin_memory_thread` is to serve the loader process.  If
    #               loader process is already exiting, we don't really care if
    #               the queues are corrupted.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # `workers_done_event`:
    #   A `multiprocessing.Event` shared among the main process and all worker
    #   processes. This is used to signal the workers that the iterator is
    #   shutting down. After it is set, they will not send processed data to
    #   queues anymore, and only wait for the final `None` before exiting.
    #   `done_event` isn't strictly needed. I.e., we can just check for `None`
    #   from the input queue, but it allows us to skip wasting resources
    #   processing data if we are already shutting down.
    #
    # `pin_memory_thread_done_event`:
    #   A `threading.Event` for a similar purpose to that of
    #   `workers_done_event`, but is for the `pin_memory_thread`. The reason
    #   that separate events are needed is that `pin_memory_thread` reads from
    #   the output queue of the workers. But the workers, upon seeing that
    #   `workers_done_event` is set, only wants to see the final `None`, and is
    #   not required to flush all data in the output queue (e.g., it may call
    #   `cancel_join_thread` on that queue if its `IterableDataset` iterator
    #   happens to exhaust coincidentally, which is out of the control of the
    #   main process). Thus, since we will exit `pin_memory_thread` before the
    #   workers (see below), two separete events are used.
    #
    # NOTE: In short, the protocol is that the main process will set these
    #       `done_event`s and then the corresponding processes/threads a `None`,
    #       and that they may exit at any time after receiving the `None`.
    #
    # NOTE: Using `None` as the final signal is valid, since normal data will
    #       always be a 2-tuple with the 1st element being the index of the data
    #       transferred (different from dataset index/key), and the 2nd being
    #       either the dataset key or the data sample (depending on which part
    #       of the data model the queue is at).
    #
    # [ worker processes ]
    #   While loader process is alive:
    #     Get from `index_queue`.
    #       If get anything else,
    #          Check `workers_done_event`.
    #            如果设置了该条件，则继续下一次迭代
    #                    即继续获取数据，直到看到 `None`，然后退出。
    #            否则，处理数据：
    #                如果正在从 `IterableDataset` 获取数据，并且迭代器
    #                    已经耗尽，则发送 `_IterableDatasetStopIteration`
    #                    对象来信号迭代结束。主进程收到这样的对象后，会发送
    #                    `None` 给当前工作进程，并且不再使用相应的 `index_queue`。
    #       如果超时，
    #          不论 `workers_done_event` 是否设置（仍需看到 `None`），
    #          必须继续下一次迭代。
    #   （循环外）
    #   如果 `workers_done_event` 已设置，（这对于 `IterableDataset` 可能为假）
    #     `data_queue.cancel_join_thread()`。 （一切都在这里结束：
    #                                          主进程不再从中读取；
    #                                          其他工作进程也会调用
    #                                          `cancel_join_thread`。）
    #
    # [ pin_memory_thread ]
    #   # 无需检查主线程。如果此线程存活，则主加载器线程必定存活，因为此线程设置为守护线程。
    #   当 `pin_memory_thread_done_event` 未设置时：
    #     从 `worker_result_queue` 获取数据。
    #       如果超时，则继续下一次获取。
    #       否则，处理数据。
    #       当 `pin_memory_thread_done_event` 未设置时：
    #         将处理过的数据放入 `data_queue`（一个带有阻塞放置的 `queue.Queue`）。
    #         如果超时，则继续下一次放置。
    #         否则，跳出循环，即继续到外层循环。
    #
    #   注意：我们不检查主线程的状态，因为
    #           1. 如果进程被致命信号杀死，`pin_memory_thread` 会结束。
    #           2. 在其他情况下，无论是在 `__del__` 中的清理还是守护线程的自动退出
    #              都会处理它。这也不会忙等待，因为 `.get(timeout)` 不会忙等待。
    #
    # [ 主进程 ]
    #   在 DataLoader 迭代器的 `__del__` 方法中：
    #     b. 退出 `pin_memory_thread`
    #          i.   设置 `pin_memory_thread_done_event`。
    #          ii.  在 `worker_result_queue` 中放入 `None`。
    #          iii. 加入 `pin_memory_thread`。
    #          iv.  `worker_result_queue.cancel_join_thread()`。
    #
    #     c. 退出工作进程。
    #          i.   设置 `workers_done_event`。
    #          ii.  在每个工作进程的 `index_queue` 中放入 `None`。
    #          iii. 加入工作进程。
    #          iv.  在每个工作进程的 `index_queue` 上调用 `.cancel_join_thread()`。
    #
    #        注意：（c）放在（b）之后更好，因为可能会留下已损坏的状态。
    #              data in `worker_result_queue`, which `pin_memory_thread`
    #              reads from, in which case the `pin_memory_thread` can only
    #              happen at timing out, which is slow. Nonetheless, same thing
    #              happens if a worker is killed by signal at unfortunate times,
    #              but in other cases, we are better off having a non-corrupted
    #              `worker_result_queue` for `pin_memory_thread`.
    #
    #   NOTE: If `pin_memory=False`, there is no `pin_memory_thread` and (b)
    #         can be omitted
    #
    # NB: `done_event`s isn't strictly needed. E.g., we can just check for
    #     `None` from `index_queue`, but it allows us to skip wasting resources
    #     processing indices already in `index_queue` if we are already shutting
    #     down.

    # 重置方法，用于重新初始化加载器的状态
    def _reset(self, loader, first_iter=False):
        # 调用父类的重置方法
        super()._reset(loader, first_iter)
        # 下一个要发送给工作线程的任务索引
        self._send_idx = 0
        # 下一个要在 __next__ 中返回的任务索引
        self._rcvd_idx = 0
        # 记录未被返回的数据信息，即索引范围在 [rcvd_idx, send_idx) 内的任务
        # 字典结构：任务索引 => (worker_id,)        若数据未被获取（未完成）
        #                         \ (worker_id, data)   若数据已被获取（顺序不一定）
        self._task_info = {}
        # 当前未完成的任务数量，始终等于 task_info 中值为单元素元组的数量
        self._tasks_outstanding = (
            0  # 总数为 task_info.values() 中长度为 1 的项的个数
        )
        # 一个布尔列表，表示每个工作线程是否仍有任务要执行
        # 若不是使用可迭代式数据集，则始终包含全部 True
        # 注：这表明工作线程仍需在当前 epoch 下继续工作，并非表示线程已死亡
        # 在 `_persistent_workers` 情况下，工作线程会在下个 epoch 中被重置为可用状态
        self._workers_status = [True for i in range(self._num_workers)]
        # 重置工作队列循环索引，以便下个 epoch 从工作线程 0 开始
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # 若非首次迭代，则恢复预取操作
        if not first_iter:
            for idx in range(self._num_workers):
                # 将 _ResumeIteration 对象放入对应工作线程的索引队列中
                self._index_queues[idx].put(
                    _utils.worker._ResumeIteration(self._shared_seed)
                )
            # 恢复迭代次数计数
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                # 获取数据并检查是否是 _ResumeIteration 对象
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _utils.worker._ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1
        # 预取数据以启动预取循环
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()
    # 尝试从 self._data_queue 中获取数据，最多等待 timeout 秒
    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        # 尝试从 `self._data_queue` 中获取数据，等待 `timeout` 时间
        # 这也可以作为没有超时的情况下获取数据的内部循环，以发送者状态作为循环条件。
        #
        # 如果任何 worker 未预期地死亡，则会引发 `RuntimeError`。此错误可能来自 `_utils/signal_handling.py` 中的 SIGCHLD 处理程序
        # （仅适用于非 Windows 平台），或者下面手动检查错误和超时。
        #
        # 返回一个二元组：
        #   (bool: 是否成功获取数据, any: 如果成功则为数据，否则为 None)
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # 在超时和错误时，我们手动检查是否有任何 worker 失败。
            # 注意，这是 Windows 检测 worker 失败的唯一机制。
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ", ".join(str(w.pid) for w in failed_workers)
                raise RuntimeError(
                    f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
                ) from e
            if isinstance(e, queue.Empty):
                return (False, None)

            import errno
            import tempfile

            try:
                # 如果接近文件描述符（FDs）限制，抛出异常。
                # 显然，尝试仅打开一个文件不足以进行充分测试。
                # 详见 NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code"
                    ) from None
            # 如果出现其他异常，则向上层抛出
            raise

# NOTE [ DataLoader on Linux and open files limit ]
#
# 在 Linux 上，当使用 DataLoader 与 multiprocessing 一起使用时，我们通过 SHM 文件在根进程和 worker 之间传递数据。
# 一旦创建这些文件，我们会立即从文件系统中删除它们，并通过保持它们活动来
    # passing around their file descriptors through AF_UNIX sockets. (See
    # docs/source/multiprocessing.rst and 'Multiprocessing Technical Notes` in
    # the wiki (https://github.com/pytorch/pytorch/wiki).)
    #
    # This sometimes leads us to exceeding the open files limit. When that happens,
    # and the offending file descriptor is coming over a socket, the `socket` Python
    # package silently strips the file descriptor from the message, setting only the
    # `MSG_CTRUNC` flag (which might be a bit misleading since the manpage says that
    # it _indicates that some control data were discarded due to lack of space in
    # the buffer for ancillary data_). This might reflect the C implementation of
    # AF_UNIX sockets.
    #
    # This behaviour can be reproduced with the script and instructions at the
    # bottom of this note.
    #
    # When that happens, the standard Python `multiprocessing` (and not
    # `torch.multiprocessing`) raises a `RuntimeError: received 0 items of ancdata`
    #
    # Sometimes, instead of the FD being stripped, you may get an `OSError:
    # Too many open files`, both in the script below and in DataLoader. However,
    # this is rare and seems to be nondeterministic.
    #
    #
    #   #!/usr/bin/env python3
    #   import sys
    #   import socket
    #   import os
    #   import array
    #   import shutil
    #   import socket
    #
    #
    #   if len(sys.argv) != 4:
    #       print("Usage: ", sys.argv[0], " tmp_dirname iteration (send|recv)")
    #       sys.exit(1)
    #
    #   if __name__ == '__main__':
    #       dirname = sys.argv[1]
    #       sock_path = dirname + "/sock"
    #       iterations = int(sys.argv[2])
    #       def dummy_path(i):
    #           return dirname + "/" + str(i) + ".dummy"
    #
    #
    #       if sys.argv[3] == 'send':
    #           while not os.path.exists(sock_path):
    #               pass
    #           client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    #           client.connect(sock_path)
    #           for i in range(iterations):
    #               fd = os.open(dummy_path(i), os.O_WRONLY | os.O_CREAT)
    #               ancdata = array.array('i', [fd])
    #               msg = bytes([i % 256])
    #               print("Sending fd ", fd, " (iteration #", i, ")")
    #               client.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, ancdata)])
    #
    #
    #       else:
    #           assert sys.argv[3] == 'recv'
    #
    #           if os.path.exists(dirname):
    #               raise Exception("Directory exists")
    #
    #           os.mkdir(dirname)
    #
    #           print("Opening socket...")
    #           server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    #           server.bind(sock_path)
    #
    #           print("Listening...")
    #           for i in range(iterations):
    #               a = array.array('i')
    def _get_data(self):
        # Fetches data from `self._data_queue`.
        # 从 `self._data_queue` 中获取数据。

        # We check workers' status every `MP_STATUS_CHECK_INTERVAL` seconds,
        # which we achieve by running `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)`
        # in a loop. This is the only mechanism to detect worker failures for
        # Windows. For other platforms, a SIGCHLD handler is also used for
        # worker failure detection.
        #
        # 每隔 `MP_STATUS_CHECK_INTERVAL` 秒检查一次工作进程的状态，
        # 这通过在循环中运行 `self._try_get_data(timeout=MP_STATUS_CHECK_INTERVAL)` 实现。
        # 这是在 Windows 平台检测工作进程失败的唯一机制。
        # 对于其他平台，还使用了 SIGCHLD 处理程序来检测工作进程的失败。

        # If `pin_memory=True`, we also need check if `pin_memory_thread` had
        # died at timeouts.
        #
        # 如果 `pin_memory=True`，还需要检查 `pin_memory_thread` 是否在超时时退出。

        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError(
                    f"DataLoader timed out after {self._timeout} seconds"
                )
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError("Pin memory thread exited unexpectedly")
            # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
            #
            # 在这种情况下，`self._data_queue` 是一个 `queue.Queue`，但我们不需要调用 `.task_done()`，
            # 因为我们不使用 `.join()`。

        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data
    # 循环直到找到下一个有效的数据批次索引
    def _next_data(self):
        while True:
            # 如果负责处理 `self._rcvd_idx` 的工作进程已经结束
            # 并且无法完成此任务（因为耗尽了 `IterableDataset` 的内容），
            # 我们尝试前进 `self._rcvd_idx` 来找到下一个有效的索引。
            #
            # 这部分需要在循环中运行，因为 `self._get_data()` 的调用和下面的 `_IterableDatasetStopIteration` 检查
            # 都可能导致额外的工作进程变成不可用状态。
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                # 如果信息长度为 2 或者工作进程仍然处于活动状态，则跳出循环
                if (
                    len(info) == 2 or self._workers_status[worker_id]
                ):  # 有数据或者仍然活动
                    break
                # 删除 `self._rcvd_idx` 对应的任务信息，并前进到下一个索引
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # 如果未找到有效的 `self._rcvd_idx`（即没有跳出循环）
                if not self._persistent_workers:
                    # 如果不使用持久化工作进程，则关闭工作进程
                    self._shutdown_workers()
                # 抛出 StopIteration 异常
                raise StopIteration

            # 现在 `self._rcvd_idx` 是我们想要获取的批次索引

            # 检查下一个样本是否已经生成
            if len(self._task_info[self._rcvd_idx]) == 2:
                # 弹出 `self._rcvd_idx` 对应的数据并处理后返回
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            # 断言不处于关闭状态且仍有待处理的任务
            assert not self._shutdown and self._tasks_outstanding > 0
            # 获取数据索引和数据本身
            idx, data = self._get_data()
            # 减少待处理的任务数量
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # 检查是否为 _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        # 如果使用持久化工作进程，则将相应工作进程标记为不可用
                        self._workers_status[data.worker_id] = False
                    else:
                        # 否则标记工作进程为不可用
                        self._mark_worker_as_unavailable(data.worker_id)
                    # 尝试放入索引并继续下一次循环
                    self._try_put_index()
                    continue

            # 如果索引不等于 `self._rcvd_idx`，则存储无序样本
            if idx != self._rcvd_idx:
                self._task_info[idx] += (data,)
            else:
                # 否则删除 `self._rcvd_idx` 对应的任务信息并处理数据后返回
                del self._task_info[idx]
                return self._process_data(data)
    # 检查是否还有未处理的任务数小于预取因子乘以工作进程数
    assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

    # 尝试获取下一个数据索引
    try:
        index = self._next_index()
    except StopIteration:
        # 如果迭代器已经结束，则直接返回
        return
    
    # 寻找下一个活跃的工作进程，如果有的话
    for _ in range(self._num_workers):
        worker_queue_idx = next(self._worker_queue_idx_cycle)
        if self._workers_status[worker_queue_idx]:
            break
    else:
        # 如果没有找到活跃的工作进程，则直接返回
        return

    # 将数据索引和数据索引放入工作进程的索引队列中
    self._index_queues[worker_queue_idx].put((self._send_idx, index))  # type: ignore[possibly-undefined]
    # 记录任务信息
    self._task_info[self._send_idx] = (worker_queue_idx,)
    # 增加未完成任务计数
    self._tasks_outstanding += 1
    # 更新发送索引
    self._send_idx += 1

def _process_data(self, data):
    # 增加已接收数据索引计数
    self._rcvd_idx += 1
    # 尝试将数据索引放入索引队列
    self._try_put_index()
    # 如果数据是异常封装对象，则重新引发异常
    if isinstance(data, ExceptionWrapper):
        data.reraise()
    # 返回数据
    return data

def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
    # 标记工作进程已经完成工作，例如由于耗尽了可迭代数据集
    # 这应该仅在 `_MultiProcessingDataLoaderIter` 继续运行时使用

    # 断言工作进程仍处于活跃状态，或者持久工作进程并且正在关闭
    assert self._workers_status[worker_id] or (
        self._persistent_workers and shutdown
    )

    # 向特定工作进程的索引队列发送终止信号
    q = self._index_queues[worker_id]
    q.put(None)

    # 更新工作进程状态为不可用
    self._workers_status[worker_id] = False

    # 断言工作完成事件的状态是否与关闭状态匹配
    assert self._workers_done_event.is_set() == shutdown

# 使用 staticmethod 移除对 `_MultiProcessingDataLoaderIter` 的引用
@staticmethod
def _clean_up_worker(w):
    # 尝试等待工作进程结束
    try:
        w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
    finally:
        # 如果工作进程仍在运行，则强制终止
        if w.is_alive():
            w.terminate()

def __del__(self):
    # 关闭所有工作进程
    self._shutdown_workers()
```