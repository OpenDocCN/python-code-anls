# `.\pytorch\torch\utils\data\datapipes\map\combinatorics.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类型
import random  # 导入随机数生成模块
from typing import Iterator, List, Optional, TypeVar  # 导入类型提示相关模块

import torch  # 导入 PyTorch 模块
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe  # 导入数据管道相关类


__all__ = ["ShufflerIterDataPipe"]  # 指定模块导出的内容


T_co = TypeVar("T_co", covariant=True)  # 定义一个协变类型变量


# @functional_datapipe('shuffle')
class ShufflerIterDataPipe(IterDataPipe[T_co]):
    r"""
    Shuffle the input MapDataPipe via its indices (functional name: ``shuffle``).

    When it is used with :class:`~torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), ``worker_init_fn`` is used to set up a random seed
    for each worker process.

    Args:
        datapipe: MapDataPipe being shuffled
        indices: a list of indices of the MapDataPipe. If not provided, we assume it uses 0-based indexing

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> shuffle_dp = dp.shuffle().set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]
        >>> list(shuffle_dp)
        [6, 1, 9, 5, 2, 4, 7, 3, 8, 0]
        >>> # Reset seed for Shuffler
        >>> shuffle_dp = shuffle_dp.set_seed(0)
        >>> list(shuffle_dp)
        [7, 8, 1, 5, 3, 4, 2, 0, 9, 6]

    Note:
        Even thought this ``shuffle`` operation takes a ``MapDataPipe`` as the input, it would return an
        ``IterDataPipe`` rather than a ``MapDataPipe``, because ``MapDataPipe`` should be non-sensitive to
        the order of data order for the sake of random reads, but ``IterDataPipe`` depends on the order
        of data during data-processing.
    """

    datapipe: MapDataPipe[T_co]  # 类型注解，指定datapipe为MapDataPipe类型的数据管道
    _enabled: bool  # 标记是否启用洗牌功能的布尔变量
    _seed: Optional[int]  # 可选的种子值，用于随机数生成器的种子
    _rng: random.Random  # 随机数生成器对象

    def __init__(
        self,
        datapipe: MapDataPipe[T_co],
        *,
        indices: Optional[List] = None,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe  # 初始化数据管道对象
        self.indices = list(range(len(datapipe))) if indices is None else indices  # 初始化索引列表
        self._enabled = True  # 默认启用洗牌功能
        self._seed = None  # 初始种子设为None
        self._rng = random.Random()  # 初始化随机数生成器对象
        self._shuffled_indices: List = self.indices  # 初始化洗牌后的索引列表为初始索引列表的副本

    def set_shuffle(self, shuffle=True):
        # 设置是否启用洗牌功能
        self._enabled = shuffle
        return self

    def set_seed(self, seed: int):
        # 设置随机数生成器的种子值
        self._seed = seed
        return self

    def __iter__(self) -> Iterator[T_co]:
        if not self._enabled:
            # 如果未启用洗牌功能，则按照原顺序迭代数据管道中的数据
            for idx in self.indices:
                yield self.datapipe[idx]
        else:
            # 如果启用洗牌功能，则按照洗牌后的顺序迭代数据管道中的数据
            while self._shuffled_indices:
                idx = self._shuffled_indices.pop()  # 弹出洗牌后的索引列表中的最后一个索引
                yield self.datapipe[idx]
    # 重置数据管道对象的状态
    def reset(self) -> None:
        # 如果启用状态且种子为空，则生成一个随机种子
        if self._enabled and self._seed is None:
            self._seed = int(torch.empty((), dtype=torch.int64).random_().item())
        # 使用当前种子重新设置随机数生成器的种子
        self._rng.seed(self._seed)
        # 清除种子值
        self._seed = None
        # 对索引进行洗牌操作，生成洗牌后的索引列表
        self._shuffled_indices = self._rng.sample(self.indices, len(self.indices))

    # 返回数据管道对象中的元素数量
    def __len__(self) -> int:
        return len(self.datapipe)

    # 获取对象的序列化状态
    def __getstate__(self):
        # 返回当前对象的状态元组，包括数据管道、索引、启用状态、种子、随机数生成器状态、洗牌后的索引、有效迭代器ID、产生的样本数量
        state = (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            self._rng.getstate(),
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        # 如果定义了自定义的状态获取钩子函数，则调用它并返回结果
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        # 否则直接返回状态元组
        return state

    # 设置对象的序列化状态
    def __setstate__(self, state):
        # 解包传入的状态元组，依次赋值给对象的相关属性
        (
            self.datapipe,
            self.indices,
            self._enabled,
            self._seed,
            rng_state,
            self._shuffled_indices,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        # 初始化一个新的随机数生成器对象
        self._rng = random.Random()
        # 设置新的随机数生成器状态
        self._rng.setstate(rng_state)
# 将"shuffle"注册为数据管道函数，使用ShufflerIterDataPipe类来实现数据管道功能
MapDataPipe.register_datapipe_as_function("shuffle", ShufflerIterDataPipe)
```