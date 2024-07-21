# `.\pytorch\torch\utils\data\datapipes\iter\combinatorics.py`

```
# mypy: allow-untyped-defs
import random  # 导入随机数模块
from typing import Dict, Iterator, List, Optional, Sized, Tuple, Type, TypeVar  # 导入类型提示相关模块

import torch  # 导入PyTorch模块
from torch.utils.data.datapipes._decorator import functional_datapipe  # 导入数据管道装饰器
from torch.utils.data.datapipes.datapipe import IterDataPipe  # 导入数据管道基类
from torch.utils.data.sampler import Sampler, SequentialSampler  # 导入样本抽样器类和顺序抽样器类


__all__ = [
    "SamplerIterDataPipe",  # 将类SamplerIterDataPipe添加到模块导出列表中
    "ShufflerIterDataPipe",  # 将类ShufflerIterDataPipe添加到模块导出列表中
]

T_co = TypeVar("T_co", covariant=True)  # 定义一个协变类型变量


class SamplerIterDataPipe(IterDataPipe[T_co]):
    r"""
    Generate sample elements using the provided ``Sampler`` (defaults to :class:`SequentialSampler`).

    Args:
        datapipe: IterDataPipe to sample from  # 从中进行抽样的IterDataPipe
        sampler: Sampler class to generate sample elements from input DataPipe.  # 用于从输入数据管道生成样本元素的抽样器类，默认为SequentialSampler
            Default is :class:`SequentialSampler` for IterDataPipe
    """

    datapipe: IterDataPipe  # 数据管道对象
    sampler: Sampler  # 抽样器对象

    def __init__(
        self,
        datapipe: IterDataPipe,
        sampler: Type[Sampler] = SequentialSampler,  # 默认使用顺序抽样器
        sampler_args: Optional[Tuple] = None,  # 抽样器的位置参数元组（可选）
        sampler_kwargs: Optional[Dict] = None,  # 抽样器的关键字参数字典（可选）
    ) -> None:
        assert isinstance(
            datapipe, Sized
        ), "Sampler class requires input datapipe implemented `__len__`"  # 断言输入的数据管道对象实现了__len__方法
        super().__init__()  # 调用父类的初始化方法
        self.datapipe = datapipe  # 初始化数据管道对象
        self.sampler_args = () if sampler_args is None else sampler_args  # 初始化抽样器的位置参数元组
        self.sampler_kwargs = {} if sampler_kwargs is None else sampler_kwargs  # 初始化抽样器的关键字参数字典
        # 使用指定的抽样器类及参数创建抽样器对象，将数据管道作为数据源
        self.sampler = sampler(*self.sampler_args, data_source=self.datapipe, **self.sampler_kwargs)  # type: ignore[misc]

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.sampler)  # 返回抽样器对象的迭代器

    def __len__(self) -> int:
        # 断言抽样器对象实现了Sized接口
        if isinstance(self.sampler, Sized):
            return len(self.sampler)  # 返回抽样器对象的长度
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")  # 如果抽样器对象不是Sized类型，则抛出类型错误


@functional_datapipe("shuffle")
class ShufflerIterDataPipe(IterDataPipe[T_co]):
    r"""
    Shuffle the input DataPipe with a buffer (functional name: ``shuffle``).

    The buffer with ``buffer_size`` is filled with elements from the datapipe first. Then,
    each item will be yielded from the buffer by reservoir sampling via iterator.

    ``buffer_size`` is required to be larger than ``0``. For ``buffer_size == 1``, the
    datapipe is not shuffled. In order to fully shuffle all elements from datapipe,
    ``buffer_size`` is required to be greater than or equal to the size of datapipe.

    When it is used with :class:`torch.utils.data.DataLoader`, the methods to
    set up random seed are different based on :attr:`num_workers`.

    For single-process mode (:attr:`num_workers == 0`), the random seed is set before
    the :class:`~torch.utils.data.DataLoader` in the main process. For multi-process
    mode (:attr:`num_worker > 0`), `worker_init_fn` is used to set up a random seed
    for each worker process.
    """
    pass  # 类定义结束，暂无额外的代码实现
    # 输入参数包括待洗牌的 IterDataPipe，缓冲区大小和解包级别，默认缓冲区大小为 10000
    datapipe: IterDataPipe[T_co]
    buffer_size: int
    _buffer: List[T_co]  # 内部缓冲区，用于存储洗牌过程中的数据
    _enabled: bool  # 指示洗牌是否启用的标志
    _seed: Optional[int]  # 可选的随机种子值
    _rng: random.Random  # 随机数生成器对象

    def __init__(
        self,
        datapipe: IterDataPipe[T_co],
        *,
        buffer_size: int = 10000,
        unbatch_level: int = 0,
    ) -> None:
        super().__init__()
        # TODO: 性能优化
        #       缓冲区可以是固定大小，避免昂贵的 `append()` 和 `len()` 操作
        self._buffer: List[T_co] = []  # 初始化空的内部缓冲区
        assert buffer_size > 0, "buffer_size should be larger than 0"  # 断言缓冲区大小应大于0
        if unbatch_level == 0:
            self.datapipe = datapipe  # 如果解包级别为0，则直接使用原始 datapipe
        else:
            self.datapipe = datapipe.unbatch(unbatch_level=unbatch_level)  # 否则对 datapipe 进行解包
        self.buffer_size = buffer_size  # 设置缓冲区大小
        self._enabled = True  # 默认启用洗牌功能
        self._seed = None  # 初始化随机种子为 None
        self._rng = random.Random()  # 创建随机数生成器对象

    def set_shuffle(self, shuffle=True):
        self._enabled = shuffle  # 设置是否启用洗牌功能的标志
        return self

    def set_seed(self, seed: int):
        self._seed = seed  # 设置随机种子值
        return self

    def __iter__(self) -> Iterator[T_co]:
        if not self._enabled:
            yield from self.datapipe  # 如果未启用洗牌，则直接迭代原始 datapipe
        else:
            for x in self.datapipe:
                if len(self._buffer) == self.buffer_size:
                    idx = self._rng.randint(0, len(self._buffer) - 1)  # 随机选择缓冲区中的索引
                    val, self._buffer[idx] = self._buffer[idx], x  # 交换缓冲区中的值和当前值
                    yield val  # 返回被交换的值
                else:
                    self._buffer.append(x)  # 将当前值添加到缓冲区
            while self._buffer:
                idx = self._rng.randint(0, len(self._buffer) - 1)  # 随机选择缓冲区中的索引
                yield self._buffer.pop(idx)  # 弹出并返回随机选择的值

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            return len(self.datapipe)  # 返回原始 datapipe 的长度
        raise TypeError(f"{type(self).__name__} instance doesn't have valid length")  # 抛出异常，表明实例无有效长度

    def reset(self) -> None:
        self._buffer = []  # 重置内部缓冲区
        if self._enabled:
            if self._seed is None:
                self._seed = int(torch.empty((), dtype=torch.int64).random_().item())  # 生成随机种子
            self._rng.seed(self._seed)  # 使用随机种子初始化随机数生成器
            self._seed = None  # 重置随机种子为 None
    # 定义特殊方法 __getstate__，用于获取对象的序列化状态
    def __getstate__(self):
        # 将对象的状态保存到元组 state 中，包括数据管道、缓冲区大小、启用状态、种子等
        state = (
            self.datapipe,
            self.buffer_size,
            self._enabled,
            self._seed,
            self._buffer,
            self._rng.getstate(),  # 获取随机数生成器的状态
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        # 如果定义了 getstate_hook 方法，则调用它处理 state 并返回处理后的结果
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        # 否则直接返回 state
        return state

    # 定义特殊方法 __setstate__，用于从序列化状态恢复对象
    def __setstate__(self, state):
        # 从 state 中恢复对象的状态，包括数据管道、缓冲区大小、启用状态、种子等
        (
            self.datapipe,
            self.buffer_size,
            self._enabled,
            self._seed,
            self._buffer,
            rng_state,  # 获取随机数生成器的状态
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        # 初始化一个新的随机数生成器对象
        self._rng = random.Random()
        # 设置随机数生成器的状态为从 state 中恢复的 rng_state
        self._rng.setstate(rng_state)

    # 定义特殊方法 __del__，用于对象销毁时的处理
    def __del__(self):
        # 清空对象的缓冲区
        self._buffer.clear()
```