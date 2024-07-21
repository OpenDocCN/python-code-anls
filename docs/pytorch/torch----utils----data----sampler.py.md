# `.\pytorch\torch\utils\data\sampler.py`

```
# mypy: allow-untyped-defs
# 引入需要的类型定义模块
from typing import (
    Generic,         # 泛型
    Iterable,        # 可迭代对象
    Iterator,        # 迭代器
    List,            # 列表
    Optional,        # 可选类型
    Sequence,        # 序列
    Sized,           # 可以获取长度的对象
    TypeVar,         # 类型变量
    Union,           # 联合类型
)

import torch  # 引入PyTorch模块


__all__ = [  # 导出的模块成员列表
    "BatchSampler",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "SubsetRandomSampler",
    "WeightedRandomSampler",
]

T_co = TypeVar("T_co", covariant=True)  # 定义一个协变的类型变量


class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements,
    and may provide a :meth:`__len__` method that returns the length of the returned iterators.

    Args:
        data_source (Dataset): This argument is not used and will be removed in 2.2.0.
            You may still have custom implementation that utilizes it.

    Example:
        >>> # xdoctest: +SKIP
        >>> class AccedingSequenceLengthSampler(Sampler[int]):
        >>>     def __init__(self, data: List[str]) -> None:
        >>>         self.data = data
        >>>
        >>>     def __len__(self) -> int:
        >>>         return len(self.data)
        >>>
        >>>     def __iter__(self) -> Iterator[int]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         yield from torch.argsort(sizes).tolist()
        >>>
        >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
        >>>     def __init__(self, data: List[str], batch_size: int) -> None:
        >>>         self.data = data
        >>>         self.batch_size = batch_size
        >>>
        >>>     def __len__(self) -> int:
        >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
        >>>
        >>>     def __iter__(self) -> Iterator[List[int]]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
        >>>             yield batch.tolist()

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized] = None) -> None:
        if data_source is not None:
            import warnings

            # 警告，指出`data_source`参数在2.2.0版本将被移除
            warnings.warn(
                "`data_source` argument is not used and will be removed in 2.2.0."
                "You may still have custom implementation that utilizes it."
            )

    def __iter__(self) -> Iterator[T_co]:
        # 抽象方法，子类必须实现，用于返回一个迭代器
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # 提供一个默认的实现，因为两种直接的默认实现都有问题：
    #
    #   + `return NotImplemented`:
    #     调用 `len(subclass_instance)` 会引发:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError`:
    #     这样做会阻止触发某些回退行为。例如，内置的 `list(X)` 首先尝试调用 `len(X)`，
    #     如果找不到该方法或者返回 `NotImplemented`，则会执行不同的代码路径，
    #     而引发 `NotImplementedError` 会传播并使调用失败，而本可以使用 `__iter__` 完成调用。
    #
    # 因此，唯一合理的做法是
    #
    #   + **不** 提供默认的 `__len__`。
    #
    #   + 抛出 `TypeError` 异常，这是当用户调用对象上未定义的方法时Python使用的方式。
    #     (@ssnl 验证至少在 Python 3.7 上可以正常工作。)
class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source
        # 初始化方法，接受一个数据集作为参数，存储在实例变量 data_source 中

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))
        # 迭代器方法，返回一个迭代器，用于按顺序访问数据集中的索引

    def __len__(self) -> int:
        return len(self.data_source)
        # 返回数据集的长度，即其中元素的数量


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )
        # 检查 replacement 是否为布尔值，如果不是则抛出 TypeError 异常

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )
        # 检查 num_samples 是否为正整数，如果不是则抛出 ValueError 异常

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples
        # 返回采样的样本数量，如果未指定则返回数据集的长度
    # 返回一个迭代器，用于生成随机采样的索引
    def __iter__(self) -> Iterator[int]:
        # 获取数据集的长度
        n = len(self.data_source)
        
        # 如果生成器未指定，则随机生成一个种子并创建生成器对象
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            # 否则，使用指定的生成器对象
            generator = self.generator
        
        # 如果采样允许替换
        if self.replacement:
            # 生成 num_samples // 32 次采样，每次采样大小为 32
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            # 生成余下的采样，大小为 num_samples % 32
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:
            # 如果采样不允许替换，生成 num_samples // n 次完全随机的索引
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            # 生成余下的采样，大小为 num_samples % n，从完全随机的索引中截取
            yield from torch.randperm(n, generator=generator).tolist()[
                : self.num_samples % n
            ]

    # 返回数据集的样本数目
    def __len__(self) -> int:
        return self.num_samples
class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    indices: Sequence[int]  # 定义存储索引序列的属性

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices  # 初始化对象时保存传入的索引序列
        self.generator = generator  # 初始化对象时保存传入的生成器

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]  # 使用随机排列的索引序列迭代返回元素

    def __len__(self) -> int:
        return len(self.indices)  # 返回索引序列的长度


class WeightedRandomSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.

    Example:
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [4, 4, 1, 4, 5]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    weights: torch.Tensor  # 定义存储权重张量的属性
    num_samples: int  # 定义存储抽样数量的属性
    replacement: bool  # 定义存储是否使用替换抽样的属性

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator=None,
    ) -> None:
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            )  # 检查并抛出异常，确保 num_samples 是正整数

        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )  # 检查并抛出异常，确保 replacement 是布尔值

        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )  # 检查并抛出异常，确保 weights 是一维张量

        self.weights = weights_tensor  # 初始化对象时保存传入的权重张量
        self.num_samples = num_samples  # 初始化对象时保存传入的抽样数量
        self.replacement = replacement  # 初始化对象时保存传入的是否使用替换抽样
        self.generator = generator  # 初始化对象时保存传入的生成器

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )  # 使用给定的权重进行多项式抽样
        yield from iter(rand_tensor.tolist())  # 返回抽样结果的迭代器

    def __len__(self) -> int:
        return self.num_samples  # 返回抽样数量作为对象的长度
class BatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        # 检查 batch_size 是否为正整数，否则抛出异常
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        # 检查 drop_last 是否为布尔值，否则抛出异常
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        # 初始化属性
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # 基于 https://github.com/pytorch/pytorch/pull/76951 的性能基准实现
        if self.drop_last:
            # 如果 drop_last 为 True，循环生成批次直到结束
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    # 尝试生成一个批次的索引
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            # 如果 drop_last 为 False，迭代生成所有可能的批次
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                # 将索引添加到当前批次中
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # 只有在 self.sampler 实现了 __len__ 方法时才能调用
        # 无法强制执行这个条件，因此关闭类型检查来实现下面的实现。
        # 类似相关注释：参见注释 [ Python 抽象基类中缺少默认的 `__len__` 方法 ]
        if self.drop_last:
            # 如果 drop_last 为 True，返回批次的数量
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            # 如果 drop_last 为 False，返回能够容纳所有元素的批次数量
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
```