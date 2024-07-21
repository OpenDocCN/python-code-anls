# `.\pytorch\torch\utils\data\dataset.py`

```
# 引入必要的模块和类
# mypy: allow-untyped-defs 允许未类型化的定义
import bisect  # 导入 bisect 模块，提供对有序序列的插入和查找操作
import itertools  # 导入 itertools 模块，提供高效的迭代工具
import math  # 导入 math 模块，提供数学函数
import warnings  # 导入 warnings 模块，用于处理警告

# 导入类型提示相关模块和类
from typing import (
    cast,  # 强制类型转换函数
    Dict,  # 字典类型
    Generic,  # 泛型类型
    Iterable,  # 可迭代类型
    List,  # 列表类型
    Optional,  # 可选类型
    Sequence,  # 序列类型
    Tuple,  # 元组类型
    TypeVar,  # 类型变量
    Union,  # 联合类型
)
from typing_extensions import deprecated  # 导入 deprecated 类型扩展

# 从 torch 模块中导入以下类型和函数
# 注意：torch/__init__.pyi 中没有 default_generator
from torch import default_generator, Generator, randperm, Tensor  # 导入默认生成器、随机排列、张量类型

# 定义 __all__，指定模块导出的公共接口
__all__ = [
    "Dataset",  # 数据集基类
    "IterableDataset",  # 可迭代数据集基类
    "TensorDataset",  # 张量数据集类
    "StackDataset",  # 堆叠数据集类
    "ConcatDataset",  # 连接数据集类
    "ChainDataset",  # 链式数据集类
    "Subset",  # 子集类
    "random_split",  # 随机分割函数
]

T_co = TypeVar("T_co", covariant=True)  # 协变类型变量 T_co
T = TypeVar("T")  # 通用类型变量 T
T_dict = Dict[str, T_co]  # 字典类型变量，键为字符串，值为协变类型变量 T_co
T_tuple = Tuple[T_co, ...]  # 元组类型变量，元素为协变类型变量 T_co 的可变长度元组
T_stack = TypeVar("T_stack", T_tuple, T_dict)  # 栈类型变量，可以是元组类型变量或字典类型变量

class Dataset(Generic[T_co]):
    r"""表示数据集的抽象类。

    所有表示从键到数据样本的映射的数据集都应该是它的子类。
    所有子类应该覆盖 :meth:`__getitem__` 方法，支持根据给定键获取数据样本。
    子类也可以选择覆盖 :meth:`__len__` 方法，它将被许多 :class:`~torch.utils.data.Sampler`
    实现和 :class:`~torch.utils.data.DataLoader` 的默认选项所使用。
    子类也可以选择实现 :meth:`__getitems__` 方法，以加速批量样本加载。
    该方法接受样本批次的索引列表，并返回样本列表。

    .. note::
      :class:`~torch.utils.data.DataLoader` 默认构造一个索引采样器，产生整数索引。
      要使其适用于具有非整数索引/键的映射样式数据集，必须提供自定义采样器。
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Dataset 的子类应该实现 __getitem__ 方法。")

    def __add__(self, other: "Dataset[T_co]") -> "ConcatDataset[T_co]":
        return ConcatDataset([self, other])
    # 重载运算符 +，返回连接数据集的拼接数据集对象

class IterableDataset(Dataset[T_co], Iterable[T_co]):
    r"""表示可迭代数据集的类。

    所有表示数据样本可迭代集合的数据集都应该是它的子类。
    当数据来自流时，这种数据集形式尤其有用。

    所有子类应该覆盖 :meth:`__iter__` 方法，它将返回此数据集中样本的迭代器。

    当子类与 :class:`~torch.utils.data.DataLoader` 一起使用时，数据集中的每个项目
    将从 :class:`~torch.utils.data.DataLoader` 迭代器中产生。
    当 :attr:`num_workers > 0` 时，每个工作进程将有数据集对象的不同副本，
    因此通常希望配置
        each copy independently to avoid having duplicate data returned from the
        workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
        process, returns information about the worker. It can be used in either the
        dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
        :attr:`worker_init_fn` option to modify each copy's behavior.


        # 每个副本独立工作，避免从工作进程返回重复数据。
        # :func:`~torch.utils.data.get_worker_info` 在工作进程中调用时返回有关工作进程的信息。
        # 可以在数据集的 :meth:`__iter__` 方法或 :class:`~torch.utils.data.DataLoader` 的
        # :attr:`worker_init_fn` 选项中使用它来修改每个副本的行为。


        Example 1: splitting workload across all workers in :meth:`__iter__`::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_DATALOADER)
            >>> # xdoctest: +SKIP("Fails on MacOS12")
            >>> class MyIterableDataset(torch.utils.data.IterableDataset):
            ...     def __init__(self, start, end):
            ...         super(MyIterableDataset).__init__()
            ...         assert end > start, "this example code only works with end >= start"
            ...         self.start = start
            ...         self.end = end
            ...
            ...     def __iter__(self):
            ...         worker_info = torch.utils.data.get_worker_info()
            ...         if worker_info is None:  # single-process data loading, return the full iterator
            ...             iter_start = self.start
            ...             iter_end = self.end
            ...         else:  # in a worker process
            ...             # split workload
            ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            ...             worker_id = worker_info.id
            ...             iter_start = self.start + worker_id * per_worker
            ...             iter_end = min(iter_start + per_worker, self.end)
            ...         return iter(range(iter_start, iter_end))
            ...
            >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
            >>> ds = MyIterableDataset(start=3, end=7)


            >>> # Single-process loading
            >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
            [tensor([3]), tensor([4]), tensor([5]), tensor([6])]


            >>> # xdoctest: +REQUIRES(POSIX)
            >>> # Mult-process loading with two worker processes
            >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
            >>> # xdoctest: +IGNORE_WANT("non deterministic")
            >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
            [tensor([3]), tensor([5]), tensor([4]), tensor([6])]


            >>> # With even more workers
            >>> # xdoctest: +IGNORE_WANT("non deterministic")
            >>> print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
            [tensor([3]), tensor([5]), tensor([4]), tensor([6])]
    def __add__(self, other: Dataset[T_co]):
        # 定义一个特殊方法 `__add__`，用于实现数据集的连接操作
        return ChainDataset([self, other])

    # 如果没有定义 `def __len__(self)`，子类在需要时会引发 `TypeError`。
    # 参见注释 [ Python 抽象基类中缺乏默认 `__len__` 的说明 ]
# 定义一个继承自 Dataset 的 TensorDataset 类，用于包装多个张量数据集
class TensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    tensors: Tuple[Tensor, ...]  # 声明类属性 tensors，存储多个张量数据集

    # 初始化方法，接收多个张量作为参数
    def __init__(self, *tensors: Tensor) -> None:
        # 断言所有张量的第一个维度大小相同
        assert all(
            tensors[0].size(0) == tensor.size(0) for tensor in tensors
        ), "Size mismatch between tensors"
        self.tensors = tensors  # 将参数中的张量赋值给类属性 tensors

    # 获取指定索引处的数据样本
    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    # 返回数据集的长度，即第一个张量的第一维度大小
    def __len__(self):
        return self.tensors[0].size(0)


# 定义一个继承自 Dataset 的 StackDataset 类，用于堆叠多个数据集
class StackDataset(Dataset[T_stack]):
    r"""Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {'image': images[0], 'text': texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """

    datasets: Union[tuple, dict]  # 声明类属性 datasets，存储多个数据集

    # 初始化方法，接收作为元组传递的数据集或作为字典传递的数据集
    def __init__(self, *args: Dataset[T_co], **kwargs: Dataset[T_co]) -> None:
        if args:  # 如果有元组类型的参数
            if kwargs:  # 如果同时有字典类型的参数，抛出错误
                raise ValueError(
                    "Supported either ``tuple``- (via ``args``) or"
                    "``dict``- (via ``kwargs``) like input/output, but both types are given."
                )
            self._length = len(args[0])  # 获取第一个数据集的长度
            # 检查所有数据集的长度是否一致
            if any(self._length != len(dataset) for dataset in args):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = args  # 将元组数据集赋值给类属性 datasets
        elif kwargs:  # 如果有字典类型的参数
            tmp = list(kwargs.values())  # 将字典的值转换为列表
            self._length = len(tmp[0])  # 获取第一个数据集的长度
            # 检查所有数据集的长度是否一致
            if any(self._length != len(dataset) for dataset in tmp):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = kwargs  # 将字典数据集赋值给类属性 datasets
        else:  # 如果没有传入任何数据集参数，抛出错误
            raise ValueError("At least one dataset should be passed")

    # 获取指定索引处的数据样本
    def __getitem__(self, index):
        if isinstance(self.datasets, dict):  # 如果 datasets 是字典类型
            return {k: dataset[index] for k, dataset in self.datasets.items()}  # 返回字典形式的数据样本
        return tuple(dataset[index] for dataset in self.datasets)  # 返回元组形式的数据样本
    # 实现特殊方法 `__getitems__`，支持按索引列表批量采样数据集
    def __getitems__(self, indices: list):
        # 如果数据集是字典形式，支持批量采样
        if isinstance(self.datasets, dict):
            # 初始化一个空列表，用于存储每个数据集的批量采样结果
            dict_batch: List[T_dict] = [{} for _ in indices]
            # 遍历字典中的每个数据集
            for k, dataset in self.datasets.items():
                # 检查数据集是否支持 `__getitems__` 方法
                if callable(getattr(dataset, "__getitems__", None)):
                    # 调用数据集的 `__getitems__` 方法进行批量采样
                    items = dataset.__getitems__(indices)  # type: ignore[attr-defined]
                    # 检查采样结果的长度是否与索引列表一致
                    if len(items) != len(indices):
                        raise ValueError(
                            "Nested dataset's output size mismatch."
                            f" Expected {len(indices)}, got {len(items)}"
                        )
                    # 将每个数据集的采样结果按键值存入对应的字典中
                    for data, d_sample in zip(items, dict_batch):
                        d_sample[k] = data
                else:
                    # 如果数据集不支持 `__getitems__` 方法，则按索引逐个获取数据
                    for idx, d_sample in zip(indices, dict_batch):
                        d_sample[k] = dataset[idx]
            return dict_batch
        
        # 如果数据集是元组形式，支持批量采样
        list_batch: List[list] = [[] for _ in indices]
        for dataset in self.datasets:
            # 检查数据集是否支持 `__getitems__` 方法
            if callable(getattr(dataset, "__getitems__", None)):
                # 调用数据集的 `__getitems__` 方法进行批量采样
                items = dataset.__getitems__(indices)  # type: ignore[attr-defined]
                # 检查采样结果的长度是否与索引列表一致
                if len(items) != len(indices):
                    raise ValueError(
                        "Nested dataset's output size mismatch."
                        f" Expected {len(indices)}, got {len(items)}"
                    )
                # 将每个数据集的采样结果逐个添加到对应的列表中
                for data, t_sample in zip(items, list_batch):
                    t_sample.append(data)
            else:
                # 如果数据集不支持 `__getitems__` 方法，则按索引逐个获取数据
                for idx, t_sample in zip(indices, list_batch):
                    t_sample.append(dataset[idx])
        
        # 将列表批量采样结果转换为元组形式返回
        tuple_batch: List[T_tuple] = [tuple(sample) for sample in list_batch]
        return tuple_batch

    # 返回数据集的长度，由 `_length` 属性确定
    def __len__(self):
        return self._length
# 定义一个类 `ConcatDataset`，用于将多个数据集连接在一起的数据集类
class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    # datasets 属性存储了要连接的数据集列表
    datasets: List[Dataset[T_co]]
    # cumulative_sizes 存储了每个数据集累积的大小
    cumulative_sizes: List[int]

    # 静态方法 cumsum 用于计算序列的累积和
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    # 初始化方法，接受一个可迭代的数据集列表 datasets
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        # 将 datasets 转换为列表存储在 self.datasets 中
        self.datasets = list(datasets)
        # 断言 datasets 列表不为空
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]
        # 检查每个数据集是否为 IterableDataset，如果是，则抛出异常
        for d in self.datasets:
            assert not isinstance(
                d, IterableDataset
            ), "ConcatDataset does not support IterableDataset"
        # 计算每个数据集的累积大小
        self.cumulative_sizes = self.cumsum(self.datasets)

    # 返回 ConcatDataset 的总长度，即所有数据集的总和
    def __len__(self):
        return self.cumulative_sizes[-1]

    # 根据索引获取数据集中的元素
    def __getitem__(self, idx):
        # 处理负索引情况
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        # 使用二分查找找到对应的数据集索引
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        # 返回数据集中的具体样本
        return self.datasets[dataset_idx][sample_idx]

    # 属性装饰器，用于返回累积大小列表，同时给出未来警告
    @property
    @deprecated(
        "`cummulative_sizes` attribute is renamed to `cumulative_sizes`",
        category=FutureWarning,
    )
    def cummulative_sizes(self):
        return self.cumulative_sizes


# 定义一个类 `ChainDataset`，用于连接多个 IterableDataset 的数据集类
class ChainDataset(IterableDataset):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """

    # 初始化方法，接受一个 IterableDataset 的可迭代对象 datasets
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        # 将 datasets 直接存储在 self.datasets 中
        self.datasets = datasets

    # 实现迭代器方法，用于遍历所有 datasets 中的元素
    def __iter__(self):
        # 断言每个数据集都是 IterableDataset 类型，如果不是则抛出异常
        for d in self.datasets:
            assert isinstance(
                d, IterableDataset
            ), "ChainDataset only supports IterableDataset"
            # 使用 yield from 语法遍历数据集 d 中的所有元素
            yield from d

    # 返回 ChainDataset 的总长度，即所有数据集的总和
    def __len__(self):
        total = 0
        # 遍历所有 datasets 计算总长度
        for d in self.datasets:
            assert isinstance(
                d, IterableDataset
            ), "ChainDataset only supports IterableDataset"
            total += len(d)  # type: ignore[arg-type]
        return total


# 定义一个类 `Subset`，用于表示数据集的子集，包含指定索引的数据
class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        # 初始化函数，接受一个数据集和索引序列作为参数
        self.dataset = dataset  # 将数据集保存到实例变量中
        self.indices = indices  # 将索引序列保存到实例变量中

    def __getitem__(self, idx):
        # 获取指定索引处的数据项
        if isinstance(idx, list):
            # 如果索引是一个列表，则返回对应的批量数据
            return self.dataset[[self.indices[i] for i in idx]]
        # 否则返回单个数据项
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        # 添加批量采样支持，当父数据集支持时。
        # 参见 torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            # 如果父数据集具有 "__getitems__" 方法，则调用该方法进行批量采样
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            # 否则逐个获取指定索引处的数据项
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        # 返回索引序列的长度，即数据集中的样本数量
        return len(self.indices)
# 随机将数据集分割成给定长度的子数据集列表
def random_split(
    dataset: Dataset[T],
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # 检查长度列表是否为分数并且总和为1，如果是则将其转换为整数长度列表
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # 将剩余的数据项数按照轮转方式添加到长度列表中，直到没有剩余项
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        # 对于长度为0的分割，发出警告，可能导致空数据集
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # 检查分割后的数据集长度总和是否与原数据集长度相等
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    # 使用指定的生成器生成随机排列的索引
    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    lengths = cast(Sequence[int], lengths)
    # 根据计算的长度列表创建并返回子数据集列表
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]
```