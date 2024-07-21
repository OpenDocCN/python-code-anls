# `.\pytorch\torch\utils\data\distributed.py`

```
import math  # 导入math库，用于数学运算
from typing import Iterator, Optional, TypeVar  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式通信模块
from torch.utils.data.dataset import Dataset  # 导入PyTorch数据集基类
from torch.utils.data.sampler import Sampler  # 导入PyTorch采样器基类


__all__ = ["DistributedSampler"]  # 定义模块的公开接口

T_co = TypeVar("T_co", covariant=True)  # 定义一个协变类型变量


class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        # 初始化方法，创建一个分布式采样器对象

        # 调用父类的初始化方法
        super().__init__()

        self.dataset = dataset  # 保存数据集对象的引用
        self.num_replicas = num_replicas or dist.get_world_size()  # 设置参与分布式训练的进程数，默认获取当前分布式组的进程数
        self.rank = rank or dist.get_rank()  # 设置当前进程在分布式训练中的排名，默认获取当前进程的排名
        self.shuffle = shuffle  # 是否对数据进行洗牌的标志位
        self.seed = seed  # 用于洗牌的随机种子
        self.drop_last = drop_last  # 是否丢弃尾部数据以使数据均匀分布的标志位

    def __iter__(self) -> Iterator[T_co]:
        # 返回一个数据迭代器的方法，迭代器产生数据的顺序由此方法控制
        indices = list(range(len(self.dataset)))  # 创建索引列表，长度为数据集大小

        if self.shuffle:
            # 如果需要洗牌，使用给定的随机种子洗牌索引列表
            torch.manual_seed(self.seed)
            indices = torch.randperm(len(indices)).tolist()

        # 计算每个进程的数据切片范围
        num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas

        if total_size != len(indices):
            # 如果数据不能被完全整除，则根据drop_last标志位进行处理
            if self.drop_last:
                indices = indices[:total_size]
            else:
                # 增加索引，使数据均匀分布到所有进程
                indices += indices[: (total_size - len(indices))]

        # 计算当前进程的数据切片
        offset = num_samples * self.rank
        indices = indices[offset:offset + num_samples]

        return iter(indices)  # 返回迭代器，产生当前进程负责的数据索引

    def __len__(self) -> int:
        # 返回数据集的长度
        return len(self.dataset)
    ) -> None:
        # 如果未指定副本数，则检查是否支持分布式计算，获取全局副本数
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        # 如果未指定当前进程的排名，则检查是否支持分布式计算，获取当前进程的排名
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        # 检查当前进程的排名是否有效
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        # 初始化Sampler对象的属性
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # 如果设置了drop_last并且数据集长度不能被副本数整除，则需要调整样本数目以保证数据均匀分配
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # 计算最接近且可被整除的数据集长度
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            # 计算数据集每个副本的样本数目（向上取整）
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        # 计算数据集在所有副本中的总体大小
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        # 如果需要shuffle数据集
        if self.shuffle:
            # 根据epoch和seed确定性地进行shuffle
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            # 创建顺序索引列表
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # 如果不需要丢弃尾部数据
        if not self.drop_last:
            # 添加额外的样本以使其能够被均匀地分割
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # 移除尾部数据以使其能够被均匀地分割
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # 对索引进行子采样，每个进程只处理属于自己排名的样本
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        # 返回Sampler的样本数目
        return self.num_samples
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        当 `shuffle=True` 时，确保每个副本在每个 epoch 使用不同的随机顺序。否则，该采样器的下一个迭代将产生相同的顺序。

        Args:
            epoch (int): Epoch 数字。
        """
        self.epoch = epoch
```