# `.\pytorch\torch\distributed\_tensor\placement_types.py`

```
# 设置一个类型检查的标志，允许未标注的函数定义
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入必要的模块和类
from dataclasses import dataclass
from typing import Any, cast, List, NamedTuple, Optional, Tuple

import torch  # 导入PyTorch库
import torch.distributed._functional_collectives as funcol  # 导入PyTorch分布式相关功能
from torch.distributed._tensor._collective_utils import (  # 导入PyTorch分布式张量的集合工具
    fill_empty_tensor_to_shards,  # 填充空张量到分片的函数
    mesh_broadcast,  # 网格广播函数
    mesh_scatter,  # 网格分散函数
    pad_tensor,  # 张量填充函数
    shard_dim_alltoall,  # 维度分组全对全通信函数
    unpad_tensor,  # 张量去填充函数
)
from torch.distributed.device_mesh import DeviceMesh  # 导入设备网格类


class Placement:
    # 放置类的基类，描述了放置类型的基本特性

    # 便捷的工具函数，用于检查放置类型是否为分片
    def is_shard(self, dim: Optional[int] = None) -> bool:
        # 检查当前实例是否为Shard类型
        is_shard_instance = isinstance(self, Shard)
        # 如果指定了维度参数且当前实例是Shard类型，则比较维度是否相同并返回结果
        if dim is not None and is_shard_instance:
            return cast(Shard, self).dim == dim
        else:
            # 否则直接返回当前实例是否为Shard类型的结果
            return is_shard_instance

    # 检查放置类型是否为复制
    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    # 检查放置类型是否为部分放置
    def is_partial(self) -> bool:
        return isinstance(self, Partial)


@dataclass(frozen=True)
class Shard(Placement):
    """
    Shard(dim)放置描述了张量在对应的DeviceMesh维度上的分片，
    每个DeviceMesh维度上的每个排位(rank)只持有全局张量的一部分。
    Shard(dim)放置遵循torch.chunk(dim)的语义，其中DeviceMesh维度上的最后几个分片可能为空。
    Shard放置可以被所有的DTensor API使用（如distribute_tensor, from_local等）。

    Args:
        dim (int): 描述张量在对应的DeviceMesh维度上进行分片的张量维度。

    ::note:: 当在一个DeviceMesh维度上对张量维度进行分片时，如果不能整除，
        目前处于实验阶段。
    """

    dim: int  # 定义Shard放置的维度参数

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        This function uses torch.chunk to split a tensor into num_chunks shards along
        the Shard placement dimension, and return a list of shards with their pad sizes.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
            few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
            This is because collectives usually require equal size tensor inputs
        """
        assert (
            self.dim <= tensor.ndim
        ), f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"

        # chunk tensor over dimension `dim` into n slices
        tensor_list = list(torch.chunk(tensor, num_chunks, dim=self.dim))
        num_empty_tensors = num_chunks - len(tensor_list)

        # if no need to have padding or tensor dim size is evenly sharded already
        # we can return early.
        if not with_padding or tensor.size(self.dim) % num_chunks == 0:
            if contiguous:
                tensor_list = [t.contiguous() for t in tensor_list]
            return (
                fill_empty_tensor_to_shards(tensor_list, self.dim, num_empty_tensors),
                [],
            )

        # compute the chunk size inline with ``torch.chunk`` to calculate padding
        full_chunk_size = (tensor.size(self.dim) + num_chunks - 1) // num_chunks

        # Compute chunk size for each chunk for ``self.dim``
        chunk_sizes = [
            tensor_list[idx].size(self.dim) if idx < len(tensor_list) else 0
            for idx in range(num_chunks)
        ]
        # Compute pad size on each chunk
        pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]

        # Reuse tensor to fill empty chunk with empty tensor
        tensor_list = fill_empty_tensor_to_shards(
            tensor_list, self.dim, num_empty_tensors
        )
        shard_list = []
        for shard, pad_size in zip(tensor_list, pad_sizes):
            # Fill the empty tensor with zeroes with padding.
            if with_padding and pad_size > 0:
                shard = pad_tensor(shard, self.dim, pad_size)
            shard = shard.contiguous() if contiguous else shard
            shard_list.append(shard)
        return shard_list, pad_sizes

    @staticmethod
    def _local_shard_size_on_dim(
        size_on_dim: int,
        num_chunks: int,
        rank: int,
        return_offset: bool = False,
        ) -> int:
        """
        Calculate the local shard size on a specific dimension for a given rank.

        Args:
            size_on_dim (int): Size of the dimension to be sharded.
            num_chunks (int): Number of chunks to split the dimension into.
            rank (int): Rank or index of the shard.
            return_offset (bool, optional): Flag indicating whether to return the offset.

        Returns:
            int: Local shard size for the specified rank on the dimension.
        """
    ) -> Tuple[int, int]:
        """
        returns the local shard size and offset on a given tensor dim
        """
        # 根据给定的张量维度计算本地分片大小和偏移量

        # 计算与 ``torch.chunk`` 内联的分片大小
        if size_on_dim % num_chunks == 0:
            full_chunk_size = size_on_dim // num_chunks
            # 如果需要返回偏移量，则返回完整的分片大小和偏移量
            return full_chunk_size, full_chunk_size * rank if return_offset else -1

        # 不均匀分片情况
        full_chunk_size = (size_on_dim + num_chunks - 1) // num_chunks
        shard_starting_idx = full_chunk_size * rank

        # 如果张量维度小于分片起始索引，则返回0和维度大小作为偏移量（如果需要的话）
        if size_on_dim < shard_starting_idx:
            return 0, size_on_dim if return_offset else -1
        else:
            # 否则计算本地分片大小和分片起始索引作为偏移量（如果需要的话）
            local_shard_size = (
                min(size_on_dim, shard_starting_idx + full_chunk_size)
                - shard_starting_idx
            )
            return local_shard_size, shard_starting_idx if return_offset else -1

    def _shard_tensor(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        在网格维度上分片和散布张量（使用网格维度上的坐标0作为真实来源）
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # 如果当前进程不属于网格，直接返回一个空张量
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        scatter_list, pad_sizes = self._split_tensor(
            tensor, num_chunks, with_padding=True, contiguous=True
        )

        mesh_dim_local_rank = my_coordinate[mesh_dim]
        output = torch.empty_like(scatter_list[mesh_dim_local_rank])
        mesh_scatter(output, scatter_list, mesh, mesh_dim=mesh_dim)

        # 如果本地张量在维度上有填充，则解除填充
        if pad_sizes and pad_sizes[mesh_dim_local_rank] > 0:
            output = unpad_tensor(output, self.dim, pad_sizes[mesh_dim_local_rank])
        return output

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: str,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        reduce and scatter a tensor on a mesh dimension
        """
        # 获取当前进程在网格上的坐标
        my_coordinate = mesh.get_coordinate()
        # 获取在指定网格维度上的网格大小（即进程数）
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        if my_coordinate is None:
            # 如果当前进程不在网格中，则直接返回本地张量（应为空张量）
            return tensor

        # 检查张量是否需要填充
        is_padded = tensor.size(self.dim) % num_chunks != 0
        if is_padded:
            # 如果需要填充，将张量分割成多个分片并计算填充大小
            scattered_list, pad_sizes = self._split_tensor(
                tensor, num_chunks, with_padding=True, contiguous=True
            )
            tensor = torch.cat(scattered_list, dim=self.dim)
        elif not tensor.is_contiguous():
            # 如果张量不连续，进行连续化操作
            tensor = tensor.contiguous()

        # 执行张量的减少和分散操作
        output = funcol.reduce_scatter_tensor(
            tensor, reduce_op, scatter_dim=self.dim, group=(mesh, mesh_dim)
        )

        if is_padded:
            # 如果存在填充，进行反填充操作以恢复原始张量大小
            output = unpad_tensor(output, self.dim, pad_sizes[my_coordinate[mesh_dim]])  # type: ignore[possibly-undefined]
        return output

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
    ) -> torch.Tensor:
        """
        This function all_gather all shards and return a tensor that
        is replicated on the previously sharded mesh dimension
        """
        # 获取在指定网格维度上的网格大小（即进程数）
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        # 检查当前本地张量的形状
        local_shape = list(local_tensor.size())

        # 获取当前逻辑维度的大小
        logical_dim_size = current_logical_shape[self.dim]
        # 检查是否需要填充，如果是，计算填充后的大小
        is_padded = logical_dim_size % num_chunks != 0

        if is_padded:
            full_chunk_size = (logical_dim_size + num_chunks - 1) // num_chunks
            pad_size = full_chunk_size - local_shape[self.dim]
            local_tensor = pad_tensor(local_tensor, self.dim, pad_size)

        if not local_tensor.is_contiguous():
            # 如果张量不连续，进行连续化操作
            local_tensor = local_tensor.contiguous()

        # 执行张量的全局收集操作
        result = funcol.all_gather_tensor(
            local_tensor,
            gather_dim=self.dim,
            group=(mesh, mesh_dim),
        )
        if is_padded:
            # 如果存在填充，进行反填充操作以恢复原始张量大小
            unpad_size = full_chunk_size * num_chunks - logical_dim_size  # type: ignore[possibly-undefined]
            result = unpad_tensor(result, self.dim, unpad_size)
        return result

    def _replicate_to_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_index: int,
    ) -> torch.Tensor:
        """
        transform from replicated tensor to a sharded tensor on
        the current rank, which would perform a local chunk
        """
        # 获取在指定网格维度上的网格大小（即进程数）
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        # 将本地张量分割成多个分片
        shards, _ = self._split_tensor(
            local_tensor,
            num_chunks,
            with_padding=False,
            contiguous=False,
        )
        # 返回指定分片索引的克隆张量
        return shards[shard_index].clone()
    def _to_new_shard_dim(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: List[int],
        new_shard_dim: int,
    ) -> torch.Tensor:
        """
        transform from existing sharded tensor to a new sharded tensor on
        that shard on a new dimension, which performs an alltoall
        """
        # 获取当前节点在网格中的坐标
        my_coordinate = mesh.get_coordinate()
        # 如果当前节点不在网格中，返回本地张量（应为空张量）
        if my_coordinate is None:
            return local_tensor

        # 获取网格在指定维度上的大小（即节点数）
        num_chunks = mesh.size(mesh_dim=mesh_dim)

        # 获取当前维度的逻辑尺寸和新维度的逻辑尺寸
        old_dim_logical_size = current_logical_shape[self.dim]
        new_dim_logical_size = current_logical_shape[new_shard_dim]

        # 检查当前维度是否存在填充
        old_dim_padding = old_dim_logical_size % num_chunks != 0
        if old_dim_padding:
            # 计算当前维度的填充后的全尺寸块大小，并对本地张量进行填充
            old_dim_full_chunk_size = (old_dim_logical_size + num_chunks - 1) // num_chunks
            old_dim_pad_size = old_dim_full_chunk_size - local_tensor.size(self.dim)
            local_tensor = pad_tensor(local_tensor, self.dim, old_dim_pad_size)

        # 检查新维度是否存在填充
        new_dim_padding = new_dim_logical_size % num_chunks != 0
        if new_dim_padding:
            # 计算新维度的填充后的全尺寸块大小，并对本地张量进行填充
            new_dim_full_chunk_size = (new_dim_logical_size + num_chunks - 1) // num_chunks
            new_dim_pad_size = new_dim_full_chunk_size * num_chunks - local_tensor.size(new_shard_dim)
            local_tensor = pad_tensor(local_tensor, new_shard_dim, new_dim_pad_size)

        # 如果本地张量不是连续的，进行连续化处理
        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        # 执行在指定维度上的 alltoall 操作，返回新的张量
        new_tensor = shard_dim_alltoall(
            local_tensor, self.dim, new_shard_dim, mesh, mesh_dim
        )

        # 如果存在当前维度的填充，对新张量进行去填充操作
        if old_dim_padding:
            old_dim_unpad_size = (old_dim_full_chunk_size * num_chunks - current_logical_shape[self.dim])
            new_tensor = unpad_tensor(new_tensor, self.dim, old_dim_unpad_size)

        # 如果存在新维度的填充，计算本地节点在新维度上的局部分片大小，并对新张量进行去填充操作
        if new_dim_padding:
            local_shard_size_on_new_dim = self._local_shard_size_on_dim(
                new_dim_logical_size, num_chunks, my_coordinate[mesh_dim]
            )[0]
            new_dim_unpad_size = new_dim_full_chunk_size - local_shard_size_on_new_dim
            new_tensor = unpad_tensor(new_tensor, new_shard_dim, new_dim_unpad_size)

        return new_tensor

    def __eq__(self, other: object) -> bool:
        # 检查当前对象与另一个对象是否相等（用于等式比较）
        if not isinstance(other, Shard):
            return False
        return self.dim == other.dim

    def __hash__(self) -> int:
        # 返回对象的哈希值（用于散列存储）
        return hash(self.dim)

    def __repr__(self) -> str:
        """
        machine readable representation of the Shard placement
        """
        # 返回对象的可机器读取表示（用于显示）
        return f"Shard(dim={self.dim})"
    def __str__(self) -> str:
        """返回Shard放置的人类可读表示"""
        # 返回包含Shard维度信息的字符串表示
        return f"S({self.dim})"
@dataclass(frozen=True)
class Replicate(Placement):
    """
    The ``Replicate()`` placement describes the DTensor replicating on a corresponding
    ``DeviceMesh`` dimension, where each rank on the DeviceMesh dimension holds a
    replica of the global Tensor. The ``Replicate`` placement can be used by all
    DTensor APIs (i.e. distribute_tensor, from_local, etc.)
    """

    def __eq__(self, other: object) -> bool:
        # Check if the other object is of type Replicate
        if not isinstance(other, Replicate):
            return False
        return True

    def __hash__(self) -> int:
        # Hash value for every instance of Replicate is the same
        return -1

    def __repr__(self) -> str:
        """
        machine readable representation of the Replicate placement
        """
        return "Replicate()"

    def __str__(self) -> str:
        """
        human readable representation of the Replicate placement
        """
        return "R"

    def _replicate_tensor(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        """
        Replicate (broadcast) a torch.Tensor on a mesh dimension (use
        the first coordinate on the mesh dimension as source of truth)
        """
        # Get the current rank's coordinate on the mesh
        my_coordinate = mesh.get_coordinate()
        # If the current rank is not part of the mesh, return an empty tensor
        if my_coordinate is None:
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        # Ensure the tensor is contiguous for efficient operations
        tensor = tensor.contiguous()
        # Broadcast the tensor across the mesh dimension
        mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim)
        return tensor


@dataclass(frozen=True)
class Partial(Placement):
    """
    The ``Partial(reduce_op)`` placement describes the DTensor that is pending
    reduction on a specified ``DeviceMesh`` dimension, where each rank on the
    DeviceMesh dimension holds the partial value of the global Tensor. User can
    redistribute the ``Partial`` DTensor to a ``Replicate`` or ``Shard(dim)``
    placement on the specified ``DeviceMesh`` dimension using ``redistribute``,
    which would trigger necessary communication operations under the hood (i.e.
    ``allreduce``, ``reduce_scatter``).

    Args:
        reduce_op (str, optional): The reduction op to be used for the partial DTensor
        to produce Replicated/Sharded DTensor. Only element-wise reduction operations
        are supportd, including: "sum", "avg", "prod", "max", "min", default: "sum".

    ::note:: The ``Partial`` placement can be generated as a result of the DTensor operators,
        and can only be used by the ``DTensor.from_local`` API.
    """

    reduce_op: str = "sum"

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # Partial placement contract #1:
        # _reduce_value: reduce the value of the tensor on the mesh dimension
        # Perform an all reduce operation on the tensor using the specified reduce_op
        return funcol.all_reduce(
            tensor, reduceOp=self.reduce_op, group=(mesh, mesh_dim)
        )
    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # Partial placement contract #2:
        # _reduce_shard_value: reduce_scatter the value of the tensor over the mesh dimension
        
        # 将张量的值通过 reduce_scatter 操作在网格维度上进行减少
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        # Partial placement contract #3:
        # _partition_value: partition the value of a replicated tensor on the mesh dimension
        
        # _partition_value 是在网格维度上对复制张量的值进行分区的操作

        # _partition_value 是 _reduce_value 的共轭操作
        # - 即对于求和减少操作，_partition_value 就是一个除法操作
        # - 对于求和减少操作，_reduce_value 就是一个求和（allreduce）操作
        # TODO: 如果 reduce_op 是 min/max 等，则 _partition_value 应该是不同的操作
        assert self.reduce_op == "sum", "only support replicate to PartialSUM for now!"
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        return tensor / num_chunks

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partial):
            return False
        return self.reduce_op == other.reduce_op

    def __hash__(self) -> int:
        return 1 + hash(self.reduce_op)

    def __repr__(self) -> str:
        """
        machine readable representation of the Partial placement
        """
        # 返回 Partial 放置的机器可读表示
        return f"Partial({self.reduce_op})"

    def __str__(self) -> str:
        """
        human readable representation of the Partial placement
        """
        # 返回 Partial 放置的人类可读表示
        return "P"
# We keep the old _Partial name for a while for BC reason
# 由于向后兼容的原因，暂时保留旧的 _Partial 名称
_Partial = Partial


class TensorMeta(NamedTuple):
    # simple named tuple to represent tensor metadata
    # intentionally to stay simple only for sharding
    # propagation purposes.
    # 简单的命名元组，用于表示张量的元数据，意在保持简单以便于分片传播目的。
    shape: torch.Size  # 张量的形状
    stride: Tuple[int, ...]  # 张量的步幅
    dtype: torch.dtype  # 张量的数据类型


# used internally to propagate the placements
# 用于内部传播位置信息
@dataclass
class DTensorSpec:
    mesh: DeviceMesh  # 设备网格
    placements: Tuple[Placement, ...]  # 放置位置的元组

    # tensor meta will only be set during sharding propagation
    # 在分片传播期间才会设置张量元信息
    tensor_meta: Optional[TensorMeta] = None

    def __post_init__(self):
        if not isinstance(self.placements, tuple):
            self.placements = tuple(self.placements)
        self._hash: Optional[int] = None

    def __setattr__(self, attr: str, value: Any):
        super().__setattr__(attr, value)
        # Make sure to recompute the hash in case any of the hashed attributes
        # change (though we do not expect `mesh` or `placements` to change)
        # 确保在哈希属性（mesh、placements 或 tensor_meta）变更时重新计算哈希值
        if hasattr(self, "_hash") and attr in ("mesh", "placements", "tensor_meta"):
            self._hash = None

    def _hash_impl(self) -> int:
        # hashing and equality check for DTensorSpec are used to cache the sharding
        # propagation results. We only need to consider the mesh, placements, shape
        # dtype and stride.
        # 对 DTensorSpec 进行哈希和相等性检查，用于缓存分片传播结果。我们只需要考虑 mesh、placements、shape、dtype 和 stride。
        if self.tensor_meta is not None:
            return hash(
                (
                    self.mesh,
                    self.placements,
                    self.tensor_meta.shape,
                    self.tensor_meta.stride,
                    self.tensor_meta.dtype,
                )
            )
        return hash((self.mesh, self.placements))

    def __hash__(self) -> int:
        # We lazily cache the spec to avoid recomputing the hash upon each
        # use, where we make sure to update the hash when the `tensor_meta`
        # changes by overriding `__setattr__`. This must be lazy so that Dynamo
        # does not try to hash non-singleton `SymInt`s for the stride.
        # 我们懒惰地缓存规范，以避免在每次使用时重新计算哈希值。当 `tensor_meta` 变更时，确保更新哈希值。这必须是懒惰的，以便 Dynamo 不会尝试为步幅哈希非单例的 SymInt。
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def __eq__(self, __o: object) -> bool:
        if not (
            isinstance(__o, DTensorSpec)
            and self.mesh == __o.mesh
            and self.placements == __o.placements
        ):
            return False
        if self.tensor_meta is None or __o.tensor_meta is None:
            return self.tensor_meta == __o.tensor_meta

        return (
            self.tensor_meta.shape == __o.tensor_meta.shape  # type: ignore[union-attr]
            and self.tensor_meta.stride == __o.tensor_meta.stride  # type: ignore[union-attr]
            and self.tensor_meta.dtype == __o.tensor_meta.dtype  # type: ignore[union-attr]
        )
    def __str__(self) -> str:
        """
        返回可读的字符串表示形式的 DTensorSpec
        """
        # 如果 placements 只有一个元素，转换为字符串
        if len(self.placements) == 1:
            placement_str = str(self.placements[0])
        else:
            # 否则将 placements 转换为字符串
            placement_str = str(self.placements)

        # 如果存在 tensor_meta，则获取其形状作为字符串
        if self.tensor_meta is not None:
            tensor_shape = str(tuple(self.tensor_meta.shape))
        else:
            # 否则形状未知
            tensor_shape = "unknown shape"

        # 返回格式化后的字符串
        return f"Spec({placement_str} on {tensor_shape})"

    @property
    def shape(self) -> torch.Size:
        """
        返回 tensor_meta 的形状
        """
        # 如果 tensor_meta 未设置，抛出数值错误异常
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.shape

    @property
    def stride(self) -> Tuple[int, ...]:
        """
        返回 tensor_meta 的步幅
        """
        # 如果 tensor_meta 未设置，抛出数值错误异常
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.stride

    @property
    def ndim(self) -> int:
        """
        返回 tensor_meta 的维度数
        """
        # 如果 tensor_meta 未设置，抛出数值错误异常
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return len(self.tensor_meta.shape)

    @property
    def num_shards(self) -> int:
        """
        返回对象分片的数量
        """
        num_shards = 1
        # 遍历 placements 中的每个元素
        for i, placement in enumerate(self.placements):
            # 如果当前 placement 是分片的，乘以当前维度的尺寸
            if placement.is_shard():
                num_shards *= self.mesh.size(i)
        return num_shards

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        返回设备网格的简单别名
        """
        # 简单地返回 mesh 字段，用于简化 DTensor/DTensorSpec 混合情况的检查
        return self.mesh
    def dim_map(self) -> List[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. It simply return a list of ints
        where dim_map[i] denotes the sharding mapping to the mesh
        dimension, and len(dim_map) == dist_tensor.ndim
        dim_map[i] = -1: means tensor dim i replicate on mesh
        dim_map[i] = j: means tensor dim i shard on mesh dim j

        For example, we have a dist tensor that have the shape of
        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:
        [Shard(1)], the dim_map of this placement would be:
        [-1, 0, -1]. This representation is pretty helpful during
        sharding propagation where we could know exactly each
        tensor dimension is sharded or not.

        Note that if placements contains `_Partial`, we have to
        explicitly deal with it, so that when we create a DTensorSpec
        with dim_map, we could properly record the pending sums.
        """
        # dims mapping of dist tensor sharding
        # return size of tensor ndim, -1 represent replicate
        # and int >=0 represent shard on that device mesh dim
        r = [-1] * self.ndim
        for i, placement in enumerate(self.placements):
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim
                if r[shard_dim] > -1:
                    raise ValueError(
                        f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
                        " DTensor operator implementation does not support things like hybrid"
                        " sharding strategies yet (i.e. [Shard(0), Shard(0)])"
                    )
                r[shard_dim] = i
        return r

    @property
    def sums(self) -> List[int]:
        """
        sums is a property we derive from `placements` of the
        distributed tensor. It simply return a list of ints where
        sums[i] denotes the pending sum (partial) on mesh dim i
        """
        # return indices of placements that are _Partial
        return [
            idx
            for idx, placement in enumerate(self.placements)
            if placement.is_partial()
        ]

    @classmethod
    def from_dim_map(
        cls,
        mesh: DeviceMesh,
        dim_map: List[int],
        sums: List[int],
        tensor_meta: Optional[TensorMeta] = None,
    ) -> "DTensorSpec":
        """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integers representing sharding on each
                tensor dimension; see `dim_map` property doc for details
            sums (List[int]): a list of integers representing the dist tensor having
                pending sums on which device mesh dimensions
            tensor_meta (TensorMeta): DTensor metadata

        Return:
            a class:`DTensorSpec` object
        """
        # by default replicate on device mesh dims
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]

        # find all mesh dims that need pending reductions
        for s in sums:
            placements[s] = Partial()

        for i, m in enumerate(dim_map):
            if m >= 0:
                placement = placements[m]
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    raise RuntimeError(
                        f"DeviceMesh dimension cannot be mapped to two dimensions of the same tensor: {i} and {placement.dim}"
                    )
                elif placement.is_partial():
                    raise RuntimeError(
                        f"DeviceMesh dimension {m} cannot be both shard and partial!"
                    )
                placements[m] = Shard(i)

        return cls(mesh, tuple(placements), tensor_meta=tensor_meta)

    def is_replicated(self):
        """
        Return True if the current DTensorSpec replicates on all mesh dims (devices).
        """
        return all(placement.is_replicate() for placement in self.placements)

    def is_sharded(self):
        """
        Return True if the current DTensorSpec is sharded on any mesh dims (devices).
        """
        return any(placement.is_shard() for placement in self.placements)

    def shallow_copy_with_tensor_meta(
        self, tensor_meta: Optional[TensorMeta]
    ) -> "DTensorSpec":
        """
        Shallow copy the DTensorSpec with a new tensor_meta.
        """
        assert tensor_meta is not None, "shallow copy with no tensor_meta!"
        return DTensorSpec(
            self.mesh,
            self.placements,
            tensor_meta=tensor_meta,
        )
```