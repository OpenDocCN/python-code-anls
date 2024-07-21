# `.\pytorch\torch\distributed\pipelining\microbatch.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入日志模块
import logging
# 导入类型提示模块
from typing import Any, Dict, List, Optional, Tuple

# 导入 PyTorch 库
import torch
# 导入 PyTorch FX 模块中的节点映射函数
from torch.fx.node import map_aggregate
# 导入 PyTorch 工具库中的树展平和树重构函数
from torch.utils._pytree import tree_flatten, tree_unflatten

# 导出的符号列表
__all__ = [
    "TensorChunkSpec",
    "split_args_kwargs_into_chunks",
    "merge_chunks",
]

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

"""
_debug_mask_minibatches specifies to send masked versions of the mini-batch
through instead of micro-batch slices--this can be used for more stable
numerical testing (see [A Note About Correctness Testing])
"""
# 是否启用调试模式来掩码化小批量数据，默认为 False
_debug_mask_minibatches = False

# 自定义的 reducer 类，用于将多个微批次的损失值合并成一个值
class _CustomReducer:
    """
    Custom reducer class that can be used to specify a custom operation that
    reduces losses of multiple microbatches into one value.

    Example:
    >>> # xdoctest: +SKIP
    >>> sum_reducer = _CustomReducer(
    >>>     torch.tensor(0.0),
    >>>     lambda a, b: a + b
    >>> )
    """

    def __init__(self, init_value, reduce_fn):
        # 初始值
        self.init_value = init_value
        # 合并函数
        self.reduce_fn = reduce_fn

# 继承自 _CustomReducer 的损失值合并类
class _LossReducer(_CustomReducer):
    pass

# 默认的分块维度为 0，用于未指定分块维度的情况
DEFAULT_CHUNK_DIM = 0

# 用于指定输入张量分块的类
class TensorChunkSpec:
    """
    Class used to specify chunking of inputs
    """

    def __init__(self, split_dim):
        # 分块维度
        self.split_dim = split_dim

    # 分块维度属性
    split_dim: int

    def __repr__(self):
        # 返回对象的字符串表示形式
        return (
            f"{self.__class__.__module__}.{self.__class__.__name__}({self.split_dim})"
        )

    def __str__(self):
        # 返回对象的字符串表示形式
        return f"TensorChunkSpec({self.split_dim})"

    @staticmethod
    def from_tuple(
        chunk_dims: Tuple[int, ...],
    ):
        """
        A helper for creating a tuple of `TensorChunkSpec` from a tuple of chunk
        dimensions (int's).
        Example:
            >>> # xdoctest: +SKIP
            >>> # There are three positional arguments to the model, and
            >>> # we are chunking them along dimension 0, 0 and 1, respectively
            >>> args_chunk_spec = TensorChunkSpec.from_tuple((0, 0, 1))
        """
        # 从元组中创建 TensorChunkSpec 列表的帮助函数
        args_chunk_spec = map_aggregate(
            chunk_dims,
            lambda dim: TensorChunkSpec(dim),
        )
        return args_chunk_spec

    @staticmethod
    def from_dict(
        chunk_dims: Dict[str, int],
    ):
        """
        A helper for creating a dictionary of `TensorChunkSpec` from a
        dictionary of chunk dimensions (int's).
        Example:
            >>> # xdoctest: +SKIP
            >>> # Chunk dimension 0 for the "id" argument, 1 for the "mask" argument
            >>> kwargs_chunk_spec = TensorChunkSpec.from_dict({"id": 0, "mask": 1})
        """
        # 从字典中创建 TensorChunkSpec 字典的帮助函数
        kwargs_chunk_spec = map_aggregate(
            chunk_dims,
            lambda dim: TensorChunkSpec(dim),
        )
        return kwargs_chunk_spec
# Class used to specify replication of inputs
class _Replicate:
    pass


def _shard_dict_of_args(
    args_dict,
    args_chunk_spec,
    num_chunks,
):
    """
    Given a dictionary of args, and a dictionary of chunking specs, shard the
    args according to the chunking specs.

    Args:
        args_dict: Dictionary of args
        args_chunk_spec: Dictionary of chunking specs
        num_chunks: Number of chunks to shard the args into

    Returns:
        args_split: List of sharded args
    """
    # Stage 1+2: flatten and shard/replicate

    # args_sharded_replicated : [num args, num flat values, num chunks]
    args_sharded_replicated = {}
    arg_specs = []

    real_num_chunks = num_chunks
    first_tensor = True

    assert len(args_dict) == len(
        args_chunk_spec
    ), f"args_dict.keys() = {list(args_dict.keys())} args_chunk_spec.keys() = {list(args_chunk_spec.keys())}"

    # chunks_flat : [num chunks, num args, num flat values]
    chunks_flat = []
    for chunk_idx in range(real_num_chunks):
        chunk_args = {}
        for key, arg in args_sharded_replicated.items():
            arg_single_chunk = []
            for v_flat in arg:
                arg_single_chunk.append(v_flat[chunk_idx])
            chunk_args[key] = arg_single_chunk
        chunks_flat.append(chunk_args)

    # args_split : [num chunks, num args]
    args_split = []

    for chunk in chunks_flat:
        per_chunk_args = {}
        assert len(arg_specs) == len(chunk)
        for (key, arg), arg_spec in zip(chunk.items(), arg_specs):
            per_chunk_args[key] = tree_unflatten(arg, arg_spec)
        args_split.append(per_chunk_args)

    return args_split


def split_args_kwargs_into_chunks(
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    chunks: int,
    args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
    kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
) -> Tuple[List[Tuple], List[Dict]]:
    """
    Given a sequence of args and kwargs, split them into a number of chunks
    according to  their respective chunking specs.

    Args:
        args: Tuple of args
        kwargs: Dict of kwargs
        chunks: Number of chunks to split the args and kwargs into
        args_chunk_spec: chunking specs for args, in same shape as args
        kwargs_chunk_spec: chunking specs for kwargs, in same shape as kwargs

    Returns:
        args_split: List of sharded args
        kwargs_split: List of sharded kwargs
    """
    # Given `args` and `kwargs`, we want to yield a set of `chunks` args and kwargs such that
    # the constituent Tensor values have been sharded/replicated according to the `args_chunk_spec`
    # and `kwargs_chunk_spec` specifications. The steps are as follows:
    #
    # 1. Use pytree.tree_flatten to flatten each arg and its spec into a 1d array of values.
    #    To use a running example: suppose our inputs look like
    #
    #    args = (tensor1, tensor2, ...)
    #    kwargs = {'arg1': tensor3, 'arg2': tensor4, ...}
    #
    #    Then we have args_chunk_spec = (spec1, spec2, ...) and kwargs_chunk_spec = {'arg1': spec3, 'arg2': spec4, ...}.
    #
    # 2. Create `args_sharded_replicated`, a structure holding the flattened values of all tensors and their specs,
    #    arranged for sharding/replication across `chunks`.
    args_sharded_replicated = {}
    arg_specs = []

    # 3. Iterate through each chunk index to create `chunks_flat`, a list where each entry represents
    #    a dictionary of tensors (flattened) for that chunk index.
    chunks_flat = []
    for chunk_idx in range(chunks):
        chunk_args = {}
        for key, arg in args_sharded_replicated.items():
            arg_single_chunk = []
            for v_flat in arg:
                arg_single_chunk.append(v_flat[chunk_idx])
            chunk_args[key] = arg_single_chunk
        chunks_flat.append(chunk_args)

    # 4. For each chunk in `chunks_flat`, reconstruct the original tensors using `tree_unflatten` with
    #    the corresponding specs from `arg_specs`, resulting in `args_split`.
    args_split = []
    for chunk in chunks_flat:
        per_chunk_args = {}
        assert len(arg_specs) == len(chunk)
        for (key, arg), arg_spec in zip(chunk.items(), arg_specs):
            per_chunk_args[key] = tree_unflatten(arg, arg_spec)
        args_split.append(per_chunk_args)

    return args_split
    # TODO: _debug_mask_minibatches
    # 处理 kwargs 为 None 的情况，将其设置为空字典
    if kwargs is None:
        kwargs = {}
    
    # 如果用户没有提供 args_chunk_spec 或 kwargs_chunk_spec，则扩展其格式并使用默认的 chunking（沿 dim 0）
    if args_chunk_spec is None:
        args_chunk_spec = (TensorChunkSpec(DEFAULT_CHUNK_DIM),) * len(args)
    
    if kwargs_chunk_spec is None:
        kwargs_chunk_spec = dict.fromkeys(kwargs, TensorChunkSpec(DEFAULT_CHUNK_DIM))
    
    # 对 args 进行分片，根据 spec 中的策略进行分片或复制
    args_split_dict = _shard_dict_of_args(
        dict(enumerate(args)),
        dict(enumerate(args_chunk_spec)),
        chunks,
    )
    real_num_chunks = len(args_split_dict)
    
    # 对 kwargs 进行分片
    kwargs_split = _shard_dict_of_args(
        kwargs,
        kwargs_chunk_spec,
        real_num_chunks,
    )
    
    # 如果 kwargs 分片的数量少于实际的 chunks 数量，则更新实际的 chunks 数量
    if len(kwargs_split) < real_num_chunks:
        real_num_chunks = len(kwargs_split)
        # 重新对 args 进行分片
        args_split_dict = _shard_dict_of_args(
            dict(enumerate(args)),
            dict(enumerate(args_chunk_spec)),
            real_num_chunks,
        )
    
    # 如果 args 分片的数量与 kwargs 分片的数量不相等，则抛出 RuntimeError
    if len(args_split_dict) != len(kwargs_split):
        raise RuntimeError(
            "args and kwargs are split into different number of chunks: "
            f"{len(args_split_dict)}, {len(kwargs_split)}"
        )
    
    # 将每个分片中的参数整理为元组并存储在 args_split 列表中
    args_split = []
    for chunk_args in args_split_dict:
        args_split.append(tuple(chunk_args[i] for i in range(len(chunk_args))))
    
    # 返回 args_split 列表和 kwargs_split 字典作为结果
    return args_split, kwargs_split
# 定义函数，将给定的多个块按照指定的块规范合并成单个值
def merge_chunks(
    chunks: List[Any],
    chunk_spec,
):
    """
    Given a list of chunks, merge them into a single value according to
    the chunk spec.

    Args:
        chunks: list of chunks  # 给定的块列表
        chunk_spec: Chunking spec for the chunks  # 用于这些块的分块规范

    Returns:
        value: Merged value  # 合并后的值
    """
    # This is essentially the inverse of `split_args_kwargs_into_chunks`, so the
    # steps are similar to the steps in that function but in reverse. Given the
    # input values:
    #
    #       chunks = [
    #           ([A, [B, C_1]], D),
    #           ([A, [B, C_2]], D),
    #       ]
    #       args_spec = ([None, [None, TensorChunkSpec]], None)
    #
    # 1. Flatten the chunks according to the chunk_spec
    #
    #       chunks_flat = [
    #           ([A, B, C_1], D),
    #           ([A, B, C_2], D),
    #       ]
    #
    # 2. Rotate the nesting order such that chunks are the inner dimension
    #
    #       value_inner = ([A, B, [C_1, C_2]], D)
    #
    # 3. Concatenate sharded arguments
    #
    #       value_combined = ([A, B, C], D)
    #
    # 4. Unflatten the combined args given the spec
    #
    #       value = ([A, [B, C]], D)

    # Preliminary: flatten the chunk spec
    # 如果 chunk_spec 不为空，将其展平
    if chunk_spec is not None:
        spec_flattened, flatten_spec = tree_flatten(chunk_spec)
    else:
        # 如果未提供 chunk_spec，则沿着默认维度（0）合并 chunks，并生成 chunk_spec
        chunk0_flat, flatten_spec = tree_flatten(chunks[0])
        spec_flattened = [TensorChunkSpec(DEFAULT_CHUNK_DIM)] * len(chunk0_flat)

    # Stage 1: flatten chunks
    # 阶段 1：展平 chunks
    # chunks_flattened : [num chunks, num args]
    chunks_flattened = []

    for chunk in chunks:
        chunk_flattened, _ = tree_flatten(chunk)
        if len(chunk_flattened) != len(spec_flattened):
            raise ValueError(f"Chunk {chunk} did not match chunk spec {chunk_spec}")

        chunks_flattened.append(chunk_flattened)

    # Stage 2 and 3: Rotate nesting order s.t. chunks are inner dimension and
    #                concatenate sharded operands
    # 阶段 2 和 3：旋转嵌套顺序，使 chunks 成为内部维度，并连接分片操作数
    # args_flattened : [num args]
    args_flattened = []
    for arg_idx, arg in enumerate(spec_flattened):
        # 遍历扁平化后的参数列表，arg_idx 是索引，arg 是参数对象
        if isinstance(arg, TensorChunkSpec):
            # 如果参数是 TensorChunkSpec 类型
            partial_values = [
                chunks_flattened[chunk_idx][arg_idx]
                for chunk_idx in range(len(chunks_flattened))
            ]
            
            if _debug_mask_minibatches:
                # 如果开启了调试模式 _debug_mask_minibatches
                # 推断单个块的大小，再次运行 `tensor_split`
                overall_shape = partial_values[0].shape
                for val in partial_values[1:]:
                    assert val.shape == overall_shape
                # 将空的元数据张量按照指定的维度分割成指定数量的段
                meta_chunks = torch.tensor_split(
                    torch.empty(*overall_shape, device="meta"),
                    sections=len(partial_values),
                    dim=arg.split_dim,
                )

                values_to_cat = []
                chunk_start_idx = 0
                assert len(partial_values) == len(meta_chunks)
                # 遍历部分值和元数据块，生成要连接的值列表
                for partial_value, meta_chunk in zip(partial_values, meta_chunks):
                    chunk_end_idx = chunk_start_idx + meta_chunk.size(arg.split_dim)

                    slice_indices = [slice(None, None, None)] * partial_value.ndim
                    slice_indices[arg.split_dim] = slice(chunk_start_idx, chunk_end_idx)
                    sliced = partial_value[slice_indices]
                    values_to_cat.append(sliced)

                    chunk_start_idx = chunk_end_idx

            else:
                # 如果未开启调试模式，直接使用部分值
                values_to_cat = partial_values

            # 将所有部分值按照指定的维度连接起来，并添加到扁平化参数列表中
            args_flattened.append(torch.cat(values_to_cat, dim=arg.split_dim))
        
        elif isinstance(arg, _CustomReducer):
            # 如果参数是自定义的 reducer 类型
            reduced_val = arg.init_value

            for chunk_idx in range(len(chunks_flattened)):
                # 依次将每个块的值应用 reduce 函数进行归约
                reduced_val = arg.reduce_fn(
                    reduced_val, chunks_flattened[chunk_idx][arg_idx]
                )

            args_flattened.append(reduced_val)
        
        else:
            # 对于普通的参数值，假设第一个块的值为所有块的共同值，进行断言检查
            value = chunks_flattened[0][arg_idx]
            for chunk_idx in range(1, len(chunks_flattened)):
                assert chunks_flattened[chunk_idx][arg_idx] == value
            args_flattened.append(value)

    # Stage 4: Unflatten combined args
    # 将组合后的扁平化参数根据扁平化规范解除扁平化
    return tree_unflatten(args_flattened, flatten_spec)
```