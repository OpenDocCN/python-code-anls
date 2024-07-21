# `.\pytorch\torch\distributed\_tensor\ops\view_ops.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入必要的模块和类
from dataclasses import dataclass
from typing import (
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# 导入 torch 库及相关模块
import torch
from torch import Tensor
from torch.distributed._tensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed._tensor.api import Shard
from torch.distributed._tensor.ops.utils import (
    generate_redistribute_costs,
    normalize_dim,
    normalize_dims,
    prod,
    register_op_strategy,
)
from torch.distributed._tensor.placement_types import DTensorSpec, Placement, Replicate
from torch.distributed.device_mesh import DeviceMesh

# 使用 torch 的 aten 操作
aten = torch.ops.aten

# 定义一个别名
Shape = Tuple[int, ...]


@dataclass
class DimSpec:
    """Specifies how an output dimension maps to an input dimension."""

    def inputs(self) -> Iterable["DimSpec"]:
        return ()


# 定义一个元组，用于描述输出维度如何映射到输入张量的维度
DimMap = Tuple[DimSpec, ...]


@dataclass
class Singleton(DimSpec):
    """Output dimension is a singleton."""

    pass


@dataclass
class InputDim(DimSpec):
    """Output dimension maps directly to an input dimension."""

    input_dim: int


@dataclass
class Broadcast(DimSpec):
    """Output is the broadcast of a singleton input dimension."""

    dim: DimSpec
    dim_size: int

    @classmethod
    def new(cls, dim: DimSpec, dim_size: int) -> DimSpec:
        return Broadcast(dim, dim_size)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.dim,)


@dataclass
class NewDim(DimSpec):
    """This is a new dimension created by the op."""

    size: int

    @classmethod
    def new(cls, size: int) -> DimSpec:
        return Singleton() if size == 1 else NewDim(size)


@dataclass
class Repeat(DimSpec):
    """Output dimension is the input dimension repeated n-times."""

    input_dim: DimSpec
    times: int

    @classmethod
    def new(cls, dim: DimSpec, times: int) -> DimSpec:
        if times == 1:
            return dim
        elif isinstance(dim, Singleton):
            # repeating a singleton is the same as broadcasting it
            return Broadcast(dim, times)
        else:
            return Repeat(dim, times)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)


@dataclass
class Flatten(DimSpec):
    """Flatten a set of input dimensions, ensuring right-most adjacent elements remain adjacent in the output."""

    input_dims: Sequence[DimSpec]

    @classmethod
    def new(cls, dims: Sequence[DimSpec]) -> DimSpec:
        if len(dims) == 0:
            # flattening a scalar leads to a singleton
            return Singleton()
        elif len(dims) == 1:
            # flattening a single dimension is no-op
            return dims[0]
        else:
            return Flatten(dims)
    # 定义一个方法 `inputs`，返回类型为 `Iterable[DimSpec]`
    def inputs(self) -> Iterable[DimSpec]:
        # 返回对象内部的 `input_dims` 属性，该属性应当是一个迭代器或可迭代对象
        return self.input_dims
@dataclass
class Split(DimSpec):
    """
    This class represents a dimension that is part of a decomposition of the input dimension.

    Note that input_dim itself could be a Flattened set of input dims.
    """

    input_dim: DimSpec  # Represents the input dimension specification
    group_shape: Shape  # Shape of the group this dimension belongs to
    split_id: int  # Identifier for this split

    @classmethod
    def new(cls, dim: DimSpec, group_shape: Tuple[int, ...], idx: int) -> DimSpec:
        assert len(group_shape) > 0
        if len(group_shape) == 1:
            # If there's only one element in group_shape, return the input dimension
            assert idx == 0
            return dim
        elif group_shape[idx] == 1:
            # If the group_shape's element at idx is 1, return a Singleton dimension
            return Singleton()
        else:
            # Remove singletons from the group_shape and return a new Split dimension
            group_mapping = list(
                enumerate((s, i) for i, s in enumerate(group_shape) if s != 1)
            )
            new_group_shape = tuple(m[1][0] for m in group_mapping)
            new_idx = next(filter(lambda x: x[1][1] == idx, group_mapping))[0]
            return Split(dim, new_group_shape, new_idx)

    def inputs(self) -> Iterable[DimSpec]:
        return (self.input_dim,)  # Returns an iterable containing the input dimension


def dim_pad_left(ndim: int, min_dims: int) -> DimMap:
    """
    Pad dimensions on the left with Singletons if necessary.

    Returns a tuple representing the padded dimensions.
    """
    return (Singleton(),) * max(0, min_dims - ndim) + tuple(
        InputDim(i) for i in range(ndim)
    )


def dim_atleast_3d(ndim: int) -> DimMap:
    """
    Ensure at least 3 dimensions in the specified shape.

    Returns a tuple representing the adjusted dimensions.
    """
    if ndim == 0:
        return (Singleton(), Singleton(), Singleton())
    elif ndim == 1:
        return (Singleton(), InputDim(0), Singleton())
    elif ndim == 2:
        return (InputDim(0), InputDim(1), Singleton())
    else:
        return tuple(InputDim(i) for i in range(ndim))


def expand(input_shape: Shape, shape: Shape) -> DimMap:
    """
    Implement broadcasting of input_shape to match shape.

    Returns a tuple representing the mapped dimensions after broadcasting.
    """
    assert len(shape) >= len(input_shape)

    # 1. create padded input dimensions
    padded_input = dim_pad_left(len(input_shape), len(shape))
    # 2. check that input shapes are compatible
    mapping = []
    for p, desired_s in zip(padded_input, shape):
        if isinstance(p, Singleton):
            actual_s = 1
            assert desired_s >= 0
        else:
            assert isinstance(p, InputDim), f"DimSpec not supported in expand: {p}"
            actual_s = input_shape[p.input_dim]
            assert actual_s == 1 or desired_s == -1 or desired_s == actual_s
        mapping.append(
            p
            if desired_s in (1, -1) or desired_s == actual_s
            else Broadcast.new(p, desired_s)
        )
    return tuple(mapping)


def normalize_sizes(sizes: Union[Shape, Tuple[Shape]]) -> Shape:
    """
    Normalize sizes to ensure a consistent shape.

    Returns a tuple representing the normalized shape.
    """
    if isinstance(sizes[0], int):
        return cast(Shape, sizes)
    elif len(sizes) == 1:
        return cast(Shape, sizes[0])  # type: ignore[redundant-cast]
    else:
        raise RuntimeError("Size must be int... or tuple")


def dim_flatten(ndim: int, start_dim=0, end_dim=-1) -> DimMap:
    """
    Flatten dimensions within the specified range.

    Returns a tuple representing the flattened dimensions.
    """
    if ndim == 0:
        return (Singleton(),)
    elif ndim == 1:
        return (InputDim(0),)
    else:
        # 只对从 start_dim 到 end_dim（包括）的维度进行扁平化处理
        # 其他维度保持不变
        if end_dim < 0:
            end_dim += ndim  # 如果 end_dim 是负数，则转换为相对索引
        results: List[DimSpec] = [InputDim(i) for i in range(start_dim)]  # 创建从 0 到 start_dim-1 的输入维度列表
        results.append(
            Flatten.new(tuple(InputDim(i) for i in range(start_dim, end_dim + 1)))
            # 在结果列表中添加一个新的 Flatten 操作，处理从 start_dim 到 end_dim 的维度
        )
        results.extend([InputDim(i) for i in range(end_dim + 1, ndim)])
        # 将剩余的维度（大于 end_dim 的部分）添加到结果列表中作为输入维度
        return tuple(results)
def dim_movedim(
    ndim: int,
    input: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> DimMap:
    # 标准化输入和目标维度，确保它们符合预期的形式
    input = normalize_dims(input, ndim)
    destination = normalize_dims(destination, ndim)

    # 断言输入和目标维度的长度一致
    assert len(input) == len(destination)
    # 创建输入维度的集合，确保没有重复的维度
    input_set = set(input)
    assert len(input_set) == len(input), "Found repeated input dims"
    # 确保目标维度中没有重复的维度，并且所有维度都在合法范围内
    assert len(set(destination)) == len(destination), "Found repeated output dims"
    assert max(input) < ndim
    assert max(destination) < ndim

    # 初始化目标维度映射表，-1 表示未被映射的维度
    dest = [-1] * ndim
    # 根据输入和目标维度创建映射关系
    for i, d in zip(input, destination):
        dest[d] = i

    # 生成未被使用的输入维度迭代器
    unused_inputs_iter = iter(i for i in range(ndim) if i not in input_set)
    # 填充未被映射的目标维度
    for i in range(ndim):
        if dest[i] == -1:
            dest[i] = next(unused_inputs_iter)

    # 返回映射后的维度元组
    return tuple(InputDim(i) for i in dest)


def dim_repeat(ndim: int, sizes: Shape) -> DimMap:
    # 标准化尺寸，确保它们符合预期的形式
    sizes = normalize_sizes(sizes)
    # 断言重复维度的数量不少于张量的总维度数
    assert (
        len(sizes) >= ndim
    ), f"Number of dimensions of repeat dims {sizes} can not be smaller than number of dimensions of tensor {ndim}."
    # 计算填充数
    pad = len(sizes) - ndim
    # 返回填充后的重复维度映射
    return tuple(Repeat.new(Singleton(), s) for s in sizes[:pad]) + tuple(
        Repeat.new(InputDim(i), s) for i, s in enumerate(sizes[pad:])
    )


def infer_size(total_size: int, sizes: Shape) -> Shape:
    """
    One dimension input to view may be "-1".

    Infer the size of this dimension given the total_size.
    """
    # 找出尺寸中值为 -1 的维度索引
    infers = [i for i, s in enumerate(sizes) if s == -1]
    # 计算尺寸的乘积
    size = prod(sizes)
    # 断言最多只能推断出一个维度的尺寸
    assert len(infers) <= 1, "can only infer one size"
    if infers:
        # 如果有可以推断的维度，计算缺失的尺寸大小
        size = -size
        missing_size = total_size // size
        # 断言推断的尺寸是整数倍
        assert (
            total_size % size == 0
        ), f"size inferred for -1 is not integral {sizes} should have {total_size} elements."
        # 替换推断维度的值为实际的大小
        return tuple(s if s != -1 else missing_size for s in sizes)
    # 最终断言尺寸与总尺寸相符
    assert size == total_size, f"sizes do not match {total_size} vs {size}"
    return sizes


def view_groups(from_size: Shape, to_size: Shape) -> DimMap:
    """
    Decompose a reshape operation into forwarding, flattening, or splitting dimensions for each output dimension.

    A view or reshape operation can be decomposed into a set of 3 types of smaller operations:
    1) Forward a dimension from input to output
    2) Flatten a set of dimensions into a single dimension
    3) Split one dimension into multiple dimensions

    view_groups identifies these operations and returns, for each output dimension, what
    is operation was performed in the input dimension. For example:

        view_groups([2, 3, 4], [2, 12]) -> (
            InputDim(0),
            Flatten((InputDim(1), InputDim(2)))
        )

    - ouptut dimension 0 maps to input dimension 0
    - output dimension 1 maps to a flattened input dimensions 1 and 2


        view_groups([2, 3], [3, 2]) -> (
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
            Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
        )
    """
    # 该函数详细说明了重塑操作的分解方式，无需进一步注释
    pass
    # 输入张量的总元素数
    from_nelem = prod(from_size)
    # 推断目标大小，将输入张量的总元素数和目标大小规范化后的大小进行比较
    to_size = infer_size(from_nelem, normalize_sizes(to_size))
    
    # 断言：输入张量的总元素数应该与目标大小的总元素数相等，否则抛出异常
    assert from_nelem == prod(to_size), "Total view shape does not add up"
    
    # 初始化索引和长度信息
    from_idx = 0
    to_idx = 0
    from_len = len(from_size)
    to_len = len(to_size)
    
    # 存储结果的列表
    result_pp = []
    
    # 循环直到处理完所有维度
    while from_idx < from_len or to_idx < to_len:
        # 初始化当前处理的维度组和目标形状
        from_group_dim, to_group_shape = [], []
    
        # 处理输入维度组
        if from_idx >= from_len:
            f = 1
        else:
            f = from_size[from_idx]
            from_group_dim.append(from_idx)
            from_idx += 1
    
        # 处理目标形状
        if to_idx >= to_len:
            t = 1
        else:
            t = to_size[to_idx]
            to_group_shape.append(t)
            to_idx += 1
    
        # 如果任一组是单例维度，则需要回溯处理
        if f == 1 and t != 1:
            # 产生 ([1], []) 形式的结果，进行回溯处理
            to_idx -= 1
            to_group_shape = []
        elif f != 1 and t == 1:
            # 产生 ([], [1]) 形式的结果，进行回溯处理
            from_idx -= 1
            from_group_dim = []
        else:
            # 处理维度不匹配的情况，进行维度调整
            while f != t:
                if f < t:
                    nf = from_size[from_idx]
                    from_group_dim.append(from_idx)
                    from_idx += 1
                    f *= nf
                else:
                    nt = to_size[to_idx]
                    to_group_shape.append(nt)
                    to_idx += 1
                    t *= nt
    
        # 如果存在目标形状，创建相应的操作并添加到结果列表中
        if len(to_group_shape) > 0:
            # 创建展平操作
            flattened = Flatten.new(
                tuple(InputDim(fi) for fi in from_group_dim if from_size[fi] > 1)
            )
            # 创建拆分操作并添加到结果列表中
            result_pp += [
                Split.new(flattened, tuple(to_group_shape), i)
                for i in range(len(to_group_shape))
            ]
    
    # 返回结果列表作为元组
    return tuple(result_pp)
def dim_tile(ndim: int, dims: Tuple[int, ...]) -> DimMap:
    # 如果给定维度少于 ndim，则在维度前补 1，使维度达到 ndim
    if len(dims) < ndim:
        dims = (1,) * (ndim - len(dims)) + dims
    # 调用 dim_repeat 函数，返回重复维度操作后的结果
    return dim_repeat(ndim, dims)


def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    # 标准化 dim1 和 dim2，确保它们在合理范围内
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    # 断言 dim1 和 dim2 必须小于 ndim，即维度索引在合理范围内
    assert dim1 < ndim
    assert dim2 < ndim
    # 创建一个包含 InputDim 对象的列表，表示输入维度映射
    dimmap = [InputDim(i) for i in range(ndim)]
    # 交换 dim1 和 dim2 对应的维度映射
    swapdim = dimmap[dim1]
    dimmap[dim1] = dimmap[dim2]
    dimmap[dim2] = swapdim
    # 返回交换后的维度映射结果
    return tuple(dimmap)


def dim_squeeze(shape: Shape, dim: Optional[int] = None) -> DimMap:
    # FIXME: 当 dim=None 且其中一个维度等于网格的大小时，这段代码有误。
    # 例如，squeeze(DTensor(tensor(4), Shard[0])) 可能最终变成 squeeze(tensor(1))，
    # 如果我们有 4 个设备；这将导致删除一个实际上不是单例的维度。
    return tuple(
        InputDim(i)
        for i, s in enumerate(shape)
        if s > 1 or (dim is not None and i != normalize_dim(dim, len(shape)))
    )


def dim_unsqueeze(ndim: int, dim: int) -> DimMap:
    # 创建一个包含 InputDim 对象的元组，表示输入维度映射
    dims = tuple(InputDim(i) for i in range(ndim))
    # 如果 dim 是负数，则将其转换为正数索引
    if dim < 0:
        dim += ndim + 1
    # 在指定维度 dim 处插入一个 Singleton() 维度
    return dims[:dim] + (Singleton(),) + dims[dim:]


def dim_view_as_real(shape: Shape) -> DimMap:
    ndim = len(shape)
    # 创建一个 DimSpec 对象的列表，表示输入维度映射
    results: List[DimSpec] = [InputDim(i) for i in range(ndim - 1)]
    # 将每个复数分割为两个实数，增加一个大小为 2 的额外维度
    results.append(Split(InputDim(ndim - 1), (shape[-1], 2), 0))
    results.append(Split(InputDim(ndim - 1), (shape[-1], 2), 1))
    # 返回组成的维度映射结果
    return tuple(results)


def dim_reduction(
    ndim: int, dim_or_dims: Optional[Union[int, Sequence[int]]], keepdim: bool
) -> DimMap:
    """
    General fallback for reduction ops where Partial() does not apply.

    This will cause incoming tensor to be replicated on the reducing dimensions.
    """
    # 如果 dim_or_dims 为 None，则默认为所有维度都要进行缩减操作
    if dim_or_dims is None:
        dim_or_dims = tuple(range(ndim))
    # 如果 dim_or_dims 是单个整数，则转换为包含单个整数的元组
    if isinstance(dim_or_dims, int):
        dim_or_dims = (dim_or_dims,)
    # 标准化 dim_or_dims，确保所有维度索引在合理范围内
    dim_or_dims = tuple(d if d >= 0 else d + ndim for d in dim_or_dims)
    # 返回维度映射结果，根据 keepdim 决定是否保留维度
    return tuple(
        InputDim(i) if i not in dim_or_dims else Singleton()
        for i in range(ndim)
        if i not in dim_or_dims or keepdim
    )


dim_maps: Dict[Callable[..., torch.Tensor], Callable[..., DimMap]] = {
    # torch.atleast_1d 对应的维度映射操作，将维度扩展为至少 1 维
    torch.atleast_1d: lambda x: dim_pad_left(x.ndim, 1),
    # torch.atleast_2d 对应的维度映射操作，将维度扩展为至少 2 维
    torch.atleast_2d: lambda x: dim_pad_left(x.ndim, 2),
    # torch.atleast_3d 对应的维度映射操作，将维度扩展为至少 3 维
    torch.atleast_3d: lambda x: dim_atleast_3d(x.ndim),
    # torch.broadcast_to 对应的维度映射操作，将输入形状扩展到目标形状
    torch.broadcast_to: lambda input, shape: expand(input.shape, shape),
    # Tensor.expand 对应的维度映射操作，将输入形状扩展到指定大小
    Tensor.expand: lambda self, *sizes: expand(self.shape, normalize_sizes(sizes)),
    # torch.flatten 对应的维度映射操作，将输入张量扁平化为一维
    torch.flatten: lambda tensor: dim_flatten(tensor.ndim),
    # torch.movedim 对应的维度映射操作，按指定顺序移动输入张量的维度
    torch.movedim: lambda input, source, destination: dim_movedim(
        input.ndim, source, destination
    ),
    # torch.permute 对应的维度映射操作，按指定维度重新排列输入张量的维度
    torch.permute: lambda input, dims: tuple(
        InputDim(i) for i in normalize_dims(dims, input.ndim)
    ),
}
    # 定义函数 torch.ravel，接受一个张量 tensor，返回将其展平后的结果
    torch.ravel: lambda tensor: dim_flatten(tensor.ndim),

    # 定义方法 Tensor.repeat，接受一个张量 self 和多个尺寸 sizes，返回按指定尺寸重复张量的结果
    Tensor.repeat: lambda self, *sizes: dim_repeat(self.ndim, sizes),

    # 定义函数 torch.reshape，接受一个输入张量 input 和一个形状 shape，返回按指定形状重新整形的结果
    torch.reshape: lambda input, shape: view_groups(input.shape, shape),

    # 定义函数 torch.squeeze，接受一个输入张量 input 和一个维度 dim（可选），返回按指定维度压缩的结果
    torch.squeeze: lambda input, dim=None: dim_squeeze(input.shape, dim),

    # 定义函数 torch.tile，接受一个输入张量 input 和一个尺寸 dims，返回按指定尺寸复制的结果
    torch.tile: lambda input, dims: dim_tile(input.ndim, dims),

    # 定义函数 torch.transpose，接受一个输入张量 input 和两个维度 dim0 和 dim1，返回按指定维度交换的结果
    torch.transpose: lambda input, dim0, dim1: dim_transpose(input.ndim, dim0, dim1),

    # 定义函数 torch.unsqueeze，接受一个输入张量 input 和一个维度 dim，返回按指定维度展开的结果
    torch.unsqueeze: lambda input, dim: dim_unsqueeze(input.ndim, dim),

    # 定义方法 Tensor.view，接受一个输入张量 input 和多个形状 shape，返回按指定形状重新整形的结果
    Tensor.view: lambda input, *shape: view_groups(input.shape, shape),

    # 定义函数 torch.view_as_complex，接受一个输入张量 input，返回视为复数的结果（通常是最后两个维度展平）
    torch.view_as_complex: lambda input: dim_flatten(input.ndim, input.ndim - 2),

    # 定义函数 torch.view_as_real，接受一个输入张量 input，返回视为实数的结果
    torch.view_as_real: lambda input: dim_view_as_real(input.shape),
# 确定输入目标分片和输出分片的函数，基于给定的全局张量形状和输入源分片
def propagate_shape_and_sharding(
    input_src_placements: Sequence[Placement],
    local_in_shape: Shape,
    rule: DimMap,
    mesh_sizes: Shape,
) -> Tuple[Sequence[Placement], Sequence[Placement]]:
    """
    Determine input target sharding and output sharding based on
    given global tensor shape and input source sharding.

    Sharding propagation follows mapped dimensions:
    - An output dimension that maps directly to an input dimension is sharded equally
    - An output dimension that is a flattened set of input dimensions can only be
      sharded if only the leftmost flattened dimension is sharded.
    - An output dimension that is a split of the input dimension can only be sharded
      if the leftmost split size is divisible by the mesh dimension
    """
    # 确保输入源分片数量与网格大小一致
    assert len(input_src_placements) == len(mesh_sizes)
    
    # 网格的维度数
    mesh_ndim = len(mesh_sizes)
    
    # 记录可分片维度的字典，每个维度对应一个布尔列表，表示该维度是否可分片
    shardable_dims: Dict[int, List[bool]] = {}
    
    # 如果输入维度消失（例如折叠、减少），则不能在该维度上进行分片（需要有复制的回退规则）
    seen_input_dims: Set[int] = set()

    # 递归函数，收集使用的输入维度
    def collect_used_inputs(cmd: DimSpec) -> None:
        # 如果是输入维度类型，将其输入维度添加到集合中
        if isinstance(cmd, InputDim):
            seen_input_dims.add(cmd.input_dim)
        # 递归处理每个命令的输入
        for inp in cmd.inputs():
            collect_used_inputs(inp)

    # 遍历规则中的每个命令，收集使用的输入维度
    for cmd in rule:
        collect_used_inputs(cmd)
    
    # 对于每个本地输入形状的维度，初始化其可分片性
    for dim in range(len(local_in_shape)):
        shardable_dims[dim] = [dim in seen_input_dims] * mesh_ndim
    # 定义一个函数，用于确定给定命令（DimSpec）的输入维度是否可以分片，并返回一个可能的输入维度对象或None
    def get_in_dim_to_shard(cmd: DimSpec) -> Optional[InputDim]:
        # 如果命令是输入维度对象，则直接返回该对象
        if isinstance(cmd, InputDim):
            return cmd
        # 如果命令是Flatten对象
        elif isinstance(cmd, Flatten):
            # 遍历除第一个外的所有输入维度对象，并标记它们为不可分片
            for dim in cmd.input_dims[1:]:
                if isinstance(dim, InputDim):
                    shardable_dims[dim.input_dim] = [False] * mesh_ndim
            # 取第一个输入维度对象
            dim0 = cmd.input_dims[0]
            return dim0 if isinstance(dim0, InputDim) else None
        # 如果命令是Split对象
        elif isinstance(cmd, Split):
            # 递归获取输入维度对象，用于确定分片
            in_dim = get_in_dim_to_shard(cmd.input_dim)
            # 获取分片的输出尺寸
            out_size = cmd.group_shape[cmd.split_id]
            if cmd.split_id == 0 and in_dim is not None:
                # 判断输入维度是否可以在每个单独的网格维度上进行分片
                shardable_dims[in_dim.input_dim] = [
                    out_size % mesh_dim_size == 0 for mesh_dim_size in mesh_sizes
                ]

                # 处理特殊情况如[Shard(0), Shard(0)]
                submesh_size = 1
                for size, shard in zip(mesh_sizes, input_src_placements):
                    if isinstance(shard, Shard) and shard.dim == in_dim:
                        submesh_size *= size
                # 确保结果维度大小能被其网格维度整除
                assert (
                    out_size % submesh_size == 0
                ), f"Resulting dimension size {out_size} is not divisible by its mesh dimension {submesh_size}."

            # 只分片Split命令的第一个组件
            return in_dim if cmd.split_id == 0 else None
        # 如果命令是Repeat对象
        elif isinstance(cmd, Repeat):
            # 递归获取输入维度对象，并标记其不可分片
            in_dim = get_in_dim_to_shard(cmd.input_dim)
            if in_dim is not None:
                shardable_dims[in_dim.input_dim] = [False] * mesh_ndim
            return None
        # 其他情况返回None
        else:
            return None

    # 对于每个规则中的维度和命令，确定其输入维度对象，并建立分片映射关系
    shard_dim_map = {}
    for dim, cmd in enumerate(rule):
        in_dim = get_in_dim_to_shard(cmd)
        if in_dim is not None:
            shard_dim_map[in_dim.input_dim] = dim

    # 根据输入源的放置情况，生成目标输入的放置列表
    input_tgt_placements = [
        Replicate()
        if isinstance(p, Shard) and not shardable_dims[p.dim][mesh_dim]
        else p
        for mesh_dim, p in enumerate(input_src_placements)
    ]
    
    # 根据输入的目标放置列表，生成输出的放置列表
    output_placements = [
        Shard(shard_dim_map[p.dim]) if isinstance(p, Shard) else p
        for p in input_tgt_placements
    ]
    # 返回两个变量：input_tgt_placements 和 output_placements
    return input_tgt_placements, output_placements
def register_op_strategy_map(
    aten_op_overload: torch._ops.OpOverload,
    local_op_name: Callable[..., torch.Tensor],
    schema_info: Optional[RuntimeSchemaInfo] = None,
) -> None:
    # 从 dim_maps 字典中获取与 local_op_name 对应的 dim_map 函数
    dim_map: Callable[..., DimMap] = dim_maps[local_op_name]

    # 定义一个 reshape_strategy 函数，并注册到 aten_op_overload 对应的策略
    @register_op_strategy(aten_op_overload, schema_info=schema_info)
    def reshape_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
        # 调用 dim_map 函数，获取维度映射规则
        rules = dim_map(*op_schema.args_schema, **op_schema.kwargs_schema)
        # 获取输入策略的形状信息
        input_strategy = cast(OpStrategy, op_schema.args_schema[0])
        global_in_shape = input_strategy.shape
        assert global_in_shape is not None, "Shape required."

        # 创建输出策略对象
        output_strategy = OpStrategy([])
        for input_placement_strategy in input_strategy.strategies:
            input_src_spec = input_placement_strategy.output_spec

            # 通过 propagate_shape_and_sharding 函数传播形状和分片信息
            input_tgt_placements, output_placements = propagate_shape_and_sharding(
                input_src_spec.placements,
                tuple(global_in_shape),
                rules,
                mesh.shape,
            )

            # 创建输入目标规格对象 input_tgt_spec
            input_tgt_spec = DTensorSpec(
                placements=tuple(input_tgt_placements),
                mesh=input_src_spec.mesh,
                tensor_meta=input_src_spec.tensor_meta,
            )

            # 生成重新分配成本列表
            redistribute_costs = [
                generate_redistribute_costs(input_strategy, input_tgt_spec)
            ]

            # 创建输出规格对象 output_spec，并添加到 output_strategy 中
            output_spec = DTensorSpec(mesh=mesh, placements=tuple(output_placements))
            output_strategy.strategies.append(
                PlacementStrategy(
                    output_specs=output_spec,
                    input_specs=(input_tgt_spec,),
                    redistribute_cost=redistribute_costs,
                )
            )

        # 返回输出策略对象
        return output_strategy


# 依次注册不同操作对应的策略映射
register_op_strategy_map(aten.squeeze.default, torch.squeeze)
register_op_strategy_map(
    aten.squeeze.dim, torch.squeeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.view.default, Tensor.view, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.reshape.default, torch.reshape, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten._unsafe_view.default, Tensor.view, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.unsqueeze.default, torch.unsqueeze, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.expand.default, Tensor.expand, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.permute.default, torch.permute, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.repeat.default, Tensor.repeat, schema_info=RuntimeSchemaInfo(1)
)
register_op_strategy_map(
    aten.transpose.int, torch.transpose, schema_info=RuntimeSchemaInfo(1)
)
# 注册操作策略映射，将torch.view_as_complex函数映射到aten.view_as_complex.default操作上
register_op_strategy_map(aten.view_as_complex.default, torch.view_as_complex)

# 注册操作策略映射，将torch.view_as_real函数映射到aten.view_as_real.default操作上
register_op_strategy_map(aten.view_as_real.default, torch.view_as_real)
```