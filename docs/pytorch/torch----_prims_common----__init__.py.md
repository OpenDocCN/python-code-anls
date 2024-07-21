# `.\pytorch\torch\_prims_common\__init__.py`

```py
# mypy: allow-untyped-defs
# 从未来导入annotations特性，允许使用类型注解
from __future__ import annotations

# 导入标准库模块和第三方库模块
import operator
import warnings
import weakref

# 导入contextlib模块中的nullcontext函数
from contextlib import nullcontext

# 导入枚举类型模块
from enum import Enum

# 导入functools模块中的cmp_to_key和reduce函数
from functools import cmp_to_key, reduce

# 导入类型注解相关的模块和类
from typing import (
    Any,
    Callable,
    cast,
    List,
    NamedTuple,
    Optional,
    overload,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

# 导入类型扩展模块中的deprecated和TypeAlias
from typing_extensions import deprecated, TypeAlias

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 在类型检查期间导入sympy模块以提供代码智能提示功能，即使用户代码中没有显式导入这些模块也可以
    import sympy

# 导入torch库
import torch

# 从torch模块中导入sym_float, sym_int和sym_max函数
from torch import sym_float, sym_int, sym_max

# 定义类型别名ShapeType，可以是torch.Size、List[int]或Tuple[int, ...]
ShapeType: TypeAlias = Union[torch.Size, List[int], Tuple[int, ...]]

# 定义类型别名StrideType，可以是List[int]或Tuple[int, ...]
StrideType: TypeAlias = Union[List[int], Tuple[int, ...]]

# 定义类型别名DimsType，可以是int、List[int]或Tuple[int, ...]
DimsType: TypeAlias = Union[int, List[int], Tuple[int, ...]]

# 定义类型别名DimsSequenceType，可以是List[int]或Tuple[int, ...]
DimsSequenceType: TypeAlias = Union[List[int], Tuple[int, ...]]

# 定义类型别名NumberTypeType，可以是Type[bool]、Type[int]、Type[float]、Type[complex]
NumberTypeType: TypeAlias = Union[Type[bool], Type[int], Type[float], Type[complex]]

# 定义类型别名NumberType，可以是bool、int、float、complex、torch.SymInt或torch.SymFloat
NumberType: TypeAlias = Union[bool, int, float, complex]

# 定义类型别名RealNumberType，可以是bool、int或float
RealNumberType: TypeAlias = Union[bool, int, float]

# 定义元组类型Number，包括bool、int、float、complex、torch.SymInt、torch.SymFloat和torch.SymBool
Number = (bool, int, float, complex, torch.SymInt, torch.SymFloat, torch.SymBool)

# 定义Dim类型为int
Dim = int

# 定义IntLike类型，可以是int或torch.SymInt
IntLike = (int, torch.SymInt)

# 定义FloatLike类型，可以是float或torch.SymFloat
FloatLike = (float, torch.SymFloat)

# 定义BoolLike类型，可以是bool或torch.SymBool
BoolLike = (bool, torch.SymBool)

# 定义IntWithoutSymInt类型为int
IntWithoutSymInt = int

# 定义FloatWithoutSymFloat类型为float
FloatWithoutSymFloat = float

# 定义DeviceLikeType类型别名，可以是str、torch.device或int
DeviceLikeType: TypeAlias = Union[str, torch.device, int]

# 定义Tensor类型别名，表示torch.Tensor类型
Tensor = torch.Tensor

# 定义torch_function_passthrough集合，包括torch库中的各种函数和属性
torch_function_passthrough = {
    torch.device,
    torch.sym_not,
    torch.sym_float,
    torch.sym_int,
    torch.sym_max,
    torch.sym_min,
    torch._sym_sqrt,  # type: ignore[attr-defined]
    torch.sym_ite,
    torch.Tensor.dim,
    torch.Tensor.ndim.__get__,  # type: ignore[attr-defined]
    torch.Tensor.numel,
    torch.Tensor.size,
    torch.Tensor.storage_offset,
    torch.Tensor.stride,
    torch.Tensor.dtype.__get__,  # type: ignore[attr-defined]
    torch.Tensor.is_sparse.__get__,  # type: ignore[attr-defined]
    torch.Tensor.shape.__get__,  # type: ignore[attr-defined]
    torch.Tensor.device.__get__,  # type: ignore[attr-defined]
    torch.Tensor.requires_grad.__get__,  # type: ignore[attr-defined]
    torch.Tensor.layout.__get__,  # type: ignore[attr-defined]
    torch.Tensor.is_contiguous,
    torch.Tensor.__format__,
    torch.Tensor.__repr__,
    torch.Tensor.requires_grad.__get__,  # type: ignore[attr-defined]
    torch.Tensor.__getitem__,
}

# 定义类型别名TensorLikeType，表示torch.Tensor类型
TensorLikeType = torch.Tensor

# 定义TensorLike类型别名，表示torch.Tensor类型
TensorLike = torch.Tensor

# 定义类型别名TensorSequenceType，可以是List[TensorLikeType]或Tuple[TensorLikeType, ...]
TensorSequenceType: TypeAlias = Union[List[TensorLikeType], Tuple[TensorLikeType, ...]]
# 定义类型别名，TensorOrNumberLikeType 可以是 TensorLikeType 或 NumberType 类型
TensorOrNumberLikeType: TypeAlias = Union[TensorLikeType, NumberType]

# 定义常量 CustomOutParamAnnotation，用于特定输出参数的注解
CustomOutParamAnnotation = "__custom_out_param__"

# 比较两个形状类型 ShapeType 的参数 a 和 b 是否具有相同的形状
def same_shape(a: ShapeType, b: ShapeType, *, allow_rhs_unbacked=False) -> bool:
    # 导入 torch.fx.experimental.symbolic_shapes 模块中的 guard_size_oblivious 函数
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious
    
    # 如果 a 和 b 的长度不相等，则返回 False
    if len(a) != len(b):
        return False
    
    # 遍历 a 和 b 中的元素对
    for x, y in zip(a, b):
        # 如果 allow_rhs_unbacked 为 True，且 y 是 torch.SymInt 类型，则继续下一次循环
        if allow_rhs_unbacked:
            # TODO: 我们应该检查这些符号是否彼此一致
            if isinstance(y, torch.SymInt):
                continue
        
        # NB: 通常情况下，你不会期望在这里执行一个无视大小的保护
        # 这里确实没有广播，但实际上在某些情况下，我们使用这个来确定是否需要对张量进行扩展
        # 因为它们不匹配，所以你确实可能会试图证明在这种情况下 u0 != 1。见 test/test_proxy_tensor.py -k test_cumsum_unbacked
        if guard_size_oblivious(x != y):
            return False
    
    # 如果以上条件都满足，则返回 True
    return True


# 获取与给定类型 t 相关的 Python 类型，如果 t 是 torch.SymFloat、torch.SymInt 或 torch.SymBool，则返回相应的 float、int 或 bool 类型
def _maybe_get_pytype(t):
    if t is torch.SymFloat:
        return float
    elif t is torch.SymInt:
        return int
    elif t is torch.SymBool:
        return bool
    else:
        return t


# TODO: 考虑使用 torch.testing.assert_close 代替，增加一个仅比较元数据的选项
# 比较两个张量类似对象 a 和 b 的元数据，包括形状、dtype 和 device，可选择检查 strides 和共轭
def compare_tensor_meta(
    a: TensorLikeType,
    b: TensorLikeType,
    check_strides=False,
    *,
    allow_rhs_unbacked=False,
    check_conj=True,
):
    """
    检查两个类似张量对象的形状、dtype 和 device 是否相同。

    未来，此函数还将验证额外的元数据，如 strides。
    """
    # 断言 a 和 b 是 TensorLike 类型的对象
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)

    # 如果 a 和 b 的形状不相等，则抛出 AssertionError
    if not same_shape(a.shape, b.shape, allow_rhs_unbacked=allow_rhs_unbacked):
        msg = f"Shapes {a.shape} and {b.shape} are not equal!"
        raise AssertionError(msg)

    # 如果 a 和 b 的 dtype 不相等，则抛出 AssertionError
    if a.dtype != b.dtype:
        msg = f"Dtypes {a.dtype} and {b.dtype} are not equal!"
        raise AssertionError(msg)

    # 如果 a 和 b 的 device 不相等，则根据特定情况处理，或者抛出 AssertionError
    if a.device != b.device:
        # 处理特殊情况 "cuda:0" vs "cuda"
        # TODO: 我们应该审查为什么会出现这种情况，并考虑修复
        if (str(a.device) == "cuda:0" or str(a.device) == "cuda") and (
            str(b.device) == "cuda:0" or str(b.device) == "cuda"
        ):
            pass
        else:
            msg = f"Devices {a.device} and {b.device} are not equal!"
            raise AssertionError(msg)

    # Stride 检查目前已禁用，参见 https://github.com/pytorch/pytorch/issues/78050
    # 如果需要检查数组的步幅信息
    if check_strides:
        # 调用函数检查数组 a 和 b 是否具有相同的重要步幅
        same_strides, idx = check_significant_strides(a, b)
        # 如果步幅不同，则抛出运行时错误并给出具体信息
        if not same_strides:
            msg = f"Stride mismatch! Strides are {a.stride()} and {b.stride()} (mismatched at {idx})!"
            raise RuntimeError(msg)

        # 如果存储偏移量不同，则抛出运行时错误并给出具体信息
        if a.storage_offset() != b.storage_offset():
            msg = f"Storage offset mismatch! Storage offsets are {a.storage_offset()} and {b.storage_offset()}!"
            raise RuntimeError(msg)

    # 如果需要检查共轭属性
    if check_conj:
        # 如果数组 a 和 b 的共轭属性不一致，则抛出运行时错误并给出具体信息
        if a.is_conj() != b.is_conj():
            raise RuntimeError(
                f"Conj mismatch! is_conj is set to {a.is_conj()} and {b.is_conj()}"
            )

    # 如果数组 a 和 b 的负号属性不一致，则抛出运行时错误并给出具体信息
    if a.is_neg() != b.is_neg():
        raise RuntimeError(
            f"Neg mismatch! is_neg is set to {a.is_neg()} and {b.is_neg()}"
        )
def _check_strides_helper(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True, significant_only=True
) -> Tuple[bool, Optional[int]]:
    # NOTE: only on CUDA because CPU elementwise strides are incorrect in PyTorch
    # See https://github.com/pytorch/pytorch/issues/77553
    # Only compares strides that are "meaningful" -- strides for dimensions with length > 1
    # and for tensors with more than one element
    
    # 检查张量的步长是否符合要求
    if (
        not only_cuda or a.device.type == "cuda" or b.device.type == "cuda"
    ) and a.numel() > 0:
        # 遍历张量的每个维度
        for idx in range(a.ndim):
            # 检查是否只比较重要的维度步长（长度大于1的维度）
            check = not significant_only or a.shape[idx] > 1
            # 如果两个张量在该维度上的步长不相等，并且满足重要性检查条件，则返回 False 和该维度索引
            if a.stride()[idx] != b.stride()[idx] and check:
                return False, idx

    return True, None


def check_significant_strides(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True
) -> Tuple[bool, Optional[int]]:
    # 检查重要的步长
    return _check_strides_helper(a, b, only_cuda=only_cuda, significant_only=True)


def check_all_strides(
    a: TensorLikeType, b: TensorLikeType, *, only_cuda=True
) -> Tuple[bool, Optional[int]]:
    # 检查所有的步长
    return _check_strides_helper(a, b, only_cuda=only_cuda, significant_only=False)


# This function is equivalent to compute_contiguous() from TensorImpl.cpp
def is_contiguous(a: TensorLikeType) -> bool:
    """
    Tests whether a tensor is contiguous or not.

    Tensors are contiguous when they have no elements,
    one element, or when they have "nested" strides.
    """
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 检查张量是否是连续的
    if guard_size_oblivious(a.numel() < 2):
        return True

    expected_stride = 1
    # 遍历张量的形状和步长
    for x, y in reversed(tuple(zip(a.shape, a.stride()))):
        # 如果维度长度为1，则跳过步长检查
        if guard_size_oblivious(x == 1):
            continue

        # 如果步长不符合预期，则返回 False
        if guard_size_oblivious(y != expected_stride):
            return False
        expected_stride = expected_stride * x

    return True


# This function is equivalent to compute_channels_last_contiguous_2d() in TensorImpl.cpp
def is_channels_last_contiguous_2d(a: Tensor) -> bool:
    # 检查是否是二维通道最后连续
    if a.ndim != 4:
        return False

    expected_stride = 1
    # 按照 NHWC 的顺序检查维度和步长
    for idx in (1, 3, 2, 0):
        length = a.shape[idx]
        if length == 1:
            continue

        stride = a.stride()[idx]
        if stride != expected_stride:
            return False

        expected_stride *= length

    return True


def is_channels_last_contiguous_3d(a: Tensor) -> bool:
    # 检查是否是三维通道最后连续
    if a.ndim != 5:
        return False

    expected_stride = 1
    # 按照 NDHWC 的顺序检查维度和步长
    for idx in (1, 4, 3, 2, 0):
        length = a.shape[idx]
        if length == 1:
            continue

        stride = a.stride()[idx]
        if stride != expected_stride:
            return False

        expected_stride *= length

    return True


_memory_formats = {
    torch.contiguous_format,
    torch.preserve_format,
    torch.channels_last,
}
    # Importing the "channels_last_3d" submodule from the "torch" module
    torch.channels_last_3d,
}

# 定义一个函数，用于验证内存格式是否合法
def validate_memory_format(memory_format: torch.memory_format):
    # 调用 torch._check 函数，检查 memory_format 是否在 _memory_formats 中
    torch._check(
        memory_format in _memory_formats,
        lambda: f"Received unknown memory format {memory_format}!",
    )


# 定义一个函数，判断张量是否符合指定的内存格式
def is_contiguous_for_memory_format(  # type: ignore[return]
    a: Tensor, *, memory_format: torch.memory_format
) -> bool:
    # 调用 validate_memory_format 函数，验证 memory_format 是否合法
    validate_memory_format(memory_format)

    # 根据不同的 memory_format，调用相应的函数进行判断
    if memory_format == torch.contiguous_format:
        return is_contiguous(a)
    if memory_format == torch.channels_last:
        return is_channels_last_contiguous_2d(a)
    if memory_format == torch.channels_last_3d:
        return is_channels_last_contiguous_3d(a)

    # 如果不支持的 memory_format，则抛出异常
    torch._check(
        False,
        lambda: f"is_contiguous received unsupported memory format {memory_format}",
    )


# NOTE: that tensors with no elements and channels last is ???
# 定义一个函数，判断张量是否是 channels-last 连续的
def is_channels_last_contiguous(a: Tensor) -> bool:
    """
    True when a tensor is channels-last contiguous.

    This requires that:

      - the tensor is conceptually either 4 (NHWC) or 5 (NDHWC) dimensions
      - if we name the tensor's dimensions NCHW or NCDHW, then the strides are such that the
        stride of the 'C' dimension (Cs) is 1 and the strides corresponding to
        each dimension (Xs) can be ordered Cs <= Ws <= Hs <= (Ds) <= Ns and are
        "nested" -- so Ws = Cs * Cl, where Cl is the length of the 'C' dimension,
        for example.
    """
    # 调用 is_channels_last_contiguous_2d 和 is_channels_last_contiguous_3d 函数，判断张量是否 channels-last 连续
    return is_channels_last_contiguous_2d(a) or is_channels_last_contiguous_3d(a)


# 定义一个函数，判断张量是否是非重叠且稠密的
def is_non_overlapping_and_dense(a: Tensor) -> bool:
    """
    True when a tensor is non-overlapping and dense.

    A tensor is non-overlapping and dense when there exists a permutation of
    its dimensions that is contiguous.
    """

    # 引入 torch.fx.experimental.symbolic_shapes 中的 guard_size_oblivious 函数
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 如果张量是稀疏的，则返回 False
    if a.is_sparse:
        return False

    # 如果张量已经是连续的或者是 channels-last 连续的，则返回 True
    if is_contiguous(a) or is_channels_last_contiguous(a):
        return True

    # 下面的代码等价于 TensorImpl.cpp 中的 compute_non_overlapping_and_dense 函数

    # 对于 rank 为一的张量，如果其 stride 是 1，则视为非重叠且稠密的
    if a.ndim == 1:
        return a.stride()[0] == 1

    # 检查是否存在一种 strides 的排列使得张量变为连续的
    # 对 (length, stride) 对按照 stride 进行排序
    #
    # 这种排序是以大小无关的方式进行的，这有助于处理类似 2048*u0 > u0 这样的比较；
    # 我们只希望它返回 True（不用担心 u0 是零的情况）。
    # 定义一个名为 K 的命名元组类，包含 size 和 stride 两个字段
    class K(NamedTuple):
        size: int  # 存储元素大小的属性
        stride: int  # 存储步长的属性
    
        # 定义小于比较操作符，使用 guard_size_oblivious 函数保证比较操作的安全性
        def __lt__(self, other):
            return guard_size_oblivious(self.stride < other.stride)
    
        # 定义大于比较操作符，使用 guard_size_oblivious 函数保证比较操作的安全性
        def __gt__(self, other):
            return guard_size_oblivious(self.stride > other.stride)
    
        # 定义小于等于比较操作符，使用 guard_size_oblivious 函数保证比较操作的安全性
        def __le__(self, other):
            return guard_size_oblivious(self.stride <= other.stride)
    
        # 定义大于等于比较操作符，使用 guard_size_oblivious 函数保证比较操作的安全性
        def __ge__(self, other):
            return guard_size_oblivious(self.stride >= other.stride)
    
        # 定义等于比较操作符，使用 guard_size_oblivious 函数保证比较操作的安全性
        def __eq__(self, other):
            return guard_size_oblivious(self.stride == other.stride)
    
    # 使用 map 函数创建 K 类的实例，并按照 stride 排序得到长度和步长的列表
    lengths_and_strides = sorted(map(K, a.shape, a.stride()))
    
    # 初始化期望的步长为 1
    expected_stride = 1
    
    # 遍历长度和步长的列表
    for length, stride in lengths_and_strides:
        # 如果长度为 1，则忽略当前步长的检查
        if guard_size_oblivious(length == 1):
            continue
    
        # 如果当前步长不等于期望的步长，返回 False
        if stride != expected_stride:
            return False
    
        # 更新期望的步长为当前长度乘以之前的期望步长
        expected_stride *= length
    
    # 若所有步长符合预期，则返回 True
    return True
# 根据 TensorIterator.cpp 中的实现，但注意到 [Computing output strides] 部分有误，
# 因为它声称即使不是“非重叠且密集”的，步长也会保留，但这是错误的。
# 元素操作的输出总是具有非重叠且密集的步长。
# 这也是错误的，因为它没有模拟 TensorIterator 的短路，可能导致不同的步长。

# 计算元素操作的逻辑到物理排列的排列顺序
def compute_elementwise_output_logical_to_physical_perm(
    *tensors, _skip_checks=False
) -> List[int]:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 如果没有跳过检查，并且张量数量为0，则抛出错误
    if not _skip_checks and len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)

    # 如果没有跳过检查，则检查所有张量是否具有相同的形状
    if not _skip_checks:
        check_same_shape(*tensors, allow_cpu_scalar_tensors=True)

    # 将输入的参数过滤为真实的张量
    if not _skip_checks:
        tensors = tuple(
            a
            for a in tensors
            if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
        )

    # 如果张量数量为0，则直接返回空列表
    if len(tensors) == 0:
        return []

    # 如果第一个张量的维度为0，则直接返回空列表
    ndim = tensors[0].ndim
    if ndim == 0:
        return []
    # 如果第一个张量的维度为1，则返回一个包含单个元素0的列表
    if ndim == 1:
        return [0]

    # 检查是否连续，遵循快速路径假设
    is_contiguous = True
    for t in tensors:
        is_contiguous = is_contiguous and t.is_contiguous(
            memory_format=torch.contiguous_format
        )

    # 如果所有张量都是连续的，则返回从0到ndim的列表
    if is_contiguous:
        return list(range(ndim))

    # 获取第一个张量的形状
    shape = tensors[0].shape

    # 判断是否需要交换两个索引的位置
    def should_swap(idx_a, idx_b):
        for tensor in tensors:
            stride_a = tensor.stride()[idx_a]
            stride_b = tensor.stride()[idx_b]

            # 如果步长为0，则忽略该维度
            if guard_size_oblivious(stride_a == 0) or guard_size_oblivious(
                stride_b == 0
            ):
                continue

            # 如果步长不同，则根据步长大小进行交换
            if guard_size_oblivious(stride_a < stride_b):
                return -1
            if guard_size_oblivious(stride_a > stride_b):
                return 1

            # 如果步长相同，则根据维度大小进行交换
            if guard_size_oblivious(shape[idx_a] > shape[idx_b]):
                return 1

        # 如果所有张量的步长都为0，或者所有步长相等且所有维度长度相同，则返回0
        return 0

    # 初始化逻辑到物理排列的排序顺序，从后向前
    perm = list(reversed(range(ndim)))

    # 插入排序，支持模糊比较
    # 对于每个维度索引 i，从第二个维度开始循环到最后一个维度
    for i in range(1, ndim):
        # 将 dim1 设置为当前维度索引 i
        dim1 = i
        # 对于当前维度索引 i，从 i-1 开始向前循环到第一个维度
        for dim0 in reversed(range(i)):
            # 调用 should_swap 函数，比较 perm[dim0] 和 perm[dim1] 的顺序
            comparison = should_swap(perm[dim0], perm[dim1])
            # 如果 comparison 大于 0，需要交换 perm[dim0] 和 perm[dim1]
            if comparison > 0:
                perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
                # 更新 dim1 为当前维度索引 dim0
                dim1 = dim0
            # 如果 comparison 小于 0，不再继续交换，跳出循环
            elif comparison < 0:
                break

    # 返回逆序排列后的 perm 列表
    return list(reversed(perm))
# 计算元素级操作的输出步长
def compute_elementwise_output_strides(*tensors) -> Tuple[int, ...]:
    """
    Computes the output strides for elementwise operations.
    """
    # 如果没有传入张量，则抛出数值错误异常
    if len(tensors) == 0:
        msg = "Can't compute elementwise output strides for zero tensors!"
        raise ValueError(msg)

    # 检查所有张量的形状是否相同，允许使用 CPU 标量张量
    check_same_shape(*tensors, allow_cpu_scalar_tensors=True)

    # 筛选出真实的张量对象，并排除 CPU 标量的情况
    tensors = tuple(
        a for a in tensors if isinstance(a, TensorLike) and not is_cpu_scalar_tensor(a)
    )

    # 如果没有有效的张量，直接返回空元组
    if len(tensors) == 0:
        return ()

    # 获取第一个张量的维度数和形状
    ndim = tensors[0].ndim
    shape = tensors[0].shape

    # 如果维度数为0，返回空元组
    if ndim == 0:
        return ()
    # 如果维度数为1，返回包含一个元素的元组
    if ndim == 1:
        return (1,)

    # 计算元素级操作的输出逻辑到物理的排列顺序
    logical_to_physical_perm = compute_elementwise_output_logical_to_physical_perm(
        *tensors, _skip_checks=True
    )
    # 应用排列顺序到形状，得到物理排列后的形状
    permuted_shape = apply_perm(shape, logical_to_physical_perm)  # to physical

    # 生成连续的步长用于排列后的形状
    new_strides = make_contiguous_strides_for(permuted_shape)
    # 将步长应用到逻辑顺序，得到逻辑顺序下的步长
    permuted_strides = apply_perm(
        new_strides, invert_perm(logical_to_physical_perm)
    )  # to logical

    # 返回排列后的步长元组
    return tuple(permuted_strides)


# 应用排列顺序到输入数组
def apply_perm(inp, perm):
    ndim = len(inp)
    permuted_inp = [-1] * ndim
    for idx, x in enumerate(perm):
        permuted_inp[idx] = inp[x]
    return permuted_inp


# 反转排列顺序
def invert_perm(perm):
    ndim = len(perm)
    new_perm = [-1] * ndim
    for idx, x in enumerate(perm):
        new_perm[x] = idx
    return new_perm


#
# 通用的辅助函数
#


# 验证维度长度是否有效
def validate_dim_length(length: int):
    """
    Validates that an object represents a valid
    dimension length.
    """

    # 如果长度是整数或者 torch.SymInt 类型，则检查是否是有效的大小
    if isinstance(length, (int, torch.SymInt)):
        torch._check_is_size(length)
    else:
        # 有时被引入者调用时传入 sympy 表达式
        assert length >= 0


# 验证形状是否有效
def validate_shape(shape: ShapeType):
    """
    Validates that a sequence represents a valid shape.
    """

    # 断言形状是序列类型
    assert isinstance(shape, Sequence), type(shape)
    # 验证每个长度是否有效
    for l in shape:
        validate_dim_length(l)


# 验证步长是否有效
def validate_strides(strides: StrideType):
    """
    Verifies the object specifies valid strides.
    """

    # 断言步长是序列类型
    assert isinstance(strides, Sequence)
    # 验证每个步长是否大于等于0
    for stride in strides:
        assert stride >= 0


# 验证索引是否有效
def validate_idx(rank: int, idx: int):
    """
    Validates that idx is a valid index for the given shape.
    Assumes the index is already canonicalized.
    """

    # 断言索引和维度都是有效的维度
    assert isinstance(idx, Dim)
    assert isinstance(rank, Dim)

    # 确保索引在有效范围内
    assert idx >= 0 and idx < rank or idx == 0


# 验证维度索引是否有效
def validate_dimension_indices(rank: int, indices: DimsSequenceType):
    for idx in indices:
        validate_idx(rank, idx)


# 验证独占索引是否有效
def validate_exclusive_idx(rank: int, ex_idx: int):
    """
    Validates that ex_idx is a valid exclusive index
    for the given shape.
    """

    # 断言独占索引和维度都是有效的维度
    assert isinstance(ex_idx, Dim)
    assert isinstance(rank, Dim)
    # 确保独占索引在有效范围内
    assert ex_idx > 0 and ex_idx <= rank
# "Wraps" a dim (up to one time) for the given rank, allowing dims to be
# specified using negative indices. If `wrap_scalar` is true then scalar
# tensors of rank 0 will allow dimensions in the range [-1, 0]. Otherwise,
# idx should be in the range [-rank, rank-1].
def canonicalize_dim(rank: int, idx: int, wrap_scalar: bool = True) -> int:
    # 检查 rank 是否为负数，若是则抛出 IndexError 异常
    if rank < 0:
        msg = f"Rank cannot be negative but got {rank}"
        raise IndexError(msg)

    # 若 rank 为 0，则根据 wrap_scalar 参数确定是否允许使用 [-1, 0] 范围的维度
    if rank == 0:
        if not wrap_scalar:
            msg = f"Dimension specified as {idx} but tensor has no dimensions"
            raise IndexError(msg)
        rank = 1

    # 如果 idx 在 [0, rank) 范围内，则直接返回 idx
    if idx >= 0 and idx < rank:
        return idx

    # 如果 idx 为负数，则将其转换为正数索引 _idx
    if idx < 0:
        _idx = idx + rank
    else:
        _idx = idx

    # 最终检查 _idx 是否在有效范围内，否则抛出 IndexError 异常
    if _idx < 0 or _idx >= rank:
        # 和 aten/src/ATen/WrapDimUtils.h:49 中相同的错误信息
        msg = f"Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {idx})"
        raise IndexError(msg)

    return _idx


# Takes a dimension or sequence of dimensions and "wraps" them,
# mapping negative offsets to positive ones
@overload
def canonicalize_dims(
    rank: int, indices: Sequence[int], wrap_scalar: bool = True
) -> Tuple[int, ...]:
    pass


@overload
def canonicalize_dims(rank: int, indices: int, wrap_scalar: bool = True) -> int:
    pass


def canonicalize_dims(rank, indices, wrap_scalar=True):
    # 如果 indices 是 Dim 类型，则调用 canonicalize_dim 处理单个维度
    if isinstance(indices, Dim):
        return canonicalize_dim(rank, indices, wrap_scalar)

    # 否则将 indices 视为维度序列，并逐个调用 canonicalize_dim 处理
    return tuple(canonicalize_dim(rank, x, wrap_scalar) for x in indices)


def is_valid_permutation(rank: int, perm: DimsSequenceType) -> bool:
    """
    Validates that perm is a permutation of length rank.
    """
    # 检查 perm 是否为 Sequence 类型，如果不是则返回 False
    if not isinstance(perm, Sequence):
        return False

    # 检查 perm 是否为 rank 长度的排列，如果不是则返回 False
    if not (tuple(sorted(perm)) == tuple(range(0, rank))):
        return False

    return True


def is_same_shape(a: Sequence, b: Sequence) -> bool:
    """
    Compares two shapes a and b, returning True if they are the same
    (their ranks and corresponding lengths match) and False otherwise.
    """
    # 比较两个形状 a 和 b 是否相同，如果相同返回 True，否则返回 False
    return tuple(a) == tuple(b)


def is_cpu_scalar_tensor(a: Any) -> bool:
    # 检查 a 是否为 TensorLike 类型且维度为 0 且设备类型为 "cpu"
    return isinstance(a, TensorLike) and a.ndim == 0 and a.device.type == "cpu"


def check_same_device(*args, allow_cpu_scalar_tensors):
    """
    Checks that all Tensors in args have the same device.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices, unless one is a CPU scalar tensor and allow_cpu_scalar_tensors is True
    """
    # 如果参数数量小于等于 1，则直接返回，无需进一步检查
    if len(args) <= 1:
        return

    # 注意：不能初始化 device 为第一个参数的设备类型（可能不存在）
    device = None
    # 遍历参数列表args中的每个参数
    for arg in args:
        # 检查参数是否为数字类型，如果是则继续下一次循环
        if isinstance(arg, Number):
            continue
        # 检查参数是否为张量（TensorLike）类型
        elif isinstance(arg, TensorLike):
            # 如果允许CPU标量张量，并且当前参数arg是CPU标量张量，则继续下一次循环
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue

            # 如果设备参数device为None，则将当前参数arg的设备赋值给device
            if device is None:
                device = arg.device

            # 如果当前参数arg的设备与设备参数device不一致，抛出运行时错误
            if device != arg.device:
                msg = (
                    "Tensor on device "
                    + str(arg.device)
                    + " is not on the expected device "
                    + str(device)
                    + "!"
                )
                raise RuntimeError(msg)
        else:
            # 如果参数类型未被预期，则抛出运行时错误，提示出错的类型
            msg = (
                "Unexpected type when checking for same device, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)
# 将设备标准化为 torch.device 对象
def canonicalize_device(device: DeviceLikeType) -> torch.device:
    # 如果 device 已经是 torch.device 对象，则直接返回
    if isinstance(device, torch.device):
        return device

    # 否则，确保 device 是字符串类型，并将其转换为 torch.device 对象后返回
    assert isinstance(device, str)
    return torch.device(device)


# 检查 args 中的所有 Tensor 是否具有相同的形状
def check_same_shape(*args, allow_cpu_scalar_tensors: bool):
    """
    Checks that all Tensors in args have the same shape.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensor objects in args have different devices
    """
    shape = None

    for arg in args:
        # 如果 arg 是数字类型，则跳过
        if isinstance(arg, Number):
            continue
        # 如果 arg 是 TensorLike 类型
        elif isinstance(arg, TensorLike):
            # 如果允许使用 CPU 标量 Tensor，并且 arg 是 CPU 标量 Tensor，则跳过
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                continue

            # 如果 shape 还未初始化，则用 arg 的形状初始化 shape
            if shape is None:
                shape = arg.shape

            # 检查当前 Tensor 的形状是否与 shape 相同，否则抛出异常
            if not is_same_shape(shape, arg.shape):
                msg = f"Shape {arg.shape} is not the expected shape {shape}!"
                raise RuntimeError(msg)
        else:
            # 如果 arg 不是数字类型也不是 TensorLike 类型，则抛出异常
            msg = (
                "Unexpected type when checking for same shape, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)


# 从一个或多个 Tensor 参数中提取公共形状（如果存在），忽略数字参数
def extract_shape(*args, allow_cpu_scalar_tensors: bool) -> Optional[ShapeType]:
    shape = None
    scalar_shape = None

    for arg in args:
        # 如果 arg 是数字类型，则跳过
        if isinstance(arg, Number):
            continue
        # 如果 arg 是 TensorLike 类型
        elif isinstance(arg, TensorLike):
            # 如果允许使用 CPU 标量 Tensor，并且 arg 是 CPU 标量 Tensor，则记录其形状到 scalar_shape
            if allow_cpu_scalar_tensors and is_cpu_scalar_tensor(arg):
                scalar_shape = arg.shape
                continue

            # 如果 shape 还未初始化，则用 arg 的形状初始化 shape
            if shape is None:
                shape = arg.shape

            # 检查当前 Tensor 的形状是否与 shape 相同，不同则返回 None
            if not is_same_shape(shape, arg.shape):
                return None
        else:
            # 如果 arg 不是数字类型也不是 TensorLike 类型，则返回 None
            return None

    # 如果 shape 不为 None，则返回 shape；否则返回 scalar_shape
    return shape if shape is not None else scalar_shape


# 从变长参数中提取可能以列表/元组或 varargs 形式传递的维度信息
def extract_dims_from_varargs(
    dims: Union[DimsSequenceType, Tuple[DimsSequenceType, ...]]
) -> DimsSequenceType:
    if dims and isinstance(dims[0], Sequence):
        # 如果 dims 不为空且 dims 的第一个元素是序列，则假设 dims 只有一个元素，并返回该元素
        assert len(dims) == 1
        dims = cast(Tuple[DimsSequenceType], dims)
        return dims[0]
    else:
        # 否则，直接返回 dims
        return cast(DimsSequenceType, dims)


# 从 varargs 中提取形状信息
def extract_shape_from_varargs(
    shape: Union[ShapeType, Tuple[ShapeType]],
    validate=True,
) -> Tuple[int, ...]:
    """
    Returns a shape from varargs.

    In PyTorch, operations that accept shapes often accept them as varargs, like
    foo(*shape). However a user can pass the shape as a sequence of integers,
    like this:

      foo(1, 2, 3)

    or as a sequence of integers

      foo((1, 2, 3))

    In the first case shape will be a tuple of integers, and in the second case it's a tuple
    """
    # 直接返回 shape，validate 参数未使用
    return shape
    # 如果 shape 是一个长度为1的元组，并且元素是一个序列，将其解包成序列
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = shape[0]

    # 如果 validate 为真，则验证 shape 的有效性
    if validate:
        validate_shape(shape)  # type: ignore[arg-type]

    # 返回经过验证或解包后的 shape
    return shape  # type: ignore[return-value]
# 推断两个形状的维度，并返回扩展后的形状尺寸元组
def infer_size_shapes(a: ShapeType, b: ShapeType) -> Tuple[int, ...]:
    # 确定维度的最大值
    ndim = max(len(a), len(b))
    # 创建一个大小为 ndim 的全零列表，用于存放扩展后的尺寸
    expandedSizes = [0] * ndim

    # 逆序遍历维度
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        # 计算在数组 a 中的索引
        dimA = len(a) - 1 - offset
        # 计算在数组 b 中的索引
        dimB = len(b) - 1 - offset
        # 获取数组 a 在当前维度的大小，如果超出索引范围则默认为 1
        sizeA = a[dimA] if dimA >= 0 else 1
        # 获取数组 b 在当前维度的大小，如果超出索引范围则默认为 1
        sizeB = b[dimB] if dimB >= 0 else 1

        # 使用 Torch 的检查函数，验证两个维度的大小是否兼容
        torch._check(
            (sizeA == sizeB) or (sizeA == 1) or (sizeB == 1),
            lambda: (
                f"The size of tensor a ({sizeA}) must match the size of "
                f"tensor b ({sizeB}) at non-jagged dimension {i}"
            ),
        )

        # 将扩展后的维度大小存入列表中
        expandedSizes[i] = sizeB if sizeA == 1 else sizeA

    # 返回扩展后的形状尺寸元组
    return tuple(expandedSizes)


def infer_size(shape: ShapeType, numel: int) -> Tuple[int, ...]:
    """
    推断形状中大小为 -1 的维度（如果存在）。
    同时检查新形状是否与元素数量兼容。
    """
    dim = None  # 初始化推断的维度为 None
    newsize = 1  # 初始化新的尺寸为 1
    for i, d in enumerate(shape):
        if d == -1:
            # 使用 Torch 的检查函数确保只有一个维度可以被推断
            torch._check(dim is None, lambda: "only one dimension can be inferred")
            dim = i  # 记录推断的维度索引
        elif d >= 0:
            newsize *= d  # 计算非负维度的总尺寸
        else:
            # 使用 Torch 的检查函数，如果维度小于零则抛出异常
            torch._check(False, lambda: f"invalid shape dimension {d}")

    if dim is None:
        # 如果没有维度可以推断，检查新形状是否与元素数量兼容
        torch._check(
            numel == newsize,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
    else:
        # 如果有维度可以推断，进一步检查条件是否满足
        from torch.fx.experimental.symbolic_shapes import definitely_true

        torch._check(
            newsize != 0,
            lambda: (
                f"cannot reshape tensor of 0 elements into shape {list(shape)} because the "
                f"unspecified dimension size -1 can be any value and is ambiguous"
                if definitely_true(numel == 0)
                else f"shape '{list(shape)}' is invalid for input of size {numel}"
            ),
        )
        # 检查新形状是否能整除元素数量
        torch._check(
            numel % newsize == 0,
            lambda: f"shape '{list(shape)}' is invalid for input of size {numel}",
        )
        # 将推断的维度大小更新到形状列表中
        shape = list(shape)
        shape[dim] = numel // newsize
        # 使用 Torch 的尺寸检查函数，确保维度大小符合要求
        torch._check_is_size(shape[dim])

    # 返回更新后的形状尺寸元组
    return tuple(shape)


# 整数类型的列表，包括各种 Torch 支持的整数类型
_integer_dtypes = (
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
# 定义了低精度数据类型的元组，包括 torch.float16, torch.bfloat16, torch.complex32
_low_precision_dtypes = (torch.float16, torch.bfloat16, torch.complex32)
# 定义了复数数据类型的元组，包括 torch.complex32, torch.complex64, torch.complex128
_complex_dtypes = (torch.complex32, torch.complex64, torch.complex128)

# 检查给定的 dtype 是否为布尔类型
def is_boolean_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype is torch.bool

# 检查给定的 dtype 是否为整数类型
def is_integer_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _integer_dtypes

# 检查给定的 dtype 是否为低精度类型
def is_low_precision_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _low_precision_dtypes

# 检查给定的 dtype 是否为浮点数类型
def is_float_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype.is_floating_point

# 检查给定的 dtype 是否为复数类型
def is_complex_dtype(dtype: torch.dtype) -> bool:
    assert isinstance(dtype, torch.dtype)
    return dtype in _complex_dtypes

# 检查给定的 dtype 是否为梯度类型，即浮点数或复数类型
def is_grad_dtype(dtype: torch.dtype) -> bool:
    """
    检查 dtype 是否可能需要梯度。
    """
    return dtype.is_floating_point or is_complex_dtype(dtype)

# 将复数 dtype 映射到对应的实数 dtype 的字典映射关系
_complex_to_real_dtype_map = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

# 将实数 dtype 映射到对应的复数 dtype 的字典映射关系
_real_to_complex_dtype_map = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}

# 返回与给定的复数 dtype 对应的实数 dtype
def corresponding_real_dtype(dtype: torch.dtype) -> torch.dtype:
    return _complex_to_real_dtype_map[dtype]

# 返回与给定的实数 dtype 对应的复数 dtype
def corresponding_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return _real_to_complex_dtype_map[dtype]

# 计算给定 dtype 对应的 Python 类型 (例如 bool, int, float, complex)
def dtype_to_type(dtype: torch.dtype) -> type:
    """
    计算给定 dtype 对应的 Python 类型 (又名 "type kind")。
    """
    assert isinstance(dtype, torch.dtype)

    if dtype is torch.bool:
        return bool
    if dtype in _integer_dtypes:
        return int
    if dtype.is_floating_point:
        return float
    if dtype in _complex_dtypes:
        return complex

    raise ValueError("Invalid dtype!")

# 计算给定 dtype 对应的 Python 类型构造器
def dtype_to_type_ctor(dtype: torch.dtype) -> Callable[[NumberType], NumberType]:
    """
    计算给定 dtype 对应的 Python 类型构造器。
    """
    assert isinstance(dtype, torch.dtype)

    if dtype is torch.bool:
        return lambda x: bool(x)
    if dtype in _integer_dtypes:
        return sym_int
    if dtype.is_floating_point:
        return sym_float
    if dtype in _complex_dtypes:
        # TODO: type error here is real, replace with sym_complex
        return lambda x: complex(x)  # type: ignore[arg-type]

    raise ValueError("Invalid dtype!")

# 计算给定 Python 类型对应的 dtype
def type_to_dtype(typ: type) -> torch.dtype:
    """
    计算给定 Number 类型对应的 dtype。
    """

    assert isinstance(typ, type)

    if typ in (bool, torch.SymBool):
        return torch.bool
    if typ in (int, torch.SymInt):
        return torch.long
    if typ in (float, torch.SymFloat):
        return torch.get_default_dtype()
    # TODO: sym_complex_float?
    # 检查 typ 是否为 complex 类型
    if typ is complex:
        # 如果是 complex 类型，返回对应的复数数据类型
        return corresponding_complex_dtype(torch.get_default_dtype())

    # 如果 typ 不是 complex 类型，则抛出 ValueError 异常，显示错误消息
    raise ValueError(f"Invalid type {typ}!")
# 定义一个函数，根据输入参数 x 的类型返回其数据类型
def get_dtype(x: Union[torch.Tensor, NumberType]):
    # 如果 x 是 torch.Tensor 类型，则返回其数据类型
    if isinstance(x, torch.Tensor):
        return x.dtype
    else:
        # 否则调用 type_to_dtype 函数返回 x 的数据类型
        return type_to_dtype(type(x))


# 定义一个元组，包含有序的数值类型 bool、int、float 和 complex
_ordered_types = (bool, int, float, complex)


# 定义一个函数，检查输入的数据类型是否为浮点数或复数
def check_fp_or_complex(
    dtype: torch.dtype, fn_name: str, allow_low_precision_dtypes: bool = True
):
    """
    检查输入是否为浮点数或复数。
    如果 allow_low_precision_dtypes 为 True，则允许 float16、bfloat16 和 complex32 类型。
    """
    # 使用 torch._check 检查数据类型是否为浮点数或复数，如果不是则抛出异常
    torch._check(
        is_float_dtype(dtype) or is_complex_dtype(dtype),
        lambda: f"{fn_name}: Expected a floating point or complex tensor as input. Got {dtype}",
    )
    # 如果 allow_low_precision_dtypes 为 False，检查是否为低精度数据类型，如果是则抛出异常
    torch._check(
        allow_low_precision_dtypes or not is_low_precision_dtype(dtype),
        lambda: f"{fn_name}: Half precision dtypes not supported. Got {dtype}",
    )


# 定义一个函数，检查输入的张量是否为矩阵（至少有两个维度）
def check_is_matrix(A: TensorLikeType, f_name: str, arg_name: str = "A"):
    # 使用 torch._check 检查张量 A 的维度是否大于等于 2，如果不是则抛出异常
    torch._check(
        len(A.shape) >= 2,
        lambda: f"{f_name}: The input tensor {arg_name} must have at least 2 dimensions.",
    )


# 定义一个函数，返回两个数值类型中更高级别的类型
def get_higher_type(a: type, b: type) -> type:
    """
    返回两个给定数值类型中更高级别的类型。

    这些类型按顺序排列为 bool -> int -> float -> complex。
    """
    # 获取类型 a 和 b 对应的 Python 类型
    a, b = _maybe_get_pytype(a), _maybe_get_pytype(b)
    # 类型检查
    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f"Expected builtin numeric types, found {a}, {b}")

    # 如果 a 和 b 相同，则返回其中任意一个
    if a is b:
        return a

    # 遍历有序的数值类型，返回比较高级的类型
    for typ in _ordered_types:
        if a is typ:
            return b
        if b is typ:
            return a

    # 如果未找到匹配的类型，则抛出异常
    raise ValueError("Unknown Python scalar type!")


# 定义一个函数，返回两个 torch 数据类型 a 和 b 中更高级别的类型，如果无法比较则返回更高级别的数据类型
def get_higher_dtype(
    a: Optional[Union[torch.dtype, TensorLikeType, NumberType]],
    b: Optional[Union[torch.dtype, TensorLikeType, NumberType]],
) -> Optional[torch.dtype]:
    """
    计算比两个输入数据类型 a 和 b 都弱的最低数据类型。
    """

    # 类型检查
    assert a is None or isinstance(a, (torch.dtype, TensorLike, Number))
    assert b is None or isinstance(b, (torch.dtype, TensorLike, Number))

    # 定义一个内部函数，从输入中提取数据类型
    def _extract_dtype(
        x: Optional[Union[torch.dtype, TensorLikeType, NumberType]]
    ) -> Optional[torch.dtype]:
        if x is None:
            return None
        if isinstance(x, torch.dtype):
            return x
        if isinstance(x, TensorLike):
            return x.dtype
        if isinstance(x, Number):
            return type_to_dtype(type(x))

        # 如果输入类型不符合预期，则抛出异常
        raise RuntimeError("Unexpected type given to _extract_dtype!")

    # 从输入参数中提取数据类型 a 和 b
    a, b = _extract_dtype(a), _extract_dtype(b)

    # 如果 a 和 b 相同，则返回其中一个
    if a is b:
        return a

    # 如果 a 为 None，则返回 b
    if a is None:
        return b

    # 如果 b 为 None，则返回 a
    if b is None:
        return a
    ordered_datatypes = (
        (torch.bool,),                # 定义一个元组 ordered_datatypes，包含不同类型的 Torch 张量数据类型的元组
        (torch.uint8, torch.int8),    # 第一个子元组包含 torch.uint8 和 torch.int8 数据类型
        (torch.int16,),               # 第二个子元组包含 torch.int16 数据类型
        (torch.int32,),               # 第三个子元组包含 torch.int32 数据类型
        (torch.int64,),               # 第四个子元组包含 torch.int64 数据类型
        (torch.float16, torch.bfloat16),  # 第五个子元组包含 torch.float16 和 torch.bfloat16 数据类型
        (torch.float32,),             # 第六个子元组包含 torch.float32 数据类型
        (torch.float64,),             # 第七个子元组包含 torch.float64 数据类型
        (torch.complex32,),           # 第八个子元组包含 torch.complex32 数据类型
        (torch.complex64,),           # 第九个子元组包含 torch.complex64 数据类型
        (torch.complex128,),          # 第十个子元组包含 torch.complex128 数据类型
    )

    for idx, dtypes in enumerate(ordered_datatypes):
        # 遍历 ordered_datatypes 中的每个子元组，同时获取索引 idx 和子元组 dtypes
        if a in dtypes and b in dtypes:
            # 如果 a 和 b 都在当前子元组 dtypes 中，返回下一个子元组中的第一个数据类型
            return ordered_datatypes[idx + 1][0]
        if a in dtypes:
            # 如果 a 在当前子元组 dtypes 中，返回 b
            return b
        if b in dtypes:
            # 如果 b 在当前子元组 dtypes 中，返回 a
            return a

    # 如果未找到匹配的数据类型，抛出运行时错误
    raise RuntimeError("Unexpected termination!")
# 检查是否为 pinned memory，如果是则抛出未实现异常
def check_pin_memory(pin_memory: bool):
    torch._check_not_implemented(
        not pin_memory, lambda: "PrimTorch does not support pinned memory"
    )


# 检查张量的布局是否为 strided，如果不是则抛出未实现异常
def check_layout(layout: torch.layout):
    torch._check_not_implemented(
        layout == torch.strided, lambda: f"PrimTorch doesn't support layout={layout}"
    )


# TODO: maybe unify with can_cast_to?
# 比较两种类型 a 和 b，如果 a 弱于 b 则返回 True
# 弱于的比较顺序为: bool, int, float, complex
def is_weakly_lesser_type(a: type, b: type) -> bool:
    """
    Compares two types, a and b, returning True if a is weakly "less" than b.

    The comparison is determined by the following type ordering: bool, int, float, complex.
    """

    a, b = _maybe_get_pytype(a), _maybe_get_pytype(b)

    # 确保 a 和 b 都是内置数值类型
    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f"Expected builtin numeric types, found {a}, {b}")

    # 根据预定义的顺序 _ordered_types 进行比较
    for typ in _ordered_types:
        if a == typ:
            return True
        if b == typ:
            return False

    raise RuntimeError("Unexpected termination!")


# 检查是否能够安全地将 cast_from 类型转换为 cast_to 类型
# 遍历几个判断函数，如果 cast_to 是复数、浮点数、整数或布尔型，则返回 True
# 如果 cast_from 是其中之一，则返回 False
def can_safe_cast_to(*, cast_to: torch.dtype, cast_from: torch.dtype) -> bool:
    for fn in (is_complex_dtype, is_float_dtype, is_integer_dtype, is_boolean_dtype):
        if fn(cast_to):
            return True
        if fn(cast_from):
            return False

    # 如果未识别的 dtypes，则抛出 ValueError
    raise ValueError(f"Received unknown dtypes {cast_to}, {cast_from}!")


# 检查所有参数是否具有相同的数据类型和相同的 Python 类型
# 涉及的对象可以是 Tensor 或数字
# 当 args 包含非 Tensor 或 Number 类型的对象，或者包含具有不同 dtype 的两个 Tensor，
# 或包含具有不同类型的两个数字时，抛出 RuntimeError
def check_same_dtype(*args):
    """
    Checks that all Tensors in args have the same device and that all Numbers have the
    same corresponding Python type.

    Raises a RuntimeError when:
      - args contains an object whose type is not Tensor or Number
      - two Tensors objects in args have different dtypes
      - two Number objects in args have different types
      - there are Tensors and Numbers in args, and one of those Tensors corresponding
          Python types is different from the type of one of those Numbers
    """
    full_dtype = None
    scalar_type = None
    # 遍历参数列表 args
    for arg in args:
        # 检查当前参数是否为数字类型
        if isinstance(arg, Number):
            # 忽略标量类型检查（将来可能会移除）
            continue
            # 下面的代码块被注释掉了，原本用于处理标量类型检查和错误抛出

            # 如果标量类型未定义，则定义为当前参数的类型
            # if scalar_type is None:
            #     scalar_type = type(arg)

            # 如果当前参数的类型与预期的标量类型不匹配，则抛出错误
            # if scalar_type is not type(arg):
            #     msg = (
            #         "Scalar of type "
            #         + str(type(arg))
            #         + " is not the expected type of "
            #         + str(scalar_type)
            #         + "!"
            #     )
            #     raise RuntimeError(msg)
        
        # 如果当前参数是类 TensorLike 的实例
        elif isinstance(arg, TensorLike):
            # 如果全局数据类型 full_dtype 未定义，则定义为当前参数的数据类型
            if full_dtype is None:
                full_dtype = arg.dtype
            
            # 如果标量类型未定义，则将其定义为当前参数的 Python 类型
            if scalar_type is None:
                scalar_type = dtype_to_type(arg.dtype)

            # 如果全局数据类型与当前参数的数据类型不匹配，则抛出错误
            if full_dtype is not arg.dtype:
                msg = (
                    "Tensor with dtype "
                    + str(arg.dtype)
                    + " is not the expected dtype of "
                    + str(full_dtype)
                    + "!"
                )
                raise RuntimeError(msg)

            # 获取当前参数的 Python 类型，并检查其是否与标量类型匹配，否则抛出错误
            arg_type = dtype_to_type(arg.dtype)
            if arg_type is not scalar_type:
                msg = (
                    "Tensor with corresponding Python type "
                    + str(arg_type)
                    + " is not the expected type of "
                    + str(scalar_type)
                    + "!"
                )
                raise RuntimeError(msg)
        
        # 如果当前参数不是数字类型也不是类 TensorLike 的实例，则抛出意外类型错误
        else:
            msg = (
                "Unexpected type when checking for same dtype, " + str(type(arg)) + "!"
            )
            raise RuntimeError(msg)
# 将数据类型映射到其用于逐元素操作的计算类型
_computation_dtype_map = {
    torch.bfloat16: torch.float32,   # bfloat16 映射到 float32
    torch.float16: torch.float32,    # float16 映射到 float32
    torch.complex32: torch.complex64,  # complex32 映射到 complex64
}


def get_computation_dtype(dtype: torch.dtype) -> torch.dtype:
    # 根据给定的 dtype 返回计算用的 dtype，若未指定则返回原始 dtype
    return _computation_dtype_map.get(dtype, dtype)


# 将数据类型映射到它们在 CPU 累加时的类型
_cpu_acc_type_map = {
    torch.bfloat16: torch.float64,    # bfloat16 映射到 float64
    torch.float16: torch.float64,     # float16 映射到 float64
    torch.float32: torch.float64,     # float32 映射到 float64
    torch.complex32: torch.complex128,  # complex32 映射到 complex128
    torch.complex64: torch.complex128,  # complex64 映射到 complex128
}


def get_acc_type(dtype: torch.dtype, device: torch.device) -> torch.dtype:
    # 确定累加类型，优先选择计算类型（computation_dtype）
    if device.type == "cpu":
        return _cpu_acc_type_map.get(dtype, dtype)
    else:
        return get_computation_dtype(dtype)


# 定义逐元素操作类型提升的种类
class ELEMENTWISE_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)         # 默认类型提升
    NO_OPMATH = (1,)       # 无操作数数学运算
    INT_TO_FLOAT = (2,)    # 整数转浮点数
    ALWAYS_BOOL = (3,)     # 始终布尔类型
    COMPLEX_TO_FLOAT = (4,)  # 复数转浮点数
    BOOL_TO_LONG = (5,)    # 布尔转长整型


# 定义约简操作输出类型的种类
class REDUCTION_OUTPUT_TYPE_KIND(Enum):
    SAME = (0,)                # 输出与输入类型相同
    COMPLEX_TO_FLOAT = (1,)    # 对于复数类型的输出，对应的是实数类型
    KEEP_PROMOTED_TYPE = (2,)  # 保持在操作数数学类型中，例如对于均值操作
    ALWAYS_BOOL = (3,)         # 始终布尔类型输出


# 描述原语的返回类型：
#
#   - NEW，创建新张量
#   - VIEW，返回输入张量的视图
#   - INPLACE，修改一个或多个输入张量
#
# 这些描述符是互斥且穷尽的。
class RETURN_TYPE(Enum):
    NEW = (0,)       # 新张量
    VIEW = (1,)      # 张量视图
    INPLACE = (2,)   # 原地操作
    NONE = (3,)      # 无返回


# TODO: 当 NumberType 包含 sym 类型时，可以简化此处
def number_type(
    x: Union[NumberType, torch.SymInt, torch.SymFloat, torch.SymBool]
) -> Type:
    # 根据输入类型返回相应的 Python 类型
    if isinstance(x, torch.SymInt):
        return int
    elif isinstance(x, torch.SymFloat):
        return float
    elif isinstance(x, torch.SymBool):
        return bool
    else:
        return type(x)


# 根据表达式类型返回相应的 Python 类型
def expr_type(x: sympy.Basic) -> Type:
    import sympy

    if x.kind is sympy.core.kind.BooleanKind:
        return bool
    elif x.is_integer:  # type: ignore[attr-defined]
        return int
    else:
        # 注意：不完全正确，但我们不支持 SymPy 的复数或布尔类型。
        return float


# TODO: 文档化类型提升的种类
def elementwise_dtypes(
    *_args,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
) -> Tuple[torch.dtype, torch.dtype]:
    """
    根据给定的参数和逐元素类型提升的种类，计算计算和结果的数据类型。

    注意，并非所有逐元素操作的输入都一定参与类型提升。
    例如，torch.add 的 "alpha" 参数并不参与类型提升，尽管它可能转换为对应的计算 dtype，
    该类型由类型提升算法决定。
    """
    pass  # 这里是函数体，尚未提供实现
    # 默认的类型提升机制，所有其他类型提升种类都会在此基础上进行调整（见下文），
    # 首先决定使用以下四种有序类型中的哪一种：
    
    # 布尔型 -> 整数型 -> 浮点型 -> 复数型
    
    # 所选类型是上述列表中“最低”的类型，以使所有数字参数具有弱“更低”的类型，
    # 并且所有张量参数对应的dtype具有弱“更低”的类型。
    
    # 一旦确定了类型，就找到了特定的结果dtype。各dtype部分有以下顺序：
    
    # 布尔型 -> uint8, int8 -> int16 -> int32 -> int64 ->
    #   float16, bfloat16 -> float32 -> float64 -> complex32 -> complex64 -> complex128
    
    # 结果dtype的选择方式包括：
    #   - 如果没有张量的dtype与所选的dtype具有相应的类型，则结果dtype是所选类型对应的（默认）dtype
    #     （例如，1.5 + 整数张量的结果dtype是默认的浮点数dtype）
    #   - 如果结果类型是复数，则dtype为：
    #     - 如果没有浮点数或复数张量，则为默认的复数dtype
    #     - 如果存在浮点数或复数张量且其中一个或多个维度，则为这些张量中最高对应复数dtype的复数dtype
    #       （例如，double + cfloat -> cdouble）
    #     - 如果只有浮点数或复数张量的维度为零，则为这些张量中最高对应复数dtype的复数dtype
    #   - 如果前两种情况不适用，则结果dtype是所有输出类型的一个或多个维度中最高的dtype，
    #     如果没有这样的张量，则它是所有输出类型的零维中最高的dtype
    #     （例如，long + half -> half，即使half张量的维度为零）
    
    # “对应复数dtypes”如下：
    #   float16    -> complex32
    #   bfloat16   -> complex64
    #   float32    -> complex64
    #   float64    -> complex128
    #   complex32  -> complex32
    #   complex64  -> complex64
    #   complex128 -> complex128
    
    # 默认的类型提升种类根据上述方法计算，然后使用结果dtype来选择计算dtype，
    # 将低精度浮点数和复数dtype映射如下：
    
    #   float16   -> float32
    #   bfloat16  -> float32
    #   complex32 -> complex64
    
    # 这称为“op math”，NO_OPMATH类型提升种类会禁用此映射，
    # 使计算dtype与选择的结果dtype相同时。NO_OPMATH适用于在张量上执行无数学运算的内核
    # （有关示例，请参见下文）。
    
    # INT_TO_FLOAT类型提升种类将布尔和整数结果dtype映射到默认的浮点数dtype，
    # 并将计算dtype映射到相应的op math dtype。
    The COMPLEX_TO_FLOAT type promotion kind maps complex result dtypes to the corresponding float dtype, following this
    mapping:

        complex32  -> float16
        complex64  -> float32
        complex128 -> float64

    Note that COMPLEX_TO_FLOAT derives the computation dtype as the DEFAULT setting does.

    The BOOL_TO_LONG type promotion kind maps boolean computation and result dtypes to long.

    The ALWAYS_BOOL type promotion kind always sets the result dtype to bool.

    Example operators for each type promotion option:
      DEFAULT                 : add
      NO_OPMATH               : where, nextafter, cat
      INT_TO_FLOAT            : sin
      COMPLEX_TO_FLOAT        : abs
      BOOL_TO_LONG            : pow
      ALWAYS_BOOL             : eq

    """



    # 将参数列表中非空的元素组成元组
    args = tuple(x for x in _args if x is not None)

    # 初始将最高类型设置为布尔类型
    highest_type: type = bool

    # 局部导入 sympy，因为在模块级别急切导入它速度过慢
    # 参见 https://dev-discuss.pytorch.org/t/delving-into-what-happens-when-you-import-torch/1589
    import sympy

    # 遍历参数列表
    for x in args:
        # 检查 x 是否为 Number、TensorLike 或 sympy.Basic 类型之一
        if not isinstance(x, (Number, TensorLike, sympy.Basic)):
            # 如果不是上述类型之一，抛出异常
            msg = f"Unexpected type {str(type(x))} when computing elementwise type promotion!"
            raise ValueError(msg)

        # 根据 x 的类型更新 highest_type
        if isinstance(x, Number):
            highest_type = get_higher_type(highest_type, number_type(x))
        elif isinstance(x, sympy.Basic):
            highest_type = get_higher_type(highest_type, expr_type(x))
        else:
            # x 是 TensorLike 类型
            highest_type = get_higher_type(highest_type, dtype_to_type(x.dtype))

    # 初始化结果 dtype 为 None
    result_dtype = None

    # 定义函数 _find_highest_dtype_filtered，根据条件过滤并找到最高的 dtype
    def _find_highest_dtype_filtered(
        args, filter, *, float_as_complex=False
    ) -> Optional[torch.dtype]:
        zero_dim_tensor_dtype = None
        one_plus_dim_tensor_dtype = None
        # 遍历参数列表 args
        for x in args:
            # 如果 x 是 TensorLike 并且符合过滤条件 filter(x.dtype)
            if isinstance(x, TensorLike) and filter(x.dtype):
                _dtype = x.dtype
                # 如果 float_as_complex 为 True 并且 _dtype 是浮点类型
                if float_as_complex and is_float_dtype(_dtype):
                    # 将 _dtype 转换为对应的复数 dtype
                    _dtype = corresponding_complex_dtype(_dtype)
                # 如果 x 的维度为 0
                if x.ndim == 0:
                    zero_dim_tensor_dtype = get_higher_dtype(
                        zero_dim_tensor_dtype, _dtype
                    )
                else:
                    # x 的维度大于 0
                    one_plus_dim_tensor_dtype = get_higher_dtype(
                        one_plus_dim_tensor_dtype, _dtype
                    )

        # 优先返回具有一维或更多维度的张量的 dtype
        if one_plus_dim_tensor_dtype is not None:
            return one_plus_dim_tensor_dtype

        return zero_dim_tensor_dtype

    # 如果 highest_type 是 float 类型
    if highest_type is float:
        # 调用 _find_highest_dtype_filtered 函数，获取符合条件的 dtype
        result_dtype = _find_highest_dtype_filtered(args, is_float_dtype)
        # 如果 result_dtype 为 None，则使用默认的 torch dtype
        result_dtype = (
            torch.get_default_dtype() if result_dtype is None else result_dtype
        )
    elif highest_type is complex:
        # 如果最高类型是复数，找到包含浮点或复数类型的最高优先级数据类型
        result_dtype = _find_highest_dtype_filtered(
            args,
            lambda x: is_float_dtype(x) or is_complex_dtype(x),
            float_as_complex=True,
        )
        if result_dtype is None:
            # 如果未找到符合条件的数据类型，则使用默认的复数对应的数据类型
            result_dtype = corresponding_complex_dtype(torch.get_default_dtype())
    elif highest_type is int:
        # 如果最高类型是整数，找到包含整数类型的最高优先级数据类型
        result_dtype = _find_highest_dtype_filtered(args, is_integer_dtype)
        # 如果未找到符合条件的数据类型，则使用默认的长整型数据类型
        result_dtype = torch.long if result_dtype is None else result_dtype
    else:
        # 如果最高类型是布尔型
        result_dtype = torch.bool

    if type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT:
        # 如果类型提升的种类是默认的元素级别类型提升种类，则返回计算的数据类型和结果数据类型
        return get_computation_dtype(result_dtype), result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH:
        # 如果类型提升的种类是没有数学操作的元素级别类型提升种类，则返回结果数据类型两次
        return result_dtype, result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
        # 如果类型提升的种类是整数到浮点数的元素级别类型提升种类
        if is_integer_dtype(result_dtype) or is_boolean_dtype(result_dtype):
            # 如果结果数据类型是整数或布尔型，则将结果数据类型设为默认的浮点数数据类型
            result_dtype = torch.get_default_dtype()
        return get_computation_dtype(result_dtype), result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
        # 如果类型提升的种类是复数到浮点数的元素级别类型提升种类
        # 注意：计算仍然可以在复数数据类型中进行
        computation_dtype = get_computation_dtype(result_dtype)
        if is_complex_dtype(result_dtype):
            # 如果结果数据类型是复数，则将其对应的实数数据类型作为结果数据类型
            result_dtype = corresponding_real_dtype(result_dtype)
        return computation_dtype, result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG:
        # 如果类型提升的种类是布尔型到长整型的元素级别类型提升种类
        if is_boolean_dtype(result_dtype):
            # 如果结果数据类型是布尔型，则返回两次长整型
            return torch.long, torch.long
        return get_computation_dtype(result_dtype), result_dtype
    elif type_promotion_kind is ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
        # 如果类型提升的种类是总是转换为布尔型的元素级别类型提升种类
        return get_computation_dtype(result_dtype), torch.bool
    else:
        # 如果未知的类型提升种类，则抛出异常
        raise ValueError(f"Unknown type promotion kind {str(type_promotion_kind)}")
# 定义函数 reduction_dtypes，用于处理张量的数据类型缩减操作
def reduction_dtypes(
    arg,
    output_dtype_kind: REDUCTION_OUTPUT_TYPE_KIND,  # 指定输出数据类型的种类
    dtype: Optional[torch.dtype] = None,  # 可选参数，指定输入张量的数据类型
) -> Tuple[torch.dtype, Optional[torch.dtype]]:
    # 尽管一些缩减操作（如 amin 或 amax）不严格要求类型提升，
    # 所有的数学运算（包括比较）仍然仅针对计算类型定义，
    # 因此这里仍然会发生类型提升。我们在这里显式执行它。
    inp_dtype = dtype if dtype is not None else arg.dtype  # 确定输入张量的数据类型
    computation_dtype = get_computation_dtype(inp_dtype)  # 获取计算使用的数据类型

    if (
        output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.SAME
        or output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
    ):
        result_dtype = dtype if dtype else arg.dtype  # 结果数据类型为指定的 dtype 或输入张量的数据类型
        if (
            output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT
            and is_complex_dtype(result_dtype)  # 如果结果数据类型是复数，则转换为对应的实数数据类型
        ):
            result_dtype = corresponding_real_dtype(result_dtype)
    elif output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.KEEP_PROMOTED_TYPE:
        result_dtype = None  # 保持提升后的类型
    else:  # output_dtype_kind == REDUCTION_OUTPUT_TYPE_KIND.ALWAYS_BOOL
        result_dtype = torch.bool  # 总是返回布尔类型的结果

    return computation_dtype, result_dtype  # 返回计算数据类型和结果数据类型的元组


# 定义函数 make_contiguous_strides_for，返回张量的连续内存步长
def make_contiguous_strides_for(
    shape: ShapeType,  # 张量的形状
    row_major: bool = True  # 是否按行主序计算步长，默认为 True
) -> Tuple[int, ...]:
    """
    返回张量的连续内存步长，如果 row_major=True，则返回 Fortran 连续矩阵批次的步长
    通常在调用 BLAS/LAPACK/cuSolver 等外部库时使用
    """
    validate_shape(shape)  # 验证张量的形状是否有效
    if not shape:
        return ()

    from torch.fx.experimental.symbolic_shapes import is_nested_int

    multiplier = 1
    strides = []
    for l in reversed(shape):
        strides.append(multiplier)  # 添加当前维度的步长
        multiplier *= l if is_nested_int(l) else sym_max(l, 1)  # 计算下一个维度的步长

    result = tuple(reversed(strides))  # 将步长反转为正确顺序

    if row_major:
        return result  # 如果按行主序计算步长，直接返回结果
    else:
        if len(shape) < 2:
            return result  # 如果张量的维度小于 2，直接返回结果
        return result[:-2] + (1, max(shape[-2], 1))  # 否则返回修改后的结果


# 定义函数 make_channels_last_1d_strides_for，返回通道最后内存格式的 1D 张量步长
def make_channels_last_1d_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    torch._check(
        len(shape) == 3,  # 检查张量的维度是否为 3
        lambda: "Only tensors of rank 3 can use the channels_last_1d memory format",  # 如果不是，返回错误信息
    )

    multiplier = 1
    strides = [0] * 3
    for idx in (1, -1, 0):
        # 注意：这里与 make_contiguous_strides_for 有意不同
        # 这与 eager 模式保持一致
        strides[idx] = multiplier  # 设置当前维度的步长
        multiplier *= shape[idx]  # 计算下一个维度的步长

    return tuple(strides)  # 返回步长的元组形式


# 定义函数 make_channels_last_2d_strides_for，暂时未实现，待完善
def make_channels_last_2d_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    # TODO: maybe inform the user of channels_last_3d if rank of the tensor is 5?
    pass  # 暂未实现，待完善
    # 检查张量的维度是否为4，否则引发错误信息
    torch._check(
        len(shape) == 4,
        lambda: "Only tensors of rank 4 can use the channels_last memory format",
    )
    
    # 初始化步幅数组和乘数
    multiplier = 1
    strides = [0] * 4
    
    # 遍历索引序列 (1, -1, -2, 0)
    for idx in (1, -1, -2, 0):
        # 注意：这里故意与 make_contiguous_strides_for 不同
        # 这与 eager 模式保持一致
        # 根据当前索引位置设置步幅值
        strides[idx] = multiplier
        # 更新乘数，用于下一个索引位置的步幅计算
        multiplier *= shape[idx]
    
    # 返回计算得到的步幅数组作为元组
    return tuple(strides)
# 根据给定形状计算通道为最后的3D张量的步幅元组
def make_channels_last_3d_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    # 检查形状是否为5维，只有5维张量才能使用channels_last_3d内存格式
    torch._check(
        len(shape) == 5,
        lambda: "Only tensors of rank 5 can use the channels_last_3d memory format",
    )

    # 初始化步幅列表为全零
    multiplier = 1
    strides = [0] * 5
    # 按照指定顺序设置步幅值
    for idx in (1, -1, -2, -3, 0):
        # 注意：与make_contiguous_strides_for有意不同
        # 这与eager模式保持一致
        strides[idx] = multiplier
        multiplier *= shape[idx]

    # 返回步幅的元组形式
    return tuple(strides)


# 根据给定形状选择对应通道为最后的步幅计算函数
def make_channels_last_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    ndim = len(shape) if isinstance(shape, Sequence) else 1
    # 根据张量维度调用相应的通道为最后步幅计算函数
    if ndim == 3:
        return make_channels_last_1d_strides_for(shape)
    elif ndim == 4:
        return make_channels_last_2d_strides_for(shape)
    elif ndim == 5:
        return make_channels_last_3d_strides_for(shape)
    else:
        # 抛出异常，指示在指定维度下不存在通道为最后格式的步幅
        raise RuntimeError(
            f"no channels last format strides exist in {ndim} dimensions"
        )


# 根据指定维度计算减少后的输出形状
def compute_reduction_output_shape(
    shape: ShapeType, dimensions: Sequence
) -> Tuple[int, ...]:
    # 验证每个指定维度是否在形状的有效范围内
    for idx in dimensions:
        validate_idx(len(shape), idx)

    # 构建新形状列表，排除指定维度后的剩余维度
    new_shape = []
    for idx in range(len(shape)):
        if idx in dimensions:
            continue
        new_shape.append(shape[idx])

    # 返回新形状的元组形式
    return tuple(new_shape)


# 验证维度列表中是否有重复的维度值
def validate_no_repeating_dims(dims: Sequence):
    if len(dims) != len(set(dims)):
        raise RuntimeError("duplicate value in the list of dims")


# 根据形状和指定维度计算减少维度后的维度元组
def reduction_dims(shape: ShapeType, dims: Optional[Sequence]) -> Tuple[int, ...]:
    # 如果未指定dims，则返回所有维度的元组
    if dims is None:
        return tuple(range(len(shape)))
    # 规范化维度并验证是否有重复维度
    dims = tuple(canonicalize_dim(len(shape), idx) for idx in dims)
    validate_no_repeating_dims(dims)
    return dims


# 设置修正值，返回浮点数修正值
def set_correction(
    unbiased: Optional[bool] = None,
    correction: Optional[NumberType] = None,
) -> float:
    # 如果同时指定了修正值和无偏值，抛出异常
    if correction is not None and unbiased is not None:
        raise RuntimeError("cannot specify both correction and unbiased arguments")
    # 如果修正值和无偏值都未指定，则默认使用1.0
    elif correction is None and unbiased is None:
        correction = 1.0
    # 如果只有修正值未指定，则根据无偏值确定修正值
    elif correction is None and unbiased is not None:
        correction = 0.0 if unbiased is False else 1.0
    # 检查修正值类型是否为整数或浮点数
    if not isinstance(correction, (IntLike, FloatLike)):
        raise ValueError("correction argument should be integer or float")
    # 检查修正值是否为非负数
    if correction < 0:
        raise ValueError("correction argument should be non-negative")
    return sym_float(correction)


# 计算给定张量几何结构所需的最小存储长度
def compute_required_storage_length(
    shape: ShapeType, strides: StrideType, storage_offset: int
) -> int:
    """计算以元素为单位的新分配张量存储的最小存储大小。

    示例
    =======

    这是新分配张量存储的大小，以元素为单位

    >>> t = torch.empty((10, 20))
    >>> compute_required_storage_length(t.shape, t.stride(), t.storage_offset())
    200

    >>> # xdoctest: +SKIP(failing)

    """
    >>> t2 = torch.empty_strided((1, 2, 3), (5, 7, 11))
    >>> size = compute_required_storage_length(t2.shape, t2.stride(), t2.storage_offset())
    >>> size == t.storage().size()
    True

    A valid tensor may have a larger storage size, but never smaller

    >>> slice = torch.empty(100)[20:40]
    >>> slice.storage().size()
    100

    >>> compute_required_storage_length(slice.shape, slice.stride(), slice.storage_offset())
    40



    """
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 如果形状没有元素，则直接返回0
    # guard_size_oblivious 函数用于检查形状是否为空
    if guard_size_oblivious(reduce(operator.mul, shape, 1) == 0):
        return 0

    # 计算最大偏移量，考虑到每个维度的步幅和长度
    max_offset = sum((x - 1) * y for x, y in zip(shape, strides))
    # 加1是为了考虑从偏移开始的第一个元素
    return 1 + storage_offset + max_offset
    ```
# 确定给定存储的形状、步幅和偏移量是否有效
def check_in_bounds_for_storage(
    a: torch.TypedStorage, shape: ShapeType, strides: StrideType, storage_offset: int
):
    """
    Determines if the given shape, strides, and offset are valid for the given storage.
    """

    # 计算所需的存储长度
    required_length = compute_required_storage_length(shape, strides, storage_offset)
    # 检查存储的大小是否小于所需长度
    if a.size() < required_length:
        # 构建错误消息
        msg = (
            f"Can't view a storage of size {a.size()} with an offset of {storage_offset}, "
            f"shape of {str(shape)}, and strides of {str(strides)}, "
            f"which requires a storage of size {required_length}"
        )
        # 抛出值错误异常
        raise ValueError(msg)


# NOTE: This function should ideally be removed, but some Meta internal models
# packaged with `torch.package` are using it, so it will have to be removed
# at some point in the future when those models no longer use this function.
@deprecated(
    "`torch._prims_common.check` is deprecated and will be removed in the future. "
    "Please use `torch._check*` functions instead.",
    category=FutureWarning,
)
def check(
    b: bool, s: Callable[[], str], exc_type: Type[Exception] = RuntimeError
) -> None:
    """
    Helper function for raising an error_type (default: RuntimeError) if a boolean condition fails.
    Error message is a callable producing a string (to avoid wasting time
    string formatting in non-error case, and also to make it easier for torchdynamo
    to trace.)

    .. note:: This function is planned for removal in the future. Please use
        `torch._check*` functions instead.
    """
    # 使用 torch._check_with 函数检查布尔条件，若失败则抛出指定类型的异常
    torch._check_with(exc_type, b, s)


# This combines is_channels_last_strides_2d and is_channels_last_strides_3d in
# c10/core/MemoryFormat.h into one function
def are_strides_like_channels_last(
    shape: Sequence[int], strides: Sequence[int]
) -> bool:
    ndim = len(shape)

    # 根据维度数选择不同的维度顺序
    if ndim == 4:
        # 检查 channels_last_2d
        dim_order = [1, 3, 2, 0]
    elif ndim == 5:
        # 检查 channels_last_3d
        dim_order = [1, 4, 3, 2, 0]
    else:
        # 如果不是4维或5维，则返回 False
        return False

    # 检查第二个维度的步幅是否为零
    if strides[1] == 0:
        return False

    min = 0
    # 遍历维度顺序
    for d in dim_order:
        # 如果形状的某维度为零，则返回 False
        if shape[d] == 0:
            return False
        # 如果步幅小于最小值，则返回 False
        if strides[d] < min:
            return False
        # 如果当前维度是第一个维度并且最小值等于第二个维度的步幅，则返回 False
        if d == 0 and min == strides[1]:
            return False
        min = strides[d]
        # 如果步幅大于1，则更新最小值乘以当前维度的形状
        if strides[d] > 1:
            min *= shape[d]
    return True


# 返回建议的内存格式
def suggest_memory_format(x: TensorLikeType) -> torch.memory_format:
    # 如果张量的布局不是 strided，则返回连续内存格式
    if x.layout != torch.strided:
        return torch.contiguous_format

    # 如果张量的步幅类似于 channels_last，则返回对应的内存格式
    if are_strides_like_channels_last(x.shape, x.stride()):
        return torch.channels_last if x.ndim == 4 else torch.channels_last_3d

    # 否则返回连续内存格式
    return torch.contiguous_format


# 计算序列中元素的乘积，空序列返回1
def prod(xs: Sequence[NumberType]) -> NumberType:
    """Product of elements in input sequence. Returns 1 for empty sequence"""
    return reduce(operator.mul, xs, 1)


# 检查形状是否可以扩展到目标形状
def is_expandable_to(shape: ShapeType, desired: ShapeType) -> bool:
    """检查一个形状是否可以扩展到另一个形状。
    这相当于检查这两个形状是否可广播。

    参数：
    shape - 当前形状的列表或元组
    desired - 目标形状的列表或元组

    返回：
    如果当前形状可以扩展到目标形状，返回True；否则返回False。
    """
    # 这是对于 aten/src/ATen/ExpandUtils.h:is_expandable 的 Python 实现

    # 如果当前形状的维度大于目标形状的维度，则无法扩展
    if len(shape) > len(desired):
        return False

    # 逐个比较当前形状和目标形状的每一个维度
    for i in range(len(shape)):
        # 检查对应维度上的大小是否相等或者当前形状是否为1（可以广播）
        if shape[-i - 1] != desired[-i - 1] and shape[-i - 1] != 1:
            return False

    # 如果所有维度都满足条件，则可以扩展
    return True
# 定义一个函数，根据给定的 mask 和 tensor t，返回一个新的 tensor，
# 类似于 torch.where(mask, t, 0)，但如果 t 是布尔类型，则结果也是布尔类型而不是提升为整数类型。
def mask_tensor(mask: TensorLikeType, t: TensorLikeType):
    # 如果 t 的数据类型是布尔类型
    if t.dtype is torch.bool:
        # 对 mask 和 t 进行逻辑与操作
        return mask.logical_and(t)
    else:
        # 否则，使用 torch.where 进行操作，用 t 替换 mask 为 False 的部分，其余为 0
        return torch.where(mask, t, 0)


# 定义一个函数，根据给定的函数和名称，返回对应的 ATen 操作名称
def get_aten_op(fn: Callable, name: str):
    # 获取函数所在的模块名称
    module = fn.__module__
    prefix = "torch._refs"
    # 断言模块名称以 "torch._refs" 开头
    assert module.startswith(prefix)
    module = module[len(prefix):]
    # 去掉开头的点号，将剩余的点号替换为下划线，并在末尾添加下划线
    if module:
        module = module[1:]
        module = module.replace(".", "_")
        module = module + "_"
    # 返回 torch._ops.ops.aten 中对应模块和名称的属性
    return getattr(torch._ops.ops.aten, f"{module}{name}")


# 定义一个函数，根据输入的 dtype 返回其值或默认的 torch 默认数据类型
def dtype_or_default(dtype: Optional[torch.dtype]) -> torch.dtype:
    return dtype if dtype is not None else torch.get_default_dtype()


# 定义一个函数，根据输入的设备类型返回其值或默认的 CPU 设备
def device_or_default(device: Optional[DeviceLikeType]) -> DeviceLikeType:
    return device if device is not None else torch.device("cpu")


# 定义一个函数，根据输入的布局类型返回其值或默认的 torch.strided 布局
def layout_or_default(layout: Optional[torch.layout]) -> torch.layout:
    return layout if layout is not None else torch.strided


# 定义一个函数，克隆输入 tensor x 并保留其步长信息
def clone_preserve_strides(x):
    # 计算所需的存储长度
    needed_size = compute_required_storage_length(
        x.size(), x.stride(), x.storage_offset()
    )
    # 在 Autograd 中，我们的 *_scatter 操作的 eager 实现都是原语，
    # 所以这些 as_strided() 调用对 Autograd 不可见。
    # 我们需要在我们的 ref/prim 实现中模拟这种行为。
    # TODO: 更好的处理方式可能是使用一个新的操作 "_unsafe_as_strided"
    # 当我们添加一个组合的 as_strided 操作时，我们应该重新审视这个问题，
    # 同时也作为 https://github.com/pytorch/pytorch/issues/90507 的一部分。
    try:
        # 暂时排除 torch._C.DispatchKey.ADInplaceOrView 的调度键
        old = torch._C._dispatch_tls_is_dispatch_key_excluded(
            torch._C.DispatchKey.ADInplaceOrView
        )
        torch._C._dispatch_tls_set_dispatch_key_excluded(
            torch._C.DispatchKey.ADInplaceOrView, True
        )
        # 使用 as_strided() 创建一个新的 buffer，并克隆其内容
        buffer = torch.as_strided(x, (needed_size,), (1,), 0).clone()
        # 返回一个按照输入 tensor x 的大小、步长和存储偏移创建的 as_strided tensor
        return torch.as_strided(buffer, x.size(), x.stride(), x.storage_offset())
    finally:
        # 恢复 torch._C.DispatchKey.ADInplaceOrView 的调度键状态
        torch._C._dispatch_tls_set_dispatch_key_excluded(
            torch._C.DispatchKey.ADInplaceOrView, old
        )


# 定义一个函数，用于通知非确定性行为
def alert_not_deterministic(caller: str):
    # 检查是否启用了确定性算法
    if torch.are_deterministic_algorithms_enabled():
        # 如果仅仅是警告模式下启用了确定性算法，则发出警告
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(
                # 发出警告，说明调用者没有确定性实现，但设置了 'torch.use_deterministic_algorithms(True, warn_only=True)'
                f"{caller} does not have a deterministic implementation, but you set "
                f"'torch.use_deterministic_algorithms(True, warn_only=True)'. "
                f"You can file an issue at https://github.com/pytorch/pytorch/issues "
                f"to help us prioritize adding deterministic support for this operation."
            )
        else:
            # 否则，如果不是警告模式，则抛出异常
            torch._check(
                False,
                lambda: (
                    # 抛出异常，说明调用者没有确定性实现，但设置了 'torch.use_deterministic_algorithms(True)'
                    f"{caller} does not have a deterministic implementation, but you set "
                    f"'torch.use_deterministic_algorithms(True)'. You can turn off "
                    f"determinism just for this operation, or you can use the "
                    f"'warn_only=True' option, if that's acceptable for your application. "
                    f"You can also file an issue at https://github.com/pytorch/pytorch/issues "
                    f"to help us prioritize adding deterministic support for this operation."
                ),
            )
class CUDARngStateHelper:
    @staticmethod
    # 返回当前的 Torch 随机状态作为元组，支持假模式
    def get_torch_state_as_tuple(fake_mode=nullcontext()):
        # 检查CUDA是否可用，否则引发运行时错误
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        # 使用假模式上下文管理器
        with fake_mode:
            # 获取当前CUDA随机种子，并封装为张量
            seed = torch.tensor(torch.cuda.initial_seed())
            # 获取CUDA随机状态偏移，并封装为张量
            offset = torch.tensor(torch.cuda._get_rng_state_offset())
            return seed, offset

    @staticmethod
    # 将给定的种子和偏移设置为 Torch 状态张量
    def set_torch_state_tensor(seed, offset):
        # 随机状态由 [64位种子, 64位偏移] 组成
        seed_portion = seed.reshape([1]).view(torch.uint8)
        offset_portion = offset.reshape([1]).view(torch.uint8)
        new_state = torch.cat([seed_portion, offset_portion])
        # 设置新的 CUDA 随机状态
        torch.cuda.set_rng_state(new_state)

    @staticmethod
    # 设置新的随机状态偏移量
    def set_new_offset(relative_offset):
        # 调用 Torch 函数设置新的 CUDA 随机状态偏移量
        torch.cuda._set_rng_state_offset(relative_offset.item())
```