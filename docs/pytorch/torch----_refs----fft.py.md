# `.\pytorch\torch\_refs\fft.py`

```py
# 导入 math 模块，用于数学计算
import math

# 从 typing 模块导入各种类型和类型注解
from typing import Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple, Union

# 导入 PyTorch 库
import torch
import torch._prims as prims  # 导入 torch._prims 模块
import torch._prims_common as utils  # 导入 torch._prims_common 模块
from torch._decomp import register_decomposition  # 导入 register_decomposition 函数
from torch._prims_common import DimsType, ShapeType, TensorLikeType  # 导入类型别名
from torch._prims_common.wrappers import _maybe_convert_to_dtype, out_wrapper  # 导入函数和装饰器

# 定义 __all__ 列表，包含需要公开的模块接口
__all__ = [
    # Transforms
    "fft",
    "fft2",
    "fftn",
    "hfft",
    "hfft2",
    "hfftn",
    "rfft",
    "rfft2",
    "rfftn",
    "ifft",
    "ifft2",
    "ifftn",
    "ihfft",
    "ihfft2",
    "ihfftn",
    "irfft",
    "irfft2",
    "irfftn",
    # Helpers
    "fftshift",
    "ifftshift",
]

# 定义 NormType 类型别名，表示规范化模式的可能取值
NormType = Union[None, Literal["forward", "backward", "ortho"]]
_NORM_VALUES = {None, "forward", "backward", "ortho"}

# 获取 torch._ops.ops.aten 的别名 aten
aten = torch._ops.ops.aten


def _apply_norm(
    x: TensorLikeType, norm: NormType, signal_numel: int, forward: bool
) -> TensorLikeType:
    """对未经规范化的 FFT 结果应用规范化"""
    # 检查 norm 是否在合法的规范化模式集合中
    torch._check(norm in _NORM_VALUES, lambda: f"Invalid normalization mode: {norm}")

    # 如果规范化模式是 "ortho"，应用正交规范化
    if norm == "ortho":
        return x * (1 / math.sqrt(signal_numel))

    # 根据 forward 参数决定是否应用规范化
    normalize = (not forward and (norm is None or norm == "backward")) or (
        forward and norm == "forward"
    )
    return x * (1 / signal_numel) if normalize else x


def _promote_type_fft(
    dtype: torch.dtype, require_complex: bool, device: torch.device
) -> torch.dtype:
    """辅助函数，将 dtype 提升为 FFT 原语支持的数据类型"""
    if dtype.is_complex:
        return dtype

    # 如果 dtype 不是浮点类型，将其提升为默认的浮点类型
    if not dtype.is_floating_point:
        dtype = torch.get_default_dtype()

    # 支持的数据类型包括 float32 和 float64
    allowed_types = [torch.float32, torch.float64]
    maybe_support_half = device.type in ["cuda", "meta"]

    # 在某些设备上可能还支持 float16
    if maybe_support_half:
        allowed_types.append(torch.float16)
    torch._check(dtype in allowed_types, lambda: f"Unsupported dtype {dtype}")

    # 如果需要复数类型，将 dtype 转换为对应的复数类型
    if require_complex:
        dtype = utils.corresponding_complex_dtype(dtype)

    return dtype


def _maybe_promote_tensor_fft(
    t: TensorLikeType, require_complex: bool = False
) -> TensorLikeType:
    """辅助函数，将张量提升为 FFT 原语支持的数据类型"""
    cur_type = t.dtype
    new_type = _promote_type_fft(cur_type, require_complex, t.device)
    return _maybe_convert_to_dtype(t, new_type)  # type: ignore[return-value]


def _resize_fft_input(
    x: TensorLikeType, dims: Tuple[int, ...], sizes: Tuple[int, ...]
) -> TensorLikeType:
    """
    修正 x 的形状，使得 x.size(dims[i]) == sizes[i]，
    可以通过填充零或者从 0 开始切片 x 来实现。
    """
    assert len(dims) == len(sizes)
    must_copy = False
    x_sizes = x.shape
    pad_amount = [0] * len(x_sizes) * 2
    # 遍历维度列表 dims 中的索引 i，执行以下操作
    for i in range(len(dims)):
        # 如果 sizes[i] 的值为 -1，则跳过当前迭代，继续下一个维度
        if sizes[i] == -1:
            continue
    
        # 如果 x_sizes[dims[i]] 小于 sizes[i]，则需要进行复制操作
        if x_sizes[dims[i]] < sizes[i]:
            # 设置标志 must_copy 为 True，表示需要复制
            must_copy = True
            # 计算填充索引 pad_idx，用于更新 pad_amount 中的填充值
            pad_idx = len(pad_amount) - 2 * dims[i] - 1
            # 计算需要填充的数量，并更新到 pad_amount 中
            pad_amount[pad_idx] = sizes[i] - x_sizes[dims[i]]
    
        # 如果 x_sizes[dims[i]] 大于 sizes[i]，则需要裁剪 x 的数据
        if x_sizes[dims[i]] > sizes[i]:
            # 使用 narrow 函数裁剪 x 在 dims[i] 维度上的数据，保留前 sizes[i] 个元素
            x = x.narrow(dims[i], 0, sizes[i])
    
    # 根据标志 must_copy 决定是否进行常量填充操作，如果不需要复制，则直接返回 x
    return torch.constant_pad_nd(x, pad_amount) if must_copy else x
# 定义函数 _fft_c2r，用于执行复数到实数的FFT（例如irfft或hfft）
def _fft_c2r(
    func_name: str,           # 函数名，用于错误消息
    input: TensorLikeType,    # 输入张量，可以是各种张量类型
    n: Optional[int],         # 数据点数，可选参数
    dim: int,                 # 指定FFT操作的维度
    norm: NormType,           # 规范化类型，用于调整FFT的输出
    forward: bool,            # 布尔值，指示是否是正向FFT
) -> TensorLikeType:
    """Common code for performing any complex to real FFT (irfft or hfft)"""
    input = _maybe_promote_tensor_fft(input, require_complex=True)  # 如果需要，将输入张量提升为复数类型
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)  # 规范化FFT操作的维度
    last_dim_size = n if n is not None else 2 * (input.shape[dim] - 1)  # 计算最后一个维度的大小

    # 检查数据点数是否有效
    torch._check(
        last_dim_size >= 1,
        lambda: f"Invalid number of data points ({last_dim_size}) specified",
    )

    # 如果指定了数据点数n，则调整输入张量的大小
    if n is not None:
        input = _resize_fft_input(input, dims=dims, sizes=(last_dim_size // 2 + 1,))

    # 如果是正向FFT，则对输入取共轭
    if forward:
        input = torch.conj(input)

    # 调用底层函数执行复数到实数的FFT，并返回结果
    output = prims.fft_c2r(input, dim=dims, last_dim_size=last_dim_size)
    return _apply_norm(output, norm=norm, signal_numel=last_dim_size, forward=forward)


# 定义函数 _fft_r2c，用于执行实数到复数的FFT（例如rfft或ihfft）
def _fft_r2c(
    func_name: str,           # 函数名，用于错误消息
    input: TensorLikeType,    # 输入张量，可以是各种张量类型
    n: Optional[int],         # 数据点数，可选参数
    dim: int,                 # 指定FFT操作的维度
    norm: NormType,           # 规范化类型，用于调整FFT的输出
    forward: bool,            # 布尔值，指示是否是正向FFT
    onesided: bool,           # 布尔值，指示是否是单边FFT
) -> TensorLikeType:
    """Common code for performing any real to complex FFT (rfft or ihfft)"""
    # 检查输入张量是否为浮点数类型
    torch._check(
        not input.dtype.is_complex,
        lambda: f"{func_name} expects a floating point input tensor, but got {input.dtype}",
    )
    # 如果需要，将输入张量提升为适合FFT的类型
    input = _maybe_promote_tensor_fft(input)

    # 规范化FFT操作的维度
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)

    # 计算指定维度的大小
    dim_size = n if n is not None else input.shape[dim]

    # 检查数据点数是否有效
    torch._check(
        dim_size >= 1, lambda: f"Invalid number of data points ({dim_size}) specified"
    )

    # 如果指定了数据点数n，则调整输入张量的大小
    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    # 调用底层函数执行实数到复数的FFT
    ret = prims.fft_r2c(input, dim=dims, onesided=onesided)

    # 应用规范化处理，调整FFT的输出
    ret = _apply_norm(ret, norm, dim_size, forward)

    # 如果是反向FFT，则对结果取共轭
    return ret if forward else torch.conj(ret)


# 定义函数 _fft_c2c，用于执行复数到复数的FFT（例如fft或ifft）
def _fft_c2c(
    func_name: str,           # 函数名，用于错误消息
    input: TensorLikeType,    # 输入张量，可以是各种张量类型
    n: Optional[int],         # 数据点数，可选参数
    dim: int,                 # 指定FFT操作的维度
    norm: NormType,           # 规范化类型，用于调整FFT的输出
    forward: bool,            # 布尔值，指示是否是正向FFT
) -> TensorLikeType:
    """Common code for performing any complex to complex FFT (fft or ifft)"""
    # 检查输入张量是否为复数类型
    torch._check(
        input.dtype.is_complex,
        lambda: f"{func_name} expects a complex input tensor, but got {input.dtype}",
    )

    # 规范化FFT操作的维度
    dims = (utils.canonicalize_dim(input.ndim, dim, wrap_scalar=False),)

    # 计算指定维度的大小
    dim_size = n if n is not None else input.shape[dim]

    # 检查数据点数是否有效
    torch._check(
        dim_size >= 1, lambda: f"Invalid number of data points ({dim_size}) specified"
    )

    # 如果指定了数据点数n，则调整输入张量的大小
    if n is not None:
        input = _resize_fft_input(input, dims, (n,))

    # 调用底层函数执行复数到复数的FFT
    ret = prims.fft_c2c(input, dim=dims, forward=forward)

    # 应用规范化处理，调整FFT的输出
    return _apply_norm(ret, norm, dim_size, forward)


# 将函数fft注册为aten.fft_fft的分解函数，并用于输出包装器
@register_decomposition(aten.fft_fft)
@out_wrapper()
def fft(
    input: TensorLikeType,     # 输入张量，可以是各种张量类型
    n: Optional[int] = None,   # 数据点数，可选参数，默认为None
    dim: int = -1,             # 指定FFT操作的维度，默认为-1
    norm: NormType = None,     # 规范化类型，用于调整FFT的输出，默认为None
) -> TensorLikeType:
    # 如果输入张量是复数类型，则调用复数到复数的FFT函数_fft_c2c
    if input.dtype.is_complex:
        return _fft_c2c("fft", input, n, dim, norm, forward=True)
    else:
        # 如果条件不满足，则执行以下代码块
        # 调用 _fft_r2c 函数，使用参数 "fft", input, n, dim, norm
        # forward 参数设置为 True，onesided 参数设置为 False
        return _fft_r2c("fft", input, n, dim, norm, forward=True, onesided=False)
# 注册对应的函数为 fft_ifft 的分解函数
# 并对输出进行装饰处理
@register_decomposition(aten.fft_ifft)
@out_wrapper()
def ifft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # 如果输入张量的数据类型是复数类型，则调用复数到复数的 FFT 反变换函数
    if input.dtype.is_complex:
        return _fft_c2c("ifft", input, n, dim, norm, forward=False)
    else:
        # 否则调用实数到复数的 FFT 反变换函数
        return _fft_r2c("ifft", input, n, dim, norm, forward=False, onesided=False)


# 注册对应的函数为 fft_rfft 的分解函数
# 并对输出进行装饰处理
@register_decomposition(aten.fft_rfft)
@out_wrapper()
def rfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # 调用实数到复数的 FFT 变换函数
    return _fft_r2c("rfft", input, n, dim, norm, forward=True, onesided=True)


# 注册对应的函数为 fft_irfft 的分解函数
# 并对输出进行装饰处理
@register_decomposition(aten.fft_irfft)
@out_wrapper()
def irfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # 调用复数到实数的 FFT 反变换函数
    return _fft_c2r("irfft", input, n, dim, norm, forward=False)


# 注册对应的函数为 fft_hfft 的分解函数
# 并对输出进行装饰处理
@register_decomposition(aten.fft_hfft)
@out_wrapper()
def hfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # 调用复数到实数的 FFT 变换函数
    return _fft_c2r("hfft", input, n, dim, norm, forward=True)


# 注册对应的函数为 fft_ihfft 的分解函数
# 并对输出进行装饰处理
@register_decomposition(aten.fft_ihfft)
@out_wrapper()
def ihfft(
    input: TensorLikeType,
    n: Optional[int] = None,
    dim: int = -1,
    norm: NormType = None,
) -> TensorLikeType:
    # 调用实数到复数的 FFT 反变换函数
    return _fft_r2c("ihfft", input, n, dim, norm, forward=False, onesided=True)


# 定义一个命名元组 _ShapeAndDims，包含形状和维度信息
class _ShapeAndDims(NamedTuple):
    shape: Tuple[int, ...]
    dims: Tuple[int, ...]


# 将输入的形状和维度参数转换为规范形式，确保它们都不是可选的
def _canonicalize_fft_shape_and_dim_args(
    input: TensorLikeType, shape: Optional[ShapeType], dim: Optional[DimsType]
) -> _ShapeAndDims:
    """Convert the shape and dim arguments into a canonical form where neither are optional"""
    # 获取输入张量的维度和大小
    input_dim = input.ndim
    input_sizes = input.shape

    if dim is not None:
        # 如果维度参数不是序列类型，则转换为元组
        if not isinstance(dim, Sequence):
            dim = (dim,)
        # 规范化维度，确保它们是唯一的
        ret_dims = utils.canonicalize_dims(input_dim, dim, wrap_scalar=False)

        # 检查维度是否唯一，若不是则抛出异常
        torch._check(
            len(set(ret_dims)) == len(ret_dims), lambda: "FFT dims must be unique"
        )
    # 如果给定了 shape 参数
    if shape is not None:
        # 如果 shape 不是序列，将其转换为只含一个元素的元组
        if not isinstance(shape, Sequence):
            shape = (shape,)

        # 检查 dim 参数是否为 None 或者与 shape 参数长度相同
        torch._check(
            dim is None or len(dim) == len(shape),
            lambda: "When given, dim and shape arguments must have the same length",
        )
        # 计算输入 tensor 的维度
        transform_ndim = len(shape)

        # 检查给定的 shape 是否不超过输入 tensor 的维度
        torch._check(
            transform_ndim <= input_dim,
            lambda: f"Got shape with {transform_ndim} values but input tensor "
            f"only has {input_dim} dimensions.",
        )

        # 如果未指定 dim 参数，将其设置为最后 len(shape) 个维度的默认值
        if dim is None:
            ret_dims = tuple(range(input_dim - transform_ndim, input_dim))

        # 将 shape 中的任何 -1 值转换为默认长度
        ret_shape = tuple(
            s if s != -1 else input_sizes[d] for (s, d) in zip(shape, ret_dims)  # type: ignore[possibly-undefined]
        )

    # 如果未指定 shape 参数
    elif dim is None:
        # 没有 shape 参数也没有 dim 参数时，返回所有维度
        ret_dims = tuple(range(input_dim))
        ret_shape = tuple(input_sizes)
    else:
        # 没有 shape 参数但有 dim 参数时，根据 dim 参数返回相应的维度
        ret_shape = tuple(input_sizes[d] for d in ret_dims)  # type: ignore[possibly-undefined]

    # 检查 ret_shape 中每个维度是否大于 0
    for n in ret_shape:
        torch._check(n > 0, lambda: f"Invalid number of data points ({n}) specified")

    # 返回处理后的 shape 和 dims
    return _ShapeAndDims(shape=ret_shape, dims=ret_dims)  # type: ignore[possibly-undefined]
# 计算整数迭代器中元素的乘积
def _prod(xs: Iterable[int]) -> int:
    prod = 1  # 初始化乘积为1
    for x in xs:  # 遍历整数迭代器
        prod *= x  # 逐步计算乘积
    return prod  # 返回最终乘积结果


# 用于实现n维复数到复数FFT（fftn或ifftn）的通用代码
def _fftn_c2c(
    function_name: str,
    input: TensorLikeType,
    shape: Tuple[int, ...],
    dim: Tuple[int, ...],
    norm: NormType,
    forward: bool,
) -> TensorLikeType:
    torch._check(
        input.dtype.is_complex,
        lambda: f"{function_name} expects a complex input tensor, "
        f"but got {input.dtype}",
    )
    x = _resize_fft_input(input, dim, shape)  # 调整FFT输入的大小
    output = prims.fft_c2c(x, dim=dim, forward=forward)  # 执行复数到复数FFT
    return _apply_norm(output, norm=norm, signal_numel=_prod(shape), forward=forward)


# 注册aten.fft_fftn的分解，并包装输出
@register_decomposition(aten.fft_fftn)
@out_wrapper()
def fftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)  # 规范化FFT的形状和维度参数
    x = _maybe_promote_tensor_fft(input, require_complex=True)  # 可能提升FFT输入的复杂性要求
    return _fftn_c2c("fftn", x, shape, dim, norm, forward=True)  # 执行复数到复数FFT


# 注册aten.fft_ifftn的分解，并包装输出
@register_decomposition(aten.fft_ifftn)
@out_wrapper()
def ifftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)  # 规范化FFT的形状和维度参数
    x = _maybe_promote_tensor_fft(input, require_complex=True)  # 可能提升FFT输入的复杂性要求
    return _fftn_c2c("ifftn", x, shape, dim, norm, forward=False)  # 执行复数到复数FFT


# 注册aten.fft_rfftn的分解，并包装输出
@register_decomposition(aten.fft_rfftn)
@out_wrapper()
def rfftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    torch._check(
        not input.dtype.is_complex,
        lambda: f"rfftn expects a real-valued input tensor, but got {input.dtype}",
    )
    shape, dim = _canonicalize_fft_shape_and_dim_args(input, s, dim)  # 规范化FFT的形状和维度参数
    input = _maybe_promote_tensor_fft(input, require_complex=False)  # 可能提升FFT输入的实数性要求
    input = _resize_fft_input(input, dim, shape)  # 调整FFT输入的大小
    out = prims.fft_r2c(input, dim=dim, onesided=True)  # 执行实数到复数FFT
    return _apply_norm(out, norm=norm, signal_numel=_prod(shape), forward=True)  # 应用规范化并返回结果


# 注册aten.fft_ihfftn的分解，并包装输出
@register_decomposition(aten.fft_ihfftn)
@out_wrapper()
def ihfftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    torch._check(
        not input.dtype.is_complex,
        lambda: f"ihfftn expects a real-valued input tensor, but got {input.dtype}",
    )
    shape, dim = _canonicalize_fft_shape_and_dim_args(input, s, dim)  # 规范化FFT的形状和维度参数
    torch._check(len(shape) > 0, lambda: "ihfftn must transform at least one axis")  # 检查转换的轴数
    input = _maybe_promote_tensor_fft(input, require_complex=False)  # 可能提升FFT输入的实数性要求
    input = _resize_fft_input(input, dim, shape)  # 调整FFT输入的大小

    tmp = prims.fft_r2c(input, dim=dim[-1:], onesided=True)  # 执行实数到复数FFT
    # 如果维度 dim 的长度为 1，则执行以下操作
    if len(dim) == 1:
        # 对 tmp 应用规范化操作，使用 _apply_norm 函数，指定 norm 参数和信号长度为 shape 的第一个元素，执行逆向操作
        tmp = _apply_norm(tmp, norm=norm, signal_numel=shape[0], forward=False)
        # 返回 tmp 的共轭
        return prims.conj(tmp)
    
    # 对 tmp 执行物理空间共轭操作
    tmp = prims.conj_physical(tmp)
    # 对 tmp 执行 C2C FFT（复数到复数的快速傅里叶变换），指定维度为 dim 的除最后一个元素外的所有元素，执行逆向操作
    tmp = prims.fft_c2c(tmp, dim=dim[:-1], forward=False)
    # 返回对 tmp 应用规范化操作的结果，使用 _apply_norm 函数，指定 norm 参数和信号长度为 shape 所有元素的乘积，执行逆向操作
    return _apply_norm(tmp, norm=norm, signal_numel=_prod(shape), forward=False)
# 定义了一个命名元组 `_CanonicalizeC2rReturn`，用于封装规范化的 c2r 转换结果
class _CanonicalizeC2rReturn(NamedTuple):
    shape: Tuple[int, ...]       # 元组成员：表示转换后的张量形状
    dim: Tuple[int, ...]         # 元组成员：表示转换应用的维度
    last_dim_size: int           # 整数成员：表示输出张量最后一个维度的大小


def _canonicalize_fft_c2r_shape_and_dim_args(
    fname: str,
    input: TensorLikeType,
    s: Optional[ShapeType],
    dim: Optional[DimsType],
) -> _CanonicalizeC2rReturn:
    """Canonicalize shape and dim arguments for n-dimensional c2r transforms,
    as well as calculating the last_dim_size which is shape[dim[-1]] for the output"""
    # 调用内部函数 `_canonicalize_fft_shape_and_dim_args` 规范化输入形状和维度参数
    (shape, dim) = _canonicalize_fft_shape_and_dim_args(input, s, dim)
    # 使用 torch._check 检查至少有一个轴需要进行变换
    torch._check(len(shape) > 0, lambda: f"{fname} must transform at least one axis")

    # 计算最后一个维度的大小，根据输入的形状和指定的维度
    if s is None or s[-1] == -1:
        last_dim_size = 2 * (input.shape[dim[-1]] - 1)
    else:
        last_dim_size = shape[-1]

    # 使用 torch._check 检查数据点数是否有效
    torch._check(
        last_dim_size >= 1,
        lambda: f"Invalid number of data points ({last_dim_size}) specified",
    )

    # 修改形状列表，将最后一个维度的大小设置为 (last_dim_size // 2 + 1)
    shape_list = list(shape)
    shape_list[-1] = last_dim_size // 2 + 1

    # 返回规范化的 c2r 转换结果
    return _CanonicalizeC2rReturn(
        shape=tuple(shape_list), dim=dim, last_dim_size=last_dim_size
    )


@register_decomposition(aten.fft_irfftn)
@out_wrapper()
def irfftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    # 调用 `_canonicalize_fft_c2r_shape_and_dim_args` 规范化参数，获取形状、维度和最后一个维度的大小
    shape, dim, last_dim_size = _canonicalize_fft_c2r_shape_and_dim_args(
        "irfftn", input, s, dim
    )
    # 对输入张量进行复杂数提升（推广）
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    # 调整输入张量的大小，以匹配规范化后的形状和维度
    input = _resize_fft_input(input, dim, shape)
    # 调用 prims.fft_c2r 进行实部到复部的傅里叶逆变换
    out = prims.fft_c2r(input, dim=dim, last_dim_size=last_dim_size)
    # 应用规范化到输出结果，包括规范化、数据点数乘积和逆向操作标志
    return _apply_norm(out, norm, _prod(out.shape[d] for d in dim), forward=False)


@register_decomposition(aten.fft_hfftn)
@out_wrapper()
def hfftn(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = None,
    norm: NormType = None,
) -> TensorLikeType:
    # 调用 `_canonicalize_fft_c2r_shape_and_dim_args` 规范化参数，获取形状、维度和最后一个维度的大小
    shape, dim, last_dim_size = _canonicalize_fft_c2r_shape_and_dim_args(
        "hfftn", input, s, dim
    )
    # 对输入张量进行复杂数提升（推广）
    input = _maybe_promote_tensor_fft(input, require_complex=True)
    # 调整输入张量的大小，以匹配规范化后的形状和维度

    tmp = prims.fft_c2c(input, dim=dim[:-1], forward=True) if len(dim) > 1 else input
    # 应用规范化到复部的傅里叶变换，包括规范化、形状去除最后一个维度和正向操作标志
    tmp = _apply_norm(tmp, norm, _prod(shape[:-1]), forward=True)
    # 对物理共轭进行转置操作
    tmp = prims.conj_physical(tmp)
    # 调用 prims.fft_c2r 进行复部到实部的傅里叶逆变换
    out = prims.fft_c2r(tmp, dim=dim[-1:], last_dim_size=last_dim_size)
    # 应用规范化到输出结果，包括规范化、最后一个维度大小和正向操作标志
    return _apply_norm(out, norm, last_dim_size, forward=True)


@register_decomposition(aten.fft_fft2)
@out_wrapper()
def fft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # 使用 torch.fft.fftn 执行二维傅里叶变换
    return torch.fft.fftn(input, s=s, dim=dim, norm=norm)


@register_decomposition(aten.fft_ifft2)
@out_wrapper()
def ifft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # 使用 torch.fft.ifftn 执行二维傅里叶逆变换
    return torch.fft.ifftn(input, s=s, dim=dim, norm=norm)
# 使用 @register_decomposition 装饰器注册 aten.fft_rfft2 函数的分解版本
# 使用 @out_wrapper() 装饰器处理输出结果
def rfft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # 调用 torch.fft.rfftn 函数执行高效的多维实数FFT变换
    return torch.fft.rfftn(input, s=s, dim=dim, norm=norm)


# 使用 @register_decomposition 装饰器注册 aten.fft_irfft2 函数的分解版本
# 使用 @out_wrapper() 装饰器处理输出结果
def irfft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # 调用 torch.fft.irfftn 函数执行高效的多维实数FFT逆变换
    return torch.fft.irfftn(input, s=s, dim=dim, norm=norm)


# 使用 @register_decomposition 装饰器注册 aten.fft_hfft2 函数的分解版本
# 使用 @out_wrapper() 装饰器处理输出结果
def hfft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # 调用 torch.fft.hfftn 函数执行高效的多维纯虚数FFT变换
    return torch.fft.hfftn(input, s=s, dim=dim, norm=norm)


# 使用 @register_decomposition 装饰器注册 aten.fft_ihfft2 函数的分解版本
# 使用 @out_wrapper() 装饰器处理输出结果
def ihfft2(
    input: TensorLikeType,
    s: Optional[ShapeType] = None,
    dim: Optional[DimsType] = (-2, -1),
    norm: NormType = None,
) -> TensorLikeType:
    # 调用 torch.fft.ihfftn 函数执行高效的多维纯虚数FFT逆变换
    return torch.fft.ihfftn(input, s=s, dim=dim, norm=norm)


# 定义函数 _default_alldims，将 Optional[DimsType] 转换为简单的整数列表，默认为所有维度
def _default_alldims(dim: Optional[DimsType], x: TensorLikeType) -> List[int]:
    if dim is None:
        # 如果维度为 None，则返回所有维度的列表
        return list(range(x.ndim))
    elif not isinstance(dim, Sequence):
        # 如果维度不是序列，则返回单个维度的列表
        return [dim]
    else:
        # 否则，返回维度列表
        return list(dim)


# 使用 @register_decomposition 装饰器注册 aten.fft_fftshift 函数的分解版本
def fftshift(input: TensorLikeType, dim: Optional[DimsType] = None) -> TensorLikeType:
    # 获取默认维度列表
    dims = _default_alldims(dim, input)
    # 计算移动量，将输入张量的轴进行移动
    shift = [input.shape[d] // 2 for d in dims]
    # 调用 torch.roll 函数实现 FFT 移位操作
    return torch.roll(input, shift, dims)


# 使用 @register_decomposition 装饰器注册 aten.fft_ifftshift 函数的分解版本
def ifftshift(input: TensorLikeType, dim: Optional[DimsType] = None) -> TensorLikeType:
    # 获取默认维度列表
    dims = _default_alldims(dim, input)
    # 计算移动量，将输入张量的轴进行逆移动
    shift = [(input.shape[d] + 1) // 2 for d in dims]
    # 调用 torch.roll 函数实现逆 FFT 移位操作
    return torch.roll(input, shift, dims)
```