# `.\pytorch\torch\ao\quantization\fx\_decomposed.py`

```py
# mypy: allow-untyped-defs
# 导入 math 库
import math
# 从 typing 库中导入 Optional 和 Tuple 类型
from typing import Optional, Tuple

# 导入 torch 库
import torch
# 从 torch._refs 模块中导入 _unsqueeze_multiple 函数
from torch._refs import _unsqueeze_multiple
# 从 torch.ao.quantization.utils 模块中导入 determine_qparams 和 validate_qmin_qmax 函数
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
# 从 torch.library 模块中导入 impl 和 Library 类
from torch.library import impl, Library

# 定义一个名为 quantized_decomposed_lib 的 Library 实例，用于存储 quantized_decomposed 相关的定义
quantized_decomposed_lib = Library("quantized_decomposed", "DEF")

# 定义整数类型的列表
_INTEGER_DTYPES = [torch.uint8, torch.int8, torch.int16, torch.int32]
# 定义浮点数类型的列表
_FLOAT_DTYPES = [torch.float8_e5m2, torch.float8_e4m3fn]

# 创建一个字典，将每种数据类型与其量化值范围的上下界关联起来
_DTYPE_TO_QVALUE_BOUNDS = {k : (torch.iinfo(k).min, torch.iinfo(k).max) for k in _INTEGER_DTYPES}
_DTYPE_TO_QVALUE_BOUNDS.update({k : (int(torch.finfo(k).min), int(torch.finfo(k).max)) for k in _FLOAT_DTYPES})

# Helper 函数用于检查传入的量化最小值和最大值是否在指定数据类型的合法范围内
def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    # 如果数据类型不在支持的数据类型列表中，则抛出 ValueError 异常
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    # 获取量化最小值和最大值的合法范围
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]

    # 使用断言确保量化最小值在合法范围内
    assert quant_min >= quant_min_lower_bound, \
        "quant_min out of bound for dtype, " \
        f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"

    # 使用断言确保量化最大值在合法范围内
    assert quant_max <= quant_max_upper_bound, \
        "quant_max out of bound for dtype, " \
        f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"

# 在 quantized_decomposed_lib 中定义一个名为 "quantize_per_tensor" 的函数签名
quantized_decomposed_lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

# 使用 @impl 装饰器，将 quantize_per_tensor 函数注册到 quantized_decomposed_lib 的 "quantize_per_tensor" 定义中
@impl(quantized_decomposed_lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor(
        input: torch.Tensor,
        scale: float,
        zero_point: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ 对 Tensor 进行仿射量化，使用相同的量化参数从浮点数映射到量化值

    Args:
       input (torch.Tensor): 原始的 float32 或 bfloat16 Tensor
       scale (float): 仿射量化的量化参数
       zero_point (int): 仿射量化的量化参数
       quant_min (int): 输出 Tensor 的最小量化值
       quant_max (int): 输出 Tensor 的最大量化值
       dtype (torch.dtype): 请求的输出数据类型 (例如 torch.uint8)

    Returns:
       返回请求的数据类型的 Tensor (例如 torch.uint8)，注意量化参数不会存储在 Tensor 中，
       而是存储在函数参数中
    """
    # 如果输入 Tensor 的数据类型是 torch.float16 或 torch.bfloat16，则将其转换为 torch.float32
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    # 使用断言确保输入 Tensor 的数据类型为 torch.float32
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    # 检查量化最小值和最大值是否在合法范围内
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)

    # 计算 scale 的倒数
    inv_scale = 1.0 / scale
    # 对输入张量进行量化操作：先乘以逆尺度（inverse scale），然后四舍五入并加上零点（zero point）
    # 使用 clamp 方法确保量化后的值在指定的最小值（quant_min）和最大值（quant_max）之间
    # 最后将结果转换为指定的数据类型（dtype）并返回
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)
# 定义函数，实现对张量进行基于元信息的量化
@impl(quantized_decomposed_lib, "quantize_per_tensor", "Meta")
def quantize_per_tensor_meta(
        input: torch.Tensor,
        scale: float,
        zero_point: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    # 如果输入张量的数据类型是浮点数16位或者bfloat16，则将其转换为32位浮点数
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    # 断言输入张量的数据类型必须是32位浮点数，否则抛出异常
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    # 返回一个与输入张量相同形状的空张量，数据类型为指定的dtype
    return torch.empty_like(input, dtype=dtype)

# 定义函数，实现对张量进行基于显式自动求导的量化
quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

# 实现函数，对张量进行基于显式自动求导的量化
@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "CompositeExplicitAutograd")
def quantize_per_tensor_tensor(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    """ Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    # 断言零点张量只包含一个元素，否则抛出异常
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    # 断言缩放张量只包含一个元素，否则抛出异常
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    # 调用基于元信息的量化函数，对输入张量进行量化
    return quantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype)

# 定义函数，实现对张量进行基于元信息的量化
@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "Meta")
def quantize_per_tensor_tensor_meta(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    # 如果输入张量的数据类型是浮点数16位或者bfloat16，则将其转换为32位浮点数
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    # 断言零点张量只包含一个元素，否则抛出异常
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    # 断言缩放张量只包含一个元素，否则抛出异常
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    # 断言输入张量的数据类型必须是32位浮点数，否则抛出异常
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    # 返回一个与输入张量相同形状的空张量，数据类型为指定的dtype
    return torch.empty_like(input, dtype=dtype)

# TODO: remove other variants and keep this one
# 定义函数，实现对张量进行基于显式自动求导的量化，支持多维度张量
quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype) -> Tensor")

# 实现函数，对张量进行基于显式自动求导的量化，支持多维度张量
@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor2", "CompositeExplicitAutograd")
def quantize_per_tensor_tensor2(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: torch.Tensor,
        quant_max: torch.Tensor,
        dtype: torch.dtype
) -> torch.Tensor:
    """
    对张量进行仿射量化，使用相同的量化参数将浮点值映射到量化值
    与 `quantize_per_tensor` 相同，但是 scale 和 zero_point 是标量张量而不是标量值
    """
    # 断言确保 zero_point 张量只有一个元素
    assert zero_point.numel() == 1, f"期望 zero_point 张量只有一个元素，但接收到 : {zero_point.numel()}"
    # 断言确保 scale 张量只有一个元素
    assert scale.numel() == 1, f"期望 scale 张量只有一个元素，但接收到 : {scale.numel()}"
    # 调用 quantize_per_tensor 函数进行张量的量化操作，返回量化后的张量
    return quantize_per_tensor(input, scale.item(), zero_point.item(), quant_min.item(), quant_max.item(), dtype)
# 定义 quantize_per_tensor_tensor2_meta 函数，用于量化输入张量。
@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor2", "Meta")
def quantize_per_tensor_tensor2_meta(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: torch.Tensor,
        quant_max: torch.Tensor,
        dtype: torch.dtype
) -> torch.Tensor:
    # 调用 quantize_per_tensor_tensor_meta 函数来执行量化操作，返回量化后的张量
    return quantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype)

# 注释：quant_min/quant_max/dtype 参数在操作中未使用，但作为输入张量的元数据保留在签名中，
# 这可能对将来的模式匹配有用。如果确定没有用例需要它们，稍后会重新审视此部分。
quantized_decomposed_lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor")

# 定义 dequantize_per_tensor 函数，执行张量的仿射反量化，使用相同的量化参数将量化值映射到浮点值
@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor(
        input: torch.Tensor,
        scale: float,
        zero_point: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """ Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per tensor quantized Tensor if combined with
       quantization parameters in the argument of this function (scale/zero_point)

       scale (float): quantization parameter for affine quantization

       zero_point (int): quantization parameter for affine quantization

       quant_min (int): minimum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): dtype for input Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    # 断言输入张量的数据类型与指定的 dtype 一致
    assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}, but got {input.dtype}"
    # 如果未指定输出数据类型，则默认为 torch.float32
    if out_dtype is None:
        out_dtype = torch.float32
    # 如果 dtype 在 _DTYPE_TO_QVALUE_BOUNDS 中
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        # TODO: investigate why
        # (input - zero_point).to(torch.float32) * scale
        # failed the test
        # 返回经过仿射反量化计算的结果
        return (input.to(out_dtype) - zero_point) * scale
    else:
        # 如果不支持输入的 dtype，则抛出 ValueError 异常
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")

# 定义 dequantize_per_tensor_meta 函数，用于元数据处理
@impl(quantized_decomposed_lib, "dequantize_per_tensor", "Meta")
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    # 定义了一个名为 out_dtype 的变量，其类型是 torch.dtype 或者 NoneType
    out_dtype: Optional[torch.dtype] = None
# 定义函数签名，声明返回类型为 torch.Tensor
) -> torch.Tensor:
    # 如果未指定输出数据类型，则默认为 torch.float32
    if out_dtype is None:
        out_dtype = torch.float32
    # 返回一个与输入张量 input 类型相同、数据类型为 out_dtype 的空张量
    return torch.empty_like(input, dtype=out_dtype)

# 使用 quantized_decomposed_lib 定义 dequantize_per_tensor.tensor 的函数签名
quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor")

# 实现 dequantize_per_tensor.tensor 函数的具体逻辑，使用 CompositeExplicitAutograd 作为实现方式
@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor_tensor(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """对张量进行仿射去量化，使用相同的量化参数将量化值映射到浮点数值
    与 `dequantize_per_tensor` 相同，但 scale 和 zero_point 是标量张量而不是标量值
    """
    # 断言 zero_point 张量仅包含一个元素
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    # 断言 scale 张量仅包含一个元素
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    # 调用 dequantize_per_tensor 函数，使用标量形式的 scale 和 zero_point 进行处理
    return dequantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype, out_dtype=out_dtype)

# 使用 quantized_decomposed_lib 定义 dequantize_per_tensor.tensor 的元实现（Meta 实现方式）
@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor", "Meta")
def dequantize_per_tensor_tensor_meta(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    # 如果未指定输出数据类型，则默认为 torch.float32
    if out_dtype is None:
        out_dtype = torch.float32
    # 断言 zero_point 张量仅包含一个元素
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    # 断言 scale 张量仅包含一个元素
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    # 断言输入张量的数据类型为指定的 dtype
    assert input.dtype == dtype, f"Expecting input to have dtype: {dtype}"
    # 如果 dtype 在 _DTYPE_TO_QVALUE_BOUNDS 中，则返回一个与 input 类型相同、数据类型为 out_dtype 的空张量
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return torch.empty_like(input, dtype=out_dtype)
    else:
        # 否则，抛出异常，提示在 dequantize_per_tensor 中不支持的数据类型
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")

# TODO: 移除其他变体，保留这个
# 使用 quantized_decomposed_lib 定义 dequantize_per_tensor.tensor2 的函数签名
quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor")

# 实现 dequantize_per_tensor.tensor2 函数的具体逻辑，使用 CompositeExplicitAutograd 作为实现方式
@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor2", "CompositeExplicitAutograd")
def dequantize_per_tensor_tensor2(
        input: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        quant_min: torch.Tensor,
        quant_max: torch.Tensor,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """对张量进行仿射去量化，使用相同的量化参数将量化值映射到浮点数值
    """
    """
    从量化值转换为浮点数值
    与 `dequantize_per_tensor` 相同，但是 scale 和 zero_point 是标量张量而不是标量数值
    """
    # 断言确保 zero_point 是一个元素，即标量
    assert zero_point.numel() == 1, f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
    # 断言确保 scale 是一个元素，即标量
    assert scale.numel() == 1, f"Expecting scale tensor to be one element, but received : {scale.numel()}"
    # 调用 dequantize_per_tensor 函数，将输入量化张量转换为浮点数张量，
    # 使用 scale.item()、zero_point.item()、quant_min.item()、quant_max.item() 作为参数，
    # 并指定数据类型为 dtype，输出数据类型为 out_dtype
    return dequantize_per_tensor(
        input, scale.item(), zero_point.item(), quant_min.item(), quant_max.item(), dtype, out_dtype=out_dtype)
# 实现一个装饰器，将函数注册到 quantized_decomposed_lib 中，以便以后调用
@impl(quantized_decomposed_lib, "dequantize_per_tensor.tensor2", "Meta")
def dequantize_per_tensor_tensor2_meta(
        input,
        scale,
        zero_point,
        quant_min,
        quant_max,
        dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    # 调用 dequantize_per_tensor_tensor_meta 函数，返回其结果
    return dequantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype, out_dtype=out_dtype)

# 定义一个函数签名，用于指定 input Tensor 的 quantized_decomposed_lib 选择量化参数的方法
quantized_decomposed_lib.define(
    "choose_qparams.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)")

# 将函数注册到 quantized_decomposed_lib 中，指定函数用于选择 input Tensor 的量化参数
@impl(quantized_decomposed_lib, "choose_qparams.tensor", "CompositeExplicitAutograd")
def choose_qparams_tensor(
        input: torch.Tensor,
        qmin: int,
        qmax: int,
        eps: float,
        dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 给定输入 Tensor，从中推导目标量化 Tensor 的每个张量的仿射量化参数
    (scale 和 zero_point)

    Args:
       input (torch.Tensor): 浮点数输入 Tensor
       quant_min (int): 目标量化 Tensor 的最小量化值
       quant_max (int): 目标量化 Tensor 的最大量化值
       dtype (torch.dtype): 目标量化 Tensor 的数据类型

    Returns:
       scale (float): 目标量化 Tensor 的量化参数
       zero_point (int): 目标量化 Tensor 的量化参数
    """
    # 断言 input 的数据类型是 torch.float32/16/b16 中的一种
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
    # 断言 dtype 是 _DTYPE_TO_QVALUE_BOUNDS 中的一种数据类型
    assert dtype in _DTYPE_TO_QVALUE_BOUNDS, \
        f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"
    # 验证 qmin 和 qmax 的有效性
    validate_qmin_qmax(qmin, qmax)

    # 计算输入 Tensor 的最小值和最大值
    min_val, max_val = torch.aminmax(input)

    # 调用 determine_qparams 函数，返回量化参数 scale 和 zero_point
    return determine_qparams(
        min_val, max_val, qmin, qmax, dtype, torch.Tensor([eps]), has_customized_qrange=False)

# 定义一个函数签名，用于指定 input Tensor 的 quantized_decomposed_lib 对称量化参数选择的方法
quantized_decomposed_lib.define(
    "choose_qparams_symmetric.tensor(Tensor input, int quant_min, int quant_max, "
    "float eps, ScalarType dtype) -> (Tensor, Tensor)")

# 将函数注册到 quantized_decomposed_lib 中，指定函数用于选择 input Tensor 的对称量化参数
@impl(quantized_decomposed_lib, "choose_qparams_symmetric.tensor", "CompositeExplicitAutograd")
def choose_qparams_symmetric_tensor(
        input: torch.Tensor,
        qmin: int,
        qmax: int,
        eps: float,
        dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 给定输入 Tensor，从中推导目标量化 Tensor 的每个张量的仿射量化参数
    (scale 和 zero_point)

    Args:
       input (torch.Tensor): 浮点数输入 Tensor
       quant_min (int): 目标量化 Tensor 的最小量化值
       quant_max (int): 目标量化 Tensor 的最大量化值
       dtype (torch.dtype): 目标量化 Tensor 的数据类型

    Returns:
       scale (float): 目标量化 Tensor 的量化参数
       zero_point (int): 目标量化 Tensor 的量化参数
    """
    """
    Calculate quantization parameters for a given input tensor.

    Args:
        input (torch.Tensor): Input tensor to be quantized.
        dtype (torch.dtype): Target dtype for the quantized tensor.
        qmin (int): Minimum quantization value.
        qmax (int): Maximum quantization value.
        eps (float): Small epsilon value for numerical stability.
        
    Raises:
        AssertionError: If input dtype is not torch.float32/16/b16, or if dtype is not supported.
        
    Returns:
        scale (float): Quantization parameter for the target quantized Tensor.
        zero_point (int): Quantization parameter for the target quantized Tensor.
    """

    # Ensure the input tensor's dtype is one of the supported floating point types
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"

    # Ensure the target dtype is supported for quantization
    assert dtype in _DTYPE_TO_QVALUE_BOUNDS, \
        f"Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}"

    # Validate the range of qmin and qmax
    validate_qmin_qmax(qmin, qmax)

    # Calculate the minimum and maximum values of the input tensor
    min_val, max_val = torch.aminmax(input)

    # Return quantization parameters using the determined_qparams function
    return determine_qparams(
        min_val,
        max_val,
        qmin,
        qmax,
        dtype,
        torch.Tensor([eps]),
        has_customized_qrange=False,
        qscheme=torch.per_tensor_symmetric
    )
# 使用装饰器将函数注册到 quantized_decomposed_lib，并指定名称和标识符
@impl(quantized_decomposed_lib, "choose_qparams.tensor", "Meta")
# 定义一个函数，用于根据输入张量选择量化参数，返回两个张量的元组
def choose_qparams_tensor_meta(
        input: torch.Tensor,  # 输入张量
        quant_min: int,  # 量化的最小值
        quant_max: int,  # 量化的最大值
        eps: float,  # 一个小的正数用于数值稳定性
        dtype: torch.dtype  # 输出张量的数据类型
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 断言输入张量的数据类型必须是 torch.float32/16/b16 中的一种
    assert input.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ], f"Expecting input to have dtype torch.float32/16/b16, but got dtype: {input.dtype}"
    # 断言量化的最小值必须小于最大值
    assert quant_min < quant_max, f"Expecting quant_min to be smaller than quant_max but received min: \
        {quant_min} max: {quant_max}"
    # 返回两个空张量，一个双精度浮点型，一个64位整型，均使用输入张量的设备
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(1, dtype=torch.int64, device=input.device)

@impl(quantized_decomposed_lib, "choose_qparams_symmetric.tensor", "Meta")
# 定义一个函数，用于根据输入张量选择对称量化参数，返回两个张量的元组
def choose_qparams_symmetric_tensor_meta(
        input: torch.Tensor,  # 输入张量
        quant_min: int,  # 量化的最小值
        quant_max: int,  # 量化的最大值
        eps: float,  # 一个小的正数用于数值稳定性
        dtype: torch.dtype  # 输出张量的数据类型
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 返回两个空张量，一个双精度浮点型，一个64位整型，均使用输入张量的设备
    return torch.empty(1, dtype=torch.double, device=input.device), torch.empty(1, dtype=torch.int64, device=input.device)

# 定义一个辅助函数，用于将输入张量的指定轴置换到第0维
def _permute_to_axis_zero(x, axis):
    # 创建一个新的轴顺序列表
    new_axis_list = list(range(x.dim()))
    # 将指定轴移动到第0维
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    # 根据新的轴顺序列表重新排列输入张量，并返回排列后的张量和新的轴顺序列表
    y = x.permute(tuple(new_axis_list))
    return y, new_axis_list

# quantized_decomposed_lib 定义一个方法，命名为 "quantize_per_channel"，接受一系列参数描述了张量和它的量化轴
quantized_decomposed_lib.define(
    "quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor")

@impl(quantized_decomposed_lib, "quantize_per_channel", "CompositeExplicitAutograd")
# 实现一个函数，用于沿任意轴进行逐通道量化
def quantize_per_channel(
        input: torch.Tensor,  # 输入张量
        scales: torch.Tensor,  # 用于每个通道的量化参数
        zero_points: torch.Tensor,  # 每个通道的零点参数
        axis: int,  # 量化的轴
        quant_min: int,  # 量化的最小值
        quant_max: int,  # 量化的最大值
        dtype: torch.dtype  # 输出张量的数据类型
) -> torch.Tensor:
    """ Affine per channel quantization for the Tensor using the same quantization
    parameters for each channel/axis to map from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel
       zero_point (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    # 如果输入张量的数据类型是 torch.float16 或 torch.bfloat16，将其转换为 torch.float32
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    # 检查输入张量的数据类型是否为torch.float32，若不是则抛出异常
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    
    # 检查轴的索引是否小于输入张量的维度数，若不是则抛出异常
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    
    # 执行输入张量的量化范围检查，确保量化参数在有效范围内
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    
    # 将输入张量按照指定轴重新排列，并返回重新排列后的张量及排列索引列表
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    
    # 创建一个与输入张量相同形状的零张量
    res = torch.zeros_like(input)
    
    # 遍历输入张量的第一维度（通常是batch维度）
    for i in range(input.size(0)):
        # 对每个元素执行量化操作：乘以逆缩放因子后四舍五入，并加上零点偏移量，然后将结果夹紧到[quant_min, quant_max]范围内
        res[i] = torch.clamp(
            torch.round(input[i] * (1.0 / scales[i])) + zero_points[i],
            quant_min,
            quant_max
        )
    
    # 将结果张量按照指定的排列索引列表重新排列
    out = res.permute(tuple(permute_axis_list))
    
    # 将输出张量转换为指定的数据类型，并返回结果
    return out.to(dtype)
@impl(quantized_decomposed_lib, "quantize_per_channel", "Meta")
# 使用装饰器将函数注册到 quantized_decomposed_lib 库中，函数名为 quantize_per_channel，注释为 "Meta"
def quantize_per_channel_meta(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype
) -> torch.Tensor:
    # 如果输入张量的数据类型是 torch.float16 或 torch.bfloat16，则转换为 torch.float32
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    # 断言输入张量的数据类型为 torch.float32
    assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
    # 断言轴的值小于输入张量的维度数
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    # 调用 _quant_min_max_bounds_check 函数，检查 quant_min、quant_max 和 dtype 的边界
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # 返回一个与输入张量相同形状的空张量，数据类型为 dtype
    return torch.empty_like(input, dtype=dtype)

# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it

quantized_decomposed_lib.define(
    "dequantize_per_channel(Tensor input, Tensor scales, Tensor? zero_points, int axis, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor")
# 使用 quantized_decomposed_lib.define 方法定义函数签名，函数名为 dequantize_per_channel，定义包含输入参数及返回值的元数据

@impl(quantized_decomposed_lib, "dequantize_per_channel", "CompositeExplicitAutograd")
# 使用装饰器将函数注册到 quantized_decomposed_lib 库中，函数名为 dequantize_per_channel，注释为 "CompositeExplicitAutograd"
def dequantize_per_channel(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: Optional[torch.Tensor],
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """ Affine per channel dequantization for the Tensor using the same quantization
    parameters for each channel/axis to map from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with
       quantization parameter in the argument of this function (scales/zero_points/axis)

       scales (torch.Tensor): a list of scale quantization parameter for
       affine quantization, one per channel

       zero_points (torch.Tensor): a list of zero_point quantization parameter for
       affine quantization, one per channel

       quant_min (int): minimum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for output Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    # 断言输入张量的数据类型为指定的 dtype
    assert input.dtype == dtype, f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
    # 如果 out_dtype 为 None，则将其设为 torch.float32
    if out_dtype is None:
        out_dtype = torch.float32
    # 确保给定的轴数小于输入张量的维度，否则触发断言错误
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    # 检查量化参数的有效性，确保量化最小值、最大值和数据类型的一致性
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # 将输入张量的指定轴移动到第一维，并返回重新排列后的张量及其轴置换列表
    input, permute_axis_list = _permute_to_axis_zero(input, axis)
    # 创建一个和输入张量相同形状的全零张量，指定数据类型为输出数据类型
    res = torch.zeros_like(input, dtype=out_dtype)

    # 遍历输入张量的第一维
    for i in range(input.size(0)):
        # 如果给定了零点列表，则使用对应索引处的零点值，否则使用默认值 0
        zp = zero_points[i] if zero_points is not None else 0
        # 计算量化后的结果，出现问题时，标记为待调查
        # (input[i] - zero_points[i]).to(out_dtype) * scales[i]
        # 未通过测试
        res[i] = (input[i].to(out_dtype) - zp) * scales[i]

    # 根据保存的轴置换列表重新排列结果张量的轴顺序
    out = res.permute(tuple(permute_axis_list))
    # 返回重排列后的输出张量
    return out
# 使用装饰器将函数注册到 quantized_decomposed_lib，函数名为 "dequantize_per_channel"，注释为 "Meta"
@impl(quantized_decomposed_lib, "dequantize_per_channel", "Meta")
def dequantize_per_channel_meta(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: Optional[torch.Tensor],
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: torch.dtype,
        *,
        out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    # 检查输入张量的数据类型是否符合预期
    assert input.dtype == dtype, f"Expecting input to have dtype {dtype}, but got dtype: {input.dtype}"
    # 如果未指定输出数据类型，则默认为 torch.float32
    if out_dtype is None:
        out_dtype = torch.float32
    # 检查轴的范围是否在输入张量的维度内
    assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
    # 检查量化范围是否符合给定数据类型的范围
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # 创建一个与输入张量相同大小的空张量，指定数据类型为 out_dtype
    return torch.empty_like(input, dtype=out_dtype)


# 使用 quantized_decomposed_lib.define 注册函数签名 "choose_qparams_per_token(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
quantized_decomposed_lib.define(
    "choose_qparams_per_token(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


# 使用装饰器将函数注册到 quantized_decomposed_lib，函数名为 "choose_qparams_per_token"，注释为 "CompositeExplicitAutograd"
@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # 计算每个 N 维度元素的绝对值的最大值，保持维度
    scales = input.abs().amax(dim=-1, keepdim=True)
    # 如果 scales 的数据类型为 torch.float16，则将其转换为 float 类型，避免在 fp16 下的溢出问题（bf16 有足够宽的范围）
    if scales.dtype == torch.float16:
        scales = (
            scales.float()
        )  # want float scales to avoid overflows for fp16, (bf16 has wide enough range)
    # 如果 dtype 为 torch.int8，则量化最大值为 2^(n_bits - 1) - 1
    if dtype == torch.int8:
        n_bits = 8
        quant_max = 2 ** (n_bits - 1) - 1
    else:
        # 如果不支持给定的 dtype，则抛出异常
        raise Exception(f"unsupported dtype in choose_qparams_per_token: {dtype}")  # noqa: TRY002

    # 将 scales 限制在最小值为 1e-5 的范围内，并将其除以 quant_max
    scales = scales.clamp(min=1e-5).div(quant_max)
    # 创建一个与 scales 张量相同大小的零张量
    zero_points = torch.zeros_like(scales)
    return scales, zero_points


# 使用装饰器将函数注册到 quantized_decomposed_lib，函数名为 "choose_qparams_per_token"，注释为 "Meta"
@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token",
    "Meta",
)
def choose_qparams_per_token_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 创建一个大小为 (1, input.size(-1)) 的空张量，数据类型为 torch.double，设备为 input 的设备
    size = (1, input.size(-1))
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


# 使用 quantized_decomposed_lib.define 注册函数签名 "_choose_qparams_per_token_asymmetric_impl(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
quantized_decomposed_lib.define(
    "_choose_qparams_per_token_asymmetric_impl(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)


# 使用装饰器将函数注册到 quantized_decomposed_lib，函数名为 "_choose_qparams_per_token_asymmetric_impl"，注释为 "CompositeImplicitAutograd"
@impl(
    quantized_decomposed_lib,
    "_choose_qparams_per_token_asymmetric_impl",
    "CompositeImplicitAutograd",
)
def _choose_qparams_per_token_asymmetric_impl(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Choose quantization parameters for per token quantization. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32/float16 Tensor
       dtype (torch.dtype): dtype (e.g. torch.uint8) for input Tensor

    Returns:
        scales and zero_points, both float32 Tensors
    """
    # 定义量化的最小值和最大值
    qmin, qmax = -128, 127
    # 计算输入张量在最后一个维度上的最小值和最大值
    min_val = torch.amin(input, dim=-1, keepdim=True)
    max_val = torch.amax(input, dim=-1, keepdim=True)
    # 计算最小值的负数部分和最大值的正数部分
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    # 获取浮点数的最小值 epsilon
    eps = torch.finfo(torch.float32).eps  # use xnnpack eps?

    # 计算缩放因子 scale
    scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
    scale = scale.clamp(min=eps)

    # 计算零点 zero point
    descaled_min = min_val_neg / scale
    descaled_max = max_val_pos / scale
    zero_point_from_min_error = qmin + descaled_min
    zero_point_from_max_error = qmax + descaled_max
    zero_point = torch.where(
        zero_point_from_min_error + zero_point_from_max_error > 0,
        qmin - descaled_min,
        qmax - descaled_max,
    )
    zero_point = torch.clamp(zero_point, qmin, qmax).round()

    # 返回计算得到的 scale 和 zero_point，转换为 float32 类型
    return scale.to(torch.float32), zero_point.to(torch.float32)
# 定义一个函数选择每个标记的非对称量化参数
quantized_decomposed_lib.define(
    "choose_qparams_per_token_asymmetric(Tensor input, ScalarType dtype) -> (Tensor, Tensor)"
)

# 实现函数的具体逻辑，根据给定的输入和数据类型选择每个标记的非对称量化参数
@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "CompositeExplicitAutograd",
)
def choose_qparams_per_token_asymmetric(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _choose_qparams_per_token_asymmetric_impl(input, dtype)


# 实现函数的元信息版本，返回预定义大小的空张量，用于非对称量化
@impl(
    quantized_decomposed_lib,
    "choose_qparams_per_token_asymmetric",
    "Meta",
)
def choose_qparams_per_token_asymmetric_meta(
    input: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = (1, input.size(-1))
    return torch.empty(size, dtype=torch.double, device=input.device), torch.empty(
        size, dtype=torch.int64, device=input.device
    )


# 检查每个标记的量化参数维度是否匹配输入张量的维度
def _per_token_quant_qparam_dim_check(input, scales, zero_points):
    num_tokens = math.prod(list(input.size())[:-1])
    assert (
        num_tokens == scales.numel()
    ), f"num_tokens: {num_tokens} scales: {scales.size()}"
    assert (
        num_tokens == zero_points.numel()
    ), f"num_tokens: {num_tokens} zero_points: {zero_points.size()}"

# 定义一个函数执行每个标记的量化操作
quantized_decomposed_lib.define(
    "quantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)

# 实现函数的具体逻辑，使用给定的量化参数将输入张量从浮点数映射到量化值
@impl(quantized_decomposed_lib, "quantize_per_token", "CompositeExplicitAutograd")
def quantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    """Per token quantization for the Tensor using the quantization parameters to map
    from floating point to quantized values. This means for a N dimension Tensor
    (M1, M2, ...Mn, N), we calculate scales/zero_points for each N elements and quantize
    every N elements with the same quantization parameter. The dimension for scales/zero_points
    will be (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per token affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per token affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    # 检查量化范围是否在合理范围内
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # 检查每个标记的量化参数维度是否匹配输入张量的维度
    _per_token_quant_qparam_dim_check(input, scales, zero_points)
    # 对输入张量进行量化操作，然后进行四舍五入，并将结果限制在给定的量化范围内，最后转换为指定的数据类型
    input = (
        input.mul(1.0 / scales).add(zero_points).round().clamp(quant_min, quant_max).to(dtype)
    )
    return input
# 使用装饰器将函数注册到 quantized_decomposed_lib 的 quantize_per_token 功能中，并命名为 "Meta"
@impl(quantized_decomposed_lib, "quantize_per_token", "Meta")
def quantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
):
    # 检查量化范围是否有效
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # 返回一个与 input 形状相同但数据类型为 dtype 的空 Tensor
    return torch.empty_like(input, dtype=dtype)


# 定义 quantized_decomposed_lib 中的 "dequantize_per_token" 函数签名及其功能说明
quantized_decomposed_lib.define(
    "dequantize_per_token(Tensor input, Tensor scales, Tensor zero_points, "
    "int quant_min, int quant_max, ScalarType dtype, ScalarType output_dtype) -> Tensor"
)


# 使用装饰器将函数注册到 quantized_decomposed_lib 的 dequantize_per_token 功能中，并命名为 "CompositeExplicitAutograd"
@impl(quantized_decomposed_lib, "dequantize_per_token", "CompositeExplicitAutograd")
def dequantize_per_token(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    """对 Tensor 进行按元素解量化，使用量化参数将浮点数映射到量化值。对于一个 N 维 Tensor
    (M1, M2, ...Mn, N)，我们为每个 N 元素计算 scales/zero_points，并使用相同的量化参数
    量化每个 N 元素。scales/zero_points 的维度为 (M1 * M2 ... * Mn)

    Args:
       input (torch.Tensor): 量化的 Tensor (uint8, int8 等)
       scales (float32 torch.Tensor): 每元素的仿射量化参数
       zero_points (int32 torch.Tensor): 每元素的仿射量化参数
       quant_min (int): 输入 Tensor 的最小量化值
       quant_max (int): 输入 Tensor 的最大量化值
       dtype (torch.dtype): 输入 Tensor 的数据类型 (例如 torch.uint8)
       output_dtype (torch.dtype): 输出 Tensor 的数据类型 (例如 torch.float32)

    Returns:
       dtype 为 `output_dtype` 的解量化 Tensor
    """
    # 减去 zero_points，将 input 转换为 output_dtype 类型，乘以 scales
    input = input - zero_points
    input = input.to(output_dtype) * scales
    return input


# 使用装饰器将函数注册到 quantized_decomposed_lib 的 dequantize_per_token 功能中，并命名为 "Meta"
@impl(quantized_decomposed_lib, "dequantize_per_token", "Meta")
def dequantize_per_token_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype = torch.float32,
):
    # 检查量化范围是否有效
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    # TODO: 暂不支持 fp16，返回一个与 input 形状相同但数据类型为 output_dtype 的空 Tensor
    return torch.empty_like(input, dtype=output_dtype)


# 定义 quantized_decomposed_lib 中的 "quantize_per_channel_group" 函数签名及其功能说明
quantized_decomposed_lib.define(
    "quantize_per_channel_group(Tensor input, Tensor scales, Tensor zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size) -> Tensor"
)


# 使用装饰器将函数注册到 quantized_decomposed_lib 的 quantize_per_channel_group 功能中，并命名为 "CompositeExplicitAutograd"
# TODO: 目前忽略 dtype 参数
@impl(
    quantized_decomposed_lib, "quantize_per_channel_group", "CompositeExplicitAutograd"
)
def quantize_per_channel_group(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    # 断言 group_size 大于 1
    assert group_size > 1
    # 用于 GPTQ 单列量化所需
    # 如果 group_size 大于输入张量的最后一个维度长度，并且 scales 的最后一个维度长度为 1，则将 group_size 设置为输入张量的最后一个维度长度
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    # 断言：输入张量的最后一个维度长度应该能被 group_size 整除
    assert input.shape[-1] % group_size == 0
    # 断言：输入张量的维度应该为 2
    assert input.dim() == 2

    # TODO: 检查 dtype，目前无法表示 torch.int4，因此忽略此处
    # 将输入张量重塑为 (-1, group_size) 的形状，用于量化处理
    to_quant = input.reshape(-1, group_size)
    # 断言：检查重塑后的张量中是否有 NaN 值
    assert torch.isnan(to_quant).sum() == 0

    # 将 scales 和 zero_points 重塑为 (-1, 1) 的形状
    scales = scales.reshape(-1, 1)
    zero_points = zero_points.reshape(-1, 1)

    # 将输入张量量化为 int8 类型
    input_int8 = (
        to_quant.mul(1.0 / scales)  # 按 scales 缩放
        .add(zero_points)           # 加上 zero_points
        .round()                    # 四舍五入
        .clamp_(quant_min, quant_max)  # 裁剪到指定的范围 [quant_min, quant_max]
        .to(dtype)                  # 转换为指定的数据类型 dtype
        .reshape_as(input)          # 重塑为与原输入张量相同的形状
    )

    # 返回量化后的 int8 张量
    return input_int8
# 使用装饰器实现函数注册到 quantized_decomposed_lib 中，函数名为 "quantize_per_channel_group_meta"，元信息为 "Meta"
@impl(quantized_decomposed_lib, "quantize_per_channel_group", "Meta")
def quantize_per_channel_group_meta(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size=128,
):
    """Groupwise quantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scales (float32 torch.Tensor): quantization parameter for per channel group affine quantization
       zero_points (int32 torch.Tensor): quantization parameter for per channel group affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    # 断言 group_size 大于 1
    assert group_size > 1
    # 如果 group_size 大于 input 张量的最后一个维度大小，并且 scales 张量的最后一个维度大小为 1，
    # 则将 group_size 设置为 input 张量的最后一个维度大小
    if group_size > input.shape[-1] and scales.shape[-1] == 1:
        group_size = input.shape[-1]

    # 断言 input 张量的最后一个维度能够整除 group_size
    assert input.shape[-1] % group_size == 0
    # 断言 input 张量的维度为 2
    assert input.dim() == 2

    # 创建一个与 input 张量形状相同、dtype 为指定类型的空张量，并返回
    return torch.empty_like(input, dtype=dtype)


# 定义 quantized_decomposed_lib 中的函数 "dequantize_per_channel_group" 的注册信息
quantized_decomposed_lib.define(
    "dequantize_per_channel_group(Tensor input, Tensor scales, Tensor? zero_points, int quant_min, "
    "int quant_max, ScalarType dtype, int group_size, ScalarType output_dtype) -> Tensor"
)


# 使用装饰器实现函数注册到 quantized_decomposed_lib 中，函数名为 "dequantize_per_channel_group"，元信息为 "CompositeExplicitAutograd"
@impl(
    quantized_decomposed_lib,
    "dequantize_per_channel_group",
    "CompositeExplicitAutograd",
)
def dequantize_per_channel_group(
    w_int8: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor],
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    group_size: int = 128,
    output_dtype: torch.dtype = torch.float32,
):
    """Groupwise dequantization within each channel for an 2-d Tensor using the quantization parameters
    to map from floating point to quantized values. This means for each row of a 2-d Tensor
    (M, N), we calculate scales/zero_points for each `group_size` elements
    and quantize every `group_size` elements with the same quantization parameter.
    The dimension for scales/zero_points will be (M * ceil(N, group_size),)
    """
    # 确保 group_size 大于 1
    assert group_size > 1
    # 如果 group_size 大于 w_int8 张量的最后一个维度，并且 scales 张量的最后一个维度为 1，调整 group_size 的值为 w_int8 张量的最后一个维度的大小
    if group_size > w_int8.shape[-1] and scales.shape[-1] == 1:
        group_size = w_int8.shape[-1]
    # 确保 w_int8 张量的最后一个维度可以被 group_size 整除
    assert w_int8.shape[-1] % group_size == 0
    # 确保 w_int8 张量是二维的
    assert w_int8.dim() == 2
    
    # 将 w_int8 张量重新形状为 (-1, group_size)，实现按 group_size 分组
    w_int8_grouped = w_int8.reshape(-1, group_size)
    # 将 scales 张量重新形状为 (-1, 1)，以便进行按通道组的仿射量化
    scales = scales.reshape(-1, 1)
    # 如果 zero_points 不为 None，则将其重新形状为 (-1, 1)，否则创建一个与 scales 同设备的 dtype 为 torch.int32 的零张量 zp
    if zero_points is not None:
        zp = zero_points.reshape(-1, 1)
    else:
        zp = torch.zeros([], dtype=torch.int32, device=scales.device)
    # 计算去量化后的 w_int8 张量，公式为 (w_int8_grouped - zp) * scales，并将结果重新形状为与 w_int8 相同的形状，并转换为 output_dtype 类型
    w_dq = w_int8_grouped.sub(zp).mul(scales).reshape_as(w_int8).to(output_dtype)
    # 返回去量化后的张量 w_dq，其类型为 output_dtype
    return w_dq
# 定义一个函数或方法，将"fake_quant_per_channel"名称绑定到指定的函数或方法上
quantized_decomposed_lib.define(
    "fake_quant_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, "
    "int quant_min, int quant_max) -> Tensor")

# 定义一个名为FakeQuantPerChannel的类，继承自torch.autograd.Function
class FakeQuantPerChannel(torch.autograd.Function):
    
    @staticmethod
    # 实现前向传播函数
    def forward(ctx, input, scales, zero_points, axis, quant_min, quant_max):
        # 如果scales的数据类型不是torch.float32，则转换为torch.float32
        if scales.dtype != torch.float32:
            scales = scales.to(torch.float32)
        # 如果zero_points的数据类型不是torch.int32，则转换为torch.int32
        if zero_points.dtype != torch.int32:
            zero_points = zero_points.to(torch.int32)
        # 断言输入的数据类型为torch.float32
        assert input.dtype == torch.float32, f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        # 断言axis小于输入张量的维度数
        assert axis < input.dim(), f"Expecting axis to be < {input.dim()}"
        # 确定要广播的维度列表，排除指定的axis维度
        broadcast_dims = list(range(0, axis)) + list(range(axis + 1, input.ndim))
        # 对scales在指定的广播维度上执行unsqueeze操作
        unsqueeze_scales = _unsqueeze_multiple(scales, broadcast_dims)
        # 对zero_points在指定的广播维度上执行unsqueeze操作
        unsqueeze_zero_points = _unsqueeze_multiple(zero_points, broadcast_dims)
        # 计算临时变量temp，执行量化伪操作
        temp = torch.round(input * (1.0 / unsqueeze_scales)) + unsqueeze_zero_points
        # 根据量化范围对temp进行截断操作，得到输出out
        out = (torch.clamp(temp, quant_min, quant_max) - unsqueeze_zero_points) * unsqueeze_scales
        # 创建一个掩码，用于标识在量化范围内的元素
        mask = torch.logical_and((temp >= quant_min), (temp <= quant_max))
        
        # 在上下文中保存掩码，以便在反向传播中使用
        ctx.save_for_backward(mask)
        # 返回前向传播计算的结果out
        return out

    @staticmethod
    # 实现反向传播函数
    def backward(ctx, gy):
        # 从上下文中获取保存的掩码
        mask, = ctx.saved_tensors
        # 返回反向传播的梯度，其形状与输入保持一致
        return gy * mask, None, None, None, None, None

# 将fake_quant_per_channel方法绑定到quantized_decomposed_lib和其名称的元数据上，实现"Autograd"方式
@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Autograd")
def fake_quant_per_channel(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
) -> torch.Tensor:
    # 调用FakeQuantPerChannel类的apply方法执行量化伪操作的前向传播计算
    return FakeQuantPerChannel.apply(input, scales, zero_points, axis, quant_min, quant_max)

# 将fake_quant_per_channel_meta方法绑定到quantized_decomposed_lib和其名称的元数据上，实现"Meta"方式
@impl(quantized_decomposed_lib, "fake_quant_per_channel", "Meta")
def fake_quant_per_channel_meta(
        input: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        axis: int,
        quant_min: int,
        quant_max: int,
) -> torch.Tensor:
    # 返回一个与输入张量形状相同但未初始化的张量，用作Meta方式的伪量化输出
    return torch.empty_like(input)
```