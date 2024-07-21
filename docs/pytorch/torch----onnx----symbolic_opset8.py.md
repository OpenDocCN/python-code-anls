# `.\pytorch\torch\onnx\symbolic_opset8.py`

```py
"""
Note [ONNX operators that are added/updated from opset 8 to opset 9]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
    Compress
    ConstantOfShape
    EyeLike
    MaxUnpool
    OneHot
    Sinh
    Cosh
    Asinh
    Acosh
    Atanh
    Shrink
    IsNaN
    Sign
    Erf
    Scatter
    Where
    NonZero
    TfIdfVectorizer
    MeanVarianceNormalization

Updated operators:
    BatchNormalization: removed spatial attribute.
    Greater, Less, Constant, MatMul, PRelu, Gemm, Flatten: more data types{integers} supported.
    Cast: more data types{string} supported.
    Upsample: moved scales from attribute to input.
    Scan
"""

import functools
import warnings

import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import _type_utils, errors, symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration

# Partial function to set opset version to 8 for ONNX symbolic functions
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=8)

# List of operators that are blocked for ONNX export
block_listed_operators = (
    "nonzero",
    "where",
    "scatter",
    "scatter_add",
    "erf",
    "sign",
    "isnan",
    "gather",
    "arange",
    "masked_fill",
    "index_fill",
    "index_copy",
    "repeat_interleave",
    "any",
    "all",
)

# Loop through block listed operators and apply blocking in opset 8
for block_listed_op in block_listed_operators:
    _onnx_symbolic(f"aten::{block_listed_op}")(
        symbolic_helper._block_list_in_opset(block_listed_op)
    )

# Decorator for ONNX symbolic functions for different types of upsampling
@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[symbolic_helper._apply_params("upsample_linear1d", 3, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear")],
)
def _interpolate(name, dim, interpolate_mode):
    # This function decorates ONNX symbolic functions for various types of upsampling
    pass
    # 定义一个符号化函数 symbolic_fn，用于生成符号化的操作图
    def symbolic_fn(g, input, output_size, *args):
        # 调用 symbolic_helper._get_interpolate_attributes 函数获取插值的尺度和对齐角
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        # 发出关于插值的警告信息
        symbolic_helper._interpolate_warning(interpolate_mode)
        # 尝试将 align_corners 转换为标量
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        # 如果 align_corners 为真，则返回未实现的警告信息
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True", input)
        # 尝试获取输出大小的常量值
        output_size = symbolic_helper._maybe_get_const(output_size, "is")
        # 如果 output_size 是一个值，返回未实现的警告信息
        if symbolic_helper._is_value(output_size):
            return symbolic_helper._unimplemented(
                name, "torch._C.Value (output_size) indexing"
            )
        # 如果 scales 为空，则根据输入和输出的尺寸计算默认的缩放比例
        if scales is None:
            scales = [
                1.0
                if i < 2
                else float(output_size[-(dim - i)])
                / float(input.type().sizes()[-(dim - i)])
                for i in range(0, dim)
            ]
        # 使用 g.op 创建一个 Upsample 操作，并传入输入、插值模式和缩放比例
        return g.op("Upsample", input, mode_s=interpolate_mode, scales_f=scales)

    # 返回符号化函数 symbolic_fn
    return symbolic_fn
# 使用装饰器将函数注册为对应的ONNX符号函数，处理 "aten::__interpolate" 操作
@_onnx_symbolic("aten::__interpolate")
def __interpolate(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    # 检查并获取是否为常量，如果是，并且 align_corners 为 True，则返回未实现的错误信息
    align_corners = symbolic_helper._maybe_get_const(align_corners, "b")
    if not symbolic_helper._is_none(align_corners) and align_corners:
        return symbolic_helper._unimplemented("interpolate", "align_corners == True")

    # 检查并获取是否为常量，如果是，并且 scale_factor 是一个值，则返回未实现的错误信息
    if not symbolic_helper._is_none(scale_factor) and symbolic_helper._is_value(
        scale_factor
    ):
        return symbolic_helper._unimplemented(
            "interpolate", "dynamic scales in opset 8"
        )

    # 检查并获取是否为常量，如果是，并且 size 是一个值，则返回未实现的错误信息
    if not symbolic_helper._is_none(size) and symbolic_helper._is_value(size):
        return symbolic_helper._unimplemented("interpolate", "dynamic size in opset 8")

    # 获取插值的缩放因子和模式
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    # 返回一个 Upsample 操作的图节点，输入为 input，模式为 mode，缩放因子为 scales
    return g.op("Upsample", input, mode_s=mode, scales_f=scales)


# 注意：在解决 "cast" 操作符的形状/类型传播问题后，应该为这种操作创建一个包装器。
#       某些符号函数依赖于输入张量的形状信息，但在转换后可能丢失。
def _try_cast_integer_to_float(g: jit_utils.GraphContext, *args):
    # 浮点数标量类型集合
    floating_scalar_types = {
        _type_utils.JitScalarType.HALF,
        _type_utils.JitScalarType.FLOAT,
        _type_utils.JitScalarType.DOUBLE,
    }
    old_type = None
    # 获取第一个参数的类型
    arg0_type = _type_utils.JitScalarType.from_value(
        args[0], _type_utils.JitScalarType.UNDEFINED
    )
    # 如果参数类型已知且不是未定义的，执行以下操作
    if arg0_type != _type_utils.JitScalarType.UNDEFINED:
        old_type = arg0_type
        # 如果旧类型不在浮点数类型集合中，则将参数转换为 Float 类型
        if old_type not in floating_scalar_types:
            old_type = old_type.scalar_name()
            args = tuple(
                g.op("Cast", arg, to_i=_C_onnx.TensorProtoDataType.FLOAT)
                for arg in args
            )
        else:
            # 如果已经是浮点数类型，则直接返回
            return (None,) + args
    else:
        # 如果参数类型未知，则发出警告
        warnings.warn(
            "Only floating datatype is supported for these operators: "
            "{Greater, Less, MatMul, PRelu, Gemm, Flatten}. This might cause "
            "the onnx model to be incorrect, if inputs have integer datatypes."
        )
    # 返回转换前的类型和转换后的参数
    return (old_type,) + args


def _cast_to_type(g: jit_utils.GraphContext, input, to_type):
    # 如果目标类型为 None，则直接返回输入
    if to_type is None:
        return input
    # 否则根据目标类型调用对应的转换函数进行类型转换
    return getattr(opset9, f"_cast_{to_type}")(g, input, False)


def _comparison_operator(g: jit_utils.GraphContext, input, other, op_name):
    # 获取标量形式的其他参数
    other = symbolic_helper._maybe_get_scalar(other)
    # 将其他参数按照输入的标量类型进行转换
    other = symbolic_helper._if_scalar_type_as(other, input)
    # 尝试将输入参数转换为浮点数类型，返回转换前的类型和转换后的参数
    _, input, other = _try_cast_integer_to_float(g, input, other)
    # 返回比较操作的图节点，操作名称为 op_name，输入为 input 和 other
    return g.op(op_name, input, other)
# 符号操作的注解，处理在 opset8 中不支持整数输入类型的情况。如有可能，将其转换为浮点数。
@_onnx_symbolic("aten::gt")
def gt(g: jit_utils.GraphContext, input, other):
    return _comparison_operator(g, input, other, "Greater")


# 符号操作的注解，处理在 opset8 中不支持整数输入类型的情况。如有可能，将其转换为浮点数。
@_onnx_symbolic("aten::lt")
def lt(g: jit_utils.GraphContext, input, other):
    return _comparison_operator(g, input, other, "Less")


# 符号操作的注解，处理在 opset8 中不支持整数输入类型的情况。如果输入是标量，则尝试将整数转换为浮点数。
# 如果无法转换，则保持原类型。
@_onnx_symbolic("aten::bmm")
def bmm(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, other = _try_cast_integer_to_float(g, self, other)
        return _cast_to_type(g, g.op("MatMul", self, other), old_type)
    else:
        return g.op("MatMul", self, other)


# 符号操作的注解，将 matmul 转换为 bmm 操作。
@_onnx_symbolic("aten::matmul")
def matmul(g: jit_utils.GraphContext, self, other):
    return bmm(g, self, other)


# 符号操作的注解，处理 PReLU 操作。根据输入张量的秩和权重的大小进行必要的调整。
# 如果输入是标量，则尝试将整数转换为浮点数。如果无法转换，则保持原类型。
@_onnx_symbolic("aten::prelu")
def prelu(g: jit_utils.GraphContext, self, weight):
    self_rank = symbolic_helper._get_tensor_rank(self)
    weight_sizes = symbolic_helper._get_tensor_sizes(weight)
    if self_rank is not None and self_rank > 2:
        weight = g.op("Unsqueeze", weight, axes_i=list(range(1, self_rank - 1)))
    elif self_rank == 0 and weight_sizes == [1]:
        weight = symbolic_helper._squeeze_helper(g, weight, [0])
    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, weight = _try_cast_integer_to_float(g, self, weight)
        return _cast_to_type(g, g.op("PRelu", self, weight), old_type)
    else:
        return g.op("PRelu", self, weight)


# 符号操作的注解，处理矩阵乘法操作 mm。
# 如果输入是标量，则创建一个常量张量 zero_constant。如果无法转换，则抛出错误。
# 返回经过类型转换后的 Gemm 操作。
@_onnx_symbolic("aten::mm")
def mm(g: jit_utils.GraphContext, self, other):
    scalar_type = symbolic_helper._try_get_scalar_type(self, other)
    if scalar_type is None:
        raise errors.SymbolicValueError(
            "mm can only operate on tensors with known types", self
        )
    zero_constant = g.op(
        "Constant",
        value_t=torch.tensor([0], dtype=scalar_type.dtype()),
    )

    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, other, zero_constant = _try_cast_integer_to_float(
            g, self, other, zero_constant
        )
        return _cast_to_type(
            g,
            g.op("Gemm", self, other, zero_constant, beta_f=0.0, alpha_f=1.0),
            old_type,
        )
    return g.op("Gemm", self, other, zero_constant, beta_f=0.0, alpha_f=1.0)


# 符号操作的注解，处理 addmm 操作。解析参数并执行相应的操作。
@_onnx_symbolic("aten::addmm")
@symbolic_helper.parse_args("v", "v", "v", "t", "t")
def addmm(g: jit_utils.GraphContext, self, mat1, mat2, beta, alpha):
    # 如果符号助手能够获取张量的标量类型
    if symbolic_helper._try_get_scalar_type(self):
        # 调用函数尝试将输入的整数转换为浮点数
        old_type, self, mat1, mat2 = _try_cast_integer_to_float(g, self, mat1, mat2)
        # 返回类型转换后的结果
        return _cast_to_type(
            g,
            # 创建一个矩阵乘法的操作节点，包括 mat1, mat2, self 作为输入，
            # 使用标量值 beta 和 alpha 作为参数
            g.op(
                "Gemm",
                mat1,
                mat2,
                self,
                beta_f=symbolic_helper._scalar(beta),
                alpha_f=symbolic_helper._scalar(alpha),
            ),
            old_type,
        )
    else:
        # 如果无法获取张量的标量类型，则直接创建矩阵乘法的操作节点
        return g.op(
            "Gemm",
            mat1,
            mat2,
            self,
            beta_f=symbolic_helper._scalar(beta),
            alpha_f=symbolic_helper._scalar(alpha),
        )
@_onnx_symbolic("aten::flatten")
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim):
    # 获取 start_dim 和 end_dim 的常量值
    start_dim_i = symbolic_helper._get_const(start_dim, "i", "start_dim")
    end_dim_i = symbolic_helper._get_const(end_dim, "i", "end_dim")

    # 获取输入张量的维度信息
    dim = input.type().dim()
    
    # 如果 end_dim_i 是负数，则转换为对应的非负数索引
    if end_dim_i < 0:
        end_dim_i = dim + end_dim_i

    # 对于输出形状为 2D 的情况，使用 ONNX 的 Flatten 操作符
    if start_dim_i == 1 and end_dim_i == dim - 1:
        if symbolic_helper._try_get_scalar_type(input):
            # 尝试将输入张量转换为浮点数类型
            old_type, input = _try_cast_integer_to_float(g, input)
            return _cast_to_type(
                g, g.op("Flatten", input, axis_i=start_dim_i), old_type
            )
        else:
            return g.op("Flatten", input, axis_i=start_dim_i)
    
    # 对于 start_dim_i 是 0，end_dim_i 是 dim-2 的情况，使用 Flatten 操作符
    if start_dim_i == 0 and end_dim_i == dim - 2:
        if symbolic_helper._try_get_scalar_type(input):
            # 尝试将输入张量转换为浮点数类型
            old_type, input = _try_cast_integer_to_float(g, input)
            return _cast_to_type(
                g, g.op("Flatten", input, axis_i=end_dim_i + 1), old_type
            )
        else:
            return g.op("Flatten", input, axis_i=end_dim_i + 1)

    # 如果以上条件都不满足，使用 opset9.flatten 函数处理
    return opset9.flatten(g, input, start_dim, end_dim)


def _constant_fill(g: jit_utils.GraphContext, sizes, dtype: int, const_value):
    # 确定标量类型，默认为 FLOAT
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType(dtype)
    
    # 如果标量类型不是浮点数类型，使用 ConstantFill 操作符生成常量张量并转换类型
    if not scalar_type.dtype().is_floating_point:
        result = g.op(
            "ConstantFill",
            sizes,
            dtype_i=_type_utils.JitScalarType.FLOAT.onnx_type(),
            input_as_shape_i=1,
            value_f=const_value,
        )
        return g.op("Cast", result, to_i=scalar_type.onnx_type())
    else:
        # 如果标量类型是浮点数类型，直接使用 ConstantFill 操作符生成常量张量
        return g.op(
            "ConstantFill",
            sizes,
            dtype_i=scalar_type.onnx_type(),
            input_as_shape_i=1,
            value_f=const_value,
        )


@_onnx_symbolic("aten::empty")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty(
    g: jit_utils.GraphContext,
    sizes,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    # 调用 zeros 函数生成全零张量，实现 empty 函数的功能
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::empty_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty_like(
    g: jit_utils.GraphContext,
    input,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    # 调用 zeros_like 函数生成与输入张量形状相同的全零张量，实现 empty_like 函数的功能
    return zeros_like(g, input, dtype, layout, device, pin_memory)


@_onnx_symbolic("aten::zeros")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def zeros(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=False):
    # 注意：在 ONNX 中无法设置设备和布局，因此在此函数中忽略 layout 和 device 参数
    # 调用 _constant_fill 函数生成全零常量张量
    return _constant_fill(g, sizes, dtype, 0)


@_onnx_symbolic("aten::zeros_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def zeros_like(
    g: jit_utils.GraphContext,
    input,
    dtype,
    layout,
    device,
    pin_memory=False,
):
    layout,  # Tensor的布局方式，例如"torch.strided"或"torch.sparse_coo"
    device,  # Tensor被分配的设备，例如"cpu"或"cuda:0"
    pin_memory=False,  # 是否将Tensor数据固定在内存中，通常在使用CUDA时设置为True以提高性能
    memory_format=None,  # 内存存储格式，例如"torch.contiguous_format"，用于控制Tensor的存储方式
# 定义名为 ones 的函数，用于处理 aten::ones 操作
@_onnx_symbolic("aten::ones")
@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def ones(g: jit_utils.GraphContext, sizes, dtype, layout, device, pin_memory=False):
    # 调用 _constant_fill 函数，创建值为 1 的常量填充
    return _constant_fill(g, sizes, dtype, 1)


# 定义名为 ones_like 的函数，用于处理 aten::ones_like 操作
@_onnx_symbolic("aten::ones_like")
@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def ones_like(
    g: jit_utils.GraphContext,
    input,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    # 获取输入张量的形状
    shape = g.op("Shape", input)
    # 调用 _constant_fill 函数，创建与输入形状相同的值为 1 的常量填充
    return _constant_fill(g, shape, dtype, 1)


# 定义名为 full 的函数，处理 aten::full 操作
@_onnx_symbolic("aten::full")
def full(
    g: jit_utils.GraphContext, sizes, value, dtype, layout, device, pin_memory=False
):
    # 获取常数值或常数值的表示
    const_value = symbolic_helper._maybe_get_const(value, "t")
    if symbolic_helper._is_value(const_value):
        # 如果 value 是一个值，创建一个值为 0 的张量
        tmp = zeros(g, sizes, dtype, layout, device)
        # 返回 tmp 与 value 相加的结果
        return opset9.add(g, tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        # 获取 dtype 的常数值或常数值的表示
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        # 调用 _constant_fill 函数，使用 const_value 创建常量填充
        return _constant_fill(g, sizes, dtype, const_value)


# 定义名为 full_like 的函数，处理 aten::full_like 操作
@_onnx_symbolic("aten::full_like")
@symbolic_helper.parse_args("v", "f", "i", "v", "v", "v", "v")
def full_like(
    g: jit_utils.GraphContext,
    input,
    fill_value,
    dtype,
    layout,
    device,
    pin_memory=False,
    memory_format=None,
):
    # 获取输入张量的形状
    shape = g.op("Shape", input)
    # 调用 _constant_fill 函数，创建与输入形状相同的常量填充
    return _constant_fill(g, shape, dtype, fill_value)


# 定义名为 repeat 的函数，处理 aten::repeat 操作
@_onnx_symbolic("aten::repeat")
def repeat(g: jit_utils.GraphContext, self, repeats):
    # 如果 repeats 不是值，则将其转换为常量张量
    if not symbolic_helper._is_value(repeats):
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
    # 如果 repeats 是打包列表，则计算其长度
    if symbolic_helper._is_packed_list(repeats):
        repeat_size_len = len(symbolic_helper._unpack_list(repeats))
    else:
        const_repeats = symbolic_helper._maybe_get_const(repeats, "is")
        repeat_size_len = len(const_repeats)
    # 如果 self 是完整的张量，则获取其大小
    if self.isCompleteTensor():
        sizes = self.type().sizes()
        diff_dims = repeat_size_len - len(sizes)
        # 如果维度差异大于零，则对 self 进行视图操作以匹配新的维度
        if diff_dims > 0:
            self = opset9.view(
                g, self, g.op("Constant", value_t=torch.tensor([1] * diff_dims + sizes))
            )
    # 返回 Tile 操作的结果，通过 repeats 进行重复操作
    return g.op("Tile", self, repeats)
```