# `.\pytorch\torch\onnx\symbolic_opset10.py`

```
# mypy: allow-untyped-defs
# 从未来模块导入注释类型，允许未注释的定义
from __future__ import annotations

# 导入 functools 库，用于创建偏函数
import functools
# 导入 sys 库，提供对 Python 解释器的访问
import sys
# 导入 warnings 库，用于处理警告
import warnings
# 导入类型相关的声明
from typing import List, Optional, Sequence, Tuple, Union

# 导入 PyTorch 主要模块
import torch
# 导入 torch._C._onnx 模块，提供对 ONNX 相关功能的访问
import torch._C._onnx as _C_onnx
# 导入 torch.onnx 模块，提供 ONNX 导出功能
import torch.onnx
# 导入 torch._C 模块，提供对 PyTorch C++ API 的访问
from torch import _C

# 在 Graph 上进行图形操作的猴子补丁，用于 ONNX 符号化操作
from torch.onnx import (
    _constants,
    _type_utils,
    errors,
    symbolic_helper,
    symbolic_opset9 as opset9,
)
# 导入全局变量
from torch.onnx._globals import GLOBALS
# 导入内部函数和装饰器
from torch.onnx._internal import _beartype, jit_utils, registration

# 编辑此文件？首先阅读此处内容！
# 请查阅 README.md 中的注释 [Edit Symbolic Files]

# 此文件导出 Opset 10 的 ONNX 操作
# Opset 10 受 ONNX 1.5.0 版本支持
# 发布日期为 2019 年 4 月 24 日

# 导出的函数列表
__all__ = [
    "dequantize",
    "div",
    "embedding_bag",
    "fake_quantize_per_tensor_affine",
    "flip",
    "fmod",
    "isfinite",
    "isinf",
    "nan_to_num",
    "quantize_per_tensor",
    "quantized_add_relu",
    "quantized_add",
    "quantized_cat",
    "quantized_conv1d_relu",
    "quantized_conv2d_relu",
    "quantized_conv3d_relu",
    "quantized_conv1d",
    "quantized_conv2d",
    "quantized_conv3d",
    "quantized_conv_transpose1d",
    "quantized_conv_transpose2d",
    "quantized_conv_transpose3d",
    "quantized_group_norm",
    "quantized_hardswish",
    "quantized_instance_norm",
    "quantized_layer_norm",
    "quantized_leaky_relu",
    "quantized_linear",
    "quantized_linear_relu",
    "quantized_mul",
    "quantized_sigmoid",
    "slice",
    "sort",
    "topk",
]

# functools.partial 函数用于注册 Opset 10 中的 ONNX 符号化函数
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=10)


# 符号化函数：aten::div
# 注释：根据参数个数分发到不同的操作函数
# g: 图形上下文对象
# self: 第一个操作数
# other: 第二个操作数
# *args: 可选的附加参数
@_onnx_symbolic("aten::div")
@_beartype.beartype
def div(g: jit_utils.GraphContext, self, other, *args):
    if len(args) == 0:
        return opset9.true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


# 内部函数：_div_rounding_mode
# 注释：根据给定的舍入模式选择除法实现
# g: 图形上下文对象
# self: 第一个操作数
# other: 第二个操作数
# rounding_mode: 舍入模式，可以是 "floor"
@symbolic_helper.parse_args("v", "v", "s")
@_beartype.beartype
def _div_rounding_mode(g: jit_utils.GraphContext, self, other, rounding_mode):
    if rounding_mode == "floor":
        return _floor_divide(g, self, other)
    else:
        return opset9._div_rounding_mode(g, self, other, rounding_mode)


# 符号化函数：aten::_floor_divide
# 注释：对浮点数执行真实除法后应用向下取整操作
# g: 图形上下文对象
# self: 第一个操作数
# other: 第二个操作数
@_onnx_symbolic("aten::_floor_divide")
@_beartype.beartype
def _floor_divide(g: jit_utils.GraphContext, self, other):
    if symbolic_helper._is_fp(self) or symbolic_helper._is_fp(other):
        out = opset9.true_divide(g, self, other)
        return g.op("Floor", out)
    else:
        # Integer division does trunction rounding
        # 使用 g.op 方法创建一个 Div 操作节点，对 self 和 other 进行整数除法运算
        div = g.op("Div", self, other)
        
        # Division is negative if: self < 0 != other < 0
        # 创建一个值为 0 的常量节点 zero，使用 g.op 方法生成 Constant 操作节点
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        # 创建两个 Less 操作节点，比较 self 和 other 是否小于零，然后使用 Xor 操作节点对结果进行异或运算
        negative = g.op("Xor", g.op("Less", self, zero), g.op("Less", other, zero))

        # For negative numbers with self % other != 0, subtract 1 to round down instead of up
        # 使用 g.op 方法创建 Mod 操作节点，计算 self 对 other 取模的结果，并禁用 fmod_i 选项
        mod = g.op("Mod", self, other, fmod_i=0)
        # 创建 Not、Equal、And 操作节点，生成一个掩码，用于修正负数情况下的整数除法结果
        fixup_mask = g.op("And", negative, g.op("Not", g.op("Equal", mod, zero)))

        # 创建一个值为 1 的常量节点 one，使用 g.op 方法生成 Constant 操作节点
        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        # 使用 g.op 方法创建 Sub 操作节点，将 div 减去 one，用于修正整数除法结果
        fixup = g.op("Sub", div, one)
        # 使用 g.op 方法创建 Where 操作节点，根据 fixup_mask 选择返回 fixup 或者 div
        return g.op("Where", fixup_mask, fixup, div)
# 注册一个ONNX符号为“aten::sort”的符号操作函数
# 使用装饰器将函数注册为符号化帮助函数，检查参数类型
@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
@_beartype.beartype
def sort(g: jit_utils.GraphContext, self, dim, decending, out=None):
    # 调用符号化帮助函数的排序辅助函数，返回结果
    return symbolic_helper._sort_helper(g, self, dim, decending=decending, out=out)


# 注册一个ONNX符号为“aten::topk”的符号操作函数
# 使用装饰器将函数注册为符号化帮助函数，检查参数类型
@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
@_beartype.beartype
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    # 调用符号化帮助函数的topk辅助函数，返回结果
    return symbolic_helper._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out
    )


# ONNX的最大池化操作函数，对输入进行最大池化处理
def _aten_max_pool_onnx(
    g: jit_utils.GraphContext,
    self: _C.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
) -> _C.Value:
    # 获取输入张量的维度
    self_rank = g.op("Size", g.op("Shape", self))
    # 如果输入张量的维度与指定的未批处理维度相同，则添加维度信息（用于处理C,H,W -> N,C,H,W and N=1的情况）
    if self_rank == unbatched_rank:
        self = g.op(
            "Unsqueeze",
            self,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    # 执行最大池化操作
    pool_result, _ = g.op(
        "MaxPool",
        self,
        outputs=2,
        ceil_mode_i=ceil_mode,
        dilations_i=dilations,
        kernel_shape_i=kernel_shape,
        pads_i=pads,
        strides_i=strides,
    )

    # 如果添加了维度信息，则恢复原始张量的形状
    if self_rank == unbatched_rank:
        pool_result = g.op(
            "Squeeze",
            pool_result,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    # 返回池化结果
    return pool_result


# 调整最大池化操作的属性以符合ONNX规范
def _adjust_attributes_of_max_pool(
    expand_size: int,
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
    dilation: Union[Sequence[int], int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    # 根据输入参数调整膨胀参数为列表形式
    if isinstance(dilation, int):
        dilation = [dilation] * expand_size

    # 根据输入参数调整核大小为列表形式
    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size  # type: ignore[assignment]

    # 根据输入参数调整填充为列表形式
    if isinstance(padding, int):
        pads = [padding] * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 1:
        pads = padding * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 2:
        # 2D填充
        pads = padding * 2  # type: ignore[operator, assignment]
    elif len(padding) == 3:
        # 3D填充
        pads = padding * 2  # type: ignore[operator, assignment]
    else:
        # 当所有维度的填充已完成时，无需再次调整
        # 例如：(1, 1, 1, 1, 1, 1)
        pads = padding  # type: ignore[assignment]

    # 根据输入参数调整步长为列表形式
    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride  # type: ignore[assignment]
    # 返回包含内核形状、步幅、填充和扩展信息的元组
    return (kernel_shape, strides, pads, dilation)
def _aten_max_pool_with_indices_onnx(
    g: jit_utils.GraphContext,
    self: _C.Value,
    kernel_shape: Sequence[int],
    strides: Sequence[int],
    pads: Sequence[int],
    dilations: Sequence[int],
    ceil_mode: bool,
    unbatched_rank: int,
    n_dims_one: Sequence[int],
    n_dims_zero: Sequence[int],
    n_dims_axes: Sequence[int],
) -> Tuple[_C.Value, Sequence[int]]:
    # 计算输入张量的维度
    self_rank = g.op("Size", g.op("Shape", self))

    # 如果输入张量的维度与 unbatched_rank 相同，插入一个维度到第一个位置（N=1）
    if self_rank == unbatched_rank:  # C,H,W -> N,C,H,W and N=1
        self = g.op(
            "Unsqueeze",
            self,
            g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64)),
        )

    # 进行最大池化操作，同时返回池化结果和索引
    pool_result, indices = g.op(
        "MaxPool",
        self,
        outputs=2,
        ceil_mode_i=ceil_mode,
        dilations_i=dilations,
        kernel_shape_i=kernel_shape,
        pads_i=pads,
        strides_i=strides,
    )

    # 执行额外的最大池化操作以获取展平后的索引
    _, flatten_indices = g.op(
        "MaxPool",
        self,
        outputs=2,
        dilations_i=dilations,
        kernel_shape_i=n_dims_one,
        strides_i=n_dims_one,
    )

    # 创建用于切片操作的常量张量
    ends = g.op("Constant", value_t=torch.tensor(n_dims_one))
    starts = g.op("Constant", value_t=torch.tensor(n_dims_zero))
    axes = g.op("Constant", value_t=torch.tensor(n_dims_axes))

    # 使用切片操作获取池化索引的差分
    delta = g.op("Slice", flatten_indices, starts, ends, axes)
    indices = g.op("Sub", indices, delta)

    # 如果输入张量的维度与 unbatched_rank 相同，则去除插入的维度
    if self_rank == unbatched_rank:
        pool_result = g.op(
            "Squeeze", pool_result, value_t=torch.tensor([0], dtype=torch.int64)
        )
        indices = g.op("Squeeze", indices, value_t=torch.tensor([0], dtype=torch.int64))

    # 返回池化结果和最终的索引
    return (pool_result, indices)


# 下面是一系列装饰器，用于将不同的池化操作映射到对应的 ONNX 符号函数
@_onnx_symbolic(
    "aten::max_pool1d",
    decorate=[symbolic_helper._apply_params("max_pool1d", 1, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool2d",
    decorate=[symbolic_helper._apply_params("max_pool2d", 2, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool3d",
    decorate=[symbolic_helper._apply_params("max_pool3d", 3, return_indices=False)],
)
@_onnx_symbolic(
    "aten::max_pool1d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool1d_with_indices",
            1,
            return_indices=True,
        )
    ],
)
@_onnx_symbolic(
    "aten::max_pool2d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool2d_with_indices",
            2,
            return_indices=True,
        )
    ],
)
@_onnx_symbolic(
    "aten::max_pool3d_with_indices",
    decorate=[
        symbolic_helper._apply_params(
            "max_pool3d_with_indices",
            3,
            return_indices=True,
        )
    ],
)
@_beartype.beartype
def _max_pool(name: str, expand_size: int, return_indices: bool):
    # 定义一个具有装饰器的函数，用于执行不同的池化操作
    @symbolic_helper.quantized_args(True, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "is", "i")
    def symbolic_fn(
        g: jit_utils.GraphContext,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: Union[int, Sequence[int]],
        dilation: Sequence[int],
        ceil_mode: bool,
    ):
        # 调整最大池化操作的参数，确保参数格式正确
        kernel_shape, strides, pads, dilations = _adjust_attributes_of_max_pool(
            expand_size, kernel_size, stride, padding, dilation
        )
    
        if return_indices:
            # 在ONNX中执行带索引的最大池化操作
            return _aten_max_pool_with_indices_onnx(
                g,
                input,
                kernel_shape,
                strides,
                pads,
                dilations,
                ceil_mode,
                expand_size + 1,
                ([1] * expand_size),      # 扩展尺寸加1后的维度信息
                ([0] * expand_size),      # 所有扩展维度的填充值为0
                ([2 + i for i in range(expand_size)]),  # 扩展维度的索引从2开始递增
            )
        else:
            # 在ONNX中执行常规最大池化操作
            return _aten_max_pool_onnx(
                g,
                input,
                kernel_shape,
                strides,
                pads,
                dilations,
                ceil_mode,
                expand_size + 1,
            )
    
    return symbolic_fn
# 对于 AvgPool
def _adjust_attributes_of_avg_pool(
    expand_size: int,
    kernel_size: Union[Sequence[int], int],
    stride: Union[Sequence[int], int],
    padding: Union[Sequence[int], int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Adjust attributes of avg_pool to match ONNX specification."""

    # 根据 kernel_size 的类型确定 kernel_shape
    if isinstance(kernel_size, int):
        kernel_shape = [kernel_size] * expand_size
    else:
        kernel_shape = kernel_size  # type: ignore[assignment]

    # 根据 padding 的类型确定 pads
    if isinstance(padding, int):
        pads = [padding] * expand_size * 2
    elif len(padding) == 1:
        pads = padding * expand_size * 2  # type: ignore[operator, assignment]
    elif len(padding) == 2:
        pads = padding * expand_size  # type: ignore[operator, assignment]
    else:
        pads = padding * 2  # type: ignore[operator, assignment]

    # 根据 stride 的类型确定 strides
    if isinstance(stride, int):
        strides = [stride] * expand_size
    elif not stride:
        strides = kernel_shape
    else:
        strides = stride  # type: ignore[assignment]

    return (kernel_shape, strides, pads)


# 装饰符，用于将函数映射到对应的 ONNX 符号操作
@_onnx_symbolic(
    "aten::avg_pool1d",
    decorate=[symbolic_helper._apply_params("avg_pool1d", 1)],
)
@_onnx_symbolic(
    "aten::avg_pool2d",
    decorate=[symbolic_helper._apply_params("avg_pool2d", 2)],
)
@_onnx_symbolic(
    "aten::avg_pool3d",
    decorate=[symbolic_helper._apply_params("avg_pool3d", 3)],
)
# 使用 beartype 装饰符，用于类型检查
@_beartype.beartype
# AvgPool 的符号函数定义
def _avg_pool(name, expand_size):
    # 装饰符，用于量化参数
    @symbolic_helper.quantized_args(True, False, False, False, False, False, False)
    # 解析参数装饰符
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    # 使用 beartype 装饰符，用于类型检查
    @_beartype.beartype
    # 符号函数定义
    def symbolic_fn(
        g,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: Union[int, Sequence[int]],
        ceil_mode: int,
        count_include_pad: int,
        divisor_override=None,
    ):
        # 调整 AvgPool 的属性以符合 ONNX 规范
        kernel_shape, strides, pads = _adjust_attributes_of_avg_pool(
            expand_size, kernel_size, stride, padding
        )

        # 使用 ONNX 操作创建 AveragePool 操作节点
        result = g.op(
            "AveragePool",
            input,
            ceil_mode_i=ceil_mode,
            count_include_pad_i=count_include_pad,
            kernel_shape_i=kernel_shape,
            pads_i=pads,
            strides_i=strides,
        )

        return result

    return symbolic_fn


# 装饰符，用于将函数映射到对应的 ONNX 符号操作
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
# 装饰符，用于类型检查
@_beartype.beartype
# Upsample 的符号函数定义
def _upsample(name, expand_size):
    # 符号函数定义
    def symbolic_fn(
        g,
        input: _C.Value,
        output_size: Optional[Sequence[int]],
        scales: Optional[Sequence[float]],
    ):
        # 创建 Upsample 操作节点
        result = g.op(
            name,
            input,
            scales=scales,
            roi=roi,
        )

        return result

    return symbolic_fn
# 为函数 _interpolate 添加装饰器 @_onnx_symbolic，指定了特定的 ONNX 符号化函数
# 该装饰器定义了对 "aten::upsample_trilinear3d" 操作的符号化处理，并使用 decorate 参数应用特定的符号化帮助函数
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear")],
)
# 使用 @_beartype 装饰器对 _interpolate 函数进行类型检查和类型注解
@_beartype.beartype
def _interpolate(name, dim, interpolate_mode):
    # 定义一个嵌套函数 symbolic_fn，用于 ONNX 符号化操作
    @symbolic_helper.quantized_args(True, False, False)
    @_beartype.beartype
    def symbolic_fn(g, input, output_size, *args):
        # 调用符号化帮助函数 _get_interpolate_attributes 获取插值操作的缩放比例和对齐方式
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        # 发出关于插值操作的警告信息
        symbolic_helper._interpolate_warning(interpolate_mode)
        # 尝试从 align_corners 中获取标量值
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        # 如果 align_corners 为真，则使用 _unimplemented 函数返回未实现的警告信息
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True", input)
        # 如果 scales 为空，则使用 _interpolate_size_to_scales 函数将输出大小插值为缩放比例
        if scales is None:
            scales = symbolic_helper._interpolate_size_to_scales(
                g, input, output_size, dim
            )
        # 使用 g.op 进行 Resize 操作，返回调整大小后的输入
        return g.op("Resize", input, scales, mode_s=interpolate_mode)

    return symbolic_fn


# 为函数 __interpolate 添加装饰器 @_onnx_symbolic，指定了特定的 ONNX 符号化函数
# 该装饰器定义了对 "aten::__interpolate" 操作的符号化处理
@_onnx_symbolic("aten::__interpolate")
# 使用 @_beartype 装饰器对 __interpolate 函数进行类型检查和类型注解
@_beartype.beartype
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
    # 调用符号化帮助函数 _interpolate_get_scales_and_mode 获取缩放比例和插值模式
    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    # 使用 g.op 进行 Resize 操作，返回调整大小后的输入
    return g.op("Resize", input, scales, mode_s=mode)


# 定义函数 _slice，用于切片操作的符号化处理
# 使用 @_beartype 装饰器对 _slice 函数进行类型检查和类型注解
@_beartype.beartype
def _slice(
    g: jit_utils.GraphContext,
    input: torch._C.Value,
    axes: Union[List, torch.Tensor, torch._C.Value],
    starts: Union[List, torch.Tensor, torch._C.Value],
    ends: Union[List, torch.Tensor, torch._C.Value],
    steps: Optional[Union[List, torch.Tensor, torch._C.Value]] = None,
):
    # 定义内部函数 is_none_value，用于检查值是否为 None 或者 Torch 常量类型
    def is_none_value(value):
        if value is None:
            return True
        return (
            isinstance(value, torch._C.Value)
            and value.node().kind() == "prim::Constant"
            and isinstance(value.type(), _C.NoneType)
        )

    # 定义内部函数 to_slice_input，用于将输入参数转换为 1D 的 Torch 值
    def to_slice_input(list_or_value, default_value=None):
        # 如果 list_or_value 为 None 并且提供了默认值，则将其转换为包含默认值的列表
        if is_none_value(list_or_value) and default_value is not None:
            list_or_value = [default_value]

        # 如果 list_or_value 是列表或者 Torch 张量，则使用 Constant 操作转换为常量
        if isinstance(list_or_value, (list, torch.Tensor)):
            return g.op("Constant", value_t=torch.tensor(list_or_value))

        # 使用 symbolic_helper 中的 _unsqueeze_helper 函数将 0 维度的张量扩展为 1 维
        rank = symbolic_helper._get_tensor_rank(list_or_value)
        if rank == 0:
            return symbolic_helper._unsqueeze_helper(g, list_or_value, [0])
        # 如果张量的维度为 1，则直接返回
        if rank == 1:
            return list_or_value
        # 如果张量维度不为 0 或 1，则引发 SymbolicValueError 异常
        raise errors.SymbolicValueError(
            f"Rank must be 0 or 1, not {rank}", list_or_value
        )

    # 定义内部函数 get_const_value，用于获取常量值
    def get_const_value(list_or_value):
        if isinstance(list_or_value, (list, torch.Tensor)):
            if len(list_or_value) == 1:
                return list_or_value[0]
            return None
        # 使用 symbolic_helper 中的 _maybe_get_const 函数尝试获取常量值
        return symbolic_helper._maybe_get_const(list_or_value, "i")

    # 检查切片操作是否是无操作
    # 检查是否为简单切片情况：开始值为0，结束值为INT64_MAX，步长为1或未指定步长
    if (
        get_const_value(starts) == 0
        and get_const_value(ends) == _constants.INT64_MAX
        and (steps is None or get_const_value(steps) == 1)
    ):
        # 若是简单切片情况直接返回输入张量
        return input
    
    # 将 axes 转换为切片输入格式
    axes = to_slice_input(axes)
    # 将 starts 转换为切片输入格式，如果未指定默认为0
    starts = to_slice_input(starts, default_value=0)
    # 将 ends 转换为切片输入格式，如果未指定默认为INT64_MAX
    ends = to_slice_input(ends, default_value=_constants.INT64_MAX)
    
    # 如果 steps 未指定，则使用 g.op 方法执行切片操作
    if steps is None:
        return g.op("Slice", input, starts, ends, axes)
    
    # 将 steps 转换为切片输入格式，如果未指定默认为1
    steps = to_slice_input(steps, default_value=1)
    # 使用 g.op 方法执行带步长的切片操作
    return g.op("Slice", input, starts, ends, axes, steps)
# 使用装饰器将函数注册为ONNX符号化函数，处理"aten::slice"操作
# 包装成类型检查函数
@_onnx_symbolic("aten::slice")
@_beartype.beartype
def slice(g: jit_utils.GraphContext, self, *args):
    # 如果参数个数为4，按照特定签名解析参数
    if len(args) == 4:
        # aten::slice(Tensor self, int dim, int? start=None, int? end=None, int step=1) -> Tensor
        dims, start, end, step = args
    # 如果参数个数为3，按照特定签名解析参数
    elif len(args) == 3:
        # aten::slice(t[] l, int? start=None, int? end=None, int step=1) -> t[]
        start, end, step = args
        dims = [0]
    else:
        # 抛出错误，表示未知的aten::slice签名
        raise errors.SymbolicValueError("Unknown aten::slice signature", self)

    # 调用辅助函数处理切片操作
    return symbolic_helper._slice_helper(
        g,
        self,
        axes=dims,
        starts=start,
        ends=end,
        steps=step,
    )


# 使用装饰器将函数注册为ONNX符号化函数，处理"aten::flip"操作
# 解析参数的装饰器
@_onnx_symbolic("aten::flip")
@symbolic_helper.parse_args("v", "is")
@_beartype.beartype
def flip(g: jit_utils.GraphContext, input, dims):
    # 调用辅助函数处理反转操作
    return symbolic_helper._slice_helper(
        g,
        input,
        axes=dims,
        starts=[-1] * len(dims),
        ends=[-_constants.INT64_MAX] * len(dims),
        steps=[-1] * len(dims),
    )


# 使用装饰器将函数注册为ONNX符号化函数，处理"aten::fmod"操作
@_onnx_symbolic("aten::fmod")
@_beartype.beartype
def fmod(g: jit_utils.GraphContext, input, other):
    # 使用ONNX操作符"Mod"处理fmod操作
    return g.op("Mod", input, other, fmod_i=1)


# 使用装饰器将函数注册为ONNX符号化函数，处理"aten::embedding_bag"操作
# 解析参数的装饰器
@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
@_beartype.beartype
def embedding_bag(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    # 如果需要按频率缩放梯度并且处于导出训练模式，返回不支持的提示信息
    if scale_grad_by_freq and GLOBALS.export_training:
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with scale_grad_by_freq for training mode"
        )
    # 如果padding_idx不为None且大于等于0，抛出运行时错误
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")

    # 发出警告，提示在opset 10中不支持动态输入/偏移形状的导出
    warnings.warn(
        "Export of embedding_bag with dynamic input/offsets shape is not supported in opset 10. "
        "Please use opset 11 or higher to export model for dynamic input shape.'"
    )
    # 获取offsets的第一个维度大小
    offsets_dim_0 = symbolic_helper._get_tensor_dim_size(offsets, 0)
    # 如果 offsets_dim_0 不为 None，则执行以下操作
    if offsets_dim_0 is not None:
        # 如果 include_last_offset 为 True，则使用 offsets_dim_0 - 1 作为 offset_len
        if include_last_offset:
            offset_len = offsets_dim_0 - 1
            offsets_extended = offsets
        # 如果 include_last_offset 不为 True，则使用 offsets_dim_0 作为 offset_len，并扩展 offsets_extended
        else:
            offset_len = offsets_dim_0
            # 构建包含最大值的 Constant 节点
            offsets_extended = [
                offsets,
                g.op("Constant", value_t=torch.tensor([sys.maxsize])),
            ]
            # 使用 Concat 操作将 offsets 和常量值连接起来
            offsets_extended = g.op("Concat", *offsets_extended, axis_i=0)
        
        # 初始化空列表 list_
        list_ = []
        # 迭代 offset_len 次
        for i in range(offset_len):
            # 使用 _unsqueeze_helper 函数处理起始索引
            start_ = symbolic_helper._unsqueeze_helper(
                g,
                opset9.select(g, offsets_extended, torch.tensor(0), torch.tensor(i)),
                [0],
            )
            # 使用 _unsqueeze_helper 函数处理结束索引
            end_ = symbolic_helper._unsqueeze_helper(
                g,
                opset9.select(
                    g, offsets_extended, torch.tensor(0), torch.tensor(i + 1)
                ),
                [0],
            )
            # 创建 Constant 节点表示轴
            axes_ = g.op("Constant", value_t=torch.tensor([0]))
            # 使用 Slice 操作获取 indices_row
            indices_row = g.op("Slice", indices, start_, end_, axes_)

            # 使用 Gather 操作从 embedding_matrix 中获取 embeddings
            embeddings = g.op("Gather", embedding_matrix, indices_row)
            
            # 如果 per_sample_weights 不为空，则处理 per_sample_weights_row
            if not symbolic_helper._is_none(per_sample_weights):
                per_sample_weights_row = g.op(
                    "Slice", per_sample_weights, start_, end_, axes_
                )
                per_sample_weights_row = symbolic_helper._unsqueeze_helper(
                    g, per_sample_weights_row, [1]
                )
                # 使用 Mul 操作将 embeddings 与 per_sample_weights_row 相乘
                embeddings = g.op("Mul", embeddings, per_sample_weights_row)
            
            # 根据 mode 的值选择不同的聚合操作
            if mode == 0:
                embeddings = symbolic_helper._reducesum_helper(
                    g, embeddings, axes_i=[0], keepdims_i=0
                )
            elif mode == 1:
                embeddings = g.op("ReduceMean", embeddings, axes_i=[0], keepdims_i=0)
            else:
                embeddings = g.op("ReduceMax", embeddings, axes_i=[0], keepdims_i=0)

            # 使用 _unsqueeze_helper 函数处理 embeddings
            embeddings = symbolic_helper._unsqueeze_helper(g, embeddings, [0])
            # 将处理后的 embeddings 添加到 list_ 中
            list_.append(embeddings)

        # 使用 Concat 操作将 list_ 中的所有 embeddings 连接起来
        output = g.op("Concat", *list_, axis_i=0)
        
        # 返回 embedding_bag 操作的输出以及三个未使用的占位符
        # 这些占位符在 torch.nn.EmbeddingBag 或 torch.nn.functional.embedding_bag 中未被使用
        return output, None, None, None
    
    # 如果 offsets_dim_0 为 None，则返回一个不支持的错误信息
    else:
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with unknown shape of offsets for opset 10 is not supported. "
            "please use opset 11 or higher."
        )
# 使用装饰器定义符号化函数，处理 ONNX 符号化与参数解析
@_onnx_symbolic("aten::fake_quantize_per_tensor_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i")
@_beartype.beartype
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,            # 输入参数列表
    scale,             # 缩放因子
    zero_point,        # 零点
    quant_min=-128,    # 最小量化值，默认为 -128
    quant_max=127,     # 最大量化值，默认为 127
):
    # 检查特殊情况：如果量化范围是 (0, 127)，则抛出不支持的异常
    if (quant_min, quant_max) == (0, 127):
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Quantize range (0, 127) not supported, requires opset 13 Clip",
            inputs,
        )
    # 检查量化范围是否在允许的范围内：(0, 255) 或 (-128, 127)，否则抛出异常
    if (quant_min, quant_max) not in [(0, 255), (-128, 127)]:
        raise errors.SymbolicValueError(
            f"For (quant_min, quant_max), ONNX allows only (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    # 尝试获取 scale 的标量值，如果不是常数则抛出不支持的异常
    scale = symbolic_helper._maybe_get_scalar(scale)
    if scale is None:
        symbolic_helper._onnx_opset_unsupported_detailed(
            "fake_quantize_per_tensor_affine",
            10,
            13,
            "Non-constant scale not supported",
            inputs,
        )
    # 将 scale 转换为 float 类型的数据，避免导出时生成 double 类型
    scale = scale.float().data
    # 根据 quant_min 的值选择合适的数据类型转换 zero_point
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    # 返回量化与反量化的操作节点
    return g.op(
        "DequantizeLinear",
        g.op("QuantizeLinear", inputs, scale, zero_point),
        scale,
        zero_point,
    )


# 使用装饰器定义符号化函数，处理 ONNX 符号化
@_onnx_symbolic("aten::isinf")
@_beartype.beartype
def isinf(g: jit_utils.GraphContext, input):
    # 返回判断输入是否为无穷大的节点
    return g.op("IsInf", g.op("Cast", input, to_i=_C_onnx.TensorProtoDataType.DOUBLE))


# 使用装饰器定义符号化函数，处理 ONNX 符号化
@_onnx_symbolic("aten::isfinite")
@_beartype.beartype
def isfinite(g: jit_utils.GraphContext, input):
    # 获取输入是否为无穷大和是否为 NaN 的节点
    inf_node = isinf(g, input)
    nan_node = opset9.isnan(g, input)
    # 返回输入既不是无穷大也不是 NaN 的节点
    return opset9.__not_(g, opset9.__or_(g, inf_node, nan_node))


# 使用装饰器定义符号化函数，处理 ONNX 符号化与参数解析
@_onnx_symbolic("aten::quantize_per_tensor")
@_beartype.beartype
def quantize_per_tensor(g: jit_utils.GraphContext, input, scale, zero_point, dtype):
    # 获取 dtype 的常量值，并将 zero_point 和 scale 转换为对应的 ONNX 数据类型
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    zero_point = g.op(
        "Cast", zero_point, to_i=_type_utils.JitScalarType(dtype).onnx_type()
    )
    scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    # 调用辅助函数进行量化操作
    return symbolic_helper.quantize_helper(g, input, scale, zero_point)


# 使用装饰器定义符号化函数，处理 ONNX 符号化
@_onnx_symbolic("aten::dequantize")
@_beartype.beartype
def dequantize(g: jit_utils.GraphContext, input):
    # 调用辅助函数进行反量化操作，并返回结果
    return symbolic_helper.dequantize_helper(g, input)[0]


# 使用装饰器定义符号化函数，处理 ONNX 符号化与参数解析
@_onnx_symbolic("aten::nan_to_num")
@symbolic_helper.parse_args("v", "f", "f", "f")
# 使用 @_beartype.beartype 装饰器，确保函数参数类型的正确性
@_beartype.beartype
# 定义函数 nan_to_num，接受 GraphContext 对象 g 和输入参数 input，nan, posinf, neginf
def nan_to_num(g: jit_utils.GraphContext, input, nan, posinf, neginf):
    # 如果 input 不是浮点数类型，直接返回原始输入张量
    if not symbolic_helper._is_fp(input):
        return input
    # 获取 input 张量的数据类型
    input_dtype = _type_utils.JitScalarType.from_value(input).dtype()
    # 如果 nan 为 None，则设置为 0.0
    if nan is None:
        nan = 0.0
    # 创建一个条件，判断 input 中的 NaN 值
    nan_cond = opset9.isnan(g, input)
    # 如果遇到 NaN 值，则使用 nan 替换
    nan_result = g.op(
        "Where",
        nan_cond,
        g.op("Constant", value_t=torch.tensor([nan], dtype=input_dtype)),
        input,
    )

    # 获取 input 数据类型的有限值范围信息
    finfo = torch.finfo(input_dtype)
    # 如果 posinf 为 None，则使用 input 数据类型的最大有限值
    if posinf is None:
        posinf = finfo.max
    # 创建条件，判断 input 是否为正无穷大
    posinf_cond = opset9.logical_and(
        g,
        isinf(g, nan_result),
        opset9.gt(g, nan_result, g.op("Constant", value_t=torch.LongTensor([0]))),
    )
    # 如果遇到正无穷大值，则使用 posinf 替换
    nan_posinf_result = g.op(
        "Where",
        posinf_cond,
        g.op("Constant", value_t=torch.tensor([posinf], dtype=input_dtype)),
        nan_result,
    )

    # 如果 neginf 为 None，则使用 input 数据类型的最小有限值
    if neginf is None:
        neginf = finfo.min
    # 创建条件，判断 input 是否为负无穷大
    neginf_cond = opset9.logical_and(
        g,
        isinf(g, nan_posinf_result),
        opset9.lt(
            g, nan_posinf_result, g.op("Constant", value_t=torch.LongTensor([0]))
        ),
    )
    # 如果遇到负无穷大值，则使用 neginf 替换
    return g.op(
        "Where",
        neginf_cond,
        g.op("Constant", value_t=torch.tensor([neginf], dtype=input_dtype)),
        nan_posinf_result,
    )


# Quantized symbolics ---------------------------------------------------------
# https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter#quantized-model-export
# 由于 opset 版本 10 引入了 `DequantizeLinear` 和 `QuantizeLinear`，因此量化模型支持从 opset 10 开始
@_onnx_symbolic("quantized::linear")
# 使用 @_beartype.beartype 装饰器，确保函数参数类型的正确性
@_beartype.beartype
# 定义函数 quantized_linear，接受 GraphContext 对象 g 和输入参数 q_input, q_weight, bias, op_scale, op_zero_point
def quantized_linear(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    # 使用辅助函数 dequantize_helper 解除量化输入 q_input
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 使用辅助函数 dequantize_helper 解除量化权重 q_weight
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    # 使用辅助函数 requantize_bias_helper 重新量化偏置 bias
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    # 使用辅助函数 dequantize_helper 解除量化偏置 q_bias
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 使用 opset9 中的线性操作函数 linear 计算输出
    output = opset9.linear(g, input, weight, bias)

    # 使用辅助函数 quantize_helper 量化输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


@_onnx_symbolic("quantized::linear_relu")
# 使用 @_beartype.beartype 装饰器，确保函数参数类型的正确性
@_beartype.beartype
# 定义函数 quantized_linear_relu，接受 GraphContext 对象 g 和输入参数 q_input, q_weight, bias, op_scale, op_zero_point
def quantized_linear_relu(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    # 使用辅助函数 dequantize_helper 解除量化输入 q_input
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 使用辅助函数 dequantize_helper 解除量化权重 q_weight
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    # 使用辅助函数 requantize_bias_helper 重新量化偏置 bias
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    # 使用辅助函数 dequantize_helper 解除量化偏置 q_bias
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 使用 opset9 中的线性操作函数 linear 计算输出
    output = opset9.linear(g, input, weight, bias)
    # 使用 opset9 中的 ReLU 激活函数操作 relu 处理输出
    output = opset9.relu(g, output)
    # 调用符号辅助器的量化辅助函数，用于量化操作
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
# 用于处理 ONNX 符号化函数 "quantized::add" 的装饰器
@_onnx_symbolic("quantized::add")
# 应用 Beartype 装饰器，用于类型检查和注解
@_beartype.beartype
# 定义 quantized_add 函数，接收图上下文 g，输入 x 和 y，以及操作的缩放因子和零点
def quantized_add(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    # 对输入 x 和 y 进行量化反过程，获取量化的值，忽略后续返回的元组项
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    # 调用 opset9.add 执行加法操作，生成输出张量 output
    output = opset9.add(g, x, y)

    # 将输出张量 output 进行量化过程，返回量化后的结果
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 用于处理 ONNX 符号化函数 "quantized::add_relu" 的装饰器
@_onnx_symbolic("quantized::add_relu")
# 应用 Beartype 装饰器，用于类型检查和注解
@_beartype.beartype
# 定义 quantized_add_relu 函数，接收图上下文 g，输入 x 和 y，以及操作的缩放因子和零点
def quantized_add_relu(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    # 对输入 x 和 y 进行量化反过程，获取量化的值，忽略后续返回的元组项
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    # 调用 opset9.add 执行加法操作，生成输出张量 output
    output = opset9.add(g, x, y)
    # 调用 opset9.relu 执行 ReLU 激活函数操作，生成输出张量 output
    output = opset9.relu(g, output)

    # 将输出张量 output 进行量化过程，返回量化后的结果
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 用于处理 ONNX 符号化函数 "quantized::mul" 的装饰器
@_onnx_symbolic("quantized::mul")
# 应用 Beartype 装饰器，用于类型检查和注解
@_beartype.beartype
# 定义 quantized_mul 函数，接收图上下文 g，输入 x 和 y，以及操作的缩放因子和零点
def quantized_mul(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    # 对输入 x 和 y 进行量化反过程，获取量化的值，忽略后续返回的元组项
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)

    # 调用 opset9.mul 执行乘法操作，生成输出张量 output
    output = opset9.mul(g, x, y)

    # 将输出张量 output 进行量化过程，返回量化后的结果
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 用于处理 ONNX 符号化函数 "quantized::hardswish" 的装饰器
@_onnx_symbolic("quantized::hardswish")
# 应用 Beartype 装饰器，用于类型检查和注解
@_beartype.beartype
# 定义 quantized_hardswish 函数，接收图上下文 g 和输入 x，以及操作的缩放因子和零点
def quantized_hardswish(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    # 对输入 x 进行量化反过程，获取量化的值，忽略后续返回的元组项
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    # 调用 opset9.hardswish 执行 HardSwish 激活函数操作，生成输出张量 output
    output = opset9.hardswish(g, x)

    # 将输出张量 output 进行量化过程，返回量化后的结果
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 用于处理 ONNX 符号化函数 "quantized::sigmoid" 的装饰器
@_onnx_symbolic("quantized::sigmoid")
# 应用 Beartype 装饰器，用于类型检查和注解
@_beartype.beartype
# 定义 quantized_sigmoid 函数，接收图上下文 g 和输入 x，以及操作的缩放因子和零点
def quantized_sigmoid(g: jit_utils.GraphContext, x, op_scale, op_zero_point):
    # 对输入 x 进行量化反过程，获取量化的值，忽略后续返回的元组项
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    # 调用 opset9.sigmoid 执行 Sigmoid 激活函数操作，生成输出张量 output
    output = opset9.sigmoid(g, x)

    # 将输出张量 output 进行量化过程，返回量化后的结果
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 用于处理 ONNX 符号化函数 "quantized::leaky_relu" 的装饰器
@_onnx_symbolic("quantized::leaky_relu")
# 应用 Beartype 装饰器，用于类型检查和注解
@_beartype.beartype
# 定义 quantized_leaky_relu 函数，接收图上下文 g、输入 x、负斜率、是否原地操作标志，以及操作的缩放因子和零点
def quantized_leaky_relu(
    g: jit_utils.GraphContext, x, negative_slope, inplace, op_scale, op_zero_point
):
    # 对输入 x 进行量化反过程，获取量化的值，忽略后续返回的元组项
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    # 调用 opset9.leaky_relu 执行 Leaky ReLU 激活函数操作，生成输出张量 output
    output = opset9.leaky_relu(g, x, negative_slope, inplace)

    # 将输出张量 output 进行量化过程，返回量化后的结果
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 用于处理 ONNX 符号化函数 "quantized::layer_norm" 的装饰器
@_onnx_symbolic("quantized::layer_norm")
# 应用 Beartype 装饰器，用于类型检查和注解
@_beartype.beartype
# 定义 quantized_layer_norm 函数，接收图上下文 g、输入 x、规范化形状、权重、偏置、epsilon 值，以及操作的缩放因子和零点
def quantized_layer_norm(
    g: jit_utils.GraphContext,
    x,
    normalized_shape,
    weight,
    bias,
    eps,
    op_scale,
    op_zero_point,
):
    # 对输入 x 进行量化反过程，获取量化的值，忽略后续返回的元组项
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

    # 调用 opset9.layer_norm 执行 Layer Normalization 操作，生成输出张量 output
    output = opset9.layer_norm(g, x, normalized_shape, weight, bias, eps, False)

    # 将输出张量 output 进行量化过程，返回量化后的结果
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 用于处理 ONNX 符号化函数 "quantized::group_norm" 的装饰器
@_onnx_symbolic("quantized::group_norm")
# 应用 Beartype 装饰器
# 使用装饰器为函数添加 ONNX 符号，指定为量化卷积层的实例标识
# 使用装饰器解析函数参数，确保参数类型正确，这里参数包括输入数据、权重、偏置、扩展参数等
# 使用装饰器对函数应用 Beartype 检查，确保输入参数类型正确
def quantized_instance_norm(
    g: jit_utils.GraphContext,   # 参数 g 表示图的上下文，用于构建符号化图操作
    q_input,                     # 量化输入数据
    weight,                      # 权重参数
    bias,                        # 偏置参数
    eps,                         # epsilon 参数，用于数值稳定性
    op_scale,                    # 操作的缩放因子
    op_zero_point,               # 操作的零点偏移量
):
    input, _, _, _ = symbolic_helper.dequantize_helper(g, q_input)   # 解量化输入数据

    # 调用 ONNX opset9 中的 instance_norm 函数执行实例归一化操作
    output = opset9.instance_norm(
        g, input, weight, bias, None, None, False, 0.0, eps, False
    )

    # 使用符号化辅助函数进行量化操作，并返回量化后的输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 使用装饰器为函数添加 ONNX 符号，指定为带 ReLU 的量化 1D 卷积操作
# 使用装饰器对函数应用 Beartype 检查，确保输入参数类型正确
def quantized_conv1d_relu(
    g: jit_utils.GraphContext,   # 参数 g 表示图的上下文，用于构建符号化图操作
    q_input,                     # 量化输入数据
    q_weight,                    # 量化权重
    bias,                        # 偏置参数
    stride,                      # 步长
    padding,                     # 填充
    dilation,                    # 空洞卷积的膨胀率
    groups,                      # 分组数
    op_scale,                    # 操作的缩放因子
    op_zero_point,               # 操作的零点偏移量
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)   # 解量化输入数据
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight) # 解量化权重
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)  # 重新量化偏置
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)     # 解量化重新量化的偏置

    # 调用 ONNX opset9 中的 conv1d 函数执行 1D 卷积操作
    output = opset9.conv1d(g, input, weight, bias, stride, padding, dilation, groups)
    # 调用 ONNX opset9 中的 relu 函数执行 ReLU 激活操作
    output = opset9.relu(g, output)

    # 使用符号化辅助函数进行量化操作，并返回量化后的输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 使用装饰器为函数添加 ONNX 符号，指定为带 ReLU 的量化 2D 卷积操作
# 使用装饰器对函数应用 Beartype 检查，确保输入参数类型正确
def quantized_conv2d_relu(
    g: jit_utils.GraphContext,   # 参数 g 表示图的上下文，用于构建符号化图操作
    q_input,                     # 量化输入数据
    q_weight,                    # 量化权重
    bias,                        # 偏置参数
    stride,                      # 步长
    padding,                     # 填充
    dilation,                    # 空洞卷积的膨胀率
    groups,                      # 分组数
    op_scale,                    # 操作的缩放因子
    op_zero_point,               # 操作的零点偏移量
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)   # 解量化输入数据
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight) # 解量化权重
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)  # 重新量化偏置
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)     # 解量化重新量化的偏置

    # 调用 ONNX opset9 中的 conv2d 函数执行 2D 卷积操作
    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)
    # 调用 ONNX opset9 中的 relu 函数执行 ReLU 激活操作
    output = opset9.relu(g, output)

    # 使用符号化辅助函数进行量化操作，并返回量化后的输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 使用装饰器为函数添加 ONNX 符号，指定为带 ReLU 的量化 3D 卷积操作
# 使用装饰器对函数应用 Beartype 检查，确保输入参数类型正确
def quantized_conv3d_relu(
    g: jit_utils.GraphContext,   # 参数 g 表示图的上下文，用于构建符号化图操作
    q_input,                     # 量化输入数据
    q_weight,                    # 量化权重
    bias,                        # 偏置参数
    stride,                      # 步长
    padding,                     # 填充
    dilation,                    # 空洞卷积的膨胀率
    groups,                      # 分组数
    op_scale,                    # 操作的缩放因子
    op_zero_point,               # 操作的零点偏移量
):
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)   # 解量化输入数据
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight) # 解量化权重
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)  # 重新量化偏置
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)     # 解量化重新量化的偏置

    # 调用 ONNX opset9 中的 conv3d 函数执行 3D 卷积操作
    output = opset9.conv3d(g, input, weight, bias, stride, padding, dilation, groups)
    # 调用 ONNX opset9 中的 relu 函数执行 ReLU 激活操作
    output = opset9.relu(g, output)

    # 使用符号化辅助函数进行量化操作，并返回量化后的输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
    op_scale,  # 操作的缩放因子，用于量化操作
    op_zero_point,  # 操作的零点偏移量，用于量化操作
# 定义处理量化的二维转置卷积操作的符号函数
@_onnx_symbolic("quantized::conv_transpose2d")
# 使用 beartype 库对函数参数进行类型检查
@_beartype.beartype
# 函数签名，接受图上下文 g 和量化输入、权重、偏置、步幅、填充、输出填充、扩展率、分组、操作缩放和零点作为参数
def quantized_conv_transpose2d(
    g: jit_utils.GraphContext,
    q_input,            # 量化的输入张量
    q_weight,           # 量化的权重张量
    bias,               # 偏置张量
    stride,             # 步幅
    padding,            # 填充
    output_padding,     # 输出填充
    dilation,           # 扩展率
    groups,             # 分组数
    op_scale,           # 操作的缩放因子
    op_zero_point,      # 操作的零点
):
    # 使用辅助函数对量化的输入和权重进行反量化，获取反量化后的张量、量化比例、以及其他无用的返回值
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    # 使用辅助函数对偏置进行重新量化，计算重新量化后的偏置
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    # 对重新量化的偏置进行反量化，获取反量化后的偏置、量化比例、以及其他无用的返回值
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 执行 opset9 中的二维转置卷积操作，使用反量化后的输入、权重、偏置，以及给定的步幅、填充、输出填充、扩展率和分组
    output = opset9.conv_transpose2d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    # 使用辅助函数对输出进行量化，返回量化后的输出张量
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
    # 使用符号辅助函数从量化的输入中解量化得到输入数据、输入数据的缩放因子，忽略后面两个返回值
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 使用符号辅助函数从量化的权重中解量化得到权重数据、权重数据的缩放因子，忽略后面两个返回值
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    # 使用符号辅助函数重新量化偏置数据，基于输入和权重的缩放因子
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    # 使用符号辅助函数从量化的偏置中解量化得到偏置数据、偏置数据的缩放因子，忽略后面两个返回值
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 使用 opset9 中的转置卷积操作进行计算，输入为解量化后的输入数据、权重数据、偏置数据，以及卷积的其他参数
    output = opset9.conv_transpose2d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    # 使用符号辅助函数对输出进行量化，使用给定的输出缩放因子和零点偏移量
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
# 将函数注册为 quantized::conv_transpose3d 的符号运算
# 应用 Beartype 装饰器确保函数签名的类型安全
@_onnx_symbolic("quantized::conv_transpose3d")
@_beartype.beartype
def quantized_conv_transpose3d(
    g: jit_utils.GraphContext,
    q_input,                 # 输入量化张量
    q_weight,                # 权重量化张量
    bias,                    # 偏置张量
    stride,                  # 跨度
    padding,                 # 填充
    output_padding,          # 输出填充
    dilation,                # 膨胀系数
    groups,                  # 分组数
    op_scale,                # 操作的缩放因子
    op_zero_point,           # 操作的零点
):
    # 对输入量化张量进行反量化，并获取输入的缩放因子和零点
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 对权重量化张量进行反量化，并获取权重的缩放因子和零点
    weight, weight_scale, _, _ = symbolic_helper.dequantize_helper(g, q_weight)
    # 对偏置进行重新量化，并根据输入和权重的缩放因子进行调整
    q_bias = symbolic_helper.requantize_bias_helper(g, bias, input_scale, weight_scale)
    # 对偏置进行反量化，并获取其缩放因子和零点
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 调用 opset9 库中的 conv_transpose3d 函数执行反量化的三维转置卷积操作
    output = opset9.conv_transpose3d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    # 将输出量化，并应用操作的缩放因子和零点
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 将函数注册为 quantized::cat 的符号运算
# 解析输入参数，并应用 Beartype 装饰器确保函数签名的类型安全
@_onnx_symbolic("quantized::cat")
@symbolic_helper.parse_args("v", "i", "v", "v")
@_beartype.beartype
def quantized_cat(
    g: jit_utils.GraphContext,
    q_inputs: _C.Value,      # 量化输入张量的列表
    dim: int,                # 拼接的维度
    op_scale: _C.Value,      # 操作的缩放因子
    op_zero_point: _C.Value, # 操作的零点
) -> _C.Value:
    # 解压量化输入张量列表
    unpacked_inputs = symbolic_helper._unpack_list(q_inputs)
    # 对每个解压后的张量进行反量化操作，获取反量化后的张量列表
    dequantized = [
        symbolic_helper.dequantize_helper(g, input)[0] for input in unpacked_inputs
    ]
    # 在给定的维度上执行拼接操作
    concatenated = g.op("Concat", *dequantized, axis_i=dim)
    # 将拼接后的张量进行量化，并应用操作的缩放因子和零点
    return symbolic_helper.quantize_helper(g, concatenated, op_scale, op_zero_point)
```