# `.\pytorch\torch\_inductor\fx_passes\quantization.py`

```
# mypy: allow-untyped-defs
import copy  # 导入 copy 模块，用于复制对象
import functools  # 导入 functools 模块，用于高阶函数操作
import itertools  # 导入 itertools 模块，用于创建迭代器的函数
import math  # 导入 math 模块，提供数学函数
import operator  # 导入 operator 模块，包含基本运算符的函数
from typing import Any, Tuple  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 深度学习框架
from torch._dynamo.utils import counters  # 从 torch._dynamo.utils 中导入 counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols  # 导入符号形状相关函数
from torch.fx.node import map_arg  # 导入 map_arg 函数
from ..lowering import lowerings as L, require_channels_last  # 导入 lowering 和 require_channels_last 函数
from ..pattern_matcher import (  # 从 pattern_matcher 模块导入多个函数和类
    Arg, CallFunction, filter_nodes, KeywordArg, ListOf, Match
)
from ..utils import pad_listlike  # 导入 pad_listlike 函数
from .freezing_patterns import register_freezing_graph_pattern  # 从 freezing_patterns 模块导入注册函数
from .post_grad import register_lowering_pattern  # 从 post_grad 模块导入注册降级模式的函数

aten = torch.ops.aten  # 设置 aten 为 torch 的 aten 操作符
prims = torch.ops.prims  # 设置 prims 为 torch 的 prims 操作符
quantized_decomposed = torch.ops.quantized_decomposed  # 设置 quantized_decomposed 为 torch 的 quantized_decomposed 操作符
quantized = torch.ops.quantized  # 设置 quantized 为 torch 的 quantized 操作符

# 仅针对每个张量量化，因为 permute 可能改变通道索引
_PER_TENSOR_QUANTIZE_OPS = [
    quantized_decomposed.quantize_per_tensor.default,  # 默认的每个张量量化操作
    quantized_decomposed.quantize_per_tensor.tensor,  # 张量的每个张量量化操作
]

_VIEW_OPS = [
    aten.transpose.int,  # 整数转置操作
    aten.permute.default,  # 默认的排列操作
    aten.view.default,  # 默认的视图操作
]

"""
quantization.py 文件主要包含与量化融合相关的传递过程，包括：
1. 解量化提升；
2. 使用 oneDNN 库对 Conv/GEMM 权重进行预包装；
3. Conv/GEMM 与输出量化节点的量化融合（如果有的话）；
4. 其他逐点操作符的量化融合，如：qmaxpool2d、qcat 等；

还涉及到 int8-mixed-fp32 和 int8-mixed-bf16 量化。int8-mixed-bf16 与 int8-mixed-fp32 的主要区别在于：
1. Conv/GEMM 的激活和权重的输入中有一个 to(dtype=torch.bfloat16) 节点。
2. Conv/GEMM 输出到下一个量化节点之前有一个 to(dtype=torch.float32) 节点。
详细设计请参阅：https://github.com/pytorch/pytorch/issues/111640
"""

def _get_pattern_output_dtype(match: Match):
    """
    从节点的元数据中获取模式的输出数据类型
    假设此匹配模式中只有一个输出节点。
    """
    pattern_output_nodes = match.output_nodes()  # 获取匹配模式的输出节点列表
    assert len(pattern_output_nodes) == 1  # 确保只有一个输出节点
    output_node = pattern_output_nodes[0]  # 获取输出节点
    assert isinstance(output_node, torch.fx.Node)  # 确保输出节点是 torch.fx.Node 类型
    output_dtype = output_node.meta["val"].dtype  # 获取输出节点的数据类型
    assert output_dtype in [torch.uint8, torch.float32, torch.bfloat16]  # 确保数据类型在支持的范围内
    return output_dtype  # 返回输出数据类型

def _may_generate_pattern_with_dtype_convert(
    pattern, dtype=Arg(), with_dtype_convert=True, users=1
):
    """
    如果需要类型转换，则生成带有类型转换的模式
    """
    if with_dtype_convert:
        return CallFunction(
            prims.convert_element_type.default,  # 调用默认的元素类型转换函数
            pattern,
            dtype,
            _users=users,
        )
    else:
        return pattern  # 否则直接返回原模式

def _may_generate_pattern_with_reshape(pattern, reshape_size=Arg(), with_reshape=True):
    """
    如果需要重塑，则生成带有重塑的模式
    """
    if with_reshape:
        return CallFunction(
            torch.ops.aten.reshape.default,  # 调用默认的重塑函数
            pattern,
            reshape_size,
        )
    else:
        return pattern  # 否则直接返回原模式

def _generate_linear_t_pattern(
    _dequant_per_channel_pattern,
    dtype,
):
    # 确保 dtype 是 torch.float32 或 torch.bfloat16 中的一个，否则触发断言错误
    assert dtype in [torch.float32, torch.bfloat16]
    # 创建一个调用函数的模式对象 t_pattern，使用默认的 permute 方法
    t_pattern = CallFunction(
        aten.permute.default,
        # 生成可能带有 dtype 转换的模式，根据 dtype 是否为 torch.bfloat16 决定是否使用 _dequant_per_channel_pattern
        _may_generate_pattern_with_dtype_convert(
            _dequant_per_channel_pattern,
            KeywordArg("autocast_wgt_dtype"),
            dtype == torch.bfloat16,
        ),
        KeywordArg("permute_axes"),
    )
    # 返回生成的模式对象 t_pattern
    return t_pattern
# 定义一个函数，用于生成一元融合模式的模式，其中包含了可能的数据类型转换
def _unary_fusion_pattern(unary_fusion, call_fn, users, is_bf16):
    computation_call = _may_generate_pattern_with_dtype_convert(
        call_fn, dtype=KeywordArg("to_float"), with_dtype_convert=is_bf16, users=users
    )
    return unary_fusion(computation_call)


# 获取用于每个张量激活的去量化模式
def get_dequantize_per_tensor_activation_pattern(is_tensor_overload=False):
    # 创建一个调用函数对象，可以选择使用张量重载的去量化模式或默认的去量化模式
    dequantize_per_tensor_activation_pattern = CallFunction(
        quantized_decomposed.dequantize_per_tensor.tensor
        if is_tensor_overload
        else quantized_decomposed.dequantize_per_tensor.default,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("x_quant_min"),
        KeywordArg("x_quant_max"),
        KeywordArg("x_dq_dtype"),
    )
    return dequantize_per_tensor_activation_pattern


# 创建去量化每通道权重的模式对象
dequantize_per_channel_weight_pattern = CallFunction(
    quantized_decomposed.dequantize_per_channel.default,
    KeywordArg("q_weight"),
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_axis"),
    KeywordArg("w_quant_min"),
    KeywordArg("w_quant_max"),
    KeywordArg("w_dtype"),
)

# 创建将每通道权重转换为bf16格式的模式对象
dequantize_per_channel_to_bf16_weight_pattern = (
    _may_generate_pattern_with_dtype_convert(
        dequantize_per_channel_weight_pattern,
        KeywordArg("autocast_wgt_dtype"),
    )
)

# 创建克隆每通道权重的模式对象
dequantize_per_channel_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)

# 创建将每通道权重转换为bf16格式的克隆模式对象
dequantize_per_channel_to_bf16_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_to_bf16_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)


# 获取用于去量化QConv Pt2e模式的函数
def get_dequantize_qconv_pt2e_pattern(users=1):
    return CallFunction(
        torch.ops.onednn.qconv2d_pointwise.default,
        KeywordArg("x"),
        KeywordArg("x_scale"),  # x_scale
        KeywordArg("x_zp"),  # x_zp
        KeywordArg("packed_weight"),  # packed_weight
        KeywordArg("w_scale"),  # w_scale
        KeywordArg("w_zp"),  # w_zp
        KeywordArg("b"),  # bias
        KeywordArg("stride"),
        KeywordArg("padding"),
        KeywordArg("dilation"),
        KeywordArg("groups"),
        KeywordArg("output_scale"),  # output_scale = 1.0
        KeywordArg("output_zero_point"),  # output_zero_point = 0
        KeywordArg("output_dtype"),  # output_dtype = None
        KeywordArg("attr"),  # attr = "none"
        Arg(),  # scalars
        Arg(),  # algorithm
        _users=users,
    )


# 获取QLinear Pt2e模式的函数
def get_qlinear_pt2e_pattern(x_scale_zp_are_tensors, users=1):
    qlinear_op = (
        torch.ops.onednn.qlinear_pointwise.tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qlinear_pointwise.default
    )
    # 调用一个函数并返回结果
    return CallFunction(
        qlinear_op,                  # 调用的函数对象，执行量化线性操作
        KeywordArg("x"),             # 关键字参数：输入张量 x
        KeywordArg("x_scale"),       # 关键字参数：输入张量的缩放因子 x_scale
        KeywordArg("x_zp"),          # 关键字参数：输入张量的零点偏移 x_zp
        KeywordArg("packed_weight"), # 关键字参数：打包的权重 packed_weight
        KeywordArg("w_scale"),       # 关键字参数：权重的缩放因子 w_scale
        KeywordArg("w_zp"),          # 关键字参数：权重的零点偏移 w_zp
        KeywordArg("b"),             # 关键字参数：偏置 b
        KeywordArg("output_scale"),  # 关键字参数：输出的缩放因子 output_scale
        KeywordArg("output_zero_point"),  # 关键字参数：输出的零点偏移 output_zero_point
        KeywordArg("output_dtype"),  # 关键字参数：输出的数据类型 output_dtype
        KeywordArg("postop_name"),   # 关键字参数：后操作的名称 postop_name
        KeywordArg("postop_args"),   # 关键字参数：后操作的参数 postop_args
        KeywordArg("postop_algorithm"),  # 关键字参数：后操作的算法 postop_algorithm
        _users=users,                # 额外的非关键字参数：用户列表 _users
    )
# 创建一个函数调用对象，调用量化解析器的默认函数，并设置关键字参数
dequantize_accum_pattern = CallFunction(
    quantized_decomposed.dequantize_per_tensor.default,  # 调用量化解析器的默认反量化函数
    KeywordArg("accum"),       # 关键字参数：accum
    KeywordArg("accum_scale"),  # 关键字参数：accum_scale
    KeywordArg("accum_zp"),     # 关键字参数：accum_zp
    Arg(),                      # 位置参数
    Arg(),                      # 位置参数
    KeywordArg("accum_dq_dtype"),  # 关键字参数：accum_dq_dtype
)


def generate_pattern_with_binary(
    binary_post_op,
    computation_call,
    extra_input_pattern,
    dtype_convert=False,
    swap_inputs=False,
):
    # 根据 swap_inputs 参数选择不同的调用方式，生成二进制模式
    binary_pattern = (
        CallFunction(
            binary_post_op,
            extra_input_pattern,
            computation_call,
        )
        if swap_inputs
        else CallFunction(
            binary_post_op,
            computation_call,
            extra_input_pattern,
        )
    )
    return _may_generate_pattern_with_dtype_convert(
        binary_pattern,  # 返回可能根据数据类型转换的二进制模式
        KeywordArg("convert_dtype_after_inplace_add"),  # 关键字参数：convert_dtype_after_inplace_add
        dtype_convert,  # 是否进行数据类型转换
    )


def generate_pattern_with_unary(computation_call, unary_post_op):
    if unary_post_op is not None:
        return CallFunction(
            unary_post_op,
            computation_call,
        )
    return computation_call  # 如果没有一元后操作，则直接返回计算调用


def generate_pattern_with_output_quant(computation_call, with_dtype_convert=False):
    # 生成输出量化模式，调用量化解析器的默认函数
    quantized_op_output_pattern_pt2e = CallFunction(
        quantized_decomposed.quantize_per_tensor.default,
        _may_generate_pattern_with_dtype_convert(
            computation_call,
            Arg(),
            with_dtype_convert,
        ),
        KeywordArg("o_inv_scale"),  # 关键字参数：o_inv_scale
        KeywordArg("o_zp"),         # 关键字参数：o_zp
        KeywordArg("o_qmin"),       # 关键字参数：o_qmin
        KeywordArg("o_qmax"),       # 关键字参数：o_qmax
        KeywordArg("o_dtype"),      # 关键字参数：o_dtype
    )
    return quantized_op_output_pattern_pt2e  # 返回量化操作的输出模式


def _check_node_kwarg_arg_value(check_node, kwarg_name, args_index, expected_value):
    if kwarg_name in check_node.kwargs:
        actual_value = check_node.kwargs[kwarg_name]
        return actual_value == expected_value  # 检查节点关键字参数值是否符合预期
    else:
        assert len(check_node.args) >= (args_index + 1)
        actual_value = check_node.args[args_index]
        return actual_value == expected_value  # 检查节点位置参数值是否符合预期


def _is_valid_quantized_conv2d_optimization_pattern():
    def fn(match):
        output_dtype = _get_pattern_output_dtype(match)  # 获取模式的输出数据类型
        if output_dtype in [torch.float32, torch.bfloat16]:  # 如果输出数据类型为 torch.float32 或 torch.bfloat16
            # 只保留输出数据类型相同的匹配模式
            qconv_node_after_weight_prepack = filter_nodes(
                match.nodes, torch.ops.onednn.qconv2d_pointwise
            )[0]  # 获取第一个符合条件的节点
            return _check_node_kwarg_arg_value(
                qconv_node_after_weight_prepack, "output_dtype", 13, output_dtype
            )  # 检查节点的关键字参数值是否符合预期
        return True

    return fn  # 返回函数 fn


def _register_quantized_conv_lowering(
    pattern,
    pass_number,
    computation_op,
    unary_attr,
):
    @register_lowering_pattern(
        pattern,  # 量化卷积降低的模式
        extra_check=_is_valid_quantized_conv2d_optimization_pattern(),  # 额外检查是否为有效的量化卷积优化模式
        pass_number=pass_number,  # 传递给注册模式的 pass_number 参数
    )
    # 定义一个函数 qconv，接受一个 Match 对象和额外的位置参数和关键字参数
    def qconv(match: Match, *args, **kwargs):
        # 获取激活量化参数
        x, x_scale, x_zp = (
            kwargs["x"],      # 输入张量
            kwargs["x_scale"],  # 输入张量的量化比例因子
            kwargs["x_zp"],     # 输入张量的零点
        )
        
        # 获取权重量化参数
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],  # 打包的权重张量
            kwargs["w_scale"],        # 权重张量的量化比例因子
            kwargs["w_zp"],           # 权重张量的零点
        )
        
        # 获取卷积参数
        b, stride, padding, dilation, groups = (
            kwargs["b"],         # 偏置
            kwargs["stride"],    # 步长
            kwargs["padding"],   # 填充
            kwargs["dilation"],  # 膨胀
            kwargs["groups"],    # 分组卷积数
        )
        
        # 获取输出数据类型
        output_dtype = _get_pattern_output_dtype(match)
        assert output_dtype in [torch.uint8, torch.float32, torch.bfloat16]
        
        # 获取输出量化参数
        o_inv_scale = kwargs["o_inv_scale"] if output_dtype == torch.uint8 else 1.0  # 输出张量的量化反比例因子
        o_zero_point = kwargs["o_zp"] if output_dtype == torch.uint8 else 0            # 输出张量的零点
        assert (
            kwargs["attr"] == "none"
        )  # 在权重预打包阶段预期没有后处理融合操作
        
        # 如果是 "hardtanh" 操作，则设置额外的标量属性
        if unary_attr.op_name == "hardtanh":
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")
            unary_attr.scalars_attr = [min_value, max_value]
    
        # 组装所有的计算参数
        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            unary_attr.op_name,
            unary_attr.scalars_attr,
            unary_attr.algorithm_attr,
        )
        
        # 更新计数器，记录匹配器调用次数和节点数
        counters["inductor"]["qconv2d_unary_matcher_count"] += 1
        counters["inductor"]["qconv2d_unary_matcher_nodes"] += len(match.nodes)
        
        # 返回计算操作的结果
        return L[computation_op](*computation_args)
# 定义一个函数 _is_valid_quantized_linear_optimization_pattern，返回一个内部函数 fn，用于检查量化线性优化模式的有效性
def _is_valid_quantized_linear_optimization_pattern():
    def fn(match):
        # 获取模式匹配后的输出数据类型
        output_dtype = _get_pattern_output_dtype(match)
        
        # 如果输出数据类型为 torch.float32 或 torch.bfloat16，则继续处理
        if output_dtype in [torch.float32, torch.bfloat16]:
            # 从匹配节点中过滤出 torch.ops.onednn.qlinear_pointwise 的节点
            qlinear_node_after_weight_prepack = filter_nodes(
                match.nodes, torch.ops.onednn.qlinear_pointwise
            )[0]
            # 检查节点的关键字参数 "output_dtype" 的值是否符合预期
            return _check_node_kwarg_arg_value(
                qlinear_node_after_weight_prepack, "output_dtype", 9, output_dtype
            )
        
        # 如果输出数据类型不是 torch.float32 或 torch.bfloat16，则返回 True
        return True

    return fn


# 定义一个函数 _register_quantized_linear_lowering，注册量化线性运算的降级操作
def _register_quantized_linear_lowering(
    pattern,
    pass_number,
    computation_op,
    unary_attr,
):
    # 定义一个内部函数 qlinear，作为 register_lowering_pattern 的回调函数
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_quantized_linear_optimization_pattern(),
        pass_number=pass_number,
    )
    def qlinear(match: Match, *args, **kwargs):
        # 获取模式匹配后的输出数据类型
        output_dtype = _get_pattern_output_dtype(match)
        
        # Activation QParams
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        
        # Weight QParams
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )

        # bias
        b = kwargs["b"] if "b" in kwargs else None

        # Output QParams
        # 如果输出数据类型是 torch.uint8，则使用对应的 o_inv_scale 和 o_zero_point，否则使用默认值
        o_inv_scale = kwargs["o_inv_scale"] if output_dtype == torch.uint8 else 1.0
        o_zero_point = kwargs["o_zp"] if output_dtype == torch.uint8 else 0
        
        # 断言预期没有在权重预打包阶段融合的后处理操作
        assert (
            kwargs["postop_name"] == "none"
        )  # Expected no post op fused in weight prepack phase

        # 构造运算所需的参数元组
        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            unary_attr.op_name,
            unary_attr.scalars_attr,
            unary_attr.algorithm_attr,
        )
        
        # 增加计数器的统计信息
        counters["inductor"]["qlinear_unary_matcher_count"] += 1
        counters["inductor"]["qlinear_unary_matcher_nodes"] += len(match.nodes)
        
        # 调用 L 中指定的 computation_op 函数，并传入计算参数
        return L[computation_op](*computation_args)

    # 返回内部函数 qlinear
    return qlinear


# 定义一个函数 _register_quantized_linear_binary_lowering，注册量化线性二进制运算的降级操作
def _register_quantized_linear_binary_lowering(
    pattern,
    pass_number,
    computation_op,
    binary_unary_attr,
):
    # 使用 register_lowering_pattern 注册降级模式
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qlinear_binary_optimization_pattern(),  # 此处应该调用合适的函数 _is_valid_qlinear_binary_optimization_pattern
        pass_number=pass_number,
    )
    # 定义一个函数 qlinear_binary，接受一个匹配对象 Match 和其他位置参数与关键字参数
    def qlinear_binary(match: Match, *args, **kwargs):
        # 获取模式输出数据类型
        output_dtype = _get_pattern_output_dtype(match)
        # 断言确保输出数据类型不为 None
        assert output_dtype is not None
    
        # Activation QParams
        # 从 kwargs 中获取激活量化参数
        x, x_scale, x_zp = (
            kwargs["x"],
            kwargs["x_scale"],
            kwargs["x_zp"],
        )
        # 根据不同的二元操作名确定 x2 的值
        x2 = (
            kwargs["accum"]
            if binary_unary_attr.binary_op_name == "sum"
            else kwargs["other"]
        )
        x2_scale = 1.0
        x2_zp = 0
    
        # Weight QParams
        # 从 kwargs 中获取权重量化参数
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
    
        # bias
        # 如果 kwargs 中有偏置 b，则赋给变量 b，否则设为 None
        b = kwargs["b"] if "b" in kwargs else None
    
        # Output QParams
        # 根据输出数据类型选择输出逆缩放因子和零点值
        o_inv_scale = kwargs["o_inv_scale"] if output_dtype == torch.uint8 else 1.0
        o_zero_point = kwargs["o_zp"] if output_dtype == torch.uint8 else 0
    
        # 调用 x2 的 realize 方法（假设为某个对象的方法）
        x2.realize()
    
        # 导入模块 .mkldnn_fusion 中的 _can_be_inplace 函数
        from .mkldnn_fusion import _can_be_inplace
    
        # 获取二元操作名
        binary_op_name = binary_unary_attr.binary_op_name
    
        # 如果二元操作名为 "sum" 且 x2 不适合原地操作，则修改二元操作名为 "add"
        if binary_op_name == "sum" and not _can_be_inplace(x2):
            # 当启用 GEMM 模板时，如果 QLinear 的输出是 3D 输入，则将其从 2D 重塑回 3D。
            # 在这种情况下，如果 x2 恰好是这种情况下的 QLinear 输出，则 _can_be_inplace(x2) 返回 False。
            # 为此情况更改后操作从 sum 改为 binary add。
            binary_op_name = "add"
    
        # 构造计算参数元组
        computation_args = (
            x,
            x_scale,
            x_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            x2,
            x2_scale,
            x2_zp,
            binary_op_name,
            binary_unary_attr.alpha,
            binary_unary_attr.unary_op_name,
            binary_unary_attr.scalars_attr,
            binary_unary_attr.algorithm_attr,
        )
    
        # 增加计数器中的 qlinear_binary_matcher_count 计数
        counters["inductor"]["qlinear_binary_matcher_count"] += 1
        # 增加计数器中的 qlinear_binary_matcher_nodes 计数，数量为匹配对象 match 的节点数
        counters["inductor"]["qlinear_binary_matcher_nodes"] += len(match.nodes)
    
        # 返回使用计算操作 L[computation_op] 执行计算参数的结果
        return L[computation_op](*computation_args)
    
    # 返回函数 qlinear_binary 作为最终结果
    return qlinear_binary
# 检查是否为有效的量化操作二进制优化模式，针对 qconv2d_pointwise 函数
def _is_valid_qconv_binary_optimization_pattern():
    return _is_valid_quantized_op_binary_optimization_pattern(
        torch.ops.onednn.qconv2d_pointwise
    )


# 检查是否为有效的量化线性操作二进制优化模式，针对 qlinear_pointwise 函数
def _is_valid_qlinear_binary_optimization_pattern():
    return _is_valid_quantized_op_binary_optimization_pattern(
        torch.ops.onednn.qlinear_pointwise,
        # 由于精度问题，不为额外输入插入 q-dq 模式
        extra_input_from_dequant=False,
    )


# 检查是否为有效的量化操作二进制优化模式
# qop: 量化操作函数
# extra_input_from_dequant: 是否从去量化模式获取额外输入，默认为 True
def _is_valid_quantized_op_binary_optimization_pattern(
    qop, extra_input_from_dequant=True
):
    # 检查是否为有效的二进制模式：
    # * qop_pointwise 应该仅有一个使用者
    # * 如果 extra_input_from_dequant 为 True，则二进制节点的额外输入应该来自去量化模式
    # * 二进制节点的两个输入应该具有 "meta" 属性且应该是张量
    # * 二进制节点的两个输入应该具有相同的形状
    # * 在这种模式下，额外输入的所有使用者应该是计算节点的祖先节点，除了连接到计算节点的二进制节点之外
    # 定义一个函数 fn，接受一个 match 参数
    def fn(match):
        # 获取模式匹配输出的数据类型
        output_dtype = _get_pattern_output_dtype(match)
        # 从 match 的节点中筛选出符合条件的计算节点
        compute_node = filter_nodes(match.nodes, qop)[0]
        # 确保 qop_pointwise 节点只有一个用户
        if len(compute_node.users) != 1:
            return False
        # 获取二进制节点的输入
        binary_node_inputs = next(iter(compute_node.users)).args
        # 断言二进制节点有且只有两个输入
        assert len(binary_node_inputs) == 2, "Expects binary node with 2 inputs"
        # 如果输出数据类型是 torch.float32 或 torch.bfloat16
        if output_dtype in [torch.float32, torch.bfloat16]:
            extra_input_of_binary_node = None
            # 查找二进制节点的额外输入，这些输入通常来自去量化模式
            for arg in binary_node_inputs:
                if arg != compute_node:
                    extra_input_of_binary_node = arg
                    break
            assert extra_input_of_binary_node is not None
            # 如果额外输入来自去量化并且不符合预期条件，则返回 False
            if extra_input_from_dequant and (
                (not isinstance(extra_input_of_binary_node, torch.fx.Node))
                or (
                    extra_input_of_binary_node.target
                    != quantized_decomposed.dequantize_per_tensor.default
                )
            ):
                return False
    
        # 确保二进制节点的两个输入具有 "meta" 属性并且是张量
        if not (
            hasattr(binary_node_inputs[0], "meta")
            and isinstance(binary_node_inputs[0].meta.get("val", None), torch.Tensor)  # type: ignore[union-attr]
        ) or not (
            hasattr(binary_node_inputs[1], "meta")
            and isinstance(binary_node_inputs[1].meta.get("val", None), torch.Tensor)  # type: ignore[union-attr]
        ):
            return False
        # 确保二进制节点的两个输入具有相同的形状
        if (
            binary_node_inputs[0].meta["val"].size()  # type: ignore[union-attr]
            != binary_node_inputs[1].meta["val"].size()  # type: ignore[union-attr]
        ):
            return False
    
        # 确保额外输入模式的所有用户都是计算节点的祖先节点，除了连接到计算节点的二进制节点之外
        from .mkldnn_fusion import _get_remaining_users
        # 确定模式的额外输入
        extra_input_of_pattern = (
            match.kwargs["other"]
            if "other" in match.kwargs
            else (
                match.kwargs["accum"]
                if output_dtype == torch.uint8 or (not extra_input_from_dequant)
                else match.kwargs["accum_after_dequant"]
            )
        )
        # 如果额外输入模式的剩余用户数量大于 1 或者额外输入模式等于计算节点的第一个参数，则返回 False
        if (
            len(
                _get_remaining_users(
                    extra_input_of_pattern,
                    compute_node,
                )
            )
            > 1
            or extra_input_of_pattern == compute_node.args[0]
        ):
            return False
        # 若通过所有条件，则返回 True
        return True
# 定义一个函数用于注册量化卷积的二进制降低模式
def _register_quantized_conv_binary_lowering(
    pattern,
    pass_number,
    computation_op,
    binary_unary_attr,
):
    # 使用装饰器注册降低模式的函数，其中传入的附加检查函数为有效的量化卷积二进制优化模式
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_qconv_binary_optimization_pattern(),
        pass_number=pass_number,
    )
    def qconv_binary(match: Match, *args, **kwargs):
        # 获取模式匹配后的输出数据类型
        output_dtype = _get_pattern_output_dtype(match)
        assert output_dtype is not None  # 断言确保输出数据类型非空

        # 根据输出数据类型选择合适的参数
        x, x_scale, x_zp = kwargs["x"], kwargs["x_scale"], kwargs["x_zp"]
        accum = (
            kwargs["accum"]
            if output_dtype == torch.uint8
            else kwargs["accum_after_dequant"]
        )
        accum_scale = (
            kwargs["accum_scale"]
            if output_dtype == torch.uint8
            else 1.0
        )
        accum_zp = (
            kwargs["accum_zp"]
            if output_dtype == torch.uint8
            else 0
        )
        packed_weight, w_scale, w_zp = (
            kwargs["packed_weight"],
            kwargs["w_scale"],
            kwargs["w_zp"],
        )
        b, stride, padding, dilation, groups = (
            kwargs["b"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
            kwargs["groups"],
        )

        # 输出的量化参数
        o_inv_scale = (
            kwargs["o_inv_scale"]
            if output_dtype == torch.uint8
            else 1.0
        )
        o_zero_point = (
            kwargs["o_zp"]
            if output_dtype == torch.uint8
            else 0
        )

        # 实现累加对象
        accum.realize()
        from .mkldnn_fusion import _can_be_inplace

        # 断言确保累加对象可以进行原地操作融合
        assert _can_be_inplace(
            accum
        ), "QConv Binary Inplace Fusion requires accum is not an alias or mutation."

        # 构建计算参数元组
        computation_args = (
            x,
            x_scale,
            x_zp,
            accum,
            accum_scale,
            accum_zp,
            packed_weight,
            w_scale,
            w_zp,
            b,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            binary_unary_attr.binary_op_name,
            binary_unary_attr.alpha,
            binary_unary_attr.unary_op_name,
            binary_unary_attr.scalars_attr,
            binary_unary_attr.algorithm_attr,
        )

        # 增加匹配计数器的数量和节点数
        counters["inductor"]["qconv2d_binary_matcher_count"] += 1
        counters["inductor"]["qconv2d_binary_matcher_nodes"] += len(match.nodes)

        # 调用具体的计算操作函数并返回结果
        return L[computation_op](*computation_args)

    # 返回注册的量化卷积二进制降低模式函数
    return qconv_binary


# 定义一个函数用于注册量化一元融合
def _register_quantization_unary_fusion():
    # 导入相关模块进行一元融合
    from .mkldnn_fusion import (
        _gelu_fusion_1 as _gelu_fusion_erf,
        _gelu_fusion_2 as _gelu_fusion_tanh,
        _hardswish_fusion,
        _hardtanh_fusion,
        _silu_fusion,
    )

    # 定义一元操作的属性类
    class UnaryAttr:
        def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""


# 定义一个函数用于注册量化二元融合
def _register_quantization_binary_fusion():
    # 此处待完善，尚未提供具体实现内容
    # 定义一个类 BinaryUnaryAttr，用于存储二元和一元操作的属性信息
    class BinaryUnaryAttr:
        # 初始化方法，接受二元操作名称、alpha 参数、一元操作名称，默认值为空列表和空字符串的参数
        def __init__(
            self,
            binary_op_name: str,
            alpha=None,
            unary_op_name: str = "none",
            scalars_attr=None,
            algorithm_attr=None,
        ):
            # 设置二元操作的名称
            self.binary_op_name = binary_op_name
            # 如果提供了 alpha 参数则使用，否则设置为默认值 1.0
            self.alpha = alpha if alpha else 1.0
            # 设置一元操作的名称
            self.unary_op_name = unary_op_name
            # 如果提供了 scalars_attr 参数则使用，否则设置为空列表
            self.scalars_attr = scalars_attr if scalars_attr else []
            # 如果提供了 algorithm_attr 参数则使用，否则设置为空字符串
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""

    # QLinear
    r"""
    Supported linear-binary(-unary) patterns

        linear(X)   extra input
               \   /
                Add
                 |
            Optional(relu)
                 |
                 Y

    1. int8-mixed-fp32
    +---+---------------+-----------+------------------------------+---------+
    | # | Add type      | Quant out | Pattern                      | Post op |
    +---+---------------+-----------+------------------------------+---------+
    | 1 | In-/out-place | Yes       | linear + fp32 -> (relu) -> q | add     |
    +---+---------------+-----------+------------------------------+---------+
    | 2 | In-/out-place | No        | linear + fp32 -> (relu)      | sum     |
    +---+---------------+-----------+------------------------------+---------+

    2. int8-mixed-bf16
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | # | X2 dtype | Add type      | Quant out | Pattern                                 | Post op |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 1 | BF16     | In-/out-place | Yes       | linear + bf16 -> (relu) -> q            | add     |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 2 | BF16     | In-/out-place | No        | linear + bf16 -> (relu)                 | sum     |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 3 | FP32     | Out-place     | Yes       | linear + fp32 -> (relu) -> q            | add     |
    |   |          | In-place right|           |                                         |         |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 4 | FP32     | Out-place     | No        | linear + fp32 -> (relu)                 | sum     |
    |   |          | In-place right|           |                                         |         +
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 5 | FP32     | In-place left | Yes       | linear + fp32 -> to_bf16 -> (relu) -> q | add     |
    +---+----------+---------------+-----------+-----------------------------------------+---------+
    | 6 | FP32     | In-place left | No        | linear + fp32 -> to_bf16 -> (relu)      | add     |
    ```
    +---+----------+---------------+-----------+-----------------------------------------+---------+

# 这段代码似乎是一个表格或者文档中的注释或说明，内容较为复杂，需要结合上下文理解其含义。
Note
(1) The positions of linear and the extra input can be swapped.
(2) we don't insert q-dq before the extra input of linear-add by recipe. But if q-dq is found at the
extra input, we don't match that pattern because we cannot match all these patterns in 3 passes.
"""
def _is_valid_quantized_maxpool2d_optimization_pattern():
    # 定义内部函数fn，用于匹配满足量化最大池化优化模式的条件
    def fn(match):
        # 仅匹配 max_pool2d_with_indices 返回值而不是索引的模式
        get_item_node = filter_nodes(match.nodes, operator.getitem)[0]
        return get_item_node.args[1] == 0

    return fn

def _register_quantized_maxpool2d_lowering(
    pattern,
    computation_op,
):
    # 注册量化最大池化降低模式的函数
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_quantized_maxpool2d_optimization_pattern(),
    )
    def qmaxpool2d(match: Match, *args, **kwargs):
        # 从关键字参数中提取必要的值
        x = kwargs["x"]
        kernel_size = kwargs["kernel_size"]
        stride = kwargs["stride"] if ("stride" in kwargs) else None
        padding = kwargs["padding"] if ("padding" in kwargs) else 0
        dilation = kwargs["dilation"] if ("dilation" in kwargs) else 1
        ceil_mode = kwargs["ceil_mode"] if ("ceil_mode" in kwargs) else False

        if padding == 0:
            padding = [0, 0]
        if dilation == 1:
            dilation = [1, 1]
        if not stride:
            stride = kernel_size

        # 将 kernel_size、stride、padding、dilation 转换为长度为2的列表
        kernel_size = pad_listlike(kernel_size, 2)
        stride = pad_listlike(stride, 2)
        padding = pad_listlike(padding, 2)
        dilation = pad_listlike(dilation, 2)

        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert len(padding) == 2
        assert len(dilation) == 2

        # 准备计算操作的参数
        computation_args = (
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )

        # 要求 channels_last 形式的计算参数
        computation_args, _ = require_channels_last(computation_op, *computation_args)

        # 增加计数器的数量记录
        counters["inductor"]["qmaxpool2d_matcher_count"] += 1
        counters["inductor"]["qmaxpool2d_matcher_nodes"] += len(match.nodes)

        # 返回计算操作的结果
        return L[computation_op](*computation_args)

    return qmaxpool2d


def _register_quantization_maxpool2d():
    # 当前情况下，默认参数不在 Dynamo 导出生成的 FX 图中。
    # 因此，如果用户使用不同的默认参数分配定义 nn.MaxPool2d，
    # 会生成具有不同数量输入节点的图，因此匹配的模式也会不同。
    # 参考问题：https://github.com/pytorch/pytorch/issues/105901
    # 定义包含不同关键字参数组合的 max_pool2d_args_list 列表
    max_pool2d_args_list = [
        [
            KeywordArg("stride"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
            KeywordArg("dilation"),
        ],
        [
            KeywordArg("stride"),
            KeywordArg("padding"),
            KeywordArg("dilation"),
            KeywordArg("ceil_mode"),
        ],
    ]
    # 遍历 max_pool2d_args_list 中的每个 max_pool2d_args
    for max_pool2d_args in max_pool2d_args_list:
        # 创建用于 dequantize maxpool2d 模式的函数调用对象
        dequantize_maxpool2d_pattern = CallFunction(
            aten.max_pool2d_with_indices.default,
            get_dequantize_per_tensor_activation_pattern(),
            KeywordArg("kernel_size"),
            *max_pool2d_args,
        )
        # 创建用于 dequantize low memory maxpool2d 模式的函数调用对象
        dequantize_lowmem_maxpool2d_pattern = CallFunction(
            prims._low_memory_max_pool2d_with_offsets.default,
            get_dequantize_per_tensor_activation_pattern(),
            KeywordArg("kernel_size"),
            *max_pool2d_args,
            KeywordArg("offset_dtype"),
        )
        # 创建从 dequantize maxpool2d 模式获取项的函数调用对象
        dequantize_maxpool2d_get_item_pattern = CallFunction(
            operator.getitem,
            dequantize_maxpool2d_pattern,
            Arg(),
        )
        # 创建从 dequantize low memory maxpool2d 模式获取项的函数调用对象
        dequantize_lowmem_maxpool2d_get_item_pattern = CallFunction(
            operator.getitem,
            dequantize_lowmem_maxpool2d_pattern,
            Arg(),
        )
        # 注册 quantized maxpool2d 的降低模式，使用 dequantize maxpool2d 获取项的模式
        _register_quantized_maxpool2d_lowering(
            generate_pattern_with_output_quant(dequantize_maxpool2d_get_item_pattern),
            quantized.max_pool2d.default,
        )
        # 注册 quantized maxpool2d 的降低模式，使用 dequantize low memory maxpool2d 获取项的模式
        _register_quantized_maxpool2d_lowering(
            generate_pattern_with_output_quant(
                dequantize_lowmem_maxpool2d_get_item_pattern
            ),
            quantized.max_pool2d.default,
        )
# 定义一个内部函数，用于检查输入和输出是否具有相同的量化标度和零点
def _is_input_output_same_scale_zp(check_node):
    def fn(match):
        # 确保所有的输入和输出具有相同的标度和零点
        # Step 1: 检查输入/输出的零点
        # 获取输入处的去量化节点
        dequant_nodes = filter_nodes(
            match.nodes, quantized_decomposed.dequantize_per_tensor.default
        )
        # 提取所有去量化节点的零点值
        zero_points = [node.args[2] for node in dequant_nodes]
        
        # 获取输出处的量化节点
        quant_nodes = filter_nodes(
            match.nodes, quantized_decomposed.quantize_per_tensor.default
        )
        assert len(quant_nodes) == 1, "expect only 1 add node at output quant pattern"
        # 将输出节点的零点值加入到列表中
        zero_points.append(quant_nodes[0].args[2])
        
        # 如果不是所有的零点值都等于列表中的第一个零点值，则返回 False
        if not all(zero_point == zero_points[0] for zero_point in zero_points):
            return False

        # Step 2: 检查输入/输出的标度
        # 提取所有去量化节点的标度值
        scales = [node.args[1] for node in dequant_nodes]
        # 将输出节点的标度值加入到列表中
        scales.append(quant_nodes[0].args[1])
        
        # 如果不是所有的标度值都与列表中的第一个标度值在相对误差范围内接近，则返回 False
        if not all(math.isclose(scale, scales[0], rel_tol=1e-5) for scale in scales):  # type: ignore[arg-type]
            return False

        return True

    return fn


# 注册量化降低模式的函数，将其作为一个特定模式的降低
def _register_quantized_cat_lowering(
    pattern,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_input_output_same_scale_zp(aten.cat.default),
    )
    def qcat(match: Match, inputs, dim, **kwargs):
        # inputs 的格式为：[[x1, x1_dq_dtype, x1_zp, x1_scale], ...]
        # 提取 uint8 类型的输入
        uint8_inputs = [input[0] for input in inputs]
        # 增加计数器的计数和节点数
        counters["inductor"]["qcat_matcher_count"] += 1
        counters["inductor"]["qcat_matcher_nodes"] += len(match.nodes)
        # 返回计算操作的结果
        return L[computation_op](uint8_inputs, dim)

    return qcat


# 创建原始去量化张量激活模式
_raw_dequantize_per_tensor_activation_pattern = CallFunction(
    quantized_decomposed.dequantize_per_tensor.default,
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
    Arg(),
)


# 注册量化重塑降低函数，将其作为一个特定模式的降低
def _register_quantized_reshape_lowering(
    pattern,
    computation_op,
):
    @register_lowering_pattern(
        pattern,
        extra_check=_is_input_output_same_scale_zp(aten.reshape.default),
    )
    def qreshape(match: Match, *args, **kwargs):
        qx = kwargs["x"]
        shape = kwargs["shape"]
        # 增加计数器的计数和节点数
        counters["inductor"]["qreshape_matcher_count"] += 1
        counters["inductor"]["qreshape_matcher_nodes"] += len(match.nodes)
        # 返回计算操作的结果
        return L[computation_op](qx, shape)

    return qreshape


# 注册量化重塑函数
def _register_quantization_reshape():
    # 创建去量化张量激活模式的重塑模式
    dequantize_reshape_pattern = CallFunction(
        torch.ops.aten.reshape.default,
        get_dequantize_per_tensor_activation_pattern(),
        KeywordArg("shape"),
    )
    # 注册量化重塑降低函数
    _register_quantized_reshape_lowering(
        generate_pattern_with_output_quant(dequantize_reshape_pattern),
        aten.reshape,
    )
    # 注册量化重塑操作的优化策略
    _register_quantized_reshape_lowering(
        # 生成包含输出量化的重塑模式
        generate_pattern_with_output_quant(dequantize_reshape_pattern),
        # 使用 torch 的 reshape 函数进行重塑操作
        aten.reshape,
    )
def _is_valid_woq_optimization_pattern():
    # 定义内部函数 fn，用于验证匹配对象是否符合优化模式的要求
    def fn(match):
        # 断言所有必需的关键字参数在匹配对象中存在
        assert all(k in match.kwargs for k in ("x", "weight", "scales"))
        # 从匹配对象的关键字参数中获取 x, weight, scales 的值
        x = match.kwargs["x"].meta["val"]
        weight = match.kwargs["weight"].meta["val"]
        scales = match.kwargs["scales"].meta["val"]
        # 返回是否符合优化条件的布尔值
        return (
            # 目前仅支持 x 类型为 bfloat16 和 weight 类型为 int8 的 woq mm 内核
            x.dtype == torch.bfloat16
            and weight.dtype == torch.int8
            and scales.dtype == torch.bfloat16
            # _weight_int8pack_mm 内核目前仅支持 CPU
            # TODO: 支持 CUDA 内核，替换当前的 mul+sum 调用方式
            and x.device.type == "cpu"
            and x.device == weight.device
            and x.device == scales.device
        )

    return fn


def _register_woq_lowering(pattern, computation_woq, computation_reshape):
    # 注册降低模式函数，匹配指定的模式，并执行额外的验证
    @register_lowering_pattern(
        pattern,
        extra_check=_is_valid_woq_optimization_pattern(),
    )
    def woq(match: Match, *args, **kwargs):
        # 从关键字参数中获取 x, weight, scales
        x = kwargs["x"]
        weight = kwargs["weight"]
        scales = kwargs["scales"]
        # 记录匹配次数和节点数量到 counters
        counters["inductor"]["woq_matcher_count"] += 1
        counters["inductor"]["woq_matcher_nodes"] += len(match.nodes)
        # 获取 weight 的输出特征数量和 x 的原始尺寸
        out_features = weight.get_size()[0]
        origin_x_size = x.get_size()
        # 设置 x 的形状为 [-1, x 最后一个维度的大小]
        x_shape = [-1, origin_x_size[-1]]
        # 设置输出形状为 x 的除最后一个维度外，加上输出特征数量
        out_shape = origin_x_size[:-1] + [
            out_features,
        ]
        # 执行计算重塑操作并返回结果
        func1 = L[computation_reshape](x, x_shape)
        func2 = L[computation_woq](func1, weight, scales)
        return L[computation_reshape](func2, out_shape)

    return woq


def _register_woq_mm_int8_pattern1():
    # 注册 woq mm int8 模式的第一个模式
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # 分发到 mm，包括 x 重塑
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.reshape.default,
            CallFunction(
                aten.mm.default,
                CallFunction(aten.reshape.default, KeywordArg("x"), Arg()),
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
            ),
            Arg(),
        ),
        KeywordArg("scales"),
    )
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_woq_mm_int8_pattern2():
    # 注册 woq mm int8 模式的第二个模式
    # F.linear(x, weight.to(dtype=x.dtype)) * scales
    # 分发到 mm，不包括 x 重塑
    # 这里的注释暂时缺失，需要补充
    pass  # 在这里添加对该模式的详细注释
    # 定义一个模式 `_woq_pattern`，用于描述一系列函数调用，实现特定的操作流程
    _woq_pattern = CallFunction(  # 创建一个函数调用对象，用于调用 `aten.mul.Tensor` 函数
        aten.mul.Tensor,  # 指定要调用的函数 `aten.mul.Tensor`
        CallFunction(  # 嵌套调用，调用 `aten.reshape.default` 函数
            aten.reshape.default,  # 调用 `aten.reshape.default` 函数
            CallFunction(  # 继续嵌套调用，调用 `aten.mm.default` 函数
                aten.mm.default,  # 调用 `aten.mm.default` 函数
                KeywordArg("x"),  # 使用关键字参数 "x"
                CallFunction(  # 继续嵌套调用，调用 `aten.permute.default` 函数
                    aten.permute.default,  # 调用 `aten.permute.default` 函数
                    CallFunction(  # 继续嵌套调用，调用 `prims.convert_element_type.default` 函数
                        prims.convert_element_type.default,  # 调用 `prims.convert_element_type.default` 函数
                        KeywordArg("weight"),  # 使用关键字参数 "weight"
                        Arg()  # 使用一个位置参数
                    ),
                    Arg()  # 使用一个位置参数
                ),
            ),
            Arg()  # 使用一个位置参数
        ),
        KeywordArg("scales")  # 使用关键字参数 "scales"
    )
    
    # 将 `_woq_pattern` 注册为特定函数 `_register_woq_lowering` 的下降模式
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)
def _register_woq_mm_int8_pattern3():
    # 定义一个模式 _woq_pattern，用于处理 int8 类型的矩阵乘法
    # 通过 F.linear(x, weight.to(dtype=x.dtype)) * scales 实现
    # 这里是调度到 bmm 的情况
    _woq_pattern = CallFunction(
        aten.mul.Tensor,
        CallFunction(
            aten.bmm.default,
            CallFunction(aten.expand.default, KeywordArg("x"), Arg()),
            CallFunction(
                aten.expand.default,
                CallFunction(
                    aten.permute.default,
                    CallFunction(
                        prims.convert_element_type.default, KeywordArg("weight"), Arg()
                    ),
                    Arg(),
                ),
                Arg(),
            ),
        ),
        KeywordArg("scales"),
    )
    # 注册 _woq_pattern 到指定的 lowering 函数中
    _register_woq_lowering(_woq_pattern, aten._weight_int8pack_mm.default, aten.reshape)


def _register_quantization_lowerings():
    # 注册量化的一元操作融合
    _register_quantization_unary_fusion()
    # 注册量化的二元操作融合
    _register_quantization_binary_fusion()
    # 注册量化的最大池化操作
    _register_quantization_maxpool2d()
    # 注册量化的拼接操作
    _register_quantization_cat()
    # 注册量化的重塑操作
    _register_quantization_reshape()


def _register_woq_lowerings():
    # 分别注册三种不同的 int8 类型矩阵乘法模式
    _register_woq_mm_int8_pattern1()
    _register_woq_mm_int8_pattern2()
    _register_woq_mm_int8_pattern3()


def _is_valid_dequant_promotion_pattern(dtype=torch.float32):
    # 检查是否为有效的去量化推广模式函数
    def _inner(match):
        assert dtype in [torch.float32, torch.bfloat16]
        # 获取去量化模式的结束节点
        dequant_pattern_end_node = match.output_node()
        # 检查结束节点的目标是否在允许的目标列表中
        if dequant_pattern_end_node.target not in [
            quantized_decomposed.dequantize_per_tensor.default,
            quantized_decomposed.dequantize_per_tensor.tensor,
            prims.convert_element_type.default,
            aten.reshape.default,
        ]:
            return False

        # 如果结束节点的目标是 aten.reshape.default
        if dequant_pattern_end_node.target is aten.reshape.default:
            # 根据 dtype 不同，选择不同的去量化节点
            dequant_node = (
                dequant_pattern_end_node.args[
                    0
                ]  # 模式: linear <- reshape <- dequant
                if dtype == torch.float32
                else dequant_pattern_end_node.args[0].args[
                    0
                ]  # 模式: linear <- reshape <- to_bf16 <- dequant
            )
        else:
            # 如果结束节点的目标不是 aten.reshape.default
            dequant_node = (
                dequant_pattern_end_node  # 模式: linear <- dequant
                if dtype == torch.float32
                else dequant_pattern_end_node.args[
                    0
                ]  # 模式: linear <- to_bf16 <- dequant
            )

        # 检查去量化节点是否在允许的去量化节点列表中，并且节点的使用者数量大于 1
        if (
            dequant_node.target
            in [
                quantized_decomposed.dequantize_per_tensor.default,
                quantized_decomposed.dequantize_per_tensor.tensor,
            ]
            and len(list(dequant_pattern_end_node.users)) > 1
        ):
            # 如果去量化模式有多于 1 个使用者，则进行去量化推广
            return True
        return False

    return _inner


def _register_dequant_promotion_pass(pattern, pass_number, dtype=torch.float32):
    # 注册去量化推广的 pass
    # 使用@register_freezing_graph_pattern装饰器注册图模式，传入以下参数：
    # - pattern: 图模式的定义
    # - extra_check: 额外的检查函数，用于验证量化解除提升的有效性，根据给定的dtype进行检查
    # - pass_number: 传递给注册函数的一个参数，可能是一个数字或标识符
def _is_valid_dequant_conv2d_pattern(dtype):
    # 定义内部函数，用于检查匹配的模式是否有效
    def _inner(match):
        # 在此进行进一步检查以确保：
        # 1. 它是一个具有四个维度的 conv2d 节点，因为我们目前只支持降低 conv2d。
        # 2. dequant 模式只有一个 conv2d 节点的用户。
        # 如果不满足这些条件，我们将不会在匹配的模式中插入权重预打包节点。
        
        # 获取 conv2d 节点
        conv_node = match.output_node()
        # 断言 conv_node 的目标是默认的 convolution
        assert conv_node.target is aten.convolution.default
        # 获取 conv2d 节点的输入元数据值
        input_meta_value = conv_node.args[0].meta.get("val")
        # 获取 conv2d 节点的权重元数据值
        weight_meta_value = conv_node.args[1].meta.get("val")
        
        # 检查输入和权重元数据值是否符合条件
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type != "cpu"
                or meta_value.dim() != 4
            ):
                # 只支持当前的 conv2d
                return False
        
        # 断言 dtype 必须是 torch.float32 或 torch.bfloat16
        assert dtype in [torch.float32, torch.bfloat16]
        
        # 根据 dtype 选择 dequant 节点
        if dtype == torch.float32:
            dequant_node = conv_node.args[0]
        else:
            convert_to_bf16 = conv_node.args[0]
            dequant_node = convert_to_bf16.args[0]
        
        # 确保 dequant 模式只有一个用户
        if len(list(dequant_node.users)) != 1:
            # 确保 dequant 模式只有一个用户，因为我们将在这里删除 dequant 模式
            return False
        return True

    return _inner


def _register_qconv_weight_prepack_pass(pattern, pass_number, dtype=torch.float32):
    # 注册 qconv 权重预打包传递的模式
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dequant_conv2d_pattern(dtype),
        pass_number=pass_number,
    )
    def _generate_dequant_convolution_node_pattern(
        _dequant_per_channel_pattern, dtype=torch.float32
    ):
        # 断言 dtype 必须是 torch.float32 或 torch.bfloat16
        assert dtype in [torch.float32, torch.bfloat16]
        
        # 生成 dequant convolution 节点模式
        dequant_convolution_node_pattern = CallFunction(
            aten.convolution.default,
            _may_generate_pattern_with_dtype_convert(
                get_dequantize_per_tensor_activation_pattern(),
                KeywordArg("autocast_act_dtype"),
                dtype == torch.bfloat16,
            ),
            _dequant_per_channel_pattern,
            KeywordArg("b"),
            KeywordArg("stride"),
            KeywordArg("padding"),
            KeywordArg("dilation"),
            KeywordArg("is_transposed"),
            KeywordArg("out_padding"),
            KeywordArg("groups"),
        )
        return dequant_convolution_node_pattern


def _generate_qconv_weight_prepack_patterns(dtype=torch.float32):
    # 断言 dtype 必须是 torch.float32 或 torch.bfloat16
    assert dtype in [torch.float32, torch.bfloat16]
    # 返回一个元组，包含两个元素：
    # 1. 调用 _generate_dequant_convolution_node_pattern 函数生成的模式，根据 dtype 决定使用不同的权重解量化模式
    # 2. 另一种模式，由于 convert_conv_weights_to_channels_last 传递而存在，根据 dtype 决定使用不同的权重克隆解量化模式
    return (
        _generate_dequant_convolution_node_pattern(
            # 如果 dtype 是 torch.float32，则使用 dequantize_per_channel_weight_pattern
            # 否则使用 dequantize_per_channel_to_bf16_weight_pattern
            dequantize_per_channel_weight_pattern
            if dtype == torch.float32
            else dequantize_per_channel_to_bf16_weight_pattern,
            dtype,
        ),
        # 在卷积和解量化节点 dequant_per_channel 之间可能会插入 to(channel_last) 节点，根据一些启发式方法决定
        _generate_dequant_convolution_node_pattern(
            # 如果 dtype 是 torch.float32，则使用 dequantize_per_channel_clone_weight_pattern
            # 否则使用 dequantize_per_channel_to_bf16_clone_weight_pattern
            dequantize_per_channel_clone_weight_pattern
            if dtype == torch.float32
            else dequantize_per_channel_to_bf16_clone_weight_pattern,
            dtype,
        ),
    )
# 根据输入的匹配对象及其属性，获取线性节点和输出重塑节点（如果存在）
def _get_linear_node(match, input_dim_exceeds_two, input_contiguous):
    output_reshape_node = None  # 初始化输出重塑节点为空
    if input_dim_exceeds_two:  # 如果输入维度超过两维
        if input_contiguous:  # 并且输入是连续的
            output_reshape_node = match.output_node()  # 获取匹配对象的输出节点作为输出重塑节点
            assert output_reshape_node.target is aten.reshape.default  # 断言输出重塑节点的目标是默认的reshape操作
            linear_node = output_reshape_node.args[0]  # 线性节点是输出重塑节点的第一个参数
        else:
            # 当输入维度超过两维但不连续时，从匹配节点中筛选出aten.bmm.default操作的线性节点
            linear_nodes = filter_nodes(match.nodes, aten.bmm.default)
            assert len(linear_nodes) == 1  # 断言只有一个线性节点
            linear_node = linear_nodes[0]  # 线性节点是筛选结果中的第一个节点
    else:
        linear_node = match.output_node()  # 输入维度不超过两维时，线性节点即为匹配对象的输出节点

    assert linear_node.target in (
        aten.addmm.default,
        aten.mm.default,
        aten.bmm.default,
    )  # 断言线性节点的目标在给定的操作集合中
    return linear_node, output_reshape_node  # 返回线性节点和输出重塑节点（如果存在）


# 根据线性节点及相关条件，获取线性节点的反量化节点及其他相关节点
def _get_linear_dq_node(
    linear_node, input_index, dtype, input_dim_exceeds_two, input_contiguous
):
    act_reshape_node = None  # 初始化激活重塑节点为空
    activation_to_bf16_node = None  # 初始化激活转换为bf16节点为空
    act_expand_node = None  # 初始化激活扩展节点为空
    if input_dim_exceeds_two:  # 如果输入维度超过两维
        if input_contiguous:  # 并且输入是连续的
            act_reshape_node = linear_node.args[input_index]  # 激活重塑节点是线性节点的指定参数
            assert act_reshape_node.target is aten.reshape.default  # 断言激活重塑节点的目标是默认的reshape操作
            if dtype == torch.float32:
                # 模式: 线性 -> 重塑 -> 反量化
                dequant_node = act_reshape_node.args[0]  # 反量化节点是激活重塑节点的第一个参数
            else:
                # 模式: 线性 -> 重塑 -> 转为bf16 -> 反量化
                activation_to_bf16_node = act_reshape_node.args[0]  # 转为bf16节点是激活重塑节点的第一个参数
                dequant_node = activation_to_bf16_node.args[0]  # 反量化节点是转为bf16节点的第一个参数
        else:
            # 当输入维度超过两维但不连续时，从线性节点中获取激活扩展节点
            act_expand_node = linear_node.args[input_index]
            assert act_expand_node.target is aten.expand.default
            if dtype == torch.float32:
                dequant_node = act_expand_node.args[0]
            else:
                activation_to_bf16_node = act_expand_node.args[0]
                dequant_node = activation_to_bf16_node.args[0]
    else:
        if dtype == torch.float32:
            # 模式: 线性 -> 反量化
            dequant_node = linear_node.args[input_index]
        else:
            # 模式: 线性 -> 转为bf16 -> 反量化
            activation_to_bf16_node = linear_node.args[input_index]
            dequant_node = activation_to_bf16_node.args[0]
    return dequant_node, act_reshape_node, activation_to_bf16_node, act_expand_node


# 检查给定的数据类型、输入维度及连续性条件，判断线性模式是否有效
def _is_valid_dequant_linear_pattern(dtype, input_dim_exceeds_two, input_contiguous):
    # 定义内部函数 _inner，用于匹配模式中的条件检查和验证
    def _inner(match):
        # 从匹配对象中获取线性节点，检查是否超过两个输入维度和是否连续
        (
            linear_node,
            _,
        ) = _get_linear_node(match, input_dim_exceeds_two, input_contiguous)

        # 确定输入索引，根据线性节点的目标确定索引位置
        input_index = 1 if linear_node.target is aten.addmm.default else 0
        # 断言数据类型为 torch.float32 或 torch.bfloat16
        assert dtype in [torch.float32, torch.bfloat16]

        # 获取线性节点的量化去量化节点
        (
            dequant_node,
            _,
            _,
            _,
        ) = _get_linear_dq_node(
            linear_node, input_index, dtype, input_dim_exceeds_two, input_contiguous
        )

        # 断言量化去量化节点的目标在指定列表中
        assert dequant_node.target in [
            quantized_decomposed.dequantize_per_tensor.default,
            quantized_decomposed.dequantize_per_tensor.tensor,
        ]

        # 确保量化去量化节点的用户数量为1，以确保仅有一个用户使用该模式
        if len(list(dequant_node.users)) != 1:
            return False

        # 对于输入维度超过两个且不连续的情况，进行额外的检查
        if input_dim_exceeds_two and not input_contiguous:
            # 检查激活函数的扩展大小应与激活函数大小完全相同
            act_expand_size = match.kwargs["act_expand_size"]
            act_node = match.kwargs["x"]
            if not (
                hasattr(act_node, "meta")
                and isinstance(act_node.meta.get("val", None), torch.Tensor)
                and (act_node.meta["val"].size() == torch.Size(act_expand_size))
            ):
                return False

            # 检查权重的排列维度应为 [1, 0]
            wgt_permute_dims = match.kwargs["permute_axes"]
            if wgt_permute_dims != [1, 0]:
                return False

            # 检查以下权重大小的条目：
            # 权重在扩展之前应该具有维度为2
            # 扩展大小应该具有维度为3
            # 扩展大小的第一个维度应与激活函数大小的第一个维度相同
            # 扩展大小的第二个维度应与权重大小的第二个维度相同
            # 扩展大小的第三个维度应与权重大小的第一个维度相同
            qweight_node = match.kwargs["q_weight"]
            wgt_expand_size = match.kwargs["wgt_expand_size"]
            if not (
                hasattr(qweight_node, "meta")
                and isinstance(qweight_node.meta.get("val", None), torch.Tensor)
                and len(qweight_node.meta["val"].size()) == 2
                and len(wgt_expand_size) == 3
                and wgt_expand_size[0] == act_node.meta["val"].size()[0]
                and wgt_expand_size[1] == qweight_node.meta["val"].size()[1]
                and wgt_expand_size[2] == qweight_node.meta["val"].size()[0]
            ):
                return False

        # 返回 True 表示匹配模式通过了所有检查
        return True

    # 返回内部函数 _inner，用于在外部进行调用和使用
    return _inner
# 注册一个包含 qlinear 权重预打包的优化模式，用于特定的模式匹配和优化传递
def _register_qlinear_weight_prepack_pass(
    pattern,
    pass_number,
    dtype=torch.float32,
    input_dim_exceeds_two=False,
    input_contiguous=True,
):
    # 嵌套函数，注册冻结图模式，额外检查是否有效的 dequant 线性模式
    @register_freezing_graph_pattern(
        pattern,
        extra_check=_is_valid_dequant_linear_pattern(
            dtype, input_dim_exceeds_two, input_contiguous
        ),
        pass_number=pass_number,
    )

# 生成 dequant 线性节点模式的函数
def _generate_dequant_linear_node_pattern(
    _dequant_per_channel_pattern,
    dtype=torch.float32,
    input_dim_exceeds_two=False,
    is_tensor_overload=False,
):
    # 断言，确保 dtype 是 torch.float32 或 torch.bfloat16
    assert dtype in [torch.float32, torch.bfloat16]
    # 生成线性转置模式
    t_pattern = _generate_linear_t_pattern(_dequant_per_channel_pattern, dtype)
    # 使用 reshape 生成带有偏置的模式
    dequant_linear_bias_pattern = _may_generate_pattern_with_reshape(
        CallFunction(
            aten.addmm.default,
            KeywordArg("b"),
            _may_generate_pattern_with_reshape(
                _may_generate_pattern_with_dtype_convert(
                    get_dequantize_per_tensor_activation_pattern(is_tensor_overload),
                    KeywordArg("autocast_act_dtype"),
                    dtype == torch.bfloat16,
                ),
                KeywordArg("act_reshape_size"),
                input_dim_exceeds_two,
            ),
            t_pattern,
        ),
        KeywordArg("output_reshape_size"),
        input_dim_exceeds_two,
    )
    # 使用 reshape 生成不带偏置的模式
    dequant_linear_no_bias_pattern = _may_generate_pattern_with_reshape(
        CallFunction(
            aten.mm.default,
            _may_generate_pattern_with_reshape(
                _may_generate_pattern_with_dtype_convert(
                    get_dequantize_per_tensor_activation_pattern(is_tensor_overload),
                    KeywordArg("autocast_act_dtype"),
                    dtype == torch.bfloat16,
                ),
                KeywordArg("act_reshape_size"),
                input_dim_exceeds_two,
            ),
            t_pattern,
        ),
        KeywordArg("output_reshape_size"),
        input_dim_exceeds_two,
    )
    # 返回带偏置和不带偏置的 dequant 线性模式
    return dequant_linear_bias_pattern, dequant_linear_no_bias_pattern


# 生成 dequant bmm 节点模式的函数
def _generate_dequant_bmm_node_pattern(
    _dequant_per_channel_pattern,
    dtype=torch.float32,
    with_bias=False,
    is_tensor_overload=False,
):
    # 当线性维度的激活超过2且不连续时
    t_pattern = _generate_linear_t_pattern(_dequant_per_channel_pattern, dtype)
    
    # 断言，确保 dtype 是 torch.float32 或 torch.bfloat16
    assert dtype in [torch.float32, torch.bfloat16]
    
    # 生成 bmm 的 dequant 模式
    dequant_bmm_pattern = CallFunction(
        aten.bmm.default,
        CallFunction(
            aten.expand.default,
            _may_generate_pattern_with_dtype_convert(
                get_dequantize_per_tensor_activation_pattern(is_tensor_overload),
                KeywordArg("autocast_act_dtype"),
                dtype == torch.bfloat16,
            ),
            KeywordArg("act_expand_size"),
        ),
        CallFunction(
            aten.expand.default,
            t_pattern,
            KeywordArg("wgt_expand_size"),
        ),
    )
    # 定义一个函数 `_generate_pattern_with_output_add`，接受两个参数 `_dequant_bmm_pattern` 和 `_with_bias`
    def _generate_pattern_with_output_add(_dequant_bmm_pattern, _with_bias):
        # 如果 `_with_bias` 参数为真（即为 True），执行以下代码块
        if _with_bias:
            # 返回一个 CallFunction 对象，调用 aten.add.Tensor 函数，传入 `_dequant_bmm_pattern` 和关键字参数 "b"
            return CallFunction(
                aten.add.Tensor,
                _dequant_bmm_pattern,
                KeywordArg("b"),
            )
        else:
            # 如果 `_with_bias` 参数为假（即为 False），直接返回 `_dequant_bmm_pattern` 对象
            return _dequant_bmm_pattern

    # 返回调用 `_generate_pattern_with_output_add` 函数的结果，传入参数 `dequant_bmm_pattern` 和 `with_bias`
    return _generate_pattern_with_output_add(dequant_bmm_pattern, with_bias)
# 定义函数，生成量化线性权重预打包的模式
def _generate_qlinear_weight_prepack_patterns(
    dtype=torch.float32,  # 默认数据类型为浮点数
    input_dim_exceeds_two=False,  # 输入维度是否超过两维的标志，默认为False
    input_contiguous=True,  # 输入是否连续的标志，默认为True
    with_bias=False,  # 是否包含偏置项的标志，默认为False
    is_tensor_overload=False,  # 是否进行张量重载的标志，默认为False
):
    # 如果输入维度超过两维且输入不是连续的，则调用特定的去量化线性节点模式生成函数
    if input_dim_exceeds_two and not input_contiguous:
        return _generate_dequant_bmm_node_pattern(
            dequantize_per_channel_weight_pattern,
            dtype,
            with_bias,
            is_tensor_overload,
        )
    else:
        # 否则调用普通的去量化线性节点模式生成函数
        return _generate_dequant_linear_node_pattern(
            dequantize_per_channel_weight_pattern,
            dtype,
            input_dim_exceeds_two,
            is_tensor_overload,
        )


# 注册去量化提升的功能
def _register_dequant_promotion():
    # 生成所有可能的去量化模式组合
    dequant_pattern_cases = itertools.product(
        [torch.float32, torch.bfloat16], [True, False], [True, False]
    )
    # 遍历所有去量化模式组合
    for dtype, input_dim_exceeds_two, is_tensor_overload in dequant_pattern_cases:
        # 注册四种去量化模式，基于数据类型和输入维度大小
        # 案例1: int8-mixed-fp32，输入维度大小为2
        # 案例2: int8-mixed-fp32，输入维度大小超过2
        # 案例3: int8-mixed-bf16，输入维度大小为2
        # 案例4: int8-mixed-bf16，输入维度大小超过2
        #        量化
        #   + - - - - | - - - - +
        #   |      去量化       |
        #   |         |         |
        #   |    优化(to_bf16)  |
        #   |         |         |
        #   |    优化(reshape)  |
        #   |      /     \      |
        #   |    节点1   节点2  |
        #   + - - | - - - | - - +
        #  优化(reshape)  优化(reshape)
        #   + - - | - - - | - - +
        #  优化(to_fp32)  优化(to_fp32)
        #   + - - | - - - | - - +
        #       量化     量化
        _register_dequant_promotion_pass(
            # 使用类型转换可能生成的模式注册
            _may_generate_pattern_with_reshape(
                _may_generate_pattern_with_dtype_convert(
                    # 获取去量化每个张量激活模式
                    get_dequantize_per_tensor_activation_pattern(
                        is_tensor_overload=is_tensor_overload
                    ),
                    KeywordArg("autocast_act_dtype"),
                    dtype == torch.bfloat16,
                ),
                KeywordArg("act_reshape_size"),
                with_reshape=input_dim_exceeds_two,
            ),
            pass_number=0,  # 在权重预打包之前运行
            dtype=dtype,
        )


# 注册量化卷积权重预打包功能
def _register_qconv_weight_prepack():
    # 遍历浮点数和bfloat16数据类型
    for dtype in [torch.float32, torch.bfloat16]:
        # 生成量化卷积权重预打包模式
        weight_prepack_patterns = _generate_qconv_weight_prepack_patterns(dtype)
        # 遍历每一个权重预打包模式
        for weight_prepack_pattern in weight_prepack_patterns:
            # 注册到pass_number为1，以便在pass_number为0时进行去量化提升
            _register_qconv_weight_prepack_pass(
                weight_prepack_pattern, pass_number=1, dtype=dtype
            )


# 注册量化线性权重预打包功能
def _register_qlinear_weight_prepack():
    # 将根据数据类型、输入维度大小和输入连续性匹配6个线性相关模式
    # 由于代码截断，后续内容未给出
    # 创建一个迭代器，用来生成所有可能的组合：dtype（数据类型）、input_dim_exceeds_two（输入维度是否大于2）、is_tensor_overload（是否张量超载）
    linear_weight_prepack_cases = itertools.product(
        [torch.float32, torch.bfloat16], [True, False], [True, False]
    )

    # Step 1: 从mm和addmm注册模式
    for dtype, input_dim_exceeds_two, is_tensor_overload in linear_weight_prepack_cases:
        # 生成QLinear权重预打包模式列表
        weight_prepack_patterns = _generate_qlinear_weight_prepack_patterns(
            dtype,
            input_dim_exceeds_two,
            is_tensor_overload=is_tensor_overload,
        )
        # 遍历每一个权重预打包模式
        for weight_prepack_pattern in weight_prepack_patterns:
            # 注册到pass_number 1，这样我们可以在pass_number 0中进行去量化推广
            _register_qlinear_weight_prepack_pass(
                weight_prepack_pattern,
                pass_number=1,
                dtype=dtype,
                input_dim_exceeds_two=input_dim_exceeds_two,
            )

    # Step 2: 从bmm注册模式
    # 当输入维度超过2且不连续时，线性可能会分解为bmm
    # 参考：
    # https://github.com/pytorch/pytorch/blob/
    # 80c07df659362a95da7cd4f3ec367abfdace38c4/torch/_decomp/decompositions.py#L3965-L3968
    # 在这种情况下，我们可以将其转换回QLinear
    for dtype, with_bias, is_tensor_overload in itertools.product(
        [torch.float32, torch.bfloat16], [True, False], [True, False]
    ):
        # 使用_generate_qlinear_weight_prepack_patterns函数生成QLinear权重预打包模式
        bmm_pattern = _generate_qlinear_weight_prepack_patterns(
            dtype=dtype,
            input_dim_exceeds_two=True,
            input_contiguous=False,
            with_bias=with_bias,
            is_tensor_overload=is_tensor_overload,
        )
        # 注册QLinear权重预打包优化pass
        _register_qlinear_weight_prepack_pass(
            bmm_pattern,
            pass_number=1
            if with_bias
            else 2,  # 如果有偏置项，输出会有加法操作，因此首先尝试匹配这种情况
            dtype=dtype,
            input_dim_exceeds_two=True,
            input_contiguous=False,
        )
@functools.lru_cache(None)
def _register_quantization_weight_pack_pass():
    # 使用 functools.lru_cache 装饰器，对该函数进行缓存，避免重复计算

    # Step 1: Dequant promotion for int8-mixed-fp32/bf16
    # 执行第一步操作：对于 int8-mixed-fp32/bf16 进行去量化促进

    # 调用 _register_dequant_promotion 函数，实现去量化促进
    _register_dequant_promotion()

    # Step 2: QConv weight prepack
    # 执行第二步操作：QConv 权重预打包

    # 调用 _register_qconv_weight_prepack 函数，实现 QConv 权重预打包
    _register_qconv_weight_prepack()

    # Step 3: QLinear weight prepack
    # 执行第三步操作：QLinear 权重预打包

    # 调用 _register_qlinear_weight_prepack 函数，实现 QLinear 权重预打包
    _register_qlinear_weight_prepack()


def quant_lift_up(graph_module: torch.fx.GraphModule):
    """
    Lift up the quant node before view like nodes. It can benefit performance
    of Attention like block. For example, we have the pattern as:

             DQ
    DQ       LINEAR
    LINEAR   VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    Q        Q
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    We want to lift up the the quant nodes from matmul before view like nodes
    as the output of Linear node.

             DQ
    DQ       LINEAR
    LINEAR   Q
    Q        VIEW
    VIEW     PERMUTE
    PERMUTE  TRANSPOSE
    DQ       DQ
       Matmul
        DIV
        ADD
      SOFTMAX

    It produces a DQ->LINEAR->Q pattern which can be fused by backend.
    """

    def is_view_op(node):
        # 判断节点是否为视图操作节点
        return node.op == "call_function" and node.target in _VIEW_OPS
    for node in graph_module.graph.nodes:
        # 遍历图中的每个节点
        # <TODO> Leslie: 这里我们验证量化节点的输入是否只有一个FX节点，
        # 且其常量标量值为缩放和零点。
        # 如果量化节点的输入有多个FX节点，需要扩展实现以提升所有连接节点
        # 在查看节点之前，以保持拓扑顺序。
        if (
            node.op == "call_function"
            and node.target in _PER_TENSOR_QUANTIZE_OPS
            and len(node.all_input_nodes) == 1
            and is_view_op(node.all_input_nodes[0])
        ):
            # 如果节点是函数调用且目标在_PER_TENSOR_QUANTIZE_OPS中，并且只有一个输入节点，并且是视图操作节点
            quant_node = node
            input_node_of_quant = quant_node.args[0]

            # 检查沿着提升路径的节点是否只有一个用户节点
            # 传播视图类似节点以找到插入新量化节点的位置
            could_lift_up = True
            current_node = quant_node
            input_node = current_node.args[0]
            while is_view_op(input_node):
                if len(input_node.users) != 1:
                    could_lift_up = False
                    break
                current_node = input_node
                input_node = current_node.args[0]

            # 进一步检查第一个视图节点的输入节点是否只有一个用户节点
            if could_lift_up and len(input_node.users) == 1:
                # 将反量化的输入从量化更改为量化的输入
                quant_node.replace_all_uses_with(input_node_of_quant)
                # 插入新的量化节点
                with graph_module.graph.inserting_before(current_node):
                    new_quant_node = graph_module.graph.node_copy(quant_node)
                    input_node.replace_all_uses_with(new_quant_node)

                    # 更新new_quant_node的输入
                    def maybe_replace_node(n: torch.fx.Node) -> torch.fx.Node:
                        if n == input_node_of_quant:
                            return input_node
                        else:
                            return n

                    new_args = map_arg(new_quant_node.args, maybe_replace_node)
                    new_kwargs = map_arg(new_quant_node.kwargs, maybe_replace_node)
                    new_quant_node.args = new_args
                    new_quant_node.kwargs = new_kwargs
                    graph_module.graph.erase_node(quant_node)

    # 对图进行静态分析
    graph_module.graph.lint()
    # 重新编译图
    graph_module.recompile()
```