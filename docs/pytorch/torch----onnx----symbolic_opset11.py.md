# `.\pytorch\torch\onnx\symbolic_opset11.py`

```
# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 11."""
from __future__ import annotations

import functools  # 导入 functools 模块，用于创建偏函数
import sys  # 导入 sys 模块，用于访问系统相关功能
import warnings  # 导入 warnings 模块，用于发出警告
from typing import Optional, Sequence  # 导入类型提示相关模块

import torch  # 导入 PyTorch 库
from torch import _C  # 导入 PyTorch 的 C++ 扩展模块 _C
from torch._C import _onnx as _C_onnx  # 导入 PyTorch ONNX 相关的 C++ 扩展模块 _onnx
from torch.onnx import (  # 导入 PyTorch 的 ONNX 模块和相关函数
    _type_utils,
    errors,
    symbolic_helper,
    symbolic_opset10 as opset10,
    symbolic_opset9 as opset9,
    utils,
)
from torch.onnx._internal import _beartype, jit_utils, registration  # 导入 PyTorch ONNX 内部模块

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = [  # 定义模块中导出的所有函数名
    "add",
    "append",
    "arange",
    "argsort",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "cat",
    "chunk",
    "clamp_max",
    "clamp_min",
    "clamp",
    "constant_pad_nd",
    "cumsum",
    "Delete",
    "embedding_bag",
    "embedding_renorm",
    "flatten",
    "gather",
    "hardtanh",
    "hstack",
    "im2col",
    "index_fill",
    "index",
    "index_copy",
    "index_put",
    "insert",
    "linalg_det",
    "linalg_vector_norm",
    "logdet",
    "masked_scatter",
    "masked_select",
    "mm",
    "narrow",
    "normal",
    "pad",
    "pixel_shuffle",
    "pop",
    "prim_constant_chunk",
    "reflection_pad",
    "relu6",
    "remainder",
    "replication_pad",
    "round",
    "scatter",
    "select",
    "size",
    "sort",
    "split_with_sizes",
    "split",
    "squeeze",
    "stack",
    "topk",
    "unbind",
    "unique_dim",
    "unsqueeze",
    "vstack",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=11)  # 创建 opset 11 下的 ONNX 符号化函数

@_onnx_symbolic("aten::hardtanh")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "f")
@_beartype.beartype
def hardtanh(g: jit_utils.GraphContext, self: _C.Value, min_val: float, max_val: float):
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.FLOAT
    )
    min_val = g.op(
        "Constant",
        value_t=torch.tensor(min_val, dtype=scalar_type.dtype()),
    )  # 创建一个常量操作节点，值为最小值
    max_val = g.op(
        "Constant",
        value_t=torch.tensor(max_val, dtype=scalar_type.dtype()),
    )  # 创建一个常量操作节点，值为最大值
    return symbolic_helper._op_with_optional_float_cast(
        g, "Clip", self, min_val, max_val, opset_before=12
    )  # 返回一个带有可选浮点转换的操作节点

@_onnx_symbolic("aten::clamp")
@_beartype.beartype
def clamp(g: jit_utils.GraphContext, self, min, max):
    @_beartype.beartype
    def _cast_if_not_none(tensor, dtype):
        if tensor is not None and not symbolic_helper._is_none(tensor):
            return g.op(
                "Cast",
                tensor,
                to_i=dtype.onnx_type(),
            )
        else:
            return tensor

    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        min = _cast_if_not_none(min, scalar_type)  # 如果最小值不为 None，则将其转换为指定类型
        max = _cast_if_not_none(max, scalar_type)  # 如果最大值不为 None，则将其转换为指定类型
    # 如果最小值是 None，则调用 clamp_max 函数对 self 进行最大值截断并返回结果
    if symbolic_helper._is_none(min):
        return clamp_max(g, self, max)
    # 如果最大值是 None，则调用 clamp_min 函数对 self 进行最小值截断并返回结果
    elif symbolic_helper._is_none(max):
        return clamp_min(g, self, min)
    else:
        # 如果最小值和最大值都是标量（张量秩为0），则调用 _op_with_optional_float_cast 函数进行裁剪操作
        if (
            symbolic_helper._get_tensor_rank(min) == 0
            and symbolic_helper._get_tensor_rank(max) == 0
        ):
            return symbolic_helper._op_with_optional_float_cast(
                g, "Clip", self, min, max, opset_before=12
            )
        else:
            # 否则先对 self 进行最小值截断，再对结果进行最大值截断，并返回结果
            return clamp_max(g, clamp_min(g, self, min), max)
# 使用装饰器指定该函数对应的ONNX符号"aten::clamp_min"
# 使用装饰器解析函数参数，期望有两个向量参数
# 使用Beartype装饰器确保输入参数的类型正确
def clamp_min(g: jit_utils.GraphContext, self, min):
    # 将min参数转换为self的ONNX类型的整数类型
    min = g.op("Cast", min, to_i=_type_utils.JitScalarType.from_value(self).onnx_type())
    # 如果min是标量（rank为0）
    if symbolic_helper._get_tensor_rank(min) == 0:
        # 创建一个未使用的max变量
        max = opset9.unused(g)
        # 调用_op_with_optional_float_cast函数，执行Clip操作，可能进行浮点数转换，使用opset 12之前的规范
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, min, max, opset_before=12
        )
    else:
        # 调用_op_with_optional_float_cast函数，执行Max操作，可能进行浮点数转换，使用opset 12之前的规范
        return symbolic_helper._op_with_optional_float_cast(
            g, "Max", self, min, opset_before=12
        )


# 使用装饰器指定该函数对应的ONNX符号"aten::clamp_max"
# 使用装饰器解析函数参数，期望有两个向量参数
# 使用Beartype装饰器确保输入参数的类型正确
def clamp_max(g: jit_utils.GraphContext, self, max):
    # 将max参数转换为self的ONNX类型的整数类型
    max = g.op("Cast", max, to_i=_type_utils.JitScalarType.from_value(self).onnx_type())
    # 如果max是标量（rank为0）
    if symbolic_helper._get_tensor_rank(max) == 0:
        # 创建一个未使用的min变量
        min = opset9.unused(g)
        # 调用_op_with_optional_float_cast函数，执行Clip操作，可能进行浮点数转换，使用opset 12之前的规范
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, min, max, opset_before=12
        )
    else:
        # 调用_op_with_optional_float_cast函数，执行Min操作，可能进行浮点数转换，使用opset 12之前的规范
        return symbolic_helper._op_with_optional_float_cast(
            g, "Min", self, max, opset_before=12
        )


# 使用装饰器指定该函数对应的ONNX符号"aten::relu6"
# 使用Beartype装饰器确保输入参数的类型正确
def relu6(g: jit_utils.GraphContext, input):
    # 根据input的值推断出scalar_type，如果无法推断则默认为float类型
    scalar_type = _type_utils.JitScalarType.from_value(
        input, _type_utils.JitScalarType.FLOAT
    )
    # 创建一个常量节点，值为0，数据类型为scalar_type的dtype
    min_val = g.op(
        "Constant",
        value_t=torch.tensor(0, dtype=scalar_type.dtype()),
    )
    # 创建一个常量节点，值为6，数据类型为scalar_type的dtype
    max_val = g.op(
        "Constant",
        value_t=torch.tensor(6, dtype=scalar_type.dtype()),
    )
    # 调用clamp函数，对input进行ReLU6激活函数的限制
    return clamp(g, input, min_val, max_val)


# 使用装饰器指定该函数对应的ONNX符号"aten::select"
# Opset 11 gather接受负索引
# 使用quantized_args装饰器处理量化参数
# 使用装饰器解析函数参数，期望有一个向量参数和一个整数参数和一个向量参数
# 使用Beartype装饰器确保输入参数的类型正确
def select(g: jit_utils.GraphContext, self, dim, index):
    # 调用Gather操作，从self中根据index收集数据，沿dim轴
    return g.op("Gather", self, index, axis_i=dim)


# 使用装饰器指定该函数对应的ONNX符号"aten::index_put"
# 使用Beartype装饰器确保输入参数的类型正确
def index_put(
    g: jit_utils.GraphContext, self, indices_list_value, values, accumulate=False
):
    # 如果indices_list_value是打包列表
    if symbolic_helper._is_packed_list(indices_list_value):
        # 解压缩indices_list_value为indices_list
        indices_list = symbolic_helper._unpack_list(indices_list_value)
    else:
        # 否则，indices_list为包含indices_list_value的列表
        indices_list = [indices_list_value]
    # 将accumulate参数解析为布尔值
    accumulate = symbolic_helper._parse_arg(accumulate, "b")

    # 如果indices_list为空列表
    if len(indices_list) == 0:
        # 直接返回values
        return values
    # 如果索引列表长度大于1，则进行以下操作
    if len(indices_list) > 1:
        # 遍历索引列表中的每个索引
        for idx_ in range(len(indices_list)):
            # 检查当前索引是否为布尔类型，如果是则转换为非零索引操作
            if symbolic_helper._is_bool(indices_list[idx_]):
                indices_list[idx_] = g.op("NonZero", indices_list[idx_])
        
        # 取第一个索引作为初始索引
        index = indices_list[0]

        # 将剩余索引与初始索引累加
        for ind in indices_list[1:]:
            index = opset9.add(g, index, ind)
        
        # 创建一个操作，获取索引的广播形状
        broadcast_index_shape = g.op("Shape", index)
        
        # 对每个索引进行操作，使其在指定维度上变成一维
        indices_list = [
            symbolic_helper._unsqueeze_helper(
                g, opset9.expand(g, ind, broadcast_index_shape, None), [-1]
            )
            for ind in indices_list
        ]
        
        # 将处理过的索引按照指定维度连接起来
        index = g.op("Concat", *indices_list, axis_i=-1)
    
    # 获取数据的子形状
    sub_data_shape = symbolic_helper._slice_helper(
        g, g.op("Shape", self), axes=[0], starts=[len(indices_list)], ends=[sys.maxsize]
    )
    
    # 获取值的形状，连接广播索引形状和数据的子形状
    values_shape = g.op("Concat", broadcast_index_shape, sub_data_shape, axis_i=0)
    
    # 检查值是否是单值，并根据需要扩展
    rank = symbolic_helper._get_tensor_rank(values)
    if rank is not None and rank == 0:
        values = opset9.expand(g, values, values_shape, None)
    
    # 重塑值的形状
    values = symbolic_helper._reshape_helper(g, values, values_shape)

    # 获取self的标量类型
    self_scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    
    # 如果self有有效的标量类型
    if self_scalar_type != _type_utils.JitScalarType.UNDEFINED:
        # 获取值的标量类型
        values_scalar_type = _type_utils.JitScalarType.from_value(
            values, _type_utils.JitScalarType.UNDEFINED
        )
        
        # 如果self和值的标量类型不匹配，则进行类型转换
        if self_scalar_type != values_scalar_type:
            values = g.op("Cast", values, to_i=self_scalar_type.onnx_type())
    
    # 如果需要累加
    if accumulate:
        # 创建一个形状为self形状的全零常量
        zeros = g.op(
            "ConstantOfShape",
            g.op("Shape", self),
            value_t=torch.tensor([0], dtype=self_scalar_type.dtype()),
        )
        
        # 执行ScatterND操作，更新值
        result = g.op("ScatterND", zeros, index, values)
        
        # 将self与结果值相加
        result = add(g, self, result)
    else:
        # 执行ScatterND操作，更新值
        result = g.op("ScatterND", self, index, values)

    # 返回结果
    return result
# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::pixel_shuffle' 操作
@_onnx_symbolic("aten::pixel_shuffle")
# 使用装饰器进行参数解析，期望参数签名为 ("v", "i")
@symbolic_helper.parse_args("v", "i")
# 使用 Beartype 进行类型注解和验证
@_beartype.beartype
# 定义像素混洗函数 pixel_shuffle，将输入张量 self 进行深度到空间的转换
def pixel_shuffle(g: jit_utils.GraphContext, self, upscale_factor):
    # 获取张量 self 的秩（rank）
    rank = symbolic_helper._get_tensor_rank(self)
    # 如果秩不为空且不为4，则返回未实现的错误信息
    if rank is not None and rank != 4:
        return symbolic_helper._unimplemented("pixel_shuffle", "only support 4d input")
    # 使用 ONNX 操作符 "DepthToSpace" 对 self 进行深度到空间的转换，指定块大小和模式
    return g.op("DepthToSpace", self, blocksize_i=upscale_factor, mode_s="CRD")


# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::upsample_nearest1d' 操作
@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest")],
)
# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::upsample_nearest2d' 操作
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest")],
)
# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::upsample_nearest3d' 操作
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest")],
)
# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::upsample_linear1d' 操作
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[symbolic_helper._apply_params("upsample_linear1d", 3, "linear")],
)
# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::upsample_bilinear2d' 操作
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear")],
)
# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::upsample_trilinear3d' 操作
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear")],
)
# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::upsample_bicubic2d' 操作
@_onnx_symbolic(
    "aten::upsample_bicubic2d",
    decorate=[symbolic_helper._apply_params("upsample_bicubic2d", 4, "cubic")],
)
# 使用 Beartype 进行类型注解和验证
@_beartype.beartype
# 定义插值函数 _interpolate，根据给定的名称、维度和插值模式进行插值操作
def _interpolate(name: str, dim: int, interpolate_mode: str):
    # 调用符号化助手函数 _interpolate_helper，传递名称、维度和插值模式进行插值操作
    return symbolic_helper._interpolate_helper(name, dim, interpolate_mode)


# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::__interpolate' 操作
@_onnx_symbolic("aten::__interpolate")
# 使用符号化助手进行量化参数的处理，标识是否是量化的参数
@symbolic_helper.quantized_args(True, False, False, False, False, False, False)
# 使用 Beartype 进行类型注解和验证
@_beartype.beartype
# 定义插值函数 __interpolate，根据给定的输入、尺寸、缩放因子、模式、对齐、重新计算缩放因子和抗锯齿进行插值操作
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
    # 调用符号化助手函数 __interpolate_helper，传递参数进行插值操作
    return symbolic_helper.__interpolate_helper(
        g, input, size, scale_factor, mode, align_corners, recompute_scale_factor
    )


# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::gather' 操作
@_onnx_symbolic("aten::gather")
# 使用装饰器进行参数解析，期望参数签名为 ("v", "i", "v", "v")
@symbolic_helper.parse_args("v", "i", "v", "v")
# 使用 Beartype 进行类型注解和验证
@_beartype.beartype
# 定义聚集函数 gather，根据给定的 self、维度 dim 和索引 index 进行聚集操作
def gather(g: jit_utils.GraphContext, self, dim, index, sparse_grad=False):
    # 如果 sparse_grad 参数不为空，则返回未实现的错误信息
    if symbolic_helper._maybe_get_const(sparse_grad, "i"):
        return symbolic_helper._unimplemented("gather", "sparse_grad == True")
    # 使用 ONNX 操作符 "GatherElements" 对 self 进行聚集操作，指定轴 axis_i 为 dim
    return g.op("GatherElements", self, index, axis_i=dim)


# 使用装饰器定义对应的 ONNX 符号化函数，处理 'aten::scatter' 操作
@_onnx_symbolic("aten::scatter")
# 使用装饰器进行参数解析，期望参数签名为 ("v", "i", "v", "v")
@symbolic_helper.parse_args("v", "i", "v", "v")
# 使用 Beartype 进行类型注解和验证
@_beartype.beartype
# 定义散布函数 scatter，根据给定的 self、维度 dim、索引 index 和源 src 进行散布操作
def scatter(g: jit_utils.GraphContext, self, dim, index, src):
    # 获取源 src 的类型
    src_type = _type_utils.JitScalarType.from_value(src)
    # 尝试获取标量的值
    src = symbolic_helper._maybe_get_scalar(src)
    # 如果源 src 是值类型，则使用 ONNX 操作符 "ScatterElements" 对 self 进行散布操作，指定轴 axis_i 为 dim
    if symbolic_helper._is_value(src):
        return g.op("ScatterElements", self, index, src, axis_i=dim)
    else:
        # 检查标量 "src" 的类型是否与当前对象的类型相同（PyTorch 允许标量 src 的类型与自身不同，但当 src 是张量时不允许）。如果不同，插入一个 Cast 节点。
        if _type_utils.JitScalarType.from_value(self) != src_type:
            # 创建一个 Cast 节点，将 src 转换为与 self 相同的类型，并使用对应的 ONNX 类型
            src = g.op(
                "Cast",
                src,
                to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
            )
        # 返回一个 ScatterElements 节点，用于在 self 上根据 index 和 dim 进行元素散布操作。src 需要通过 opset9.expand_as 函数与 index 扩展到相同形状。
        return g.op(
            "ScatterElements", self, index, opset9.expand_as(g, src, index), axis_i=dim
        )
# 使用装饰器将函数注册为对应 ONNX 符号"aten::cumsum"的符号推导函数
# 同时，使用 beartype 装饰器确保输入参数类型正确
@_onnx_symbolic("aten::cumsum")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype
def cumsum(g: jit_utils.GraphContext, self, dim, dtype=None):
    # 创建一个 Constant 操作节点，用于表示维度 dim 的张量常量
    dim_tensor = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.int))
    
    # 检查 dtype 是否存在且不是常量，如果是，则解析出其数值类型
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        # 使用 Cast 操作将 self 张量转换为指定的数据类型
        cast = g.op(
            "Cast", self, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    else:
        # 如果 dtype 是常量或未提供，则直接使用 self 张量
        cast = self
    
    # 执行累加操作，生成累加后的张量
    csum = g.op("CumSum", cast, dim_tensor)
    
    # 返回累加后的结果张量
    return csum


# 使用装饰器将函数注册为对应 ONNX 符号"aten::masked_select"的符号推导函数
# 同时，使用 beartype 装饰器确保输入参数类型正确
@_onnx_symbolic("aten::masked_select")
@_beartype.beartype
def masked_select(g: jit_utils.GraphContext, self, mask):
    # 使用 opset9 模块提供的非零元素索引方法，获取 mask 在 self 上的索引
    index = opset9.nonzero(g, opset9.expand_as(g, mask, self))
    
    # 使用 GatherND 操作从 self 中按照 index 所指示的位置收集数据
    return g.op("GatherND", self, index)


# 使用装饰器将函数注册为对应 ONNX 符号"aten::masked_scatter"的符号推导函数
# 同时，使用 beartype 装饰器确保输入参数类型正确
@_onnx_symbolic("aten::masked_scatter")
@_beartype.beartype
def masked_scatter(g: jit_utils.GraphContext, self, mask, source):
    # 使用 opset9 模块提供的非零元素索引方法，获取 mask 在 self 上的索引
    index = opset9.nonzero(g, opset9.expand_as(g, mask, self))
    
    # 提示：source 可能包含超出所需的元素，也可能具有任意形状。
    # ONNX::ScatterND 不支持此类输入，因此需要对 source 张量进行扁平化和切片处理。
    source = symbolic_helper._reshape_helper(g, source, torch.LongTensor([-1]))
    source = symbolic_helper._slice_helper(
        g,
        source,
        axes=torch.LongTensor([0]),
        starts=torch.LongTensor([0]),
        ends=opset9.size(g, index, torch.LongTensor([0])),
    )
    
    # 使用 ScatterND 操作将 source 张量的数据散布到 self 中指定的 index 位置
    return g.op("ScatterND", self, index, source)


# 使用装饰器将函数注册为对应 ONNX 符号"aten::len"的符号推导函数
# 同时，使用 beartype 装饰器确保输入参数类型正确
@_onnx_symbolic("aten::len")
@_beartype.beartype
def _len(g: jit_utils.GraphContext, self):
    # 如果 self 是 tensor 列表或者是 onnx::SplitToSequence 操作的结果
    if (
        symbolic_helper._is_tensor_list(self)
        or self.node().kind() == "onnx::SplitToSequence"
    ):
        # 返回序列的长度，使用 SequenceLength 操作
        return g.op("SequenceLength", self)
    
    # 否则，计算 self 的第一个维度的大小，并去除尺寸为 1 的维度
    sz_0 = size(g, self, g.op("Constant", value_t=torch.LongTensor([0])))
    return symbolic_helper._squeeze_helper(g, sz_0, [0])


# 使用装饰器将函数注册为对应 ONNX 符号"aten::__getitem_"的符号推导函数
# 同时，使用 beartype 装饰器确保输入参数类型正确
@_onnx_symbolic("aten::__getitem_")
@_beartype.beartype
def __getitem_(g: jit_utils.GraphContext, self, i):
    # 如果 self 是 tensor 列表，则使用 SequenceAt 操作获取第 i 个元素
    if symbolic_helper._is_tensor_list(self):
        return g.op("SequenceAt", self, i)
    else:
        # 否则，从 torch.onnx.symbolic_opset9 模块中导入 __getitem_ 函数，并调用之
        from torch.onnx.symbolic_opset9 import __getitem_ as getitem

        return getitem(g, self, i)


# 使用装饰器将函数注册为对应 ONNX 符号"aten::_set_item"的符号推导函数
# 同时，使用 beartype 装饰器确保输入参数类型正确
@_onnx_symbolic("aten::_set_item")
@_beartype.beartype
def _set_item(g: jit_utils.GraphContext, tensor_list, i, v):
    # 使用 SequenceErase 操作删除 tensor_list 中索引为 i 的元素
    tensor_list = g.op("SequenceErase", tensor_list, i)
    
    # 使用 SequenceInsert 操作在 tensor_list 中的索引 i 处插入新的元素 v
    return g.op("SequenceInsert", tensor_list, v, i)


# 使用装饰器将函数注册为对应 ONNX 符号"aten::append"的符号推导函数
# 同时，使用 beartype 装饰器确保输入参数类型正确
@_onnx_symbolic("aten::append")
@_beartype.beartype
def append(g: jit_utils.GraphContext, self, tensor):
    # 使用 SequenceInsert 操作在 self 序列末尾插入新的 tensor 元素
    return g.op("SequenceInsert", self, tensor)


# 使用装饰器将函数注册为对应 ONNX 符号"aten::add"的符号推导函数
# 同时，使用 beartype 装饰器确保输入参数类型正确
@_onnx_symbolic("aten::add")
@_beartype.beartype
def add(g: jit_utils.GraphContext, self, other, alpha=None):
    # 检查当前对象是否为值并且是张量列表
    if symbolic_helper._is_value(self) and symbolic_helper._is_tensor_list(self):
        # 获取另一个节点的张量列表
        tensor_list_node = other.node()
        # 如果节点类型不是"prim::ListConstruct"，则返回未实现的错误信息
        if tensor_list_node.kind() != "prim::ListConstruct":
            return symbolic_helper._unimplemented(
                "add", "does not support adding dynamic tensor list to another"
            )
        # 解包另一个张量列表
        tensors = symbolic_helper._unpack_list(other)
        # 初始化列表l为当前对象self
        l = self
        # 遍历张量列表，将每个张量插入到列表l中
        for t in tensors:
            l = g.op("SequenceInsert", l, t)
        # 返回更新后的列表l
        return l

    # 调用opset9中的add函数，将当前对象self和另一个对象other相加，带有alpha参数
    return opset9.add(g, self, other, alpha)
# 将 insert 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::insert")
@_beartype.beartype
def insert(g: jit_utils.GraphContext, self, pos, tensor):
    # 使用 ONNX 操作 "SequenceInsert" 插入 tensor 到 self 中的 pos 位置
    return g.op("SequenceInsert", self, tensor, pos)


# 将 pop 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::pop")
@_beartype.beartype
def pop(g: jit_utils.GraphContext, tensor_list, dim):
    # 使用 ONNX 操作 "SequenceErase" 在 tensor_list 中删除指定维度 dim 的元素
    return g.op("SequenceErase", tensor_list, dim)


# 将 Delete 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::Delete")
@_beartype.beartype
def Delete(g: jit_utils.GraphContext, tensor_list, dim):
    # 使用 ONNX 操作 "SequenceErase" 在 tensor_list 中删除指定维度 dim 的元素
    return g.op("SequenceErase", tensor_list, dim)


# 将 cat 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::cat")
@symbolic_helper.quantized_args(True)
@_beartype.beartype
def cat(g: jit_utils.GraphContext, tensor_list, dim):
    # 如果 tensor_list 是打包列表，则调用 opset9.cat 在维度 dim 上进行拼接
    if symbolic_helper._is_packed_list(tensor_list):
        return opset9.cat(g, tensor_list, dim)
    else:
        # 否则，获取常量维度 dim，使用 ONNX 操作 "ConcatFromSequence" 在该维度上进行拼接
        dim = symbolic_helper._get_const(dim, "i", "dim")
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim)


# 将 stack 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::stack")
@_beartype.beartype
def stack(g: jit_utils.GraphContext, tensor_list, dim):
    # 如果 tensor_list 是打包列表，则调用 opset9.stack 在维度 dim 上进行堆叠
    if symbolic_helper._is_packed_list(tensor_list):
        return opset9.stack(g, tensor_list, dim)
    else:
        # 否则，获取常量维度 dim，使用 ONNX 操作 "ConcatFromSequence" 在该维度上进行堆叠，并添加新的轴
        dim = symbolic_helper._get_const(dim, "i", "dim")
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim, new_axis_i=1)


# 将 _unique2 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::_unique2")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def _unique2(g: jit_utils.GraphContext, self, sorted, return_inverse, return_counts):
    # 使用 ONNX 操作 "Unique" 对 self 进行唯一化操作，并返回唯一化后的结果、索引、逆索引和计数
    u, indices, inverse_indices, counts = g.op(
        "Unique", self, sorted_i=sorted, outputs=4
    )
    return u, inverse_indices, counts


# 将 unique_dim 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::unique_dim")
@symbolic_helper.parse_args("v", "i", "i", "i", "i")
@_beartype.beartype
def unique_dim(
    g: jit_utils.GraphContext, self, dim, sorted, return_inverse, return_counts
):
    # 使用 ONNX 操作 "Unique" 对 self 在指定维度 dim 上进行唯一化操作，并返回唯一化后的结果、索引、逆索引和计数
    u, indices, inverse_indices, counts = g.op(
        "Unique", self, axis_i=dim, sorted_i=sorted, outputs=4
    )
    return u, inverse_indices, counts


# 将 topk 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
@_beartype.beartype
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    # 调用 symbolic_helper._topk_helper 辅助函数，使用 ONNX 操作实现 topk 操作
    return symbolic_helper._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out
    )


# 将 sort 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
@_beartype.beartype
def sort(g: jit_utils.GraphContext, self, dim, decending, out=None):
    # 调用 symbolic_helper._sort_helper 辅助函数，使用 ONNX 操作实现排序操作
    return symbolic_helper._sort_helper(g, self, dim, decending=decending, out=out)


# 将 argsort 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::argsort")
@symbolic_helper.parse_args("v", "i", "i", "none")
@_beartype.beartype
def argsort(g: jit_utils.GraphContext, self, dim, decending, out=None):
    # 调用 symbolic_helper._sort_helper 辅助函数，使用 ONNX 操作获取排序后的索引
    _, indices = symbolic_helper._sort_helper(
        g, self, dim, decending=decending, out=out
    )
    return indices


# 将 round 函数注册为 onnx 符号化操作，并使用 beartype 进行类型检查
@_onnx_symbolic("aten::round")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def round(g: jit_utils.GraphContext, self, decimals=0):
    # 如果 self 不是浮点数类型，则直接返回 self
    if not symbolic_helper._is_fp(self):
        return self
    # 如果小数位数 decimals 等于 0，则执行 Round 操作并返回结果
    if decimals == 0:
        return g.op("Round", self)
    
    # 计算一个常量值，表示 10 的 decimals 次方，并与 self 相乘
    mul = g.op("Mul", self, g.op("Constant", value_t=torch.tensor(pow(10, decimals))))
    
    # 对上一步得到的乘积进行 Round 操作
    round = g.op("Round", mul)
    
    # 计算一个常量值，表示 10 的 -1 * decimals 次方，并与 round 的结果相乘
    # 用于将 round 后的结果还原为原始小数位数
    return g.op(
        "Mul", round, g.op("Constant", value_t=torch.tensor(pow(10, -1 * decimals)))
    )
# 注册 ONNX 符号化函数，处理 PyTorch 中的 "aten::remainder" 操作
@_onnx_symbolic("aten::remainder")
# 应用 Beartype 对函数参数进行类型检查和类型注解
@_beartype.beartype
# 定义函数 remainder，接受图上下文 g、input 和 other 作为参数
def remainder(g: jit_utils.GraphContext, input, other):
    # 如果 input 或 other 是浮点数，则调用 opset9.remainder 处理
    if symbolic_helper._is_fp(input) or symbolic_helper._is_fp(other):
        return opset9.remainder(g, input, other)
    # 否则，使用 "Mod" 操作计算 input 除以 other 的余数
    return g.op("Mod", input, other, fmod_i=0)


# 注册 ONNX 符号化函数，处理 PyTorch 中的 "aten::split" 操作
@_onnx_symbolic("aten::split")
# 解析参数类型为 "v", "v", "i", "i"，并应用 Beartype 进行类型检查和注解
@symbolic_helper.parse_args("v", "v", "i", "i")
@_beartype.beartype
# 定义函数 split，接受图上下文 g、self、split_size_or_sizes、dim 和可选参数 _outputs
def split(g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None):
    # 如果 split_size_or_sizes 不是静态分割，则使用 SplitToSequence 操作进行动态分割
    if not symbolic_helper._is_split_static(split_size_or_sizes, _outputs):
        split_out = g.op("SplitToSequence", self, split_size_or_sizes, axis_i=dim)
        # 如果 _outputs 未指定，则直接返回 split_out
        if _outputs is None:
            return split_out
        # 如果 split_size_or_sizes 是打包列表且与 _outputs 数量相符，则转换为多个 Slice 节点
        if (
            symbolic_helper._is_packed_list(split_size_or_sizes)
            and len(symbolic_helper._unpack_list(split_size_or_sizes)) == _outputs
        ):
            # 对 split_size_or_sizes 中的每个值进行扩展操作，生成 split_sizes 列表
            split_sizes = [
                symbolic_helper._unsqueeze_helper(g, v, [0])
                for v in symbolic_helper._unpack_list(split_size_or_sizes)
            ]
            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            res = []
            # 对每个输出进行切片操作，生成 res 列表
            for i in range(_outputs):
                end = g.op(
                    "Add", start, split_sizes[i]
                )  # split_sizes is a list of same length as _outputs
                res.append(g.op("Slice", self, start, end, axis))
                start = end
            return res
        # 否则，返回 SequenceAt 操作得到的列表
        return [
            g.op(
                "SequenceAt",
                split_out,
                g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)),
            )
            for i in range(_outputs)
        ]
    else:
        # 如果 split_size_or_sizes 是静态分割，则调用 opset9.split 处理
        return opset9.split(g, self, split_size_or_sizes, dim, _outputs)


# 注册 ONNX 符号化函数，处理 PyTorch 中的 "aten::split_with_sizes" 操作
@_onnx_symbolic("aten::split_with_sizes")
# 解析参数类型为 "v", "v", "i", "i"，并应用 Beartype 进行类型检查和注解
@symbolic_helper.parse_args("v", "v", "i", "i")
@_beartype.beartype
# 定义函数 split_with_sizes，与 split 函数功能相同，仅作为 "aten::split_with_sizes" 的别名
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    return split(g, self, split_sizes, dim, _outputs)


# 注册 ONNX 符号化函数，处理 PyTorch 中的 "aten::unbind" 操作
@_onnx_symbolic("aten::unbind")
# 解析参数类型为 "v", "i", "i"，并应用 Beartype 进行类型检查和注解
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
# 定义函数 unbind，接受图上下文 g、self、dim 和可选参数 _outputs
def unbind(g: jit_utils.GraphContext, self, dim=0, _outputs=None):
    # 如果 _outputs 未指定，则使用 SplitToSequence 操作进行解绑
    if _outputs is None:
        return g.op(
            "SplitToSequence",
            self,
            g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
            axis_i=dim,
            keepdims_i=0,
        )
    else:
        # 否则，调用 opset9.unbind 处理解绑操作
        return opset9.unbind(g, self, dim, _outputs)


# 准备 ONNX 格式的填充，基于 PyTorch 中的 pad 参数生成
@_beartype.beartype
# 定义函数 _prepare_onnx_paddings，接受图上下文 g、input 和 pad 参数
def _prepare_onnx_paddings(g: jit_utils.GraphContext, input, pad):
    """Generate paddings in ONNX order based on pad in pytorch."""
    Args:
        input: 输入张量。
        pad: 在 PyTorch 中的填充信息。
            其顺序为 dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ..., dim_m_begin, dim_m_end，
            其中 m 的取值范围为 [0, n]。

    """
    # 如果 pad 不是 packed list 且是 list 且是 scalar list，则进行拼接操作
    if (
        not symbolic_helper._is_packed_list(pad)
        and symbolic_helper._is_list(pad)
        and symbolic_helper._is_scalar_list(pad)
    ):
        pad = g.op("ConcatFromSequence", pad, axis_i=0, new_axis_i=1)

    # 计算 pad 的长度
    pad_len = opset9.size(g, pad, g.op("Constant", value_t=torch.tensor([0])))

    # 获取输入张量的维度数
    rank = symbolic_helper._get_tensor_rank(input)
    if rank is None:
        rank = g.op("Size", g.op("Shape", input))
    else:
        rank = g.op("Constant", value_t=torch.tensor(rank, dtype=torch.int64))

    # 计算需要扩展的长度
    extension = g.op(
        "Sub",
        g.op("Mul", rank, g.op("Constant", value_t=torch.tensor(2, dtype=torch.int64))),
        pad_len,
    )

    # 将 pad 转换为 int64 类型
    pad = g.op("Cast", pad, to_i=_C_onnx.TensorProtoDataType.INT64)

    # 拼接 pad 和 extension，构成 paddings
    paddings = g.op(
        "Concat",
        pad,
        g.op(
            "ConstantOfShape", extension, value_t=torch.tensor([0], dtype=torch.int64)
        ),
        axis_i=0,
    )

    # 通过辅助函数对 paddings 进行重塑
    paddings = symbolic_helper._reshape_helper(
        g, paddings, g.op("Constant", value_t=torch.tensor([-1, 2]))
    )

    # 对 paddings 进行转置，实现顺序颠倒
    paddings = g.op("Transpose", opset10.flip(g, paddings, [0]), perm_i=[1, 0])

    # 再次通过辅助函数对 paddings 进行重塑
    paddings = symbolic_helper._reshape_helper(
        g, paddings, g.op("Constant", value_t=torch.tensor([-1]))
    )

    # 将 paddings 转换为 int64 类型并返回
    padding_c = g.op("Cast", paddings, to_i=_C_onnx.TensorProtoDataType.INT64)
    return padding_c
# 将函数constant_pad_nd标记为ONNX符号化函数，用于处理常量填充的操作
@_onnx_symbolic("aten::constant_pad_nd")
# 应用beartype装饰器，确保函数参数类型符合预期
@_beartype.beartype
def constant_pad_nd(g: jit_utils.GraphContext, input, padding, value=None):
    # 设置填充模式为"constant"
    mode = "constant"
    # 将填充值转换为标量（如果可能）
    value = symbolic_helper._maybe_get_scalar(value)
    # 将填充值的数据类型与输入数据一致化
    value = symbolic_helper._if_scalar_type_as(value, input)
    # 准备ONNX需要的填充参数
    pad = _prepare_onnx_paddings(g, input, padding)
    # 返回ONNX操作节点，表示进行填充操作
    return g.op("Pad", input, pad, value, mode_s=mode)


# 将函数reflection_pad标记为ONNX符号化函数，用于处理反射填充的操作
@_onnx_symbolic("aten::reflection_pad1d")
@_onnx_symbolic("aten::reflection_pad2d")
@_onnx_symbolic("aten::reflection_pad3d")
# 应用beartype装饰器，确保函数参数类型符合预期
@_beartype.beartype
def reflection_pad(g: jit_utils.GraphContext, input, padding):
    # 设置填充模式为"reflect"
    mode = "reflect"
    # 准备ONNX需要的填充参数
    paddings = _prepare_onnx_paddings(g, input, padding)
    # 返回ONNX操作节点，表示进行反射填充操作
    return g.op("Pad", input, paddings, mode_s=mode)


# 将函数replication_pad标记为ONNX符号化函数，用于处理复制填充的操作
@_onnx_symbolic("aten::replication_pad1d")
@_onnx_symbolic("aten::replication_pad2d")
@_onnx_symbolic("aten::replication_pad3d")
# 应用beartype装饰器，确保函数参数类型符合预期
@_beartype.beartype
def replication_pad(g: jit_utils.GraphContext, input, padding):
    # 设置填充模式为"edge"
    mode = "edge"
    # 准备ONNX需要的填充参数
    paddings = _prepare_onnx_paddings(g, input, padding)
    # 返回ONNX操作节点，表示进行边缘复制填充操作
    return g.op("Pad", input, paddings, mode_s=mode)


# 将函数pad标记为ONNX符号化函数，用于根据给定模式进行不同类型的填充操作
@_onnx_symbolic("aten::pad")
# 应用beartype装饰器，确保函数参数类型符合预期
@_beartype.beartype
def pad(
    g: jit_utils.GraphContext,
    input: _C.Value,
    pad: _C.Value,
    mode: _C.Value,
    value: _C.Value,
):
    # 解析填充模式参数为字符串类型
    mode = symbolic_helper._parse_arg(mode, "s")
    # 根据填充模式选择相应的填充操作函数
    if mode == "replicate":
        return replication_pad(g, input, pad)
    elif mode == "reflect":
        return reflection_pad(g, input, pad)
    elif mode == "constant":
        return constant_pad_nd(g, input, pad, value)
    elif mode == "circular":
        # 对于"circular"模式，调用opset9提供的特定填充函数
        return opset9._pad_circular(g, input, pad)
    else:
        # 若填充模式不被识别，则引发错误
        raise errors.SymbolicValueError(f"Unrecognized padding mode {mode}", input)


# 将函数linalg_det标记为ONNX符号化函数，用于计算输入张量的行列式
@_onnx_symbolic("aten::linalg_det")
# 应用beartype装饰器，确保函数参数类型符合预期
@_beartype.beartype
def linalg_det(g: jit_utils.GraphContext, self):
    # 返回ONNX操作节点，表示计算输入张量的行列式
    return g.op("Det", self)


# 将函数logdet标记为ONNX符号化函数，用于计算输入张量的对数行列式
@_onnx_symbolic("aten::logdet")
# 应用beartype装饰器，确保函数参数类型符合预期
@_beartype.beartype
def logdet(g: jit_utils.GraphContext, input):
    # 调用linalg_det函数计算输入张量的行列式，并返回其对数
    return opset9.log(g, linalg_det(g, input))


# 将函数arange标记为ONNX符号化函数，用于创建等差数列张量
@_onnx_symbolic("aten::arange")
# 应用beartype装饰器，确保函数参数类型符合预期
@_beartype.beartype
def arange(g: jit_utils.GraphContext, *args):
    # 内部函数，用于获取arange操作的数据类型
    def _get_arange_dtype(dtype):
        # 将数据类型转换为整数类型
        dtype = symbolic_helper._maybe_get_const(dtype, "i")
        return dtype

    # 如果参数长度为2且所有参数都是整数
    if len(args) == 2 and all(isinstance(val, int) for val in args):
        # 设置默认数据类型为torch.int64
        dtype = torch.int64
        # 创建表示起始索引的常量节点
        start = g.op(
            "Constant",
            value_t=torch.tensor(args[0], dtype=dtype),
        )
        # 创建表示终止（不包含）索引的常量节点
        end = g.op(
            "Constant",
            value_t=torch.tensor(args[1], dtype=dtype),
        )
        # 创建表示从起始到终止索引的步长的常量节点
        delta_default = g.op(
            "Constant",
            value_t=torch.tensor(1, dtype=dtype),
        )
        # 返回ONNX操作节点，表示创建指定范围的等差数列张量
        return g.op("Range", start, end, delta_default)
    # 如果参数个数为2或5时执行以下逻辑
    elif len(args) == 2 or len(args) == 5:
        if len(args) == 2:
            # 如果参数个数为2，则调用 aten::arange(Scalar end, Tensor out)
            dtype = None
        else:
            # 如果参数个数为5，则调用 aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
            # 获取参数 args[1] 对应的数据类型
            dtype = _get_arange_dtype(args[1])
        # 调用 symbolic_helper._arange_cast_helper 方法进行类型转换和参数准备
        type_, end, start, step = symbolic_helper._arange_cast_helper(
            g, end=args[0], dtype=dtype
        )
        # 创建表示起始值的常量节点
        start_default = g.op(
            "Constant",
            value_t=torch.tensor(0, dtype=type_.dtype()),
        )
        # 创建表示步长的常量节点
        delta_default = g.op(
            "Constant",
            value_t=torch.tensor(1, dtype=type_.dtype()),
        )
        # 返回 Range 操作的节点，表示生成一个从 start_default 到 end（不包括）的数列，步长为 delta_default
        return g.op("Range", start_default, end, delta_default)
    # 如果参数个数为4或7时执行以下逻辑
    elif len(args) == 4 or len(args) == 7:
        if len(args) == 4:
            # 如果参数个数为4，则调用 aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
            dtype = None
        else:
            # 如果参数个数为7，则调用 aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
            # 获取参数 args[3] 对应的数据类型
            dtype = _get_arange_dtype(args[3])
        # 调用 symbolic_helper._arange_cast_helper 方法进行类型转换和参数准备
        _, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], step=args[2], dtype=dtype
        )
        # 返回 Range 操作的节点，表示生成一个从 start 到 end（不包括）的数列，步长为 step
        return g.op("Range", start, end, step)
    # 如果参数个数为6时执行以下逻辑
    elif len(args) == 6:
        # 调用 aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        # 获取参数 args[2] 对应的数据类型
        dtype = _get_arange_dtype(args[2])
        # 调用 symbolic_helper._arange_cast_helper 方法进行类型转换和参数准备
        type_, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], dtype=dtype
        )
        # 创建表示步长的常量节点，默认步长为 1
        delta_default = g.op(
            "Constant",
            value_t=torch.tensor(1, dtype=type_.dtype()),
        )
        # 返回 Range 操作的节点，表示生成一个从 start 到 end（不包括）的数列，步长为 delta_default
        return g.op("Range", start, end, delta_default)
    else:
        # 如果参数个数不符合以上情况，则调用 symbolic_helper._unimplemented 方法报告未实现的情况
        return symbolic_helper._unimplemented(
            "aten::arange", f"with {len(args)} arguments"
        )
# 定义一个带有装饰器的函数，该函数用于处理特定的ONNX符号"_dim_arange"
# 装饰器"_onnx_symbolic"指定了该函数在ONNX中的符号表示
# 装饰器"symbolic_helper.parse_args"指定了函数的参数类型和顺序
# 装饰器"_beartype.beartype"确保函数被调用时参数类型正确
def _dim_arange(g: jit_utils.GraphContext, like, dim):
    # 获取输入张量"like"的形状
    like_shape = g.op("Shape", like)
    # 从形状中获取指定维度"dim"的大小作为停止值
    stop = g.op(
        "Gather", like_shape, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0
    )
    # 调用"arange"函数，返回一个等差数列的张量
    return arange(g, stop, 4, None, None, None)


# 定义一个带有装饰器的函数，该函数处理"aten::size"的符号
# 装饰器"_onnx_symbolic"指定了该函数在ONNX中的符号表示
# 装饰器"symbolic_helper.quantized_args"指定了函数的特定参数配置
# 装饰器"_beartype.beartype"确保函数被调用时参数类型正确
def size(g: jit_utils.GraphContext, self, dim=None):
    # 如果未指定维度"dim"，则返回输入张量"self"的形状
    if dim is None:
        return g.op("Shape", self)
    # 否则调用帮助函数"_size_helper"，返回指定维度"dim"的大小
    return symbolic_helper._size_helper(g, self, dim)


# 定义一个带有装饰器的函数，该函数处理"aten::squeeze"的符号
# 装饰器"_onnx_symbolic"指定了该函数在ONNX中的符号表示
# 装饰器"_beartype.beartype"确保函数被调用时参数类型正确
def squeeze(g: jit_utils.GraphContext, self, dim=None):
    # 如果未指定维度"dim"，则返回对输入张量"self"进行挤压操作后的结果
    if dim is None:
        return g.op("Squeeze", self)

    # 如果维度"dim"不是常数，则调用帮助函数"_squeeze_helper"，返回挤压操作后的结果
    if not symbolic_helper._is_constant(dim):
        return symbolic_helper._squeeze_helper(g, self, [dim])

    # 获取维度"dim"的常数值
    dim = symbolic_helper._get_const(dim, "i", "dim")

    # 获取输入张量"self"的秩（维度数量）
    input_rank = symbolic_helper._get_tensor_rank(self)
    adjusted_dim = dim
    # 如果输入张量的秩已知且维度"dim"为负数，则调整为相对索引
    if input_rank is not None and dim < 0:
        adjusted_dim += input_rank
    # 获取维度"dim"的大小
    dim_size = symbolic_helper._get_tensor_dim_size(self, adjusted_dim)

    # 如果维度"dim"为负数且输入张量的秩未知，或者维度大小未知，则进行动态形状的处理
    if (dim < 0 and input_rank is None) or dim_size is None:
        # 创建一个条件节点，条件是维度"dim"的大小是否为1
        dim_constant = g.op("Constant", value_t=torch.tensor([dim]))
        size = symbolic_helper._size_helper(g, self, dim_constant)
        const_one = g.op("Constant", value_t=torch.ones(1, dtype=torch.int64))
        cond = g.op("Equal", size, const_one)
        # 创建"If"节点，并添加"then"和"else"块
        if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
            g, "If", cond, n_blocks=2
        )
        # 在"then"块中进行挤压操作
        squeeze_ = symbolic_helper._squeeze_helper(if_context, self, [dim])
        utils._add_output_to_block(if_context.block, squeeze_)
        # 在"else"块中添加"Identity"操作，保持张量不变
        identity_ = else_context.op("Identity", self)
        utils._add_output_to_block(else_context.block, identity_)
        return if_op

    # 对于静态输入形状，直接返回原始输入张量"self"
    dim = adjusted_dim
    if dim_size > 1:
        # 如果维度"dim"的大小大于1，发出警告
        warnings.warn(
            "This model contains a squeeze operation on dimension "
            + str(dim)
            + ". The size of "
            + "this dimension in the given input is "
            + str(dim_size)
            + ". The model will "
            + "be exported without the squeeze node. If the model is intended to be used with dynamic "
            + "input shapes, please export with dynamic_axes argument."
        )
        return self
    # 否则调用帮助函数"_squeeze_helper"，返回挤压操作后的结果
    return symbolic_helper._squeeze_helper(g, self, [dim])


# 定义一个带有装饰器的函数，该函数处理"aten::unsqueeze"的符号
# 装饰器"_onnx_symbolic"指定了该函数在ONNX中的符号表示
# 装饰器"_beartype.beartype"确保函数被调用时参数类型正确
def unsqueeze(g: jit_utils.GraphContext, self, dim):
    # 在输入张量"self"上进行插入维度操作
    # 如果 dim 是一个常数，调用 symbolic_helper._get_const 函数获取其值，其中 "i" 是期望的类型，"dim" 是描述常数的名称
    if symbolic_helper._is_constant(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")
    
    # 调用 symbolic_helper._unsqueeze_helper 函数，对张量 self 进行 unsqueeze 操作，使用 dim 参数指定的维度
    return symbolic_helper._unsqueeze_helper(g, self, [dim])
# 使用装饰器将函数注册为处理 "aten::mm" 符号的 ONNX 符号函数
# 同时使用 beartype 装饰器进行类型检查
@_onnx_symbolic("aten::mm")
@_beartype.beartype
def mm(g: jit_utils.GraphContext, self, other):
    # 使用 ONNX 操作符 "Gemm" 执行矩阵乘法操作
    return g.op("Gemm", self, other, beta_f=0.0, alpha_f=1.0)


# 使用装饰器将函数注册为处理 "aten::index" 符号的 ONNX 符号函数
# 同时使用 beartype 装饰器进行类型检查
@_onnx_symbolic("aten::index")
@_beartype.beartype
def index(g: jit_utils.GraphContext, self, index):
    # 检查 index 是否为打包的列表，如果是，则解包为 indices
    if symbolic_helper._is_packed_list(index):
        indices = symbolic_helper._unpack_list(index)
    else:
        indices = [index]

    # 处理单个索引情况
    if len(indices) == 1:
        index = indices[0]
        # 如果 index 不为 None 并且是布尔类型或者 UINT8 类型
        if not symbolic_helper._is_none(index) and (
            symbolic_helper._is_bool(index)
            or _type_utils.JitScalarType.from_value(index)
            == _type_utils.JitScalarType.UINT8
        ):
            # 对 index 执行 nonzero 操作，返回非零元素的索引
            index = opset9.nonzero(g, index)
            # 使用 "GatherND" 操作进行多维索引
            return g.op("GatherND", self, index)
    # 调用 opset9 中的 index 操作进行索引
    return opset9.index(g, self, index)


# 使用装饰器将函数注册为处理 "aten::index_fill" 符号的 ONNX 符号函数
# 同时使用 beartype 装饰器进行类型检查
@_onnx_symbolic("aten::index_fill")
@_beartype.beartype
def index_fill(g: jit_utils.GraphContext, self, dim, index, value):
    # 解析 dim 参数，并转换为整数
    dim_value = symbolic_helper._parse_arg(dim, "i")
    # 辅助函数，用于重塑索引填充操作中的形状
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    # 获取标量值，并确保与 self 的类型匹配
    value = symbolic_helper._maybe_get_scalar(value)
    value = symbolic_helper._if_scalar_type_as(value, self)
    # 使用 opset9 中的 expand 操作，扩展 value 的形状
    expanded_value = opset9.expand(g, value, expanded_index_shape, None)
    # 调用 scatter 函数，执行索引填充操作
    return scatter(g, self, dim, expanded_index, expanded_value)


# 使用装饰器将函数注册为处理 "aten::index_copy" 符号的 ONNX 符号函数
# 同时使用 beartype 装饰器进行类型检查
@_onnx_symbolic("aten::index_copy")
@_beartype.beartype
def index_copy(g: jit_utils.GraphContext, self, dim, index, source):
    # 解析 dim 参数，并转换为整数
    dim_value = symbolic_helper._parse_arg(dim, "i")
    # 辅助函数，用于重塑索引填充操作中的形状
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    # 调用 scatter 函数，执行索引复制操作
    return scatter(g, self, dim, expanded_index, source)


# 使用装饰器将函数注册为处理 "aten::bitwise_right_shift" 和 "aten::__rshift_" 符号的 ONNX 符号函数
# 同时使用 beartype 装饰器进行类型检查
@_onnx_symbolic("aten::bitwise_right_shift")
@_onnx_symbolic("aten::__rshift_")
@_beartype.beartype
def __rshift_(g: jit_utils.GraphContext, self, other):
    # 确保将 other 强制转换为 self 的类型
    # （当 self 是长整型时，确保 other 不是浮点型）
    if _type_utils.JitScalarType.from_value(
        other, _type_utils.JitScalarType.UNDEFINED
    ) != _type_utils.JitScalarType.from_value(self):
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
        )

    # 如果 self 的类型为 UINT8，则执行位右移操作
    if (
        _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.UINT8
    ):
        return g.op("BitShift", self, other, direction_s="RIGHT")

    # 创建一个常量张量 "two"，值为 2，数据类型为 float32
    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # 在 onnx::Pow 中，指数（与 self 相同的类型）必须为 float 或 double 类型
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    # 执行 2 的 other 次幂
    two_pow = g.op("Pow", two, other)
    # 将结果转换为与 self 相同的数据类型
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
    )
    # 执行除法操作，实现右移操作
    rshift = g.op("Div", self, two_pow)
    return rshift


# 返回变量 rshift 的值作为函数的结果
# 将函数标记为对"aten::bitwise_left_shift"的ONNX符号化处理
# 将函数标记为对"aten::__lshift_"的ONNX符号化处理
# 使用beartype装饰器确保参数类型正确性
def __lshift_(g: jit_utils.GraphContext, self, other):
    # 确保将other转换为与self相同的类型
    # (当self为long类型时，确保other不是float类型)
    if _type_utils.JitScalarType.from_value(
        other, _type_utils.JitScalarType.UNDEFINED
    ) != _type_utils.JitScalarType.from_value(self):
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
        )

    # 如果self的类型是UINT8，则执行位左移操作
    if (
        _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.UINT8
    ):
        return g.op("BitShift", self, other, direction_s="LEFT")

    # 创建一个常数节点表示2，类型为float32
    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    
    # 如果self不是浮点数，则将other强制转换为浮点数
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    
    # 计算2的other次幂
    two_pow = g.op("Pow", two, other)
    
    # 将计算结果强制转换为与self相同的类型
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
    )
    
    # 执行self与2的other次幂的乘法操作
    lshift = g.op("Mul", self, two_pow)
    return lshift


# 使用beartype装饰器确保参数类型正确性
def _get_im2col_indices_along_dim(
    g: jit_utils.GraphContext, input_d, kernel_size_d, dilation_d, padding_d, stride_d
):
    # 输入始终是4维张量(N, C, H, W)
    
    # 计算沿空间维度滑动块的索引
    # 计算每个维度d上滑动核的索引：
    # 每个维度d的范围从0到input[d]+2xpadding[d]-dilation[d]x(kernel_size[d]-1)
    # 步长为stride
    
    blocks_d = g.op(
        "Add", input_d, g.op("Constant", value_t=torch.tensor(padding_d * 2))
    )
    blocks_d = g.op(
        "Sub",
        blocks_d,
        g.op("Constant", value_t=torch.tensor(dilation_d * (kernel_size_d - 1))),
    )
    
    # 在输入上滑动核并找到沿维度d的起始索引
    blocks_d_indices = g.op(
        "Range",
        g.op("Constant", value_t=torch.tensor(0)),
        blocks_d,
        g.op("Constant", value_t=torch.tensor(stride_d)),
    )
    
    # 对核施加扩展，并找到沿维度d的索引
    kernel_grid = torch.arange(0, kernel_size_d * dilation_d, dilation_d)
    kernel_grid = g.op("Constant", value_t=kernel_grid.unsqueeze(0))
    
    # 广播并添加核起始位置（索引）与kernel_grid沿维度d的值，
    # 以获取沿维度d的块索引
    blocks_d_indices = symbolic_helper._unsqueeze_helper(
        g, blocks_d_indices, [0]
    )  # 重塑为[1, -1]
    kernel_mask = symbolic_helper._reshape_helper(
        g, kernel_grid, g.op("Constant", value_t=torch.tensor([-1, 1]))
    )
    block_mask = g.op("Add", blocks_d_indices, kernel_mask)

    return block_mask


# 使用beartype装饰器确保参数类型正确性
def _get_im2col_padded_input(g: jit_utils.GraphContext, input, padding_h, padding_w):
    # 输入始终是4维张量(N, C, H, W)
    # 此函数尚未实现具体内容，只提供了注释说明
    # 创建一个常量张量用于填充，格式为：(padding_h, padding_w, padding_h, padding_w, padding_h, padding_w, padding_h, padding_w)
    # 这是为了符合 ONNX 格式的要求，其格式为：(dim1_begin, dim2_begin, ..., dim1_end, dim2_end, ..., dim1_begin, dim2_begin, ..., dim1_end, dim2_end, ...)
    pad = g.op("Constant", value_t=torch.LongTensor([0, 0, padding_h, padding_w] * 2))
    # 对输入张量进行填充操作，使用上面创建的填充张量 pad
    return g.op("Pad", input, pad)
# 定义一个装饰器，用于检查函数参数类型和返回值类型
@_beartype.beartype
# 获取im2col操作的输出形状
def _get_im2col_output_shape(g: jit_utils.GraphContext, input, kernel_h, kernel_w):
    # 获取输入张量的批次维度大小（N）
    batch_dim = size(g, input, g.op("Constant", value_t=torch.tensor(0)))
    # 获取输入张量的通道维度大小（C）
    channel_dim = size(g, input, g.op("Constant", value_t=torch.tensor(1)))
    # 计算展开后的通道大小，即通道维度乘以卷积核的大小（kernel_h * kernel_w）
    channel_unfolded = g.op(
        "Mul", channel_dim, g.op("Constant", value_t=torch.tensor(kernel_h * kernel_w))
    )

    # 返回合并后的张量形状，包括批次和展开通道维度
    return g.op(
        "Concat",
        symbolic_helper._unsqueeze_helper(g, batch_dim, [0]),  # 在指定轴上增加批次维度
        symbolic_helper._unsqueeze_helper(g, channel_unfolded, [0]),  # 在指定轴上增加展开通道维度
        g.op("Constant", value_t=torch.tensor([-1])),  # 添加固定的维度值
        axis_i=0,  # 沿着0轴进行合并
    )


# 定义一个ONNX符号操作的函数，用于执行im2col操作
@_onnx_symbolic("aten::im2col")
# 解析输入参数，其中v表示返回值，"is"表示整数数组
@symbolic_helper.parse_args("v", "is", "is", "is", "is")
# 使用装饰器检查函数参数类型和返回值类型
@_beartype.beartype
def im2col(g: jit_utils.GraphContext, input, kernel_size, dilation, padding, stride):
    # 输入张量始终为4维张量（N，C，H，W）
    # 其它参数都是包含两个整数的数组

    # 获取输入张量的高度和宽度
    input_h = size(g, input, g.op("Constant", value_t=torch.tensor(2)))
    input_w = size(g, input, g.op("Constant", value_t=torch.tensor(3)))

    # 分别提取步幅、填充和扩张的高度和宽度
    stride_h, stride_w = stride[0], stride[1]
    padding_h, padding_w = padding[0], padding[1]
    dilation_h, dilation_w = dilation[0], dilation[1]
    kernel_h, kernel_w = kernel_size[0], kernel_size[1]

    # 获取沿高度和宽度维度的im2col索引块
    blocks_row_indices = _get_im2col_indices_along_dim(
        g, input_h, kernel_h, dilation_h, padding_h, stride_h
    )
    blocks_col_indices = _get_im2col_indices_along_dim(
        g, input_w, kernel_w, dilation_w, padding_w, stride_w
    )

    # 获取im2col操作的输出形状
    output_shape = _get_im2col_output_shape(g, input, kernel_h, kernel_w)
    # 获取填充后的输入张量用于im2col操作
    padded_input = _get_im2col_padded_input(g, input, padding_h, padding_w)

    # 执行Gather操作，沿着高度和宽度维度收集索引块
    output = g.op("Gather", padded_input, blocks_row_indices, axis_i=2)
    output = g.op("Gather", output, blocks_col_indices, axis_i=4)
    # 对输出进行转置操作，交换深度（dim=3）和行（dim=4），然后根据输出形状进行重塑
    output = g.op("Transpose", output, perm_i=[0, 1, 2, 4, 3, 5])
    return symbolic_helper._reshape_helper(g, output, output_shape)


# 定义一个ONNX符号操作的函数，用于执行narrow操作
@_onnx_symbolic("aten::narrow")
# 使用装饰器检查函数参数类型和返回值类型
@_beartype.beartype
def narrow(g: jit_utils.GraphContext, input, dim, start, length):
    # 计算结束索引，即开始索引加上长度
    end = g.op("Add", start, length)
    # 调用symbolic_helper模块中的_slice_helper函数，用于处理切片操作
    return symbolic_helper._slice_helper(g, input, axes=dim, starts=start, ends=end)
# 声明一个函数，并应用一系列装饰器来定义 ONNX 符号化函数和量化参数
@_onnx_symbolic("aten::flatten")
@symbolic_helper.quantized_args(True, False, False)
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim):
    # 获取输入张量的维度数
    dim = symbolic_helper._get_tensor_rank(input)
    # 如果输入张量已经是一维，则直接返回
    if dim == 1:
        return input
    # 当输出形状为二维时，使用 ONNX 的 Flatten 操作符
    if start_dim == 1:
        if end_dim == -1 or (dim is not None and end_dim == dim - 1):
            return g.op("Flatten", input, axis_i=start_dim)
    # 当 start_dim 为 0 时，对于 end_dim 为 -2 或者与 dim - 2 相等的情况，使用 Flatten 操作符
    elif start_dim == 0:
        if end_dim == -2 or (dim is not None and end_dim == dim - 2):
            return g.op("Flatten", input, axis_i=end_dim + 1)
    # 如果无法确定输入张量的维度数，则返回未实现函数的错误信息
    if dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )
    # 如果 end_dim 是负数，则加上 dim 得到实际的维度索引
    if end_dim < 0:
        end_dim = dim + end_dim

    # 调用辅助函数来实现 flatten 操作
    return symbolic_helper._flatten_helper(g, input, start_dim, end_dim, dim)


# 声明一个函数，并应用一系列装饰器来定义 ONNX 符号化函数和参数解析
@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
@_beartype.beartype
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self,
    ord,
    dim: Optional[Sequence[int]],
    keepdim: bool,
    dtype,
):
    # 调用辅助函数来实现 linalg_vector_norm 操作
    return symbolic_helper._linalg_vector_norm_helper(g, self, ord, dim, keepdim, dtype)


# 声明一个函数，并应用一系列装饰器来定义 ONNX 符号化函数和参数解析
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
    # 调用辅助函数来实现 embedding_bag 操作
    return symbolic_helper._embedding_bag_helper(
        g,
        embedding_matrix,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


# 声明一个函数，并应用一系列装饰器来定义 ONNX 符号化函数和参数解析
@_onnx_symbolic("aten::embedding_renorm")
@symbolic_helper.parse_args("v", "v", "f", "f")
@_beartype.beartype
def embedding_renorm(g: jit_utils.GraphContext, weight, indices, max_norm, norm_type):
    # 使用 ONNX 的 Unique 操作符获取独特的索引
    unique_indices = g.op("Unique", indices)
    # 使用 Gather 操作符从权重中提取部分权重
    partial_weight = g.op("Gather", weight, unique_indices)
    # 将 norm_type 转换为整数
    norm_i = int(norm_type)
    # 根据不同的范数类型选择对应的 ONNX 操作符
    if norm_i == 1:
        norm_type = "ReduceL1"
    elif norm_i == 2:
        norm_type = "ReduceL2"
    else:
        # 如果范数类型不支持，则抛出错误
        raise errors.SymbolicValueError(
            f"Unsupported: ONNX export of embedding_renorm with norm: {norm_i}. "
            "Only 1. and 2. are supported.",
            weight,
        )
    # 使用指定的范数操作符计算部分权重的范数
    partial_weight_norm = g.op(norm_type, partial_weight, axes_i=[1], keepdims_i=1)
    # 添加一个小的常数以防止除以零
    # https://github.com/pytorch/pytorch/blob/0a07488ed2c47765e337e290bd138c0e6e459cbd/aten/src/ATen/native/Embedding.cpp#L177
    # Add 1e-7 to prevent division by zero.
    # 计算部分权重的归一化值，加上一个微小的常数，以避免除以零错误
    partial_weight_norm_ = g.op(
        "Add", partial_weight_norm, g.op("Constant", value_t=torch.tensor(1e-7))
    )
    
    # 将最大范数转换为张量格式
    max_norm = torch.tensor(max_norm)
    
    # 计算归一化系数，将最大范数除以部分权重的范数
    scales = g.op("Div", max_norm, partial_weight_norm_)
    
    # 计算归一化后的部分权重值，乘以归一化系数
    partial_weight_renorm = g.op("Mul", partial_weight, scales)
    
    # 根据条件进行权重的重新赋值，如果部分权重的范数大于最大范数，则使用归一化后的值，否则保持原值
    partial_weight_renorm = g.op(
        "Where",
        g.op("Greater", partial_weight_norm, max_norm),
        partial_weight_renorm,
        partial_weight,
    )
    
    # 使用 ScatterND 操作，根据给定的索引和值更新权重张量的部分值
    return g.op(
        "ScatterND",
        weight,
        symbolic_helper._unsqueeze_helper(g, unique_indices, [1]),
        partial_weight_renorm,
    )
# 注解 onnx_symbolic 装饰器，指定了这个函数在 ONNX 符号化时的操作名称
# 注解 beartype 装饰器，用于类型检查和类型注解
def chunk(g: jit_utils.GraphContext, self, chunks, dim):
    # 计算动态分块的块大小
    dim_size = g.op("Gather", g.op("Shape", self), dim, axis_i=0)
    # 计算 chunk_size_s，chunks 减去常量 1 的结果
    chunk_size_s = g.op(
        "Sub", chunks, g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
    )
    # 计算每个块的大小 chunk_size
    chunk_size = g.op("Div", g.op("Add", dim_size, chunk_size_s), chunks)
    # 创建分块向量
    chunk_vec = [
        opset9.expand(g, chunk_size, chunk_size_s, None),
        g.op("Sub", dim_size, g.op("Mul", chunk_size, chunk_size_s)),
    ]
    # 连接 chunk_vec 中的张量，沿指定维度 axis_i=0 连接
    chunk_vec = g.op("Concat", *chunk_vec, axis_i=0)
    # 调用 split 函数，将 self 按 dim 维度分割成 chunk_vec 中指定大小的块
    return split(g, self, chunk_vec, dim)


# 注解 onnx_symbolic 装饰器，指定了这个函数在 ONNX 符号化时的操作名称
# 注解 beartype 装饰器，用于类型检查和类型注解
def normal(
    g: jit_utils.GraphContext,
    mean,
    std,
    sizes=None,
    generator=None,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    # 如果 sizes 不是 None，则使用 opset9.expand 扩展 mean 的维度
    if sizes is not None and not symbolic_helper._is_none(sizes):
        mean = opset9.expand(g, mean, sizes, None)
    # 使用 RandomNormalLike 操作生成与 mean 形状相同的随机正态分布样本，再乘以 std
    result = opset9.mul(g, std, g.op("RandomNormalLike", mean))
    # 返回 result 与 mean 相加的结果
    return add(g, result, mean)


# 注解 onnx_symbolic 装饰器，指定了这个函数在 ONNX 符号化时的操作名称
# 注解 beartype 装饰器，用于类型检查和类型注解
def atleast_1d(g: jit_utils.GraphContext, self: torch._C.Value):
    # 如果 self 是 0 维，则使用 reshape 转换为 1 维
    if symbolic_helper._is_value(self) and symbolic_helper._is_packed_list(self):
        # 如果 self 是 packed list，则解包成 tensor_list
        tensor_list = symbolic_helper._unpack_list(self)
        new_tensor_list = []
        for tensor in tensor_list:
            new_tensor = tensor
            # 获取 tensor 的秩
            tensor_rank = symbolic_helper._get_tensor_rank(tensor)
            # 如果 tensor 是 0 维，则使用 reshape 转换为 1 维
            if tensor_rank == 0:
                new_tensor = symbolic_helper._reshape_helper(
                    g, new_tensor, g.op("Constant", value_t=torch.tensor([1]))
                )
            new_tensor_list.append(new_tensor)
        # 使用 SequenceConstruct 操作将 new_tensor_list 中的 tensor 合并成一个序列
        return g.op("SequenceConstruct", *new_tensor_list)

    # 获取 self 的秩
    tensor_rank = symbolic_helper._get_tensor_rank(self)
    # 如果 self 是 0 维，则使用 reshape 转换为 1 维
    if tensor_rank == 0:
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([1]))
        )
    return self


# 注解 onnx_symbolic 装饰器，指定了这个函数在 ONNX 符号化时的操作名称
# 注解 beartype 装饰器，用于类型检查和类型注解
def atleast_2d(g: jit_utils.GraphContext, self: torch._C.Value):
    # 如果 self 是 0 维，则使用 reshape 转换为 2 维
    # 如果 self 是 1 维，则使用 unsqueeze 转换为 2 维

    # NOTE: self could be a packed list or a tensor
    # 检查 self 是否是一个值且是一个打包的列表
    if symbolic_helper._is_value(self) and symbolic_helper._is_packed_list(self):
        # 解包列表中的张量
        tensor_list = symbolic_helper._unpack_list(self)
        new_tensor_list = []
        # 遍历每个张量
        for tensor in tensor_list:
            # 复制张量
            new_tensor = tensor
            # 获取张量的秩（维度）
            tensor_rank = symbolic_helper._get_tensor_rank(tensor)
            # 如果张量的秩为 0
            if tensor_rank == 0:
                # 重塑张量为形状为 [1, 1] 的张量
                new_tensor = symbolic_helper._reshape_helper(
                    g, new_tensor, g.op("Constant", value_t=torch.tensor([1, 1]))
                )
            # 如果张量的秩为 1
            elif tensor_rank == 1:
                # 在第 0 维度上增加一个维度
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[0]
                )
            # 将处理过的张量加入新的张量列表中
            new_tensor_list.append(new_tensor)
        # 使用新张量列表构建一个序列
        return g.op("SequenceConstruct", *new_tensor_list)

    # 获取 self 的张量秩（维度）
    tensor_rank = symbolic_helper._get_tensor_rank(self)
    # 如果张量秩为 0
    if tensor_rank == 0:
        # 重塑张量为形状为 [1, 1] 的张量
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([1, 1]))
        )
    # 如果张量秩为 1
    elif tensor_rank == 1:
        # 在第 0 维度上增加一个维度
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[0])
    # 返回处理过的 self
    return self
# 将函数注册为对应的ONNX符号"aten::atleast_3d"
# 使用beartype装饰器确保函数参数类型的有效性检查
@_onnx_symbolic("aten::atleast_3d")
@_beartype.beartype
def atleast_3d(g: jit_utils.GraphContext, self: torch._C.Value):
    # 如果输入张量是0维，则重塑为3维
    # 如果输入张量是1维，则在两端增加维度使其成为3维
    # 如果输入张量是2维，则在最后一个维度上增加维度使其成为3维

    # 检查self可能是一个打包列表或张量
    if symbolic_helper._is_value(self) and symbolic_helper._is_packed_list(self):
        # 解包打包列表
        tensor_list = symbolic_helper._unpack_list(self)
        new_tensor_list = []
        for tensor in tensor_list:
            new_tensor = tensor
            tensor_rank = symbolic_helper._get_tensor_rank(tensor)
            if tensor_rank == 0:
                # 将0维张量重塑为3维
                new_tensor = symbolic_helper._reshape_helper(
                    g, new_tensor, g.op("Constant", value_t=torch.tensor([1, 1, 1]))
                )
            elif tensor_rank == 1:
                # 在0维上增加维度
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[0]
                )
                # 在最后一个维度上增加维度
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[-1]
                )
            elif tensor_rank == 2:
                # 在最后一个维度上增加维度
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[-1]
                )
            new_tensor_list.append(new_tensor)
        # 返回构造的序列张量
        return g.op("SequenceConstruct", *new_tensor_list)

    # 获取self的张量秩
    tensor_rank = symbolic_helper._get_tensor_rank(self)
    if tensor_rank == 0:
        # 将0维张量重塑为3维
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([1, 1, 1]))
        )
    elif tensor_rank == 1:
        # 在0维上增加维度
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[0])
        # 在最后一个维度上增加维度
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[-1])
    elif tensor_rank == 2:
        # 在最后一个维度上增加维度
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[-1])
    return self


# 将函数注册为对应的ONNX符号"prim::ConstantChunk"
# 使用beartype装饰器确保函数参数类型的有效性检查
@_onnx_symbolic("prim::ConstantChunk")
@_beartype.beartype
def prim_constant_chunk(g: jit_utils.GraphContext, self, chunks, dim):
    # 获取输入张量的形状
    input_shape = g.op("Shape", self)
    # 创建一个常量张量，表示指定维度
    axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
    # 使用Gather操作获取输入张量在指定维度上的维度大小
    input_shape_dim = g.op("Gather", input_shape, axis, axis_i=0)
    # 创建一个常量张量，表示起始位置为0
    start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
    # 创建一个常量张量，表示每块的大小
    chunk_size = g.op("Constant", value_t=torch.tensor([chunks], dtype=torch.long))
    # 创建一个常量张量，表示每块的大小减去1
    chunk_size_minus_1 = g.op(
        "Constant", value_t=torch.tensor([chunks - 1], dtype=torch.long)
    )
    # 计算输入张量在指定维度上移动的大小
    input_shape_dim_shift = g.op("Add", input_shape_dim, chunk_size_minus_1)
    # 计算每块的维度大小
    chunk_dim = g.op("Div", input_shape_dim_shift, chunk_size)
    res = []
    for i in range(chunks):
        # 创建一个常量张量，表示索引值
        index = g.op("Constant", value_t=torch.tensor([i + 1], dtype=torch.long))
        # 计算每块的结束位置
        end = g.op("Mul", chunk_dim, index)
        # 切片操作，获取每块的数据
        res.append(g.op("Slice", self, start, end, axis))
        # 更新起始位置
        start = end
    return res


# 将函数注册为对应的ONNX符号"aten::hstack"
# 使用beartype装饰器确保函数参数类型的有效性检查
@_onnx_symbolic("aten::hstack")
@_beartype.beartype
def hstack(g: jit_utils.GraphContext, tensor_list: _C.Value):
    # 调用atleast_1d函数确保输入列表中的所有张量至少是1维
    tensor_list = atleast_1d(g, tensor_list)
    # 使用 ONNX Graph 的操作符 "SequenceAt"，从 tensor_list 中取出第一个张量
    first_tensor = g.op(
        "SequenceAt",
        tensor_list,
        g.op("Constant", value_t=torch.tensor(0, dtype=torch.long)),
    )
    # 获取第一个张量的形状信息
    first_tensor_shape = g.op("Shape", first_tensor)
    # 获取第一个张量的维度数
    first_tensor_dim = g.op("Size", first_tensor_shape)

    # 创建一个常数张量，数值为 1，类型为 long
    const_one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.long))
    # 检查第一个张量的维度数是否等于 1
    equal_to_one = g.op("Equal", first_tensor_dim, const_one)

    # 使用 jit_utils.add_op_with_blocks 函数添加带有条件分支的操作 "If"
    (
        if_op_greater,
        (if_context_equal, else_context_equal),
        _,
    ) = jit_utils.add_op_with_blocks(g, "If", equal_to_one, n_blocks=2, outputs=1)
    # 在 if 分支中执行 "ConcatFromSequence" 操作，沿着第 0 维度连接张量列表 tensor_list
    result_if = if_context_equal.op(
        "ConcatFromSequence", tensor_list, axis_i=0, new_axis_i=0
    )
    # 将结果添加到 if 分支的输出块中
    utils._add_output_to_block(if_context_equal.block, result_if)
    # 在 else 分支中执行 "ConcatFromSequence" 操作，沿着第 1 维度连接张量列表 tensor_list
    result_else = else_context_equal.op(
        "ConcatFromSequence", tensor_list, axis_i=1, new_axis_i=0
    )
    # 将结果添加到 else 分支的输出块中
    utils._add_output_to_block(else_context_equal.block, result_else)
    # 获取整个 if 操作的输出结果
    result = if_op_greater.node().output()

    # 返回整个操作的结果
    return result
# 使用装饰器将函数注册为 ONNX 符号处理函数，处理的符号为 "aten::vstack"
# 使用装饰器 @_beartype.beartype 对函数进行类型检查和验证
@_onnx_symbolic("aten::vstack")
@_beartype.beartype
# 定义函数 vstack，接受两个参数：图上下文 g 和张量列表 tensor_list
def vstack(g: jit_utils.GraphContext, tensor_list: _C.Value):
    # 调用 atleast_2d 函数，确保张量列表 tensor_list 至少是二维的
    tensor_list = atleast_2d(g, tensor_list)
    # 使用图上下文 g 的 op 方法，执行 ConcatFromSequence 操作，沿着 axis_i=0 的方向进行堆叠，并且在新轴 new_axis_i=0 上创建新轴
    return g.op("ConcatFromSequence", tensor_list, axis_i=0, new_axis_i=0)
```