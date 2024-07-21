# `.\pytorch\torch\onnx\symbolic_opset16.py`

```py
# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 16.

Note [ONNX Operators that are added/updated in opset 16]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-16-of-the-default-onnx-operator-set
New operators:
    GridSample https://github.com/onnx/onnx/pull/3557

Updated operators:
    Identity
    If
    LeakyRelu
    Loop
    PRelu
    RoiAlign
    Scan
    ScatterElements
    ScatterND
    Where
    GreaterOrEqual
    LessOrEqual
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

import functools

import torch
from torch.nn.functional import (
    GRID_SAMPLE_INTERPOLATION_MODES,
    GRID_SAMPLE_PADDING_MODES,
)
from torch.onnx import _type_utils, errors, symbolic_helper, utils
from torch.onnx._internal import _beartype, jit_utils, registration

# Partial function application to register ONNX symbols with opset 16
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=16)


# Function to handle ONNX symbolic registration for 'aten::grid_sampler'
# note (mkozuki): Why `grid_sampler` instead of `grid_sample`?
# Because `torch.nn.functional.grid_sample` calls `torch.grid_sampler`.
@_onnx_symbolic("aten::grid_sampler")
@symbolic_helper.parse_args("v", "v", "i", "i", "b")
@_beartype.beartype
def grid_sampler(
    g: jit_utils.GraphContext,
    input,
    grid,
    mode_enum,
    padding_mode_enum,
    align_corners,
):
    # Check the input and grid tensor rank beforehand.
    if symbolic_helper._get_tensor_rank(input) == 5:
        return symbolic_helper._onnx_unsupported("GridSample with 5D volumetric input")
    
    # Convert mode and padding_mode enums to strings using predefined dictionaries
    mode_s = {v: k for k, v in GRID_SAMPLE_INTERPOLATION_MODES.items()}[mode_enum]  # type: ignore[call-arg]
    padding_mode_s = {v: k for k, v in GRID_SAMPLE_PADDING_MODES.items()}[padding_mode_enum]  # type: ignore[call-arg]
    
    # Create ONNX operation node for 'GridSample' with specified attributes
    return g.op(
        "GridSample",
        input,
        grid,
        align_corners_i=int(align_corners),
        mode_s=mode_s,
        padding_mode_s=padding_mode_s,
    )


# Function to handle ONNX symbolic registration for 'aten::scatter_add'
@_onnx_symbolic("aten::scatter_add")
@symbolic_helper.parse_args("v", "i", "v", "v")
@_beartype.beartype
def scatter_add(g: jit_utils.GraphContext, self, dim, index, src):
    # Determine the type of the 'src' tensor
    src_type = _type_utils.JitScalarType.from_value(
        src, _type_utils.JitScalarType.UNDEFINED
    )
    
    # Retrieve sizes of 'src' and 'index' tensors
    src_sizes = symbolic_helper._get_tensor_sizes(src)
    index_sizes = symbolic_helper._get_tensor_sizes(index)

    # Check if dimensions of 'src' and 'index' tensors match
    if len(src_sizes) != len(index_sizes):
        return symbolic_helper._unimplemented(
            "scatter_add",
            f"`index` ({index_sizes}) should have the same dimensionality as `src` ({src_sizes})",
        )

    # PyTorch only allows 'index' shape <= 'src' shape, manage shapes accordingly
    # to slice 'src' to accommodate.
    # 如果源张量的形状与索引张量的形状不匹配，或者索引张量中包含 None 值
    if src_sizes != index_sizes or None in index_sizes:
        # 使用 ONNX 操作符 "Shape" 获取索引张量的形状并调整
        adjusted_shape = g.op("Shape", index)
        # 创建一个表示起始位置的常量张量，值为 [0, 0, ..., 0]，长度与索引张量维度相同
        starts = g.op("Constant", value_t=torch.tensor([0] * len(index_sizes)))
        # 使用 ONNX 操作符 "Slice" 对源张量进行切片，从 starts 到 adjusted_shape
        src = g.op("Slice", src, starts, adjusted_shape)

    # 调用 symbolic_helper 模块的方法，可能会返回标量值
    src = symbolic_helper._maybe_get_scalar(src)
    # 检查 src 是否为值类型（标量），如果是，则执行 ScatterElements 操作
    if symbolic_helper._is_value(src):
        return g.op("ScatterElements", self, index, src, axis_i=dim, reduction_s="add")
    else:
        # 检查标量值 "src" 的类型是否与 self 相同，如果不同则插入 Cast 节点进行类型转换
        if _type_utils.JitScalarType.from_value(self) != src_type:
            src = g.op(
                "Cast",
                src,
                to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
            )

        # 执行 ScatterElements 操作，将 src 散布到 self 中的指定维度 dim，进行加法归约
        return g.op(
            "ScatterElements",
            self,
            index,
            src,
            axis_i=dim,
            reduction_s="add",
        )
@_onnx_symbolic("aten::scatter_reduce")
@symbolic_helper.parse_args("v", "i", "v", "v", "s", "b")
@_beartype.beartype
def scatter_reduce(
    g: jit_utils.GraphContext,
    self: torch._C.Value,
    dim: int,
    index: torch._C.Value,
    src: torch._C.Value,
    reduce: str,
    include_self: bool,
):
    # 如果 reduce 是 "mean"，则抛出异常，因为 ONNX 不支持 mean 形式的 reduce
    if reduce == "mean":
        raise errors.OnnxExporterError(
            "ONNX does not support mean reduction for scatter_reduce"
        )
    # 如果 include_self 是 False，则抛出异常，因为 ONNX 不支持 include_self=False
    if not include_self:
        raise errors.OnnxExporterError(
            "ONNX does not support include_self=False for scatter_reduce"
        )

    # 定义不同 reduce 模式对应的 ONNX reduce 操作
    reduce_mode = {
        "mean": "none",  # 'mean' 在 ONNX 1.14 定义中不支持
        "sum": "add",
        "prod": "mul",
        "amin": "min",
        "amax": "max",
    }
    onnx_reduce = reduce_mode[reduce]

    # 计算 self 的 rank（维度），用于后续的判断和操作
    self_rank = g.op("Size", g.op("Shape", self))

    # 判断 self 的 rank 是否为 0，生成对应的条件操作符和上下文
    self_rank_is_zero = g.op(
        "Equal", self_rank, g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
    )
    if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
        g, "If", self_rank_is_zero, n_blocks=2, outputs=3
    )
    neg_1 = if_context.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))

    # 在 if 上下文中对 self、index 和 src 进行 Reshape 操作，并添加到对应的块中
    self_reshape = if_context.op("Reshape", self, neg_1)
    utils._add_output_to_block(if_context.block, self_reshape)
    index_reshape = if_context.op("Reshape", index, neg_1)
    utils._add_output_to_block(if_context.block, index_reshape)
    src_reshape = if_context.op("Reshape", src, neg_1)
    utils._add_output_to_block(if_context.block, src_reshape)

    # 在 else 上下文中对 self、index 和 src 进行 Identity 操作，并添加到对应的块中
    self_identity = else_context.op("Identity", self)
    utils._add_output_to_block(else_context.block, self_identity)
    index_identity = else_context.op("Identity", index)
    utils._add_output_to_block(else_context.block, index_identity)
    src_identity = else_context.op("Identity", src)
    utils._add_output_to_block(else_context.block, src_identity)

    # 在图中执行 ScatterElements 操作，使用 if_op 中的条件和结果
    result = g.op("ScatterElements", *if_op, axis_i=dim, reduction_s=onnx_reduce)

    # 再次根据 self_rank 是否为 0 进行条件操作符和上下文的生成
    if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
        g, "If", self_rank_is_zero, n_blocks=2, outputs=1
    )
    # 在 if 上下文中对 result 进行 Squeeze 操作，并添加到对应的块中
    result_squeezed = if_context.op("Squeeze", result)
    utils._add_output_to_block(if_context.block, result_squeezed)
    # 在 else 上下文中对 result 进行 Identity 操作，并添加到对应的块中
    result_identity = else_context.op("Identity", result)
    utils._add_output_to_block(else_context.block, result_identity)
    # 获取 if_op 的最终输出结果
    result_final = if_op.node().output()

    # 返回最终结果
    return result_final
```