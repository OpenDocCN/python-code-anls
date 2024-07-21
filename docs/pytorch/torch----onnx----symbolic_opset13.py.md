# `.\pytorch\torch\onnx\symbolic_opset13.py`

```
# 设置 mypy 以允许未标注的函数定义
# 编辑此文件？请先阅读 README.md 中的“Edit Symbolic Files”部分说明

# 导入必要的库和模块
import functools

import torch
import torch._C._onnx as _C_onnx  # 导入 Torch 的 ONNX 操作模块
from torch.onnx import (
    _constants,           # 导入 ONNX 的常量定义
    _type_utils,          # 导入类型工具函数
    errors,               # 导入错误处理函数
    symbolic_helper,      # 导入符号操作的辅助函数
    symbolic_opset11 as opset11,  # 导入 opset 11 的符号操作
    symbolic_opset9 as opset9,    # 导入 opset 9 的符号操作
    utils,                # 导入工具函数
)
from torch.onnx._internal import _beartype, jit_utils, registration  # 导入内部函数和注册模块

# 定义一个局部函数 _onnx_symbolic，用于设置 opset 为 13 的 ONNX 符号操作
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=13)

# 符号化函数：softmax，对应于 ATen 的 softmax 操作
# 使用装饰器 _onnx_symbolic("aten::softmax")，将该函数注册为 opset 13 的符号操作
@_onnx_symbolic("aten::softmax")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype  # 使用 beartype 进行类型检查
def softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    softmax = g.op("Softmax", input, axis_i=dim)  # 创建 Softmax 操作节点
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        softmax = g.op(
            "Cast", softmax, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )  # 如果存在 dtype，将结果转换为指定的数据类型
    return softmax  # 返回 Softmax 操作节点

# 符号化函数：log_softmax，对应于 ATen 的 log_softmax 操作
# 使用装饰器 _onnx_symbolic("aten::log_softmax")，将该函数注册为 opset 13 的符号操作
@_onnx_symbolic("aten::log_softmax")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype  # 使用 beartype 进行类型检查
def log_softmax(g: jit_utils.GraphContext, input, dim, dtype=None):
    return_op = g.op("LogSoftmax", input, axis_i=dim)  # 创建 LogSoftmax 操作节点
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        return_op = g.op(
            "Cast", return_op, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )  # 如果存在 dtype，将结果转换为指定的数据类型
    return return_op  # 返回 LogSoftmax 操作节点

# 符号化函数：frobenius_norm，对应于 ATen 的 frobenius_norm 操作
# 使用装饰器 _onnx_symbolic("aten::frobenius_norm")，将该函数注册为 opset 13 的符号操作
@_onnx_symbolic("aten::frobenius_norm")
@symbolic_helper.parse_args("v", "v", "i")
@_beartype.beartype  # 使用 beartype 进行类型检查
def frobenius_norm(g: jit_utils.GraphContext, self, dim=None, keepdim=False):
    dim_val = symbolic_helper._maybe_get_const(dim, "is")
    if not symbolic_helper._is_value(dim_val) and len(dim_val) == 0:
        return g.op("ReduceL2", self, keepdims_i=0)  # 如果维度为空，则返回 ReduceL2 操作节点
    sqr = g.op("Mul", self, self)  # 计算 self 和自身的乘积
    sumsqr = symbolic_helper._reducesum_helper(g, sqr, dim, keepdims_i=keepdim)  # 求和操作的辅助函数
    return g.op("Sqrt", sumsqr)  # 返回平方根操作节点

# 符号化函数：split，对应于 ATen 的 split 操作
# 使用装饰器 _onnx_symbolic("aten::split")，将该函数注册为 opset 13 的符号操作
@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
@_beartype.beartype  # 使用 beartype 进行类型检查
def split(g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None):
    # 检查是否需要进行静态分割
    if not symbolic_helper._is_split_static(split_size_or_sizes, _outputs):
        # 如果不是静态分割，使用"SplitToSequence"操作进行分割
        split_out = g.op("SplitToSequence", self, split_size_or_sizes, axis_i=dim)
        # 如果输出数量未知，直接返回分割结果
        if _outputs is None:
            return split_out
        
        # 如果分割大小和输出数量都是静态已知的，转换为多个切片节点
        if (
            symbolic_helper._is_packed_list(split_size_or_sizes)
            and len(symbolic_helper._unpack_list(split_size_or_sizes)) == _outputs
        ):
            # 将分割大小解包并添加未知维度
            split_sizes = [
                symbolic_helper._unsqueeze_helper(g, v, [0])
                for v in symbolic_helper._unpack_list(split_size_or_sizes)
            ]

            # 创建切片操作的起始位置和轴
            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            res = []
            # 遍历每个输出，生成切片操作
            for i in range(_outputs):
                end = g.op(
                    "Add", start, split_sizes[i]
                )  # split_sizes 是一个与 _outputs 长度相同的列表
                res.append(g.op("Slice", self, start, end, axis))
                start = end
            return res
        # 如果无法进行静态切片，返回使用"SequenceAt"操作获取每个输出
        return [
            g.op(
                "SequenceAt",
                split_out,
                g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)),
            )
            for i in range(_outputs)
        ]

    # 如果分割大小是一个节点，获取其值
    split_val = symbolic_helper._node_get(split_size_or_sizes.node(), "value")
    # 如果分割值的维度大于0，使用"Split"操作进行分割
    if split_val.dim() > 0:
        return g.op("Split", self, split_size_or_sizes, axis_i=dim, outputs=_outputs)
    
    # 获取指定维度的大小
    split_size = symbolic_helper._get_const(split_size_or_sizes, "i", "split_size")
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    
    # 如果无法确定维度大小，根据输出数量和分割大小计算总大小
    if size is None:
        if _outputs is not None:
            size = split_size * _outputs
        else:
            raise errors.SymbolicValueError(
                "Unknown dimension size not supported", self
            )
    
    # 计算每个分割的大小和余数
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)
    
    # 将分割大小转换为常量张量，并使用"Split"操作进行分割
    splits = g.op("Constant", value_t=torch.tensor(splits))
    return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)
# 使用装饰器将函数注册为 ONNX 符号化处理函数，处理 PyTorch 中的 aten::split_with_sizes 操作
@_onnx_symbolic("aten::split_with_sizes")
# 使用装饰器进行类型检查和注解
@_beartype.beartype
# 定义函数 split_with_sizes，接受图上下文 g、操作的自身对象 self、分割尺寸 split_sizes、分割维度 dim 和可选输出 _outputs
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    # 调用 split 函数进行实际的分割操作
    return split(g, self, split_sizes, dim, _outputs)


# 使用装饰器将函数注册为 ONNX 符号化处理函数，处理 PyTorch 中的 aten::unsafe_split 操作
@_onnx_symbolic("aten::unsafe_split")
# 使用装饰器进行类型检查和注解
@_beartype.beartype
# 定义函数 unsafe_split，接受图上下文 g、操作的自身对象 self、分割尺寸或尺寸列表 split_size_or_sizes、分割维度 dim 和可选输出 _outputs
def unsafe_split(
    g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None
):
    # 调用 split 函数进行实际的分割操作
    return split(g, self, split_size_or_sizes, dim, _outputs)


# 使用装饰器将函数注册为 ONNX 符号化处理函数，处理 PyTorch 中的 aten::unsafe_split_with_sizes 操作
@_onnx_symbolic("aten::unsafe_split_with_sizes")
# 使用装饰器进行类型检查和注解
@_beartype.beartype
# 定义函数 unsafe_split_with_sizes，接受图上下文 g、操作的自身对象 self、分割尺寸 split_sizes、分割维度 dim 和可选输出 _outputs
def unsafe_split_with_sizes(
    g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None
):
    # 调用 split_with_sizes 函数进行实际的分割操作
    return split_with_sizes(g, self, split_sizes, dim, _outputs)


# 使用装饰器将函数注册为 ONNX 符号化处理函数，处理 PyTorch 中的 aten::tensor_split 操作
@_onnx_symbolic("aten::tensor_split")
# 解析函数参数，并指定参数类型和数量
@symbolic_helper.parse_args("v", "v", "i", "i")
# 使用装饰器进行类型检查和注解
@_beartype.beartype
# 定义函数 tensor_split，接受图上下文 g、操作的自身对象 self、分割位置或分割段数 indices_or_sections、分割维度 dim 和可选输出 _outputs
def tensor_split(
    g: jit_utils.GraphContext, self, indices_or_sections, dim, _outputs=None
):
    # 创建表示分割维度的常量节点
    axis = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.long))
    # 在指定维度上扩展维度为 1
    axis = opset11.unsqueeze(g, axis, 0)
    # 创建值为 1 的常量节点
    const_1 = g.op("Constant", value_t=torch.tensor(1, dtype=torch.long))

    # 如果分割位置或分割段数 indices_or_sections 是静态的
    if symbolic_helper._is_split_static(indices_or_sections, _outputs):
        # 获取分割位置的值
        split_val = symbolic_helper._node_get(indices_or_sections.node(), "value")

        # 如果分割位置是一个张量
        if split_val.dim() > 0:
            # 创建表示起始位置的常量节点
            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            res = []
            # 确保输出数量不为空
            assert _outputs is not None
            # 对于每一个输出，获取分割位置并进行切片操作
            for i in range(_outputs - 1):
                end = g.op(
                    "Gather",
                    indices_or_sections,
                    g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)),
                    axis_i=0,
                )
                res.append(g.op("Slice", self, start, end, axis))
                start = end

            # 获取最后一个输出的结束位置
            end = symbolic_helper._size_helper(g, self, axis)
            res.append(g.op("Slice", self, start, end, axis))
            return res

        # 获取分割大小的常量值
        split_size = symbolic_helper._get_const(
            indices_or_sections, "i", "indices_or_sections"
        )

        # 获取指定维度上的张量大小
        size = symbolic_helper._get_tensor_dim_size(self, dim)
        # 如果大小未知
        if size is None:
            # 如果输出不为空，则计算总大小
            if _outputs is not None:
                size = split_size * _outputs
            else:
                # 否则抛出错误
                raise errors.SymbolicValueError(
                    "Unknown dimension size not supported", self
                )

        # 计算每个分割段最小的分割大小
        min_split_size = size // split_size
        num_splits_one_extra = size % split_size

        # 创建表示分割段的张量
        splits = num_splits_one_extra * [min_split_size + 1]
        leftover = (split_size - num_splits_one_extra) * [min_split_size]

        splits = g.op(
            "Constant", value_t=torch.tensor(splits + leftover, dtype=torch.long)
        )
        # 返回在指定维度上分割的操作
        return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)

    # 如果 indices_or_sections 是张量且其秩为 1
    if (
        symbolic_helper._is_tensor(indices_or_sections)
        and symbolic_helper._get_tensor_rank(indices_or_sections) == 1
        #
        ):
            loop_len = symbolic_helper._size_helper(
                g, indices_or_sections, g.op("Constant", value_t=torch.tensor(0))
            )
            loop_len = opset11.unsqueeze(g, loop_len, 0)
            loop_condition = g.op("Cast", const_1, to_i=_C_onnx.TensorProtoDataType.BOOL)
    
            # To make the first slice in the below loop work,
            # we pad a zero to the first position so that it will be the initial start of slice.
            padding_0 = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            indices_or_sections = g.op("Concat", padding_0, indices_or_sections, axis_i=0)
    
            final_splits = g.op("SequenceEmpty")
            # Loop inputs
            loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
                g, "Loop", loop_len, loop_condition, final_splits, outputs=1, n_blocks=1
            )
    
            loop_block = loop_context.block
            block_input_iter = utils._add_input_to_block(loop_block)
            cond = utils._add_input_to_block(loop_block)
            final_splits = utils._add_input_to_block(loop_block)
    
            start = loop_context.op(
                "Gather", indices_or_sections, block_input_iter, axis_i=0
            )
            end = loop_context.op(
                "Gather",
                indices_or_sections,
                loop_context.op("Add", block_input_iter, const_1),
                axis_i=0,
            )
    
            slice = loop_context.op("Slice", self, start, end, axis)
            final_splits = loop_context.op("SequenceInsert", final_splits, slice)
    
            # Loop outputs
            cond_out = loop_context.op("Identity", loop_condition)
            utils._add_output_to_block(loop_block, cond_out)
            utils._add_output_to_block(loop_block, final_splits)
    
            loop_out = loop.node().output()
            start = g.op(
                "Gather",
                indices_or_sections,
                g.op("Constant", value_t=torch.tensor(-1, dtype=torch.long)),
                axis_i=0,
            )
            start = opset11.unsqueeze(g, start, 0)
            end = symbolic_helper._size_helper(g, self, axis)
    
            last_slice = g.op("Slice", self, start, end, axis)
    
            return g.op("SequenceInsert", loop_out, last_slice)


注释：


            # 循环的入口点
            ):
                # 计算循环的长度
                loop_len = symbolic_helper._size_helper(
                    g, indices_or_sections, g.op("Constant", value_t=torch.tensor(0))
                )
                # 在第0位置添加一个零，确保循环中第一个切片起始位置有效
                padding_0 = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
                indices_or_sections = g.op("Concat", padding_0, indices_or_sections, axis_i=0)
    
                # 初始化一个空序列作为循环的最终结果
                final_splits = g.op("SequenceEmpty")
                
                # 添加带有块的操作，并获取循环的上下文
                loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
                    g, "Loop", loop_len, loop_condition, final_splits, outputs=1, n_blocks=1
                )
    
                loop_block = loop_context.block
                # 向循环块中添加输入
                block_input_iter = utils._add_input_to_block(loop_block)
                cond = utils._add_input_to_block(loop_block)
                final_splits = utils._add_input_to_block(loop_block)
    
                # 获取循环迭代的起始和结束位置
                start = loop_context.op(
                    "Gather", indices_or_sections, block_input_iter, axis_i=0
                )
                end = loop_context.op(
                    "Gather",
                    indices_or_sections,
                    loop_context.op("Add", block_input_iter, const_1),
                    axis_i=0,
                )
    
                # 在当前对象上执行切片操作
                slice = loop_context.op("Slice", self, start, end, axis)
                # 将切片结果插入到最终的分割序列中
                final_splits = loop_context.op("SequenceInsert", final_splits, slice)
    
                # 循环块的输出
                cond_out = loop_context.op("Identity", loop_condition)
                utils._add_output_to_block(loop_block, cond_out)
                utils._add_output_to_block(loop_block, final_splits)
    
                # 获取循环的输出
                loop_out = loop.node().output()
                # 获取最后一个切片的起始位置
                start = g.op(
                    "Gather",
                    indices_or_sections,
                    g.op("Constant", value_t=torch.tensor(-1, dtype=torch.long)),
                    axis_i=0,
                )
                start = opset11.unsqueeze(g, start, 0)
                # 计算当前对象在指定轴上的大小
                end = symbolic_helper._size_helper(g, self, axis)
    
                # 获取最后一个切片
                last_slice = g.op("Slice", self, start, end, axis)
    
                # 将最后一个切片插入到循环输出序列中
                return g.op("SequenceInsert", loop_out, last_slice)
    # 如果不是向量张量（scalar tensor）的情况
    else:  # scalar tensor
        # 使用符号辅助函数获取轴向的尺寸
        dim_size = symbolic_helper._size_helper(g, self, axis)
        # 计算最小分割尺寸，使用除法运算符"Div"
        min_split_size = g.op("Div", dim_size, indices_or_sections)
        # 计算最小分割尺寸加一，使用加法运算符"Add"
        min_split_size_plus_1 = g.op(
            "Add",
            min_split_size,
            const_1,
        )
        # 计算分割余数，使用取模运算符"Mod"
        num_splits_one_extra = g.op("Mod", dim_size, indices_or_sections)
        # 使用"Tile"操作将最小分割尺寸加一复制为指定数量的张量
        splits = g.op("Tile", min_split_size_plus_1, num_splits_one_extra)
        # 使用"Tile"操作将最小分割尺寸复制为指定数量的张量
        leftover = g.op(
            "Tile",
            min_split_size,
            g.op(
                "Sub",
                opset11.unsqueeze(g, indices_or_sections, 0),  # 使用"unsqueeze"扩展维度
                num_splits_one_extra,
            ),
        )

        # 使用"Concat"操作在指定轴向（axis_i=0）上连接分割张量和余数张量
        splits = g.op("Concat", splits, leftover, axis_i=0)
        # 如果没有指定输出（_outputs为None），则使用"SplitToSequence"操作进行分割序列
        if _outputs is None:
            return g.op("SplitToSequence", self, splits, axis_i=dim)
        # 否则，使用"Split"操作进行张量分割，指定轴向（axis_i=dim）和输出
        return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)
# 注释：定义了一个函数 unbind，用于将输入张量按指定维度进行拆分或压缩
@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
def unbind(g: jit_utils.GraphContext, self, dim=0, _outputs=None):
    # 注释：如果没有指定 _outputs 参数，则返回按指定维度 dim 将 self 拆分成序列的操作
    if _outputs is None:
        return g.op(
            "SplitToSequence",
            self,
            g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
            axis_i=dim,
            keepdims_i=0,
        )

    # 注释：创建一个包含多个分片数量的常量张量 splits
    splits = g.op("Constant", value_t=torch.tensor([1] * _outputs))
    # 注释：按指定维度 dim 将 self 拆分成 _outputs 个输出
    outputs = g.op("Split", self, splits, axis_i=dim, outputs=_outputs)
    # 注释：如果 _outputs 为 1，则将 outputs 转换为列表
    outputs = [outputs] if _outputs == 1 else outputs
    # 注释：对每个输出进行挤压操作，去除维度为 1 的维度
    squeezed_outputs = [
        g.op("Squeeze", out, g.op("Constant", value_t=torch.tensor([dim])))
        for out in outputs
    ]
    # 注释：返回挤压后的输出
    return squeezed_outputs


@_onnx_symbolic("aten::nonzero_numpy")
# 注释：对应于 torch.nonzero(x, as_tuple=True) 的符号化处理函数
@_beartype.beartype
def nonzero_numpy(g: jit_utils.GraphContext, input, _outputs=None):
    # 注释：调用 unbind 函数对输入 input 进行符号化处理，生成的张量进行操作
    return unbind(g, opset9.nonzero(g, input), 1, _outputs=_outputs)


@_onnx_symbolic("aten::where")
@symbolic_helper.parse_args("v", "v", "v", "i")
@_beartype.beartype
def where(g: jit_utils.GraphContext, condition, self=None, other=None, _outputs=None):
    # 注释：假设 torch.where 的第一个参数只接受布尔值或字节张量
    if not symbolic_helper._is_bool(condition):
        # 注释：如果条件不是布尔值，则将其转换为布尔张量
        condition = g.op("Cast", condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    if self is None:
        # 注释：如果 self 为空，则对条件进行 nonzero 操作，然后调用 _unbind_helper 函数进行处理
        condition = opset9.nonzero(g, condition)
        return symbolic_helper._unbind_helper(
            g, condition, g.op("Constant", value_t=torch.tensor(1)), _outputs
        )
    # 注释：对条件、self 和 other 进行 Where 操作
    return g.op("Where", condition, self, other)


@_onnx_symbolic("aten::fake_quantize_per_channel_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i")
@_beartype.beartype
def fake_quantize_per_channel_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    axis,
    quant_min=-128,
    quant_max=127,
):
    # 注释：(0, 127) 是一个特殊情况。PyTorch 限制激活值的范围在 (0, 127)。
    # 参考：https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        # 注释：如果 (quant_min, quant_max) 不在允许的范围内，抛出错误
        raise errors.SymbolicValueError(
            "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    # 注释：根据 quant_min 的值，将 zero_point 转换为相应的数据类型
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    # 注释：对输入进行按通道仿射量化操作
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
    # 检查量化的范围是否是 (0, 127)，即量化的最小值和最大值是否分别为 0 和 127
    if (quant_min, quant_max) == (0, 127):
        # 如果是，则对量化后的张量进行裁剪操作，确保值在 [0, 127] 范围内
        quantized = g.op(
            "Clip",
            quantized,  # 待裁剪的量化张量
            opset9.unused(g),  # 不使用的参数，与 Clip 操作相关
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),  # 表示裁剪的上限值
        )
    
    # 返回反量化的线性操作，将裁剪后的量化张量还原成浮点数张量
    return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)
# 使用装饰器定义 ONNX 符号化函数，处理 fake_quantize_per_tensor_affine 操作
@_onnx_symbolic("aten::fake_quantize_per_tensor_affine")
# 使用装饰器解析函数参数，期望参数列表为 "v", "v", "v", "i", "i"
@symbolic_helper.parse_args("v", "v", "v", "i", "i")
# 应用 Beartype 装饰器，用于运行时类型检查和类型提示
@_beartype.beartype
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
):
    # 检查 (quant_min, quant_max) 是否在允许的特殊范围内
    # PyTorch 将激活值限制在 (0, 127) 范围内
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        # 如果不在允许的范围内，抛出符号化数值错误
        raise errors.SymbolicValueError(
            "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    # 如果 quant_min 为 0，将 zero_point 强制转换为 UINT8 类型
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        # 否则，将 zero_point 强制转换为 INT8 类型
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    # 如果 scale 不是 FLOAT 类型，则将其转换为 FLOAT 类型
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    # 对输入数据进行量化，使用 QuantizeLinear 操作
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point)
    # 如果 (quant_min, quant_max) 等于 (0, 127)，则对量化结果进行裁剪
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            "Clip",
            quantized,
            opset9.unused(g),  # 使用未使用的 opset9 引用
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),  # 创建常量张量
        )
    # 对量化后的结果进行反量化，使用 DequantizeLinear 操作
    return g.op("DequantizeLinear", quantized, scale, zero_point)


# 定义一个符号化函数生成器，用于生成具体 ONNX 操作的符号化函数
@_beartype.beartype
def _reduce_op_symbolic(onnx_op_name):
    # 定义符号化函数，接受图形上下文 g、self、dim 和 keepdim 参数
    @_beartype.beartype
    def symbolic(g, self, dim=None, keepdim=None):
        # 将 self 可能的类型转换为减少操作的输入
        self = symbolic_helper._maybe_cast_reduce_op_input(g, self)
        if dim is None:
            # 如果 dim 为 None，则表示执行全局减少路径
            return symbolic_helper._handle_reduce_dim_none(g, self, onnx_op_name)
        else:
            # 否则，获取 keepdim 的常量值，处理特定维度的减少操作
            keepdim = symbolic_helper._get_const(keepdim, "i", "keepdim")
            return g.op(onnx_op_name, self, dim, keepdims_i=keepdim)

    return symbolic


# 使用装饰器定义 ONNX 符号化函数，处理带有数据类型的减少操作
@_onnx_symbolic(
    "aten::sum",
    decorate=[symbolic_helper._apply_params("ReduceSum", "sum")],  # 应用符号化参数
)
# 应用 Beartype 装饰器，用于运行时类型检查和类型提示
@_beartype.beartype
def _reduce_with_dtype(onnx_op, name):
    # 生成指定 ONNX 操作的符号化函数
    symbolic = _reduce_op_symbolic(onnx_op)

    # 根据参数数量重载符号化辅助函数
    @symbolic_helper._overload_by_arg_count
    @_beartype.beartype
    def reduce(g, *args, **kwargs):
        # 定义内部函数 reduce_nodim，用于处理无维度的 reduce 操作
        @symbolic_helper.parse_args("v", "none")
        @_beartype.beartype
        def reduce_nodim(g, self, dtype):
            # 如果 dtype 是 onnx::Constant 类型，则获取其常量值，并转换为对应的 ONNX 数据类型
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            # 如果 dtype 不是 prim::Constant 类型，则返回未实现的错误信息
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype", dtype)
            # 对输入数据进行符号化操作
            result = symbolic(g, self)
            # 如果有指定的 dtype_onnx，检查结果的数据类型是否与之匹配，如果不匹配则进行类型转换
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        # 定义内部函数 reduce_dim，用于处理带维度的 reduce 操作
        @symbolic_helper.parse_args("v", "v", "i", "none")
        @_beartype.beartype
        def reduce_dim(g, self, dim, keepdim, dtype):
            # 如果 dtype 是 onnx::Constant 类型，则获取其常量值，并转换为对应的 ONNX 数据类型
            dtype_onnx = None
            if dtype.node().kind() == "onnx::Constant":
                dtype = symbolic_helper._get_const(dtype, "i", "dtype")
                dtype_onnx = _type_utils.JitScalarType(dtype).onnx_type()
                self = g.op("Cast", self, to_i=dtype_onnx)
            # 如果 dtype 不是 prim::Constant 类型，则返回未实现的错误信息
            elif dtype.node().kind() != "prim::Constant":
                return symbolic_helper._unimplemented(name, "dtype", dtype)
            # 对输入数据进行符号化操作，包括维度参数和是否保留维度参数
            result = symbolic(g, self, dim, keepdim)
            # 如果有指定的 dtype_onnx，检查结果的数据类型是否与之匹配，如果不匹配则进行类型转换
            if dtype_onnx is not None:
                result_dtype_onnx = _type_utils.JitScalarType.from_value(
                    result
                ).onnx_type()
                if result_dtype_onnx != dtype_onnx:
                    result = g.op("Cast", result, to_i=dtype_onnx)
            return result

        # 返回内部函数 reduce_nodim 和 reduce_dim，用于外部调用
        return reduce_nodim, reduce_dim

    # 返回 reduce 函数的引用
    return reduce
# 在注释中提到，此函数用于将 torch 的 aten::unflatten 操作转换为对应的 ONNX 符号操作
# 需要使用 g 参数（图上下文）来构建符号操作
@_onnx_symbolic("aten::unflatten")
@_beartype.beartype
def unflatten(g: jit_utils.GraphContext, input, dim, unflattened_size):
    # 获取输入张量的维度
    input_dim = symbolic_helper._get_tensor_rank(input)
    # 如果无法获取输入张量的维度，则返回未实现的符号错误信息
    if input_dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX 和 PyTorch 在分割输入时使用不同策略。导出时必须知道输入的秩。",
        )

    # 将维度参数 dim 转换为正常的维度索引（可能是负数）
    input_dim = g.op("Constant", value_t=torch.tensor([input_dim], dtype=torch.int64))
    dim = g.op("Add", input_dim, dim)  # 将输入张量的秩与 dim 相加
    dim = g.op("Mod", dim, input_dim)  # 对结果取模，确保维度索引在合理范围内

    # 获取输入张量的形状
    input_size = g.op("Shape", input)

    # 构建头部部分的起始索引和结束索引
    head_start_idx = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
    head_end_idx = g.op(
        "Reshape", dim, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    head_part_rank = g.op("Slice", input_size, head_start_idx, head_end_idx)

    # 计算尾部部分的起始索引和结束索引
    dim_plus_one = g.op(
        "Add", dim, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    tail_start_idx = g.op(
        "Reshape",
        dim_plus_one,
        g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)),
    )
    tail_end_idx = g.op(
        "Constant", value_t=torch.tensor([_constants.INT64_MAX], dtype=torch.int64)
    )
    tail_part_rank = g.op("Slice", input_size, tail_start_idx, tail_end_idx)

    # 构建最终的形状，通过连接头部部分、unflattened_size 和尾部部分
    final_shape = g.op(
        "Concat", head_part_rank, unflattened_size, tail_part_rank, axis_i=0
    )

    # 调用 reshape 辅助函数来重新塑形输入张量
    return symbolic_helper._reshape_helper(g, input, final_shape)


# 在注释中提到，此函数用于将 torch 的 aten::unsafe_chunk 操作转换为对应的 ONNX 符号操作
# 需要使用 g 参数（图上下文）来构建符号操作
@_onnx_symbolic("aten::unsafe_chunk")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def unsafe_chunk(g: jit_utils.GraphContext, self, chunks, dim, _outputs=None):
    # 如果未提供输出参数，则返回将 self 按指定维度 dim 分割成序列的操作
    if _outputs is None:
        return g.op(
            "SplitToSequence",
            self,
            g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
            axis_i=dim,
            keepdims_i=0,
        )

    # 获取指定维度上 self 张量的大小
    size = symbolic_helper._get_tensor_dim_size(self, dim)
    # 如果无法获取维度大小，则返回未实现的符号错误信息
    if size is None:
        return symbolic_helper._unimplemented("unsafe_chunk", "unknown dimension size")

    # 计算每个分块的大小
    split_size = (size + chunks - 1) // chunks
    splits = [split_size] * (size // split_size)
    leftover = size % split_size
    if leftover:
        splits.append(leftover)

    # 将分块大小转换为常量张量
    splits = g.op("Constant", value_t=torch.tensor(splits, dtype=torch.long))
    # 返回将 self 按照 splits 张量在指定维度 dim 上分割的操作
    return g.op("Split", self, splits, axis_i=dim, outputs=_outputs)


@_onnx_symbolic("aten::tile")
@_beartype.beartype
def tile(g: jit_utils.GraphContext, self, dims):
    # 获取 self 张量的形状信息
    self_shape = g.op("Shape", self)
    # 获取 self 张量的维度数
    self_rank = g.op("Size", self_shape)
    # 获取 dims 张量的维度数
    dims_rank = g.op("Size", dims)
    # 计算维度数的差异
    diff = g.op("Sub", self_rank, dims_rank)
    # 创建一个常量张量值为0
    const_zero = g.op("Constant", value_t=torch.tensor([0]))

    # 1. 如果 dims 的长度比 self.shape 短，则用1填充 dims
    dims_shorter_than_self_shape = g.op("Greater", diff, const_zero)
    (
        if_op_greater,
        (if_context_greater, else_context_greater),
        _,
    ) = jit_utils.add_op_with_blocks(
        g, "If", dims_shorter_than_self_shape, n_blocks=2, outputs=1
    )
    # 创建一个常量张量值为1
    const_one = if_context_greater.op("Constant", value_t=torch.LongTensor([1]))
    # 将 diff 重塑为1维张量
    diff_1d_greater = if_context_greater.op("Reshape", diff, const_one)
    # 扩展为与 dims 相同长度的1维张量
    expand_ones_greater = if_context_greater.op("Expand", const_one, diff_1d_greater)
    # 在 axis=0 上连接 expand_ones_greater 和 dims
    dims_ = if_context_greater.op("Concat", expand_ones_greater, dims, axis_i=0)
    # 将结果添加到 if_context_greater 块的输出
    utils._add_output_to_block(if_context_greater.block, dims_)
    # 在 else_context_greater 块中保持 dims 不变
    identity_dim = else_context_greater.op("Identity", dims)
    utils._add_output_to_block(else_context_greater.block, identity_dim)
    # 获取 if_op_greater 块的输出作为最终 dims
    dims_final = if_op_greater.node().output()

    # 2. 如果 dims 的长度比 self.shape 长，则用1填充 self.shape
    dims_longer_than_self_shape = g.op("Less", diff, const_zero)
    (
        if_op_less,
        (if_context_less, else_context_less),
        _,
    ) = jit_utils.add_op_with_blocks(
        g, "If", dims_longer_than_self_shape, n_blocks=2, outputs=1
    )
    # 再次创建一个常量张量值为1
    const_one = if_context_less.op("Constant", value_t=torch.LongTensor([1]))
    # 将 diff 的绝对值重塑为1维张量
    diff_1d_less = if_context_less.op(
        "Reshape",
        if_context_less.op("Abs", diff),
        const_one,
    )
    # 扩展为与 self_shape 相同长度的1维张量
    expand_ones_less = if_context_less.op("Expand", const_one, diff_1d_less)
    # 在 axis=0 上连接 expand_ones_less 和 self_shape
    self_final_shape = if_context_less.op(
        "Concat", expand_ones_less, self_shape, axis_i=0
    )
    # 将 self 重塑为 self_final_shape
    self_ = if_context_less.op("Reshape", self, self_final_shape)
    # 将结果添加到 if_context_less 块的输出
    utils._add_output_to_block(if_context_less.block, self_)
    # 在 else_context_less 块中保持 self 不变
    identity_self = else_context_less.op("Identity", self)
    utils._add_output_to_block(else_context_less.block, identity_self)
    # 获取 if_op_less 块的输出作为最终 self
    self_final = if_op_less.node().output()

    # 将 dims_final 转换为 INT64 类型
    dims_final = g.op("Cast", dims_final, to_i=_C_onnx.TensorProtoDataType.INT64)
    # 使用 Tile 操作将 self_final 沿 dims_final 进行复制
    return g.op("Tile", self_final, dims_final)
    # 如果 repeats_sizes 是 None，则抛出符号值错误异常
    if repeats_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown repeats size.",
            self,
        )
    # 如果 input_sizes 是 None，则抛出符号值错误异常
    if input_sizes is None:
        raise errors.SymbolicValueError(
            "Unsupported: ONNX export of repeat_interleave for unknown input size.",
            self,
        )

    # 将 final_dim 初始化为 dim
    final_dim = dim
    # 如果 dim 是 None，则将输入展平（flatten）
    # 默认情况下，使用展平后的输入数组，并返回一个展平的输出数组
    if symbolic_helper._is_none(dim):
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([-1]))
        )
        dim = torch.tensor(0, dtype=torch.int64)
    else:
        # 尝试将 dim 转换为标量
        dim = symbolic_helper._maybe_get_scalar(dim)

    # 处理 dim 为负数的情况，将其转换为非负数
    if dim < 0:
        dim += len(input_sizes)

    # 复制 input_sizes 到 output_sizes
    output_sizes = input_sizes.copy()
    # 将 input_sizes 中为 None 的元素替换为 (0, -1)
    for idx, input_size in enumerate(input_sizes):
        if input_size is None:
            output_sizes[idx], input_sizes[idx] = 0, -1

    # 检查是否所有的索引都应该重复相同次数
    if repeats_dim == 0 or (repeats_dim == 1 and repeats_sizes[0] == 1):
        # 使用单值重复辅助函数处理
        return symbolic_helper._repeat_interleave_single_value_repeat_helper(
            g, self, repeats, dim
        )

    # 检查输出大小是否为 0 或者重复向量是否为动态的
    cond_dynamic_repeats = repeats_dim == 1 and repeats_sizes[0] is None
    if output_sizes[dim] == 0 or cond_dynamic_repeats:
        # 获取重复次数
        reps = symbolic_helper._size_helper(g, self, dim)
        # 在维度 0 上进行unsqueeze操作
        reps = opset11.unsqueeze(g, reps, 0)

        # 检查 repeats 是否为动态的
        if cond_dynamic_repeats:
            repeat_dim = symbolic_helper._size_helper(
                g, repeats, g.op("Constant", value_t=torch.LongTensor([0]))
            )
            repeat_cond = g.op(
                "Equal", repeat_dim, g.op("Constant", value_t=torch.LongTensor([1]))
            )
            # 使用 where 节点代替 if 语句
            repeats = where(g, repeat_cond, g.op("Expand", repeats, reps), repeats)
    else:
        # 如果不满足上述条件，则调用 opset9.repeat_interleave 函数进行重复插入
        return opset9.repeat_interleave(g, self, repeats, final_dim)

    # 创建一个与 repeats 形状相同的常数张量，值为 [1]
    reps_like = g.op(
        "ConstantOfShape",
        g.op("Shape", repeats),
        value_t=torch.tensor([1], dtype=torch.long),
    )
    # 使用 split 函数将 repeats 和 self 在维度 0 上拆分
    r_splits = split(g, repeats, reps_like, 0)
    i_splits = split(g, self, reps_like, dim)
    # 设置输出尺寸和输入尺寸的初始值
    output_sizes[dim], input_sizes[dim] = -1, 1

    # 创建循环以在每个维度上迭代每个值，并使用重复张量执行单独的交错操作
    # 循环的基本模式如下
    # input (trip_count, cond)
    #   int trip_count = ...;
    #   bool cond = ...;
    #   for (int i=0; i < trip_count && cond; ++i) {
    #     cond = ...;
    #   }

    # 循环条件设置
    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    loop_condition = g.op("Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    loop_len = reps

    # 创建一个空序列以存储最终的扩展结果
    final_splits = g.op("SequenceEmpty")

    # 循环输入设置
    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
        g, "Loop", loop_len, loop_condition, final_splits, n_blocks=1
    )

    loop_block = loop_context.block
    block_input_iter = utils._add_input_to_block(loop_block)
    cond = utils._add_input_to_block(loop_block)
    final_splits = utils._add_input_to_block(loop_block)

    # 从序列中获取分裂的实际值
    r_split = loop_context.op("SequenceAt", r_splits, block_input_iter)
    i_split = loop_context.op("SequenceAt", i_splits, block_input_iter)

    # 在指定维度上添加一个维度
    i_split = opset11.unsqueeze(loop_context, i_split, dim + 1)

    # 创建一个张量列表用于连接
    r_concat = [
        loop_context.op("Constant", value_t=torch.LongTensor(input_sizes[: dim + 1])),
        r_split,
        loop_context.op("Constant", value_t=torch.LongTensor(input_sizes[dim + 1 :])),
    ]
    r_concat = loop_context.op("Concat", *r_concat, axis_i=0)

    # 使用扩展操作扩展输入
    i_split = opset9.expand(loop_context, i_split, r_concat, None)

    # 重新塑造张量
    i_split = symbolic_helper._reshape_helper(
        loop_context, i_split, g.op("Constant", value_t=torch.LongTensor(output_sizes))
    )

    # 将结果插入到最终的分裂序列中
    final_splits = loop_context.op("SequenceInsert", final_splits, i_split)

    # 循环输出设置
    cond_out = loop_context.op(
        "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
    )
    utils._add_output_to_block(loop_block, cond_out)
    utils._add_output_to_block(loop_block, final_splits)

    # 获取循环的输出并按维度连接
    loop_out = loop.node().output()
    loop_out = g.op("ConcatFromSequence", loop_out, axis_i=dim)

    # 返回循环的最终输出
    return loop_out
# 装饰器，将函数标记为对应的ONNX符号化函数
@_onnx_symbolic("aten::diagonal")
# 装饰器，帮助解析函数参数，指定参数类型为(v, i, i, i)
@symbolic_helper.parse_args("v", "i", "i", "i")
# 装饰器，用于类型检查和类型注解
@_beartype.beartype
# 定义函数diagonal，接受图形上下文g、self（表示张量）、偏移量offset、以及两个维度dim1和dim2
def diagonal(g: jit_utils.GraphContext, self, offset, dim1, dim2):
    # 获取张量self的秩（维度数）
    rank = symbolic_helper._get_tensor_rank(self)
    # 当秩已知时，替换负索引值
    if rank is not None:
        dim1 = dim1 if dim1 >= 0 else dim1 + rank
        dim2 = dim2 if dim2 >= 0 else dim2 + rank

    # 使用opset9的size函数计算self在dim1和dim2维度上的尺寸
    dim1_size = opset9.size(
        g, self, dim=g.op("Constant", value_t=torch.LongTensor([dim1]))
    )
    dim2_size = opset9.size(
        g, self, dim=g.op("Constant", value_t=torch.LongTensor([dim2]))
    )

    # 创建适当形状的掩码
    mask_shape = g.op("Concat", dim1_size, dim2_size, axis_i=0)
    mask = opset9.zeros(g, mask_shape, None, None, None)
    mask = g.op("EyeLike", mask, k_i=offset)

    # 若秩已知，将dim1和dim2作为最后一个维度附加到形状中
    if rank is not None:
        axes = list(range(rank))
        axes.remove(dim1)
        axes.remove(dim2)
        self = g.op("Transpose", self, perm_i=axes + [dim1, dim2])
    else:
        # 若秩未知，返回未实现的错误
        return symbolic_helper._unimplemented("diagonal", "unknown input rank")

    # 使用掩码和self张量进行乘法，计算沿对角线的值
    result = g.op("Mul", self, mask)
    # 使用帮助函数_reducesum_helper对结果进行求和降维处理
    result = symbolic_helper._reducesum_helper(g, result, axes_i=[-1], keepdims_i=0)

    # 计算基于偏移和维度的gather索引
    offset_op = g.op("Constant", value_t=torch.LongTensor([offset]))
    if offset >= 0:
        # 如果偏移大于等于零，设置偏移为零以便计算选择窗口
        diag_size = g.op(
            "Max",
            g.op("Min", dim1_size, g.op("Sub", dim2_size, offset_op)),
            g.op("Constant", value_t=torch.LongTensor([0])),
        )
        offset = 0
    else:
        # 否则，计算对角线的尺寸范围
        diag_size = g.op(
            "Max",
            g.op("Min", g.op("Add", dim1_size, offset_op), dim2_size),
            g.op("Constant", value_t=torch.LongTensor([0])),
        )
    diag_size = g.op("Concat", diag_size, axis_i=0)

    # 计算要选择的对角线值
    select_window_ones_fill = opset9.ones(g, diag_size, 4, None, None)
    select_window = g.op(
        "CumSum",
        select_window_ones_fill,
        g.op("Constant", value_t=torch.LongTensor([0])),
    )
    select_window = g.op(
        "Add",
        select_window,
        g.op("Constant", value_t=torch.LongTensor([abs(offset) - 1])),
    )
    # 根据 rank 计算 gather_shape 列表，用于确定 GatherND 操作的参数
    gather_shape = [
        opset9.size(g, result, dim=g.op("Constant", value_t=torch.LongTensor([axis])))
        for axis in list(range(rank))[:-2]  # 遍历除去最后两个维度的所有维度
    ]
    gather_shape.append(diag_size)  # 将 diag_size 添加到 gather_shape 中
    gather_shape = g.op("Concat", *gather_shape, axis_i=0)  # 使用 Concat 操作连接 gather_shape 列表中的元素

    # 创建一个全零张量 gather_indices，用于存储 GatherND 操作的索引
    gather_indices = opset9.zeros(g, gather_shape, 4, None, None)

    # 检查是否存在对角线超出边界的情况
    # 如果 offset 值大于行数或列数，可能会导致对角线越界，使得 diag_size 变为零
    # 例如，如果 offset = 9，dim1_size = 2（列），dim2_size = 4（行）
    # 则 diag_size = max(min(2, (4-9)), 0) = 0，根据上述计算
    # 对角线越界的情况始终导致 diag_size = max(0, 负值) = 0
    # 在没有对角线越界的情况下，我们选择计算对角线值的适当行或列
    # 在对角线越界的情况下，我们返回一个维度为零的张量，因为我们实际上返回了一个空张量
    overrun_cond = g.op(
        "Not",
        g.op(
            "Equal",
            diag_size,
            g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64)),
        ),
    )

    # 根据 overrun_cond 添加带有两个分支的 If 操作
    if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
        g, "If", overrun_cond, n_blocks=2
    )

    # 在 if 分支中计算 gather_indices_if_block，并进行必要的维度调整
    gather_indices_if_block = if_context.op("Add", gather_indices, select_window)
    gather_indices_if_block = symbolic_helper._unsqueeze_helper(
        if_context, gather_indices_if_block, [rank - 1]
    )

    # 在 if 分支中执行 GatherND 操作，获取最终的非越界情况下的结果
    final_non_overrun = if_context.op(
        "GatherND", result, gather_indices_if_block, batch_dims_i=rank - 2
    )

    # 在 else 分支中创建一个与 gather_shape 相同大小的全零张量，表示越界情况下的结果
    final_overrun = opset9.zeros(else_context, gather_shape, 6, None, None)

    # 将结果添加到对应的分支中
    utils._add_output_to_block(if_context.block, final_non_overrun)
    utils._add_output_to_block(else_context.block, final_overrun)

    # 返回 If 操作的结果
    return if_op
# Quantized ops

# 定义量化线性操作的符号化函数
@_onnx_symbolic("quantized::linear")
@_beartype.beartype
def quantized_linear(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    # 对输入进行去量化操作，获取去量化后的输入、输入的缩放因子、以及轴向信息
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 对权重进行去量化操作，获取去量化后的权重、权重的缩放因子、以及轴向信息
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    # 对偏置进行重新量化操作，获取重新量化后的偏置
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    # 对重新量化后的偏置再次进行去量化操作，获取最终的偏置、偏置的缩放因子、以及轴向信息
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 使用线性操作计算输出
    output = opset9.linear(g, input, weight, bias)

    # 对输出进行量化操作，得到量化后的输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 定义包含ReLU的量化线性操作的符号化函数
@_onnx_symbolic("quantized::linear_relu")
@_beartype.beartype
def quantized_linear_relu(
    g: jit_utils.GraphContext, q_input, q_weight, bias, op_scale, op_zero_point
):
    # 对输入进行去量化操作，获取去量化后的输入、输入的缩放因子、以及轴向信息
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 对权重进行去量化操作，获取去量化后的权重、权重的缩放因子、以及轴向信息
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    # 对偏置进行重新量化操作，获取重新量化后的偏置
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    # 对重新量化后的偏置再次进行去量化操作，获取最终的偏置、偏置的缩放因子、以及轴向信息
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 使用线性操作计算输出
    output = opset9.linear(g, input, weight, bias)
    # 对输出进行ReLU操作
    output = opset9.relu(g, output)

    # 对输出进行量化操作，得到量化后的输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 定义包含ReLU的量化一维卷积操作的符号化函数
@_onnx_symbolic("quantized::conv1d_relu")
@_beartype.beartype
def quantized_conv1d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # 对输入进行去量化操作，获取去量化后的输入、输入的缩放因子、以及轴向信息
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 对权重进行去量化操作，获取去量化后的权重、权重的缩放因子、以及轴向信息
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    # 对偏置进行重新量化操作，获取重新量化后的偏置
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    # 对重新量化后的偏置再次进行去量化操作，获取最终的偏置、偏置的缩放因子、以及轴向信息
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 使用一维卷积操作计算输出，并结合ReLU操作
    output = opset9.conv1d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    # 对输出进行量化操作，得到量化后的输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 定义包含ReLU的量化二维卷积操作的符号化函数
@_onnx_symbolic("quantized::conv2d_relu")
@_beartype.beartype
def quantized_conv2d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # 对输入进行去量化操作，获取去量化后的输入、输入的缩放因子、以及轴向信息
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 对权重进行去量化操作，获取去量化后的权重、权重的缩放因子、以及轴向信息
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    # 对偏置进行重新量化操作，获取重新量化后的偏置
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    # 对重新量化后的偏置再次进行去量化操作，获取最终的偏置、偏置的缩放因子、以及轴向信息
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 使用二维卷积操作计算输出，并结合ReLU操作
    output = opset9.conv2d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    # 对输出进行量化操作，得到量化后的输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 定义包含ReLU的量化三维卷积操作的符号化函数
@_onnx_symbolic("quantized::conv3d_relu")
@_beartype.beartype
def quantized_conv3d_relu(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # 对输入进行去量化操作，获取去量化后的输入、输入的缩放因子、以及轴向信息
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 对权重进行去量化操作，获取去量化后的权重、权重的缩放因子、以及轴向信息
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    # 对偏置进行重新量化操作，获取重新量化后的偏置
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    # 对重新量化后的偏置再次进行去量化操作，获取最终的偏置、偏置的缩放因子、以及轴向信息
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 使用三维卷积操作计算输出，并结合ReLU操作
    output = opset9.conv3d(g, input, weight, bias, stride, padding, dilation, groups)
    output = opset9.relu(g, output)

    # 对输出进行量化操作，得到量化后的输出
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
    g: jit_utils.GraphContext,  # g 是一个 GraphContext 对象，可能包含图形计算的上下文信息
    q_input,                    # q_input 是一个变量或参数，表示量化后的输入数据
    q_weight,                   # q_weight 是一个变量或参数，表示量化后的权重数据
    bias,                       # bias 是一个变量或参数，表示神经网络层的偏置项
    stride,                     # stride 是一个变量或参数，表示卷积操作的步幅大小
    padding,                    # padding 是一个变量或参数，表示卷积操作的填充大小
    dilation,                   # dilation 是一个变量或参数，表示卷积操作的扩张大小
    groups,                     # groups 是一个变量或参数，表示卷积操作的分组大小
    op_scale,                   # op_scale 是一个变量或参数，表示量化操作的缩放因子
    op_zero_point,              # op_zero_point 是一个变量或参数，表示量化操作的零点
# 定义 quantized::conv1d 符号操作的函数
@_onnx_symbolic("quantized::conv1d")
# 用装饰器确保 beartype 的类型检查
@_beartype.beartype
def quantized_conv1d(
    # g: jit_utils.GraphContext，表示图上下文对象，用于符号操作
    g: jit_utils.GraphContext,
    # 输入量化张量
    q_input,
    # 权重量化张量
    q_weight,
    # 偏置
    bias,
    # 步幅
    stride,
    # 填充
    padding,
    # 膨胀
    dilation,
    # 组数
    groups,
    # 操作的比例因子
    op_scale,
    # 操作的零点
    op_zero_point,
):
    # 使用符号帮助函数对输入量化张量进行反量化，获取输入数据、缩放因子以及其他信息
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 使用符号帮助函数对权重量化张量进行反量化，获取权重数据、缩放因子以及其他信息
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    # 使用符号帮助函数对偏置进行重新量化，获取重新量化后的偏置张量
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    # 再次使用符号帮助函数对重新量化后的偏置进行反量化，获取最终的偏置数据
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 使用 opset9 库中的 conv1d 函数进行量化卷积操作，得到输出张量
    output = opset9.conv1d(g, input, weight, bias, stride, padding, dilation, groups)

    # 使用符号帮助函数对输出进行量化，使用给定的比例因子和零点
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
    op_scale,  # 操作的缩放因子，用于量化操作
    op_zero_point,  # 操作的零点偏移量，用于量化操作
# 定义量化反卷积操作的符号计算函数，注册为 quantized::conv_transpose2d 的符号计算
@_onnx_symbolic("quantized::conv_transpose2d")
# 使用 beartype 库进行类型检查和注解
@_beartype.beartype
# 定义 quantized_conv_transpose2d 函数，接收以下参数：
# g: 符号图上下文，用于构建符号计算图
# q_input: 量化输入
# q_weight: 量化权重
# bias: 偏置项
# stride: 步长
# padding: 填充
# output_padding: 输出填充
# dilation: 空洞卷积的膨胀率
# groups: 分组数
# op_scale: 操作的量化比例因子
# op_zero_point: 操作的量化零点
def quantized_conv_transpose2d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # 从符号辅助函数中获取解量化后的输入、输入的量化比例因子、以及未使用的占位符
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 从符号辅助函数中获取解量化后的权重、权重的量化比例因子、以及轴信息
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    # 使用符号辅助函数 requantize_bias_helper 处理偏置项
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    # 再次从符号辅助函数中获取解量化后的偏置项、未使用的占位符
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 调用 opset9.conv_transpose2d 执行反卷积操作
    output = opset9.conv_transpose2d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    # 使用符号辅助函数 quantize_helper 对输出进行量化处理，并返回结果
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# 定义量化反卷积3D操作的符号计算函数，注册为 quantized::conv_transpose3d 的符号计算
@_onnx_symbolic("quantized::conv_transpose3d")
# 使用 beartype 库进行类型检查和注解
@_beartype.beartype
# 定义 quantized_conv_transpose3d 函数，接收以下参数：
# g: 符号图上下文，用于构建符号计算图
# q_input: 量化输入
# q_weight: 量化权重
# bias: 偏置项
# stride: 步长
# padding: 填充
# output_padding: 输出填充
# dilation: 空洞卷积的膨胀率
# groups: 分组数
# op_scale: 操作的量化比例因子
# op_zero_point: 操作的量化零点
def quantized_conv_transpose3d(
    g: jit_utils.GraphContext,
    q_input,
    q_weight,
    bias,
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    op_scale,
    op_zero_point,
):
    # 从符号辅助函数中获取解量化后的输入、输入的量化比例因子、以及未使用的占位符
    input, input_scale, _, _ = symbolic_helper.dequantize_helper(g, q_input)
    # 从符号辅助函数中获取解量化后的权重、权重的量化比例因子、以及轴信息
    weight, weight_scale, _, axis = symbolic_helper.dequantize_helper(g, q_weight)
    # 使用符号辅助函数 requantize_bias_helper 处理偏置项
    q_bias = symbolic_helper.requantize_bias_helper(
        g, bias, input_scale, weight_scale, axis
    )
    # 再次从符号辅助函数中获取解量化后的偏置项、未使用的占位符
    bias, _, _, _ = symbolic_helper.dequantize_helper(g, q_bias)

    # 调用 opset9.conv_transpose3d 执行反卷积操作
    output = opset9.conv_transpose3d(
        g, input, weight, bias, stride, padding, output_padding, groups, dilation
    )

    # 使用符号辅助函数 quantize_helper 对输出进行量化处理，并返回结果
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
```