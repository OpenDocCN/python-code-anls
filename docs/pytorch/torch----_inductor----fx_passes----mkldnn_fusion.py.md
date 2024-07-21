# `.\pytorch\torch\_inductor\fx_passes\mkldnn_fusion.py`

```py
# mypy: allow-untyped-defs
# 导入 functools 模块
import functools
# 导入 operator 模块
import operator
# 从 functools 模块导入 reduce 函数
from functools import reduce
# 导入 Any 和 Tuple 类型
from typing import Any, Tuple

# 导入 torch 模块
import torch

# 导入 torch.fx.experimental.symbolic_shapes 模块中的 has_free_symbols 函数
from torch.fx.experimental.symbolic_shapes import has_free_symbols

# 从当前包的上级目录导入 ir 模块
from .. import ir

# 从当前包的 lowerings 模块导入 lowerings 别名 L
from ..lowering import lowerings as L

# 从当前包的 pattern_matcher 模块导入以下内容：
from ..pattern_matcher import (
    Arg,                # 导入 Arg 类
    CallFunction,       # 导入 CallFunction 类
    filter_nodes,       # 导入 filter_nodes 函数
    get_arg_value,      # 导入 get_arg_value 函数
    KeywordArg,         # 导入 KeywordArg 类
    MULTIPLE,           # 导入 MULTIPLE 常量
)

# 从当前包的 virtualized 模块导入 ops 和 V
from ..virtualized import ops, V

# 从当前包的 freezing_patterns 模块导入 register_freezing_graph_pattern 函数
from .freezing_patterns import register_freezing_graph_pattern

# 从当前包的 post_grad 模块导入 register_lowering_pattern 函数
from .post_grad import register_lowering_pattern

# 从当前包的 quantization 模块导入以下函数：
from .quantization import (
    _register_quantization_lowerings,    # 导入 _register_quantization_lowerings 函数
    _register_quantization_weight_pack_pass,   # 导入 _register_quantization_weight_pack_pass 函数
    _register_woq_lowerings,            # 导入 _register_woq_lowerings 函数
)

# 如果 torch._C._has_mkldnn 为真，则定义以下变量：
if torch._C._has_mkldnn:
    aten = torch.ops.aten             # 定义 aten 变量
    mkldnn = torch.ops.mkldnn         # 定义 mkldnn 变量
    prims = torch.ops.prims           # 定义 prims 变量

    # 定义 _conv_args 列表，包含 10 个 Arg 对象
    _conv_args = [Arg() for _ in range(10)]
    # 定义 _linear_args 列表，包含 6 个 Arg 对象
    _linear_args = [Arg() for _ in range(6)]
    # 定义 _conv_transpose_args 列表，包含 11 个 Arg 对象
    _conv_transpose_args = [Arg() for _ in range(11)]

    # 定义 _conv_call 函数，接受 users 参数，默认值为 1
    def _conv_call(users=1):
        return CallFunction(
            mkldnn._convolution_pointwise.default, *_conv_args, _users=users
        )

    # 定义 _linear_call 函数，接受 users 参数，默认值为 1
    def _linear_call(users=1):
        return CallFunction(
            mkldnn._linear_pointwise.default, *_linear_args, _users=users
        )

    # 定义 _conv_transpose_call 函数，接受 users 参数，默认值为 1
    def _conv_transpose_call(users=1):
        return CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=users,
        )

    # 定义 _to_float 函数，接受 input_call 和 users 参数，默认值为 1
    def _to_float(input_call, users=1):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_float"),
            _users=users,
        )

    # 定义 _to_bf16 函数，接受 input_call 参数
    def _to_bf16(input_call):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_bf16"),
            _users=1,
        )

    # 定义 _to_fp16 函数，接受 input_call 参数
    def _to_fp16(input_call):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_fp16"),
            _users=1,
        )

    # 定义 _unary_fusion_pattern 函数，接受 unary_fusion、call_fn、users 和 lowp_dtype 参数
    def _unary_fusion_pattern(unary_fusion, call_fn, users, lowp_dtype):
        # 如果 lowp_dtype 为真，插入 to_float 到 computation_call
        computation_call = (
            _to_float(call_fn(), users=users) if lowp_dtype else call_fn(users=users)
        )
        # 对 unary_fusion 调用 computation_call
        out = unary_fusion(computation_call)
        # 如果 lowp_dtype 是 torch.bfloat16，则返回 _to_bf16(out)
        if lowp_dtype == torch.bfloat16:
            return _to_bf16(out)
        # 如果 lowp_dtype 是 torch.float16，则返回 _to_fp16(out)
        elif lowp_dtype == torch.float16:
            return _to_fp16(out)
        # 否则返回 out
        else:
            return out

    # 定义 _gelu_fusion_1 函数，接受 computation_call 参数
    def _gelu_fusion_1(computation_call):
        # 返回 CallFunction 调用链
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.erf,
                    CallFunction(aten.mul, computation_call, 0.7071067811865476),
                ),
                1,
            ),
        )
    # GELU 激活函数的融合实现，使用以下数学运算按照公式计算 GELU 函数
    def _gelu_fusion_2(computation_call):
        return CallFunction(
            aten.mul,  # 乘法运算：computation_call * 0.5
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,  # 加法运算：tanh(computation_call * (computation_call * computation_call * 0.044715 + 0.7978845608028654)) + 1
                CallFunction(
                    aten.tanh,
                    CallFunction(
                        aten.mul,
                        CallFunction(
                            aten.add,
                            computation_call,
                            CallFunction(
                                aten.mul,
                                CallFunction(
                                    aten.mul,
                                    CallFunction(
                                        aten.mul, computation_call, computation_call
                                    ),
                                    computation_call,
                                ),
                                0.044715,
                            ),
                        ),
                        0.7978845608028654,
                    ),
                ),
                1,
            ),
        )

    # HardSwish 激活函数的融合实现，按照公式计算 HardSwish 函数
    def _hardswish_fusion(computation_call):
        return CallFunction(
            aten.div,  # 除法运算：computation_call * clamp_max(clamp_min(computation_call + 3, 0), 6) / 6
            CallFunction(
                aten.mul,
                computation_call,
                CallFunction(
                    aten.clamp_max,
                    CallFunction(
                        aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                    ),
                    6,
                ),
            ),
            6,
        )

    # SiLU 激活函数的融合实现，按照公式计算 SiLU 函数
    def _silu_fusion(computation_call):
        return CallFunction(
            aten.mul,  # 乘法运算：computation_call * sigmoid(computation_call)
            computation_call,
            CallFunction(aten.sigmoid, computation_call)
        )

    # HardSigmoid 激活函数的融合实现，按照公式计算 HardSigmoid 函数
    def _hardsigmoid_fusion(computation_call):
        return CallFunction(
            aten.div,  # 除法运算：clamp_max(clamp_min(computation_call + 3, 0), 6) / 6
            CallFunction(
                aten.clamp_max,
                CallFunction(
                    aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                ),
                6,
            ),
            6,
        )

    # Leaky ReLU 激活函数的融合实现，按照公式计算 Leaky ReLU 函数
    def _leaky_relu_fusion(computation_call):
        return CallFunction(
            aten.where,  # 条件选择运算：where(computation_call > 0, computation_call, computation_call * negative_slope)
            CallFunction(aten.gt, computation_call, 0),  # 条件：computation_call > 0
            computation_call,  # 如果条件为真，返回 computation_call 本身
            CallFunction(aten.mul, computation_call, KeywordArg("negative_slope")),  # 否则返回 computation_call * negative_slope
        )

    # HardTanh 激活函数的融合实现，按照公式计算 HardTanh 函数
    def _hardtanh_fusion(computation_call):
        return CallFunction(
            aten.clamp_max,  # 上限截断运算：clamp_max(clamp_min(computation_call, min_value), max_value)
            CallFunction(aten.clamp_min, computation_call, KeywordArg("min_value")),  # 下限截断：clamp_min(computation_call, min_value)
            KeywordArg("max_value"),  # 上限值：max_value
        )

    # 组合运算的融合实现，直接调用给定的 elementwise_op 运算
    def _combined_fusion(computation_call, elementwise_op):
        return CallFunction(elementwise_op, computation_call)

    # 二元运算的融合实现，按照给定的 binary_fn 进行二元操作，第一个参数为 "other"，第二个参数为 computation_call
    def _binary_fusion_v1(computation_call, binary_fn):
        return CallFunction(binary_fn, KeywordArg("other"), computation_call)

    # binary_op(computation_op, other)，此处应当补充 binary_op 的具体定义和用途
    # 定义一个函数 `_binary_fusion_v2`，用于创建一个函数调用对象，执行二元函数计算
    def _binary_fusion_v2(computation_call, binary_fn):
        return CallFunction(binary_fn, computation_call, KeywordArg("other"))

    # 定义一个函数 `_is_single_computation_op`，返回一个函数，用于检查匹配对象是否符合特定的计算操作条件
    def _is_single_computation_op(computation_op, lowp_dtype=None):
        def fn(match):
            # 过滤出匹配对象中所有符合 `computation_op` 的计算节点
            computation_nodes = filter_nodes(match.nodes, computation_op)

            # 如果指定了 `lowp_dtype`，则检查输出节点的数据类型是否符合
            if lowp_dtype:
                output_node_meta = match.output_node().meta.get("val")
                if output_node_meta.dtype != lowp_dtype:
                    return False

            # 如果计算节点少于1个，返回 False
            if len(computation_nodes) < 1:
                return False

            # 如果任何一个计算节点的第三个参数不是 "none"，返回 False
            if any(n.args[-3] != "none" for n in computation_nodes):
                return False

            # 否则返回 True，表示匹配成功
            return True

        return fn

    # 定义一个函数 `_is_valid_computation_unary_fusion`，返回一个函数，用于检查匹配对象是否符合有效的单目融合计算条件
    def _is_valid_computation_unary_fusion(computation_op, lowp_dtype=None):
        def fn(match):
            # 检查是否符合单一计算操作的条件
            matched = _is_single_computation_op(computation_op, lowp_dtype)(match)
            # 获取匹配对象中的计算节点
            computation_node = filter_nodes(match.nodes, computation_op)[0]

            # 如果指定了 `lowp_dtype`，则检查转换数据类型节点的数量和类型
            if lowp_dtype:
                conversion_dtype_nodes = filter_nodes(
                    match.nodes, prims.convert_element_type.default
                )
                if len(conversion_dtype_nodes) != 2:
                    return False

                # 融合模式始终是计算操作 + to_float32 + 单目操作 + to_bfloat16 的形式
                if computation_node == conversion_dtype_nodes[0].args[0]:
                    to_float = conversion_dtype_nodes[0].args[1]
                    to_lp = conversion_dtype_nodes[1].args[1]
                else:
                    to_float = conversion_dtype_nodes[1].args[1]
                    to_lp = conversion_dtype_nodes[0].args[1]

                # 检查是否符合要求的数据类型转换
                matched = matched and to_float == torch.float and to_lp == lowp_dtype

            # 返回最终的匹配结果
            return matched

        return fn

    # 定义一个函数 `_register_unary_fusion_lowering`，注册单目融合降低操作的模式和处理函数
    def _register_unary_fusion_lowering(
        pattern, unary_attr, computation_op, lowp_dtype=None
    ):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_unary_fusion(computation_op, lowp_dtype),
        )
        def fn(match, *args, **kwargs):
            # 构造执行计算操作所需的参数列表
            computation_args = list(args)[:-3] + [
                unary_attr.op_name,
                unary_attr.scalars_attr,
                unary_attr.algorithm_attr,
            ]
            return L[computation_op](*computation_args)

        return fn
    def _register_leaky_relu_fusion_lowering(pattern, computation_op, lowp_dtype=None):
        # 定义一个装饰器函数，用于注册一个模式匹配和计算操作的融合降低函数
        @register_lowering_pattern(
            pattern, extra_check=_is_single_computation_op(computation_op, lowp_dtype)
        )
        def fn(match, *args, **kwargs):
            # 从 kwargs 中获取 negative_slope 参数
            negative_slope = kwargs.get("negative_slope")
            # 如果 negative_slope 是 ir.TensorBox 类型，则 matched 为 False
            if isinstance(negative_slope, ir.TensorBox):
                matched = False
            else:  # 否则认为 inp 是一个数值类型
                matched = True
            # 如果指定了 lowp_dtype
            if lowp_dtype:
                # 获取 to_float 和 to_bf16 或 to_fp16 的数据类型
                dtype1 = kwargs.get("to_float")
                dtype2 = (
                    kwargs.get("to_bf16")
                    if lowp_dtype == torch.bfloat16
                    else kwargs.get("to_fp16")
                )
                # 检查匹配条件是否同时满足
                matched = matched and dtype1 == torch.float and dtype2 == lowp_dtype
            # 将参数列表转换为列表形式
            computation_args = list(args)
            # 如果匹配，则修改计算参数列表以执行 leaky_relu 操作
            if matched:
                computation_args = computation_args[:-3] + [
                    "leaky_relu",
                    [negative_slope],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                # 否则执行原始的计算操作
                out = L[computation_op](*computation_args)
                # 如果指定了 lowp_dtype，则将输出转换为 torch.float 类型
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                # 应用 leaky relu 函数
                out = L[aten.where](
                    L[aten.gt](out, 0),
                    out,
                    L[aten.mul](out, negative_slope),
                )
                # 如果指定了 lowp_dtype，则将输出转换为 dtype2 类型
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=dtype2)  # type: ignore[possibly-undefined]
                return out

        return fn
    # 定义一个函数 _register_hardtanh_fusion_lowering，用于注册硬切线函数融合降级模式
    def _register_hardtanh_fusion_lowering(pattern, computation_op, lowp_dtype=None):
        # 在注册的降级模式中，添加额外的检查函数 _is_single_computation_op，用于检查单一计算操作和低精度数据类型
        @register_lowering_pattern(
            pattern, extra_check=_is_single_computation_op(computation_op, lowp_dtype)
        )
        # 定义一个匿名函数 fn，接收匹配对象 match，以及任意数量的位置参数 *args 和关键字参数 **kwargs
        def fn(match, *args, **kwargs):
            # 从关键字参数中获取最小值和最大值
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")
            # 如果最小值或最大值是 TensorBox 类型，则 matched 为 False
            if isinstance(min_value, ir.TensorBox) or isinstance(max_value, ir.TensorBox):
                matched = False
            else:  # 否则，假设输入是一个数字
                assert max_value is not None
                # 检查最小值是否小于等于最大值，结果赋给 matched
                matched = min_value <= max_value
            # 如果设置了低精度数据类型
            if lowp_dtype:
                # 获取类型转换参数
                dtype1 = kwargs.get("to_float")
                dtype2 = (
                    kwargs.get("to_bf16")
                    if lowp_dtype == torch.bfloat16
                    else kwargs.get("to_fp16")
                )
                # 检查 matched 是否为 True，并且 dtype1 是 torch.float，dtype2 是 lowp_dtype
                matched = matched and dtype1 == torch.float and dtype2 == lowp_dtype
            # 复制计算参数列表
            computation_args = list(args)
            # 如果匹配成功
            if matched:
                # 将计算参数列表最后三个元素替换为硬切线函数和其参数
                computation_args = computation_args[:-3] + [
                    "hardtanh",
                    [min_value, max_value],
                    "",
                ]
                # 调用 L[computation_op] 函数，并返回结果
                return L[computation_op](*computation_args)
            else:
                # 否则，执行原始计算操作
                out = L[computation_op](*computation_args)
                # 如果设置了低精度数据类型
                if lowp_dtype:
                    # 将输出转换为 torch.float 类型
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                # 对输出进行最小值和最大值的裁剪
                out = L[aten.clamp_max](L[aten.clamp_min](out, min_value), max_value)
                # 如果设置了低精度数据类型
                if lowp_dtype:
                    # 将输出再次转换为 dtype2 类型
                    out = L[prims.convert_element_type.default](out, dtype=dtype2)  # type: ignore[possibly-undefined]
                # 返回处理后的输出
                return out
    
        # 返回定义的匿名函数 fn
        return fn
    
    # 定义一个字典 _binary_attr，用于存储 aten.add、ops.add、aten.sub 和 ops.sub 对应的字符串值
    _binary_attr = {
        aten.add: "add",
        ops.add: "add",
        aten.sub: "sub",
        ops.sub: "sub",
    }
    def _is_valid_binary(match, fn):
        # 从匹配结果中筛选出二进制节点
        binary_nodes = filter_nodes(match.nodes, fn)
        # 如果没有找到符合条件的二进制节点，返回 False
        if len(binary_nodes) < 1:
            return False

        def get_meta_value(argument: torch.fx.node.Argument):
            # 获取节点的元数据值，预期这些节点是 torch.fx.Node 类型并具有元数据
            if isinstance(argument, torch.fx.Node):
                return argument.meta.get("val", None)
            return None

        # 检查每个二进制节点的参数，确保都是 torch.Tensor 类型
        if any(
            not isinstance(get_meta_value(n.args[0]), torch.Tensor)
            or not isinstance(get_meta_value(n.args[1]), torch.Tensor)
            for n in binary_nodes
        ):
            return False

        # 检查每个二进制节点的 alpha 参数是否为 1.0
        if any(
            get_arg_value(n, 2, kwarg_name="alpha") != 1.0
            and get_arg_value(n, 2, kwarg_name="alpha") is not None
            for n in binary_nodes
        ):
            return False

        # 检查每个二进制节点的参数 tensor 大小、设备和数据类型是否匹配
        if any(
            get_meta_value(n.args[0]).size() != get_meta_value(n.args[1]).size()
            or get_meta_value(n.args[0]).device != get_meta_value(n.args[1]).device
            or get_meta_value(n.args[0]).dtype != get_meta_value(n.args[1]).dtype
            for n in binary_nodes
        ):
            return False

        # 检查每个二进制节点的两个参数是否不相同
        if any(n.args[0] == n.args[1] for n in binary_nodes):
            return False

        # 若所有条件都通过，则返回 True
        return True

    def _is_valid_computation_binary(computation_op, binary_op, other_index=None):
        # 定义检查函数 fn，验证计算操作和二进制操作是否有效
        def fn(match):
            # 检查计算操作是否单个，如果不是，则返回 False
            if not _is_single_computation_op(computation_op)(match):
                return False
            # 检查二进制操作是否有效
            if not _is_valid_binary(match, binary_op):
                return False
            # 若都通过验证，则返回 True
            return True

        # 返回定义的检查函数 fn
        return fn
    def _get_remaining_users(extra_input_node, compute_node):
        # 定义内部函数_is_ancestor_node，用于检查_ancestor_node是否是_current_node的祖先节点
        def _is_ancestor_node(_current_node, _ancestor_node):
            # 初始化节点列表，从_current_node开始
            _node_list = [_current_node]
            # 记录已访问过的节点集合
            _visited_nodes = set()
            # 进行广度优先搜索
            while len(_node_list) != 0:
                _current_node = _node_list.pop(0)
                if _current_node not in _visited_nodes:
                    _visited_nodes.add(_current_node)
                    # 如果找到_ancestor_node，则返回True
                    if _current_node == _ancestor_node:
                        return True
                    # 如果_current_node是torch.fx.Node类型且不是特定操作类型（"placeholder", "output", "get_attr"）
                    elif isinstance(
                        _current_node, torch.fx.Node
                    ) and _current_node.op not in ["placeholder", "output", "get_attr"]:
                        # 将_current_node的所有输入节点加入节点列表中
                        for input in _current_node.all_input_nodes:
                            _node_list.append(input)  # noqa: PERF402，忽略性能警告
            # 如果遍历完仍未找到_ancestor_node，则返回False
            return False

        # 返回所有使用了extra_input_node但不是compute_node祖先节点的用户列表
        return [
            user
            for user in list(extra_input_node.users)
            if not _is_ancestor_node(compute_node, user)
        ]
    def _is_valid_computation_binary_inplace(computation_op, binary_op, other_index):
        # 返回一个函数，该函数用于判断是否可以在原地进行二进制计算
        def fn(match):
            # 如果给定的匹配对象不满足有效的二进制计算条件，则返回 False
            if not _is_valid_computation_binary(computation_op, binary_op)(match):
                return False
            
            # 过滤出所有符合条件的二进制节点
            binary_nodes = filter_nodes(match.nodes, binary_op)

            # 内部函数，用于获取计算节点中的另一个输入节点
            def _get_compute_node(_binary_node, _other_index):
                assert (
                    len(_binary_node.all_input_nodes) == 2
                ), "Binary node should have 2 input nodes."
                _compute_index = 1 if (_other_index == 0) else 0
                return _binary_node.args[_compute_index]

            # 内部函数，用于检查另一个输入是否可以进行原地操作
            def _other_input_not_inplaceable(_binary_node, _other_index):
                _compute_node = _get_compute_node(_binary_node, _other_index)
                return (
                    len(
                        _get_remaining_users(
                            _binary_node.args[_other_index], _compute_node
                        )
                    )
                    > 1
                    or _binary_node.args[_other_index] == _compute_node.args[0]
                )

            # 如果存在任何一个二进制节点的另一个输入不支持原地操作，则返回 False
            if any(_other_input_not_inplaceable(n, other_index) for n in binary_nodes):
                return False
            
            # 如果存在任何一个二进制节点的另一个输入是占位符或输出，则返回 False
            if any(
                n.args[other_index].op in ["placeholder", "output"]
                for n in binary_nodes
            ):
                return False
            
            # 否则，返回 True，表示可以在原地进行二进制计算
            return True

        return fn

    def _register_binary_unary_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        fusion_op,
        unary_attr=None,
    ):
        # 使用装饰器注册下降模式，并附加额外的有效性检查
        @register_lowering_pattern(
            pattern, extra_check=_is_valid_computation_binary(computation_op, binary_op)
        )
        def fn(match, *args, **kwargs):
            # 获取关键字参数中的 "other" 对象，并确保其为 TensorBox 类型
            other = kwargs.get("other")
            assert isinstance(other, ir.TensorBox)
            
            # 获取二进制操作的属性，并组装计算参数列表
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            
            # 如果参数列表长度超过 6，则根据情况添加一些额外的计算参数
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            
            # 调用 L 中的 fusion_op 函数，并传入组装好的计算参数
            return L[fusion_op](*computation_args)

        return fn

    def _can_be_inplace(_other):
        # 如果 _other 的数据类型是 ir.View，则递归调用直至数据类型不再是 ir.View
        if isinstance(_other.data, ir.View):
            return _can_be_inplace(_other.data)
        else:
            # 否则，判断 _other 是否可以原地操作
            return not (
                isinstance(_other.data, ir.ReinterpretView)
                or len(_other.get_inputs_that_alias_output()) > 0
            )
    # 定义一个函数，用于注册二元和一元操作的融合降低函数
    def _register_binary_unary_maybe_inplace_fusion_lowering(
        pattern,  # 融合模式的模式匹配对象
        computation_op,  # 计算操作函数
        binary_op,  # 二元操作函数
        inplace_fusion_op,  # 原地融合操作函数
        outplace_fusion_op,  # 非原地融合操作函数
        unary_attr=None,  # 一元操作的属性对象，默认为None
        other_index=None,  # 其他索引，默认为None
    ):
        # 使用装饰器注册一个降低模式函数
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_binary_inplace(
                computation_op, binary_op, other_index
            ),
        )
        # 定义降低模式函数
        def fn(match, *args, **kwargs):
            other = kwargs.get("other")  # 获取关键字参数中的"other"
            assert isinstance(other, ir.TensorBox)  # 断言"other"是ir.TensorBox类型
            binary_attr = _binary_attr[binary_op]  # 获取二元操作函数对应的属性
            args_list = list(args)  # 将位置参数转换为列表
            # 构建计算所需的参数列表
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:  # 如果位置参数数量大于6
                if unary_attr is not None:  # 如果存在一元操作属性对象
                    # 添加一元操作的属性到计算参数列表中
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    # 如果不存在一元操作属性对象，则添加默认值到计算参数列表中
                    computation_args += [1.0, None, [], None]
            # 确保"other"对象已实现
            other.realize()
            # 如果"other"不能进行原地操作，则调用非原地融合操作函数
            if not _can_be_inplace(other):
                return L[outplace_fusion_op](*computation_args)
            # 否则，调用原地融合操作函数
            return L[inplace_fusion_op](*computation_args)

        return fn  # 返回定义的降低模式函数

    # 定义一组计算操作函数列表
    computation_ops = [
        mkldnn._convolution_pointwise.default,
        mkldnn._linear_pointwise.default,
        mkldnn._convolution_transpose_pointwise.default,
    ]

    # 定义一元操作属性类
    class UnaryAttr:
        def __init__(self, op_name: str, scalars_attr=None, algorithm_attr=None):
            self.op_name = op_name  # 操作名称
            self.scalars_attr = scalars_attr if scalars_attr else []  # 标量属性列表，默认为空列表
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""  # 算法属性，默认为空字符串
    def _register_inplace_fusion():
        # 定义二元操作列表，包括 PyTorch 和操作系统的加法
        binary_ops = [aten.add, ops.add]
        # 定义原位融合操作符为 MKL-DNN 的点卷积二元操作
        inplace_fusion_op = mkldnn._convolution_pointwise_.binary
        # 定义非原位融合操作符为 MKL-DNN 的点卷积二元操作
        outplace_fusion_op = mkldnn._convolution_pointwise.binary
        # 调用 _conv_call 函数，设置 users 参数为 1，得到卷积操作的调用对象
        conv_call = _conv_call(users=1)
        # 获取计算操作列表的第一个元素，作为卷积操作
        conv_op = computation_ops[0]
        # 遍历二元操作列表
        for binary_op in binary_ops:
            # 使用 _binary_fusion_v1 函数对 conv_call 和 binary_op 进行二元融合
            binary_v1 = _binary_fusion_v1(conv_call, binary_op)
            # 对二元融合后的结果和 relu 函数进行组合融合
            binary_unary_v1 = _combined_fusion(binary_v1, aten.relu)
            # 注册可能的原位融合降低操作
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
                unary_attr=UnaryAttr("relu"),
            )
            # 注册可能的非原位融合降低操作
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
            )
            # 使用 _binary_fusion_v2 函数对 conv_call 和 binary_op 进行二元融合
            binary_v2 = _binary_fusion_v2(conv_call, binary_op)
            # 对二元融合后的结果和 relu 函数进行组合融合
            binary_unary_v2 = _combined_fusion(binary_v2, aten.relu)
            # 注册可能的原位融合降低操作
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
                unary_attr=UnaryAttr("relu"),
            )
            # 注册可能的非原位融合降低操作
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
            )

    def _register_binary_fusion():
        # 定义二元操作列表，包括 PyTorch 的加法、减法和操作系统的加法
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        # 定义融合操作列表，包括 MKL-DNN 的点卷积和点线性二元操作
        fusion_ops = [
            mkldnn._convolution_pointwise.binary,
            mkldnn._linear_pointwise.binary,
        ]
        # 获取计算操作用户为 1 的列表，分别包括卷积和线性计算
        _computation_user_1 = [_conv_call(users=1), _linear_call(users=1)]
        # 遍历计算操作和对应的融合操作
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            # 遍历二元操作列表
            for binary_op in binary_ops:
                # 使用 _binary_fusion_v2 函数对 computation_call 和 binary_op 进行二元融合
                pattern = _binary_fusion_v2(computation_call, binary_op)
                # 注册二元-一元融合降低操作
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

            # 针对 PyTorch 的加法，再次使用 _binary_fusion_v1 函数对 computation_call 和 binary_op 进行二元融合
            for binary_op in [aten.add, ops.add]:
                pattern = _binary_fusion_v1(computation_call, binary_op)
                # 注册二元-一元融合降低操作
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )
    # 注册二元和一元操作融合的函数
    def _register_binary_unary_fusion():
        # 定义一组二元操作
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        # 定义融合操作列表
        fusion_ops = [mkldnn._convolution_pointwise.binary]
        # 定义计算用户1的列表
        _computation_user_1 = [_conv_call(users=1)]
        # 遍历计算用户1、计算操作和融合操作的组合
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            # 遍历每个二元操作
            for binary_op in binary_ops:
                # 创建第一种模式的融合
                pattern_v1 = _combined_fusion(
                    _binary_fusion_v2(computation_call, binary_op), aten.relu
                )
                # 注册二元和一元融合的降低操作，指定一元操作为ReLU
                _register_binary_unary_fusion_lowering(
                    pattern_v1,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )
            # 再次遍历每个二元操作（只包括add操作）
            for binary_op in [aten.add, ops.add]:
                # 创建第二种模式的融合
                pattern_v2 = _combined_fusion(
                    _binary_fusion_v1(computation_call, binary_op), aten.relu
                )
                # 注册二元和一元融合的降低操作，指定一元操作为ReLU
                _register_binary_unary_fusion_lowering(
                    pattern_v2,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )

    # 检查是否可打包成MKLDNN RNN层
    def _is_packable_mkldnn_rnn_layer(match):
        # 获取LSTM节点
        lstm_node = match.output_node()
        # 定义权重位置列表
        POS_WEIGHTS = [1, 2]
        # 定义输入位置列表
        POS_INPUTS = [0, 5, 6]
        # 组合所有参数位置
        POS_ARGS = POS_WEIGHTS + POS_INPUTS

        # 检查权重是否为常数
        if any(
            lstm_node.args[POS_WEIGHT].op != "get_attr" for POS_WEIGHT in POS_WEIGHTS
        ):
            return False

        # 检查权重和输入的元信息是否可用
        if any(lstm_node.args[POS_ARG].meta.get("val") is None for POS_ARG in POS_ARGS):
            return False

        # 检查设备类型是否为CPU
        if any(
            lstm_node.args[POS_ARG].meta.get("val").device.type != "cpu"
            for POS_ARG in POS_ARGS
        ):
            return False

        # 检查数据类型是否为torch.bfloat16并且MKLDNN不支持
        if any(
            lstm_node.args[POS_ARG].meta.get("val").dtype == torch.bfloat16
            and not mkldnn._is_mkldnn_bf16_supported()
            for POS_ARG in POS_ARGS
        ):
            return False
        
        # 检查数据类型是否为torch.float16并且MKLDNN不支持
        if any(
            lstm_node.args[POS_ARG].meta.get("val").dtype == torch.float16
            and not mkldnn._is_mkldnn_fp16_supported()
            for POS_ARG in POS_ARGS
        ):
            return False

        # 所有条件满足，可以打包成MKLDNN RNN层
        return True
    def _is_packable_convolution(match):
        """
        Check if the node is supported for MKLDNN convolution.
        """
        # 获取卷积节点
        conv_node = match.output_node()
        # 获取输入和权重的元数据值
        input_meta_value = conv_node.args[0].meta.get("val")
        weight_meta_value = conv_node.args[1].meta.get("val")
        # 如果输入或权重的元数据值为空，则不支持
        if input_meta_value is None or weight_meta_value is None:
            return False
        # 获取输入的形状
        input_size = input_meta_value.shape
        # 如果权重的操作不是 "get_attr"，则不支持
        if conv_node.args[1].op != "get_attr":
            return False
        # 检查输入和权重的元数据
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type != "cpu"
                or (meta_value.dim() != 4 and meta_value.dim() != 5)
            ):
                return False
        # 如果数据类型是 torch.bfloat16，且不支持 MKLDNN 的 bf16，则不支持
        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not mkldnn._is_mkldnn_bf16_supported():
                return False
        # 如果数据类型是 torch.float16，且不支持 MKLDNN 的 fp16，则不支持
        if (
            input_meta_value.dtype == torch.float16
            or weight_meta_value.dtype == torch.float16
        ):
            if not mkldnn._is_mkldnn_fp16_supported():
                return False
        # 检查是否为转置卷积
        is_transposed = conv_node.args[-3]
        if is_transposed:
            # TODO: 支持 MKLDNN 转置卷积的动态形状情况
            if has_free_symbols(input_size):
                return False
            # 获取组数和输入通道数
            groups = conv_node.args[-1]
            in_channels = weight_meta_value.size(0)
            # 不支持 group_depthwise_conv_transpose
            if groups > 1 and groups == in_channels:
                return False
            # 检查输出填充情况和步长
            output_paddings = conv_node.args[-2]
            strides = conv_node.args[3]
            if any(
                output_padding >= stride
                for output_padding, stride in zip(output_paddings, strides)
            ):
                return False
        # 如果以上条件都通过，则支持该卷积操作
        return True
    def _is_packable_linear(match):
        """
        Check if the node is supported for MKLDNN linear.
        """
        # 获取线性节点
        linear_node = match.output_node()
        # mkldnn线性操作仅支持beta为1或0，alpha为1
        if linear_node.target == aten.addmm.default:
            alpha = linear_node.kwargs.get("alpha", 1.0)
            beta = linear_node.kwargs.get("beta", 1.0)
            # 检查alpha和beta的取值是否符合要求
            if (beta != 0.0 and beta != 1.0) or alpha != 1.0:
                return False
        # 对于aten.mm，weight_idx为1；对于aten.addmm，weight_idx为2
        weight_idx = 2 if linear_node.target == aten.addmm.default else 1
        # 检查权重参数是否是get_attr操作
        if linear_node.args[weight_idx].op != "get_attr":
            return False
        # 获取输入和权重的元数据值
        input_meta_value = linear_node.args[weight_idx - 1].meta.get("val")
        weight_meta_value = linear_node.args[weight_idx].meta.get("val")
        # 检查元数据是否存在
        if input_meta_value is None or weight_meta_value is None:
            return False
        # 获取输入的批处理大小
        batch_size = input_meta_value.shape[0]
        # 检查数据类型是否为torch.float64
        if (
            input_meta_value.dtype == torch.float64
            or weight_meta_value.dtype == torch.float64
        ):
            return False
        # 检查权重是否是低精度权重类型
        is_lp_weight = weight_meta_value.dtype in (
            torch.bfloat16,
            torch.float16,
        )
        # 在x86架构上，对于fp32，需要启用mkl并且批处理大小不是自由符号
        # 在aarch64架构上，如果启用了acl，也使用mkldnn操作进行fp32
        if (
            not is_lp_weight
            and not mkldnn._is_mkldnn_acl_supported()
            and ((not torch._C.has_mkl) or has_free_symbols(batch_size))
        ):
            return False
        # 检查输入和权重的元数据
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type != "cpu"
                or meta_value.dim() != 2
            ):
                return False
        # 如果weight_idx为2，检查偏置的元数据
        if weight_idx == 2:
            bias_meta_value = linear_node.args[0].meta.get("val")
            if (
                bias_meta_value is None
                or meta_value.device.type != "cpu"
                or bias_meta_value.dim() != 1
                or bias_meta_value.size(0) != weight_meta_value.size(1)
            ):
                return False

        # 如果数据类型为torch.bfloat16，检查mkldnn是否支持bfloat16
        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not mkldnn._is_mkldnn_bf16_supported():
                return False
        # 如果数据类型为torch.float16，检查mkldnn是否支持float16
        if (
            input_meta_value.dtype == torch.float16
            or weight_meta_value.dtype == torch.float16
        ):
            if not mkldnn._is_mkldnn_fp16_supported():
                return False
        # 如果所有条件满足，则返回True
        return True

    _aten_conv_args = (
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        KeywordArg("is_transposed"),
        Arg(),
        Arg(),
    )
    _aten_mkldnn_rnn_layer_args = (
        Arg(),  # input 输入参数
        Arg(),  # weight0 权重0
        Arg(),  # weight1 权重1
        Arg(),  # weight2 权重2
        Arg(),  # weight3 权重3
        Arg(),  # hx_ 初始隐藏状态
        Arg(),  # cx_ 初始细胞状态
        KeywordArg("reverse"),  # reverse 是否反向
        Arg(),  # batch_sizes 批次大小
        Arg(),  # mode 模式
        Arg(),  # hidden_size 隐藏层大小
        Arg(),  # num_layers 层数
        Arg(),  # has_biases 是否有偏置
        Arg(),  # bidirectional 是否双向
        Arg(),  # batch_first 是否批次优先
        Arg(),  # train 训练模式
    )

    def _eliminate_duplicate_packed_nodes(gm):
        """
        Combine packed weight nodes with the same inputs to reduce memory usage.
        for example:
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32, bias=True)

            def forward(self, x):
                return self.linear(self.linear(x))

        the above's packed weight nodes are duplicate if two linear calls have same input size.
        """
        # 如果未启用MKLDNN或MKLDNN不可用，则直接返回原始图模型
        if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
            return gm

        # 定义MKLDNN支持的打包权重操作列表
        packed_weight_ops = [
            torch._C._nn.mkldnn_reorder_conv2d_weight,
            torch._C._nn.mkldnn_reorder_conv3d_weight,
            mkldnn._reorder_convolution_transpose_weight,
            mkldnn._reorder_linear_weight,
            mkldnn._reorder_mkldnn_rnn_layer_weight,
        ]
        # 如果有MKL支持，添加额外的打包权重操作
        if torch._C.has_mkl:
            packed_weight_ops.append(torch.ops.mkl._mkl_reorder_linear_weight)

        # 遍历图模型中的节点
        for node in gm.graph.nodes:
            # 如果节点的目标在打包权重操作列表中，并且该节点的输入使用次数大于1
            if node.target in packed_weight_ops and len(node.args[0].users) > 1:
                # 遍历所有使用该节点作为输入的用户节点
                for user_node in list(node.args[0].users.keys()):
                    # 如果用户节点的目标与当前节点相同，并且它们不是同一个节点，并且它们的参数相同
                    if (
                        user_node.target == node.target
                        and user_node != node
                        and user_node.args == node.args
                    ):
                        # 用当前节点替换所有使用用户节点的地方
                        user_node.replace_all_uses_with(node)
                        # 从图模型中移除用户节点
                        gm.graph.erase_node(user_node)

    @functools.lru_cache(None)
    def _mkldnn_fusion_init():
        # TODO: aarch64: enable op fusion for acl once it supports fused operators. Disabling it for now.
        # Otherwise even the matmul or innerproduct can not be accelerated with acl
        # 如果MKLDNN可用，并且当前不支持ACL的融合操作，执行以下初始化操作
        if (
            torch.backends.mkldnn.enabled
            and torch.backends.mkldnn.is_available()
            and not torch.ops.mkldnn._is_mkldnn_acl_supported()
        ):
            _register_unary_fusion()  # 注册一元融合操作
            _register_inplace_fusion()  # 注册原地操作融合
            _register_binary_unary_fusion()  # 注册二元一元融合
            _register_binary_fusion()  # 注册二元融合
            _register_quantization_lowerings()  # 注册量化降低
            _register_woq_lowerings()  # 注册无权重量化降低

    @functools.lru_cache(None)
    def _mkldnn_weight_pack_init():
        # 如果MKLDNN可用，则执行以下权重打包的初始化操作
        if torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
            _register_weight_pack_pass()  # 注册权重打包传递
            _recover_linear()  # 恢复线性操作
            _register_quantization_weight_pack_pass()  # 注册量化权重打包传递
```