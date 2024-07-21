# `.\pytorch\torch\_inductor\fx_passes\binary_folding.py`

```
# 指定允许未标注的函数定义
# 导入 functools 模块，用于支持 LRU 缓存
# 导入 itertools 模块，用于生成迭代器
import functools
import itertools

# 导入 PyTorch 库
import torch
# 导入计数器工具函数
from ..._dynamo.utils import counters

# 导入模式匹配相关类和函数
from ..pattern_matcher import Arg, CallFunction, KeywordArg
# 导入注册二元折叠模式的函数
from .freezing_patterns import register_binary_folding_pattern

# 设置 aten 和 prims 作为 PyTorch 操作的别名
aten = torch.ops.aten
prims = torch.ops.prims

# 函数用于标记混合数据类型的卷积操作
def mark_mixed_dtype_conv(conv):
    # 获取卷积操作的数据类型
    conv_dtype = conv.meta["val"].dtype
    # 如果数据类型不是 float16 或 bfloat16，则返回
    if conv_dtype not in (torch.float16, torch.bfloat16):
        return

    # 如果卷积操作不止一个用户，则返回
    if not len(conv.users) == 1:
        return

    # 获取卷积操作的用户
    conv_user = next(iter(conv.users.keys()))
    # 如果用户的值不是 torch.Tensor 类型，则返回
    if not isinstance(conv_user.meta["val"], torch.Tensor):
        return

    # 如果用户的数据类型不是 float32，则返回
    if not conv_user.meta["val"].dtype == torch.float32:
        return

    # 循环查找卷积操作的用户，直到找到不是二元操作的节点
    while conv_user.target in _binary_ops:
        # 如果卷积操作的用户不止一个用户，则返回
        if not len(conv_user.users) == 1:
            return
        conv_user = next(iter(conv_user.users.keys()))

    # 如果用户操作不是默认的转换元素类型或者数据类型与卷积操作的数据类型不符，则返回
    if not (
        conv_user.target == prims.convert_element_type.default
        and conv_user.args[1] == conv_dtype
    ):
        return

    # 设置卷积操作的元数据，允许混合数据类型的折叠
    conv.meta["_allow_conv_mixed_dtype_folding"] = conv_dtype


# 函数用于标记允许混合数据类型的卷积操作
def mark_mixed_dtype_allowed_convs(gm):
    """
    遍历图中的节点，标记需要进行混合数据类型折叠的卷积操作
    """
    for node in gm.graph.find_nodes(
        op="call_function", target=aten.convolution.default
    ):
        mark_mixed_dtype_conv(node)


# 函数用于还原被折叠到高精度的卷积操作的原始精度
def recover_original_precision_folded_convs(gm):
    """
    在折叠卷积权重和偏置到更高数据类型之后，恢复它们的原始精度
    """
    graph = gm.graph
    for node in graph.find_nodes(op="call_function", target=aten.convolution.default):
        # 获取允许混合数据类型折叠的卷积操作的原始数据类型
        orig_dtype = node.meta.get("_allow_conv_mixed_dtype_folding", None)
        if orig_dtype is None:
            continue

        # 在节点之前插入操作，将输入转换回原始精度
        with graph.inserting_before(node):
            for idx in [1, 2]:
                old_input = node.args[idx]
                if old_input is None:
                    continue

                # 创建一个新节点，将输入转换为原始数据类型
                new_input = graph.create_node(
                    "call_function",
                    prims.convert_element_type.default,
                    (old_input, orig_dtype),
                )
                node.replace_input_with(old_input, new_input)


# 定义允许与卷积操作进行二元折叠的操作列表
_binary_ops = [aten.add.Tensor, aten.sub.Tensor, aten.mul.Tensor, aten.div.Tensor]


# 使用 functools 提供的 LRU 缓存装饰器，初始化二元折叠
@functools.lru_cache(None)
def binary_folding_init():
    # 创建用于卷积操作的参数列表
    _conv_args = [Arg() for _ in range(9)]
    # 定义计算操作的默认卷积操作
    _computation_ops = [aten.convolution.default]
    # 定义调用函数的默认卷积操作
    _computation_calls = [CallFunction(aten.convolution.default, *_conv_args, _users=1)]

    """
    为了将 add/sub/mul/div 与 conv 融合，其常数张量的维度必须满足以下条件：
    - 通过调整大小，广播到权重/偏置张量的形状
    - 广播到 conv 输出形状
    它需要具有可以调整到权重/偏置张量形状的形状，因为我们需要使用 conv 运行操作
    """
    """
    根据注释，这段代码是用于检查张量是否符合与卷积输出不广播的条件。以下是详细解释：

    def _op_not_broadcasting_with_conv(weight_tensor, other_tensor):
        # 根据 frozen_conv_folding.cpp 中的 opDoesNotBroadCastWithConv 函数
        # 获取权重张量和另一个张量的形状
        weight_shape = weight_tensor.shape
        other_shape = other_tensor.shape

        # 如果权重张量的维度少于另一个张量，则不符合条件
        if len(weight_shape) < len(other_shape):
            return False

        # 如果权重张量的维度比另一个张量多1，则进行以下判断
        if len(weight_shape) == len(other_shape) + 1:
            # 权重形状是 [o, i, *]，另一个张量形状是 [o, 1...] 的情况
            for i in reversed(range(len(other_shape))):
                # 第一个维度（通常是输出通道数）必须匹配
                if i == 0 and weight_shape[0] == other_shape[i]:
                    continue
                # 其他维度必须为1，否则不符合条件
                if other_shape[i] != 1:
                    return False
        else:
            # 权重形状是 [o, i, *]，另一个张量形状是 [1, i, *] 的情况
            for i in reversed(range(len(other_shape))):
                # 第二个维度（通常是输入通道数）必须匹配
                if i == 1 and weight_shape[0] == other_shape[i]:
                    continue
                # 其他维度必须为1，否则不符合条件
                if other_shape[i] != 1:
                    return False
        
        # 如果通过以上条件判断，则返回 True，表示不会与卷积输出进行广播
        return True
    """
    # 检查卷积和广播操作的前置条件，根据 frozen_conv_folding.cpp 中的 checkConvAndBroadcastingOpPreConditions 函数定义。
    def _check_conv_and_broadcast_op(conv_node, other):
        # 检查卷积操作的权重参数 conv.weight 是否是通过 get_attr 操作获取
        if conv_node.args[1].op != "get_attr":
            return False
        # 检查卷积操作的偏置参数 conv.bias 是否存在且通过 get_attr 操作获取
        if conv_node.args[1] is not None and conv_node.args[1].op != "get_attr":
            return False
        # 如果 other 不是整数、浮点数或通过 get_attr 操作获取，则返回 False
        if (
            not isinstance(other, int)
            and not isinstance(other, float)
            and other.op != "get_attr"
        ):
            return False

        # 检查权重参数 conv.weight 是否仅被一个用户使用
        if not len(conv_node.args[1].users) == 1:
            return False

        # 获取权重参数 conv.weight 的元数据值
        weight_meta_value = conv_node.args[1].meta.get("val")
        if weight_meta_value is None:
            return False
        # 避免融合会导致类型提升的操作，限制在浮点数上以避免标量重载的整数/浮点困难
        if not weight_meta_value.is_floating_point():
            return False

        # 如果 other 是 torch.fx.Node 且通过 get_attr 操作获取，则进一步检查
        if isinstance(other, torch.fx.Node) and other.op == "get_attr":
            other_meta_value = other.meta.get("val")
            # 如果 other 的值不是浮点数类型，则返回 False
            if not other_meta_value.is_floating_point():
                return False
            # 如果 other 的数据类型提升不等于权重参数的数据类型，则根据条件进一步检查
            if (
                torch.promote_types(other_meta_value.dtype, weight_meta_value.dtype)
                != weight_meta_value.dtype
            ):
                # 如果未允许卷积混合数据类型融合，则返回 False
                if not conv_node.meta.get("_allow_conv_mixed_dtype_folding", False):
                    return False

                # 如果 other 的数据类型不是 torch.float，且权重参数的数据类型不是 torch.float16 或 torch.bfloat16，则返回 False
                if (
                    other_meta_value.dtype != torch.float
                    and weight_meta_value.dtype not in (torch.float16, torch.bfloat16)
                ):
                    return False

            # 检查权重参数和 other 的广播规则，若不符合则返回 False
            if not _op_not_broadcasting_with_conv(weight_meta_value, other_meta_value):
                return False
        else:
            # TODO: 支持标量情况，当前暂时返回 False
            return False

        # 若以上所有条件均满足，则返回 True，表示可融合
        return True

    # 检查是否匹配可折叠的模式
    def _is_foldable_pattern(match):
        # 获取匹配的二元节点
        binary_node = match.output_node()
        # 获取计算节点和其他节点
        computation_node = binary_node.args[0]
        other = binary_node.args[1]
        # 如果二元节点的第一个参数不是计算操作，则交换计算节点和其他节点
        if binary_node.args[0].target not in _computation_ops:
            computation_node = binary_node.args[1]
            other = binary_node.args[0]
        # 如果二元节点的计算目标是默认的卷积操作，则调用 _check_conv_and_broadcast_op 函数进行进一步检查
        if binary_node.args[0].target == aten.convolution.default:
            return _check_conv_and_broadcast_op(computation_node, other)

        # 若不符合卷积操作的条件，则返回 False
        return False
    def resize_scalar_or_tensor_to_shape(graph, other, shape):
        # TODO: support scalar case
        # 检查输入张量或标量是否为单个元素
        if other.meta.get("val").numel() == 1:
            # 如果是标量，将其转换成形状为 (1,) 的张量
            res = graph.create_node(
                "call_function",
                aten.reshape.default,
                (other, (1,)),
            )
            # 使用给定的形状扩展张量
            res = graph.create_node(
                "call_function",
                aten.expand.default,
                (res, shape),
            )
        else:
            # 如果是张量，直接按给定形状进行reshape操作
            res = graph.create_node(
                "call_function",
                aten.reshape.default,
                (other, shape),
            )
        return res

    def _create_new_conv_node(graph, conv_node, binary_node, other):
        assert conv_node.target == aten.convolution.default
        conv_args = list(conv_node.args)
        weight_meta_value = conv_node.args[1].meta.get("val")
        bias = conv_args[2]
        if binary_node.target in [aten.add.Tensor, aten.sub.Tensor]:
            # 将标量或张量调整为与卷积权重相匹配的形状
            other_reshape = resize_scalar_or_tensor_to_shape(
                graph, other, (weight_meta_value.size(0),)
            )
            # 创建新的偏置节点，使用二元操作处理
            new_bias = graph.create_node(
                "call_function",
                binary_node.target,
                (0 if bias is None else bias, other_reshape),
            )
            conv_args[2] = new_bias
        else:
            assert binary_node.target in [aten.mul.Tensor, aten.div.Tensor]
            # 将标量或张量调整为与卷积权重广播兼容的形状
            weight_broadcast_shape = [1 for _ in range(len(weight_meta_value.shape))]
            weight_broadcast_shape[0] = weight_meta_value.size(0)
            other_reshape1 = resize_scalar_or_tensor_to_shape(
                graph, other, tuple(weight_broadcast_shape)
            )
            # 创建新的权重节点，使用二元操作处理
            new_weight = graph.create_node(
                "call_function", binary_node.target, (conv_args[1], other_reshape1)
            )
            new_weight.meta.update(conv_args[1].meta)
            conv_args[1] = new_weight
            if bias is not None:
                # 将标量或张量调整为与卷积权重相匹配的形状
                other_reshape = resize_scalar_or_tensor_to_shape(
                    graph, other, (weight_meta_value.size(0),)
                )
                # 创建新的偏置节点，使用二元操作处理
                new_bias = graph.create_node(
                    "call_function", binary_node.target, (bias, other_reshape)
                )
                new_bias.meta.update(bias.meta)
                conv_args[2] = new_bias
        # 创建新的卷积节点，将处理后的参数传递给卷积操作
        return graph.create_node("call_function", conv_node.target, tuple(conv_args))

    # 遍历所有计算调用和二元操作的组合
    for _computation_call, binary_op in itertools.product(
        _computation_calls, _binary_ops
        ):

            @register_binary_folding_pattern(
                # 注册一个二元折叠模式，用于匹配特定的函数调用模式
                CallFunction(binary_op, _computation_call, KeywordArg("other")),
                # 额外的检查函数，确保匹配的模式可折叠
                extra_check=_is_foldable_pattern,
            )
            # 定义折叠操作的函数
            def folded_op(match, *args, **kwargs):
                # 计数器统计，增加二元折叠计数
                counters["inductor"]["binary_folding"] += 1
                # 从关键字参数中获取 'other' 参数
                other = kwargs.get("other")
                # 获取匹配对象的输出节点
                binary_node = match.output_node()
                # 确定计算节点，根据目标是不是在计算操作列表中的一个
                computation_node = (
                    binary_node.args[0]
                    if binary_node.args[0].target in _computation_ops
                    else binary_node.args[1]
                )
                # 获取匹配的图对象
                graph = match.graph
                # 在二元节点之前插入新的节点
                with graph.inserting_before(binary_node):
                    # TODO: support linear?
                    # 确保计算节点的目标是默认的卷积操作
                    assert computation_node.target == aten.convolution.default
                    # 创建一个新的卷积节点来替换原始的二元节点
                    new_computation_node = _create_new_conv_node(
                        graph, computation_node, binary_node, other
                    )
                    # 更新新计算节点的元数据
                    new_computation_node.meta.update(computation_node.meta)
                    # 移除原始的二元节点和计算节点
                    graph.erase_node(binary_node)
                    graph.erase_node(computation_node)
```