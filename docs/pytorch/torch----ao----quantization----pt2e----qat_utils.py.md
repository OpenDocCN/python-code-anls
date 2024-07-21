# `.\pytorch\torch\ao\quantization\pt2e\qat_utils.py`

```
# Type hinting: allow untyped definitions in mypy
mypy: allow-untyped-defs

# 导入必要的模块和函数
import dataclasses                  # 用于数据类（data classes）
import itertools                    # 提供迭代工具的函数
import operator                     # 提供标准操作符的函数
from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING  # 引入类型提示

import torch                        # PyTorch 主库
from torch.fx import Graph, GraphModule, Node  # PyTorch FX 模块相关类和函数
from torch.fx.subgraph_rewriter import (
    replace_pattern_with_filters,
    ReplacedPatterns,
)                                   # 用于子图重写的相关函数
import torch.nn.functional as F     # PyTorch 的神经网络函数
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # 量化相关的库，忽略 F401 警告
from torch.ao.quantization.pt2e.export_utils import _WrapperModule  # 用于导出工具的封装模块
from torch.ao.quantization.quantizer import (
    DerivedQuantizationSpec,
    EdgeOrNode,
    SharedQuantizationSpec,
    QuantizationSpecBase,
)                                   # 量化相关的规范和类

from .utils import (
    _conv1d_bn_example_inputs,
    _conv2d_bn_example_inputs,
    _is_bn_node,
    _is_conv_or_conv_transpose_node,
    _is_conv_transpose_fn,
    fold_bn_weights_into_conv_node,
    _get_aten_graph_module_for_pattern,
)                                   # 导入本地的工具函数

if TYPE_CHECKING:
    from torch.fx.passes.utils.matcher_with_name_node_map_utils import InternalMatch
                                    # 如果是类型检查阶段，则导入特定的内部匹配类

__all__ = []  # type: ignore[var-annotated]  # 导出时不包含任何内容的空列表


# 用于 quantized_conv1d_bn 模式的示例输入
_quantized_conv1d_bn_example_inputs = (
    torch.randn(1, 1, 3),   # 输入 x
    torch.randn(1, 1, 1),   # 卷积权重 conv_weight
    torch.randn(1),         # 批归一化权重 bn_weight
    torch.randn(1),         # 批归一化偏置 bn_bias
    torch.randn(1),         # 批归一化运行均值 bn_running_mean
    torch.randn(1),         # 批归一化运行方差 bn_running_var
)

# 用于 quantized_conv2d_bn 模式的示例输入
_quantized_conv2d_bn_example_inputs = (
    torch.randn(1, 1, 3, 3),   # 输入 x
    torch.randn(1, 1, 1, 1),   # 卷积权重 conv_weight
    torch.randn(1),            # 批归一化权重 bn_weight
    torch.randn(1),            # 批归一化偏置 bn_bias
    torch.randn(1),            # 批归一化运行均值 bn_running_mean
    torch.randn(1),            # 批归一化运行方差 bn_running_var
)


def _get_quantized_conv_bn_example_inputs_kwargs(
    is_per_channel: bool,
    has_bias: bool,
    bias_is_quantized: bool,
    is_cuda: bool,
) -> Dict[str, Any]:
    """
    获取用于 quantized conv-bn 模式的示例输入参数，作为关键字参数返回。
    """
    kwargs = {}  # 初始化空字典用于存储参数

    # 如果是按通道量化，则使用字面量表示比例和零点，不需要在这里作为参数包含
    if is_per_channel:
        kwargs["weight_scale"] = torch.tensor([1], dtype=torch.float)
        kwargs["weight_zero_point"] = torch.tensor([0], dtype=torch.int)
        if has_bias and bias_is_quantized:
            kwargs["bias_scale"] = torch.tensor([1], dtype=torch.float)
            kwargs["bias_zero_point"] = torch.tensor([0], dtype=torch.int)
    
    # 如果存在偏置，则添加卷积的偏置
    if has_bias:
        kwargs["conv_bias"] = torch.randn(1)
    
    # 如果是在 CUDA 上，则将所有参数转移到 GPU 上
    if is_cuda:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.cuda()
    
    return kwargs  # 返回关键字参数字典


def _get_conv_bn_pattern(conv_fn: Callable) -> Callable:
    # 定义一个函数 _conv_bn_pattern，接收多个张量作为输入，并返回一个张量作为输出
    def _conv_bn_pattern(
        x: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ) -> torch.Tensor:
        # 调用 conv_fn 函数，对输入张量 x 进行卷积操作，使用给定的卷积权重和偏置
        x = conv_fn(x, conv_weight, conv_bias)
        # 使用 F.batch_norm 函数进行批归一化操作，对卷积结果 x 进行处理
        # 使用给定的批归一化权重、偏置、和累积统计值（均值和方差），并在训练模式下执行
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True)
        # 返回经过卷积和批归一化处理后的张量 x
        return x
    # 返回一个 _WrapperModule 包装后的 _conv_bn_pattern 函数
    return _WrapperModule(_conv_bn_pattern)
# 将此函数与 `no_conv_bias` 情况合并
def _get_qat_conv_bn_pattern(conv_fn: Callable) -> Callable:
    # 返回一个内部函数 `_qat_conv_bn_pattern`，用于量化训练中融合卷积和批归一化操作
    def _qat_conv_bn_pattern(
        x: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        近似融合卷积和批归一化的方法。只需要进行一次前向传播。
        conv_orig = conv / scale_factor，其中 scale_factor = bn.weight / sqrt(bn.running_var + bn_eps)。
        基于 `nniqat.ConvBn2d._forward_approximate` 实现。
        """
        # TODO: 允许设置 eps
        bn_eps = 1e-5
        running_std = torch.sqrt(bn_running_var + bn_eps)
        scale_factor = bn_weight / running_std
        weight_shape = [1] * len(conv_weight.shape)
        weight_in_channel_axis = 1 if _is_conv_transpose_fn(conv_fn) else 0
        weight_shape[weight_in_channel_axis] = -1
        bias_shape = [1] * len(conv_weight.shape)
        bias_shape[1] = -1
        scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
        zero_bias = torch.zeros_like(conv_bias, dtype=x.dtype)
        x = conv_fn(x, scaled_weight, zero_bias)  # 执行卷积操作，使用缩放后的权重和零偏置
        x = x / scale_factor.reshape(bias_shape)  # 对卷积结果进行缩放
        x = x + conv_bias.reshape(bias_shape)  # 加上卷积的偏置
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)  # 执行批归一化
        return x
    return _WrapperModule(_qat_conv_bn_pattern)

# 获取不包含卷积偏置项的量化训练中的卷积批归一化模式
def _get_qat_conv_bn_pattern_no_conv_bias(conv_fn: Callable) -> Callable:
    # 返回一个内部函数 `_qat_conv_bn_pattern_no_conv_bias`，处理不含卷积偏置项的情况
    def _qat_conv_bn_pattern_no_conv_bias(
        x: torch.Tensor,
        conv_weight: torch.Tensor,
        # 不使用，仅用于匹配方便
        conv_bias: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        与 `_get_qat_conv_bn_pattern` 相同，但处理不含卷积偏置项的情况。
        """
        # TODO: 允许设置 eps
        bn_eps = 1e-5
        running_std = torch.sqrt(bn_running_var + bn_eps)
        scale_factor = bn_weight / running_std
        weight_shape = [1] * len(conv_weight.shape)
        weight_in_channel_axis = 1 if _is_conv_transpose_fn(conv_fn) else 0
        weight_shape[weight_in_channel_axis] = -1
        bias_shape = [1] * len(conv_weight.shape)
        bias_shape[1] = -1
        scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
        x = conv_fn(x, scaled_weight, None)  # 执行卷积操作，不使用卷积偏置
        x = x / scale_factor.reshape(bias_shape)  # 对卷积结果进行缩放
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=True, eps=bn_eps)  # 执行批归一化
        return x
    return _WrapperModule(_qat_conv_bn_pattern_no_conv_bias)

# 辅助函数，在 `x` 后追加量化-反量化操作，使用虚拟值进行量化参数
def _append_qdq(x, is_per_channel, is_bias, kwargs):
    """
    辅助函数，在 `x` 后追加量化-反量化操作，使用虚拟值进行量化参数
    """
    """
    Dummy args to be passed into q-dq ops
    per_channel_axis = 0
    scale_key = "bias_scale" if is_bias else "weight_scale"
    zp_key = "bias_zero_point" if is_bias else "weight_zero_point"
    scale = kwargs[scale_key] if is_per_channel else 1.0
    zp = kwargs[zp_key] if is_per_channel else 0
    qmin = -127  # 定义量化的最小值
    qmax = 127   # 定义量化的最大值
    dtype = torch.int8  # 定义数据类型为 int8
    
    qd = torch.ops.quantized_decomposed  # 获取量化解析运算的操作符
    
    if is_per_channel:
        # 如果是按通道量化，则调用按通道量化和反量化的操作
        x = qd.quantize_per_channel(x, scale, zp, per_channel_axis, qmin, qmax, dtype)
        x = qd.dequantize_per_channel(x, scale, zp, per_channel_axis, qmin, qmax, dtype)
    else:
        # 如果是按张量量化，则调用按张量量化和反量化的操作
        x = qd.quantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
        x = qd.dequantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
    return x
    """
# 返回一个函数，该函数实现了量化 QAT（Quantization Aware Training）的卷积 + BN 模式，其中 BN 权重被折叠到卷积中
def _get_folded_quantized_qat_conv_bn_pattern(
    is_per_channel: bool,
    has_bias: bool,
    bias_is_quantized: bool,
    conv_fn: Callable,
    bn_is_training: bool,
) -> Callable:
    """
    Quantized QAT conv - bn pattern with bn weights being folded into conv.
    """
    # TODO: allow setting eps
    bn_eps = 1e-5

    def _folded_quantized_qat_conv_bn_pattern(
        x: torch.Tensor,
        conv_weight: torch.Tensor,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        bn_running_mean: torch.Tensor,
        bn_running_var: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # 计算 BN 的运行标准差，并添加一个很小的值 eps，以避免除以零
        running_std = torch.sqrt(bn_running_var + bn_eps)
        # 计算缩放因子，将 BN 权重除以运行标准差得到
        scale_factor = bn_weight / running_std
        # 构建权重和偏置的形状
        weight_shape = [1] * len(conv_weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(conv_weight.shape)
        bias_shape[1] = -1
        # 将卷积权重乘以缩放因子，以将 BN 权重折叠到卷积权重中
        scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
        # 如果有偏置，处理偏置
        scaled_weight = _append_qdq(
            scaled_weight, is_per_channel, is_bias=False, kwargs=kwargs,
        )
        if has_bias:
            # 创建与卷积偏置相同形状的零偏置张量
            zero_bias = torch.zeros_like(kwargs["conv_bias"], dtype=x.dtype)
            if bias_is_quantized:
                # 如果偏置是量化的，将其处理为与卷积权重类似的方式
                zero_bias = _append_qdq(
                    zero_bias, is_per_channel, is_bias=True, kwargs=kwargs,
                )
            # 执行带有权重和偏置的卷积操作
            x = conv_fn(x, scaled_weight, zero_bias)
        else:
            # 执行仅带权重的卷积操作
            x = conv_fn(x, scaled_weight, None)
        # 反向操作：将 x 除以缩放因子，以还原 BN 的效果
        x = x / scale_factor.reshape(bias_shape)
        if has_bias:
            # 如果有偏置，将其添加回 x 中
            x = x + kwargs["conv_bias"].reshape(bias_shape)
        # 应用批量归一化，使用给定的运行均值、方差、权重、偏置，并根据训练状态进行调整
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=bn_is_training, eps=bn_eps)
        return x
    # 返回一个封装后的模块，即 _folded_quantized_qat_conv_bn_pattern 函数
    return _WrapperModule(_folded_quantized_qat_conv_bn_pattern)
    ) -> torch.Tensor:
        # 定义函数签名，表示该函数接受输入并返回 torch.Tensor 类型的输出
        conv_weight = _append_qdq(
            conv_weight, is_per_channel, is_bias=False, kwargs=kwargs,
        )
        # 调用 _append_qdq 函数，对卷积权重进行量化/量化感知训练扩展处理
        if has_bias:
            # 如果存在偏置项
            bias = kwargs["conv_bias"]
            # 从参数中获取卷积层偏置值
            if bias_is_quantized:
                # 如果偏置被量化
                bias = _append_qdq(
                    bias, is_per_channel, is_bias=True, kwargs=kwargs,
                )
                # 调用 _append_qdq 函数，对偏置进行量化/量化感知训练扩展处理
        else:
            # 如果不存在偏置项
            bias = None
            # 将偏置设为 None
        # 调用卷积函数，传入输入 x、处理过的权重 conv_weight 和偏置 bias
        x = conv_fn(x, conv_weight, bias)
        # 对 x 进行批归一化处理，使用批次统计信息 bn_running_mean、bn_running_var，权重 bn_weight，偏置 bn_bias，训练模式 bn_is_training 和 epsilon bn_eps
        x = F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=bn_is_training, eps=bn_eps)
        # 返回处理后的张量 x
        return x
    # 返回包装后的模块 _WrapperModule，处理的模式是折叠量化量化感知训练卷积批归一化模式
    return _WrapperModule(_folded_quantized_qat_conv_bn_pattern)
# 检查匹配的子图是否包含具有偏置项的卷积节点，返回布尔值
def _has_conv_bias_filter(
    match: "InternalMatch",
    original_graph: Graph,
    pattern_graph: Graph,
) -> bool:
    """
    Match filter for the subgraph rewriter that returns True if the conv node in
    the original graph has bias.
    """
    # 遍历匹配的节点映射
    for n in match.nodes_map.values():
        # 如果节点是卷积或转置卷积节点
        if _is_conv_or_conv_transpose_node(n):
            # 返回节点的参数数量大于2且第三个参数不为None（表示有偏置项）
            return len(n.args) > 2 and n.args[2] is not None
    # 如果未找到包含偏置项的卷积节点，引发异常
    raise ValueError("Could not find conv node in matched conv + bn pattern")

# 检查匹配的子图是否不包含具有偏置项的卷积节点，返回布尔值
def _no_conv_bias_filter(
    match: "InternalMatch",
    original_graph: Graph,
    pattern_graph: Graph,
) -> bool:
    """
    Match filter for the subgraph rewriter that returns True if the conv node in
    the original graph does NOT have bias.
    """
    # 返回与 _has_conv_bias_filter 相反的结果
    return not _has_conv_bias_filter(match, original_graph, pattern_graph)

# 判断节点是否为量化节点，返回布尔值
def _is_quantize(n: Node) -> bool:
    return n.target in [
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.quantize_per_channel.default,
    ]

# 判断节点是否为去量化节点，返回布尔值
def _is_dequantize(n: Node) -> bool:
    return n.target in [
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    ]

# 提取卷积-BN融合模式中的节点信息，返回字典形式的映射
def _get_conv_bn_pattern_nodes(r: ReplacedPatterns) -> Dict[str, Tuple[Node, Node]]:
    """
    Helper function to extract the nodes in the conv-bn fusion pattern after
    subgraph rewriting, in the form of a map:

        {name: (original_node, replacement_node)}

    The following names must exist in the map:

        "conv", "conv_weight", "conv_input", "bn", "getitem"

    The following names may exist in the map:

        "conv_weight_q", "conv_weight_dq", "conv_bias",
        "conv_bias_q", "conv_bias_dq"
    """
    # 内部函数，从节点列表中获取卷积、BN和getitem节点
    def _get_nodes(nodes: List[Node]) -> Tuple[Node, Node, Node]:
        """
        Return a 3-tuple of (conv_node, bn_node, getitem_node).
        This asserts that the match contains exactly one of each node.
        """
        conv_node, bn_node, getitem_node = None, None, None
        # 遍历节点列表
        for n in nodes:
            # 如果节点的操作是函数调用
            if n.op != "call_function":
                continue
            # 判断节点类型并分配给相应变量
            if _is_conv_or_conv_transpose_node(n):
                assert conv_node is None
                conv_node = n
            if _is_bn_node(n):
                assert bn_node is None
                bn_node = n
            if n.target == operator.getitem:
                assert getitem_node is None
                getitem_node = n
        # 确保找到卷积、BN和getitem节点
        assert conv_node is not None
        assert bn_node is not None
        assert getitem_node is not None
        return (conv_node, bn_node, getitem_node)
    # 定义一个函数，获取与给定节点关联的原始节点、量化节点和反量化节点的三元组
    def _get_q_dq_nodes(n: Node) -> Tuple[Node, Node, Node]:
        """
        Return a 3-tuple of (orig_node, q_node, dq_node).
        """
        # 断言节点是反量化节点
        assert _is_dequantize(n)
        # 获取量化节点作为q_node
        q_node = n.args[0]
        # 断言q_node是Node类型的对象
        assert isinstance(q_node, Node)
        # 断言量化节点的第一个参数是原始节点
        orig_node = q_node.args[0]
        # 断言orig_node是Node类型的对象
        assert isinstance(orig_node, Node)
        # 返回原始节点、量化节点和反量化节点的三元组
        return (orig_node, q_node, n)

    # 获取原始节点列表，通过过滤节点映射后的值
    original_nodes = list(_filter_nodes_map(r.nodes_map).values())
    # 从原始节点列表中获取卷积节点、批量归一化节点和获取项节点
    o_conv, o_bn, o_getitem = _get_nodes(original_nodes)
    # 从替换节点列表中获取卷积节点、批量归一化节点和获取项节点
    r_conv, r_bn, r_getitem = _get_nodes(r.replacements)

    # 创建从原始节点到替换节点的映射
    mapping = {
        "conv": (o_conv, r_conv),
        "bn": (o_bn, r_bn),
        "getitem": (o_getitem, r_getitem),
    }

    # 提取卷积操作的输入和权重
    # 注意：这里通过模式节点间接提取原始节点，因为替换后原始节点的参数不再可用
    (p_conv, _, _) = _get_nodes(list(r.nodes_map.keys()))
    (p_conv_input, p_conv_weight, *_) = p_conv.args
    (r_conv_input, r_conv_weight, *_) = r_conv.args
    assert isinstance(p_conv_input, Node)
    assert isinstance(p_conv_weight, Node)
    assert isinstance(r_conv_input, Node)
    assert isinstance(r_conv_weight, Node)
    # 获取原始卷积操作的输入节点和权重节点
    o_conv_input = r.nodes_map[p_conv_input]
    o_conv_weight = r.nodes_map[p_conv_weight]

    # 如果卷积权重是量化的，提取其q - dq节点
    if _is_dequantize(p_conv_weight):
        # 获取原始和替换卷积权重的q - dq节点
        p_conv_weight, p_conv_weight_q, p_conv_weight_dq = _get_q_dq_nodes(p_conv_weight)
        r_conv_weight, r_conv_weight_q, r_conv_weight_dq = _get_q_dq_nodes(r_conv_weight)
        # 获取原始卷积权重、q权重和dq权重的替换节点
        o_conv_weight = r.nodes_map[p_conv_weight]
        o_conv_weight_q = r.nodes_map[p_conv_weight_q]
        o_conv_weight_dq = r.nodes_map[p_conv_weight_dq]
        # 添加卷积权重q节点和dq节点到映射中
        mapping["conv_weight_q"] = (o_conv_weight_q, r_conv_weight_q)
        mapping["conv_weight_dq"] = (o_conv_weight_dq, r_conv_weight_dq)
    # 添加卷积输入节点和权重节点到映射中
    mapping["conv_input"] = (o_conv_input, r_conv_input)
    mapping["conv_weight"] = (o_conv_weight, r_conv_weight)

    # 提取卷积偏置
    # 检查 p_conv 和 r_conv 的参数列表是否大于2，确保可以访问第三个参数
    if len(p_conv.args) > 2 and len(r_conv.args) > 2:
        # 获取 p_conv 和 r_conv 的第三个参数作为偏置项
        p_conv_bias = p_conv.args[2]
        r_conv_bias = r_conv.args[2]
        # 断言第三个参数是 Node 类型的对象
        assert isinstance(p_conv_bias, Node)
        assert isinstance(r_conv_bias, Node)
        # 根据 p_conv_bias 查找在 r 节点映射中的对应节点
        o_conv_bias = r.nodes_map[p_conv_bias]

        # 如果卷积偏置项是量化的，则提取 q - dq 节点
        if _is_dequantize(p_conv_bias):
            # 获取 p_conv_bias 的量化和去量化节点
            p_conv_bias, p_conv_bias_q, p_conv_bias_dq = _get_q_dq_nodes(p_conv_bias)
            r_conv_bias, r_conv_bias_q, r_conv_bias_dq = _get_q_dq_nodes(r_conv_bias)
            # 更新 o_conv_bias 到量化和去量化节点在 r 节点映射中的对应关系
            o_conv_bias = r.nodes_map[p_conv_bias]
            o_conv_bias_q = r.nodes_map[p_conv_bias_q]
            o_conv_bias_dq = r.nodes_map[p_conv_bias_dq]
            # 更新 mapping 字典，记录量化和去量化节点的映射关系
            mapping["conv_bias_q"] = (o_conv_bias_q, r_conv_bias_q)
            mapping["conv_bias_dq"] = (o_conv_bias_dq, r_conv_bias_dq)
        
        # 更新 mapping 字典，记录卷积偏置项在 r 节点映射中的对应关系
        mapping["conv_bias"] = (o_conv_bias, r_conv_bias)
    
    # 返回最终的映射关系字典
    return mapping
def _filter_nodes_map(nodes_map: Dict[Node, Node]) -> Dict[Node, Node]:
    """
    Return a filtered `nodes_map` returned from the subgraph rewriter.
    The filtered `nodes_map` will contain only nodes that are actually
    matched in the pattern, excluding None or placeholder nodes.
    """
    # 创建一个新的空字典，用于存储筛选后的节点映射关系
    new_nodes_map: Dict[Node, Node] = {}
    # 遍历给定的节点映射
    for pattern_node, graph_node in nodes_map.items():
        # 如果图中的节点是None，则跳过
        if graph_node is None:
            continue
        # 如果模式节点的操作是"placeholder"，则跳过
        if pattern_node.op == "placeholder":
            continue
        # 将模式节点和图节点添加到新的节点映射中
        new_nodes_map[pattern_node] = graph_node
    # 返回筛选后的节点映射
    return new_nodes_map

# TODO: this is error prone, use the replace_literals_with_placeholders hack instead
def _copy_over_literal_conv_args(original_node: Node, new_node: Node):
    """
    Copy over literal args in conv, such as stride and padding, from the matched node
    in the original graph to its replacement in the new graph.

    This is needed due to the following limitation in the subgraph rewriter when used
    with dynamo export: literal (non-tensor) args are not supported in the match and
    replacement patterns. This is because dynamo export automatically inlines these
    literal args, making them dead placeholder nodes. In the future, we should check
    if dynamo export can optionally disable this inlining, or if subgraph rewriter
    can do the copying for us. See https://github.com/pytorch/pytorch/issues/100419.

    Note: Unlike other tensor args like conv weights and biases, literal args are
    preserved in the original nodes after replacement, so we can access them here.
    """
    # 确保原始节点和新节点都是卷积或卷积转置节点
    assert _is_conv_or_conv_transpose_node(original_node)
    assert _is_conv_or_conv_transpose_node(new_node)
    # x, weight, bias, [stride, padding, dilation, transposed, output_padding, groups]
    # 将新节点的参数列表转换为列表以便修改
    new_args = list(new_node.args)
    if len(new_args) < 3:
        # 当没有偏置时，偏置参数为None
        new_args.append(None)
    # 更新新节点的参数，保留原始节点中的字面参数（如步幅和填充）
    new_node.args = tuple(new_args[:3]) + original_node.args[3:]

def _update_conv_input_qspec_map_after_replacement(original_node: Node, replacement_node: Node):
    """
    Update the `input_qspec_map` in the annotation after subgraph rewriting.

    The original annotation referred to the nodes in the original graph,
    so the keys in the `input_qspec_map` will need to be updated to reflect
    the corresponding nodes in the replacement graph.
    """
    # 确保原始节点和替换节点都是卷积或卷积转置节点
    assert _is_conv_or_conv_transpose_node(original_node)
    assert _is_conv_or_conv_transpose_node(replacement_node)
    # 如果原始节点的元数据中没有"quantization_annotation"，则直接返回
    if "quantization_annotation" not in original_node.meta:
        return
    # 获取原始输入量化映射
    original_input_qspec_map = original_node.meta["quantization_annotation"].input_qspec_map
    # 创建一个新的空字典，用于存储更新后的输入量化映射
    input_qspec_map = {}
    # 获取配置列表，应按照输入、权重、偏置的顺序排序
    # 注意：这是一个非常临时的解决方案，我们需要一个更好的解决方案
    # 在 subgraph_rewriter 中，处理问题跟踪的链接: https://github.com/pytorch/pytorch/issues/101820
    # 将 original_input_qspec_map 转换为列表，并存储在 all_configs 中
    all_configs = list(original_input_qspec_map.items())
    # 设置输入激活量化配置，使用第一个配置项
    input_qspec_map[replacement_node.args[0]] = all_configs[0][1]
    # 设置权重量化配置，使用第二个配置项
    input_qspec_map[replacement_node.args[1]] = all_configs[1][1]
    # 如果替换节点有偏置参数，并且 all_configs 中有第三个配置项，则设置偏置量化配置
    if len(replacement_node.args) > 2 and len(all_configs) > 2:
        input_qspec_map[replacement_node.args[2]] = all_configs[2][1]
    # 将更新后的 input_qspec_map 存储在 replacement_node 的元数据中的 quantization_annotation 下
    replacement_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
# 更新替换后的子图中的特殊量化规格`SharedQuantizationSpec`和`DerivedQuantizationSpec`
# 在节点 `node` 的量化注释中，原始注释引用原始图中的节点，因此这些特殊量化规格使用的节点需要更新为替换图中对应的节点。
def _update_special_qspecs_after_replacement(
    node: Node,
    original_to_replacement_node: Dict[Node, Node],
):
    """
    Update the `SharedQuantizationSpec`s and `DerivedQuantizationSpec`s
    used in `node`'s quantization annotation after subgraph rewriting.

    The original annotation referred to the nodes in the original graph,
    so the nodes used in these special quantization specs will need to
    be updated to the corresponding nodes in the replacement graph.
    """
    def _get_new_edge_or_node(edge_or_node: EdgeOrNode):
        # 如果是节点类型，返回其替换后的节点，否则抛出异常
        if isinstance(edge_or_node, Node):
            _node = edge_or_node
            return original_to_replacement_node.get(_node, _node)
        # 如果是二元组且元素均为节点类型，返回替换后的源节点和目标节点，否则抛出异常
        elif isinstance(edge_or_node, tuple) and len(edge_or_node) == 2 and all(isinstance(x, Node) for x in edge_or_node):
            src, dest = edge_or_node
            return (
                original_to_replacement_node.get(src, src),
                original_to_replacement_node.get(dest, dest),
            )
        else:
            raise ValueError("unexpected type for edge_or_node: ", type(edge_or_node))

    def _get_new_qspec(qspec: QuantizationSpecBase):
        # 如果是`SharedQuantizationSpec`类型，返回替换后的边缘或节点
        if isinstance(qspec, SharedQuantizationSpec):
            new_edge_or_node = _get_new_edge_or_node(qspec.edge_or_node)
            return SharedQuantizationSpec(new_edge_or_node)
        # 如果是`DerivedQuantizationSpec`类型，替换其来源并返回
        elif isinstance(qspec, DerivedQuantizationSpec):
            new_derived_from = [_get_new_edge_or_node(x) for x in qspec.derived_from]
            return dataclasses.replace(qspec, derived_from=new_derived_from)
        else:
            return qspec

    # 如果节点的元数据中没有 `quantization_annotation`，直接返回
    if "quantization_annotation" not in node.meta:
        return
    # 获取节点的量化注释
    annotation = node.meta["quantization_annotation"]
    # 遍历输入量化映射表，更新其中的量化规格
    for input_node, qspec in annotation.input_qspec_map.items():
        annotation.input_qspec_map[input_node] = _get_new_qspec(qspec)
    # 更新输出量化规格
    annotation.output_qspec = _get_new_qspec(annotation.output_qspec)

# 融合`Conv`和`BN`的量化感知训练（QAT）子图等效于融合后的图形模块
def _fuse_conv_bn_qat(m: GraphModule) -> GraphModule:
    # 检查图中是否存在 BatchNorm 节点
    has_bn = any(_is_bn_node(n) for n in m.graph.nodes)
    # 如果没有 BatchNorm 节点，直接返回原图模块
    if not has_bn:
        return m
    # 根据当前环境是否支持 CUDA 设置相应的选项列表
    is_cuda_options = [True, False] if torch.cuda.is_available() else [False]
    # 遍历 CUDA 选项列表
    for is_cuda in is_cuda_options:
        # 依次对不同类型的卷积操作调用辅助函数进行量化感知训练子图的融合
        m = _fuse_conv_bn_qat_helper(m, F.conv1d, _conv1d_bn_example_inputs, is_cuda=is_cuda)
        m = _fuse_conv_bn_qat_helper(m, F.conv2d, _conv2d_bn_example_inputs, is_cuda=is_cuda)
        m = _fuse_conv_bn_qat_helper(m, F.conv_transpose1d, _conv1d_bn_example_inputs, is_cuda=is_cuda)
        m = _fuse_conv_bn_qat_helper(m, F.conv_transpose2d, _conv2d_bn_example_inputs, is_cuda=is_cuda)
    # 返回更新后的图形模块
    return m

# 辅助函数：根据给定的卷积函数和示例输入，替换图中的 (conv + bn) 模式为融合后的 QAT 子图等效。
def _fuse_conv_bn_qat_helper(
    m: GraphModule,
    conv_fn: Callable,
    example_inputs: Tuple[Any, ...],
    is_cuda: bool,
) -> GraphModule:
    """
    Given a graph of decomposed aten ops, replace the (conv + bn) pattern with
    the fused QAT subgraph equivalent. The input graph should already be annotated.
    """
    """
    The annotations in the original nodes will be preserved in the corresponding
    nodes in the new subgraph.

    Note: This also handles the (conv + bn + relu) pattern.
    """
    # 清除模型图中的死代码，优化模型图
    m.graph.eliminate_dead_code()
    # 重新编译模型
    m.recompile()

    # 获取带有 conv + bn 模式的模式图
    conv_bn_pattern = _get_conv_bn_pattern(conv_fn)
    # 获取模式匹配图，用于查找 conv + bn 模式在模型中的位置
    match_pattern = _get_aten_graph_module_for_pattern(conv_bn_pattern, example_inputs, is_cuda)

    # Step (1): Replace patterns with conv bias
    #
    # 这里我们分别处理有和没有 conv bias 的情况，因为这两种情况的替换模式差别很大。
    # TODO: 一旦 replace_pattern API 也返回替换节点，就可以使用公共的 API

    # 获取带有 QAT conv + bn 模式的模式图
    qat_conv_bn_pattern = _get_qat_conv_bn_pattern(conv_fn)
    # 获取用于替换的模式匹配图，带有 conv bias
    replacement_pattern_with_conv_bias = _get_aten_graph_module_for_pattern(
        qat_conv_bn_pattern,
        example_inputs,
        is_cuda,
    )
    # 替换带有 conv bias 的模式
    replacements_with_conv_bias = replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern_with_conv_bias,
        match_filters=[_has_conv_bias_filter],
        ignore_literals=True,
    )
    # 重新编译模型
    m.recompile()

    # Step (2): Replace patterns without conv bias

    # 获取没有 conv bias 的 QAT conv + bn 模式图
    qat_conv_bn_pattern_no_conv_bias = _get_qat_conv_bn_pattern_no_conv_bias(conv_fn)
    # 获取用于替换的模式匹配图，没有 conv bias
    replacement_pattern_no_conv_bias = _get_aten_graph_module_for_pattern(
        qat_conv_bn_pattern_no_conv_bias,
        example_inputs,
        is_cuda,
    )
    # 替换没有 conv bias 的模式
    replacements_no_conv_bias = replace_pattern_with_filters(
        m,
        match_pattern,
        replacement_pattern_no_conv_bias,
        match_filters=[_no_conv_bias_filter],
        ignore_literals=True,
    )
    # 重新编译模型
    m.recompile()

    # Step (3): Post processing
    #
    # 由于子图重写器的功能有限，这里我们手动更新替换图：
    #
    #   (a) 从原始子图复制元数据。这确保了堆栈跟踪和注释在新子图中得到保留。
    #
    #   (b) 从原始子图复制 conv 的 literal args。
    #       TODO: 对 batchnorm 的 literal args 也执行相同操作。
    #
    #   (c) 更新原始子图中所有旧节点的引用，使其引用新子图中对应的节点。
    #
    # 在未来，我们应尽可能将这些功能推入子图重写器中，以便无需手动复制任何内容。
    # 更多细节，请参阅 https://github.com/pytorch/pytorch/issues/100419.

    # 创建一个空字典，用于存储所有原始节点到替换节点的映射关系
    all_original_to_replacement_nodes = {}
    # 对于每个替换节点 r，包括有偏置和无偏置替换
    for r in replacements_with_conv_bias + replacements_no_conv_bias:
        # 获取替换节点 r 中卷积 - 批归一化模式节点的原始节点和替换节点
        for original_node, replacement_node in _get_conv_bn_pattern_nodes(r).values():
            # 步骤 (3a): 复制 [卷积 - 批归一化 - 获取项目] 中所有节点的元数据
            replacement_node.meta = original_node.meta
            # 如果原始节点是卷积或卷积转置节点
            if _is_conv_or_conv_transpose_node(original_node):
                # 步骤 (3b): 复制卷积节点的文字参数
                _copy_over_literal_conv_args(original_node, replacement_node)
                # 步骤 (3c): 更新卷积节点输入 qspec 映射中的旧引用
                _update_conv_input_qspec_map_after_replacement(original_node, replacement_node)
            # 将原始节点到替换节点的映射存入字典
            all_original_to_replacement_nodes[original_node] = replacement_node

    # 步骤 (3c): 更新图中所有节点的特殊 qspec 引用
    for n in m.graph.nodes:
        _update_special_qspecs_after_replacement(n, all_original_to_replacement_nodes)

    # 返回修改后的模型 m
    return m
# 辅助函数，用于复制图中所有具有多个用户的去量化节点。
def _duplicate_dequantize_node(m: GraphModule):
    """
    Helper function to duplicate all dequantize nodes in the graph if the
    node has more than one user. For example:

    Before:
      quantize -> dequantize -> a
                          \\--> b
                          \\--> c

    After:
      quantize -> dequantize_1 -> a
            \\--> dequantize_2 -> b
            \\--> dequantize_3 -> c

    This is useful for subgraph rewriting. E.g. if we wish to match the
    pattern [dequantize - a] above, subgraph matching would fail because
    the dequantize node has users outside the matched portion of the graph.
    Instead, we match [dequantize_1 - a], which is safe.
    """
    # 获取去量化操作的函数对象
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor
    # 遍历图中的每个节点
    for n in m.graph.nodes:
        # 如果节点不是调用函数节点，或者目标不是去量化操作，或者只有一个用户，则跳过
        if n.op != "call_function" or n.target != dq_op or len(n.users) == 1:
            continue
        # 遍历节点的每个用户
        for user in list(n.users):
            # 在节点 n 前插入新节点
            with m.graph.inserting_before(n):
                # 创建新的调用函数节点，复制去量化操作
                new_node = m.graph.create_node("call_function", dq_op, n.args, n.kwargs)
            # 用新节点替换用户对原节点的输入
            user.replace_input_with(n, new_node)
        # 删除原节点
        m.graph.erase_node(n)
    # 重新编译模型
    m.recompile()

# 辅助函数，用于移除图中多余的去量化节点，保留唯一的一个去量化节点用于多个操作
def _remove_extra_dequantize(m: GraphModule):
    """
    Removes duplicate dequant nodes in the graph, for an operator that has
    multiple dequant nodes as a user, replace them with a single dequant node
    that can be shared across all the uses. This should be seen as the "reverse"
    of `_duplicate_dequantize_node`.
    """
    # 获取去量化操作的函数对象
    dq_op = torch.ops.quantized_decomposed.dequantize_per_tensor
    # 遍历图中的每个节点
    for n in m.graph.nodes:
        # 找到所有作为用户的去量化节点
        dq_users = [user for user in n.users if user.op == "call_function" and user.target == dq_op]
        # 如果有多个去量化用户
        if len(dq_users) > 1:
            # 在第一个去量化用户后插入新节点
            with m.graph.inserting_after(dq_users[0]):
                # 创建新的调用函数节点，复制去量化操作
                new_node = m.graph.create_node("call_function", dq_op, dq_users[0].args, {})
            # 替换所有去量化用户的使用为新节点，并删除原节点
            for dq_user in dq_users:
                dq_user.replace_all_uses_with(new_node)
                m.graph.erase_node(dq_user)
    # 重新编译模型
    m.recompile()

# 辅助函数，用于复制一个量化或去量化节点的所有字面参数到替换节点
def _copy_over_q_dq_args(original_node: Node, replacement_node: Node):
    """
    Given a pair of quantize or dequantize nodes, copy over all literal args
    from the original node to the replacement node.
    """
    # 断言原节点和替换节点目标相同
    assert original_node.target == replacement_node.target
    # 根据不同的量化或去量化操作，确定需要复制的参数起始索引
    if original_node.target in (
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
    ):
        # 参数：input, [scale, zp, qmin, qmax, dtype]
        start_copy_arg_index = 1
    elif original_node.target in (
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    ):
        # 参数：input, scale, zp, [axis, qmin, qmax, dtype]
        start_copy_arg_index = 3
    else:
        # 如果条件不满足，抛出值错误异常，指明预期的量化/反量化节点，实际得到的节点类型
        raise ValueError(f"Expected quantize/dequantize nodes, got '{original_node.target}'")
    # 更新替换节点的参数，保留部分原始节点的参数
    replacement_node.args = (
        replacement_node.args[:start_copy_arg_index] + original_node.args[start_copy_arg_index:]
    )
def _fold_conv_bn_qat(m: GraphModule) -> GraphModule:
    # 检查图中是否存在 Batch Normalization 节点
    has_bn = any(_is_bn_node(n) for n in m.graph.nodes)
    if not has_bn:
        return m
    # 根据 CUDA 是否可用确定选项
    is_cuda_options = [True, False] if torch.cuda.is_available() else [False]
    for is_cuda in is_cuda_options:
        # 调用辅助函数处理不同类型的卷积层及其 Batch Normalization
        m = _fold_conv_bn_qat_helper(m, F.conv1d, _quantized_conv1d_bn_example_inputs, is_cuda=is_cuda)
        m = _fold_conv_bn_qat_helper(m, F.conv2d, _quantized_conv2d_bn_example_inputs, is_cuda=is_cuda)
        m = _fold_conv_bn_qat_helper(m, F.conv_transpose1d, _quantized_conv1d_bn_example_inputs, is_cuda=is_cuda)
        m = _fold_conv_bn_qat_helper(m, F.conv_transpose2d, _quantized_conv2d_bn_example_inputs, is_cuda=is_cuda)
    return m

def _fold_conv_bn_qat_helper(
    m: GraphModule,
    conv_fn: Callable,
    example_inputs: Tuple[Any, ...],
    is_cuda: bool,
) -> GraphModule:
    """
    Replace the quantized (conv + bn) pattern with conv with bn weights folded into the weights of conv.
    """
    # 清除无用代码
    m.graph.eliminate_dead_code()
    m.recompile()
    # 复制去量化节点以备份
    _duplicate_dequantize_node(m)

    # 步骤（1）：将 QAT 模式替换为简单的 [conv - bn] 模式
    replacements = []
    # 枚举所有替换选项
    replacement_options = itertools.product(
        [True, False],  # is_per_channel
        [True, False],  # has_bias
        [True, False],  # bias_is_quantized
        [True, False],  # bn_is_training
    )
    for is_per_channel, has_bias, bias_is_quantized, bn_is_training in replacement_options:
        # 对于没有偏置的情况，忽略偏置是否量化的可能性，以避免重复模式
        if not has_bias and bias_is_quantized:
            continue
        # 获取当前替换选项的输入参数
        kwargs = _get_quantized_conv_bn_example_inputs_kwargs(is_per_channel, has_bias, bias_is_quantized, is_cuda)
        # 获取当前替换选项的量化 QAT 模式
        match_pattern = _get_quantized_qat_conv_bn_pattern(
            is_per_channel, has_bias, bias_is_quantized, conv_fn, bn_is_training
        )
        # 根据输入参数获取模式的图模块
        match_pattern = _get_aten_graph_module_for_pattern(match_pattern, example_inputs, is_cuda, **kwargs)
        # 获取折叠后的量化 QAT 模式
        replacement_pattern = _get_folded_quantized_qat_conv_bn_pattern(
            is_per_channel, has_bias, bias_is_quantized, conv_fn, bn_is_training
        )
        # 根据输入参数获取模式的图模块
        replacement_pattern = _get_aten_graph_module_for_pattern(replacement_pattern, example_inputs, is_cuda, **kwargs)
        # 替换模式并记录替换结果
        replacements.extend(
            replace_pattern_with_filters(
                m,
                match_pattern,
                replacement_pattern,
                ignore_literals=True,
            )
        )
    # 重新编译图模块
    m.recompile()
    # 移除额外的去量化节点
    _remove_extra_dequantize(m)
    # 遍历替换列表中的每个替换对象
    for r in replacements:
        # 获取当前替换对象的卷积-BN模式节点映射
        node_map = _get_conv_bn_pattern_nodes(r)

        # 步骤（2）：从原始子图复制元数据
        for original_node, replacement_node in node_map.values():
            # 将替换节点的元数据设置为原始节点的元数据
            replacement_node.meta = original_node.meta

        # 步骤（3）：复制权重（和可选偏置）q - dq 节点的参数
        _copy_over_q_dq_args(*node_map["conv_weight_q"])
        _copy_over_q_dq_args(*node_map["conv_weight_dq"])
        # 如果存在偏置节点，则复制其参数
        if "conv_bias_q" in node_map:
            assert "conv_bias_dq" in node_map
            _copy_over_q_dq_args(*node_map["conv_bias_q"])
            _copy_over_q_dq_args(*node_map["conv_bias_dq"])

        # 步骤（4）：将BN的权重合并到卷积节点中
        conv_bias = None
        # 获取卷积和BN节点
        (_, conv_node) = node_map["conv"]
        (_, bn_node) = node_map["bn"]
        (_, conv_weight) = node_map["conv_weight"]
        # 如果存在卷积的偏置节点，则获取其信息
        if "conv_bias" in node_map:
            (_, conv_bias) = node_map["conv_bias"]
        # 将BN的权重合并到卷积节点中
        fold_bn_weights_into_conv_node(conv_node, conv_weight, conv_bias, bn_node, m)

        # 复制卷积节点的字面参数
        for original_node in _filter_nodes_map(r.nodes_map).values():
            # 如果是卷积或转置卷积节点，则复制其字面参数
            if _is_conv_or_conv_transpose_node(original_node):
                _copy_over_literal_conv_args(original_node, conv_node)

    # 从模型图中清除死代码
    m.graph.eliminate_dead_code()
    # 重新编译模型
    m.recompile()
    # 返回更新后的模型
    return m
```