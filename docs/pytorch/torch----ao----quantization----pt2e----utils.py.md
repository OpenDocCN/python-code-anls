# `.\pytorch\torch\ao\quantization\pt2e\utils.py`

```py
# mypy: allow-untyped-defs
import operator  # 导入operator模块，用于操作符操作
import types  # 导入types模块，用于操作类型对象

import torch  # 导入PyTorch库
from torch._export import capture_pre_autograd_graph  # 导入捕获预自动微分图函数
from torch.fx import (  # 从torch.fx模块导入以下对象：
    GraphModule,  # 图模块
    Node,         # 节点
)
import torch.nn.functional as F  # 导入torch.nn.functional模块，并简称为F
from torch.nn.utils.fusion import fuse_conv_bn_weights  # 导入融合卷积和批归一化权重函数
from typing import Any, Callable, Dict, Optional, Tuple, List, Union  # 导入类型提示

from torch.utils._pytree import LeafSpec  # 从torch.utils._pytree导入LeafSpec对象
from torch.export.unflatten import _AttrKind, _assign_attr  # 从torch.export.unflatten导入_AttrKind和_assign_attr对象

# Makes sure that quantized_decomposed ops are registered
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # 确保quantized_decomposed操作已注册（仅注释，不执行）

from torch.ao.quantization.quantizer import QuantizationAnnotation  # 从torch.ao.quantization.quantizer导入QuantizationAnnotation类


__all__ = [  # 定义模块的公共接口列表
    "fold_bn_weights_into_conv_node",  # 折叠批归一化权重到卷积节点
    "remove_tensor_overload_for_qdq_ops",  # 移除用于量化和去量化操作的张量过载
]

_QUANTIZE_OPS = [  # 定义量化操作的列表
    torch.ops.quantized_decomposed.quantize_per_tensor.default,    # 默认张量量化操作
    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,     # 张量量化操作
    torch.ops.quantized_decomposed.quantize_per_channel.default,   # 默认通道量化操作
]

_DEQUANTIZE_OPS = [  # 定义去量化操作的列表
    torch.ops.quantized_decomposed.dequantize_per_tensor.default,  # 默认张量去量化操作
    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,   # 张量去量化操作
    torch.ops.quantized_decomposed.dequantize_per_channel.default, # 默认通道去量化操作
]

# Example inputs for conv-bn1d patterns
_conv1d_bn_example_inputs = (  # 定义用于卷积批归一化1D模式的示例输入
    torch.randn(1, 1, 3),  # x，随机正态分布张量
    torch.randn(1, 1, 1),  # conv_weight，随机正态分布卷积权重
    torch.randn(1),        # conv_bias，随机正态分布卷积偏置
    torch.randn(1),        # bn_weight，随机正态分布批归一化权重
    torch.randn(1),        # bn_bias，随机正态分布批归一化偏置
    torch.randn(1),        # bn_running_mean，随机正态分布批归一化运行均值
    torch.randn(1),        # bn_running_var，随机正态分布批归一化运行方差
)

# Example inputs for conv-bn2d patterns
_conv2d_bn_example_inputs = (  # 定义用于卷积批归一化2D模式的示例输入
    torch.randn(1, 1, 3, 3),  # x，随机正态分布张量
    torch.randn(1, 1, 1, 1),  # conv_weight，随机正态分布卷积权重
    torch.randn(1),           # conv_bias，随机正态分布卷积偏置
    torch.randn(1),           # bn_weight，随机正态分布批归一化权重
    torch.randn(1),           # bn_bias，随机正态分布批归一化偏置
    torch.randn(1),           # bn_running_mean，随机正态分布批归一化运行均值
    torch.randn(1),           # bn_running_var，随机正态分布批归一化运行方差
)

def _is_connected(source: torch.fx.Node, dest: torch.fx.Node) -> bool:
    """
    Assuming dest is one of the ops inserted by quant workflow, this function
    finds if source and dest are connected. Assumption is that only quant workflow
    inserted ops exist between source and dest
    """
    quant_workflow_ops = _QUANTIZE_OPS + _DEQUANTIZE_OPS  # 合并量化和去量化操作列表
    quant_workflow_ops.append(torch.ops.quantized_decomposed.choose_qparams.tensor)  # 添加张量选择量化参数操作到列表
    while dest.target in quant_workflow_ops:  # 当目标节点的目标操作在量化工作流操作列表中时循环
        if not isinstance(dest.args[0], torch.fx.Node):  # 如果目标节点的第一个参数不是节点类型，则抛出值错误
            raise ValueError(f"expected arg[0] of quant workflow ops to be a node but found {dest.args[0]}")
        dest = dest.args[0]  # 更新目标节点为其第一个参数（节点）
    return (dest == source)  # 返回目标节点是否等于源节点的布尔值


def _find_q_dq_node_for_user(
    produer: torch.fx.Node, user: torch.fx.Node
) -> Tuple[Any, Any]:
    """
    Find q, dq pair corresponding to [producer -> q -> dq -> user]
    Utils works by finding dq arg of user and ensuring it is connected to
    producer
    """
    dq_node = None  # 初始化去量化节点为空
    # 遍历用户传入的位置参数 user.args
    for n in user.args:
        # 检查当前节点 n 是否为 torch.fx.Node 类型，且操作为 "call_function"，且目标函数在 _DEQUANTIZE_OPS 中
        if isinstance(n, torch.fx.Node) and n.op == "call_function" and n.target in _DEQUANTIZE_OPS:
            # 如果当前节点与 produer 之间存在连接
            if _is_connected(produer, n):
                # 将找到的节点 dq_node 设为当前节点 n，并结束循环
                dq_node = n
                break
    # 如果未找到 dequantize 节点，则继续搜索关键字参数 user.kwargs
    if dq_node is None:
        for n in user.kwargs:
            # 检查当前节点 n 是否为 torch.fx.Node 类型，且操作为 "call_function"，且目标函数在 _DEQUANTIZE_OPS 中
            if isinstance(n, torch.fx.Node) and n.op == "call_function" and n.target in _DEQUANTIZE_OPS:
                # 如果当前节点与 produer 之间存在连接
                if _is_connected(produer, n):
                    # 将找到的节点 dq_node 设为当前节点 n，并结束循环
                    dq_node = n
                    break
    # 如果还是未找到 dequantize 节点，则返回 (None, None)
    if dq_node is None:
        return (None, None)

    # 初始化量化节点 q_node
    q_node = None
    # 检查 dequantize 节点的第一个位置参数是否为 "call_function" 类型且目标函数在 _QUANTIZE_OPS 中
    if dq_node.args[0].op == "call_function" and dq_node.args[0].target in _QUANTIZE_OPS:
        # 如果是，则将 q_node 设为该节点
        q_node = dq_node.args[0]
    # 返回找到的量化节点 q_node 和 dequantize 节点 dq_node
    return (q_node, dq_node)
# 判断给定的节点是否为调用函数的节点，并且目标函数是 torch.ops.aten.sym_size.default 或其他相关符号操作函数
def _is_sym_size_node(node: Node):
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten.sym_size.default
        or node.target == torch.ops.aten.sym_numel.default
        or node.target == torch.ops.aten.sym_numel
        or node.target == torch.ops.aten.sym_size
    )


# 过滤掉符号尺寸操作节点的用户节点，返回剩余节点列表
def _filter_sym_size_users(node: torch.fx.Node) -> List[torch.fx.Node]:
    node_users = list(filter((lambda x: (_is_sym_size_node(x) is False)), node.users))
    return node_users


# 判断给定的注释是否有效，即输入量化映射和输出量化规格不能为空
def _is_valid_annotation(annotation: QuantizationAnnotation) -> bool:
    if annotation is None:
        return False
    input_qspec_map = annotation.input_qspec_map
    output_qspec = annotation.output_qspec
    if len(input_qspec_map) == 0 and output_qspec is None:
        return False
    return True


# 从节点中获取来自张量的常量值，根据节点目标找到对应的属性路径并返回
def _get_tensor_constant_from_node(node, m):
    if node is None:
        return None
    assert node.op == "get_attr"
    target_atoms = node.target.split('.')
    attr_itr = m
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


# 获取所有参数，结合原始位置参数、关键字参数和参数模式定义
def _get_all_arguments(orig_args, orig_kwargs, args_schema):
    all_args = []
    for i, schema in enumerate(args_schema):
        if schema.name in orig_kwargs:
            all_args.append(orig_kwargs[schema.name])
        elif not schema.kwarg_only and i < len(orig_args):
            all_args.append(orig_args[i])
        else:
            all_args.append(schema.default_value)
    return all_args


# 判断给定节点是否是支持用于训练的批量归一化操作节点
def _is_supported_batch_norm_for_training(node: Node):
    """
    Return True if the given node refers to an aten batch norm op QAT supports.
    """
    supported_ops = [
        torch.ops.aten._native_batch_norm_legit.default,
        # 注意：在批量归一化合并之后，我们将不再需要这个操作
        # 目前我们需要继续支持它，因为它比 `_native_batch_norm_legit` 提供更好的训练数值
        torch.ops.aten.cudnn_batch_norm.default,
        torch.ops.aten.miopen_batch_norm.default,
    ]
    return node.target in supported_ops


# 判断给定节点是否为卷积操作节点
def _is_conv_node(n: Node):
    """
    Return whether the node refers to an aten conv op.
    """
    return n.op == "call_function" and n.target in [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
    ]


# 判断给定节点是否为转置卷积操作节点
def _is_conv_transpose_node(n: Node):
    """
    Return whether the node refers to an aten conv_transpose op.
    """
    return n.op == "call_function" and n.target in [
        torch.ops.aten.conv_transpose1d,
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d,
        torch.ops.aten.conv_transpose2d.input,
    ]


# 判断给定节点是否为卷积或转置卷积操作节点
def _is_conv_or_conv_transpose_node(n: Node):
    """
    Return whether the node refers to an aten conv or conv transpose op.
    """
    # 返回一个布尔值，判断节点 `n` 是否为卷积操作节点或转置卷积操作节点
    return _is_conv_node(n) or _is_conv_transpose_node(n)
def _is_conv_transpose_fn(conv_fn: Callable):
    # 检查给定的卷积函数是否是转置卷积函数之一
    return conv_fn in [F.conv_transpose1d, F.conv_transpose2d]

def _is_bn_node(n: Node):
    # 检查节点是否是批量归一化节点，支持训练或与非训练相关的批量归一化操作
    return _is_supported_batch_norm_for_training(n) or n.target == torch.ops.aten._native_batch_norm_legit_no_training.default

def fold_bn_weights_into_conv_node(
    conv_node: Node,
    conv_weight_node: Node,
    conv_bias_node: Optional[Node],
    bn_node: Node,
    m: GraphModule
) -> None:
    # conv args: input, weight, bias, stride, padding, dilation, ...
    # 获取卷积节点的权重和偏置节点的张量常数
    conv_w = _get_tensor_constant_from_node(conv_weight_node, m)
    conv_b = _get_tensor_constant_from_node(conv_bias_node, m)
    # 检查是否为转置卷积
    transpose = _is_conv_transpose_node(conv_node)

    # eval bn args: input, weight, bias, running mean, running var, momentum, eps
    # train bn args: input, weight, bias, running mean, running var, training, momentum, eps
    # 获取批量归一化节点的参数模式和参数值
    bn_args_schema = bn_node.target._schema.arguments  # type: ignore[union-attr]
    bn_args = _get_all_arguments(bn_node.args, bn_node.kwargs, bn_args_schema)
    bn_w = _get_tensor_constant_from_node(bn_args[1], m)
    bn_b = _get_tensor_constant_from_node(bn_args[2], m)
    bn_rm = _get_tensor_constant_from_node(bn_args[3], m)
    bn_rv = _get_tensor_constant_from_node(bn_args[4], m)
    # 根据批量归一化节点的类型确定 eps 参数的位置
    if bn_node.target == torch.ops.aten._native_batch_norm_legit_no_training.default:
        eps_arg_index = 6
    elif _is_supported_batch_norm_for_training(bn_node):
        eps_arg_index = 7
    else:
        raise ValueError("BN node target is unexpected ", bn_node.target)
    bn_eps = bn_args[eps_arg_index]

    # 合并卷积和批量归一化的权重和偏置
    fused_weight, fused_bias = fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b, transpose=transpose)

    # 更新卷积节点的权重和偏置
    conv_args = list(conv_node.args)
    # 填充默认的偏置参数
    if len(conv_args) == 2:
        conv_args.append(None)

    # 将合并后的权重和偏置分配给卷积节点
    weight_attr_name = conv_weight_node.target
    assert isinstance(weight_attr_name, str)
    _assign_attr(fused_weight, m, weight_attr_name, _AttrKind.PARAMETER)
    if conv_bias_node is not None:
        bias_attr_name = conv_bias_node.target
        _assign_attr(fused_bias, m, str(bias_attr_name), _AttrKind.PARAMETER)
    else:
        bias_attr_name = weight_attr_name + "_bias"
        _assign_attr(fused_bias, m, bias_attr_name, _AttrKind.PARAMETER)
        # 在卷积节点之前插入获取偏置节点的操作
        with m.graph.inserting_before(conv_node):
            get_bias_node = m.graph.get_attr(bias_attr_name)
        # 注意：这里假设卷积的偏置未经量化！
        conv_args[2] = get_bias_node
    conv_node.args = tuple(conv_args)

    # native_batch_norm 有 3 个输出，我们期望在输出上进行getitem调用，并替换getitem 0 的使用为卷积的输出
    #
    # Before:
    # conv -> bn - (first output) -> users1
    #          \ - (second output) -> users2
    #          \ - (third output) -> users3
    # After:
    # (待补充，此处应根据实际情况填写后续处理的代码逻辑)
    # 遍历 bn_node 的所有使用者（即后继节点）
    for user in bn_node.users:
        # 检查使用者节点是否满足以下条件，如果不满足则跳过当前迭代
        if user.op != "call_function" or user.target != operator.getitem or user.args[1] != 0:
            continue
        # 将满足条件的使用者节点替换为 conv_node
        user.replace_all_uses_with(conv_node)
# 融合卷积和批量归一化的权重，直接修改图模块和图形对象
def _fuse_conv_bn_(m: GraphModule) -> None:
    # 检查图中是否存在批量归一化节点
    has_bn = any(_is_bn_node(n) for n in m.graph.nodes)
    if not has_bn:
        return  # 如果没有批量归一化节点，直接返回

    # 遍历图中的每个节点
    for n in m.graph.nodes:
        # 如果节点不是调用函数或者不是目标为 torch.ops.aten._native_batch_norm_legit_no_training.default 的节点，则跳过
        if n.op != "call_function" or n.target != torch.ops.aten._native_batch_norm_legit_no_training.default:
            continue
        bn_node = n
        n = bn_node.args[0]  # 获取批量归一化节点的第一个参数
        if not _is_conv_or_conv_transpose_node(n):  # 如果第一个参数不是卷积或转置卷积节点，则跳过
            continue
        conv_node = n
        conv_weight_node = conv_node.args[1]  # 获取卷积节点的权重参数
        conv_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None  # 获取卷积节点的偏置参数，如果存在的话
        fold_bn_weights_into_conv_node(conv_node, conv_weight_node, conv_bias_node, bn_node, m)  # 将批量归一化的权重融合到卷积节点中

    m.graph.eliminate_dead_code()  # 删除图中的死代码
    m.recompile()  # 重新编译图模块

def _get_node_name_to_scope(model: GraphModule) -> Dict[str, Tuple[str, type]]:
    # TODO: 将此信息移动到 FX 节点本身
    node_name_to_scope: Dict[str, Tuple[str, type]] = {}
    # 遍历模型图中的每个节点
    for n in model.graph.nodes:
        nn_module_stack = n.meta.get("nn_module_stack", None)  # 获取节点的 nn_module_stack 元数据
        current_scope = ("", type(None))  # 默认作用域为空字符串和 None 类型
        if nn_module_stack:
            bt = list(nn_module_stack.values())[-1]  # 获取 nn_module_stack 的最后一个值
            current_scope = (bt[0].split(".")[-1], bt[1])  # 设置当前作用域为 nn_module_stack 的最后一个值的名称和类型
        node_name_to_scope[n.name] = current_scope  # 将节点名称映射到当前作用域

    return node_name_to_scope  # 返回节点名称到作用域的映射字典

def _get_aten_graph_module_for_pattern(
    pattern: Callable,
    example_inputs: Tuple[Any, ...],
    is_cuda: bool = False,
    **kwargs,
) -> GraphModule:
    """
    将模式转换为带有分解 aten 操作的 FX 图形。
    """
    if is_cuda:
        example_inputs = tuple([x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs])  # 如果是 CUDA，将示例输入转移到 GPU 上

    aten_pattern = capture_pre_autograd_graph(
        pattern,
        example_inputs,
        kwargs,
    )  # 使用 capture_pre_autograd_graph 捕获模式的预自动微分图

    aten_pattern.graph.eliminate_dead_code()  # 删除图中的死代码
    aten_pattern.recompile()  # 重新编译图模块

    # 对于模式，ep.module() 会添加用于变异输入的 copy_ 节点。对于模式来说，这不重要。
    for node in aten_pattern.graph.nodes:
        # 如果节点是调用函数且目标是 torch.ops.aten.copy_.default，并且没有用户使用它，则擦除该节点
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default and len(node.users) == 0:
            aten_pattern.graph.erase_node(node)

    aten_pattern.graph.eliminate_dead_code()  # 再次删除图中的死代码
    aten_pattern.recompile()  # 重新编译图模块

    return aten_pattern  # 返回处理后的 FX 图形模块

def remove_tensor_overload_for_qdq_ops(match_pattern: GraphModule) -> None:
    """ 移除量化/反量化操作的 .tensor 过载，以便我们可以使用从 torchdynamo 导出的 match_pattern 来匹配 convert_pt2e 的输出 """
    # 定义一个映射表_MAP，将特定的 torch 操作函数映射到对应的量化或反量化函数上
    _MAP = {
        torch.ops.quantized_decomposed.quantize_per_tensor.default: torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default: torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor: torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_tensor.tensor2: torch.ops.quantized_decomposed.quantize_per_tensor,
        torch.ops.quantized_decomposed.dequantize_per_tensor.tensor2: torch.ops.quantized_decomposed.dequantize_per_tensor,
        torch.ops.quantized_decomposed.quantize_per_channel.default: torch.ops.quantized_decomposed.quantize_per_channel,
        torch.ops.quantized_decomposed.dequantize_per_channel.default: torch.ops.quantized_decomposed.dequantize_per_channel,
        torch.ops.aten.clamp.Tensor: torch.ops.aten.clamp,
    }
    
    # 遍历给定的 match_pattern 图中的每个节点
    for n in match_pattern.graph.nodes:
        # 如果当前节点不是 "call_function" 类型，则跳过该节点
        if n.op != "call_function":
            continue
        # 如果当前节点的目标函数在_MAP中，则将其目标函数替换为_MAP中映射的函数
        if n.target in _MAP:
            n.target = _MAP[n.target]
# 判断参数是否为字面值（literal），即 int 或 float 类型
def _is_literal(arg):
    if isinstance(arg, (int, float)):
        return True
    # 如果参数是 tuple 或 list 类型，递归检查每个元素是否为字面值
    if isinstance(arg, (tuple, list)):
        return all(map(_is_literal, arg))
    # 其他情况均返回 False
    return False

# 替换图中的字面值（literal）为新的占位符节点，这些占位符节点在遍历图时动态创建，
# 以便匹配和替换图中的字面值参数
def _replace_literals_with_new_placeholders(
    gm: torch.fx.GraphModule,
    merge_dup: bool = False,
    exclude_literals: Optional[List[Any]] = None
):
    """Replace the literals in the graph with placeholder nodes that's created on the fly while we
    traverse the graph, so that the literal arguments in the graph can be matched and replaced

    To use this, the pattern and replacement graph should have the exact same number of literal args
    and they should be used in the exact same order in the pattern and replacement graph.

    If the literal arguments are not used in the same order in pattern and replacement graph, please
    use `_replace_literals_with_existing_placeholders` instead

    Args:
        `gm`: input GraphModule that we'll transform
        `merge_dup`: boolean flag to indicate that if the same literal appears multiple times in
         the graph, whether they should correspond to the same placeholder or not
        `exclude_literals`: a list of literals that will not be replaced with placeholders

    Example:

    # 1. Original Graph
    def pattern(self, x):
        return x + 3

    def replacement(self, x):
        return x - 3

    example_inputs = (torch.randn(1, 3, 3, 3),)
    pattern_gm = _get_aten_graph_module_for_pattern(pattern, example_inputs)
    replacement_gm = _get_aten_graph_module_for_pattern(pattern, example_inptus)

    # 2. Before calling replace literals we'll see the following graph:
    def pattern(self, x):
        return x + 3

    def replacement(self, x):
        return x - 3

    pattern_gm = _replace_literals_with_new_placeholders(pattern_gm)
    replacement_gm = _replace_literals_with_new_placeholders(replacement_gm)

    # 3. After replacing literals with new placeholder nodes

    def pattern(self, x, new_ph):
        return x + new_ph

    def pattern(self, x, new_ph):
        return x - new_ph

    """
    # 初始化最后一个占位符为 None
    last_ph = None
    # 计数器初始化为 0
    cnt = 0
    # 字面值到占位符节点的映射字典，初始化为空字典
    literal_to_ph: Dict[Union[float, bool, int, torch.dtype], Node] = {}
    # 如果未提供要排除的字面值列表，则初始化为空列表
    if exclude_literals is None:
        exclude_literals = []

    # 获取输入规范
    in_spec = gm._in_spec
    # 获取输入参数的规范
    args_spec = in_spec.children_specs[0]
    # 遍历图中的每个节点
    for node in gm.graph.nodes:
        # 检查节点是否为占位符
        if node.op == "placeholder":
            # 更新最后一个占位符节点，并增加计数器
            last_ph = node
            cnt += 1
            # 继续下一个节点的处理
            continue
        
        # 在最后一个占位符节点后插入新节点
        with gm.graph.inserting_after(last_ph):
            # 初始化新的参数列表
            new_args = []
            # 遍历当前节点的参数
            for arg in node.args:
                # 如果参数是字面量且不在排除列表中
                if _is_literal(arg) and arg not in exclude_literals:
                    # 如果允许合并重复字面量，并且字面量已经有对应的占位符
                    if merge_dup and arg in literal_to_ph:
                        # 使用现有的占位符替换字面量
                        new_args.append(literal_to_ph[arg])
                    else:
                        # 创建新的占位符节点，并添加到参数列表中
                        ph_node = gm.graph.placeholder("arg" + str(cnt))
                        new_args.append(ph_node)
                        # 更新参数规范的子节点规范
                        args_spec.children_specs.append(LeafSpec())
                        cnt += 1
                        # 如果允许合并重复字面量，则将字面量映射到新的占位符节点
                        if merge_dup:
                            literal_to_ph[arg] = ph_node
                else:
                    # 将非字面量参数直接添加到新参数列表中
                    new_args.append(arg)
            
            # 将新参数列表转换为元组
            new_args = tuple(new_args)

        # 更新当前节点的参数为新的参数列表
        node.args = new_args

    # 调用对象的后初始化方法来更新节点数、叶子数和子节点数
    args_spec.__post_init__()
    in_spec.__post_init__()

    # 返回更新后的图管理器对象
    return gm
def _replace_literals_with_existing_placeholders(
    gm: torch.fx.GraphModule,
    exclude_literals: Optional[List[Any]] = None,
    literal_to_ph_idx: Optional[Dict[Union[float, int, bool, torch.dtype], int]] = None
):
    """Replace the literals in the graph with **existing** placeholder nodes, so that the literal arguments
    in the graph can be matched and replaced

    To use this, all literal args in the graph should be unique and each of them should correspond
    to exactly one placeholder node

    # 1. Original Graph
    def pattern(self, x_i8, scale, zero_point, quant_min, quant_max):
        return torch.dequantize_per_tensor(x_i8, scale, zero_point, quant_min, quant_max)

    def replacement(x_i8, scale, zero_point, quant_min, quant_max):
        x_i8 = torch.clamp(x_i8, quant_min, quant_max)
        return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)

    example_inputs = (
        torch.randn(1, 3, 3, 3),
        1.0,
        0,
        -128,
        127,
    )
    pattern_gm = _get_aten_graph_module_for_pattern(pattern, example_inputs)
    replacement_gm = _get_aten_graph_module_for_pattern(pattern, example_inptus)

    # 2. Before calling replace literals we'll see the following graph:
    def pattern(self, x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        return torch.dequantize_per_tensor(x_i8, 1.0, 0, -128, 127)

    def replacement(x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        x_i8 = torch.clamp(x_i8, -128, 127)
        return ((x_i8.to(torch.float32) - 0) * 1.0).to(dtype=torch.float32)

    # Note that literal args appear in different order in pattern and replacement graph, so
    # we can't use _replace_literals_with_new_placeholders

    literal_to_ph_idx = {1.0: 1, 0: 2, -128: 3, 127: 4}
    pattern_gm = _replace_literals_with_existing_placeholders(pattern_gm, literal_to_ph_idx)
    replacement_gm = _replace_literals_with_existing_placeholders(replacement_gm, literal_to_ph_idx)

    # 3. After replacing literals with existing placeholder nodes

    def pattern(self, x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        return torch.dequantize_per_tensor(x_i8, scale, zero_point, quant_min, quant_max)

    def replacement(x_i8, scale, zero_point, quant_min, quant_max):
        # scale/zero_point/quant_min/quant_max are burnt in since they are scalar values
        x_i8 = torch.clamp(x_i8, quant_min, quant_max)
        return ((x_i8.to(torch.float32) - zero_point) * scale).to(dtype=torch.float32)
    """
    if exclude_literals is None:
        # 如果 exclude_literals 未指定，默认为一个空列表
        exclude_literals = []

    if literal_to_ph_idx is None:
        # 如果 literal_to_ph_idx 未指定，默认为一个空字典
        literal_to_ph_idx = {}

    # 获取图中所有操作为 "placeholder" 的节点，这些节点是占位符节点
    phs = [node for node in gm.graph.nodes if node.op == "placeholder"]
    # 遍历图中的每个节点
    for node in gm.graph.nodes:
        # 检查节点操作是否为 "call_function"，如果不是，则跳过当前节点
        if node.op != "call_function":
            continue
        
        # 初始化一个新的参数列表，用于存储替换后的参数
        new_args = []
        
        # 遍历当前节点的参数列表
        for arg in node.args:
            # 检查参数是否为字面量并且未在排除列表中，并且存在于字面量到占位符索引的映射中
            if _is_literal(arg) and arg not in exclude_literals and arg in literal_to_ph_idx:
                # 获取参数对应的占位符索引
                ph_idx = literal_to_ph_idx[arg]
                # 根据索引获取对应的占位符节点
                ph_node = phs[ph_idx]
                # 将占位符节点添加到新的参数列表中
                new_args.append(ph_node)
            else:
                # 如果不满足上述条件，则将原始参数添加到新的参数列表中
                new_args.append(arg)
        
        # 将新的参数列表转换为元组，并将其赋值给当前节点的参数
        new_args = tuple(new_args)
        node.args = new_args
    
    # 返回更新后的图对象
    return gm
# TODO: 在导出过程中处理此问题，并且不要在另一个 GraphModule 中包装模型
# 在准备和转换过程中

def _disallow_eval_train(model: GraphModule):
    """
    禁止在给定的 GraphModule 上调用 `model.train()` 或 `model.eval()`。
    这对于已导出的模型很有用，因为这些方法的行为并非按预期执行。
    """
    error_message = \
        """
        调用 train() 或 eval() 不支持导出的模型。
        请调用 `torch.ao.quantization.move_exported_model_to_train(model)`（或 eval）。

        如果无法替换对 `model.train()` 和 `model.eval()` 的调用，您可以通过调用
        `torch.ao.quantization.allow_exported_model_train_eval(model)` 来覆盖这些方法的行为，
        这将自动为您执行上述操作。请注意，这对于在训练和评估模式之间切换行为有限，仅应用于特殊操作，如 dropout 和 batchnorm。
        """

    def _train(self, mode: bool = True):
        raise NotImplementedError(error_message)

    def _eval(self, mode: bool = True):
        raise NotImplementedError(error_message)

    # 将 _train 方法绑定到 model.train()
    model.train = types.MethodType(_train, model)  # type: ignore[method-assign]
    # 将 _eval 方法绑定到 model.eval()
    model.eval = types.MethodType(_eval, model)  # type: ignore[method-assign]
    
    # 返回修改后的模型对象
    return model
```