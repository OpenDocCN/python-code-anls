# `.\pytorch\torch\_export\passes\replace_set_grad_with_hop_pass.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import contextlib  # 上下文管理模块
import copy  # 复制对象模块

import torch  # PyTorch库
from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled  # 引入PyTorch的包装函数

from ..utils import (
    node_inline_,  # 从自定义工具包中导入节点内联函数
    node_replace_,  # 从自定义工具包中导入节点替换函数
    nodes_filter,   # 从自定义工具包中导入节点过滤函数
    nodes_first,    # 从自定义工具包中导入首个节点获取函数
    nodes_map,      # 从自定义工具包中导入节点映射函数
    sequential_split,  # 从自定义工具包中导入顺序拆分函数
)


def _is_set_grad_enabled_node(node: torch.fx.Node):
    # 检查节点是否为调用函数操作，并且目标函数是torch._C._set_grad_enabled
    return (
        node
        and node.op == "call_function"
        and node.target == torch._C._set_grad_enabled
    )


def _is_set_grad_enabled_sub_mod(node: torch.fx.Node, omit_if_same_with_ambient=False):
    # 检查节点是否为调用模块操作，并且该模块调用了torch._C._set_grad_enabled函数
    if node.op == "call_module":
        assert isinstance(node.target, str)
        subgm = getattr(node.graph.owning_module, node.target)
        # 获取子模块的首个非占位符节点
        first_non_ph = nodes_first(
            subgm.graph.nodes, lambda node: node.op != "placeholder"
        )
        if (
            first_non_ph
            and first_non_ph.op == "call_function"
            and first_non_ph.target == torch._C._set_grad_enabled
        ):
            # 如果忽略与当前环境相同的情况，则比较子模块的梯度开启状态
            return (
                first_non_ph.args[0] != torch.is_grad_enabled()
                if omit_if_same_with_ambient
                else True
            )
    return False


def _replace_with_hop(node: torch.fx.Node):
    # 确定节点为调用模块操作
    assert node.op == "call_module"
    graph: torch.fx.Graph = node.graph
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    # 获取子图中调用torch._C._set_grad_enabled函数的节点集合
    set_grad_nodes = nodes_filter(sub_graph.nodes, _is_set_grad_enabled_node)


def _remove_set_grad_and_inline(node: torch.fx.Node):
    # 确定节点为调用模块操作
    assert node.op == "call_module"
    graph: torch.fx.Graph = node.graph
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    # 将子图中所有调用torch._C._set_grad_enabled函数的节点移除，并内联当前节点
    nodes_map(
        sub_graph.nodes,
        lambda n: sub_graph.erase_node(n) if _is_set_grad_enabled_node(n) else n,
    )
    node_inline_(node)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature
):
    """
    Helper function for replace_set_grad_with_hop_pass().
    Split the graph module into multiple subgraphs based on the set_grad_enabled nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    """
    # 如果没有set_grad_enabled节点，则返回原始图模块
    need_replacing = False
    for node in gm.graph.nodes:
        if _is_set_grad_enabled_node(node):
            need_replacing = True
    # 如果需要进行替换操作
    if need_replacing:
        # sequential_split 函数返回一个新的图模块，可能具有不同的输出参数名称。我们需要修复图的签名。
        new_gm = sequential_split(gm, _is_set_grad_enabled_node)

        # 创建一个空的上下文管理器对象
        replace_ctx = contextlib.nullcontext()
        
        # 如果存在图签名，则进行深拷贝
        new_signature = None
        if graph_signature is not None:
            new_signature = copy.deepcopy(graph_signature)
            
            # 获取新图中的输出节点，并验证其与签名中输出规范的匹配性
            new_gm_out_node = next(reversed(new_gm.graph.find_nodes(op="output")))
            assert new_gm_out_node.op == "output" and len(
                new_gm_out_node.args[0]
            ) == len(new_signature.output_specs)
            
            # 遍历输出节点和输出规范，确保它们的名称匹配
            for arg_node, out_spec in zip(
                new_gm_out_node.args[0], new_signature.output_specs
            ):
                if out_spec.arg.name != arg_node.name:
                    out_spec.arg.name = arg_node.name
            
            # 设置新图的替换钩子，用于替换操作
            replace_ctx = new_gm._set_replace_hook(new_signature.get_replace_hook())  # type: ignore[assignment]

        # 进入替换操作的上下文环境
        with replace_ctx:
            
            # 定义内部函数，根据条件进行内联或替换为 HOP
            def _maybe_inline_or_replace_with_hop(node: torch.fx.Node):
                if _is_set_grad_enabled_sub_mod(node, omit_if_same_with_ambient=True):
                    _replace_with_hop(node)
                else:
                    _remove_set_grad_and_inline(node)

            # 对图中的每个节点进行映射操作
            nodes_map(
                list(new_gm.graph.nodes),
                lambda node: (
                    _maybe_inline_or_replace_with_hop(node)
                    if node.op == "call_module"
                    else node
                ),
            )
        
        # 重新编译新的图模块
        new_gm.recompile()
        
        # 返回更新后的图模块和图签名
        return new_gm, new_signature

    # 如果不需要替换操作，则直接返回原始的图模块和图签名
    return gm, graph_signature
# 将输入的图模块gm分解成顺序子图，并可能内联子图，返回新的图模块和签名
def replace_set_grad_with_hop_pass(gm: torch.fx.GraphModule, graph_signature):
    new_gm, new_signature = _sequential_split_and_maybe_inline_subgraphs(
        gm, graph_signature
    )
    # 递归调用替换子图中的"get_attr"节点
    for node in new_gm.graph.nodes:
        if node.op == "get_attr":
            # 获取节点目标对应的子图模块
            subgm = getattr(new_gm, node.target)
            # 如果子图模块不是torch.fx.GraphModule类型，则跳过
            if not isinstance(subgm, torch.fx.GraphModule):
                continue
            # 递归调用本函数，替换子图模块中的节点
            new_subgm, _ = replace_set_grad_with_hop_pass(subgm, None)
            # 将新的子图模块设置回原始图模块的属性中
            setattr(new_gm, node.target, new_subgm)

    # 重新编译新的图模块
    new_gm.recompile()
    # 对新的图模块进行Lint检查
    new_gm.graph.lint()
    # 返回更新后的图模块和签名
    return new_gm, new_signature
```