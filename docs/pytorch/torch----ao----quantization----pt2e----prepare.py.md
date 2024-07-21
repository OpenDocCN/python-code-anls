# `.\pytorch\torch\ao\quantization\pt2e\prepare.py`

```py
# mypy: allow-untyped-defs
# 引入 PyTorch 相关模块和函数
import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.prepare import (
    _insert_obs_or_fq,
    _save_state,
    _is_activation_post_process_node,
    _create_obs_or_fq_from_qspec,
)
from torch.fx import (
    GraphModule,
    Graph,
    Node,
)
from torch.fx.node import Argument

# 引入量化配置相关模块和函数
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from typing import Dict, Tuple, Union, Any, Optional
from torch.ao.quantization.quantizer import (
    EdgeOrNode,
    SharedQuantizationSpec,
    QuantizationSpecBase,
)
from torch.ao.quantization import ObserverOrFakeQuantize

# TODO: make pt2e folder private?
# 定义模块公开的接口
__all__ = [
    "prepare",
]

def _find_root_edge_or_node(edge_or_node: EdgeOrNode, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]) -> EdgeOrNode:
    """Find the root node for the sharing tree
    Args:
        edge_or_node: edge/node that we want to find the root
        shared_with_map: each edge/node points to the parent, the root node will points to itself

    Returns:
        root edge/node
    """
    # 获取当前节点的父节点
    parent = shared_with_map[edge_or_node]
    # 如果父节点即为当前节点，说明已经找到根节点
    if parent == edge_or_node:
        return edge_or_node
    # 递归寻找根节点，并进行路径压缩
    root = _find_root_edge_or_node(parent, shared_with_map)
    # 路径压缩，将当前节点直接指向根节点，加速下次查找
    shared_with_map[edge_or_node] = root
    return root

def _union(parent: EdgeOrNode, child: EdgeOrNode, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]) -> None:
    """Merge the subtree for `child` with `parent`, the order is important here
    """
    # 获取 parent 和 child 的根节点
    root_parent = _find_root_edge_or_node(parent, shared_with_map)
    root_child = _find_root_edge_or_node(child, shared_with_map)
    # 将 child 的根节点指向 parent 的根节点，合并两棵树
    shared_with_map[root_child] = root_parent

def _update_shared_with(child: EdgeOrNode, qspec: QuantizationSpecBase, shared_with_map: Dict[EdgeOrNode, EdgeOrNode]):
    """Update the `shared_with_map` based on the qspec, this applies the `SharedQuantizationSpec`
    configuration and established the relationship between `edge_or_node` with the edge/node that it
    is pointing to, we'll use this information in the end to get the group id
    """
    # 如果 qspec 是 SharedQuantizationSpec 类型，则更新 shared_with_map
    if isinstance(qspec, SharedQuantizationSpec):
        parent = qspec.edge_or_node
        # 将 child 指向 parent，表示它们共享量化配置
        _union(parent, child, shared_with_map)

def _unwrap_shared_qspec(
    qspec: QuantizationSpecBase,
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase],
    shared_with_map: Dict[EdgeOrNode, EdgeOrNode]
) -> QuantizationSpecBase:
    """Unwraps qspec to get the final root qspec (non SharedQuantizationSpec)

    This function resolves the final quantization specification for a node or edge
    by recursively resolving through the shared configurations until the ultimate
    non-shared specification is found.

    Args:
        qspec: The current quantization specification possibly wrapped in SharedQuantizationSpec
        edge_or_node_to_qspec: Mapping from edge or node to its respective quantization specification
        shared_with_map: Mapping used to resolve shared relationships between edges or nodes

    Returns:
        Quantization specification that is not wrapped in SharedQuantizationSpec
    """
    # 如果 qspec 是 SharedQuantizationSpec 类型，继续解析其真正的量化配置
    if isinstance(qspec, SharedQuantizationSpec):
        root_qspec = _unwrap_shared_qspec(edge_or_node_to_qspec[qspec.edge_or_node], edge_or_node_to_qspec, shared_with_map)
        return root_qspec
    else:
        return qspec  # 返回非 SharedQuantizationSpec 类型的量化配置
    # 如果给定的 qspec 是 SharedQuantizationSpec 的实例
    if isinstance(qspec, SharedQuantizationSpec):
        # 获取与 qspec 共享的边或节点
        sharing_with = qspec.edge_or_node
        # 找到与 sharing_with 相关的根边或节点
        root = _find_root_edge_or_node(sharing_with, shared_with_map)
        # 根据根边或节点找到对应的 qspec
        qspec = edge_or_node_to_qspec[root]
        # 递归展开共享的 qspec，继续解析
        return _unwrap_shared_qspec(qspec, edge_or_node_to_qspec, shared_with_map)
    # 如果 qspec 不是 SharedQuantizationSpec 的实例，直接返回它
    return qspec
def _has_same_dtype(qspec_a: QuantizationSpecBase, qspec_b: QuantizationSpecBase):
    # 检查两个量化规格对象是否都具有 'dtype' 属性，并且这两个属性的值相等
    return (
        hasattr(qspec_a, "dtype") and
        hasattr(qspec_b, "dtype") and
        qspec_a.dtype == qspec_b.dtype
    )

def _has_same_is_dynamic(qspec_a: QuantizationSpecBase, qspec_b: QuantizationSpecBase):
    # 检查两个量化规格对象是否都具有 'is_dynamic' 属性，并且这两个属性的值相等
    return (
        hasattr(qspec_a, "is_dynamic") and
        hasattr(qspec_b, "is_dynamic") and
        qspec_a.is_dynamic == qspec_b.is_dynamic
    )

def _get_edge_or_node_to_qspec(model: torch.fx.GraphModule) -> Dict[EdgeOrNode, QuantizationSpecBase]:
    """Get a map from EdgeOrNode to quantization spec based on annotations on the nodes
    获取从 EdgeOrNode 到量化规格的映射，基于节点上的注释
    """
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase] = {}
    for n in model.graph.nodes:
        if hasattr(n, "meta") and "quantization_annotation" in n.meta:
            qa = n.meta["quantization_annotation"]
            # 遍历节点上的量化注释，将输入到节点的边或节点与对应的量化规格映射存入字典
            for input_to_n, qspec in qa.input_qspec_map.items():
                input_edge = (input_to_n, n)
                edge_or_node_to_qspec[input_edge] = qspec
            # 如果节点有输出的量化规格，将输出节点与其量化规格映射存入字典
            if qa.output_qspec is not None:
                output_node = n
                qspec = qa.output_qspec
                edge_or_node_to_qspec[output_node] = qspec
    return edge_or_node_to_qspec

def _union_input_edge_with(input_edge, input_edge_root_qspec, edge_or_node, edge_or_node_to_qspec, shared_with_map):
    """Union input edge with another edge or node, used in implicit sharing to point the current input
    edge to other user edges of the producer node, or the output of producer node since these are
    referring to the same Tensor
    将输入边与另一个边或节点进行联合，用于隐式共享，将当前输入边指向生产者节点的其他用户边，或者生产者节点的输出，因为它们引用相同的张量
    """
    root_qspec = None
    if edge_or_node in edge_or_node_to_qspec:
        qspec = edge_or_node_to_qspec[edge_or_node]
        # 获取包含隐式共享的根量化规格
        root_qspec = _unwrap_shared_qspec(qspec, edge_or_node_to_qspec, shared_with_map)
    # TODO: add assertions for types of root qspecs
    if (
        root_qspec is not None and
        _has_same_dtype(root_qspec, input_edge_root_qspec) and
        _has_same_is_dynamic(root_qspec, input_edge_root_qspec)
    ):
        # 如果根量化规格不为 None，并且其 dtype 和 is_dynamic 与输入边的根量化规格相同，则进行联合操作
        # 将输入边指向根节点或边
        _union(edge_or_node, input_edge, shared_with_map)


def _get_edge_or_node_to_group_id(edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase]) -> Dict[EdgeOrNode, int]:
    """Map from edge/node to the group ID, generated from quantization annotations,
    edge/node with the same group ID should use the same observer/fake_quant instance
    从边/节点映射到组 ID，根据量化注释生成，具有相同组 ID 的边/节点应使用相同的观察者/fake_quant 实例
    """
    edge_or_node_to_group_id: Dict[EdgeOrNode, int] = {}
    for edge_or_node in edge_or_node_to_qspec:
        # 对于每个边或节点，根据其量化规格生成组 ID 并存入字典
        group_id = hash(edge_or_node_to_qspec[edge_or_node])
        edge_or_node_to_group_id[edge_or_node] = group_id
    return edge_or_node_to_group_id
    """
    we'll assume sharing between the output of op1 and input of (op1 -> op2) since these are the same Tensor.

    Figuring out the correct group ID for all edge/node is a standard union find problem:
    https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/

    Args:
        edge_or_node_to_qspec: Dictionary from edge_or_node to the qspec, derived from annotations
    Returns:
        edge_or_node_to_group_id: Dictionary from edge_or_node to group_id (int), all edge or node that
        belongs to the same group should have the same id

    Example:
        op2 -> cat1 -> cat2
           op1 /        /
                     op3
        edge_or_node_to_qspec: {
            op1: int8_qspec,
            op2: int8_qspec,
            (op1, cat1): int8_qspc,
            (op2, cat1): SharedQuantizationSpec((op1, cat1)),
            cat1: SharedQuantizationSpec((op1, cat1)),
            (op3, cat2): int8_qspec,
            (cat1, cat2): SharedQuantizationSpec((op3, cat2)),
            cat2: SharedQuantizationSpec((op3, cat2)),
        }

        edge_or_node_to_group_id = _get_edge_or_node_to_group_id(edge_or_node_to_qspec)
        edge_or_node_to_group_id: {
            op1: 1,
            op2: 1,
            (op1, cat1): 1,
            (op2, cat1): 1,
            cat1: 1,
            (op3, cat2): 1,
            (cat1, cat2): 1,
            cat2: 1,
        }
        # everything are in the same group because (cat1) and (cat1, cat2) are implicitly shared, which
        # connects the two sharing group around cat1 and cat2 op due to transitive sharing
    """
    # means the observer of key should be shared with observer with value, by default it will
    # be shared with itself
    shared_with_map: Dict[EdgeOrNode, EdgeOrNode] = {k: k for k in edge_or_node_to_qspec.keys()}
    # 遍历 edge_or_node_to_qspec 字典中的每一个键值对
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        # 检查当前的 edge_or_node 是否为 torch.fx.Node 类型
        if isinstance(edge_or_node, torch.fx.Node):
            # 如果是节点类型，则将 output_node 设置为当前的 edge_or_node
            output_node = edge_or_node
            # 更新 output_node 的共享信息
            _update_shared_with(output_node, qspec, shared_with_map)
        else:
            # 如果不是节点类型，则将其作为 input_edge
            input_edge = edge_or_node
            # 解包共享的 qspec 信息
            input_edge_root_qspec = _unwrap_shared_qspec(qspec, edge_or_node_to_qspec, shared_with_map)

            # 断言 input_edge 是一个元组
            assert isinstance(input_edge, tuple)
            # 解包元组，获取其中的两个元素 arg 和 n
            arg, n = input_edge

            # 检查是否允许隐式共享
            if n.meta["quantization_annotation"].allow_implicit_sharing:
                # 注意：这里的顺序很重要，首先与其他用户共享，然后再与先前的输出共享，
                # 因为反向顺序可能会导致循环依赖的问题

                # 与生产者节点的其他用户共享
                # (arg, user)
                if not isinstance(arg, Node) or not isinstance(n, Node):
                    raise Exception(f"Expected input_edge to have type Tuple[Node, Node], but got: {arg, n}")  # noqa: TRY002
                for user in arg.users:
                    if user is n:
                        continue
                    arg_to_user_edge = (arg, user)
                    # 将当前的 input_edge 与 arg_to_user_edge 进行共享
                    _union_input_edge_with(
                        input_edge,
                        input_edge_root_qspec,
                        arg_to_user_edge,
                        edge_or_node_to_qspec,
                        shared_with_map
                    )

                # 与生产者节点的输出共享
                _union_input_edge_with(input_edge, input_edge_root_qspec, arg, edge_or_node_to_qspec, shared_with_map)

            # 更新 input_edge 的共享信息
            _update_shared_with(input_edge, qspec, shared_with_map)

    # 现在已经确定了所有边和节点之间的共享关系，可以分配组 ID 了
    cur_group_id = 0
    # 创建一个空字典，用于存储每个边或节点对应的组 ID
    edge_or_node_to_group_id: Dict[EdgeOrNode, int] = {}
    # 遍历共享映射中的每个边缘或节点
    for edge_or_node in shared_with_map.keys():
        # 找到当前边缘或节点的根节点
        root = _find_root_edge_or_node(edge_or_node, shared_with_map)
        # 如果根节点不在边缘或节点到组ID的映射中，则将其添加，并分配新的组ID
        if root not in edge_or_node_to_group_id:
            edge_or_node_to_group_id[root] = cur_group_id
            cur_group_id += 1
        # 将当前边缘或节点映射到与其根节点相同的组ID
        edge_or_node_to_group_id[edge_or_node] = edge_or_node_to_group_id[root]
    
    # 返回边缘或节点到组ID的映射
    return edge_or_node_to_group_id
# 生成 EdgeOrNode 到观察者/伪量化实例的映射，确保具有相同 group_id 的 EdgeOrNode 共享同一个观察者或伪量化实例
def _get_obs_or_fq_map(
    edge_or_node_to_group_id: Dict[EdgeOrNode, int],
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase],
    is_qat: bool
) -> Dict[EdgeOrNode, ObserverOrFakeQuantize]:
    """Generates the EdgeOrNode to observer/fake_quant instances
    Makes sure that for EdgeOrNode that has the same group_id should have the same observer or fake quant
    instances
    """
    # 初始化空字典，用于存储 EdgeOrNode 到 ObserverOrFakeQuantize 的映射关系
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize] = {}
    # 初始化空字典，用于存储 group_id 到 ObserverOrFakeQuantize 的映射关系
    group_id_to_obs_or_fq: Dict[int, ObserverOrFakeQuantize] = {}
    
    # 遍历 edge_or_node_to_qspec 字典中的每一个元素
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        # 获取 edge_or_node 对应的 group_id
        group_id = edge_or_node_to_group_id[edge_or_node]
        # 如果 group_id 不在 group_id_to_obs_or_fq 中，则调用 _create_obs_or_fq_from_qspec 函数创建相应的 ObserverOrFakeQuantize 实例
        if group_id not in group_id_to_obs_or_fq:
            group_id_to_obs_or_fq[group_id] = _create_obs_or_fq_from_qspec(qspec, obs_or_fq_map, is_qat)
        # 将 edge_or_node 映射到相应的 ObserverOrFakeQuantize 实例
        obs_or_fq_map[edge_or_node] = group_id_to_obs_or_fq[group_id]
    
    # 返回生成的 EdgeOrNode 到 ObserverOrFakeQuantize 实例的映射字典
    return obs_or_fq_map

# 给定一个节点 `node` 和一个参数 `arg`，如果必要，在 `node` 和 `arg` 之间插入输入观察者
def _maybe_insert_input_observer_for_arg_or_kwarg(
    node: Union[Node, Any],
    arg: Argument,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> Argument:
    """
    Given a `node` and an `arg`, inserts an input observer between
    `node` and `arg` if necessary.
    """
    # 对于类似 torch.cat([x0, x1]) 的操作，遍历列表内的每个元素
    if isinstance(arg, (list, tuple)):
        new_arg_to_return = []
        for inner_arg in arg:
            # 递归调用 _maybe_insert_input_observer_for_arg_or_kwarg 处理内部元素
            new_inner_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
                node, inner_arg, qconfig, model, named_modules, obs_or_fq_map, is_qat,
            )
            new_arg_to_return.append(new_inner_arg)
        return type(arg)(new_arg_to_return)

    # 如果 arg 不是 Node 对象，则直接返回 arg
    if not isinstance(arg, Node):
        return arg
    # 确保 arg 是 Node 对象
    assert isinstance(arg, Node)
    # 默认情况下（没有观察者），new_arg 为原始的 arg
    new_arg = arg

    # 查找到原始的 arg 节点，跳过插入的观察者/伪量化节点
    original_arg = arg
    while _is_activation_post_process_node(original_arg, named_modules):
        original_arg = original_arg.args[0]  # type: ignore[assignment]
    assert isinstance(original_arg, Node), f"expect original argument to be a Node, but got: {type(original_arg)}"

    # 构建输入边
    input_edge = (original_arg, node)
    # 如果 input_edge 不在 obs_or_fq_map 中，则直接返回 new_arg
    if input_edge not in obs_or_fq_map:
        return new_arg
    # 获取 input_edge 对应的观察者/伪量化实例
    input_edge_obs_or_fq = obs_or_fq_map[input_edge]
    # 如果 input_edge_obs_or_fq 为 None，则直接返回 new_arg
    if input_edge_obs_or_fq is None:
        return new_arg

    # 获取 original_arg 对应的输出观察者/伪量化实例
    arg_as_output_obs_or_fq = obs_or_fq_map.get(original_arg, None)
    # 如果 arg_as_output_obs_or_fq 是输出观察者/伪量化实例，并且使用相同的实例作为 input_edge，则重用已插入的观察者/伪量化实例
    # 如果arg_as_output_obs_or_fq不为None，并且其与input_edge_obs_or_fq是同一个对象
    if arg_as_output_obs_or_fq is not None and id(arg_as_output_obs_or_fq) == id(input_edge_obs_or_fq):
        # 返回当前arg_as_output_obs_or_fq对象，因为不需要插入新的观察者或伪量化节点
        return new_arg

    # 否则，我们将插入一个新的观察者/伪量化节点

    existing_obs_node = None
    # 遍历arg的用户，查找是否已经存在相同的观察者实例被插入到其他节点中
    # 示例:
    # conv1 -> obs1 -> existing_obs -> conv2
    #             \ -> conv3
    #
    # 如果已经存在相同的观察者实例，我们不需要插入新的观察者节点，而是复用已有的观察者节点
    for maybe_obs_node in arg.users.keys():
        # 如果不是激活后处理节点，则跳过
        if not _is_activation_post_process_node(maybe_obs_node, named_modules):
            continue
        # 获取可能的观察者模块
        maybe_obs_mod = named_modules[maybe_obs_node.target]  # type: ignore[index]
        # 如果观察者模块与输入的观察者或伪量化节点是同一个对象
        if id(maybe_obs_mod) == id(input_edge_obs_or_fq):
            # 返回可能的观察者节点
            return maybe_obs_node

    # 插入新的观察者或伪量化节点，并返回新的参数
    new_arg = _insert_obs_or_fq(arg, input_edge_obs_or_fq, model, named_modules, model.graph)
    return new_arg
def _maybe_insert_input_observers_for_node(
    node: Node,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> None:
    """
    If needed, inserts observers to the input args and kwargs of `node`.
    Note: modifies `node` inplace.

    For example, if cur_node needs an observer after prev_node, we change from

      prev_node -> cur_node

    To

      prev_node -> obs -> cur_node

    """
    # 需要检查每个输入参数。如果参数的目标数据类型与当前节点的目标数据类型不匹配，则插入一个观察者。
    new_args = []
    # 旧参数到新参数的映射，用于更新数值调试句柄映射
    remap = {}
    for arg in node.args:
        new_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node, arg, qconfig, model, named_modules, obs_or_fq_map, is_qat,
        )
        new_args.append(new_arg)
        remap[arg] = new_arg

    if "numeric_debug_handle" in node.meta:
        # 定义一个函数来重新映射参数
        def remap_fn(x):
            return remap.get(x, x)

        # 更新数值调试句柄映射中的参数
        numeric_debug_handle = node.meta["numeric_debug_handle"]
        node.meta["numeric_debug_handle"] = {remap_fn(k): v for k, v in numeric_debug_handle.items()}

    # Clone 操作有 memory_format 参数，zeros_like 操作有 pin_memory 参数，gelu 操作有 approximate 参数，这些在导出的图中保持不变。
    # 这里是对它们的一个解决方法。
    assert (
        node.target == torch.ops.aten.clone.default or
        node.target == torch.ops.aten.zeros_like.default or
        node.target == torch.ops.aten.gelu.default or
        len(node.kwargs) == 0
    ), " expecting kwargs for aten op IR to be empty"

    # 将新的参数赋值给节点，直接修改原节点
    node.args = tuple(new_args)

def _maybe_insert_output_observer_for_node(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> Optional[Node]:
    if node in obs_or_fq_map:
        output_act_obs_or_fq = obs_or_fq_map[node]
        return _insert_obs_or_fq(node, output_act_obs_or_fq, model, named_modules, graph)
    return None

def _maybe_insert_input_and_output_observers_for_node(
    node: Node,
    model: torch.fx.GraphModule,
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
):
    # 如果当前节点有量化注释，则进行输入和输出观察者的插入
    this_node_quantization_annotation = node.meta["quantization_annotation"] if "quantization_annotation" in node.meta else None
    if this_node_quantization_annotation is None:
        return

    # 获取模型中命名的所有模块
    named_modules = dict(model.named_modules(remove_duplicate=False))
    # 插入当前节点的输入观察者
    _maybe_insert_input_observers_for_node(
        node,
        None,  # qconfig 参数为 None
        model,
        named_modules,
        obs_or_fq_map,
        is_qat,
    )

    # 如果节点的 meta 中存在 'val' 并且其值是 FakeTensor 类型，则输出为张量
    # 如果输出不是张量，则直接返回，不进行后续处理
    if not output_is_a_tensor:
        return

    # 如果需要的话，为当前节点插入输出观察节点
    maybe_output_obs_node = _maybe_insert_output_observer_for_node(
        node, model, named_modules, model.graph, obs_or_fq_map, is_qat)

    # 如果没有插入输出观察节点，则直接返回
    if maybe_output_obs_node is None:
        return

    # 更新原始节点的使用者，使其使用新插入的输出观察节点
    # 例如，将原始的用户节点列表保存在 orig_users 中
    orig_users = list(node.users.keys())
    for user_node in orig_users:
        # 如果用户节点已经是新插入的输出观察节点，则跳过
        if user_node is maybe_output_obs_node:
            continue
        # 替换用户节点对当前节点的引用为对新插入的输出观察节点的引用
        user_node.replace_input_with(node, maybe_output_obs_node)
def prepare(
    model: GraphModule,
    node_name_to_scope: Dict[str, Tuple[str, type]],
    is_qat: bool,
) -> GraphModule:
    # 获取原始的节点列表，因为我们在处理过程中会修改图结构，所以使用原始节点列表而不是 model.graph.nodes。
    nodes_before_observation = list(model.graph.nodes)

    # 在高层次上，我们构建一个从 EdgeOrNode 到 observer_or_fake_quant 实例的映射
    # 所有属于同一组的边/节点将使用同一个实例
    # 当插入观察者时，我们只需查询此映射即可获取正确的 observer_or_fake_quant 实例
    edge_or_node_to_qspec = _get_edge_or_node_to_qspec(model)
    edge_or_node_to_group_id = _get_edge_or_node_to_group_id(edge_or_node_to_qspec)
    obs_or_fq_map = _get_obs_or_fq_map(edge_or_node_to_group_id, edge_or_node_to_qspec, is_qat)

    # 遍历原始节点列表，并尝试为每个节点插入输入和输出观察者
    for node in nodes_before_observation:
        # TODO: 简化插入观察者的逻辑
        _maybe_insert_input_and_output_observers_for_node(node, model, obs_or_fq_map, is_qat)

    # 创建新的 GraphModule 实例，用于返回最终处理后的模型
    model = GraphModule(model, model.graph)

    # 保存当前模型的状态和配置信息
    _save_state(
        model,
        {},  # node_name_to_qconfig，暂时为空字典
        node_name_to_scope,
        PrepareCustomConfig(),  # 创建一个空的 PrepareCustomConfig 实例
        {},  # equalization_node_name_to_qconfig，暂时为空字典
        QConfigMapping(),  # 创建一个空的 QConfigMapping 实例
        is_qat,
        set()  # observed_node_names，暂时为一个空集合
    )
    
    # 返回处理后的模型对象
    return model
```