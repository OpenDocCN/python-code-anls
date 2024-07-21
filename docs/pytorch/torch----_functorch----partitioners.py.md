# `.\pytorch\torch\_functorch\partitioners.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和库
import copy  # 复制对象和数据结构
import functools  # 提供高阶函数的工具
import heapq  # 提供堆队列算法
import itertools  # 提供高效的迭代工具
import logging  # 记录日志消息
import math  # 数学函数库
import operator  # 提供Python的内置运算符函数
import os  # 提供与操作系统交互的功能
from collections import defaultdict  # 提供默认值的字典
from dataclasses import dataclass, replace  # 提供数据类的支持
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union  # 提供类型提示支持

import torch  # 引入PyTorch库
import torch._inductor.inductor_prims  # PyTorch的内部操作原语
import torch.fx as fx  # PyTorch的特征图工具
import torch.utils._pytree as pytree  # PyTorch的树结构操作工具
from torch.fx.experimental._backward_state import BackwardState  # PyTorch的特征图后向状态
from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types  # PyTorch的代理张量相关工具
from torch.fx.experimental.sym_node import magic_methods, method_to_operator  # PyTorch的符号节点操作
from torch.fx.experimental.symbolic_shapes import (
    find_symbol_binding_fx_nodes,  # 寻找绑定符号的特征图节点
    free_symbols,  # 自由符号的集合
    hint_int,  # 整数提示
    is_symbol_binding_fx_node,  # 判断是否为符号绑定的特征图节点
)
from torch.fx.passes import graph_drawer  # 特征图的图形绘制工具
from torch.utils.checkpoint import CheckpointPolicy  # PyTorch的检查点策略
from . import config  # 导入本地配置文件
from ._aot_autograd.logging_utils import get_aot_graph_name  # 获取AOT图的名称的日志工具
from .compile_utils import fx_graph_cse, get_aten_target  # 特征图的图形CSE优化工具和获取ATen目标工具

if TYPE_CHECKING:
    import sympy  # 导入符号计算库（仅用于类型检查）

AOT_PARTITIONER_DEBUG = config.debug_partitioner  # 从配置中获取AOT分区调试标志
log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

aten = torch.ops.aten  # PyTorch的ATen操作命名空间
prims = torch.ops.prims  # PyTorch的Prims操作命名空间


@dataclass
class OpTypes:
    """Class for keeping track of different operator categories"""

    fusible_ops: Set[Callable]  # 可融合操作集合
    compute_intensive_ops: Set[Callable]  # 计算密集型操作集合
    random_ops: Set[Callable]  # 随机操作集合
    view_ops: Set[Callable]  # 视图操作集合
    recomputable_ops: Set[Callable]  # 可重新计算操作集合

    def is_fusible(self, node: fx.Node):
        return get_aten_target(node) in self.fusible_ops  # 判断特征图节点是否属于可融合操作

    def is_compute_intensive(self, node: fx.Node):
        return get_aten_target(node) in self.compute_intensive_ops  # 判断特征图节点是否属于计算密集型操作

    def is_random(self, node: fx.Node):
        return get_aten_target(node) in self.random_ops  # 判断特征图节点是否属于随机操作

    def is_view(self, node: fx.Node):
        return get_aten_target(node) in self.view_ops  # 判断特征图节点是否属于视图操作

    def is_recomputable(self, node: fx.Node):
        return get_aten_target(node) in self.recomputable_ops  # 判断特征图节点是否属于可重新计算操作


@dataclass
class NodeInfo:
    # Be careful about iterating over these explicitly, as their order may not
    # be deterministic
    inputs: List[fx.Node]  # 特征图节点的输入列表
    _required_fw_nodes: Set[fx.Node]  # 所需的前向节点集合
    required_bw_nodes: Set[fx.Node]  # 所需的反向节点集合
    unclaimed_nodes: Set[fx.Node]  # 未声明归属的节点集合
    fw_order: Dict[fx.Node, int]  # 前向节点的顺序映射

    @functools.cached_property
    def required_fw_nodes(self) -> List[fx.Node]:
        return sorted(
            (n for n in self._required_fw_nodes), key=lambda n: self.fw_order[n]
        )  # 根据前向节点的顺序属性对所需前向节点进行排序

    def is_required_fw(self, n: fx.Node) -> bool:
        return n in self._required_fw_nodes  # 判断节点是否为所需的前向节点

    def is_required_bw(self, n: fx.Node) -> bool:
        return n in self.required_bw_nodes  # 判断节点是否为所需的反向节点

    def is_unclaimed(self, n: fx.Node) -> bool:
        return n in self.unclaimed_nodes  # 判断节点是否为未声明归属的节点

    def get_fw_order(self, n: fx.Node) -> int:
        assert n in self._required_fw_nodes, f"Node {n} not in fw nodes!"
        return self.fw_order[n]  # 获取节点的前向顺序


@dataclass
class MinCutOptions:
    ban_if_used_far_apart: bool  # 远距离使用时禁止选项
    ban_if_long_fusible_chains: bool  # 长融合链时禁止选项
    # 定义一个名为 ban_if_materialized_backward 的布尔变量，用于标识是否禁止向后材料化
    ban_if_materialized_backward: bool
    # 定义一个名为 ban_if_not_in_allowlist 的布尔变量，用于标识是否禁止不在允许列表中的情况
    ban_if_not_in_allowlist: bool
    # 定义一个名为 ban_if_reduction 的布尔变量，用于标识是否禁止减少的情况
    ban_if_reduction: bool
def must_recompute(node: fx.Node) -> bool:
    # 检查节点的元数据中是否有 "recompute" 键，并且其对应的值在重新计算策略列表中
    return node.meta.get("recompute", None) in [
        CheckpointPolicy.MUST_RECOMPUTE,
        CheckpointPolicy.PREFER_RECOMPUTE,
    ]


def has_recomputable_ops(fx_g: fx.GraphModule) -> bool:
    found = False
    for node in fx_g.graph.nodes:
        # 检查是否有需要重新计算的操作节点
        if must_recompute(node):
            return True
    return False


def has_recomputable_rng_ops(fx_g: fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        # 检查是否有需要重新计算的随机数生成操作节点
        if (
            must_recompute(node)
            and hasattr(node.target, "tags")
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return True
    return False


def sym_node_size(node: fx.Node) -> int:
    # 计算符号节点的大小，根据节点的值类型返回不同大小的字节数
    if isinstance(node.meta["val"], (torch.SymInt, torch.SymBool)):
        return 1
    assert isinstance(node.meta["val"], torch.SymFloat)
    return 4


class InvalidNodeBase:
    def __repr__(self):
        return "Invalid Node"


InvalidNode = InvalidNodeBase()


def _extract_graph_with_inputs_outputs(
    joint_graph: fx.Graph, inputs: List[fx.Node], outputs: List[fx.Node]
) -> fx.Graph:
    """
    Given a graph, extracts out a subgraph that takes the specified nodes as
    inputs and returns the specified outputs.

    This includes specifying non-placeholder nodes as inputs.

    The general strategy is to initialize all inputs with proxies as we
    encounter them, and trace through the graph, only keeping values which take
    in valid proxies. Then, all dead code is eliminated.
    """
    new_graph = fx.Graph()
    env = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in inputs:
        # 在新图中按照输入顺序添加占位符节点
        new_node = new_graph.placeholder(node.name)
        # 由于可能将先前的 call_function 转换为占位符，因此不能在这里使用 node_copy
        new_node.meta = node.meta
        env[node] = new_node

    for node in joint_graph.nodes:
        if node in env:
            # 节点必须是我们的输入之一。（任何不是最初输入的 env 成员必定是由此循环创建的，不会在 joint_graph.nodes 中）
            continue
        elif node.op == "placeholder":
            env[node] = InvalidNode
        elif node.op == "call_function":
            # 将所有参数展开为参数树叶，并检查是否包含无效节点
            all_args = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            all_args = [
                isinstance(env[x], InvalidNodeBase)
                for x in all_args
                if isinstance(x, fx.Node)
            ]
            if any(all_args):
                env[node] = InvalidNode
                continue
            # 将节点复制到新图中，使用 lambda 函数映射环境中的节点
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "get_attr":
            # 将节点复制到新图中，使用 lambda 函数映射环境中的节点
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "output":
            pass
    output_values = []
    # 遍历输出列表中的每个元素
    for x in outputs:
        # 检查当前元素是否属于 fx.Node 类型
        if isinstance(x, fx.Node):
            # 如果当前节点不在环境变量中，抛出运行时错误
            if x not in env:
                raise RuntimeError(f"Node {x} couldn't be found in env")
            # 确保环境中的节点不是无效节点
            assert not isinstance(
                env[x], InvalidNodeBase
            ), f"Node {x} was invalid, but is output"
            # 将环境中节点的值添加到输出值列表中
            output_values.append(env[x])
        else:
            # 如果当前元素不是 fx.Node 类型，直接将其添加到输出值列表中
            output_values.append(x)
    
    # 将输出值列表设置为新图形对象的输出
    new_graph.output(output_values)
    
    # 清除新图形中的死代码
    new_graph.eliminate_dead_code()
    
    # 对新图形进行代码检查
    new_graph.lint()
    
    # 返回处理后的新图形对象
    return new_graph
def _is_primal(node: fx.Node) -> bool:
    # 判断节点是否为主节点（placeholder），且节点目标不包含"tangents"，且不是后向传播种子偏移量，也不是前向传播种子偏移量
    return (
        node.op == "placeholder"
        and "tangents" not in str(node.target)
        and not _is_bwd_seed_offset(node)
        and not _is_fwd_seed_offset(node)
    )


def _is_tangent(node: fx.Node) -> bool:
    # 判断节点是否为切线节点（placeholder），且节点目标包含"tangents"
    return node.op == "placeholder" and "tangents" in str(node.target)


def _is_bwd_seed_offset(node: fx.Node) -> bool:
    # 判断节点是否为后向传播种子偏移量节点（placeholder），且节点目标包含"bwd_seed"或"bwd_base_offset"
    return node.op == "placeholder" and (
        "bwd_seed" in str(node.target) or "bwd_base_offset" in str(node.target)
    )


def _is_fwd_seed_offset(node: fx.Node) -> bool:
    # 判断节点是否为前向传播种子偏移量节点（placeholder），且节点目标包含"fwd_seed"或"fwd_base_offset"
    return node.op == "placeholder" and (
        "fwd_seed" in str(node.target) or "fwd_base_offset" in str(node.target)
    )


def _is_backward_state(node: fx.Node) -> bool:
    # 判断节点是否为后向状态节点（placeholder），且节点的 meta 属性值是 BackwardState 类型
    return node.op == "placeholder" and isinstance(node.meta.get("val"), BackwardState)


def _extract_fwd_bwd_outputs(
    joint_module: fx.GraphModule, *, num_fwd_outputs
) -> Tuple[List[fx.Node], List[fx.Node]]:
    # 提取前向和后向的输出节点
    outputs = pytree.arg_tree_leaves(
        *(node.args for node in joint_module.graph.find_nodes(op="output"))
    )
    fwd_outputs = outputs[:num_fwd_outputs]  # 前向输出节点列表
    bwd_outputs = outputs[num_fwd_outputs:]  # 后向输出节点列表
    return fwd_outputs, bwd_outputs


def _remove_by_name(saved_values: List[fx.Node], name: str):
    # 根据节点名称从列表中移除保存的值
    for saved_value in saved_values:
        if saved_value.name == name:
            saved_values.remove(saved_value)
            break


def _extract_fwd_bwd_modules(
    joint_module: fx.GraphModule,
    saved_values: List[fx.Node],
    saved_sym_nodes: List[fx.Node],
    *,
    num_fwd_outputs: int,
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    # 提取前向和后向的模块
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(
        joint_module, num_fwd_outputs=num_fwd_outputs
    )
    placeholders = joint_module.graph.find_nodes(op="placeholder")
    primal_inputs = [*filter(_is_primal, placeholders)]  # 主节点输入列表
    tangent_inputs = [*filter(_is_tangent, placeholders)]  # 切线节点输入列表
    fwd_seed_offset_inputs = [*filter(_is_fwd_seed_offset, placeholders)]  # 前向种子偏移量节点输入列表
    bwd_seed_offset_inputs = [*filter(_is_bwd_seed_offset, placeholders)]  # 后向种子偏移量节点输入列表
    backward_state_inputs = [*filter(_is_backward_state, placeholders)]  # 后向状态节点输入列表

    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs,
        bwd_outputs,
    )

    for node in bwd_graph.find_nodes(op="placeholder"):
        # 这段代码用于过滤掉实际上没有被后向传播使用的保存值
        if not node.users:
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
        elif _is_backward_state(node):
            # 如果节点是后向状态节点，则直接保存
            _remove_by_name(saved_values, node.name)
            assert backward_state_inputs

    # 现在我们有了最终的保存值列表，需要确保传播所有被后向输入引用的符号。
    # 下面的变量保存了符号集合的信息，用于后续的大小变量分配
    saved_symbols: Set[sympy.Symbol] = set()
    # 保存直接绑定的符号节点
    saved_sym_nodes_binding = []
    # 保存派生的符号节点
    saved_sym_nodes_derived = []
    
    # 遍历保存的符号节点，识别并添加绑定的符号到集合中
    for node in saved_sym_nodes:
        # 检查节点是否是符号绑定节点
        symbol = is_symbol_binding_fx_node(node)
        if symbol:
            # 将符号添加到保存的符号集合中
            saved_symbols.add(symbol)
            # 将节点添加到保存直接绑定的符号节点列表中
            saved_sym_nodes_binding.append(node)
        else:
            # 将节点添加到保存派生的符号节点列表中
            saved_sym_nodes_derived.append(node)
    
    # 遍历所有的潜在反向输入，并跟踪需要绑定的其他符号
    symbol_bindings = find_symbol_binding_fx_nodes(joint_module.graph)
    for node in itertools.chain(saved_sym_nodes_derived, saved_values, tangent_inputs):
        # 检查节点的元数据中是否包含 'val'，如果没有则跳过
        if "val" not in node.meta:
            continue
        # 计算新的符号集合，去除已保存的符号集合中已有的符号
        new_symbols = free_symbols(node.meta["val"]) - saved_symbols
        # 对新符号按名称进行排序，保持确定性顺序
        for s in sorted(new_symbols, key=lambda s: s.name):
            # 对于良好形式的图，符号应始终存在，但我们也可能有生成不良形式图的方法，例如直接使用 make_fx，这种情况下不要阻塞
            if s not in symbol_bindings:
                continue
            # 将绑定的符号节点添加到保存直接绑定的符号节点列表中
            saved_sym_nodes_binding.append(symbol_bindings[s])
        # 将新符号添加到保存的符号集合中
        saved_symbols |= new_symbols
    
    # 更新保存的符号节点列表，现在已按绑定顺序重新排序，以确保所有绑定在前面
    saved_sym_nodes.clear()
    saved_sym_nodes.extend(saved_sym_nodes_binding + saved_sym_nodes_derived)
    
    # 重新生成前向和反向图形
    fwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        primal_inputs + fwd_seed_offset_inputs,
        fwd_outputs + saved_values + saved_sym_nodes,
    )
    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes
        + saved_values
        + tangent_inputs
        + bwd_seed_offset_inputs
        + backward_state_inputs,
        bwd_outputs,
    )
    
    # 创建前向和反向图形的懒加载图形模块
    fwd_module = fx._lazy_graph_module._make_graph_module(joint_module, fwd_graph)
    bwd_module = fx._lazy_graph_module._make_graph_module(joint_module, bwd_graph)
    
    # 返回前向和反向图形的模块
    return fwd_module, bwd_module
# 默认分区函数，将给定的联合模块按照与原始 `.forward()` 和 `.backward()` 方法执行相似的方式进行分割
def default_partition(
    joint_module: fx.GraphModule, _joint_inputs, *, num_fwd_outputs
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the :attr:`joint_module` in a manner that closely resembles the
    behavior observed in the original ``.forward()`` and ``.backward()`` of the
    callable, i.e., the resulting forward graph contains those operators that
    are executed in the original ``.forward()`` callable passed to
    :func:`aot_function`.

    The default partitioner collects the operators that are between the forward
    inputs and the forward outputs. This helps in finding the tensors which have
    to be stashed for the backward pass. These stashed tensors become the output
    of the generated forward graph. The remaining operators are then placed in
    the backward graph.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """
    # 检查联合模块是否具有可重计算操作，如果有，则使用最小割重计算分区
    if has_recomputable_ops(joint_module):
        return min_cut_rematerialization_partition(
            joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs
        )
    
    # 从联合图中提取原始输入节点（即不是派生输入的节点）
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    # 提取前向种子偏移输入节点（用于前向输出的种子偏移值）
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    # 将原始输入节点和前向种子偏移输入节点合并为总的输入节点列表
    inputs = primal_inputs + fwd_seed_offset_inputs
    
    # 提取前向输出节点和反向输出节点
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(
        joint_module, num_fwd_outputs=num_fwd_outputs
    )
    
    # 提取仅包含特定输入和前向输出节点的前向图
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph, inputs, fwd_outputs
    )
    
    # 获取前向图中不是输出节点的所有节点名称，这些节点将作为保存值
    forward_node_names = {
        node.name for node in forward_only_graph.nodes if node.op != "output"
    }
    
    # 初始化保存值列表和保存的符号节点列表
    saved_values = []
    saved_sym_nodes = []
    # 遍历联合模块的图中的每个节点
    for node in joint_module.graph.nodes:
        # 如果节点的名称不在前向节点名称列表中，则跳过此节点
        if node.name not in forward_node_names:
            continue
        
        # 如果节点是符号节点（symint），则需要将其与张量分开存储，
        # 这样 PythonFunction 只会在张量上调用 save_for_backward，而将符号节点存储在 autograd 的上下文中
        if is_sym_node(node):
            saved_sym_nodes.append(node)
        
        # 否则，如果节点不包含 'tensor_meta' 元数据且其操作为 'call_function'
        elif "tensor_meta" not in node.meta and node.op == "call_function":
            # 由于无法保存张量值的元组，需要展开要保存的内容
            users = node.users
            # 断言所有用户节点都是调用 operator.getitem
            assert all(user.target == operator.getitem for user in users)
            saved_values.extend(users)
        
        # 否则，处理其他类型的节点
        else:
            # 获取所有向后使用当前节点的节点列表
            backward_usages = [
                n for n in node.users if n.name not in forward_node_names
            ]
            
            # 如果节点的元数据中包含 'tensor_meta' 并且所有向后使用节点都是符号节点
            if "tensor_meta" in node.meta and all(
                is_sym_node(n) for n in backward_usages
            ):
                # 如果在前向过程中我们只需要其大小/步幅而不是实际张量数据，
                # 那么仅保存大小/步幅会更加高效。
                #
                # 注意，保存张量数据也可能导致编译问题：
                # 如果用户在前向过程中改变了输入并在后向过程中使用其大小/步幅，
                # 那么我们需要在保存之前克隆输入以符合 autograd 的要求。
                # （这是我们最初发现此 bug 的方式）。
                saved_sym_nodes.extend(backward_usages)
            
            # 否则，将当前节点添加到 saved_values 列表中
            else:
                saved_values.append(node)
    
    # 使用 dict.fromkeys 方法去除重复项，再转换回列表，并将结果分配给 saved_values 和 saved_sym_nodes
    saved_values = list(dict.fromkeys(saved_values).keys())
    saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes).keys())
    
    # 调用 _extract_fwd_bwd_modules 函数，返回其结果
    return _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_fwd_outputs,
    )
# 定义一个常量，表示整数的无穷大
INT_INF = int(1e6)


# 计算张量占用的字节数
def _tensor_nbytes(numel: int, dtype) -> int:
    return numel * dtype.itemsize


# 计算节点的大小
def _size_of(node: fx.Node) -> int:
    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, py_sym_types):
            return 1
        # 如果值是列表或元组，则计算所有张量占用的总字节数
        elif isinstance(val, (list, tuple)):
            return sum(
                _tensor_nbytes(hint_int(n.numel(), fallback=4096), n.dtype)
                for n in val
                if isinstance(n, torch.Tensor)
            )
        # 如果值是张量，则计算张量的字节数
        elif isinstance(val, torch.Tensor):
            return _tensor_nbytes(hint_int(val.numel(), fallback=4096), val.dtype)

        # 抛出异常，表示未知的元数据类型
        raise RuntimeError(f"Unknown metadata type {type(val)}")
    
    # 如果节点操作是 "get_attr"，则返回大小为 0
    if node.op == "get_attr":
        return 0
    
    # 抛出异常，表明节点应该总是有 `val` 元数据
    raise RuntimeError("We should always have `val` metadata on the nodes")


# 用于一些调查目的
def _count_ops(graph: fx.Graph):
    from collections import defaultdict

    cnt: Dict[str, int] = defaultdict(int)
    for node in graph.nodes:
        if node.op == "call_function":
            cnt[node.target.__name__] += 1
    # 打印操作计数，按计数值降序排列
    print(sorted(cnt.items(), key=lambda x: x[1], reverse=True))


# 使用 functools 模块的 lru_cache 装饰器，缓存点操作函数
@functools.lru_cache(None)
def pointwise_ops():
    ops = []
    for attr_name in dir(torch.ops.aten):
        opoverloadpacket = getattr(torch.ops.aten, attr_name)
        if not isinstance(opoverloadpacket, torch._ops.OpOverloadPacket):
            continue

        for overload in opoverloadpacket.overloads():
            op_overload = getattr(opoverloadpacket, overload)
            if torch.Tag.pointwise in op_overload.tags:
                # 将满足条件的操作添加到 ops 列表中
                ops.append(opoverloadpacket)
                break

    return ops


# 根据深度映射对参数进行排序
def sort_depths(args, depth_map: Dict[fx.Node, int]) -> List[Tuple[fx.Node, int]]:
    arg_depths = {
        arg: depth_map[arg] for arg in args if isinstance(arg, torch.fx.node.Node)
    }
    # 按节点深度降序排列并返回列表
    return sorted(arg_depths.items(), key=lambda x: x[1], reverse=True)


# 重新排序以模仿自动求导引擎的图形
def reordering_to_mimic_autograd_engine(gm: fx.GraphModule) -> fx.GraphModule:
    """
    此过程查找图中第一个 bwd 节点（通过查看切线的用户），然后重新排序图形，
    从此节点向图形末尾遍历。在此遍历中的每个操作中，我们将此操作插入新图中，
    并尝试仅从其他非 bwd 边缘中提取相关的子图。这在很大程度上模仿了自动求导引擎的行为。

    为什么首先需要此过程？

    这是分区器如何工作的产物。分区器的起点是联合图，即 fwd 和 bwd 图。
    在检查点技术中，我们保留了 fwd 图的部分，保持它们在原始位置上。
    """
    # 实现函数的具体功能
    pass  # 此处 pass 是为了表示功能尚未实现，占位用
    """
    the joint graph, while obtaining a bwd graph. As a result, the resulting bwd
    graph has copies of recomputed fwd subgraphs followed by the original bwd
    graph. If we run this naively, this leads to bad memory footprint, because
    the fwd subgraphs are live for way longer duration than necessary. This pass
    reorders the operations such that we prioritize the ops for the original bwd
    graph while only realizing those ops from the fwd graph that are necessary
    at any given point in the graph.
    """

    # 创建一个新的空白图形对象
    new_graph = fx.Graph()
    # 初始化一个空的环境字典，用于跟踪原始图中节点和新图中复制节点的映射关系
    env: Dict[fx.Node, fx.Node] = {}

    # 按照输入中指定的顺序，在新图中添加新的占位符节点
    for node in gm.graph.find_nodes(op="placeholder"):
        # 将原始图中的节点复制到新图中，并根据环境字典映射其输入节点
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    # 创建一个节点顺序的索引字典，记录每个节点在原始图中的顺序位置
    order = {}
    for idx, node in enumerate(gm.graph.nodes):
        order[node] = idx

    # 定义一个函数，将指定节点及其依赖节点插入到新图中
    def insert_node_in_graph(node):
        cur_nodes = [node]
        insertable_nodes = set()
        while len(cur_nodes) > 0:
            node = cur_nodes.pop()
            if node in insertable_nodes or node in env:
                continue
            insertable_nodes.add(node)

            # 偏向遍历具有更高深度的节点 - 优先处理关键路径
            cur_nodes += node.all_input_nodes

        # 根据节点在原始图中的顺序对可插入节点进行排序
        insertable_nodes = sorted(insertable_nodes, key=lambda n: order[n])
        for node in insertable_nodes:
            # 将节点复制到新图中，并映射其输入节点到新图中对应节点
            env[node] = new_graph.node_copy(node, lambda x: env[x])

    # 在图中查找第一个反向节点（bwd node）
    tangent_inputs = list(filter(_is_tangent, gm.graph.nodes))
    first_node_in_bwd = None
    minimum_order = math.inf
    for tangent in tangent_inputs:
        for user in tangent.users:
            if order[user] < minimum_order:
                minimum_order = order[user]
                first_node_in_bwd = user

    # 如果在“反向传播”中找不到任何节点，则返回原始图
    if first_node_in_bwd is None:
        return gm

    # 从第一个反向节点开始，逐个构建图中的操作节点
    for node in list(gm.graph.nodes)[order[first_node_in_bwd]:]:
        insert_node_in_graph(node)

    # 已经通过遍历构建了输出节点
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm
# 用户驱动激活检查点期间，确保前向（fwd）中的 rng 操作与后向（bwd）中重新计算的 rng 操作产生相同的输出。
def functionalize_rng_ops(
    joint_module: fx.GraphModule,
    fw_module: fx.GraphModule,
    bw_module: fx.GraphModule,
    num_sym_nodes: int,
) -> Tuple[fx.GraphModule, fx.GraphModule]:
    # 在前向（fwd）和后向（bwd）图之间共享 rng 状态，使用 functionalize 封装器来包装随机操作。

    # 有三个主要步骤来实现这一点
    # 第一步 - 构建 fwd 和 bwd 之间 rng 节点的映射。
    # 第二步 - 修改 fwd 传递，使其
    #   1) 用 run_and_save_rng_state 包装器替换 rand 操作
    #   2) 用此操作的 output[1] 替换原始操作的用户
    #   3) 收集所有 rng_state - 每个操作的 output[0]，并将它们作为输出节点。
    # 需要特别注意这里，因为 fwd 输出在最后有 symints。
    # 第三步 - 修改 bwd 传递，使其
    #   1) 在 tangents 前添加输入节点以存储存储的 rng 状态
    #   2) 用 run_with_save_rng_state 包装器替换 rand 操作
    #   3) 将存储的状态作为这些操作的输入使用。

    # 生成唯一的 id 来生成名称
    uid = itertools.count()

    def get_rng_ops(gmod):
        # 获取图中的所有随机节点
        random_nodes = {}
        for node in gmod.graph.nodes:
            if (
                node.op == "call_function"
                and hasattr(node.target, "tags")
                and torch.Tag.nondeterministic_seeded in node.target.tags
            ):
                random_nodes[node.name] = node
        return random_nodes

    def get_device(node):
        """
        检查节点输出的示例值以查找设备类型。
        """
        if "val" not in node.meta:
            return None

        candidates = node.meta["val"]
        if not isinstance(candidates, tuple):
            candidates = (candidates,)

        for candidate in candidates:
            if isinstance(candidate, torch.Tensor):
                if candidate.device.type == "cuda":
                    return "cuda"

        return "cpu"

    def get_sample_rng_state(device):
        # 根据设备类型获取随机数生成器状态
        if device == "cuda":
            return torch.cuda.get_rng_state()
        return torch.get_rng_state()

    # 第一步 - 构建 fwd 和 bwd 之间 rng 节点的映射。
    joint_graph_rng_ops = get_rng_ops(joint_module)
    fw_graph_rng_ops = get_rng_ops(fw_module)
    bw_graph_rng_ops = get_rng_ops(bw_module)
    recomputable_rng_ops_map = dict()
    # 遍历 joint_module 的图中的所有节点
    for node in joint_module.graph.nodes:
        # 检查是否需要重新计算该节点
        if (
            must_recompute(node)
            and hasattr(node.target, "tags")
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            # 获取与当前节点相关的随机数操作节点
            base_node = joint_graph_rng_ops[node.name]
            fw_node = fw_graph_rng_ops[node.name]
            bw_node = bw_graph_rng_ops[node.name]
            # 将可重新计算的随机数操作节点映射到其前向和后向节点
            recomputable_rng_ops_map[base_node] = {"fwd": fw_node, "bwd": bw_node}

    # 获取 torch._prims.rng_prims 中的函数引用
    run_and_save_rng = torch._prims.rng_prims.run_and_save_rng_state
    run_with_rng_state = torch._prims.rng_prims.run_with_rng_state

    # 初始化反向图中的切线起始节点为 None
    bw_tangent_start_node = None
    # 在反向图中查找操作为 "placeholder" 的节点，寻找名字包含 "tangent" 的节点
    for node in bw_module.graph.find_nodes(op="placeholder"):
        if "tangent" in node.name:
            bw_tangent_start_node = node
            break
    # 如果未找到切线起始节点，则抛出 RuntimeError 异常
    if bw_tangent_start_node is None:
        raise RuntimeError(
            "Couldn't find tangent node in graph inputs. This is unexpected, please file a bug if you see this"
        )

    # 初始化存储前向图随机数状态的输出列表
    fw_rng_state_outputs = []
    # 遍历可重新计算随机数操作节点的映射表
    for base_node, node_pair in recomputable_rng_ops_map.items():
        # Step 2 - 修改前向传播，使其如下
        fw_node = node_pair["fwd"]
        bw_node = node_pair["bwd"]
        fw_graph = fw_module.graph
        # 在 fw_node 前插入节点
        with fw_graph.inserting_before(fw_node):
            # 创建调用函数节点，用于运行并保存随机数状态
            functional_fw_node = fw_graph.create_node(
                "call_function",
                run_and_save_rng,
                args=(fw_node.target, *fw_node.args),
                kwargs=fw_node.kwargs,
            )
            # 提取保存的随机数状态
            state = fw_graph.create_node(
                "call_function",
                operator.getitem,
                args=(functional_fw_node, 0),
                kwargs={},
            )
            # 提取随机数输出
            rng_output = fw_graph.create_node(
                "call_function",
                operator.getitem,
                args=(
                    functional_fw_node,
                    1,
                ),
                kwargs={},
            )
            # 用随机数输出替换原始节点的所有使用
            fw_node.replace_all_uses_with(rng_output)
            # 删除原始节点
            fw_graph.erase_node(fw_node)
            # 将状态节点添加到输出列表
            fw_rng_state_outputs.append(state)

        # Step 3 - 修改反向传播，使其如下
        bw_graph = bw_module.graph
        # 在 bw_tangent_start_node 前插入节点
        with bw_graph.inserting_before(bw_tangent_start_node):
            # 创建名称为 state_name 的占位符节点，表示随机数状态输出
            state_name = f"rng_state_output_{next(uid)}"
            bw_rng_state_node = bw_graph.placeholder(state_name)
            # 设置占位符节点的 meta 信息，存储样本随机数状态
            bw_rng_state_node.meta["val"] = get_sample_rng_state(get_device(fw_node))

        # 在 bw_node 前插入节点
        with bw_graph.inserting_before(bw_node):
            # 创建调用函数节点，用于使用给定随机数状态运行反向传播
            rng_output = bw_graph.create_node(
                "call_function",
                run_with_rng_state,
                args=(bw_rng_state_node, bw_node.target, *bw_node.args),
                kwargs=bw_node.kwargs,
            )
            # 用随机数输出替换原始节点的所有使用
            bw_node.replace_all_uses_with(rng_output)
            # 删除原始节点
            bw_graph.erase_node(bw_node)

    # 在前向图的输出中添加随机数状态。AOT Autograd 假设符号整数位于前向图输出的末尾。
    # 因此，插入新的随机数状态节点到输出中
    # 从前向模块的计算图中找到第一个输出节点
    fw_output_node = next(iter(fw_module.graph.find_nodes(op="output")))

    # 获取前向模块的所有输出
    fw_outputs = fw_output_node.args[0]

    # 计算符号节点开始的索引，这些节点用于计算梯度
    sym_node_start_idx = len(fw_outputs) - num_sym_nodes

    # 构建新的输出列表，将前向模块的输出分为三部分：前部分、随机数生成器状态输出、符号节点部分
    outputs = (
        fw_outputs[:sym_node_start_idx]
        + fw_rng_state_outputs
        + fw_outputs[sym_node_start_idx:]
    )

    # 设置前向模块的计算图的输出为新构建的输出列表
    fw_module.graph.output(outputs)

    # 删除前向模块的原始输出节点
    fw_module.graph.erase_node(fw_output_node)

    # 重新编译前向模块，确保更新后的计算图生效
    fw_module.recompile()

    # 重新编译后向模块，以保持前向模块的更新与后向模块的同步
    bw_module.recompile()

    # 返回更新后的前向模块和后向模块
    return fw_module, bw_module
def cleanup_recompute_tags(joint_module: fx.GraphModule) -> fx.GraphModule:
    """
    如果有两个连续的检查点块之间没有操作符，则仍然希望在检查点块的边界存储张量。
    以下的处理步骤使得最后的输出节点不可重新计算，以支持这一点。
    """
    # 遍历计算图中的每个节点
    for node in joint_module.graph.nodes:
        # 如果节点需要重新计算
        if must_recompute(node):
            # 遍历该节点的每个使用者
            for user in node.users:
                # 如果使用者也需要重新计算，并且使用者的图标识大于节点的图标识
                if (
                    must_recompute(user)
                    and user.meta["ac_graph_id"] > node.meta["ac_graph_id"]
                ):
                    # 将节点标记为必须保存，以避免重新计算
                    node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    # 返回更新后的计算图模块
    return joint_module


def solve_min_cut(
    joint_graph: fx.Graph,
    node_info: NodeInfo,
    min_cut_options: MinCutOptions,
    dont_ban=None,
):
    # 如果 dont_ban 参数为 None，则初始化为空集合
    if dont_ban is None:
        dont_ban = set()
    # 获取默认操作列表
    op_types = get_default_op_list()

    # 如果启用了 AOT_PARTITIONER_DEBUG
    if AOT_PARTITIONER_DEBUG:
        # 收集所有节点的操作重载数据包，并转化为字符串集合
        joint_module_ops = {
            str(node.target._overloadpacket)
            for node in joint_graph.nodes
            if node.op == "call_function" and hasattr(node.target, "_overloadpacket")
        }
        # 计算不可重建的操作集合
        ops_ignored = joint_module_ops - {str(i) for i in op_types.recomputable_ops}
        # 打印不可重新材料化的操作信息
        print("Ops banned from rematerialization: ", ops_ignored)
        print()

    def is_fusible(a, b):
        """
        判断两个节点是否可融合。
        我们可以将 "memory fusion" 执行到一个 cat 操作上，但 cat 操作不能作为融合的生产者。
        """
        if get_aten_target(b) == aten.cat:
            return True
        return op_types.is_fusible(a) and op_types.is_fusible(b)

    try:
        import networkx as nx
    except ImportError as e:
        # 如果导入 networkx 失败，则抛出运行时错误
        raise RuntimeError(
            "Need networkx installed to perform smart recomputation " "heuristics"
        ) from e

    def is_materialized_backwards(node):
        """
        判断节点是否反向材料化。
        如果节点是视图操作，则返回 False。
        否则，逐步向前查找节点的使用者，直到找到非前向必要节点或不能融合的节点为止。
        """
        if op_types.is_view(node):
            return False
        cur_nodes = {node}
        while len(cur_nodes) > 0:
            cur = cur_nodes.pop()
            for user in cur.users:
                if not node_info.is_required_fw(user) and not is_fusible(cur, user):
                    return True
                if op_types.is_view(user):
                    cur_nodes.add(user)

        return False
    # 判断是否应该禁止重新计算节点
    def should_ban_recomputation(node):
        # 如果节点操作不是函数调用，则返回 False
        if node.op != "call_function":
            return False
        # 如果节点的目标是 operator.getitem，则返回 False
        if node.target == operator.getitem:
            return False
        # 如果节点的元数据中指定了强制保存，则返回 True
        if node.meta.get("recompute", None) == CheckpointPolicy.MUST_SAVE:
            return True
        # 如果启用了重新计算视图，并且节点是视图类型，则返回 False
        if config.recompute_views and op_types.is_view(node):
            return False
        # 如果节点的目标在不重新计算的白名单中，则返回 False
        if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
            return False

        # 如果设置了禁止不在允许列表中的操作类型重新计算选项
        if min_cut_options.ban_if_not_in_allowlist:
            # 如果节点不可重新计算，则返回 True
            if not op_types.is_recomputable(node):
                return True
        else:
            # 如果节点是随机操作或计算密集型操作，则返回 True
            if op_types.is_random(node) or op_types.is_compute_intensive(node):
                return True

        # 如果禁止材料化后向传播且节点在后向传播中被材料化，则返回 True
        if min_cut_options.ban_if_materialized_backward and is_materialized_backwards(
            node
        ):
            log.info("materialized backwards: %s %s", node, tuple(node.users))
            return True

        # 任意的启发式方法，用于优化性能，目前看起来不再必要
        # 注意：自 PR #121692 起，此方法似乎不再必要
        if node.dist_from_bw < 1000 and node.dist_from_bw > config.max_dist_from_bw:
            return True

        # 如果操作的输出大小是输入张量总和的四分之一以下，则禁止重新计算
        if min_cut_options.ban_if_reduction:
            input_tensors_size = sum(
                _size_of(i) for i in node.args if isinstance(i, fx.Node)
            )
            output_size = _size_of(node)
            return output_size * 4 < input_tensors_size
        # 默认返回 False
        return False

    # 判断节点是否已经材料化
    def is_materialized(node):
        # 如果节点操作是“placeholder”，则返回 True
        if node.op == "placeholder":
            return True

        # 如果节点有任何用户不可融合，则返回 True
        return not all(is_fusible(node, user) for user in node.users)
    # 定义一个函数，用于计算节点的权重，返回一个浮点数
    def get_node_weight(node) -> float:
        # 计算节点的内存大小
        mem_sz = _size_of(node)
        
        # 如果配置允许重新计算视图，并且当前节点是视图类型，则返回无穷大
        if config.recompute_views and op_types.is_view(node):
            # 如果 `config.recompute_views=True`，我们不保存视图。这通常是个好主意，
            # 因为视图是可以重新计算的，而且这样做稍微简化了分析过程。
            # 注意：如果它们不能自由重新计算（例如嵌套张量），我们应该修改对视图操作的检查为 `is_view` 并验证。
            # 基本上，对于嵌套张量，`aten.view` 不是一个“视图操作”。
            return math.inf

        # 如果节点的值是 Python 符号类型的实例
        if isinstance(node.meta["val"], py_sym_types):
            # 我们永远不想保存符号浮点数
            if not isinstance(node.meta["val"], torch.SymInt):
                return INT_INF

        # 启发式算法，偏向于靠近反向传播的节点
        # 关于当前值的完全猜测
        mem_sz = int(mem_sz * (1.1 ** max(min(node.dist_from_bw, 100), 1)))
        
        # 如果节点已经实现了，则返回内存大小
        if is_materialized(node):
            return mem_sz
        else:
            # 否则返回内存大小乘以2
            return mem_sz * 2

    # 创建一个有向图对象
    nx_graph = nx.DiGraph()
    # 创建一个集合用于存储禁止重新计算的节点
    banned_nodes = set()

    # 定义一个函数，如果允许的话禁止重新计算节点
    def ban_recomputation_if_allowed(node):
        # 如果节点是视图类型，则返回 False，不禁止重新计算
        if op_types.is_view(node):
            return False
        # 如果节点在 `dont_ban` 集合中，则返回 False，不禁止重新计算
        if node in dont_ban:
            return False
        # 如果用户注解要求必须重新计算该节点，则返回 False，不禁止重新计算
        if must_recompute(node):
            return False

        # 如果节点的值在元数据中存在，并且是 torch 的符号浮点数类型，则返回 False，不禁止重新计算
        if "val" in node.meta and isinstance(node.meta["val"], torch.SymFloat):
            return False

        # 将节点添加到禁止重新计算的节点集合中
        banned_nodes.add(node)
        
        # 添加一个从源节点到当前节点的入边到图中，并设置容量为无穷大
        nx_graph.add_edge("source", node.name + "_in", capacity=math.inf)
        
        # 返回 True，表示已经成功禁止重新计算该节点
        return True
    for node in joint_graph.nodes:
        # 遍历联合图中的节点

        if node.op == "output":
            # 如果节点的操作是 "output"，则跳过该节点
            continue

        if node in node_info.required_bw_nodes:
            # 如果节点在需要反向计算的节点列表中
            if node not in node_info.inputs:
                # 并且节点不在输入节点列表中
                # 将节点名称 + "_in" 连接到 "sink"，设置容量为无穷大
                nx_graph.add_edge(node.name + "_in", "sink", capacity=math.inf)
                # 继续下一个节点的处理
                continue

            # 否则，将节点名称 + "_out" 连接到 "sink"，设置容量为无穷大
            nx_graph.add_edge(node.name + "_out", "sink", capacity=math.inf)

        if must_recompute(node):
            # 如果需要重新计算节点
            # 将节点名称 + "_in" 连接到 "sink"，设置容量为无穷大
            nx_graph.add_edge(node.name + "_in", "sink", capacity=math.inf)
            # 继续下一个节点的处理
            continue

        if _is_primal(node) or _is_fwd_seed_offset(node):
            # 如果节点是主节点或者是前向梯度种子偏移节点
            # 根据条件禁止重新计算该节点
            ban_recomputation_if_allowed(node)

        # 如果一个节点不能重新计算（过于昂贵或涉及随机性），通过向源添加无穷大边来防止重新计算
        # 我们只需要在前向传递中禁止节点，因为这些是唯一可能会重新计算的节点。
        if node_info.is_required_fw(node) and should_ban_recomputation(node):
            # 如果节点在前向传递中是必需的，并且应该禁止重新计算该节点
            ban_recomputation_if_allowed(node)

        # 检查节点是否实际上是一个元组。如果我们总是使用伪张量，则可以简化为 isinstance 检查。
        is_non_tensor_node = (
            "val" not in node.meta and "tensor_meta" not in node.meta
        ) or ("val" in node.meta and not isinstance(node.meta["val"], torch.Tensor))

        if is_sym_node(node):
            # 如果节点是对称节点
            # 设置权重为对称节点的大小转换为浮点数
            weight = float(sym_node_size(node))
        elif is_non_tensor_node:
            # 如果节点不是张量节点
            # 如果节点的值是 BackwardState 类型，则权重为 0.0，否则为无穷大
            weight = (
                0.0 if isinstance(node.meta.get("val"), BackwardState) else math.inf
            )
        else:
            # 否则，获取节点的权重
            weight = get_node_weight(node)

        # 在节点之间创建边，从节点名称 + "_in" 到节点名称 + "_out"，设置容量为计算得到的权重
        nx_graph.add_edge(node.name + "_in", node.name + "_out", capacity=weight)
        # 遍历节点的用户
        for user in node.users:
            # 将节点名称 + "_out" 连接到用户节点名称 + "_in"，设置容量为无穷大
            nx_graph.add_edge(node.name + "_out", user.name + "_in", capacity=math.inf)

    # todo(chilli): This is the most questionable of the 3 heuristics for banning recompute.
    # Some example models to look at where this helps perf: poolformer_m36,
    # mixer_b16_224, cait_m36_384
    # 这是三种禁止重新计算的启发式方法中最值得怀疑的一个。
    # 一些示例模型可以看出这种方法提升了性能：poolformer_m36、mixer_b16_224、cait_m36_384
    # 如果设置了禁止远程使用，则对每个需要向前节点的节点进行遍历
    for used_node in node_info.required_fw_nodes:
        # 获取每个使用节点的顺序列表
        orders = [
            node_info.get_fw_order(user)
            for user in used_node.users
            if node_info.is_required_fw(user)
        ]
        # 获取所有需要向前节点的使用节点
        fw_users = [
            user for user in used_node.users if node_info.is_required_fw(user)
        ]
        # 如果存在顺序列表
        if len(orders) > 0:
            # 查找第一个不可融合使用的位置
            first_unfusible_use = find_first_unfusible(fw_users, max(orders))
            # 对于每个使用节点的副本
            for user in tuple(used_node.users):
                # 如果节点是必需的向前节点且节点顺序大于第一个不可融合使用位置并且可融合
                if (
                    node_info.is_required_fw(user)
                    and node_info.get_fw_order(user) > first_unfusible_use
                    and is_fusible(used_node, user)
                ):
                    # 如果节点在被禁用节点中，则跳过
                    if user in banned_nodes:
                        continue
                    # 记录日志
                    log.info(
                        "used above/below fusible %s:(%s) -> %s -> %s:(%s)",
                        used_node,
                        node_info.get_fw_order(used_node),
                        first_unfusible_use,
                        user,
                        node_info.get_fw_order(user),
                    )
                    # 如果允许的话，禁止重新计算节点
                    ban_recomputation_if_allowed(user)
    # 如果设定为禁止长的可融合链条，则执行以下逻辑
    visited = set()  # 用于记录已访问过的节点集合
    for start_node in joint_graph.nodes:  # 遍历联合图中的所有节点作为起始节点
        if not node_info.is_required_fw(start_node):  # 如果起始节点不是必需的前向节点，则跳过
            continue
        fusible = [(node_info.get_fw_order(start_node), start_node)]  # 创建一个优先队列，包含起始节点的前向顺序和节点本身
        start_order = node_info.get_fw_order(start_node)  # 获取起始节点的前向顺序
        while len(fusible) > 0:  # 当优先队列不为空时循环
            _, cur = heapq.heappop(fusible)  # 弹出优先队列中的节点
            if cur in visited:  # 如果当前节点已经访问过，则继续下一轮循环
                continue
            visited.add(cur)  # 将当前节点标记为已访问
            # 100 是一个任意选择的值，旨在防止出现退化情况
            if (
                node_info.get_fw_order(cur) > start_order + 100
                and len(fusible) == 0
            ):
                log.info(
                    "too long %s %s %s %s",
                    cur,
                    start_node,
                    node_info.get_fw_order(cur),
                    node_info.get_fw_order(start_node),
                )
                ban_recomputation_if_allowed(cur)  # 如果允许，则禁止对当前节点的重计算
                break

            for user in cur.users:  # 遍历当前节点的用户节点
                if (
                    node_info.is_required_fw(user)  # 如果用户节点是必需的前向节点
                    and is_fusible(cur, user)  # 并且当前节点与用户节点可融合
                    and user not in banned_nodes  # 并且用户节点不在禁止节点集合中
                ):
                    heapq.heappush(fusible, (node_info.get_fw_order(user), user))  # 将用户节点加入优先队列

    try:
        cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")  # 尝试计算最小割集
    except Exception:
        print("Failed to compute min-cut on following graph:")
        print("\n".join(nx.readwrite.edgelist.generate_edgelist(nx_graph)))
        visualize_min_cut_graph(nx_graph)
        raise

    reachable, non_reachable = partition  # 将最小割集分为可达和不可达两部分
    cutset: Set[Tuple[str, str]] = set()  # 初始化一个空的割集

    # 遍历可达部分中的节点及其邻居节点，将可达部分与不可达部分之间的边加入割集
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    cut_nodes = set()  # 初始化一个空集合，用于存储割点的节点

    # 遍历割集中的每一对节点，确保节点名称满足特定条件，然后将节点名称添加到割点集合中
    for node_in, node_out in cutset:
        assert node_in[:-3] == node_out[:-4]  # 确保节点名称的特定前缀
        node_name = node_in[:-3]  # 获取节点的名称
        cut_nodes.add(node_name)  # 将节点名称添加到割点集合中

    name_to_node = get_name_to_node(joint_graph)  # 获取节点名称到节点对象的映射
    node_idx = {node: idx for idx, node in enumerate(joint_graph.nodes)}  # 创建节点到索引的映射
    saved_values = sorted(
        (name_to_node[node] for node in cut_nodes), key=lambda x: node_idx[x]
    )  # 根据节点索引对割点集合中的节点对象进行排序，以确保结果的确定性
    # 返回两个变量 saved_values 和 banned_nodes
    return saved_values, banned_nodes
def visualize_min_cut_graph(nx_graph):
    # 导入必要的库
    import networkx as nx
    import pydot

    # 将 NetworkX 图转换为 Pydot 图的字符串表示形式
    dot_format = nx.nx_pydot.to_pydot(nx_graph).to_string()
    # 从 Pydot 格式的字符串创建图对象
    dot_graph = pydot.graph_from_dot_data(dot_format)[0]
    
    # 遍历图中的每条边
    for edge in dot_graph.get_edges():
        # 获取边的权重，这里假设边属性名为 "capacity"
        weight = nx_graph[edge.get_source()][edge.get_destination()]["capacity"]
        
        # 设置边的标签为权重值的字符串表示
        edge.set_label(str(weight))
        
        # 如果权重为无穷大，则将边颜色设置为红色
        if weight == float("inf"):
            edge.set_color("red")
    
    # 输出信息，表示正在将图保存为 SVG 文件
    print("Visualizing the failed graph to min_cut_failed.svg")
    # 将 Pydot 图保存为 SVG 文件
    dot_graph.write_svg("min_cut_failed.svg")


def get_default_op_list() -> OpTypes:
    # 导入必要的库和模块
    from typing import List, Callable
    import torch._ops.aten as aten
    import torch._ops.prim as prims
    import operator

    # 默认的可重计算操作列表
    default_recomputable_ops: List[Callable] = [
        aten.add, aten.sub, aten.div, aten.atan2, aten.mul,
        aten.max, aten.min, aten.pow, aten.remainder, aten.fmod,
        aten.__and__, aten.__or__, aten.__xor__, aten.__lshift__,
        aten.__rshift__, aten.eq, aten.ne, aten.ge, aten.gt, aten.le,
        aten.lt, aten.abs, aten.bitwise_not, aten.ceil, aten.floor,
        aten.frac, aten.neg, aten.relu, aten.round, aten.silu,
        aten.trunc, aten.log, aten.log10, aten.log1p, aten.log2,
        aten.lgamma, aten.exp, aten.expm1, aten.erf, aten.erfc,
        aten.cos, aten.acos, aten.cosh, aten.sin, aten.asin,
        aten.sinh, aten.tan, aten.atan, aten.tanh, aten.atanh,
        aten.sqrt, aten.rsqrt, aten.reciprocal, aten.sigmoid,
        aten.softplus, aten.threshold, aten.threshold_backward,
        aten.clamp, aten.where, aten.lerp, aten.addcmul, aten.gelu,
        aten.gelu_backward, aten.sum, aten.mean, aten._grad_sum_to_size,
        aten.sum_to_size, aten.amax, aten.to, aten.type_as,
        operator.getitem, aten.squeeze, aten.unsqueeze, aten.rsub,
        aten._to_copy,
    ]  # noqa: E501,B950
    
    # 可重计算的视图操作列表
    recomputable_view_ops = [
        aten.squeeze, aten.unsqueeze, aten.alias,
        aten.view, aten.slice, aten.t, prims.broadcast_in_dim,
        aten.expand, aten.as_strided, aten.permute,
    ]
    view_ops = recomputable_view_ops

    # 返回默认操作列表
    return default_recomputable_ops
    # 将一系列操作添加到默认可重新计算操作列表中
    default_recomputable_ops += [
        prims.div,  # 加法运算
        prims.convert_element_type,  # 转换元素类型
        aten.clone,  # 克隆张量
        aten._to_copy,  # 复制张量
        aten.full_like,  # 创建形状相同的全零张量
        prims.var,  # 计算方差
        prims.sum,  # 求和
        aten.var,  # 计算方差
        aten.std,  # 计算标准差
        prims.broadcast_in_dim,  # 在指定维度广播张量
        aten.select,  # 选取张量中的部分元素
        aten._unsafe_view,  # 不安全的视图操作
        aten.view,  # 改变张量的视图
        aten.expand,  # 扩展张量维度
        aten.slice,  # 切片张量
        aten.reshape,  # 改变张量形状
        aten.broadcast_tensors,  # 广播张量
        aten.scalar_tensor,  # 根据标量创建张量
        aten.ones,  # 创建全一张量
        aten.new_zeros,  # 创建全零张量
        aten.lift_fresh_copy,  # 创建张量的新副本
        aten.arange,  # 创建等差序列张量
        aten.triu,  # 提取张量的上三角部分
        aten.var_mean,  # 计算方差和均值
        aten.isinf,  # 判断张量中元素是否为无穷大
        aten.any,  # 判断张量是否至少有一个 True 元素
        aten.full,  # 创建填充指定值的张量
        aten.as_strided,  # 创建一个具有指定步幅的张量
        aten.zeros,  # 创建全零张量
        aten.argmax,  # 找出张量中最大值的索引
        aten.maximum,  # 比较两个张量并返回较大的值
        prims.iota,  # 生成整数序列
        prims._low_memory_max_pool2d_offsets_to_indices,  # 低内存消耗的最大池化操作
    ]  # noqa: E501,B950

    # Natalia 建议允许重新计算索引操作 :)
    # 将索引和聚集操作添加到默认可重新计算操作列表中
    default_recomputable_ops += [aten.index, aten.gather]

    # 将视图操作添加到默认可重新计算操作列表中
    default_recomputable_ops += view_ops

    # 将点对点操作添加到默认可重新计算操作列表中
    default_recomputable_ops += pointwise_ops()

    # 将 zeros_like 操作添加到默认可重新计算操作列表中
    default_recomputable_ops += [
        aten.zeros_like,
    ]

    # 将魔术方法转换为操作符并添加到默认可重新计算操作列表中
    default_recomputable_ops += [method_to_operator(m) for m in magic_methods]

    # 创建包含所有默认可重新计算操作的集合
    recomputable_ops = set(default_recomputable_ops)

    # 定义包含随机操作的列表
    random_ops = [aten.native_dropout, aten.rand_like, aten.randn_like]

    # 定义包含计算密集型操作的列表
    compute_intensive_ops = [
        aten.mm,  # 矩阵乘法
        aten.convolution,  # 卷积操作
        aten.convolution_backward,  # 卷积反向传播
        aten.bmm,  # 批量矩阵乘法
        aten.addmm,  # 矩阵相加再乘
        aten._scaled_dot_product_flash_attention,  # 缩放点积闪存注意力
        aten._scaled_dot_product_efficient_attention,  # 缩放点积高效注意力
        aten._flash_attention_forward,  # 闪存注意力前向传播
        aten._efficient_attention_forward,  # 高效注意力前向传播
        aten.upsample_bilinear2d,  # 双线性插值上采样
    ]  # noqa: E501,B950

    # 创建融合操作的集合，包括默认可重新计算操作和随机操作
    fusible_ops = recomputable_ops | set(random_ops)

    # 返回 OpTypes 对象，包含融合操作、计算密集型操作、随机操作、视图操作和默认可重新计算操作的集合
    return OpTypes(
        set(fusible_ops),
        set(compute_intensive_ops),
        set(random_ops),
        set(view_ops),
        set(recomputable_ops),
    )
# 创建一个空字典，用于存储图中节点名称到节点对象的映射
def get_name_to_node(graph: fx.Graph):
    name_to_node = {}
    # 遍历图中的所有节点
    for node in graph.nodes:
        # 将节点名称作为键，节点对象作为值存入字典
        name_to_node[node.name] = node
    return name_to_node


# 贪心算法解决背包问题，返回总运行时间、需要保存的项目列表和允许重新计算的项目列表
def greedy_knapsack(
    memory: List[float], runtimes: List[float], max_memory: float
) -> Tuple[float, List[int], List[int]]:
    n = len(runtimes)
    items = list(range(n))

    # 根据运行时间与内存比值的降序对项目进行排序
    items = sorted(items, key=lambda i: runtimes[i] / memory[i], reverse=True)

    total_memory = 0.0
    total_runtime = 0.0
    items_to_save = []
    items_to_allow_recomputing = []

    for i in items:
        if total_memory + memory[i] <= max_memory:
            total_memory += memory[i]
            total_runtime += runtimes[i]
            items_to_save.append(i)
        else:
            items_to_allow_recomputing.append(i)
    return total_runtime, items_to_save, items_to_allow_recomputing


# 使用整数线性规划解决背包问题，返回总运行时间、需要保存的项目列表和允许重新计算的项目列表
def ilp_knapsack(
    memory: List[float], runtimes: List[float], max_memory: float
) -> Tuple[float, List[int], List[int]]:
    import numpy as np

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        # 若导入失败，则抛出运行时错误
        raise RuntimeError(
            "To use the ILP for memory budget checkpointing you need to install scipy"
        ) from None

    np_memory = np.array(memory)
    np_runtimes = np.array(runtimes)
    c = -np_runtimes  # 目标函数为运行时间的负数

    # 内存约束条件
    memory_constraint = LinearConstraint(A=np_memory, ub=np.array(max_memory))
    constraints = [memory_constraint]

    integrality = np.ones_like(c)
    # 调用整数线性规划求解器
    res = milp(
        c=c, constraints=constraints, integrality=integrality, bounds=Bounds(0, 1)
    )
    if not res.success:
        # 如果求解失败，则抛出运行时错误
        raise RuntimeError("Somehow scipy solving failed")

    items_to_save = []
    items_to_allow_recomputing = []
    # 根据解向量判断哪些项目需要保存，哪些可以重新计算
    for idx, i in enumerate(res.x):
        if i == 1:
            items_to_save.append(idx)
        else:
            items_to_allow_recomputing.append(idx)
    return -res.fun, items_to_save, items_to_allow_recomputing


# 动态规划解决背包问题，返回总运行时间、需要保存的项目列表和允许重新计算的项目列表
def dp_knapsack(
    memory: List[float], runtimes: List[float], max_memory: float
) -> Tuple[float, List[int], List[int]]:
    # 缩放因子，将浮点数权重转换为整数
    S = 10000

    # 量化内存权重
    quantized_memory = torch.tensor(
        [int(round(m * S)) for m in memory], dtype=torch.long, device="cpu"
    )
    runtimes = torch.tensor(runtimes, dtype=torch.float32, device="cpu")

    # 量化的伪多项式时间动态规划解决0-1背包问题
    quantized_max_memory = int(round(max_memory * S))

    n = len(memory)

    # 初始化动态规划表
    dp = torch.zeros(
        (n + 1, quantized_max_memory + 1), dtype=torch.float32, device="cpu"
    )
    # 遍历范围为1到n的循环，其中n是输入参数
    for i in range(1, n + 1):
        # 获取当前的量化内存和运行时间，索引从0开始
        current_memory = quantized_memory[i - 1]
        current_runtime = runtimes[i - 1]

        # 复制前一行的数据到当前行
        dp[i, :] = dp[i - 1, :]

        # 更新所有j >= current_memory的dp[i, j]
        if current_memory == 0:
            # 如果当前内存为0，直接加上当前运行时间
            dp[i, :] = dp[i - 1, :] + current_runtime
        else:
            # 否则，使用torch.maximum更新dp[i, current_memory:]，考虑前一行的值和加上当前运行时间的值
            dp[i, current_memory:] = torch.maximum(
                dp[i - 1, current_memory:],
                dp[i - 1, :-current_memory] + current_runtime,
            )

    # 回溯以找出背包中包含的项目
    saved_items = []
    recomputable_items = []
    j: int = quantized_max_memory
    # 从最后一行向前遍历到第一行
    for i in range(n, 0, -1):
        # 如果当前行与前一行在j列的值不同，则表示选择了当前物品
        if dp[i][j] != dp[i - 1][j]:
            saved_items.append(i - 1)  # 将该物品的索引添加到保存的项目中（从0开始索引）
            j -= int(quantized_memory[i - 1].item())  # 减去该物品占用的内存
        else:
            recomputable_items.append(i - 1)  # 将可以重新计算的物品的索引添加到重新计算的项目中

    saved_items.reverse()  # 将保存的项目反转，以便按添加顺序获取物品

    # 在最大内存限制内可以实现的最大运行时间
    max_runtime = dp[n][quantized_max_memory].item()

    # 返回最大运行时间、保存的项目列表和可以重新计算的项目列表
    return max_runtime, saved_items, recomputable_items
# 根据给定的内存和运行时间列表，优化运行时配置，返回最佳解决方案
def _optimize_runtime_with_given_memory(
    memory: List[float],
    runtimes: List[float],
    max_memory: float,
) -> Tuple[float, List[int], List[int]]:
    # 从全局配置中获取内存预算求解器类型
    SOLVER = config.activation_memory_budget_solver
    # 根据求解器类型选择不同的背包问题求解方法
    if SOLVER == "greedy":
        return greedy_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "ilp":
        return ilp_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "dp":
        return dp_knapsack(memory, runtimes, max_memory)
    else:
        # 若求解器类型未知，则引发运行时错误
        raise RuntimeError(f"Not aware of memory budget knapsack solver: {SOLVER}")


# 从 torch.utils._mode_utils 中导入 no_dispatch 函数
from torch.utils._mode_utils import no_dispatch


# 估算节点的运行时间
def estimate_runtime(node):
    # 从全局配置中获取运行时间估算模式
    RUNTIME_MODE = config.activation_memory_budget_runtime_estimator

    # 定义材料化参数函数，将抽象节点转化为具体值
    def materialize_arg(x):
        if isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.Tensor):
            # 若节点值为张量，则返回一个与其形状相同的零张量
            shape = list(x.meta["val"].shape)

            def realize_symbol(d):
                return hint_int(d, fallback=4096)

            shape = [realize_symbol(s) for s in shape]
            return x.meta["val"].new_zeros(shape)
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymInt):
            # 若节点值为符号整数，则返回一个实际整数值
            return hint_int(x.meta["val"], fallback=4096)
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymFloat):
            # 若节点值为符号浮点数，则返回1.0
            return 1.0
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymBool):
            # 若节点值为符号布尔值，则返回True
            return True
        else:
            # 其他情况下直接返回节点值本身
            return x

    # 根据运行时估算模式进行不同的处理
    if RUNTIME_MODE == "testing":
        # 在测试模式下返回一个虚拟的运行时间1毫秒
        return 1

    elif RUNTIME_MODE == "profile":
        # 在性能分析模式下，执行节点的计算，并测量执行时间（毫秒）
        from triton.testing import do_bench

        with no_dispatch():
            args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
            ms = do_bench(lambda: node.target(*args, **kwargs))
            return ms

    elif RUNTIME_MODE == "flops":
        # 在 FLOPs 统计模式下，使用 Torch 的 FLOP 计数器统计节点的浮点运算数
        # 注意：这里返回的是最大的 FLOP 数或者1（如果 FLOP 数为0）
        from torch.utils.flop_counter import FlopCounterMode

        args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
        with FlopCounterMode(display=False) as mode:
            node.target(*args, **kwargs)
        counted_flops = mode.get_total_flops()
        return max(counted_flops, 1)
    else:
        # 若运行时估算模式未知，则引发运行时错误
        raise RuntimeError(f"Not aware of runtime estimator: {RUNTIME_MODE}")


# 选择保存的节点集合
def choose_saved_values_set(
    joint_graph: fx.Graph, node_info: NodeInfo, memory_budget=1
) -> List[fx.Node]:
    # 检查内存预算的有效性，必须在0到1之间（包括0和1）
    if memory_budget > 1 or memory_budget < 0:
        raise RuntimeError(
            f"The valid ranges for memory budget are 0 <= m <= 1. The provided value is {memory_budget}"
        )
    # 定义最小割选项，包括禁止远距离使用的重计算、禁止长融合链、禁止后向材料化重计算、不在允许列表中的禁止重计算和禁止减少计算
    min_cut_options = MinCutOptions(
        ban_if_used_far_apart=config.ban_recompute_used_far_apart,
        ban_if_long_fusible_chains=config.ban_recompute_long_fusible_chains,
        ban_if_materialized_backward=config.ban_recompute_materialized_backward,
        ban_if_not_in_allowlist=config.ban_recompute_not_in_allowlist,
        ban_if_reduction=config.ban_recompute_reductions,
    )
    # 如果配置为进行积极的重新计算
    if config.aggressive_recomputation:
        # 使用 `replace` 函数更新 `min_cut_options` 的参数，禁止远离使用的节点、禁止长可融合链、不材料化向后传播、不在允许列表中的节点均不禁止
        min_cut_options = replace(
            min_cut_options,
            ban_if_used_far_apart=False,
            ban_if_long_fusible_chains=False,
            ban_if_materialized_backward=False,
            ban_if_not_in_allowlist=False,
        )
    
    # 如果内存预算为零，直接返回节点信息的输入
    if memory_budget == 0:
        return node_info.inputs

    # 调用 `solve_min_cut` 函数计算最小割，得到运行时优化的保存数值和一些额外信息，如不需要额外信息可以使用 `_`
    runtime_optimized_saved_values, _ = solve_min_cut(
        joint_graph,
        node_info,
        min_cut_options,
    )

    # 如果内存预算为1，直接返回运行时优化的保存数值
    if memory_budget == 1:
        return runtime_optimized_saved_values

    # 定义一个函数 `estimate_activations_size`，计算保存数值列表中节点的激活大小的估计值
    def estimate_activations_size(saved_values: List[fx.Node]) -> float:
        return sum([_size_of(i) for i in saved_values]) / 1e9

    # 计算节点信息输入的激活大小估计值和运行时优化保存数值的激活大小估计值
    min_act_size = estimate_activations_size(node_info.inputs)
    max_act_size = estimate_activations_size(runtime_optimized_saved_values)

    # 如果运行时优化保存数值的激活大小估计值小于等于节点信息输入的激活大小估计值，则返回运行时优化保存数值
    if max_act_size <= min_act_size:
        return runtime_optimized_saved_values

    # 定义函数 `get_normalized_size`，计算给定大小的标准化值
    def get_normalized_size(sz):
        return (sz / 1e9) / (max_act_size - min_act_size)

    # 定义函数 `get_mem_ratio`，计算给定激活列表的内存占比
    def get_mem_ratio(activations: List[fx.Node]):
        return (estimate_activations_size(activations) - min_act_size) / (
            max_act_size - min_act_size
        )

    # 使用 `replace` 函数生成更积极的 `min_cut_options`，禁止远离使用的节点和长可融合链，但不再限制不在允许列表中的节点
    more_aggressive_options = replace(
        min_cut_options,
        ban_if_used_far_apart=False,
        ban_if_long_fusible_chains=False,
        ban_if_materialized_backward=False,
    )

    # 调用 `solve_min_cut` 函数计算更积极选项下的最小割，得到更积极优化的保存数值和一些额外信息
    more_aggressive_saved_values, _ = solve_min_cut(
        joint_graph, node_info, more_aggressive_options
    )

    # 如果更积极选项下的内存占比小于内存预算，返回更积极优化的保存数值
    if get_mem_ratio(more_aggressive_saved_values) < memory_budget:
        return more_aggressive_saved_values

    # 使用 `replace` 函数生成更进一步积极的 `min_cut_options`，不再限制不在允许列表中的节点
    aggressive_options = replace(
        more_aggressive_options,
        ban_if_not_in_allowlist=False,
    )

    # 调用 `solve_min_cut` 函数计算更进一步积极选项下的最小割，得到更进一步积极优化的保存数值和被禁止节点列表
    aggressive_recomputation_saved_values, banned_nodes = solve_min_cut(
        joint_graph, node_info, aggressive_options
    )

    # 如果更进一步积极选项下的内存占比小于内存预算，返回更进一步积极优化的保存数值
    if get_mem_ratio(aggressive_recomputation_saved_values) < memory_budget:
        return aggressive_recomputation_saved_values

    # 导入 `get_node_storage` 函数，用于获取节点的存储信息
    from torch._inductor.fx_utils import get_node_storage

    # 获取节点信息输入中每个节点的存储信息，并存储在 `input_storages` 字典中
    input_storages = {get_node_storage(node) for node in node_info.inputs}

    # 定义函数 `get_recomputable_banned_nodes`，返回可以重新计算的被禁止节点列表
    def get_recomputable_banned_nodes(banned_nodes: List[fx.Node]) -> List[fx.Node]:
        return [
            i
            for i in banned_nodes
            if (
                # 仅允许重新计算实际需要进行反向传播的节点，并且它们的距离小于1e9
                i.dist_from_bw < int(1e9)  # type: ignore[attr-defined]
                and get_node_storage(i) not in input_storages
            )
        ]

    # 获取可以重新计算的被禁止节点列表
    recomputable_banned_nodes = get_recomputable_banned_nodes(banned_nodes)

    # 返回三种不同优化策略下的结果，按节点大小降序排序
    all_recomputable_banned_nodes = sorted(
        recomputable_banned_nodes, key=_size_of, reverse=True
    )
    # 如果所有可重新计算的被禁止节点的数量为零，则直接返回节点信息的输入
    if len(all_recomputable_banned_nodes) == 0:
        return node_info.inputs
    
    # 计算所有可重新计算的被禁止节点的内存大小，并进行归一化处理
    memories_banned_nodes = [
        get_normalized_size(_size_of(i)) for i in all_recomputable_banned_nodes
    ]
    
    # 估计所有可重新计算的被禁止节点的运行时长
    runtimes_banned_nodes = [
        estimate_runtime(node) for node in all_recomputable_banned_nodes
    ]
    
    # 导入 no_dispatch 函数，该函数用于确保在特定上下文中没有分发操作
    from torch.utils._mode_utils import no_dispatch

    # 定义一个函数，根据内存预算获取保存的值和预期的运行时间
    def get_saved_values_knapsack(memory_budget):
        with no_dispatch():
            # 调用优化函数，根据给定的内存和已禁止节点的内存和运行时数据进行优化
            (
                expected_runtime,
                saved_node_idxs,
                recomputable_node_idxs,
            ) = _optimize_runtime_with_given_memory(
                memories_banned_nodes, runtimes_banned_nodes, max(memory_budget, 0)
            )
        # 将可重新计算的节点添加到“不禁止”的集合中
        dont_ban = set()
        for idx in recomputable_node_idxs:
            dont_ban.add(all_recomputable_banned_nodes[idx])
        # 断言“不禁止”的节点集合是“所有可重新计算的被禁止节点”的子集
        assert dont_ban.issubset(all_recomputable_banned_nodes)

        # 解决最小割问题，获取保存的值和预期的运行时间
        saved_values, _ = solve_min_cut(
            joint_graph,
            node_info,
            aggressive_options,
            dont_ban,
        )
        return saved_values, expected_runtime

    # 如果配置为可视化内存预算帕累托曲线
    if config.visualize_memory_budget_pareto:
        options = []
        # 对内存预算进行循环，每次递减5个单位，从100递减到0
        for sweep_memory_budget in range(100, -1, -5):
            # 调用获取保存值和预期运行时间的函数，根据当前的内存预算
            saved_values, expected_runtime = get_saved_values_knapsack(
                sweep_memory_budget / 100
            )
            # 将当前选项的结果添加到选项列表中，包括内存预算、重新计算运行时与预期运行时间的差异、保存值的内存比率
            options.append(
                (
                    sweep_memory_budget,
                    sum(runtimes_banned_nodes) - expected_runtime,
                    get_mem_ratio(saved_values),
                )
            )

        # 导入 matplotlib 库，用于绘图
        import matplotlib.pyplot as plt

        # 提取 x 和 y 值用于绘图
        x_values = [item[2] for item in options]
        y_values = [item[1] for item in options]

        # 创建一个图形窗口，并设定大小
        plt.figure(figsize=(10, 6))
        # 绘制帕累托曲线，使用圆圈标记每个点
        plt.plot(x_values, y_values, marker="o")

        # 为每个点添加标签
        for i, txt in enumerate(x_values):
            plt.annotate(
                f"{txt:.2f}",
                (x_values[i], y_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # 设定 x 和 y 轴的标签
        plt.xlabel("Memory Budget")
        plt.ylabel("Runtime of Recomputed Components")
        # 设定图表标题
        plt.title("Pareto Frontier of Memory Budget vs. Recomputation Runtime")
        # 启用网格线
        plt.grid(True)
        # 获取当前图形对象
        fig = plt.gcf()
        # 显示图形
        plt.show()
        # 根据图表名称保存图形
        fig_name = f"memory_budget_pareto_{get_aot_graph_name()}.png"
        fig.savefig(fig_name)
        # 记录警告日志，指示生成的帕累托曲线保存位置
        log.warning("Generated Pareto frontier curve at %s", fig_name)

    # 返回获取保存值和预期运行时间的函数，使用默认的内存预算作为参数
    return get_saved_values_knapsack(memory_budget=memory_budget)[0]
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recomputing helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimination.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.
        _joint_inputs: The inputs to the joint graph. This is unused.
        compiler: This option determines the default set of recomputable ops.
            Currently, there are two options: ``nvfuser`` and ``inductor``.
        recomputable_ops: This is an optional set of recomputable ops. If this
            is not None, then this set of ops will be used instead of the
            default set of ops.
        num_fwd_outputs: The number of outputs from the forward graph.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """

    # 进行死代码消除，优化图形模块
    joint_module.graph.eliminate_dead_code()
    # 重新编译模块
    joint_module.recompile()

    # 获取联合模块的图形对象
    fx_g = joint_module.graph

    # 添加公共子表达式消除（CSE）通道
    if config.cse:
        # 使用公共子表达式消除算法处理图形对象
        cse_graph = fx_graph_cse(fx_g)
        # 更新联合模块的图形对象为经过CSE处理的图形对象
        joint_module.graph = cse_graph

    # 获取更新后的联合图形模块
    joint_graph = joint_module.graph

    # 检查联合模块是否有可重算的操作
    graph_has_recomputable_ops = has_recomputable_ops(joint_module)
    # 检查联合模块是否有可重算的随机数生成操作
    graph_has_recomputable_rng_ops = has_recomputable_rng_ops(joint_module)

    # 如果图形模块有可重算的操作
    if graph_has_recomputable_ops:
        # 清理模块中的重算标签
        joint_module = cleanup_recompute_tags(joint_module)
    # 根据联合模块的图获取节点名称到节点对象的映射
    name_to_node = get_name_to_node(joint_module.graph)
    
    # 存储需要反向传播的节点集合
    required_bw_nodes = set()
    
    # 遍历联合模块的图中的每个节点
    for node in joint_module.graph.nodes:
        # 如果节点的操作为"placeholder"并且目标包含"tangents"
        if node.op == "placeholder" and "tangents" in node.target:
            required_bw_nodes.add(node)
        
        # 如果节点在需要反向传播的节点集合中
        if node in required_bw_nodes:
            # 将该节点的所有用户节点也加入到需要反向传播的节点集合中
            for user in node.users:
                required_bw_nodes.add(user)

    # 筛选出主输入节点列表，这些节点通过 _is_primal 函数进行过滤
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    
    # 筛选出前向种子偏移输入节点列表，这些节点通过 _is_fwd_seed_offset 函数进行过滤
    fwd_seed_offset_inputs = list(
        filter(_is_fwd_seed_offset, joint_module.graph.nodes)
    )
    
    # 将主输入节点和前向种子偏移输入节点合并成一个输入节点列表
    inputs = primal_inputs + fwd_seed_offset_inputs
    
    # 提取前向和后向输出节点
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(
        joint_module, num_fwd_outputs=num_fwd_outputs
    )
    
    # 将后向输出节点中不为 None 且操作不为"output"的节点加入到需要反向传播的节点集合中
    required_bw_nodes.update(
        o for o in bwd_outputs if o is not None and o.op != "output"
    )
    
    # 提取只包含主输入节点和前向输出节点的图
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph, inputs, fwd_outputs
    )
    
    # 获取在 forward_only_graph 中不为"output"操作的节点，构成需要前向传播的节点集合
    required_fw_nodes: Set[fx.Node] = {
        name_to_node[node.name]
        for node in forward_only_graph.nodes
        if node.op != "output"
    }
    
    # 筛选出在联合模块图中既不在需要前向传播节点集合，也不在需要反向传播节点集合的节点，构成未声明节点集合
    unclaimed_nodes = {
        node
        for node in joint_module.graph.nodes
        if node not in required_fw_nodes and node not in required_bw_nodes
    }
    
    # 初始化前向计数和前向顺序字典
    fw_cnt = 0
    fw_order = {}
    
    # 遍历联合模块图中的每个节点
    for node in joint_module.graph.nodes:
        # 如果节点在需要前向传播的节点集合中
        if node in required_fw_nodes:
            # 将节点及其顺序加入前向顺序字典中
            fw_order[node] = fw_cnt
            fw_cnt += 1
    
    # 返回节点信息对象，包括输入节点列表、需要前向传播的节点集合、需要反向传播的节点集合、未声明节点集合和前向顺序字典
    return NodeInfo(
        inputs, required_fw_nodes, required_bw_nodes, unclaimed_nodes, fw_order
    )

# 对联合模块进行节点分类，获取节点信息
node_info = classify_nodes(joint_module)

# 如果需要反向传播的节点集合为空
if len(node_info.required_bw_nodes) == 0:
    # 调用默认分区器对联合模块进行分区
    return default_partition(
        joint_module, _joint_inputs, num_fwd_outputs=num_fwd_outputs
    )

# 对联合模块的节点进行反向遍历
for node in reversed(joint_module.graph.nodes):
    # 如果节点的操作为"output"
    if node.op == "output":
        # 设置节点距离反向传播的距离为一个极大值
        node.dist_from_bw = int(1e9)
    # 如果节点不在需要前向传播的节点集合中
    elif not node_info.is_required_fw(node):
        # 设置节点距离反向传播的距离为0
        node.dist_from_bw = 0
    else:
        # 设置节点距离反向传播的距离为一个极大值
        node.dist_from_bw = int(1e9)
        # 遍历节点的用户节点
        for user in node.users:
            # 更新节点距离反向传播的距离为其用户节点距离加1的最小值
            node.dist_from_bw = min(node.dist_from_bw, user.dist_from_bw + 1)

# 获取配置的激活内存预算
memory_budget = config.activation_memory_budget

# 遍历联合图中的每个节点
for node in joint_graph.nodes:
    # 如果节点的元数据中存在浮点型的"memory_budget"信息
    if isinstance(node.meta.get("memory_budget", None), float):
        # 更新内存预算为节点的"memory_budget"值
        memory_budget = node.meta["memory_budget"]
        # 中断循环
        break

# 使用选择的保存值集合函数，选择节点需要保存的值
saved_values = choose_saved_values_set(
    joint_graph, node_info, memory_budget=memory_budget
)

# 在张量上调用 save_for_backward，并在自动求导上下文中存储符号整数
    # 从 saved_values 列表中过滤出所有符号节点，保存在 saved_sym_nodes 列表中
    saved_sym_nodes = list(filter(is_sym_node, saved_values))
    # 从 saved_values 列表中过滤出所有非符号节点，重新赋值给 saved_values 列表
    saved_values = list(filter(lambda n: not is_sym_node(n), saved_values))

    # 提取前向和后向模块，用于自动微分
    fw_module, bw_module = _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_fwd_outputs,
    )

    # 如果图中存在可重新计算的操作
    if graph_has_recomputable_ops:
        # 如果图中存在可重新计算的随机数生成操作
        if graph_has_recomputable_rng_ops:
            # 对随机数生成操作进行功能化处理，以便前向和后向模块处理
            fw_module, bw_module = functionalize_rng_ops(
                joint_module, fw_module, bw_module, len(saved_sym_nodes)
            )
    
    # 重新排序 bw_module 以模仿自动微分引擎的行为
    bw_module = reordering_to_mimic_autograd_engine(bw_module)

    # 如果启用了 AOT_PARTITIONER_DEBUG
    if AOT_PARTITIONER_DEBUG:
        from torch._inductor.fx_utils import get_node_storage

        # 获取 saved_values 中每个节点的存储空间信息
        storages = {get_node_storage(node) for node in saved_values}
        # 打印理论激活存储量，单位为 GB
        print(
            "Theoretical Activations Stored: ",
            sum(_size_of(i) for i in saved_values) / 1e9,
        )
        # 对 saved_values 中的节点按大小排序，并打印
        sorted_sizes = sorted([(_size_of(i), str(i)) for i in saved_values])
        # 提取 fw_module 中调用函数的节点名称集合
        fw_module_nodes = {
            node.name for node in fw_module.graph.nodes if node.op == "call_function"
        }
        # 提取 bw_module 中调用函数的节点名称集合
        bw_module_nodes = {
            node.name for node in bw_module.graph.nodes if node.op == "call_function"
        }
        # 计算前向和后向模块中重复使用的节点名称集合
        remat_nodes = fw_module_nodes & bw_module_nodes

        # 统计 fw_module 中节点重复计算的操作次数
        counts: Dict[str, int] = defaultdict(int)
        for node in fw_module.graph.nodes:
            if node.name in remat_nodes and hasattr(node.target, "_overloadpacket"):
                counts[str(node.target._overloadpacket)] += 1
        # 打印重复计算的操作次数信息
        print(
            f"# remat/fw/bw: {len(remat_nodes)}/{len(fw_module_nodes)}/{len(bw_module_nodes)}"
        )
        # 打印按重复次数排序的操作及其次数
        print(
            "Count of Ops Rematerialized: ",
            sorted(counts.items(), key=lambda x: x[1], reverse=True),
        )
    
    # 返回前向和后向模块
    return fw_module, bw_module
def draw_graph(
    traced: torch.fx.GraphModule,  # 输入参数traced是一个torch.fx.GraphModule对象，表示待绘制的图
    fname: str,  # 输入参数fname是文件名，用于保存生成的图形
    figname: str = "fx_graph",  # 可选参数figname指定图形的名称，默认为"fx_graph"
    clear_meta: bool = True,  # 可选参数clear_meta表示是否清除图的元数据，默认为True
    prog: Optional[Union[str, List[str]]] = None,  # 可选参数prog指定图形的输出格式，可以是字符串或字符串列表，默认为None
    parse_stack_trace: bool = False,  # 可选参数parse_stack_trace表示是否解析堆栈跟踪信息，默认为False
    dot_graph_shape: Optional[str] = None,  # 可选参数dot_graph_shape指定生成的图形的形状，可以是字符串，默认为None
) -> None:  # 函数返回类型为None，即没有返回值
    if clear_meta:
        # 如果clear_meta为True，深拷贝traced.graph创建一个新的图，并将其赋值给traced
        new_graph = copy.deepcopy(traced.graph)
        traced = fx.GraphModule(traced, new_graph)
        # 清空新图中每个节点的元数据
        for node in traced.graph.nodes:
            node.meta = {}
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = "." + config.torch_compile_graph_format  # 如果没有提供文件扩展名，则使用默认的torch编译图形格式
    print(f"Writing FX graph to file: {base}{ext}")  # 打印输出正在写入的FX图文件名
    # 创建一个FxGraphDrawer对象g，用于绘制图形
    g = graph_drawer.FxGraphDrawer(
        traced,
        figname,
        parse_stack_trace=parse_stack_trace,
        dot_graph_shape=dot_graph_shape,
    )
    # 获取主要的DOT图形对象x
    x = g.get_main_dot_graph()
    # 获取用于写入图形的方法，方法名称由文件扩展名决定
    write_method = getattr(x, "write_" + ext.lstrip("."))
    fname = f"{base}{ext}"  # 组合生成最终的文件名
    if prog is None:
        write_method(fname)  # 调用write_method将图形写入文件
    else:
        write_method(fname, prog=prog)  # 如果指定了prog参数，则将其一并传递给write_method
```