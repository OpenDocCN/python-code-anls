# `.\pytorch\torch\fx\passes\tools_common.py`

```
# mypy: allow-untyped-defs
# 引入类型定义，允许未标注类型的函数
from typing import List, Tuple, Union, Dict, Any, Set, Mapping, Optional
# 引入collections模块
import collections
# 引入dataclass装饰器
from dataclasses import dataclass
# 引入operator模块
import operator

# 引入PyTorch库
import torch
# 引入torch.fx模块
import torch.fx
# 引入torch.fx.node模块中的_get_qualified_name函数
from torch.fx.node import _get_qualified_name
# 引入torch.fx._compatibility模块中的compatibility装饰器
from torch.fx._compatibility import compatibility

# 定义公开的模块成员列表
__all__ = ['get_acc_ops_name', 'get_node_target', 'is_node_output_tensor', 'FxNetAccFusionsFinder', 'legalize_graph']

# 定义Tensors类型为Union类型，包含Tuple[torch.Tensor]和List[torch.Tensor]
Tensors = Union[Tuple[torch.Tensor], List[torch.Tensor]]
# 定义TensorOrTensors类型为Union类型，包含torch.Tensor和Tensors
TensorOrTensors = Union[torch.Tensor, Tensors]
# 定义NodeList类型为List[torch.fx.Node]
NodeList = List[torch.fx.Node]
# 定义NodeSet类型为Set[torch.fx.Node]
NodeSet = Set[torch.fx.Node]
# 定义Names类型为List[str]
Names = List[str]
# 定义CALLABLE_NODE_OPS为包含字符串的集合
CALLABLE_NODE_OPS = {"call_module", "call_function", "call_method"}


@compatibility(is_backward_compatible=False)
def get_acc_ops_name(k):
    """
    根据输入的对象k返回其对应的操作名称。

    如果k是字符串类型，直接返回k。
    如果k的模块包含"acc_ops"，返回"acc_ops.{k.__name__}"。
    否则，将k的模块名称中的'torch._ops'替换为'torch.ops'，并返回"{module}.{k.__name__}"。
    """
    if isinstance(k, str):
        return k
    elif k.__module__ and "acc_ops" in k.__module__:
        return f"acc_ops.{k.__name__}"
    else:
        module = k.__module__.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
        return f"{module if module else ''}.{k.__name__}"


@compatibility(is_backward_compatible=False)
def get_node_target(submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> str:
    """
    给定一个节点`node`，返回其目标类型名。

    对于"call_method"节点，返回节点正在调用方法的名称。
    对于"call_function"节点，返回节点目标的类型名称。
    对于"call_module"节点，返回节点目标指向的模块的类型名称。

    如果目标名称字符串中包含"_VariableFunctionsClass"，则将其替换为"torch"。
    """
    assert node.op in CALLABLE_NODE_OPS, (
        "Expect op types of " + ", ".join(CALLABLE_NODE_OPS) + f", but found {node.op}"
    )

    if node.op == "call_module":
        assert isinstance(node.target, str)
        submod = submodules[node.target]
        submod_type = getattr(submod, "_base_class_origin", type(submod))
        return get_acc_ops_name(submod_type)
    elif node.op == "call_function":
        target: Any = node.target
        return (
            f"acc_ops.{target.__name__}"
            if target.__module__ is not None and "acc_ops" in target.__module__
            else _get_qualified_name(target)
        )
    else:
        assert isinstance(node.target, str)
        return node.target


@compatibility(is_backward_compatible=False)
def is_node_output_tensor(node: torch.fx.Node) -> bool:
    """
    检查节点输出是否产生了Tensor。

    注意：在调用此函数之前，需要在包含fx图上运行`ShapeProp`。
    这是因为它通过检查节点上的`type`元数据来工作。
    此元数据由`ShapeProp`生成。
    """
    type_ = node.meta.get("type", None)
    return type_ is not None and issubclass(type_, torch.Tensor)
# 使用修饰器定义类的兼容性，指定它不向后兼容
@compatibility(is_backward_compatible=False)
# 定义一个类 FxNetAccFusionsFinder，用于查找在 ACC 节点之间传递非张量数据的连接组，称为融合组
class FxNetAccFusionsFinder:
    """
    查找在 ACC 节点之间传递非张量数据的连接组，这些组称为融合组。
    """

    # 初始化方法，接受一个 torch.fx.GraphModule 对象和一个 ACC 节点集合作为参数
    def __init__(self, module: torch.fx.GraphModule, acc_nodes: NodeSet):
        # 将传入的模块对象保存到实例变量 module 中
        self.module = module
        # 获取模块图中所有节点的列表并保存到实例变量 nodes 中
        self.nodes = list(module.graph.nodes)
        # 将传入的 ACC 节点集合保存到实例变量 acc_nodes 中
        self.acc_nodes = acc_nodes

    # 定义一个内部类 FusionGroup，用于表示融合组
    @dataclass
    class FusionGroup:
        # 融合组中经过模型所有节点进行拓扑排序后的最小节点索引
        top_node_idx: int

        # 融合组中的节点集合
        nodes: NodeSet

        # 融合组的输入节点集合
        inputs: NodeSet

        # 尚未处理的融合组中的节点集合
        nodes_need_process: NodeSet

        # 向融合组中添加节点的方法
        def add_node(self, node):
            """
            向融合组中添加一个节点。
            """
            # 如果节点已经在节点集合中，直接返回
            if node in self.nodes:
                return

            # 将节点添加到待处理节点集合中
            self.nodes_need_process.add(node)
            # 将节点添加到节点集合中
            self.nodes.add(node)
            # 从输入节点集合中移除该节点
            self.inputs.discard(node)
            # 更新输入节点集合，包括节点的所有输入节点，但要排除已经在融合组中的节点
            self.inputs.update(
                {
                    n
                    for n in node.all_input_nodes
                    if n.op in CALLABLE_NODE_OPS and n not in self.nodes
                }
            )

    # 递归添加节点到融合组的方法
    def recursive_add_node(
        self,
        fusion_group: "FxNetAccFusionsFinder.FusionGroup",
        inputs: Union[NodeSet, NodeList],
        visited: Optional[NodeSet] = None,
    ):
        """
        从输入节点开始，按照反向拓扑顺序进行递归。如果任何上游节点
        在融合组中，则将该路径上的所有节点添加到融合组中。
        """
        # 遍历输入节点集合
        for arg in inputs:
            # 如果已经访问过该节点，则跳过
            if visited is not None:
                if arg in visited:
                    continue
                visited.add(arg)

            # 跳过占位符和 get_attr 节点，因为它们不会在融合组中
            if arg.op not in CALLABLE_NODE_OPS:
                continue

            # 如果节点的索引小于融合组中的最小节点索引，说明它已经是融合组的上游节点，不需要再检查它
            if self.nodes.index(arg) < fusion_group.top_node_idx:
                continue

            # 如果节点已经在融合组中，返回 True
            if arg in fusion_group.nodes:
                return True

            # 检查节点的上游节点，如果任何上游节点在融合组中，则将该节点添加到融合组并返回 True
            if self.recursive_add_node(fusion_group, arg.all_input_nodes, visited):
                fusion_group.add_node(arg)
                return True

        return False
    # 定义 __call__ 方法，用于执行对象的调用操作，返回一个字典，将节点映射到节点集合
    def __call__(self) -> Dict[torch.fx.Node, NodeSet]:
        # 初始化结果字典
        result: Dict[torch.fx.Node, NodeSet] = {}
        # 将累积节点列表转换为列表
        acc_nodes = list(self.acc_nodes)

        # 遍历累积节点列表中的每个节点
        for node in acc_nodes:
            # 如果节点已经在结果字典中，则跳过处理
            if node in result:
                continue
            # 如果节点操作不在可调用节点操作集合中，则跳过处理
            if node.op not in CALLABLE_NODE_OPS:
                continue
            # 如果节点的元数据中包含 "tensor_meta" 键，则跳过处理
            if "tensor_meta" in node.meta:
                continue
            # 如果节点不在累积节点集合中，则跳过处理
            if node not in self.acc_nodes:
                continue

            # 创建融合组对象，用于将节点聚合到一个组中
            fusion_group: FxNetAccFusionsFinder.FusionGroup = self.FusionGroup(
                top_node_idx=self.nodes.index(node),
                nodes={node},
                inputs=set(node.all_input_nodes),
                nodes_need_process={node},
            )
            # 当融合组中有待处理节点时执行循环
            while fusion_group.nodes_need_process:
                # 弹出融合组中待处理的节点
                node = fusion_group.nodes_need_process.pop()
                # 递归地添加节点及其输入节点到融合组中
                self.recursive_add_node(
                    fusion_group,
                    fusion_group.inputs,
                    visited=set(),
                )

                # 可选地添加节点的下游节点到融合组中
                if "tensor_meta" not in node.meta:
                    for user in node.users:
                        # 如果用户节点的操作不在可调用节点操作集合中，则跳过处理
                        if user.op not in CALLABLE_NODE_OPS:
                            continue
                        # 如果用户节点已经在融合组的节点集合中，则跳过处理
                        if user in fusion_group.nodes:
                            continue

                        # 将用户节点添加到融合组中
                        fusion_group.add_node(user)
                        # 递归地添加用户节点及其输入节点到融合组中
                        self.recursive_add_node(
                            fusion_group,
                            fusion_group.inputs,
                            visited=set(),
                        )

                # 添加节点的部分上游节点到融合组中
                for arg in node.all_input_nodes:
                    # 如果参数节点的操作不在可调用节点操作集合中，则跳过处理
                    if arg.op not in CALLABLE_NODE_OPS:
                        continue
                    # 如果参数节点的元数据中包含 "tensor_meta" 键，则跳过处理
                    if "tensor_meta" in arg.meta:
                        continue
                    # 如果参数节点已经在融合组的节点集合中，则跳过处理
                    if arg in fusion_group.nodes:
                        continue

                    # 将参数节点添加到融合组中
                    fusion_group.add_node(arg)
                    # 更新融合组的顶部节点索引为参数节点的索引
                    fusion_group.top_node_idx = min(
                        fusion_group.top_node_idx, self.nodes.index(arg)
                    )
                    # 递归地添加参数节点及其输入节点到融合组中
                    self.recursive_add_node(
                        fusion_group,
                        fusion_group.inputs,
                        visited=set(),
                    )

            # 如果融合组中的节点集合不完全包含在累积节点集合中，则从累积节点集合中移除融合组中的节点
            if not (set(fusion_group.nodes) <= self.acc_nodes):
                self.acc_nodes -= fusion_group.nodes
            else:
                # 否则，将融合组中的每个节点映射到结果字典中
                for n in fusion_group.nodes:
                    result[n] = fusion_group.nodes

        # 返回最终的结果字典
        return result
@compatibility(is_backward_compatible=False)
# 定义一个装饰器，用于标记函数不向后兼容
def legalize_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    用一个包含相同节点但拓扑排序的图替换给定 GraphModule 的图。

    这在下面的 merge_matmul 变换中使用，该变换会干扰其输入 GraphModule 的拓扑排序，
    因此在进一步变换之前需要恢复此顺序。

    参数:
        gm: 要拓扑排序的图模块。会直接修改它。

    返回:
        拓扑排序后的图模块
    """

    # 这些操作符用于在任何数据相关操作发生之前进行运行时断言。我们希望优先对这些操作进行排序，
    # 确保这些断言在图中任何数据相关操作之前出现。
    PRIORITIZED_OPS = [
        operator.add,
        operator.mul,
        operator.sub,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.le,
        operator.lt,
        operator.ge,
        operator.gt,
        operator.eq,
        operator.ne,
        torch.ops.aten.sym_constrain_range.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.msg,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten._assert_scalar.default,
    ]

    # 创建节点入度的字典，初始化为零
    indeg = dict.fromkeys(gm.graph.nodes, 0)
    # 创建一个新的图
    new_graph = torch.fx.Graph()

    # 计算每个节点的未满足依赖数量
    for node in gm.graph.nodes:
        for user in node.users:
            indeg[user] += 1

    # 使用双端队列来实现广度优先搜索
    queue: collections.deque = collections.deque()
    
    # 将所有没有依赖的节点添加到队列中
    for node in gm.graph.nodes:
        if indeg[node] == 0:
            queue.append(node)

    # 存储节点映射关系的字典
    env: Dict[torch.fx.Node, torch.fx.Node] = {}

    # 弹出队列中的节点，并添加已满足所有依赖的节点
    while len(queue) > 0:
        cur = queue.popleft()
        env[cur] = new_graph.node_copy(cur, lambda x: env[x])
        for user in cur.users:
            indeg[user] -= 1
            if indeg[user] == 0:
                if user.op == "call_function" and user.target in PRIORITIZED_OPS:
                    queue.appendleft(user)
                else:
                    queue.append(user)

    # 如果新图的大小小于旧图，则存在循环（即某些节点的依赖关系未被满足）
    if len(new_graph.nodes) < len(gm.graph.nodes):
        raise RuntimeError(f"Input graph has cycles, unable to add {[node for node in indeg if indeg[node] != 0]}")

    # 复制代码生成器并更新图模块的图
    new_graph._codegen = gm.graph._codegen
    gm.graph = new_graph

    # 返回修改后的图模块
    return gm
```