# `.\pytorch\torch\ao\ns\fx\graph_matcher.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import collections  # 导入collections模块
import enum  # 导入enum模块

import torch  # 导入torch库
toq = torch.ops.quantized  # 设置torch.quantized操作的别名

from torch.fx import GraphModule  # 从torch.fx库中导入GraphModule类
from torch.fx.graph import Graph, Node  # 从torch.fx.graph库中导入Graph和Node类

from torch.ao.quantization.utils import getattr_from_fqn  # 从torch.ao.quantization.utils中导入getattr_from_fqn函数
from .ns_types import NSSubgraph, NSNodeTargetType  # 从当前包中导入NSSubgraph和NSNodeTargetType
from .mappings import (
    get_base_name_to_sets_of_related_ops,  # 从当前包中导入get_base_name_to_sets_of_related_ops函数
    get_unmatchable_types_map,  # 从当前包中导入get_unmatchable_types_map函数
)
from .pattern_utils import (
    get_type_a_related_to_b,  # 从当前包中导入get_type_a_related_to_b函数
    get_reversed_fusions,  # 从当前包中导入get_reversed_fusions函数
    end_node_matches_reversed_fusion,  # 从当前包中导入end_node_matches_reversed_fusion函数
)
from torch.ao.quantization import (
    ObserverBase,  # 从torch.ao.quantization中导入ObserverBase类
    FakeQuantizeBase,  # 从torch.ao.quantization中导入FakeQuantizeBase类
)

from typing import Dict, Tuple, List, Optional, Set, Any  # 导入必要的类型声明

def _get_output_nodes(g: Graph) -> List[Node]:
    # 返回给定图中操作为'output'的节点列表
    return [n for n in g.nodes if n.op == 'output']

class _NSGraphMatchableSubgraphsIterator:
    """
    Iterates through the graph of gm, starting with the output nodes
    and continuing backwards.
    1. Returns matchable subgraphs, in order. A subgraph is defined by
       (start_node, end_node).
    2. Skips over non-matchable subgraphs
    """
    def __init__(
        self,
        gm: GraphModule,
        non_matchable_functions: Set[NSNodeTargetType],
        non_matchable_modules: Set[NSNodeTargetType],
        non_matchable_methods: Set[NSNodeTargetType],
    ):
        # 初始化迭代器
        self.gm: GraphModule = gm  # 设置GraphModule对象
        self.non_matchable_functions: Set[NSNodeTargetType] = non_matchable_functions  # 设置不可匹配函数集合
        self.non_matchable_modules: Set[NSNodeTargetType] = non_matchable_modules  # 设置不可匹配模块集合
        self.non_matchable_methods: Set[NSNodeTargetType] = non_matchable_methods  # 设置不可匹配方法集合
        self.seen_nodes: Set[Node] = set()  # 初始化已访问节点集合
        self.stack: List[Node] = []  # 初始化节点栈
        for start_node in _get_output_nodes(self.gm.graph):
            self.stack.append(start_node)  # 将起始节点添加到栈中

    def __iter__(self):
        return self  # 返回迭代器本身

    def _recursively_add_node_arg_to_stack(self, arg: Any) -> None:
        """
        Adds all of the nodes in this arg to the stack, properly navigating
        through list, dicts and tuples.
        """
        if isinstance(arg, Node):
            self.stack.append(arg)  # 若参数是Node类型，则将其添加到栈中
        elif isinstance(arg, torch.fx.immutable_collections.immutable_list) or type(arg) is tuple:
            for inner_arg in arg:
                self._recursively_add_node_arg_to_stack(inner_arg)  # 若参数是不可变列表或元组，则递归地将其中节点添加到栈中
        elif isinstance(arg, torch.fx.immutable_collections.immutable_dict):
            for value in arg.values():
                self._recursively_add_node_arg_to_stack(value)  # 若参数是不可变字典，则递归地将值节点添加到栈中
    # 判断给定节点是否可匹配的私有方法，返回布尔值
    def _is_matchable(self, node: Node) -> bool:
        # 如果节点操作是调用函数
        if node.op == 'call_function':
            # 检查目标函数是否不在不可匹配函数列表中
            return node.target not in self.non_matchable_functions
        # 如果节点操作是调用模块
        elif node.op == 'call_module':
            # 确保目标是字符串类型
            assert isinstance(node.target, str)
            # 从完全限定名中获取目标模块对象
            target_mod = getattr_from_fqn(self.gm, node.target)
            # 检查目标模块是否不属于任何不可匹配模块类型
            return not \
                any(isinstance(target_mod, t)  # type: ignore[arg-type]
                    for t in self.non_matchable_modules)
        # 如果节点操作是调用方法
        elif node.op == 'call_method':
            # 检查调用的方法是否不在不可匹配方法列表中
            return node.target not in self.non_matchable_methods
        else:
            # 其他情况返回False，即节点不可匹配
            return False
# 自定义异常类，用于表示两个图无法匹配时触发的异常
class GraphMatchingException(Exception):
    """
    当两个图无法匹配时引发的异常。
    """
    pass

# 枚举类型，表示子图关系
class SubgraphTypeRelationship(enum.Enum):
    # 相同类型且已知
    # 示例：F.linear 和 F.linear，或 nn.Conv2d 和 nn.Conv2d
    EQUAL = enum.auto()
    # 相同类型，但类型对 Numerical Suite 不可知（如用户定义类型等）
    EQUAL_BUT_UKNOWN = enum.auto()
    # 已知，子图关系集相同，但类型不同
    # 示例：F.linear 和 toq.linear
    RELATED_BUT_NOT_EQUAL = enum.auto()
    # 不相关
    NOT_RELATED = enum.auto()

# 函数用于获取子图关系类型
def _get_subgraph_relationship_type(
    subgraph_a: NSSubgraph,
    subgraph_b: NSSubgraph,
    gm_a: GraphModule,
    gm_b: GraphModule,
    type_a_related_to_b: Set[Tuple[NSNodeTargetType, NSNodeTargetType]],
) -> SubgraphTypeRelationship:
    node_a = subgraph_a.base_op_node
    node_b = subgraph_b.base_op_node

    # TODO: 下一个版本需处理通过基本操作前的匹配
    # 如果节点操作不同，且不是 'call_function' 或 'call_method'，则返回不相关
    if node_a.op != node_b.op:
        if not (
            node_a.op in ('call_function', 'call_method') and
            node_b.op in ('call_function', 'call_method')
        ):
            return SubgraphTypeRelationship.NOT_RELATED

    # 如果节点操作为 'call_function' 或 'call_method'
    if node_a.op in ('call_function', 'call_method'):
        key = (node_a.target, node_b.target)

        # 如果 key 不在 type_a_related_to_b 中
        if key not in type_a_related_to_b:
            # 如果目标相同，则返回 EQUAL_BUT_UKNOWN，否则返回 NOT_RELATED
            if node_a.target == node_b.target:
                return SubgraphTypeRelationship.EQUAL_BUT_UKNOWN
            else:
                return SubgraphTypeRelationship.NOT_RELATED
        
        # 如果目标相同
        if node_a.target == node_b.target:
            node_a_has_prev = subgraph_a.base_op_node == subgraph_a.start_node
            node_b_has_prev = subgraph_b.base_op_node == subgraph_b.start_node
            # 如果 node_a 有前导节点而 node_b 没有，则返回 RELATED_BUT_NOT_EQUAL
            if node_a_has_prev and (not node_b_has_prev):
                return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
            # 如果 node_a 没有前导节点而 node_b 有，则返回 RELATED_BUT_NOT_EQUAL
            elif (not node_a_has_prev) and node_b_has_prev:
                return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
            # 如果两者均没有前导节点，则返回 EQUAL，否则返回 EQUAL
            elif (not node_a_has_prev) and (not node_b_has_prev):
                return SubgraphTypeRelationship.EQUAL
            else:
                # TODO（未来的PR）：检查匹配的 start_op_node 和 base_op_node
                return SubgraphTypeRelationship.EQUAL

        # 如果 key 在 type_a_related_to_b 中
        if key in type_a_related_to_b:
            return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
        else:
            return SubgraphTypeRelationship.NOT_RELATED
    # 如果节点 A 的操作是 'call_module'，则执行以下逻辑
    elif node_a.op == 'call_module':
        # 断言子图 A 和 B 的基本操作节点与起始节点相同，否则抛出异常
        assert (subgraph_a.base_op_node == subgraph_a.start_node and
                subgraph_b.base_op_node == subgraph_b.start_node), \
            "Matching call_module patterns where base_op_node != start_node is not supported yet"
        
        # 对于 call_module 操作，需要查找模块以进行类型检查
        assert isinstance(node_a.target, str)
        # 从模型 gm_a 中获取完全限定名为 node_a.target 的模块
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        
        assert isinstance(node_b.target, str)
        # 从模型 gm_b 中获取完全限定名为 node_b.target 的模块
        mod_b = getattr_from_fqn(gm_b, node_b.target)
        
        # 创建键，用于确定模块类型的关系
        key = (type(mod_a), type(mod_b))

        # 如果 key 不在 type_a_related_to_b 中，则根据模块类型返回关系
        if key not in type_a_related_to_b:
            if type(mod_a) == type(mod_b):
                return SubgraphTypeRelationship.EQUAL_BUT_UKNOWN
            else:
                return SubgraphTypeRelationship.NOT_RELATED
        elif type(mod_a) == type(mod_b):
            return SubgraphTypeRelationship.EQUAL
        else:
            return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL

    # 默认情况下，返回子图类型关系为 NOT_RELATED
    return SubgraphTypeRelationship.NOT_RELATED
# 返回一个唯一的子图名称。该名称基于两个因素：
# 1. 包含子图中基本操作底层类型名称的集合的名称（例如，如果与线性操作相关，则为'torch.nn.functional.linear'）
# 2. 具有相关基本操作底层类型的先前子图数量

def _get_name_for_subgraph(
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]],
    existing_names: Set[str],
) -> str:
    """
    Returns a unique name for a subgraph. This name is based on two things:
    1. the name of the set containing the underlying type of the base op in the
       subgraph (i.e. 'torch.nn.functional.linear' if this is related to a linear op)
    2. the number of previous subgraphs with related underlying type of the base op

    For example, in the graph

    linear0 -> relu0 -> linear1 -> relu1

    The subgraphs are (linear0, relu0) and (linear1, relu1).  If we iterate
    from the output node backwards, the name given to (linear1, relu1) will be
    `base_op_torch.nn.functional.linear_0`, and the name given to (linear0, relu0)
    will be `base_op_torch.nn.functional.linear_1`.

    Why are we not just using the node name? Answer: because of two requirements:
    A. fusions must be supported
    B. some Numeric Suite APIs can be called without having all of the models in memory

    For example, let's say we need to match nodes of

    (1) ... -> linear0 -> relu0 -> ...

    And

    (2) ... -> linear_relu0 -> ...

    Without being able to inspect them together. With the current naming scheme, if
    we iterate through both of these graphs in the same order, and assuming the rest
    of the graphs match, both of these subgraphs will get the same name without
    (1) and (2) knowing anything about each other.
    """
    # 获取子图中基本操作节点的目标类型
    target_type = _get_node_target_type(subgraph_a.base_op_node, gm_a)
    target_base_type = None
    # 遍历基本名称到相关操作集合的字典
    for base_name, sets_of_related_ops in base_name_to_sets_of_related_ops.items():
        # 如果目标类型在相关操作集合中，选择该基本名称
        if target_type in sets_of_related_ops:
            target_base_type = base_name
    # 创建基于目标基本类型的名称
    target_base_name = 'base_op_' + str(target_base_type)
    counter = 0
    # 提议的名称包含基本名称和计数器
    proposed_name = target_base_name + '_' + str(counter)
    # 确保名称唯一，如果已存在则增加计数器直到找到唯一名称
    while proposed_name in existing_names:
        counter += 1
        proposed_name = target_base_name + '_' + str(counter)
    # 将新名称添加到现有名称集合中
    existing_names.add(proposed_name)
    # 返回建议的唯一名称
    return proposed_name

# 获取节点的目标类型
def _get_node_target_type(node: Node, gm: GraphModule) -> Optional[NSNodeTargetType]:
    if node.op in ('call_function', 'call_method'):
        return node.target
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)
        return type(mod)
    # 如果不是特定类型的节点，则返回空
    return None

# 匹配图A和图B的可匹配子图对
def get_matching_subgraph_pairs(
    gm_a: GraphModule,
    gm_b: GraphModule,
    base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
    unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
) -> Dict[str, Tuple[NSSubgraph, NSSubgraph]]:
    """
    Matches matchable subgraphs of graph_a to graph_b.

    For a node, "matchable" is defined as a node which is not an observer,
    fake_quants, quant or dequant.
    """
    """
    如果未提供不可匹配类型映射，则获取默认的不可匹配类型映射。
    包含不可匹配函数、模块和方法的字典。
    """
    if unmatchable_types_map is None:
        unmatchable_types_map = get_unmatchable_types_map()
    
    """
    使用不可匹配类型映射中的不可匹配函数、模块和方法，创建图 A 的可匹配子图迭代器。
    """
    graph_a_iterator = _NSGraphMatchableSubgraphsIterator(
        gm_a, non_matchable_functions, non_matchable_modules,
        non_matchable_methods)
    
    """
    使用不可匹配类型映射中的不可匹配函数、模块和方法，创建图 B 的可匹配子图迭代器。
    """
    graph_b_iterator = _NSGraphMatchableSubgraphsIterator(
        gm_b, non_matchable_functions, non_matchable_modules,
        non_matchable_methods)
    
    """
    初始化一个有序字典来存储匹配结果。
    """
    results = collections.OrderedDict()
    
    """
    如果未提供基础名称到相关操作集合的映射，则获取默认的基础名称到相关操作集合的映射。
    """
    if base_name_to_sets_of_related_ops is None:
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    # 从 base_name_to_sets_of_related_ops 中获取与类型 A 相关的信息
    type_a_related_to_b = \
        get_type_a_related_to_b(base_name_to_sets_of_related_ops)

    # 初始化用于存储已存在名称的集合
    existing_names_a: Set[str] = set()
    existing_names_b: Set[str] = set()

    while True:
        # 获取下一个子图，从 graph_a_iterator 和 graph_b_iterator 中分别获取
        cur_subgraph_a, cur_subgraph_b = None, None
        try:
            cur_subgraph_a = next(graph_a_iterator)
        except StopIteration:
            pass
        try:
            cur_subgraph_b = next(graph_b_iterator)
        except StopIteration:
            pass

        # 查找节点 a 和 b 的类型，用于生成有用的错误消息
        type_start_a, type_start_b = None, None
        if cur_subgraph_a is not None:
            type_start_a = _get_node_target_type(cur_subgraph_a.start_node, gm_a)
        if cur_subgraph_b is not None:
            type_start_b = _get_node_target_type(cur_subgraph_b.start_node, gm_b)

        # 检查结果并确定下一步操作
        if cur_subgraph_a is not None and cur_subgraph_b is not None:
            # 如果两个节点都被获取到，检查子图关系类型
            # 注意：子图关系类型是在起始节点上进行检查的，例如
            # 如果检查线性-relu模式，我们会检查线性的子图关系类型
            subgraph_relationship = _get_subgraph_relationship_type(
                cur_subgraph_a, cur_subgraph_b,
                gm_a, gm_b, type_a_related_to_b)
            if subgraph_relationship == SubgraphTypeRelationship.NOT_RELATED:
                msg = f"""
    # 检查并匹配两个模型中的子图对，确保它们是相关的或相等的，否则抛出异常
    for (cur_subgraph_a, type_start_a), (cur_subgraph_b, type_start_b) in zip(gm_a, gm_b):
        if cur_subgraph_a is not None and cur_subgraph_b is not None:
            # 检查子图之间的关系类型
            subgraph_relationship = _determine_relationship(cur_subgraph_a, cur_subgraph_b)
            if subgraph_relationship == SubgraphTypeRelationship.UNRELATED:
                # 如果子图不相关，则抛出异常
                msg = f"""
The subgraphs
({cur_subgraph_a}, {type_start_a}) and
({cur_subgraph_b}, {type_start_b})
are not related. Please ensure that the two models you pass in have the same number
of subgraphs, and each pair of subgraphs is related to each other."""
                raise GraphMatchingException(msg)
            elif subgraph_relationship == SubgraphTypeRelationship.EQUAL_BUT_UKNOWN:
                # 跳过匹配但类型未知的情况
                continue
            # 获取子图在图中的名称
            key_name_a = _get_name_for_subgraph(
                cur_subgraph_a, gm_a, base_name_to_sets_of_related_ops,
                existing_names_a)
            key_name_b = _get_name_for_subgraph(
                cur_subgraph_b, gm_b, base_name_to_sets_of_related_ops,
                existing_names_b)
            # 断言子图名称应该相等，否则抛出异常
            assert key_name_a == key_name_b, \
                f"Subgraph names {key_name_a} and {key_name_b} do not match"
            # 将匹配成功的子图对加入结果字典中
            results[key_name_a] = (cur_subgraph_a, cur_subgraph_b)
            # 继续下一个子图对的匹配
            continue
        elif cur_subgraph_a is None and cur_subgraph_b is None:
            # 如果两个模型的子图都为空，则表示已经遍历完所有子图
            break
        else:
            # 如果只有一个模型的子图为空，则抛出异常
            msg = f"""
Attempting to match
({cur_subgraph_a}, {type_start_a}) and
({cur_subgraph_b}, {type_start_b}),
one of which is empty. Please ensure that the two models you pass in have the same number
of subgraphs."""
            raise GraphMatchingException(msg)

    # 原始的子图对是从输出到输入遍历两个图形创建的，反转结果以返回执行顺序的子图
    results = collections.OrderedDict(reversed(list(results.items())))

    # 返回最终匹配成功的子图对结果
    return results
```