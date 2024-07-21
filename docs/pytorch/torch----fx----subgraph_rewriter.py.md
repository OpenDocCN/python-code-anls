# `.\pytorch\torch\fx\subgraph_rewriter.py`

```
# 导入自定义模块和类
from .graph_module import GraphModule
from .graph import Graph
from .node import Node
from ._symbolic_trace import symbolic_trace
from ._compatibility import compatibility

# 导入标准库和第三方库
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union, TYPE_CHECKING
import torch

# 如果在类型检查模式下，导入内部匹配工具
if TYPE_CHECKING:
    from .passes.utils.matcher_with_name_node_map_utils import InternalMatch

# 定义模块内可以导入的变量名
__all__ = ['Match', 'replace_pattern', 'replace_pattern_with_filters', "ReplacedPatterns"]

# 匹配对象，表示找到的匹配节点和子图中节点到整个图中节点的映射
@compatibility(is_backward_compatible=True)
class Match(NamedTuple):
    anchor: Node  # 匹配发现的起始节点
    nodes_map: Dict[Node, Node]  # 将子图中节点映射到整个图中节点的字典

# 替换模式的对象，包含匹配的起始节点、节点映射以及添加到图中的节点列表
@compatibility(is_backward_compatible=False)
@dataclass
class ReplacedPatterns:
    anchor: Node  # 匹配发现的起始节点
    nodes_map: Dict[Node, Node]  # 将子图中节点映射到整个图中节点的字典
    replacements: List[Node]  # 添加到图中的节点列表

# 替换图模块中的属性
def _replace_attributes(gm: GraphModule, replacement: torch.nn.Module) -> None:
    # 删除所有未使用的子模块
    gm.delete_all_unused_submodules()

    # 如果替换模块是 GraphModule 类型，进行代码检查
    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    # 尝试获取模块的属性值
    def try_get_attr(gm: torch.nn.Module, target: str) -> Optional[Any]:
        module_path, _, attr_name = target.rpartition(".")
        try:
            mod: torch.nn.Module = gm.get_submodule(module_path)
        except AttributeError:
            return None
        attr = getattr(mod, attr_name, None)
        return attr

    # 遍历图模块中的每个节点
    for node in gm.graph.nodes:
        if node.op == "call_module" or node.op == "get_attr":

            # 尝试获取图模块和替换模块的属性
            gm_attr = try_get_attr(gm, node.target)
            replacement_attr = try_get_attr(replacement, node.target)

            # CASE 1: 如果目标已经存在于结果 GraphModule 中作为属性，则保留现有的子模块
            if gm_attr is not None:
                continue

            # CASE 2: 如果目标只存在于替换模块中作为属性，则进行深拷贝并添加到 GraphModule 中
            elif replacement_attr is not None:
                new_attr = copy.deepcopy(replacement_attr)
                if isinstance(replacement_attr, torch.nn.Module):
                    gm.add_submodule(node.target, new_attr)
                else:
                    setattr(gm, node.target, new_attr)

            # CASE 3: 如果目标在 gm 和 replacement 中都不存在，则抛出运行时错误
            else:
                raise RuntimeError('Attempted to create a "', node.op,
                                   '" node during subgraph rewriting '
                                   f"with target {node.target}, but "
                                   "the referenced attribute does not "
                                   "exist in the replacement GraphModule")
    gm.graph.lint()



# 调用 gm 对象的 graph 属性的 lint 方法，用于检查和修复图形数据的一致性和规范性
gm.graph.lint()
# 使用装饰器声明函数兼容性，指定是否向后兼容
@compatibility(is_backward_compatible=True)
# 定义函数replace_pattern，用于在GraphModule的图中匹配并替换指定的子图
def replace_pattern(
    gm: GraphModule,  # 参数gm：GraphModule对象，表示要操作的图
    pattern: Union[Callable, GraphModule],  # 参数pattern：可调用对象或GraphModule，表示要匹配的子图
    replacement: Union[Callable, GraphModule]  # 参数replacement：可调用对象或GraphModule，表示用来替换pattern的子图
) -> List[Match]:  # 返回类型为Match对象的列表，表示匹配到pattern的位置
    """
    Matches all possible non-overlapping sets of operators and their
    data dependencies (``pattern``) in the Graph of a GraphModule
    (``gm``), then replaces each of these matched subgraphs with another
    subgraph (``replacement``).

    Args:
        ``gm``: The GraphModule that wraps the Graph to operate on
        ``pattern``: The subgraph to match in ``gm`` for replacement
        ``replacement``: The subgraph to replace ``pattern`` with

    Returns:
        List[Match]: A list of ``Match`` objects representing the places
        in the original graph that ``pattern`` was matched to. The list
        is empty if there are no matches. ``Match`` is defined as:

        .. code-block:: python

            class Match(NamedTuple):
                # Node from which the match was found
                anchor: Node
                # Maps nodes in the pattern subgraph to nodes in the larger graph
                nodes_map: Dict[Node, Node]

    Examples:

    .. code-block:: python

        import torch
        from torch.fx import symbolic_trace, subgraph_rewriter

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)

        def pattern(w1, w2):
            return torch.cat([w1, w2]).sum()

        def replacement(w1, w2):
            return torch.stack([w1, w2])

        traced_module = symbolic_trace(M())

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

    The above code will first match ``pattern`` in the ``forward``
    method of ``traced_module``. Pattern-matching is done based on
    use-def relationships, not node names. For example, if you had
    ``p = torch.cat([a, b])`` in ``pattern``, you could match
    ``m = torch.cat([a, b])`` in the original ``forward`` function,
    despite the variable names being different (``p`` vs ``m``).

    The ``return`` statement in ``pattern`` is matched based on its
    value only; it may or may not match to the ``return`` statement in
    the larger graph. In other words, the pattern doesn't have to extend
    to the end of the larger graph.

    When the pattern is matched, it will be removed from the larger
    function and replaced by ``replacement``. If there are multiple
    matches for ``pattern`` in the larger function, each non-overlapping
    match will be replaced. In the case of a match overlap, the first
    found match in the set of overlapping matches will be replaced.
    ("First" here being defined as the first in a topological ordering
    of the Nodes' use-def relationships. In most cases, the first Node
    """
    # 实现函数主体
    pass  # 占位符，实际上函数体未被提供
    match_and_replacements = _replace_pattern(gm, pattern, replacement)
    # 使用给定的模式函数和替换函数，在图模型 gm 中查找匹配并进行替换
    return [Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements]
    # 返回一个列表，其中每个元素都是 Match 对象，包含匹配的锚点和节点映射
# 实验性 API，不向后兼容的函数装饰器
@compatibility(is_backward_compatible=False)
# 用于在图模块中替换模式匹配到的子图为指定替换的函数
def replace_pattern_with_filters(
    gm: GraphModule,
    pattern: Union[Callable, Graph, GraphModule],
    replacement: Union[Callable, Graph, GraphModule],
    match_filters: Optional[List[Callable[["InternalMatch", Graph, Graph], bool]]] = None,
    ignore_literals: bool = False,
) -> List[ReplacedPatterns]:
    """
    See replace_pattern for documentation. This function is an overload with an additional match_filter argument.

    Args:
        ``match_filters``: A list of functions that take in
            (match: InternalMatch, original_graph: Graph, pattern_graph: Graph) and return a boolean indicating
            whether the match satisfies the condition.
            See matcher_utils.py for definition of InternalMatch.
    """

    # 调用内部函数 _replace_pattern，执行实际的替换操作
    return _replace_pattern(gm, pattern, replacement, match_filters, ignore_literals)


# 内部函数，执行具体的模式替换操作
def _replace_pattern(
    gm: GraphModule,
    pattern: Union[Callable, Graph, GraphModule],
    replacement: Union[Callable, Graph, GraphModule],
    match_filters: Optional[List[Callable[["InternalMatch", Graph, Graph], bool]]] = None,
    ignore_literals: bool = False,
) -> List[ReplacedPatterns]:

    # 导入必要的模块和类
    from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch

    # 如果 match_filters 为 None，则初始化为空列表
    if match_filters is None:
        match_filters = []

    # 获取 `gm` 的图，以及 `pattern` 和 `replacement` 的图形表示
    original_graph: Graph = gm.graph

    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    elif isinstance(pattern, Graph):
        pattern_graph = pattern
    else:
        # 如果 pattern 是可调用对象，则通过 symbolic_trace 转换成图形表示
        pattern_graph = symbolic_trace(pattern).graph

    if isinstance(replacement, GraphModule):
        replacement_graph = replacement.graph
    elif isinstance(replacement, Graph):
        replacement_graph = replacement
    else:
        # 如果 replacement 是可调用对象，则通过 symbolic_trace 转换成图形表示
        replacement_graph = symbolic_trace(replacement).graph

    # 创建子图匹配器对象，并进行匹配
    matcher = SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False,
                              remove_overlapping_matches=True, ignore_literals=ignore_literals)
    _matches: List[InternalMatch] = matcher.match(original_graph)

    # 使用 match_filters 过滤不符合条件的匹配结果
    _matches = [
        m for m in _matches
        if all(match_filter(m, original_graph, pattern_graph)
               for match_filter in match_filters)
    ]

    # 获取 replacement_graph 中所有操作为 "placeholder" 的节点，作为替换的占位符
    replacement_placeholders = [n for n in replacement_graph.nodes if n.op == "placeholder"]

    # 用于记录匹配结果中节点替换后的变化
    match_changed_node: Dict[Node, Node] = {}

    # 存储匹配和替换的结果
    match_and_replacements = []

    # 重新编译 GraphModule，以反映 original_graph 的新状态
    gm.recompile()

    # 如果 replacement 是 nn.Module，需要确保所有子模块已正确复制
   `
# 判断 replacement 是否是 torch.nn.Module 类型的实例
if isinstance(replacement, torch.nn.Module):
    # 调用 _replace_attributes 函数，传入图模型（gm）和 replacement 对象
    _replace_attributes(gm, replacement)

# 返回 match_and_replacements 对象
return match_and_replacements
```