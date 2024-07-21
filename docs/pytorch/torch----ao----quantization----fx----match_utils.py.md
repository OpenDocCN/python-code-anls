# `.\pytorch\torch\ao\quantization\fx\match_utils.py`

```py
# mypy: allow-untyped-defs
# 导入系统模块和torch模块
import sys
import torch
# 导入torch.fx.graph模块中的Graph和Node类
from torch.fx.graph import (
    Graph,
    Node,
)
# 导入torch.ao.quantization.utils模块中的Pattern类
from torch.ao.quantization.utils import Pattern
# 导入当前目录下的quantize_handler模块中的QuantizeHandler类
from .quantize_handler import (
    QuantizeHandler,
)
# 导入..qconfig模块中的QConfigAny类
from ..qconfig import (
    QConfigAny,
)
# 导入..utils模块中的MatchAllNode类
from ..utils import (
    MatchAllNode
)
# 导入当前目录下的graph_module模块中的_is_observed_standalone_module函数
from .graph_module import (
    _is_observed_standalone_module,
)
# 导入torch.nn.utils.parametrize模块中的type_before_parametrizations函数
from torch.nn.utils.parametrize import type_before_parametrizations
# 导入类型提示
from typing import Any, Dict, List, Callable, Optional, Tuple, Type, Set, Iterable

# 定义空列表__all__，用于声明公开的模块成员
__all__: List[str] = []

# 定义_MatchResult类型，它是一个元组，包含Node、List[Node]、Optional[Pattern]和QuantizeHandler类型
_MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler]

# 定义_MatchResultWithQConfig类型，它是一个元组，包含Node、List[Node]、Optional[Pattern]、QuantizeHandler和QConfigAny类型
_MatchResultWithQConfig = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler,
                                QConfigAny]

# 注意: 模式的顺序很重要！match函数会选择第一个匹配的模式，因此我们需要将融合模式放在单一模式之前。
# 例如，add_relu应该在relu之前注册。装饰器的应用顺序与我们看到的相反。还有，在与这些模式匹配的图节点时，
# 我们将从图的最后一个节点开始向后遍历。
def _is_match(modules, node, pattern, max_uses=sys.maxsize):
    """ Matches a node in fx against a pattern
    匹配fx中的节点与模式
    """
    if isinstance(pattern, tuple):
        self_match, *arg_matches = pattern
        if self_match is getattr:
            assert len(pattern) == 2, 'Expecting getattr pattern to have two elements'
            arg_matches = []
    else:
        self_match = pattern
        arg_matches = []

    if isinstance(self_match, type) and issubclass(self_match, MatchAllNode):
        return True

    if node == pattern:
        return True

    if not isinstance(node, Node) or len(node.users) > max_uses:
        return False

    if isinstance(self_match, type) and issubclass(self_match, torch.nn.Module):
        if node.op != 'call_module':
            return False
        if not type_before_parametrizations(modules[node.target]) == self_match:
            return False
    elif callable(self_match):
        if node.op != 'call_function' or node.target is not self_match:
            return False
        elif node.target is getattr:
            if node.args[1] != pattern[1]:
                return False
    elif isinstance(self_match, str):
        if node.op != 'call_method' or node.target != self_match:
            return False
    elif node.target != self_match:
        return False

    if not arg_matches:
        return True

    if len(arg_matches) != len(node.args):
        return False

    return all(_is_match(modules, node, arg_match, max_uses=1) for node, arg_match in zip(node.args, arg_matches))
def _find_matches(
        graph: Graph,
        modules: Dict[str, torch.nn.Module],
        patterns: Dict[Pattern, QuantizeHandler],
        root_node_getter_mapping: Dict[Pattern, Callable],
        standalone_module_names: Optional[List[str]] = None,
        standalone_module_classes: Optional[List[Type]] = None,
        custom_module_classes: Optional[List[Any]] = None) -> Dict[str, _MatchResult]:
    """
    Matches the nodes in the input graph to quantization patterns, and
    outputs the information needed to quantize them in future steps.

    Inputs:
      - graph: an fx.Graph object
      - modules: a mapping of fully qualified module name to instance,
          for example, {'foo': ModuleFoo, ...}
      - patterns: a mapping from a tuple of nodes in reverse order to
          uninitialized QuantizeHandler subclass.
      - root_node_getter_mapping: a mapping from patterns to functions that retrieve root nodes
      - standalone_module_names: optional list of standalone module names
      - standalone_module_classes: optional list of standalone module classes
      - custom_module_classes: optional list of custom module classes

    Outputs a map of
      node_name ->
        (node, matched_values, matched_pattern, QuantizeHandler instance,
         qconfig)

    For example, {
      'relu_1': (relu_1, [relu_1], torch.nn.functional.relu,
                 <CopyNodeQuantizeHandler instance>, QConfig(...)),
      ...
    }
    """
    if custom_module_classes is None:
        custom_module_classes = []

    if standalone_module_classes is None:
        standalone_module_classes = []

    if standalone_module_names is None:
        standalone_module_names = []

    # Initialize an empty dictionary to store the matching results
    match_map: Dict[str, _MatchResult] = {}
    
    # Initialize an empty set to keep track of all matched node names
    all_matched : Set[str] = set()

    def _recursive_record_node_in_match_map(
            last_node,
            match_map,
            node_pattern,
            matched_node_pattern,
            pattern,
            match_value):
        """
        Recursively records matched nodes into the match_map.

        Args:
          - last_node: the last node processed
          - match_map: the dictionary storing matched results
          - node_pattern: current node pattern to match
          - matched_node_pattern: pattern of the matched node
          - pattern: quantization pattern associated with the node
          - match_value: value indicating the match

        If the node_pattern is a Node object, it adds the match to match_map;
        otherwise, it recursively processes each element in node_pattern.
        """
        if isinstance(node_pattern, Node):
            match_map[node_pattern.name] = (
                last_node, matched_node_pattern, pattern, match_value)
        elif not isinstance(node_pattern, Iterable):
            return
        else:
            for n in node_pattern:
                _recursive_record_node_in_match_map(last_node, match_map, n, matched_node_pattern, pattern, match_value)

    # TODO: 1. merge with fuse matcher 2. document the code
    def record_match(
            pattern,
            node,
            last_node,
            matched_node_pattern,
            match_map):
        # 如果模式是一个元组，解构第一个元素作为操作符，剩余元素作为参数
        if isinstance(pattern, tuple):
            s, *args = pattern
            # 判断是否只有一个参数
            is_single_arg = len(args) == 1
            # 用于存储当前节点模式的列表
            current_node_pattern: List[Node] = []
            # 递归调用record_match处理第一个元素作为新的模式
            record_match(
                s,
                node,
                last_node,
                matched_node_pattern,
                match_map)
            # 如果第一个元素不是getattr，则遍历参数与节点的args进行匹配
            if pattern[0] is not getattr:
                for subpattern, arg in zip(args, node.args):
                    record_match(
                        subpattern,
                        arg,
                        node,
                        current_node_pattern,
                        match_map)
            # 如果current_node_pattern长度大于1，将其加入到matched_node_pattern中
            if len(current_node_pattern) > 1:
                # current_node_pattern是通过与节点的args匹配得到的节点模式
                # 使用is_single_arg来恢复模式的原始结构
                # 如果原始模式只有一个参数，则会是(original_op, (original_arg, ...))
                # 否则，会是参数列表 (original_op, arg0, arg1, arg2, ...)
                if is_single_arg:
                    matched_node_pattern.append(tuple(current_node_pattern))
                else:
                    matched_node_pattern.extend(list(current_node_pattern))
            else:
                matched_node_pattern.append(current_node_pattern[0])
        else:
            # 如果模式不是元组，则直接将节点添加到matched_node_pattern中
            matched_node_pattern.append(node)
    # 对图中的节点进行逆序遍历
    for node in reversed(graph.nodes):
        # 如果节点名称不在match_map中，并且不在all_matched中
        if node.name not in match_map and node.name not in all_matched:
            # 遍历patterns字典中的每一个模式和对应的处理类
            for pattern, quantize_handler_cls in patterns.items():
                # 获取根节点的获取器，如果没有则为None
                root_node_getter = root_node_getter_mapping.get(pattern, None)
                # 判断当前节点是否与当前模式匹配，并且节点名称不在match_map中
                if _is_match(modules, node, pattern) and node.name not in match_map:
                    # 创建一个空列表用于存储匹配到的节点模式
                    matched_node_pattern: List[Node] = []
                    # 记录匹配结果到match_map中
                    record_match(
                        pattern,
                        node,
                        node,
                        matched_node_pattern,
                        match_map)
                    # 创建量化处理器对象，处理器类型为quantize_handler_cls
                    quantize_handler = quantize_handler_cls(  # type: ignore[operator]
                        matched_node_pattern,
                        modules,
                        root_node_getter)
                    # 记录最后一个节点
                    last_node = node
                    # 递归记录匹配节点模式中的所有节点到match_map中
                    _recursive_record_node_in_match_map(
                        last_node,
                        match_map,
                        # 需要记录匹配模式中的所有节点到match_map中
                        matched_node_pattern,
                        # 这是与节点对应的值的一部分
                        matched_node_pattern,
                        pattern,
                        quantize_handler)
                    # 中断当前模式的遍历，继续下一个节点
                    break

    # 将自定义模块实例添加到匹配结果中
    assert modules is not None
    for node in graph.nodes:
        # 如果节点操作为'call_module'且其目标模块类型在custom_module_classes中
        if node.op == 'call_module' and \
           type(modules[node.target]) in custom_module_classes:
            # 将节点添加到match_map中，并创建量化处理器对象标记为自定义模块
            match_map[node.name] = (
                node, node, None, QuantizeHandler(node, modules, is_custom_module=True))

    def is_standalone_module(node_target: str, modules: Dict[str, torch.nn.Module]):
        # 断言确保modules不为None
        assert modules is not None
        # 判断节点目标是否为独立模块名称之一或其模块类型是否在standalone_module_classes中
        return (
            node_target in standalone_module_names or  # type: ignore[operator]
            type(modules[node_target]) in standalone_module_classes  # type: ignore[operator]
        )

    # 将独立模块添加到匹配结果中
    for node in graph.nodes:
        # 如果节点操作为'call_module'且节点目标为独立模块或是观察到的独立模块
        if node.op == 'call_module' and \
           (is_standalone_module(node.target, modules) or
                _is_observed_standalone_module(modules[node.target])):
            # 将节点添加到match_map中，并创建量化处理器对象标记为独立模块
            match_map[node.name] = (
                node, node, None,
                QuantizeHandler(node, modules, is_standalone_module=True))

    # 返回匹配结果的字典match_map
    return match_map
```