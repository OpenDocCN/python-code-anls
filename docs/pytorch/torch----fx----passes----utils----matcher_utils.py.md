# `.\pytorch\torch\fx\passes\utils\matcher_utils.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
    Node,
    Graph,
)
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os

# 限定模块导出的名称，仅包括'SubgraphMatcher'和'InternalMatch'
__all__ = ['SubgraphMatcher', 'InternalMatch']

# 设置日志记录器的初始配置
def _init_logger():
    logger = logging.getLogger(__name__)

    # 从环境变量中获取日志级别，默认为'WARNING'
    level = os.environ.get('PYTORCH_MATCHER_LOGLEVEL', 'WARNING').upper()
    logger.setLevel(level)
    
    # 设置控制台输出的日志格式
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(filename)s > %(message)s")
    console.setFormatter(formatter)
    console.setLevel(level)
    
    # 将控制台处理器添加到日志记录器
    logger.addHandler(console)
    # 阻止日志消息向上传播到父记录器
    logger.propagate = False
    return logger

# 初始化全局日志记录器
logger = _init_logger()

# 定义'InternalMatch'类，用于存储子图匹配的详细信息
@compatibility(is_backward_compatible=False)
@dataclass
class InternalMatch:
    # 匹配找到的锚节点列表
    anchors: List[Node]
    # 将模式子图中的节点映射到较大图中的节点
    nodes_map: Dict[Node, Node] = field(default_factory=dict)

    # 模式中匹配的占位符节点列表
    placeholder_nodes: List[Node] = field(default_factory=list)

    # 输出中返回的匹配子图中的节点列表
    returning_nodes: List[Node] = field(default_factory=list)

    # 将字符串名称映射到目标图中的节点的字典
    # 仅当匹配器为'SubgraphMatcherWithNameNodesMap'时才可用
    name_node_map: Dict[str, Node] = field(default_factory=dict)

    # 实现'copy'方法以复制当前对象
    def __copy__(self):
        return InternalMatch(anchors=self.anchors, nodes_map=self.nodes_map.copy(),
                             placeholder_nodes=self.placeholder_nodes.copy(),
                             returning_nodes=self.returning_nodes.copy())

# 定义'SubgraphMatcher'类，用于执行子图匹配
@compatibility(is_backward_compatible=False)
class SubgraphMatcher:
    # 类的具体实现在此省略
    # 初始化方法，用于创建 SubgraphMatcher 对象，匹配指定的图模式
    def __init__(self, pattern: Graph,
                 match_output: bool = False,
                 match_placeholder: bool = False,
                 remove_overlapping_matches: bool = True,
                 ignore_literals: bool = False) -> None:
        """
        Args:
            pattern: 目标匹配模式，以 fx.Graph 形式表示。
            match_output: 如果为 True，模式图中的输出节点将被视为目标模式的一部分。
                          如果为 False，在匹配过程中将忽略输出节点。
            match_placeholder: 如果为 True，模式图中的占位符节点将被视为目标模式的一部分。
                               如果为 False，占位符节点将被视为通配符。
            remove_overlapping_matches: 如果为 True，在重叠匹配的情况下，只返回第一个匹配。
            ignore_literals: 如果为 True，不会检查字面值是否相等，而是将其视为通配符。
        """

        self.pattern = pattern  # 将输入的模式图保存到对象属性中
        self.match_output = match_output  # 是否匹配输出节点的选项
        self.match_placeholder = match_placeholder  # 是否匹配占位符节点的选项
        self.remove_overlapping_matches = remove_overlapping_matches  # 是否移除重叠匹配的选项
        self.ignore_literals = ignore_literals  # 是否忽略字面值的选项

        if len(pattern.nodes) == 0:
            raise ValueError("SubgraphMatcher cannot be initialized with an empty pattern")
            # 如果模式图中没有节点，则抛出值错误异常

        for node in pattern.nodes:
            if node.op != "output":
                assert len(node.users) > 0, \
                       "SubgraphMatcher cannot be initialized with an pattern with dead code"
                # 对于非输出节点，确保节点有用户（被其他节点使用），否则抛出断言错误

        # TODO: assert pattern is a connected graph
        # TODO: 断言模式图是一个连通图

        # 找出模式图中所有的占位符节点
        self.pattern_placeholder_nodes = [n for n in pattern.nodes if n.op == "placeholder"]

        # 找出模式图中返回节点的所有节点
        output_node = next(iter(reversed(pattern.nodes)))
        self.pattern_returning_nodes: List[Node] = output_node.all_input_nodes

        # 初始化模式图的锚点节点列表
        self.pattern_anchors: List[Node] = []
        if match_output:
            self.pattern_anchors = [output_node]
        else:
            # 如果一个节点的唯一用户是输出节点，则该节点是图的汇点，应作为锚点进行匹配
            self.pattern_anchors = [n for n in output_node.all_input_nodes if len(n.users) == 1]
    def _match_attributes(self, pn: Node, gn: Node) -> bool:
        # Attributes matching is complicated. Right now we only support matching constant tensor
        # 断言：确保 pn.target 是一个字符串
        assert isinstance(pn.target, str), f"pn.target {pn.target} must be a string."
        # 断言：确保 gn.target 是一个字符串
        assert isinstance(gn.target, str), f"gn.target {gn.target} must be a string."

        # TODO(tmanlaibaatar) should probably make this actual API
        # 获取属性值的函数，根据模型和属性名获取属性值
        def _getattr(model: torch.fx.GraphModule, attr_name: str):
            *prefix, field = attr_name.split(".")
            t = model
            for item in prefix:
                t = getattr(t, item, None)  # type: ignore[assignment]
                assert t is not None

            return getattr(t, field)

        # 获取 pn 对应的属性值
        pn_value = _getattr(pn.graph.owning_module, pn.target)
        # 获取 gn 对应的属性值
        gn_value = _getattr(gn.graph.owning_module, gn.target)

        # 检查属性值的类型是否匹配
        if type(pn_value) != type(gn_value):
            return False

        # 不要求张量值的精确匹配
        # 如果是 torch.Tensor 类型，只需检查类型是否匹配
        if isinstance(pn_value, torch.Tensor):
            return isinstance(gn_value, torch.Tensor)
        else:
            raise RuntimeError(f"Unsupported type {pn_value} when matching attributes")
        return False

    def _nodes_are_equal(self, pn: Node, gn: Node) -> bool:
        # if exact match for placeholder is not required, then use placeholder as a wildcard
        # 如果不需要对占位符进行精确匹配，则将占位符视为通配符
        if not self.match_placeholder and pn.op == "placeholder":
            return True

        # 比较两个节点是否相等
        if pn.op == gn.op:
            if pn.op == "placeholder" or pn.op == "output":
                return True
            elif pn.op == "get_attr":
                # 对 get_attr 操作，比较其属性是否匹配
                return self._match_attributes(pn, gn)
            # 比较两个节点的目标是否相同
            return pn.target == gn.target
        return False

    def _is_contained(self, nodes_map: Dict[Node, Node]) -> bool:
        # `lookup` represents all the nodes in `original_graph`
        # that are part of `pattern`

        # Placeholders can be used by other nodes in the graphs
        # `lookup` 包含所有在 `pattern` 中的节点，除了占位符
        lookup: Dict[Node, Node] = {gn : pn for pn, gn in nodes_map.items() if pn.op != "placeholder"}

        # 遍历 `lookup` 中的每一对节点
        for gn, pn in lookup.items():
            # nodes returned by output are allowed to be used in other areas of the graph
            # 如果 pn 在 pattern_returning_nodes 中，则允许在图的其他地方使用
            if pn in self.pattern_returning_nodes:
                continue

            # 遍历 gn 的用户节点
            for user in gn.users:
                # 如果这个用户节点不在 `lookup` 中，则表示它泄漏到了模式子图之外
                if user not in lookup:
                    return False
        return True
    # 私有方法：移除重叠匹配项，保留非重叠的匹配列表
    def _remove_overlapping_matches(self, matches: List[InternalMatch]) -> List[InternalMatch]:
        # 初始化一个空列表，用于存储非重叠的匹配项
        non_overlapping_matches: List[InternalMatch] = list()
        # 用于跟踪已经匹配过的节点集合
        nodes_matched: Set[Node] = set()

        # 遍历每一个匹配项
        for match in matches:
            found_overlap = False
            # 检查当前匹配项中的节点映射
            for pn, gn in match.nodes_map.items():
                # 如果父节点的操作不是"placeholder"或"output"，且子图节点已经匹配过
                if pn.op not in {"placeholder", "output"} and gn in nodes_matched:
                    found_overlap = True
                    break

            # 如果未发现重叠
            if not found_overlap:
                # 将当前匹配项加入非重叠匹配列表
                non_overlapping_matches.append(match)
                # 将当前匹配项中的子图节点加入已匹配节点集合
                for pn, gn in match.nodes_map.items():
                    if pn.op not in {"placeholder", "output"}:
                        nodes_matched.add(gn)

        # 返回非重叠匹配列表
        return non_overlapping_matches

    # 私有方法：匹配字面量，返回是否匹配成功
    def _match_literals(self, pn: Any, gn: Any, match: InternalMatch) -> bool:
        # 断言：pn和gn不能同时为Node类型
        assert not (isinstance(pn, Node) and isinstance(gn, Node)), "pn and gn cannot both be Node"

        # 如果pn是Node而gn不是Node
        if isinstance(pn, Node) and not isinstance(gn, Node):
            # 如果pn的操作是"placeholder"
            if pn.op == "placeholder":
                # 检查在当前遍历中是否已经匹配这些节点
                if pn in match.nodes_map:
                    return match.nodes_map[pn] == gn

                # 将pn映射到gn，并返回匹配成功
                match.nodes_map[pn] = gn
                return True
            else:
                # 如果pn的操作不是"placeholder"，返回匹配失败
                return False
        # 如果pn不是Node而gn是Node，返回匹配失败
        elif not isinstance(pn, Node) and isinstance(gn, Node):
            return False
        else:
            # 如果pn和gn的类型相同且值相等，则匹配成功
            return type(gn) == type(pn) and gn == pn
```