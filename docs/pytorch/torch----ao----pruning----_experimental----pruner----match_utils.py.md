# `.\pytorch\torch\ao\pruning\_experimental\pruner\match_utils.py`

```py
"""
Contains utility functions to check if a pattern is in the graph and return the matching nodes
"""
# 导入必要的库和模块
import torch
from torch import nn
from torch.ao.quantization.utils import (
    MatchAllNode,
)
from torch.fx import Node
from torch.nn.utils import parametrize
from typing import Any, Dict, List, Optional, Tuple, Union

# 定义函数 _match，用于检查单个模式节点是否匹配
def _match(modules: Dict[str, nn.ModuleDict], node: Node, current: Union[nn.Module, Any]) -> bool:
    r"""
    checks to see if a single node of a pattern matches
    """
    # 如果 current 是 MatchAllNode 的子类，返回 True
    if isinstance(current, type) and issubclass(current, MatchAllNode):
        return True
    # 如果 node 不是 Node 类型，返回 False
    if not isinstance(node, Node):
        return False
    # 如果 current 是 torch.nn.Module 的子类，并且 node 是调用模块操作，
    # 并且 parametrize.type_before_parametrizations 返回的类型与 current 相同，则返回 True
    if isinstance(current, type) and issubclass(current, torch.nn.Module):
        return (
            node.op == "call_module"
            and parametrize.type_before_parametrizations(modules[node.target])
            == current
        )
    # 如果 current 是可调用对象，并且 node 是调用函数操作，并且 node 的目标是 current，则返回 True
    elif callable(current):
        return node.op == "call_function" and node.target is current
    # 如果 current 是字符串，并且 node 的目标与 current 相同，则返回 True
    elif isinstance(current, str):
        return node.target == current
    # 其他情况返回 False
    return False

# 定义函数 apply_match，用于在图中匹配模式，并返回匹配的节点列表
def apply_match(
    modules: Dict[str, nn.ModuleDict],
    pattern: Union[Tuple[Any], Any],
    node: Node,
    matched_node_pattern: List[Node],
) -> Optional[List[Node]]:
    r"""
    This function will return the matched nodes if the pattern matches the node given
    If there is no match, it will return None
    """
    # 如果 pattern 是元组
    if isinstance(pattern, tuple):
        # 如果元组长度为 1，并且 _match 返回 True，则返回匹配节点列表加上当前节点
        if len(pattern) == 1:
            if _match(modules, node, pattern[0]):
                return matched_node_pattern + [node]

        # 取出第一个元素作为 first，其余的作为 rest
        first, *rest = pattern
        # 如果当前节点 node 匹配 first
        if _match(modules, node, first):
            # 如果 rest 为 None，则返回匹配节点列表加上当前节点
            if rest is None:
                return matched_node_pattern + [node]

            # 遍历当前节点的所有用户节点
            for user in node.users:
                # 递归调用 apply_match，匹配 rest 部分，并返回匹配节点列表加上当前节点
                return apply_match(
                    modules, tuple(rest), user, matched_node_pattern + [node]
                )
    # 如果 pattern 不是元组，直接检查当前节点是否匹配 pattern
    elif _match(modules, node, pattern):
        return [node]
    
    # 如果都不匹配，返回 None
    return None
```