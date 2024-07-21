# `.\pytorch\torch\onnx\_internal\fx\passes\_utils.py`

```py
# mypy: allow-untyped-defs
"""Common utility functions for FX passes.

These functions should NOT be directly invoked outside of `passes` package.
"""
from __future__ import annotations

import collections  # 导入collections模块，用于处理数据结构相关操作

import re  # 导入re模块，用于正则表达式操作

from typing import Callable, Dict, Optional, Tuple  # 导入类型提示相关模块

import torch.fx  # 导入torch.fx模块，用于构建和操作FX图
import torch.fx.traceback as fx_traceback  # 导入FX图的跟踪模块
from torch.onnx._internal import _beartype  # 导入_beartype模块，用于类型注解


@_beartype.beartype
def wrap_graph_module_for_node_meta_preservation(
    graph_module: torch.fx.GraphModule,
) -> Callable:
    """Wrap a GraphModule with contexts to preserve node meta information, such as stacktrace info.

    This is typically useful before calling `make_fx`. Without this wrapper, the
    stacktrace information will be lost afterwards.
    """
    # 定义一个函数，用于给GraphModule添加上下文，以保留节点元信息（如堆栈跟踪信息）
    def wrapped(*args):
        with fx_traceback.preserve_node_meta():  # 使用fx_traceback模块保存节点元信息的上下文
            return torch.fx.Interpreter(graph_module).run(*args)  # 运行GraphModule的解释器并返回结果

    return wrapped  # 返回包装后的函数


def _get_node_base_name(node_name: str) -> Tuple[str, Optional[int]]:
    pattern = r"(.*)\.(\d+)"
    match = re.match(pattern, node_name)  # 使用正则表达式匹配节点名称中的基础名称和计数后缀
    if match is not None:
        base_name, count_str = match.groups()  # 获取匹配到的基础名称和计数后缀
        return base_name, int(count_str)  # 返回基础名称和计数后缀的整数形式
    return node_name, None  # 若没有匹配到，则直接返回节点名称和None


@_beartype.beartype
def set_node_name(
    node: torch.fx.Node,
    new_name: str,
    name_to_node_cache: Dict[str, torch.fx.Node],
):
    """Safely set the unique name of a node.

    If the new name is already taken by another node, the name of the other node will be
    updated. If `new_name` is a string of format f"{base_name}.{count}", where `count`
    is an integer, the other node will be renamed as f"{base_name}.{count+1}". If not,
    the other node will be renamed as "{new_name}.1". This function will iteratively
    update the names until there is no conflict.

    ``name_to_node_cache`` is required as an argument to avoid recomputation. The caller
    is responsible for ensuring the cache is accurate and in sync with the owning module
    of the node. The values in the cache will be updated accordingly.

    Args:
        node: The node to update.
        new_name: The new name to use.
        name_to_node_cache: A cache of node names to nodes.
    """
    module = node.graph.owning_module  # 获取节点所属的模块

    node_name_to_set = collections.deque([(node, new_name)])  # 创建一个双向队列，用于保存需要设置名称的节点

    while node_name_to_set:  # 循环直到队列为空
        node, new_name = node_name_to_set.pop()  # 弹出队列中的节点和新名称
        if new_name in name_to_node_cache and name_to_node_cache[new_name] != node:
            # 如果新名称已经存在于缓存中，并且对应的节点不是当前节点
            base_name, postfix_count = _get_node_base_name(new_name)  # 获取新名称的基础名称和计数后缀
            if postfix_count is None:
                postfix_count = 0
            node_name_to_set.append(
                (name_to_node_cache[new_name], f"{base_name}.{postfix_count + 1}")
            )  # 将需要重命名的节点和新的名称加入队列
        node.name = new_name  # 设置节点的新名称
        name_to_node_cache[new_name] = node  # 更新名称到节点的缓存


@_beartype.beartype
def replace_placeholder_name_and_target(
    module: torch.fx.GraphModule, reference_module: torch.fx.GraphModule
):
    """Replace the argument names in module with those in reference_module.

    This function updates the names of the arguments in `module` to match those in
    `reference_module`. It ensures that all placeholders in `module` are replaced
    with the corresponding names in `reference_module`.

    Args:
        module: The module whose arguments need replacement.
        reference_module: The module providing the reference names.
    """
    """
    This function assumes the two modules have the same signature structure.
    The caller is responsible for ensuring this. Otherwise, the behavior of this
    function is undefined. This function only does minimal sanity check that the two
    modules have the same number of arguments.

    Name conflicts between new names and existing node names in the graph are handled.
    Check the documentation of :func:`set_node_name` for more details.

    Raises:
        RuntimeError: If the two modules have different number of arguments.
    """
    # 获取当前模块中所有操作为 "placeholder" 的节点列表
    placeholders = [node for node in module.graph.nodes if node.op == "placeholder"]
    # 获取参考模块中所有操作为 "placeholder" 的节点列表
    reference_placeholders = [
        node for node in reference_module.graph.nodes if node.op == "placeholder"
    ]

    # 检查当前模块和参考模块的占位符节点数量是否相同，若不同则引发运行时错误
    if len(placeholders) != len(reference_placeholders):
        raise RuntimeError(
            "The two modules have different number of arguments. "
            f"module: {len(placeholders)}, reference_module: {len(reference_placeholders)}"
        )

    # 创建一个字典，将当前模块中的节点名映射到节点对象
    name_to_node: Dict[str, torch.fx.Node] = {}
    for node in module.graph.nodes:
        name_to_node[node.name] = node

    # 遍历当前模块和参考模块的占位符节点，更新当前模块的目标和节点名称
    for placeholder, reference_placeholder in zip(placeholders, reference_placeholders):
        placeholder.target = reference_placeholder.target
        set_node_name(placeholder, reference_placeholder.name, name_to_node)

    # 重新编译当前模块，应用更新后的节点目标和名称
    module.recompile()
```