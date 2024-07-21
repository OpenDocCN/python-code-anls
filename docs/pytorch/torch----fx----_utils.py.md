# `.\pytorch\torch\fx\_utils.py`

```py
# mypy: allow-untyped-defs
# 导入 sys 模块，用于访问系统相关功能
import sys
# 导入类型提示，用于声明函数参数和返回类型
from typing import Dict, Optional

# 导入 PyTorch 库
import torch

# 导入 LazyString 类，用于延迟格式化字符串
from torch._logging import LazyString


def lazy_format_graph_code(name, gm, maybe_id=None, **kwargs):
    """
    返回一个 LazyString 对象，用于格式化图形代码。
    """

    def format_name():
        # 根据 maybe_id 的有无格式化 name
        if maybe_id is not None:
            return f"{name} {maybe_id}"
        else:
            return name

    # 如果 kwargs 中没有 "print_output" 键，设为 False
    if "print_output" not in kwargs:
        kwargs["print_output"] = False

    # 如果 kwargs 中有 "colored" 键且 stdout 不是终端，将 "colored" 设为 False
    if "colored" in kwargs and not sys.stdout.isatty():
        kwargs["colored"] = False

    # 返回 LazyString 对象，其中 lambda 函数延迟执行 _format_graph_code
    return LazyString(
        lambda: _format_graph_code(
            f"===== {format_name()} =====\n",
            gm.forward.__code__.co_filename,
            gm.print_readable(**kwargs),
        )
    )


def _format_graph_code(name, filename, graph_str):
    """
    返回一个格式化后的字符串，用于格式化图形代码。
    """
    return f"TRACED GRAPH\n {name} {filename} {graph_str}\n"


def first_call_function_nn_module_stack(graph: torch.fx.Graph) -> Optional[Dict]:
    """
    返回第一个 call_function 节点的 nn_module_stack。
    """
    for node in graph.nodes:
        # 遍历节点，找到 op 为 "call_function" 且 meta 中包含 "nn_module_stack" 的节点
        if node.op == "call_function" and "nn_module_stack" in node.meta:
            return node.meta["nn_module_stack"]
    return None


def get_node_context(node, num_nodes=2) -> str:
    """
    返回图中最后 num_nodes 个节点的字符串表示。
    """
    node_contexts = []
    cur = node
    # 遍历获取当前节点的前 num_nodes 个节点的格式化字符串表示
    for i in range(num_nodes):
        node_contexts.append(cur.format_node())
        # 如果当前节点的操作为 "root"，停止遍历
        if cur.op == "root":
            break
        cur = cur.prev
    # 将获取到的节点字符串逆序连接成一个字符串返回
    return "\n".join(node_contexts[::-1])
```