# `.\pytorch\torch\fx\experimental\merge_matmul.py`

```
# mypy: allow-untyped-defs
# 引入 torch 库
import torch

# 从 torch.fx.node 模块中引入 Node 类
from torch.fx.node import Node
# 从 torch.fx._symbolic_trace 模块中引入 symbolic_trace 函数
from torch.fx._symbolic_trace import symbolic_trace
# 从 torch.fx.passes.tools_common 模块中引入 legalize_graph 函数
from torch.fx.passes.tools_common import legalize_graph
# 引入 itertools 模块，用于操作迭代器和生成器的工具
import itertools
# 引入 operator 模块，提供了一些内置的操作符函数
import operator

# 从 typing 模块中引入类型提示工具
from typing import Dict, List, Tuple


def split_result_tensors(
    result: torch.Tensor, inputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    """
    A free function for use in the merge_matmul graph transformation below that
    splits the output from a merged matmul into the individual results for each
    input tensor.

    Arguments:
        result: The merged matmul result tensor.
        inputs: The list of inputs that were merged into one for the matmul.

    Returns:
        List of matmul results for each input tensor.
    """
    # 当 fx tracer 正在运行时，x.shape[0] 将是 torch.fx.Attribute 类型，但我们需要一个 int 类型
    # 即使在追踪时也需要
    if isinstance(result, torch.fx.Proxy):
        # 如果 result 是 torch.fx.Proxy 类型，则 splits 列表初始化为全零列表
        splits = [0] * len(inputs)
    else:
        # 否则，splits 列表中的每个元素为对应输入张量的第一维度大小
        splits = [x.shape[0] for x in inputs]

    # 使用 torch.split 函数将 result 张量按照 splits 列表进行分割，并返回分割后的结果
    return torch.split(result, splits)


def may_depend_on(a: Node, b: Node, search_depth: int = 6):
    """
    Determine if one node depends on another in a torch.fx.Graph.

    Arguments:
        a: The node that may have a dependency on b.
        b: The node that a may have a dependency on.
        search_depth: In the case of an indirect dependency, this function
                        searches upto this many nodes away in search of a
                        data dependency. If none is found, the function
                        makes the conservative assumption that there is a
                        dependency.

    Returns:
        True if a may depend on b, False if it definitely does not.
    """
    # 如果 a 和 b 是同一个节点对象，则认为它们存在依赖关系
    if a == b:
        return True

    # 如果节点 a 没有输入节点，则认为它不依赖于节点 b
    if len(a.all_input_nodes) == 0:
        return False

    # 如果搜索深度为 0，则认为可能存在数据依赖关系
    if search_depth == 0:
        return True

    # 递归检查节点 a 的所有输入节点
    for inp in a.all_input_nodes:
        # 如果节点 inp 可能依赖于节点 b，则认为节点 a 可能依赖于节点 b
        if may_depend_on(inp, b, search_depth - 1):
            return True

    # 如果以上条件都不满足，则认为节点 a 不依赖于节点 b
    return False


def are_nodes_independent(nodes: List[Node]):
    """
    Check if all of the given nodes are pairwise-data independent.

    Arguments:
        nodes: The nodes to check for data dependencies.

    Returns:
        True if any pair in nodes has a data dependency.
    """
    # 对于节点列表 nodes 中的每一对节点 (i, j)
    for i, j in itertools.combinations(nodes, 2):
        # 如果节点 i 依赖于节点 j 或者节点 j 依赖于节点 i，则它们不是独立的
        if may_depend_on(i, j) or may_depend_on(j, i):
            return False

    # 如果所有节点对都没有数据依赖关系，则它们是独立的
    return True


def merge_matmul(in_mod: torch.nn.Module):
    """
    A graph transformation that merges matrix multiplication operations that share the same right-hand
    ```
    """
    gm = symbolic_trace(in_mod)

    rhs_users: Dict[Node, List[Node]] = {}
    lhs_users: Dict[Node, List[Node]] = {}

    # Populate rhs_users and lhs_users - maps from LHS/RHS matrix multiply operands to
    # the matmul of which they are the LHS/RHS.
    遍历图中的每个节点
    for node in gm.graph.nodes:
        如果节点的操作不是函数调用或者目标不是 torch.matmul，则跳过
        if node.op != "call_function" or node.target is not torch.matmul:
            continue

        从节点的参数中获取左右操作数
        lhs, rhs = node.args

        # TODO: Properly handle aliasing caused by get_attr. For now,
        # use the attribute name as the operand if the node is a
        # get_attr.
        如果左操作数的操作是 "get_attr"，则使用目标作为左操作数
        lhs = lhs.target if lhs.op == "get_attr" else lhs
        如果右操作数的操作是 "get_attr"，则使用目标作为右操作数
        rhs = rhs.target if rhs.op == "get_attr" else rhs

        将节点与其左操作数的映射添加到 lhs_users 字典中
        lhs_users.setdefault(lhs, []).append(node)
        将节点与其右操作数的映射添加到 rhs_users 字典中
        rhs_users.setdefault(rhs, []).append(node)
    for rhs, mms in rhs_users.items():
        # 对于每个右手边 rhs，遍历其对应的多个矩阵乘法节点 mms
        # 如果矩阵乘法节点数量小于 2，跳过
        if len(mms) < 2:
            continue

        # 所有的矩阵乘法节点必须在直接或间接上不相互依赖，才能进行合并
        if not are_nodes_independent(mms):
            continue

        # 收集所有左手边操作数
        lhs_vals = [mm.args[0] for mm in mms]

        # 合并矩阵乘法操作
        # 收集左手边操作数和单个右手边操作数
        lhs = [gm.graph.get_attr(l) if isinstance(l, str) else l for l in lhs_vals]
        rhs = gm.graph.get_attr(rhs) if isinstance(rhs, str) else rhs

        # 将所有左手边操作数拼接起来
        merge_mm_cat = gm.graph.call_function(torch.cat, (lhs,), {})

        # 使用拼接后的左手边操作数与右手边操作数进行矩阵乘法
        # 这将产生与原始图中涉及 rhs 的所有单独矩阵乘法相同的结果，但它们将被全部连接在一起
        merge_mm = gm.graph.call_function(torch.matmul, (merge_mm_cat, rhs,), {})

        # 使用左手边操作数的形状来拆分合并后的矩阵乘法的结果
        merge_mm_split = gm.graph.call_function(
            split_result_tensors, (merge_mm, lhs), {}
        )
        merge_mm_res = [
            # 获取合并矩阵乘法拆分结果中的各个块
            gm.graph.call_function(operator.getitem, (merge_mm_split, out), {})
            for out in range(len(lhs))
        ]

        # 替换所有原始未合并矩阵乘法节点的使用位置为合并后的拆分块
        for old, new in zip(mms, merge_mm_res):
            old.replace_all_uses_with(new)
            gm.graph.erase_node(old)

        # 上述新节点都是按顺序插入的，因此需要对图进行拓扑排序，以确保所有定义在使用之前
        legalize_graph(gm)

    # 重新编译图形
    gm.recompile()
    # 对图进行 lint 检查
    gm.graph.lint()
    # 返回更新后的图管理器
    return gm
```