# `.\pytorch\torch\_inductor\fx_passes\micro_pipeline_tp.py`

```
# mypy: allow-untyped-defs
# 导入运算符模块，用于操作符号
import operator
# 导入 dataclass 类型，用于创建数据类
from dataclasses import dataclass
# 导入类型相关模块：List（列表）、Set（集合）、Tuple（元组）、Union（联合类型）
from typing import cast, List, Set, Tuple, Union

# 导入 PyTorch 深度学习库
import torch
# 导入 inductor_prims 模块的所有内容
from .. import inductor_prims
# 从 pattern_matcher 模块中导入多个符号
from ..pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    ListOf,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
)

# 将 torch.ops.aten 赋值给 aten
aten = torch.ops.aten
# 创建 PatternMatcherPass 类的实例对象 patterns
patterns = PatternMatcherPass()


def _is_backward(graph: torch.fx.Graph) -> bool:
    # 初始化占位符列表
    placeholders = []
    # 遍历图中的节点
    for node in graph.nodes:
        # 如果节点的操作不是 "placeholder"，则跳出循环
        if node.op != "placeholder":
            break
        # 将符合条件的节点添加到占位符列表中
        placeholders.append(node)
    # 返回是否所有占位符的名称都以 "primal" 开头的布尔值
    return not all(node.name.startswith("primal") for node in placeholders)


def _compute_mm_arithmetic_intensity(M: int, N: int, K: int) -> float:
    # 计算矩阵乘法的算术强度
    return M * N * K / (M * K + N * K + M * N)


def _filter_nodes_by_target(nodes: List[torch.fx.Node], target) -> List[torch.fx.Node]:
    # 根据目标过滤节点列表中的节点
    return [x for x in nodes if x.target == target]


def _find_ancestors(node: torch.fx.Node) -> Set[torch.fx.Node]:
    # 查找给定节点的所有祖先节点集合
    ancestors = set()
    ancestors.add(node)
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.all_input_nodes:
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    # 返回不是占位符操作的祖先节点集合
    return {node for node in ancestors if node.op != "placeholder"}


def _can_schedule_y_before_x(
    x: torch.fx.Node, y: torch.fx.Node
) -> Tuple[bool, Set[torch.fx.Node]]:
    """
    Check if y can be reordered before x and return the ancestors of y
    (inclusive).
    """
    # 获取节点 y 的所有祖先节点集合
    y_ancestors = _find_ancestors(y)
    # 判断节点 x 是否在节点 y 的祖先节点集合中
    if x in y_ancestors:
        # 如果是，则返回 False 和节点 y 的祖先节点集合
        return False, y_ancestors

    # 否则，返回 True 和节点 y 的祖先节点集合
    return True, y_ancestors


@dataclass
class _2DMatmul:
    # 定义数据类 _2DMatmul，包含节点和 B_node 节点
    node: torch.fx.Node
    B_node: torch.fx.Node
    B_node_ancestors: Set[torch.fx.Node]

    def replace_with(self, new_node: torch.fx.Node) -> None:
        """
        Replace the matmul with the new node.
        """
        # 替换 matmul 节点为新节点
        self.node.replace_all_uses_with(new_node)


@dataclass
class _NDMatmul:
    # 定义数据类 _NDMatmul，包含节点列表 nodes 和 B_node 节点
    nodes: List[torch.fx.Node]
    B_node: torch.fx.Node
    B_node_ancestors: Set[torch.fx.Node]
    # 获取新节点所在的图形
    graph = new_node.graph
    # 断言当前节点列表长度为3，分别对应于reshape -> mm -> reshape的顺序
    assert len(self.nodes) == 3
    # 获取mm节点，即第二个节点
    mm_node = self.nodes[1]
    # 获取输出reshape节点，即第三个节点
    output_reshape_node = self.nodes[2]

    # 断言mm节点的目标是torch的矩阵乘法函数
    assert mm_node.target == aten.mm.default
    # 断言输出reshape节点的目标是torch的reshape函数

    assert output_reshape_node.target == aten.reshape.default

    # 替换输出reshape节点的所有使用为新节点
    output_reshape_node.replace_all_uses_with(new_node)

    # 如果mm节点有多个使用者
    if len(mm_node.users) > 1:
        # 在新节点之后插入一个新的reshape节点
        with graph.inserting_after(new_node):
            # 创建一个新的mm节点，使用原始mm节点的形状
            new_mm_node = graph.call_function(
                aten.reshape.default,
                args=(new_node, list(mm_node.meta["val"].shape)),
            )
        # 替换所有原始mm节点的使用为新的mm节点
        mm_node.replace_all_uses_with(new_mm_node)
def _find_consumer_matmuls(node: torch.fx.Node) -> List[Union[_2DMatmul, _NDMatmul]]:
    """
    Find the matmuls that use `node` as the lhs argument.
    This function effective normalizes 2D and ND matmuls.
    """
    matmuls: List[Union[_2DMatmul, _NDMatmul]] = []  # 初始化一个空列表，用于存储找到的 matmul 对象

    for user in node.users:  # 遍历所有使用 `node` 作为左操作数的节点
        # ND matmuls
        if user.target == aten.reshape.default:  # 如果使用 `node` 的节点目标是 reshape 操作
            for mm_node in user.users:  # 遍历 reshape 操作的所有使用者
                if mm_node.target != aten.mm.default:  # 如果使用者不是 mm 操作，则跳过
                    continue

                B_node = cast(torch.fx.Node, mm_node.args[1])  # 获取 mm 操作的第二个参数节点 B_node
                can_schedule, B_node_ancestors = _can_schedule_y_before_x(user, B_node)  # 检查是否可以在 user 之前调度 B_node
                if not can_schedule:
                    continue

                for reshape_node in mm_node.users:  # 遍历 mm 操作的所有使用者
                    if reshape_node.target != aten.reshape.default:  # 如果使用者不是 reshape 操作，则跳过
                        continue

                    # 创建 matmul 的输出形状
                    matmul_out_shape = torch.Size(
                        [
                            *node.meta["val"].shape[:-1],  # 使用 `node` 的形状，除了最后一个维度之外的所有维度
                            B_node.meta["val"].shape[-1],  # B_node 的最后一个维度
                        ]
                    )
                    if reshape_node.meta["val"].shape != matmul_out_shape:  # 如果 reshape 操作的形状不等于 matmul 的输出形状，则跳过
                        continue

                    matmuls.append(  # 将找到的 _NDMatmul 对象添加到 matmuls 列表中
                        _NDMatmul(
                            nodes=[user, mm_node, reshape_node],  # 使用 reshape 操作、mm 操作和 user 节点
                            B_node=B_node,  # B_node 参数
                            B_node_ancestors=B_node_ancestors,  # B_node 的祖先节点
                        )
                    )
        # 2D matmuls
        elif user.target == aten.mm.default:  # 如果使用 `node` 的节点目标是 mm 操作
            B_node = cast(torch.fx.Node, user.args[1])  # 获取 mm 操作的第二个参数节点 B_node
            can_schedule, B_node_ancestors = _can_schedule_y_before_x(user, B_node)  # 检查是否可以在 user 之前调度 B_node
            if not can_schedule:
                continue

            matmuls.append(  # 将找到的 _2DMatmul 对象添加到 matmuls 列表中
                _2DMatmul(
                    node=user,  # 使用 `node` 的节点
                    B_node=B_node,  # B_node 参数
                    B_node_ancestors=B_node_ancestors,  # B_node 的祖先节点
                ),
            )
    return matmuls


def _find_all_gather_node_from_match(match) -> Tuple[torch.fx.Node, torch.fx.Node]:
    """
    Processes match for ZeroDimAllGather and NonZeroDimAllGather. Returns the
    all-gather node (all_gather_into_tensor.default) and the all-gather result
    node (wait_tensor.default for gather_dim == 0 and aten.cat.default for
    gather_dim == 1). This function effectively normalizes zero-dim and
    non-zero-dim all_gather_tensor.
    """
    # gather_dim == 0
    if len(match.nodes) == 2:  # 如果 match 的节点数量为 2
        return match.nodes[0], match.nodes[1]  # 返回 match 的两个节点
    # gather_dim == 1
    ag_node = _filter_nodes_by_target(
        match.nodes,
        torch.ops._c10d_functional.all_gather_into_tensor.default,  # 过滤得到 all_gather_into_tensor.default 的节点
    )[0]
    ag_res_node = _filter_nodes_by_target(
        match.nodes,
        aten.cat.default,  # 过滤得到 aten.cat.default 的节点
    )[0]
    shard_node = ag_node.args[0]  # 获取 all_gather_into_tensor.default 的第一个参数节点
    return ag_node, ag_res_node  # 返回 all_gather_into_tensor.default 节点和 aten.cat.default 节点


def fuse_all_gather_matmul_zero_dim(match, shard, group_name):
    fuse_all_gather_matmul(match, shard, 0, group_name)  # 调用 fuse_all_gather_matmul 函数，将 gather_dim 设置为 0
def fuse_all_gather_matmul(match, shard, gather_dim, group_name):
    """
    将以下模式进行融合：

        A = all_gather_tensor(A_shard, gather_dim, group_name)
        C_0 = torch.matmul(A, B_0)
        C_1 = torch.matmul(A, B_1)
        C_2 = torch.matmul(A, B_2)
        ...

    融合成以下形式：

        A, Cs = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, [B_0, B_1, B_2, ...], gather_dim, group_name,
        )
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return

    # 导入相关模块和函数
    c10d = torch.ops._c10d_functional
    from torch.distributed._symmetric_memory import (
        is_symm_mem_enabled_for_group,
        restride_A_shard_for_fused_all_gather_matmul,
    )

    # 如果 gather_dim 大于等于 shard.meta["val"] 的最后一维索引，不支持在 K 维度上分解 matmul
    if gather_dim >= len(shard.meta["val"].shape) - 1:
        return

    # 如果指定的分组不支持对称内存，则返回
    if not is_symm_mem_enabled_for_group(group_name):
        return

    # 标准化零维和非零维的 all_gather_tensor
    ag_node, ag_res_node = _find_all_gather_node_from_match(match)

    # 查找可融合的消费者 matmul
    matmuls = _find_consumer_matmuls(ag_res_node)
    if len(matmuls) == 0:
        return

    # 获取 shard_node 对象，并提取 B_nodes 列表
    shard_node = cast(torch.fx.Node, ag_node.args[0])
    B_nodes = [matmul.B_node for matmul in matmuls]

    # 在 all_gather_tensor 前插入新的图节点来融合
    graph = ag_node.graph
    with graph.inserting_before(ag_node):
        # 如果 shard_node 的 meta 中存在 "val"
        if "val" in shard_node.meta:
            # 对 A_shard 进行重新调整以用于融合的 all_gather_matmul
            restrided = restride_A_shard_for_fused_all_gather_matmul(
                shard_node.meta["val"],
                gather_dim,
            )
            # 创建一个新的节点来调用 inductor_prims.force_stride_order 函数
            shard_node = graph.call_function(
                inductor_prims.force_stride_order,
                args=(shard_node, restrided.stride()),
            )

        # 创建调用 torch.ops.symm_mem.fused_all_gather_matmul.default 的新节点
        fused_node = graph.call_function(
            torch.ops.symm_mem.fused_all_gather_matmul.default,
            args=(shard_node, B_nodes, gather_dim, group_name),
        )
        # 创建获取融合结果 A 的新节点
        new_ag_node = graph.call_function(
            operator.getitem,
            args=(fused_node, 0),
        )
        # 创建获取融合结果 Cs 的新节点
        new_out_nodes = graph.call_function(
            operator.getitem,
            args=(fused_node, 1),
        )
        # 替换原 matmul 结果节点为新的输出节点
        for idx, matmul in enumerate(matmuls):
            new_out_node = graph.call_function(
                operator.getitem,
                args=(new_out_nodes, idx),
            )
            matmul.replace_with(new_out_node)
        # 替换所有使用 ag_res_node 的地方为新的 A 结果节点
        ag_res_node.replace_all_uses_with(new_ag_node)

    # 提升在 ag_res_node 和融合节点上方的拓扑顺序排序的 B 的祖先节点
    order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes_to_raise = sorted(
        {x for matmul in matmuls for x in matmul.B_node_ancestors},
        key=lambda x: order[x],
    )
    # 遍历需要提升的节点列表
    for node in nodes_to_raise:
        # 检查节点在顺序字典中的顺序是否大于融合节点的顺序
        if order[node] > order[fused_node]:
            # 如果是，则在融合节点之前插入当前节点
            fused_node.prepend(node)

    # 清除图中的死代码
    graph.eliminate_dead_code()
    # 返回空，函数执行完毕
    return
def fuse_matmul_reduce_scatter_zero_dim(match, rs_input, reduce_op, group_name):
    # 调用 fuse_matmul_reduce_scatter 函数，将 scatter_dim 参数设为 0
    fuse_matmul_reduce_scatter(match, rs_input, reduce_op, 0, group_name)


def fuse_matmul_reduce_scatter(match, rs_input, reduce_op, scatter_dim, group_name):
    """
    合并模式：

        reduce_scatter_tensor(A @ B, scatter_dim, group_name)

    到

        torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, scatter_dim, group_name,
        )
    """
    # 检查是否支持分布式计算和 NCCL
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return

    # 导入相关模块和函数
    c10d = torch.ops._c10d_functional
    from torch.distributed._symmetric_memory import (
        is_symm_mem_enabled_for_group,
        restride_A_for_fused_matmul_reduce_scatter,
    )

    # 如果指定的 group_name 不支持对称内存，则退出函数
    if not is_symm_mem_enabled_for_group(group_name):
        return

    # 当前 fused_matmul_reduce_scatter 函数不返回矩阵乘积结果，
    # 所以如果矩阵乘积结果被多个用户使用，则无法应用此优化。
    if len(rs_input.users) != 1:
        return

    # 检查是否为二维矩阵乘积
    if rs_input.target == aten.mm.default:
        A_node, B_node = rs_input.args[0], rs_input.args[1]
    # 检查是否为多维矩阵乘积
    elif rs_input.target == aten.reshape.default:
        mm_node = rs_input.args[0]
        # 如果不是矩阵乘积或者有多个用户使用结果，则退出函数
        if mm_node.target != aten.mm.default or len(mm_node.users) != 1:
            return
        A_node, B_node = mm_node.args[0], mm_node.args[1]
        # 检查 A_node 是否为 reshape 操作
        if A_node.target != aten.reshape.default:
            return
        A_node = A_node.args[0]
    # 如果不是矩阵乘积，则退出函数
    else:
        return

    # 获取 rs_res_node，并检查是否可以在 B_node 之前调度 rs_res_node
    rs_res_node = _filter_nodes_by_target(match.nodes, c10d.wait_tensor.default)[0]
    if not _can_schedule_y_before_x(rs_res_node, B_node):
        return

    # 获取图表
    graph = rs_res_node.graph

    # 在 rs_res_node 前插入节点
    with graph.inserting_before(rs_res_node):
        # 如果 A_node.meta 中存在 "val" 键，则重新调整 A_node 的步幅顺序
        if "val" in A_node.meta:
            val = A_node.meta["val"]
            restrided = restride_A_for_fused_matmul_reduce_scatter(
                A_node.meta["val"],
                scatter_dim,
            )
            A_node = graph.call_function(
                inductor_prims.force_stride_order,
                args=(A_node, restrided.stride()),
            )

        # 创建融合节点
        fused_node = graph.call_function(
            torch.ops.symm_mem.fused_matmul_reduce_scatter.default,
            args=(A_node, B_node, reduce_op, scatter_dim, group_name),
        )
        # 将 rs_res_node 的所有使用替换为 fused_node
        rs_res_node.replace_all_uses_with(fused_node)

    # 获取图中节点的顺序
    order = {node: idx for idx, node in enumerate(graph.nodes)}
    # 对 B_node 的所有祖先节点进行排序
    nodes_to_raise = sorted(
        _find_ancestors(B_node),
        key=lambda x: order[x],
    )
    # 将所有节点提升到 fused_node 之前
    for node in nodes_to_raise:
        if order[node] > order[fused_node]:
            fused_node.prepend(node)

    # 消除死代码
    graph.eliminate_dead_code()


def _register_passes():
    # 检查是否支持分布式计算和 NCCL，若不支持则退出函数
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return
    c10d = torch.ops._c10d_functional
    # 导入 torch C10D 模块中的 _c10d_functional 函数

    # 定义与 gather_dim == 0 时匹配的函数模式
    ZeroDimAllGather = CallFunction(
        c10d.wait_tensor.default,  # 调用 c10d.wait_tensor.default 函数
        CallFunction(
            c10d.all_gather_into_tensor.default,  # 调用 c10d.all_gather_into_tensor.default 函数
            KeywordArg("shard"),  # 指定关键字参数 "shard"
            Ignored(),  # 忽略的参数
            KeywordArg("group_name"),  # 指定关键字参数 "group_name"
        ),
    )

    # 定义与 gather_dim > 0 时匹配的函数模式
    # 注意：如果 funcol.all_gather_tensor 发生变化，可能需要更新此模式
    NonZeroDimAllGather = CallFunction(
        aten.cat.default,  # 调用 aten.cat.default 函数
        ListOf(
            CallFunction(
                operator.getitem,  # 调用 operator.getitem 函数
                CallFunction(
                    aten.split.Tensor,  # 调用 aten.split.Tensor 函数
                    CallFunction(
                        c10d.wait_tensor.default,  # 调用 c10d.wait_tensor.default 函数
                        CallFunction(
                            c10d.all_gather_into_tensor.default,  # 调用 c10d.all_gather_into_tensor.default 函数
                            KeywordArg("shard"),  # 指定关键字参数 "shard"
                            Ignored(),  # 忽略的参数
                            KeywordArg("group_name"),  # 指定关键字参数 "group_name"
                        ),
                    ),
                    Ignored(),  # 忽略的参数
                    _users=MULTIPLE,  # _users 参数为 MULTIPLE
                ),
                Ignored(),  # 忽略的参数
            ),
        ),
        KeywordArg("gather_dim"),  # 指定关键字参数 "gather_dim"
        _users=MULTIPLE,  # _users 参数为 MULTIPLE
    )

    # 注册 ZeroDimAllGather 模式对应的图模式函数
    register_graph_pattern(
        ZeroDimAllGather,
        pass_dict=patterns,  # 使用 patterns 字典进行传递
    )(fuse_all_gather_matmul_zero_dim)

    # 注册 NonZeroDimAllGather 模式对应的图模式函数
    register_graph_pattern(
        NonZeroDimAllGather,
        pass_dict=patterns,  # 使用 patterns 字典进行传递
    )(fuse_all_gather_matmul)

    # 定义与 scatter_dim == 0 时匹配的函数模式
    ZeroDimReduceScatter = CallFunction(
        c10d.wait_tensor.default,  # 调用 c10d.wait_tensor.default 函数
        CallFunction(
            c10d.reduce_scatter_tensor.default,  # 调用 c10d.reduce_scatter_tensor.default 函数
            KeywordArg("rs_input"),  # 指定关键字参数 "rs_input"
            KeywordArg("reduce_op"),  # 指定关键字参数 "reduce_op"
            Ignored(),  # 忽略的参数
            KeywordArg("group_name"),  # 指定关键字参数 "group_name"
        ),
    )

    # 定义与 scatter_dim > 0 时匹配的函数模式
    # 注意：如果 funcol.reduce_scatter_tensor 发生变化，可能需要更新此模式
    NonZeroDimReduceScatter = CallFunction(
        c10d.wait_tensor.default,  # 调用 c10d.wait_tensor.default 函数
        CallFunction(
            c10d.reduce_scatter_tensor.default,  # 调用 c10d.reduce_scatter_tensor.default 函数
            CallFunction(
                aten.cat.default,  # 调用 aten.cat.default 函数
                ListOf(
                    CallFunction(
                        operator.getitem,  # 调用 operator.getitem 函数
                        CallFunction(
                            aten.split.Tensor,  # 调用 aten.split.Tensor 函数
                            KeywordArg("rs_input"),  # 指定关键字参数 "rs_input"
                            Ignored(),  # 忽略的参数
                            KeywordArg("scatter_dim"),  # 指定关键字参数 "scatter_dim"
                            _users=MULTIPLE,  # _users 参数为 MULTIPLE
                        ),
                        Ignored(),  # 忽略的参数
                    )
                ),
            ),
            KeywordArg("reduce_op"),  # 指定关键字参数 "reduce_op"
            Ignored(),  # 忽略的参数
            KeywordArg("group_name"),  # 指定关键字参数 "group_name"
        ),
    )

    # 注册 ZeroDimReduceScatter 模式对应的图模式函数
    register_graph_pattern(
        ZeroDimReduceScatter,
        pass_dict=patterns,  # 使用 patterns 字典进行传递
    )(fuse_matmul_reduce_scatter_zero_dim)
    # 注册图模式，将 NonZeroDimReduceScatter 图模式注册到 fuse_matmul_reduce_scatter 函数上
    register_graph_pattern(
        NonZeroDimReduceScatter,
        pass_dict=patterns,  # 使用 patterns 参数传递给图模式注册函数
    )(fuse_matmul_reduce_scatter)
# 调用一个函数或方法来注册一些处理步骤或操作
_register_passes()
```