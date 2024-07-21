# `.\pytorch\torch\distributed\_spmd\partial_lower.py`

```py
# 这个文件从 Meta 内部存储库中复制而来，与内部版本不同步。一旦内部版本完全成熟，我们应该再次上游并停用内部版本。@yifuwang

import logging  # 导入日志记录模块
import operator  # 导入运算符模块
from typing import Callable, List, Optional, Set, Tuple  # 导入类型提示模块

import torch  # 导入 PyTorch 库
from functorch import make_fx  # 导入 functorch 库中的 make_fx 函数
from torch._inductor.compile_fx import compile_fx_inner  # 导入 torch._inductor.compile_fx 模块中的 compile_fx_inner 函数
from torch._inductor.decomposition import select_decomp_table  # 导入 torch._inductor.decomposition 模块中的 select_decomp_table 函数

MIN_ATEN_OPS_TO_LOWER = 10  # 定义最小的 ATen 操作次数以进行降低处理

logger: logging.Logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


def _create_subgraph_module(
    inputs: List[torch.fx.Node], body: List[torch.fx.Node], outputs: List[torch.fx.Node]
) -> torch.fx.GraphModule:
    subgraph: torch.fx.Graph = torch.fx.Graph()  # 创建一个新的 PyTorch FX 图对象
    node_to_subgraph_node = {}  # 创建一个空字典，用于映射节点到子图节点的关系
    for idx, inp in enumerate(inputs):
        subgraph_inp = subgraph.placeholder(name=f"arg_{idx}")  # 在子图中创建输入占位符节点
        subgraph_inp.meta = inp.meta  # 将输入节点的元数据复制到子图输入节点
        node_to_subgraph_node[inp] = subgraph_inp  # 将输入节点映射到子图输入节点的字典中

    for node in body:
        subgraph_node = subgraph.node_copy(
            node, arg_transform=lambda x: node_to_subgraph_node[x]
        )  # 复制主体中的节点到子图中，通过映射将其连接到正确的输入节点
        node_to_subgraph_node[node] = subgraph_node  # 更新节点到子图节点的映射关系

    subgraph.output(result=tuple(node_to_subgraph_node[x] for x in outputs))  # 将子图的输出设置为指定的输出节点
    subgraph.eliminate_dead_code()  # 清除子图中的死代码节点
    subgraph.lint()  # 对子图进行静态检查
    return torch.fx.GraphModule(root={}, graph=subgraph)  # 创建并返回一个基于子图的 FX 模块


def _is_container_node(node: torch.fx.Node) -> bool:
    if any(user.target == operator.getitem for user in node.users):
        assert all(user.target == operator.getitem for user in node.users), (
            "Malformed graph: a container node is used as input for non-getitem nodes."
            "\nNode: {fmt_node}\nUsers: {fmt_users}".format(
                fmt_node=node.format_node(),
                fmt_users="\n".join(u.format_node() for u in node.users),
            )
        )
        return True  # 如果节点被用作非 getitem 节点的输入，返回 True
    return False  # 否则返回 False


def _lower_subgraph_nodes(
    gm: torch.fx.GraphModule,
    subgraph_name: str,
    subgraph_nodes: List[torch.fx.Node],
    dumper: Callable[[str], str],
) -> None:
    prologue: List[torch.fx.Node] = []  # 创建一个空的 prologue 列表，用于存储子图的前言节点
    inputs: List[torch.fx.Node] = []  # 创建一个空的 inputs 列表，用于存储子图的输入节点
    body: List[torch.fx.Node] = []  # 创建一个空的 body 列表，用于存储子图的主体节点
    visible: Set[torch.fx.Node] = set()  # 创建一个空的 visible 集合，用于跟踪已处理的节点

    # Inductor 要求所有的图输入都是张量。当作为子图输入添加一个容器节点时，
    # 将其后代 getitem 节点添加到子图的 prologue 中，并将其叶子 getitem 节点添加到子图输入中。
    def add_input(arg: torch.fx.Node) -> None:
        stack = [arg]
        while len(stack) != 0:
            node = stack.pop()
            if _is_container_node(node):
                # 只在 subgraph_nodes 中准备前导节点
                prologue.extend(user for user in node.users if user in subgraph_nodes)
                stack.extend(node.users)
            else:
                if node not in visible:
                    inputs.append(node)
                    visible.add(node)
    for node in subgraph_nodes:
        if node.op == "get_attr":
            # 将 get_attr 提前以避免复制属性到子图模块中。
            # 将节点添加到输入列表，并标记为可见。
            inputs.append(node)
            visible.add(node)
            continue

        for arg in node.all_input_nodes:
            if arg not in visible:
                add_input(arg)

        if node not in prologue:
            # 将节点添加到主体列表，并标记为可见。
            body.append(node)
            visible.add(node)

    outputs: List[torch.fx.Node] = []

    # Inductor 要求所有图的输出为张量。当将容器节点作为子图输出时，
    # 将其后代的 getitem 节点添加到子图主体，并将其叶子的 getitem 节点添加到子图输出。
    def add_output(output: torch.fx.Node) -> None:
        stack = [output]
        while len(stack) != 0:
            node = stack.pop()
            if _is_container_node(node):
                # 将容器节点的用户节点扩展到主体列表。
                body.extend(node.users)
                stack.extend(node.users)
            elif not all(user in visible for user in node.users):
                # 如果节点的用户节点不全在可见节点中，则添加节点到输出列表。
                if node not in outputs:
                    outputs.append(node)

    # 对主体列表中的每个节点进行处理，确保节点的所有用户节点都在可见节点中。
    for node in body:
        if not all(user in visible for user in node.users):
            add_output(node)

    # 断言输入列表中的节点数量与其集合的唯一性相符。
    assert len(inputs) == len(set(inputs))
    # 断言输出列表中的节点数量与其集合的唯一性相符。
    assert len(outputs) == len(set(outputs))

    # 创建子图模块并设置可读的标签。
    subgraph_module = _create_subgraph_module(inputs, body, outputs)
    readable_tag = dumper(str(subgraph_module.graph))
    # 将子图模块作为属性设置到主图模块中。
    setattr(gm, subgraph_name, _InductorModule(subgraph_module))

    # 插入点为子图节点列表中的最后一个节点的下一个节点。
    insertion_point = subgraph_nodes[-1].next
    # 将 prologue 列表中的节点前置到插入点之前。
    for node in prologue:
        insertion_point.prepend(node)

    # 在插入点之前插入子图调用节点。
    with gm.graph.inserting_before(insertion_point):
        subgraph_call = gm.graph.create_node(
            op="call_module",
            target=subgraph_name,
            args=tuple(inputs),
            kwargs={"tag": readable_tag},
        )
        # 用子图输出替换父图节点。
        for idx, output in enumerate(outputs):
            new_output = gm.graph.create_node(
                op="call_function",
                target=operator.getitem,
                args=(subgraph_call, idx),
            )
            new_output.meta = output.meta
            output.replace_all_uses_with(new_output)

    # 从父图中删除降低的节点。
    for node in reversed(body + outputs):
        if len(node.users) == 0:
            gm.graph.erase_node(node)
class _InductorModule(torch.nn.Module):
    def __init__(self, gm: torch.fx.GraphModule) -> None:
        super().__init__()
        self.gm = gm  # 初始化函数，将传入的图模块对象保存到实例变量中
        self.compiled: Optional[
            Callable[[List[torch.Tensor]], List[torch.Tensor]]
        ] = None  # 初始化函数，声明一个可选类型的变量，用于存储编译后的函数

    def forward(self, *args: torch.Tensor, tag: str) -> List[torch.Tensor]:
        if self.compiled is None:
            inductor_decompositions = select_decomp_table()
            # TODO: figure out why turning on cudagraphs cause exceptions.
            # 如果未编译，则选择分解表，创建 FX 图，记录日志
            decomp_gm = make_fx(self.gm, decomposition_table=inductor_decompositions)(
                *args
            )
            logger.info("Lowering subgraph (%s) to Inductor...", tag)
            # 编译子图为 Inductor 代码，设置编译后的函数
            self.compiled = compile_fx_inner(
                decomp_gm,
                list(args),
                cudagraphs=False,
            )
            logger.info("Completed lowering subgraph (%s) to Inductor", tag)
        with torch.profiler.record_function(tag):
            assert self.compiled is not None
            # 调用编译后的函数并返回结果
            return self.compiled(list(args))


def _is_inductor_compatible(node: torch.fx.Node) -> Tuple[bool, str]:
    # `has_tag` is not supported yet
    # if has_tag(node, "non_lowerable"):

    if node.target in (
        torch.ops.aten._fused_adam_.default,
        torch.ops.aten._fused_adam.default,
        torch.ops.aten._foreach_add_.Scalar,
        torch.ops.aten._foreach_add.Scalar,
    ):
        return False, "fused adam is not supported yet"

    # TODO(yifu): apparently having a meta kernel is not a necessary
    # condition for Inductor compatiblity. We should refine the check.
    # Sneaking this one in for now to support comm_fusion_with_cat.
    if node.target == torch.ops.aten.flatten.using_ints:
        return True, ""

    if isinstance(node.target, torch._ops.OpOverload):
        if not node.target.has_kernel_for_dispatch_key(torch._C.DispatchKey.Meta):
            return False, f"{node.target} doesn't have a meta kernel registered"
    return True, ""


def _subgraph_predicate(nodes: List[torch.fx.Node]) -> bool:
    num_aten_ops = len([n for n in nodes if str(n.target).startswith("aten.")])
    return num_aten_ops >= MIN_ATEN_OPS_TO_LOWER


def partial_lower(
    gm: torch.fx.GraphModule,
    node_predicate: Callable[[torch.fx.Node], bool] = lambda x: True,
    subgraph_predicate: Callable[[List[torch.fx.Node]], bool] = lambda x: True,
    dumper: Callable[[str], str] = lambda x: "subgraph",
) -> torch.fx.GraphModule:
    """
    Lower Inductor compatible portions of the graph module to Inductor.
    """
    # 函数：将图模块的兼容 Inductor 部分降低为 Inductor
    Args:
        node_predicate: 用于确定是否考虑节点进行降级的用户谓词。
        subgraph_predicate: 用于确定是否考虑候选节点列表进行降级的用户谓词。
        dumper: 用于将子图转储供人查看的回调函数。例如，可以是将数据写入磁盘或 blob 存储并返回路径/句柄的函数。每个子图的路径/句柄将在父图中的子图调用节点中以及子图的性能分析块的标签中提供。

    """
    nodes_per_subgraph: List[List[torch.fx.Node]] = [[]]
    # 初始化一个空的节点列表用于存储子图的节点

    ptr = next(iter(gm.graph.nodes))
    # 获取图中第一个节点作为指针（当前处理节点）

    def _node_predicate(node: torch.fx.Node) -> Tuple[bool, str]:
        # 内部函数：用于判断节点是否应该降级
        should_lower, reason = _is_inductor_compatible(node)
        # 检查节点是否兼容降级
        if not should_lower:
            return should_lower, reason
        # 如果不需要降级，则返回不需要降级和原因
        if not node_predicate(node):
            return False, "user predicate"
        # 如果用户定义的谓词不满足，则返回不需要降级和相应的原因
        return True, ""
        # 如果需要降级且通过所有检查，则返回需要降级和空原因字符串

    while ptr.op != "output":
        # 循环直到指针指向输出节点
        if ptr.op == "placeholder":
            ptr = ptr.next
            continue
        # 如果当前节点是占位符，则指针移动到下一个节点并继续循环
        should_lower, reason = _node_predicate(ptr)
        # 检查当前节点是否需要降级
        if should_lower:
            nodes_per_subgraph[-1].append(ptr)
            # 如果需要降级，则将当前节点添加到最后一个子图的节点列表中
        else:
            if len(nodes_per_subgraph[-1]) > 0:
                logger.warning(
                    "partial_lower: graph break at %s. Reason: %s", str(ptr), reason
                )
            # 如果当前节点不需要降级且之前的子图列表不为空，则记录警告日志
            nodes_per_subgraph.append([])
            # 创建一个新的空子图列表

        ptr = ptr.next
        # 将指针移动到下一个节点继续处理

    nodes_per_subgraph = [
        nodes
        for nodes in nodes_per_subgraph
        if subgraph_predicate(nodes) and _subgraph_predicate(nodes)
    ]
    # 过滤出满足子图谓词条件的子图节点列表

    for idx, subgraph_nodes in enumerate(nodes_per_subgraph):
        subgraph_name = f"subgraph_{idx}"
        _lower_subgraph_nodes(gm, subgraph_name, subgraph_nodes, dumper)
        # 遍历每个子图节点列表，将其降级处理并命名

    gm.graph.lint()
    # 对修改后的图进行代码检查
    gm.recompile()
    # 重新编译修改后的图
    return gm
    # 返回修改后的图模型
```