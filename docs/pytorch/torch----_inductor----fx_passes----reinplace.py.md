# `.\pytorch\torch\_inductor\fx_passes\reinplace.py`

```
# mypy: allow-untyped-defs
# 引入 itertools 模块，提供迭代工具函数
import itertools
# 引入 operator 模块，提供操作符函数
import operator
# 从 collections 模块中引入 defaultdict 类型，提供默认值的字典
from collections import defaultdict
# 从 dataclasses 模块中引入 dataclass 装饰器，用于创建不可变数据类
from dataclasses import dataclass
# 从 typing 模块中引入 Any, Callable, Dict, List, Tuple 类型
from typing import Any, Callable, Dict, List, Tuple

# 引入 torch 库
import torch
# 从 torch._higher_order_ops.triton_kernel_wrap 模块中引入 triton_kernel_wrapper_functional 函数
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional
# 从 torch._inductor 模块中引入 inductor_prims 模块
from torch._inductor import inductor_prims
# 从 torch._inductor.fx_utils 模块中引入 get_node_storage, is_node_realized 函数
from torch._inductor.fx_utils import get_node_storage, is_node_realized
# 从 torch._inductor.lowering 模块中引入 inplaceable_foreach_ops_lowerings 模块
from torch._inductor.lowering import (
    inplaceable_foreach_ops as inplaceable_foreach_ops_lowerings,
)
# 从 torch._inductor.virtualized 模块中引入 V 类
from torch._inductor.virtualized import V
# 从 torch.fx.immutable_collections 模块中引入 immutable_dict 函数
from torch.fx.immutable_collections import immutable_dict
# 从 torch.fx.passes.reinplace 模块中引入 _is_view_op 函数
from torch.fx.passes.reinplace import _is_view_op
# 从 torch.utils 模块中引入 _pytree 模块
from torch.utils import _pytree as pytree

# 引入 torch.ops.aten 模块
aten = torch.ops.aten

# 定义不可变数据类 InplaceableOp
@dataclass(frozen=True)
class InplaceableOp:
    inplace_op: Callable[..., Any]  # 原地操作函数
    mutated_arg: int  # 发生变异的参数位置
    extra_check: Callable[[torch.fx.Node], bool] = lambda node: True  # 额外检查函数，默认返回 True

# 定义字典_SCATTER_OP_TO_VIEW，将 scatter 操作映射为对应的 view 操作
_SCATTER_OP_TO_VIEW = {
    torch.ops.aten.diagonal_scatter.default: torch.ops.aten.diagonal.default,
    torch.ops.aten.select_scatter.default: torch.ops.aten.select.int,
    torch.ops.aten.slice_scatter.default: torch.ops.aten.slice.Tensor,
    torch.ops.aten.as_strided_scatter.default: torch.ops.aten.as_strided.default,
}
# 定义字典_VIEW_OP_TO_SCATTER，将 view 操作映射为对应的 scatter 操作
_VIEW_OP_TO_SCATTER = {v: k for k, v in _SCATTER_OP_TO_VIEW.items()}

# 定义函数 graph_call_function，用于在图中调用函数
def graph_call_function(graph: torch.fx.Graph, fn, *args, **kwargs):
    # 将 args 和 kwargs 中的 torch.fx.Node 替换为对应的 meta["val"] 值
    fake_args, fake_kwargs = pytree.tree_map(
        lambda node: node.meta["val"] if isinstance(node, torch.fx.Node) else node,
        (args, kwargs),
    )
    # 在 fake 模式下调用函数 fn
    with V.fake_mode:
        fake_result = fn(*fake_args, **fake_kwargs)

    # 在图中调用函数，并将结果存储在节点的 meta 字典中
    node = graph.call_function(fn, args, kwargs)
    node.meta["val"] = fake_result
    return node

# 定义不可变数据类 ViewOp，表示视图操作
@dataclass
class ViewOp:
    target: torch._ops.OpOverload  # 目标操作
    args: Tuple[Any, ...]  # 参数元组
    kwargs: Dict[str, Any]  # 关键字参数字典

# 定义函数 _inplace_generalized_scatter，实现广义的原地 scatter 操作
def _inplace_generalized_scatter(
    inp: torch.Tensor, src: torch.Tensor, view_ops: List[ViewOp]
) -> torch.Tensor:
    tmp = inp
    # 遍历 view_ops 列表中的视图操作
    for view in view_ops:
        # 将 args 和 kwargs 中的 torch.fx.Node 替换为对应的 meta["val"] 值
        fake_args, fake_kwargs = pytree.tree_map(
            lambda node: node.meta["val"] if isinstance(node, torch.fx.Node) else node,
            (view.args, view.kwargs),
        )
        # 对 tmp 执行视图操作 view.target
        tmp = view.target(tmp, *fake_args, **fake_kwargs)
    # 尝试将 src 复制到 tmp 中
    try:
        tmp.copy_(src)
    except RuntimeError as e:
        # 抛出形状错误异常，指示无法将 src 的形状广播到 tmp 的形状
        raise RuntimeError(
            f"shape error in scatter op, can not broadcast {src.shape} to {tmp.shape}"
        ) from e
    return inp

# 定义函数 _generalized_scatter，实现广义的 scatter 操作
def _generalized_scatter(
    inp: torch.Tensor, src: torch.Tensor, view_ops: List[ViewOp]
) -> torch.Tensor:
    out = inp.clone()  # 克隆输入张量 inp 到 out
    return _inplace_generalized_scatter(out, src, view_ops)  # 执行原地 scatter 操作并返回结果张量

# 定义函数 _decompose_scatter_functional_helper，辅助分解 scatter 函数
def _decompose_scatter_functional_helper(
    graph: torch.fx.Graph,
    inp: torch.Tensor,
    src: torch.Tensor,
    view_ops: List[ViewOp],
) -> torch.fx.Node:
    view_op, view_ops_tail = view_ops[0], view_ops[1:]
    # 如果存在 view_ops_tail，执行以下操作
    if view_ops_tail:
        # 调用 graph_call_function 函数，执行视图操作目标函数
        view = graph_call_function(
            graph, view_op.target, inp, *view_op.args, **view_op.kwargs
        )
        # 使用视图函数辅助函数 _decompose_scatter_functional_helper 处理散射函数
        src = _decompose_scatter_functional_helper(graph, view, src, view_ops[1:])  # type: ignore[assignment]

    # 调用 graph_call_function 函数，执行视图操作目标函数的散射函数
    return graph_call_function(
        graph,
        _VIEW_OP_TO_SCATTER[view_op.target],
        inp,
        src,
        *view_op.args,
        **view_op.kwargs,
    )
def _decompose_scatter_functional(
    graph: torch.fx.Graph, node: torch.fx.Node
) -> torch.fx.Node:
    """
    Decompose _generalized_scatter to a sequence of view_scatter operations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    view = aten.slice(inp, 0, 0, 10)
    view_updated = aten.slice_scatter(view, src, 1, 10, -10)
    inp_updated = aten.slice_scatter(inp, view_updated, 0, 0, 10)
    """
    assert node.target is _generalized_scatter
    inp, src, view_ops = node.args
    return _decompose_scatter_functional_helper(graph, *node.args)  # type: ignore[arg-type]


def _decompose_scatter_mutating(
    graph: torch.fx.Graph, node: torch.fx.Node
) -> torch.fx.Node:
    """
    Decompose _generalized_scatter using mutations

    e.g. _generalized_scatter(inp, src, [(aten.slice, 0, 0, 10), (aten.slice, 1, 10, -10)])

    will become

    inp_updated = aten.clone(inp)
    slice1 = aten.slice(inp_updated, 0, 0, 10)
    slice2 = aten.slice(slice1, 1, 10, -10)
    slice2.copy_(src)
    """
    assert node.target in (_generalized_scatter, _inplace_generalized_scatter)
    inp, src, view_ops = node.args
    assert not node.kwargs

    if node.target is _generalized_scatter:
        inp = graph_call_function(graph, aten.clone, inp)

    tmp = inp
    for view in view_ops:  # type: ignore[union-attr]
        tmp = graph_call_function(graph, view.target, tmp, *view.args, **view.kwargs)  # type: ignore[union-attr]

    graph_call_function(graph, aten.copy_.default, tmp, src)
    return inp  # type: ignore[return-value]


# View ops whose view_scatter op is lowered into mutations anyway,
# so is never a pessimisation to decompose.
_ALWAYS_MUTATING_SCATTER_OPS = {
    aten.as_strided.default,
    aten.diagonal.default,
}


def scatter_always_uses_mutation(node: torch.fx.Node) -> bool:
    """
    Determine if scatter operation always uses mutation based decomposition

    This function checks if any view operation is in _ALWAYS_MUTATING_SCATTER_OPS.

    Args:
        node: The node representing the scatter operation in the FX graph.

    Returns:
        bool: True if any view operation uses mutation, False otherwise.
    """
    _, _, view_ops = node.args
    return any(view.target in _ALWAYS_MUTATING_SCATTER_OPS for view in view_ops)  # type: ignore[union-attr]


def should_reinplace_scatter(node: torch.fx.Node) -> bool:
    """
    Decide whether to use mutating or functional scatter decompositions

    This function makes a decision based on whether scatter_always_uses_mutation
    returns True, and if the input and output nodes are realized.

    Args:
        node: The node representing the scatter operation in the FX graph.

    Returns:
        bool: True if mutating scatter decomposition should be used, False if functional.
    """
    inp, src, view_ops = node.args

    # Mutating scatter ops unconditionally realize input and output
    if scatter_always_uses_mutation(node):
        return True

    if is_node_realized(inp) and is_node_realized(node):  # type: ignore[arg-type]
        return True

    # If the output is copied back into the input, this forces both to be
    # realized as the output is a user of the input
    if inp.op in ("placeholder", "get_attr") and any(
        user.target is aten.copy_.default and user.args[0] is inp for user in node.users
    ):
        return True
    # 否则，假设融合将使功能变体变得有利可图
    return False
def decompose_generalized_scatter(graph: torch.fx.Graph) -> None:
    """将_generalized_scatter替换为普通的aten操作"""
    # 遍历图中所有调用_generalized_scatter和_inplace_generalized_scatter的节点
    for node in itertools.chain(
        graph.find_nodes(op="call_function", target=_generalized_scatter),
        graph.find_nodes(op="call_function", target=_inplace_generalized_scatter),
    ):
        # 判断是否使用了_inplace_generalized_scatter或者scatter_always_uses_mutation函数
        use_mutation = (
            node.target is _inplace_generalized_scatter
            or scatter_always_uses_mutation(node)
        )

        # 在当前节点之前插入新节点
        with graph.inserting_before(node):
            # 根据是否使用mutation来选择不同的分解方法
            if use_mutation:
                new_node = _decompose_scatter_mutating(graph, node)
            else:
                new_node = _decompose_scatter_functional(graph, node)

        # 替换当前节点的所有使用为新节点
        node.replace_all_uses_with(new_node)
        # 删除当前节点
        graph.erase_node(node)


def canonicalize_view_scatter_ops(graph: torch.fx.Graph) -> None:
    """
    将视图散射操作规范化为通用形式，定义为:
      def scatter(inp, src, views):
        tmp = inp.clone()
        for view in views:
          tmp = view(tmp)
        tmp.copy_(src)

    还将形如以下的连续视图散射操作融合为一个更高效的形式:
        a = scatter(view2(self), src, [view1])
        b = scatter(self, a, [view2])
    可以重写为
        b = scatter(self, src, [view2, view1])
        a = view2(b)

    这样更高效，因为只进行一次散射操作，并且更容易替换，因为只有一个`self`的使用
    """

    node_to_view_base: Dict[torch.fx.Node, torch.fx.Node] = {}
    node_to_view_op: Dict[torch.fx.Node, List[ViewOp]] = defaultdict(list)

    def handle_views(node: torch.fx.Node):
        inp = node.args[0]
        # 将当前节点的输入作为视图基础节点
        node_to_view_base[node] = node_to_view_base.get(inp, inp)  # type: ignore[arg-type]
        # 将当前节点的视图操作添加到视图操作字典中
        node_to_view_op[node] = [
            *node_to_view_op[inp],  # type: ignore[index]
            ViewOp(
                node.target,  # type: ignore[arg-type]
                args=node.args[1:],
                kwargs=node.kwargs,
            ),
        ]
    def handle_view_scatter(node: torch.fx.Node):
        # 确保节点的参数数量至少为2
        assert len(node.args) >= 2
        # 提取输入和源节点
        inp, src = node.args[:2]

        # 创建视图操作对象，根据目标操作选择对应的视图操作
        scatter_view_op = ViewOp(
            _SCATTER_OP_TO_VIEW[node.target],
            args=node.args[2:],  # 提取除了前两个参数外的其他参数作为args
            kwargs=node.kwargs,  # 使用节点的关键字参数
        )

        def can_fuse():
            # 检查是否可以融合节点
            if src.target is not _generalized_scatter:  # 如果源节点的目标不是_generalized_scatter
                return False
            src_inp, src_src, src_scatter_view_op = src.args  # 提取源节点的参数

            # 获取输入节点和源节点的基本视图，检查它们是否一致，并且对应的视图操作是否可以串联
            inp_base = node_to_view_base.get(inp, inp)  # 获取节点到基本视图的映射，如果没有映射则使用节点本身
            src_base = node_to_view_base.get(src_inp, src_inp)
            return (
                inp_base is src_base
                and node_to_view_op[src_inp] == [*node_to_view_op[inp], scatter_view_op]
            )

        # 如果不能融合，则在节点之前插入新节点，并替换原节点的所有用途
        if not can_fuse():
            with graph.inserting_before(node):
                new_node = graph_call_function(
                    graph,
                    _generalized_scatter,
                    inp,
                    src,
                    [scatter_view_op],
                )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            return

        # 对于可以融合的情况，更新源节点的参数，并替换所有使用该节点的地方
        src_inp, src_src, src_scatter_view_op = src.args
        with graph.inserting_before(src):
            new_node = graph_call_function(
                graph,
                _generalized_scatter,
                inp,
                src_src,
                [scatter_view_op, *src_scatter_view_op],
            )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

            # 如果源节点有使用者，则创建新的视图调用，并替换所有使用该节点的地方
            if src.users:
                new_src = graph_call_function(
                    graph,
                    _SCATTER_OP_TO_VIEW[node.target],
                    new_node,
                    *node.args[2:],  # 使用除了前两个参数外的其他参数
                    **node.kwargs,
                )

                handle_views(new_src)  # 处理新的视图节点
                src.replace_all_uses_with(new_src)  # 替换所有使用源节点的地方

            graph.erase_node(src)  # 删除源节点

    # 遍历图中的所有节点
    for node in graph.nodes:
        if _is_view_op(node.target):
            handle_views(node)  # 处理视图节点
        elif node.target in _SCATTER_OP_TO_VIEW:
            handle_view_scatter(node)  # 处理散射操作对应的视图节点
# 创建一个字典，用于存储可以原地操作的运算符及其对应的原地操作对象
inplaceable_ops = {
    # 使用 aten.index_put.default 创建 InplaceableOp 对象，初始版本为 0
    aten.index_put.default: InplaceableOp(aten.index_put_.default, 0),
    # 使用 aten._unsafe_index_put.default 创建 InplaceableOp 对象，初始版本为 0
    aten._unsafe_index_put.default: InplaceableOp(inductor_prims._unsafe_index_put_, 0),
    # 使用 _generalized_scatter 创建 InplaceableOp 对象，初始版本为 0，额外的检查条件为 should_reinplace_scatter
    _generalized_scatter: InplaceableOp(
        _inplace_generalized_scatter,
        0,
        extra_check=should_reinplace_scatter,
    ),
}

try:
    # 尝试获取 torch.ops._c10d_functional 的引用
    c10d_functional = torch.ops._c10d_functional
    # 创建一个字典，用于存储可以原地操作的集体运算符及其对应的原地操作对象
    inplaceable_collective_ops = {
        # 使用 c10d_functional.all_reduce.default 创建 InplaceableOp 对象，初始版本为 0
        c10d_functional.all_reduce.default: InplaceableOp(
            c10d_functional.all_reduce_.default, 0
        ),
        # 使用 c10d_functional.all_reduce_coalesced.default 创建 InplaceableOp 对象，初始版本为 0
        c10d_functional.all_reduce_coalesced.default: InplaceableOp(
            c10d_functional.all_reduce_coalesced_.default, 0
        ),
    }
    # 将 inplaceable_collective_ops 字典中的内容添加到 inplaceable_ops 字典中
    inplaceable_ops.update(inplaceable_collective_ops)
except AttributeError:
    # 如果出现 AttributeError，说明 _c10d_functional ops 仅在 torch 使用 USE_DISTRIBUTED=1 编译时可用
    # 在这种情况下，忽略异常并不执行任何操作
    pass

# 创建一个空的字典，用于存储可以原地操作的 foreach 运算符及其对应的原地操作对象
inplaceable_foreach_ops: Dict[torch._ops.OpOverload, InplaceableOp] = {}
# 遍历 inplaceable_foreach_ops_lowerings.items() 中的键值对
for outplace_op, inplace_op in inplaceable_foreach_ops_lowerings.items():
    # 将 outplace_op 和 inplace_op 组合成键值对，添加到 inplaceable_foreach_ops 字典中
    inplaceable_foreach_ops[outplace_op] = InplaceableOp(inplace_op, 0)

# 创建一个集合，包含不依赖于张量数据的元数据运算符
META_ONLY_OPS = {
    aten.sym_size.int,
    aten.sym_stride.int,
    aten.sym_numel.default,
    aten.sym_storage_offset.default,
}

def reinplace_inplaceable_ops_core(graph: torch.fx.Graph) -> None:
    """
    Reinplaces in-placeable operations.
    如果在当前节点之后没有使用变异参数的视图，则可以在原地执行操作。
    上述算法可以通过观察副作用来证明。在向前遍历图时，只有后续节点才能观察到当前节点的副作用。
    如果当前节点后续既未使用，也没有视图用于后续图中，则可以安全地执行原地操作，因为没有办法观察到副作用。
    对于图输入，条件稍有不同，只有在上述条件为真且 epilogue 中存在 copy_ 操作时，才能执行原地操作，
    这表明调用方希望观察到变异。

    与 JIT Inductor 不同，AOTInductor 目前从输入参数中解开权重和缓冲区，因此 AOTInductor 在 get_attr 上检查变异而不是占位符。
    未来可能会更改此行为。
    """
    
    # 创建一个空字典，用于存储参数复制到复制节点的参数
    copy_args_to_copy_nodes = {}
    # 创建一个空集合，用于存储变异输入参数
    mutated_inputs = set()
    # 创建一个默认字典，用于存储存储器到节点的映射列表
    storage_to_nodes = defaultdict(list)
    # 创建一个字典，用于存储节点顺序
    node_order: Dict[Any, int] = {}
    # 使用 enumerate 遍历 graph.nodes 的逆序，并生成索引 i 和节点 node
    for i, node in enumerate(reversed(graph.nodes)):
        # 将节点 node 添加到 node_order 字典中，键为节点 node，值为逆序索引
        node_order[node] = len(graph.nodes) - i - 1
        # 将节点 node 根据其存储位置存入 storage_to_nodes 字典中
        storage_to_nodes[get_node_storage(node)].append(node)
        # 如果节点 node 的目标是 aten.copy_.default 并且第一个参数的操作是 "placeholder" 或 "get_attr"
        if node.target == aten.copy_.default and node.args[0].op in (
            "placeholder",
            "get_attr",
        ):
            # 将目标为 getitem 的源节点 src 和目标节点 dst 初始化
            dst = node.args[0]
            src = node.args[1]
            # 如果源节点 src 的目标是 operator.getitem 并且满足以下任一条件：
            # 1. src.args[0].target 是 triton_kernel_wrapper_functional 并且满足特定索引关系
            # 2. src.args[0].target 在 inplaceable_foreach_ops 中
            # 3. src.args[0].target 是 torch.ops.higher_order.auto_functionalized
            if src.target == operator.getitem and (
                (
                    src.args[0].target == triton_kernel_wrapper_functional
                    and src.args[0].kwargs["kwargs"][src.args[1]] == node.args[0]
                )
                or (src.args[0].target in inplaceable_foreach_ops)
                or (src.args[0].target == torch.ops.higher_order.auto_functionalized)
            ):
                # 更新源节点 src 为其第一个参数
                src = src.args[0]

            # 将 (dst, src) 元组映射到节点 node，表示复制关系
            copy_args_to_copy_nodes[(dst, src)] = node

            # 将节点 node 的第一个参数添加到 mutated_inputs 集合中，表示其被改变
            mutated_inputs.add(node.args[0])

    # 定义函数 any_use_of_views_after_node，用于判断节点 node 之后是否有视图使用
    def any_use_of_views_after_node(node, shared_view_nodes, *, copy_node):
        # 获取节点 node 和复制节点 copy_node 的顺序位置
        node_loc = node_order[node]
        copy_node_loc = node_order[copy_node] if copy_node is not None else None

        # 定义函数 is_meta_only_user，用于判断节点是否只是元数据操作的用户
        def is_meta_only_user(node):
            # 如果节点的目标是视图操作，则递归判断其所有用户
            if _is_view_op(node.target):
                return all(is_meta_only_user(u) for u in node.users)
            # 否则判断节点目标是否在 META_ONLY_OPS 中
            return node.target in META_ONLY_OPS

        # 遍历 shared_view_nodes 中的每个视图节点 view
        for view in shared_view_nodes:
            # 遍历视图节点 view 的每个用户节点 user
            for user in view.users:
                # 获取用户节点 user 的顺序位置
                user_loc = node_order[user]
                # 如果用户节点 user 在节点 node 之前，则跳过
                if user_loc <= node_loc:
                    continue
                # 如果存在复制节点 copy_node 并且用户节点 user 在其之后，则跳过
                if copy_node_loc is not None and copy_node_loc <= user_loc:
                    continue
                # 如果用户节点 user 只是对形状元数据的操作，则跳过
                if is_meta_only_user(user):
                    continue
                # 若找到一个使用节点，则返回 True
                return True
        # 若没有找到使用节点，则返回 False
        return False
    def can_inplace(node, mutated_arg):
        # 如果变异参数是列表或元组，递归检查每个元素是否可以进行原地操作
        if isinstance(mutated_arg, (list, tuple)):
            return all(can_inplace(node, arg) for arg in mutated_arg)

        # 获取变异参数的存储对象，如果为None，则不支持原地操作
        if get_node_storage(mutated_arg) is None:
            return False
        
        # 获取共享视图节点列表
        shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]
        
        # 如果变异参数的操作是 "placeholder" 或 "get_attr"
        if mutated_arg.op in ("placeholder", "get_attr"):
            # 如果不存在拷贝节点，不支持原地操作
            if not (
                copy_node := copy_args_to_copy_nodes.get((mutated_arg, node), False)
            ):
                return False

            # 如果节点后存在视图节点的使用，则不支持原地操作
            if any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=copy_node
            ):
                return False

            return True
        # 如果共享视图节点中存在操作是 "placeholder" 或 "get_attr" 的节点
        elif any(view.op in ("placeholder", "get_attr") for view in shared_view_nodes):
            # 不支持原地操作，因为变异参数是输入图中的视图之一
            # 需要更复杂的算法来处理这种情况
            return False
        else:
            # 如果节点后不存在视图节点的使用，则支持原地操作
            return not any_use_of_views_after_node(
                node, shared_view_nodes, copy_node=None
            )

    # 初始化一个空的替换字典，用于存储将要替换的节点
    replace_dict: Dict[torch.fx.Node, torch.fx.Node] = {}

    def reinplace_and_refine_tensors_to_clone(old_tensors_to_clone, kwargs):
        # 存储需要克隆的张量名称列表
        tensors_to_clone: List[str] = []
        
        # 遍历旧的需要克隆的张量列表
        for arg in old_tensors_to_clone:
            # 断言参数在关键字参数中存在
            assert arg in kwargs
            mutated_arg = kwargs[arg]
            
            # 如果可以原地操作
            if can_inplace(node, mutated_arg):
                # 获取拷贝节点
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                
                # 如果存在拷贝节点，将其放入替换字典中，用于替换其参数节点
                if copy_node is not None:
                    replace_dict[copy_node] = copy_node.args[0]
                
                # 遍历节点的使用者，如果使用者是 operator.getitem 并且参数为当前张量名称，则替换使用者节点
                for user in node.users:
                    if user.target == operator.getitem and user.args[1] == arg:
                        replace_dict[user] = mutated_arg
            else:
                # 如果不能原地操作，则将当前张量名称添加到需要克隆的张量列表中
                tensors_to_clone.append(arg)
        
        # 返回需要克隆的张量名称列表
        return tensors_to_clone
    # 遍历图中的每个节点
    for node in graph.nodes:
        # 如果节点的目标操作在 inplaceable_ops 字典中存在
        if (inplaceable_op := inplaceable_ops.get(node.target, None)) is not None:
            # 获取被改变的参数
            mutated_arg = node.args[inplaceable_op.mutated_arg]
            # 检查是否可以原地操作并且通过额外的检查函数
            if can_inplace(node, mutated_arg) and inplaceable_op.extra_check(node):
                # TODO(yifu): 这段代码没有正确处理那些对多个输入进行变异操作的复制结束语句。
                # 需要修改复制节点追踪逻辑以支持这种情况。
                # 获取与变异参数和节点对应的复制节点
                copy_node = copy_args_to_copy_nodes.get((mutated_arg, node))
                # 如果存在复制节点，则用复制节点的第一个参数替换替换字典中的复制节点
                if copy_node is not None:
                    replace_dict[copy_node] = copy_node.args[0]
                # 将节点的目标操作设置为原地操作的目标操作
                node.target = inplaceable_op.inplace_op
        # 如果节点的目标操作是 torch.ops.higher_order.auto_functionalized
        elif node.target == torch.ops.higher_order.auto_functionalized:
            # 获取可变操作
            _mutable_op = node.args[0]
            from torch._higher_order_ops.auto_functionalize import get_mutable_arg_names

            # 获取可变参数的名称列表
            tensors_to_clone = get_mutable_arg_names(_mutable_op)
            # 不尝试对 Optional[Tensor] 参数为 None 的进行原地操作
            tensors_to_clone = [
                t for t in tensors_to_clone if node.kwargs[t] is not None
            ]
            # 对需要克隆的张量进行重新原地操作和细化
            tensors_to_clone = reinplace_and_refine_tensors_to_clone(
                tensors_to_clone, node.kwargs
            )

            # 存储元数据。后面有一个步骤，将 auto_functionalized 操作分解为克隆 + 可变操作；
            # 此元数据告诉分解过程仅对以下输入进行克隆
            node.meta["only_clone_these_tensors"] = tensors_to_clone
        # 如果节点的目标操作在 inplaceable_triton_ops 集合中
        elif node.target in inplaceable_triton_ops:
            # inplaceable_triton_ops 需要一个额外的参数 tensors_to_clone，其中包含要克隆的张量列表
            # 对这些张量进行重新原地操作和细化
            tensors_to_clone = reinplace_and_refine_tensors_to_clone(
                node.kwargs["tensors_to_clone"], node.kwargs["kwargs"]
            )

            # 创建新的 kwargs 字典，将 tensors_to_clone 更新到 kwargs 中
            kwargs = dict(node.kwargs)
            kwargs["tensors_to_clone"] = tensors_to_clone
            # 将节点的 kwargs 更新为不可变字典
            node.kwargs = immutable_dict(kwargs)
        # 如果节点的目标操作在 inplaceable_foreach_ops 字典中存在
        elif (inplaceable_op := inplaceable_foreach_ops.get(node.target, None)) is not None:
            # 获取变异参数列表
            mutated_args = node.args[inplaceable_op.mutated_arg]

            # 如果不是所有变异参数与节点在复制参数到复制节点的映射中
            if not all((arg, node) in copy_args_to_copy_nodes for arg in mutated_args):
                continue

            # 如果可以原地操作这些变异参数
            if can_inplace(node, mutated_args):
                # 对每个变异参数执行以下操作
                for arg in mutated_args:
                    # 获取与参数和节点对应的复制节点
                    copy_node = copy_args_to_copy_nodes[(arg, node)]
                    # 将替换字典中的复制节点替换为其第一个参数
                    replace_dict[copy_node] = copy_node.args[0]

                # 将节点的目标操作设置为原地操作的目标操作
                node.target = inplaceable_op.inplace_op
    # 遍历替换字典中的每一个键值对，node 是原始值，replacement 是替换值
    for node, replacement in replace_dict.items():
        # 在替换字典中查找替换值的最终替换结果，直到找到不再需要替换的值
        while replacement in replace_dict:
            replacement = replace_dict[replacement]
        # 将原始值更新为最终替换结果
        replace_dict[node] = replacement

        # 将图中所有使用原始值的地方替换为最终替换结果
        node.replace_all_uses_with(replacement)
        
        # 从图中移除原始值对应的节点
        graph.erase_node(node)
# 在图中执行操作以规范化视图散列操作
def reinplace_inplaceable_ops(graph: torch.fx.Graph) -> None:
    canonicalize_view_scatter_ops(graph)
    # 调用函数 canonicalize_view_scatter_ops 对图进行视图散列操作的规范化

    # 在图中执行操作以重新放置不可就地操作的核心函数
    reinplace_inplaceable_ops_core(graph)
    # 调用函数 reinplace_inplaceable_ops_core 处理图中的不可就地操作

    # 分解广义散列操作
    decompose_generalized_scatter(graph)
    # 调用函数 decompose_generalized_scatter 对图中的广义散列操作进行分解
```