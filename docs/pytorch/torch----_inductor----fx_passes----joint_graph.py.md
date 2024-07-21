# `.\pytorch\torch\_inductor\fx_passes\joint_graph.py`

```py
# mypy: allow-untyped-defs
# 导入必要的库和模块
import itertools  # 导入 itertools 库，提供迭代工具
import logging  # 导入 logging 模块，用于记录日志信息
import typing  # 导入 typing 模块，用于类型提示
from collections import Counter  # 从 collections 模块中导入 Counter 类
from typing import Dict, List, Set, Union  # 导入多个类型提示

import torch  # 导入 PyTorch 库
import torch._guards  # 导入 PyTorch 私有模块
from torch._inductor.constant_folding import ConstantFolder  # 从 torch._inductor.constant_folding 模块中导入 ConstantFolder 类
from torch.fx.experimental.symbolic_shapes import statically_known_true  # 从 torch.fx.experimental.symbolic_shapes 模块中导入 statically_known_true 函数
from torch.fx.passes.graph_transform_observer import GraphTransformObserver  # 从 torch.fx.passes.graph_transform_observer 模块中导入 GraphTransformObserver 类
from torch.multiprocessing.reductions import StorageWeakRef  # 从 torch.multiprocessing.reductions 模块中导入 StorageWeakRef 类

from .. import config  # 导入相对路径下的 config 模块
from ..pattern_matcher import (  # 导入相对路径下的 pattern_matcher 模块的多个符号
    CallFunction,
    init_once_fakemode,
    KeywordArg,
    Match,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)
from .replace_random import replace_random_passes  # 从当前目录下的 replace_random 模块中导入 replace_random_passes 函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器
patterns = PatternMatcherPass()  # 创建 PatternMatcherPass 实例
aten = torch.ops.aten  # 获取 torch 的 aten 操作符
prims = torch.ops.prims  # 获取 torch 的 prims 操作符

pass_patterns = [  # 创建 pass_patterns 列表，包含多个 PatternMatcherPass 实例
    patterns,
    PatternMatcherPass(),
]


@init_once_fakemode  # 装饰器，用于初始化模式
def lazy_init():
    from .fuse_attention import _sfdp_init  # 导入当前目录下 fuse_attention 模块的 _sfdp_init 函数
    from .misc_patterns import _misc_patterns_init  # 导入当前目录下 misc_patterns 模块的 _misc_patterns_init 函数
    from .pad_mm import _pad_mm_init  # 导入当前目录下 pad_mm 模块的 _pad_mm_init 函数

    _pad_mm_init()  # 执行 _pad_mm_init 函数，初始化 pad_mm 模块
    _sfdp_init()  # 执行 _sfdp_init 函数，初始化 fuse_attention 模块
    _misc_patterns_init()  # 执行 _misc_patterns_init 函数，初始化 misc_patterns 模块


@torch.utils._python_dispatch._disable_current_modes()  # 装饰器，用于禁用当前模式
def remove_no_ops(
    gm: torch.fx.GraphModule, zeros: Set[torch.fx.Node], ones: Set[torch.fx.Node]
):
    "Removes no-ops: (+ 0, - 0, * 1, / 1)"
    graph = gm.graph  # 获取 GraphModule 对象的计算图

    def fake_tensors_eq(t1, t2, fields=("shape", "dtype", "device")):
        if any(not isinstance(t, torch.Tensor) for t in (t1, t2)):
            return False
        for field in fields:
            if getattr(t1, field) != getattr(t2, field):
                return False
        return True

    def replace_no_op(node, replace_input_index):
        replacement = node.args[replace_input_index]

        # https://github.com/pytorch/pytorch/issues/86128 causes
        # non-Tensor inputs even for ops with only Tensor inputs.
        # TODO - decompose/type promote to avoid this
        if not all(isinstance(arg, torch.fx.Node) for arg in node.args):
            return

        if not fake_tensors_eq(node.meta["val"], replacement.meta["val"]):
            if fake_tensors_eq(
                node.meta["val"],
                replacement.meta["val"],
                ("shape", "device"),
            ):
                with graph.inserting_after(node):
                    replacement = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(replacement, node.meta["val"].dtype),
                    )
            else:
                return

        node.replace_all_uses_with(replacement)  # 替换节点的所有使用为替代节点
        replacement.meta.update(node.meta)  # 更新替代节点的元信息
        graph.erase_node(node)  # 从计算图中擦除节点
    # 查找图中所有使用 aten.add.Tensor 调用函数的节点
    for node in graph.find_nodes(op="call_function", target=aten.add.Tensor):
        # 处理 Tensor-Scalar 相加，这是不同的模式
        if len(node.args) == 2:
            # 如果其中一个参数在 zeros 中，或者 alpha 参数不为 1，则跳过
            if (
                not any(e in zeros for e in node.args)
                or node.kwargs.get("alpha", 1) != 1
            ):
                continue

            # 确定需要替换的参数的索引位置
            replace_index = 1 if node.args[0] in zeros else 0
            # 调用替换函数，将指定索引的参数替换为 no-op
            replace_no_op(node, replace_index)

    # 查找图中所有使用 aten.sub.Tensor 调用函数的节点
    for node in graph.find_nodes(op="call_function", target=aten.sub.Tensor):
        # 如果节点有两个参数
        if len(node.args) == 2:
            # 如果第二个参数不在 zeros 中，或者 alpha 参数不为 1，则跳过
            if node.args[1] not in zeros or node.kwargs.get("alpha", 1) != 1:
                continue

            # 调用替换函数，将第一个参数替换为 no-op
            replace_no_op(node, 0)

    # 查找图中所有使用 aten.mul.Tensor 调用函数的节点
    for node in graph.find_nodes(op="call_function", target=aten.mul.Tensor):
        # 如果节点有两个参数
        if len(node.args) == 2:
            # 如果两个参数中至少一个在 ones 中
            if not any(e in ones for e in node.args):
                continue

            # 确定需要替换的输入参数的索引位置
            replace_input_index = 1 if node.args[0] in ones else 0
            # 调用替换函数，将指定索引的输入参数替换为 no-op
            replace_no_op(node, replace_input_index)

    # 查找图中所有使用 aten.div.Tensor 调用函数的节点
    for node in graph.find_nodes(op="call_function", target=aten.div.Tensor):
        # 如果节点有两个参数且第二个参数在 ones 中
        if len(node.args) == 2 and node.args[1] in ones:
            # 调用替换函数，将第一个参数替换为 no-op
            replace_no_op(node, 0)

    # 处理从图中返回的 meta 张量，这些张量没有数据，可以用 empty_strided 替换
    for output_node in graph.find_nodes(op="output"):
        had_meta_return = False

        # 定义访问函数，用于检查节点的 meta 数据
        def visit(n):
            nonlocal had_meta_return
            val = n.meta.get("val")
            # 如果 meta 数据是 torch.Tensor 并且设备类型是 "meta"
            if isinstance(val, torch.Tensor) and val.device.type == "meta":
                # 在 output 节点之前插入操作
                with graph.inserting_before(output_node):
                    # 替换所有使用该节点的地方为调用 empty_strided 的结果
                    n.replace_all_uses_with(
                        graph.call_function(
                            torch.ops.aten.empty_strided.default,
                            args=(val.size(), val.stride()),
                            kwargs={"dtype": val.dtype, "device": val.device},
                        )
                    )
                had_meta_return = True

        # 对 output 节点的参数应用访问函数
        torch.fx.map_arg(output_node.args, visit)
        # 如果曾经有 meta 返回，消除死代码
        if had_meta_return:
            graph.eliminate_dead_code()
# 禁用当前的 Torch 模式，确保运行在正确的上下文中
@torch.utils._python_dispatch._disable_current_modes()
def remove_redundant_views(gm: torch.fx.GraphModule):
    """
    Removes redundant views by reusing existing ones.
    """

    # 一个字典，将张量映射到其所有别名视图
    views: Dict[torch.fx.Node, Dict[torch.dtype, torch.fx.Node]] = {}
    # 获取图模块的计算图
    graph = gm.graph

    # 遍历图中所有调用函数为 torch.ops.aten.view.dtype 的节点
    for node in graph.find_nodes(op="call_function", target=torch.ops.aten.view.dtype):
        # 获取视图的源张量和目标类型
        src = node.args[0]
        to_type = node.args[1]
        # 获取已存在的视图
        existing_views = views.get(src)
        is_needed = True

        if existing_views:
            # 如果已存在该源张量的视图
            # 尝试使用已存在的视图替换当前视图
            alias = existing_views.get(to_type)
            if alias:
                # 不再需要当前视图，直接替换所有使用该视图的节点
                is_needed = False
                node.replace_all_uses_with(alias)
                # 更新别名节点的元数据
                alias.meta.update(node.meta)
                # 从图中移除当前视图节点
                graph.erase_node(node)
        else:
            # 如果不存在已存在的视图，则创建一个新的
            from_type = src.meta["val"].dtype
            existing_views = {from_type: src}
            views[src] = existing_views

        if is_needed:
            # 如果当前视图仍然需要保留，则将其加入到已存在视图字典中
            existing_views.setdefault(to_type, node)
            views[node] = existing_views

    # 清理未使用的视图
    while True:
        unused_views = [alias for alias in views if not alias.users]
        if len(unused_views) == 0:
            break
        for unused in unused_views:
            # 从视图字典和图中移除未使用的视图节点
            views.pop(unused)
            graph.erase_node(unused)


class UniformValueConstantFolder(ConstantFolder):
    """
    Runs constant folding and replaces tensors that have a unifrom value
    with a tensor constructor call: aten.full([shape], value, ...)
    """

    def __init__(self, gm, skip_constructors=False):
        # 调用父类构造函数初始化常量折叠器
        super().__init__(gm, skip_constructors)
        # 存储节点的存储指针映射
        self.node_storages_ptrs: Dict[torch.fx.Node, int] = {}
        # 存储常量数据的弱引用映射
        self.constant_data_ptrs: Dict[torch.fx.Node, StorageWeakRef] = {}
        # 存储节点替换的形状映射
        self.node_replacements_shapes: Dict[torch.fx.Node, List[int]] = {}

    def insertable_tensor_check(self, t: torch.Tensor) -> bool:
        # TODO - 可以考虑在这里替换为 arange 的张量
        return (
            t.numel() != 0
            and bool((t == t.flatten()[0]).all())
            and torch._C._has_storage(t)
            and t.layout == torch.strided
        )

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        # 记录节点替换为张量，并保存其展平后的第一个元素
        self.node_replacements[node] = tensor.flatten()[0].item()
        # 记录常量数据的弱引用
        self.constant_data_ptrs[node] = StorageWeakRef(tensor.untyped_storage())
        # 记录张量的形状，确保形状维度均为整数类型
        shape = list(tensor.shape)
        assert all(type(dim) is int for dim in shape)
        self.node_replacements_shapes[node] = shape


@torch.utils._python_dispatch._disable_current_modes()
def constant_fold_uniform_value(gm: torch.fx.GraphModule):
    # 导入 torch 的 aten 操作模块
    aten = torch.ops.aten

    # 创建 UniformValueConstantFolder 对象，并运行常量折叠优化
    cf = UniformValueConstantFolder(gm)
    cf.run()

    # 获取常量折叠后的节点替换映射
    node_replacements = cf.node_replacements

    # 注意: [constant folding refining of symints]
    # 常量折叠会部分评估图形，以使那些在编译时完全已知依赖关系的值也可能变为编译时常量。
    # 在某些情况下，这将包括我们之前尚未推断为具有常量值的 symint，在常量折叠中被推断为常量的情况。
    # 例如:
    # unbacked_symint_eq_11 = torch.full((), 11).item()
    # torch.full((unbacked_symint_eq_11,), 0)
    node_replacements_shapes = cf.node_replacements_shapes

    # 获取图形管理器中的图形
    graph = gm.graph

    # 初始化空集合用于存储零和一的节点
    zeros = set()
    ones = set()

    # 如果一个张量没有别名，将其常量化
    constant_data_ptr_count: typing.Counter[StorageWeakRef] = Counter()

    # 遍历常量折叠后的节点替换映射，统计每个节点的常量数据指针出现次数
    for node in cf.node_replacements:
        constant_data_ptr_count[cf.constant_data_ptrs[node]] += 1
    # 遍历替换节点字典中的每个节点及其对应的值
    for node, value in node_replacements.items():
        # 当前没有一种有效的方法来使用全/零/一来实例化非连续张量
        # 目前还没有显示出它有多重要
        if "val" not in node.meta:
            # 这种情况只会在AOTI中发生
            continue

        # 从节点的元数据中获取虚拟张量
        fake_tensor = node.meta["val"]
        # 如果虚拟张量不是连续的，则跳过
        if not fake_tensor.is_contiguous(memory_format=torch.contiguous_format):
            continue

        # 如果常量数据指针的计数大于1，则跳过
        if constant_data_ptr_count[cf.constant_data_ptrs[node]] > 1:
            continue

        # 在节点之后插入新节点
        with graph.inserting_after(node):
            # 如果节点是调用函数且目标是aten.full.default，并且有两个参数
            if (
                node.op == "call_function"
                and node.target == aten.full.default
                and len(node.args) == 2
            ):
                # 使用原始的full构造函数的值，因为张量到值再回来的转换可能会丢失精度
                value = node.args[1]

            # 细化符号整数，参见上文的[constant folding refining of symints]
            # 检查运行时大小和编译时大小是否一致
            for runtime_size, compile_time_size in zip(
                node_replacements_shapes[node], fake_tensor.shape
            ):
                torch._check(runtime_size == compile_time_size)

            # zeros和ones只是被跟踪到full中，因此我们插入它们
            # 创建一个新的节点，调用aten.full.default函数
            new_node = graph.call_function(
                aten.full.default,
                args=(node_replacements_shapes[node], value),
                kwargs={
                    "dtype": fake_tensor.dtype,
                    "layout": torch.strided,
                    "device": fake_tensor.device,
                    "pin_memory": False,
                },
            )

            # 更新新节点的元数据
            new_node.meta.update(node.meta)
            # 替换所有使用当前节点的地方为新节点
            node.replace_all_uses_with(new_node)
            # 删除当前节点
            graph.erase_node(node)

            # 如果值为0，则将新节点添加到zeros集合中
            if value == 0:
                zeros.add(new_node)
            # 如果值为1，则将新节点添加到ones集合中
            elif value == 1:
                ones.add(new_node)

    # 在图中移除不必要的操作
    remove_no_ops(gm, zeros, ones)
    # 在图中移除冗余的视图操作
    remove_redundant_views(gm)
# 对图模块应用联合前向和后向图形的 FX 变换
def joint_graph_passes(graph: torch.fx.GraphModule):
    # 执行懒初始化操作
    lazy_init()
    # 计数器初始化
    count = 0
    # 如果配置中定义了联合自定义预处理，则执行以下操作
    if config.joint_custom_pre_pass is not None:
        # 使用 GraphTransformObserver 监视器，应用自定义预处理到图形上，并记录日志URL
        with GraphTransformObserver(
            graph, "joint_custom_pre_pass", config.trace.log_url_for_graph_xform
        ):
            # 调用配置中定义的联合自定义预处理函数，并传入图形对象
            config.joint_custom_pre_pass(graph.graph)
            # 增加计数器
            count += 1

    # 导入并执行后向梯度过程中的无操作操作移除函数
    from .post_grad import remove_noop_ops
    remove_noop_ops(graph.graph)

    # 如果配置中开启了图形常量折叠
    if config.joint_graph_constant_folding:
        # 使用 GraphTransformObserver 监视器，应用常量折叠到图形上，并记录日志URL
        with GraphTransformObserver(
            graph, "constant_fold_uniform_value", config.trace.log_url_for_graph_xform
        ):
            # 调用常量折叠函数，将一致值折叠应用到图形中
            constant_fold_uniform_value(graph)

    # 如果配置中开启了模式匹配器
    if config.pattern_matcher:
        # 对每个传递模式列表中的模式应用到图形中，并增加计数器
        for patterns in pass_patterns:
            count += patterns.apply(graph.graph)  # type: ignore[arg-type]

    # 如果不使用回退随机，则执行替换随机传递函数并增加计数器
    if not config.fallback_random:
        count += replace_random_passes(graph)

    # 如果配置中定义了联合自定义后处理，则执行以下操作
    if config.joint_custom_post_pass is not None:
        # 使用 GraphTransformObserver 监视器，应用自定义后处理到图形上，并记录日志URL
        with GraphTransformObserver(
            graph, "joint_custom_post_pass", config.trace.log_url_for_graph_xform
        ):
            # 调用配置中定义的联合自定义后处理函数，并传入图形对象
            config.joint_custom_post_pass(graph.graph)
            # 增加计数器
            count += 1

    # 如果计数器不为零，则进行稳定的拓扑排序、图形检查及重新编译操作
    if count:
        stable_topological_sort(graph.graph)
        graph.graph.lint()
        graph.recompile()
    # 返回处理后的图形对象
    return graph


# 注册图形模式匹配函数，用于移除 AMP 创建的类型转换链
@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        CallFunction(
            torch.ops.prims.convert_element_type.default,
            KeywordArg("arg"),
            KeywordArg("dtype1"),
        ),
        KeywordArg("dtype2"),
    ),
    pass_dict=patterns,
)
def pointless_convert(match: Match, arg, dtype1: torch.dtype, dtype2: torch.dtype):
    """移除 AMP 经常创建的类型转换链"""
    graph = match.graph
    node = match.output_node()
    # 允许的数据类型集合
    allowed = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    # 如果 dtype1 和 dtype2 均在允许的数据类型集合中
    if dtype1 in allowed and dtype2 in allowed:
        # 创建替换节点，调用类型转换函数，并更新元数据
        repl = graph.call_function(
            torch.ops.prims.convert_element_type.default, (arg, dtype2)
        )
        repl.meta.update(node.meta)
        # 替换所有使用当前节点的节点为新替换节点，并从图中擦除匹配的节点
        node.replace_all_uses_with(repl)
        match.erase_nodes(graph)


# 注册图形模式匹配函数，用于移除无操作视图
@register_graph_pattern(
    CallFunction(torch.ops.aten.view.default, KeywordArg("arg"), KeywordArg("size")),
    pass_dict=patterns,
)
def pointless_view(match: Match, arg, size):
    """移除无操作视图"""
    graph = match.graph
    node = match.output_node()
    # 获取节点参数中参数0的形状元数据
    arg_size = list(node.args[0].meta["val"].shape)  # type: ignore[union-attr]
    # 如果给定大小与参数0的形状元数据相同
    if size == arg_size:
        # 替换所有使用当前节点的节点为参数0节点，并从图中擦除匹配的节点
        node.replace_all_uses_with(node.args[0])
        match.erase_nodes(graph)


# 当 softmax 结合温度或其他缩放因子使用时，可能出现如下模式
#
#   scale(x) - scale(x).amax(dim, keepdim=True)
#
# 预期该模式最多为零，但我们可能在重新计算 scale(x) 的内外值之间存在数值差异
# 定义一个函数来处理部分 softmax 的模式匹配，其中包括反向匹配和转换数据类型的选项
def _partial_softmax_pattern(linear_func, reverse=False, to_dtype=False):
    # 如果 reverse 为 True，表示进行反向匹配，交换 inp 和 other 的位置
    if reverse:
        scaled = CallFunction(
            linear_func, KeywordArg("other"), KeywordArg("inp"), _users=MULTIPLE
        )
    else:
        # 否则按正常顺序进行匹配
        scaled = CallFunction(
            linear_func, KeywordArg("inp"), KeywordArg("other"), _users=MULTIPLE
        )
    
    # 如果需要进行数据类型转换
    if to_dtype:
        scaled = CallFunction(
            prims.convert_element_type, scaled, KeywordArg("dtype"), _users=MULTIPLE
        )
    
    # 计算 scaled 张量在指定维度上的最大值，保持维度信息
    amax = CallFunction(
        aten.amax.default, scaled, KeywordArg("dim"), KeywordArg("keepdim")
    )
    
    # 返回将 scaled 张量减去 amax 张量的结果
    return CallFunction(aten.sub.Tensor, scaled, amax)


def _other_is_broadcasted_in_dim(match):
    # 检查缩放因子是否在减少维度上是常数，这样缩放就不会改变哪个索引对应最大值
    other = match.kwargs["other"]
    if isinstance(other, (int, float)):
        return True

    inp = match.kwargs["inp"]
    if not all(isinstance(x, torch.fx.Node) for x in (inp, other)):
        return False

    inp_example = inp.meta["val"]
    other_example = other.meta["val"]
    if isinstance(other_example, (torch.SymInt, torch.SymFloat)):
        return True

    if not all(isinstance(x, torch.Tensor) for x in (inp_example, other_example)):
        return False

    inp_ndim = inp_example.ndim
    other_shape = other_example.shape
    if inp_ndim < len(other_shape):
        return False

    # 将 other_shape 填充到与 inp 相同的 ndim
    other_shape = [1] * (inp_ndim - len(other_shape)) + list(other_shape)

    dim = match.kwargs["dim"]
    if isinstance(dim, int):
        dim = (dim,)

    # 检查在指定维度上 other_shape 是否都为 1
    return all(statically_known_true(other_shape[d] == 1) for d in dim)


def mul_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None):
    # 定义替换函数 repl，根据输入的参数 inp, other 进行 softmax 模式匹配
    def repl(inp, other):
        # 如果指定了 dtype，则将 inp 转换为指定的数据类型
        if dtype is not None:
            inp = inp.to(dtype)

        # 根据 other 的正负号设置 sign
        sign: Union[int, float, torch.Tensor]
        if isinstance(other, (int, float)):
            sign = 1 if other >= 0 else -1
        else:
            one = torch.scalar_tensor(1, dtype=inp.dtype, device=inp.device)
            sign = torch.where(other >= 0, one, -one)

        # 对 inp 进行缩放，然后计算 inp 在指定维度上的最大值 max_
        inp = inp * sign
        max_ = torch.amax(inp, dim=dim, keepdim=keepdim)
        
        # 返回 softmax 计算结果
        return (inp - max_) * (sign * other)

    # 使用 repl 函数替换匹配到的模式
    match.replace_by_example(repl, [inp, other])


# 遍历 reverse 和 to_dtype 的所有组合
for reverse, to_dtype in itertools.product((False, True), repeat=2):
    # 注册部分 softmax 模式匹配图案
    register_graph_pattern(
        _partial_softmax_pattern(aten.mul.Tensor, reverse=reverse, to_dtype=to_dtype),
        pass_dict=pass_patterns[1],
        extra_check=_other_is_broadcasted_in_dim,
    )(mul_softmax_pattern)


注释：


# 这行代码似乎存在语法错误，因为括号未正确匹配
def div_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None):
    # 定义替换函数repl，用于处理softmax模式匹配
    def repl(inp, other):
        # 如果指定了dtype，则将inp转换为指定的数据类型
        if dtype is not None:
            inp = inp.to(dtype)

        # 定义变量sign，根据other的类型确定其正负号
        sign: Union[int, float, torch.Tensor]
        if isinstance(other, (int, float)):
            sign = 1 if other >= 0 else -1
        else:
            # 创建一个标量张量one，用于生成sign张量
            one = torch.scalar_tensor(1, dtype=inp.dtype, device=inp.device)
            # 根据other的值，生成sign张量，表示其正负号
            sign = torch.where(other >= 0, one, -one)

        # 将inp乘以sign，调整输入张量的值
        inp = inp * sign
        # 计算inp沿指定维度dim的最大值max_
        max_ = torch.amax(inp, dim=dim, keepdim=keepdim)
        # 返回经过softmax处理后的张量
        return (inp - max_) / (sign * other)

    # 使用match对象的replace_by_example方法，将repl函数应用于匹配的模式
    match.replace_by_example(repl, [inp, other])


# 遍历两种数据类型False和True，分别注册softmax模式
for to_dtype in (False, True):
    # 调用register_graph_pattern函数，注册_partial_softmax_pattern模式
    # 并指定pass_dict和extra_check参数
    register_graph_pattern(
        _partial_softmax_pattern(aten.div.Tensor, to_dtype=to_dtype),
        pass_dict=pass_patterns[1],
        extra_check=_other_is_broadcasted_in_dim,
    )(div_softmax_pattern)
```