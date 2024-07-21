# `.\pytorch\torch\_inductor\fx_passes\split_cat.py`

```
# 添加类型检查允许未类型化的定义
# 导入必要的库和模块
import itertools  # 提供迭代工具的函数
import logging  # 提供日志记录功能
import operator  # 提供标准操作符的函数集合
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union  # 引入类型提示相关模块
from typing_extensions import TypeAlias  # 引入类型别名支持

import torch  # 引入PyTorch深度学习框架
from torch._dynamo.utils import counters  # 导入计数器工具

from ..pattern_matcher import (  # 导入模式匹配相关的类和函数
    Arg,  # 表示模式匹配中的参数
    CallFunction,  # 表示模式匹配中的函数调用
    CallFunctionVarArgs,  # 表示模式匹配中可变参数的函数调用
    CallMethodVarArgs,  # 表示模式匹配中可变参数的方法调用
    FailedMatch,  # 表示模式匹配失败
    get_arg_value,  # 获取模式匹配中参数的值
    Ignored,  # 表示模式匹配中的忽略标志
    KeywordArg,  # 表示模式匹配中的关键字参数
    ListOf,  # 表示模式匹配中的列表
    Match,  # 表示模式匹配中的匹配对象
    MatchContext,  # 表示模式匹配的上下文
    MULTIPLE,  # 表示模式匹配中的多个匹配
    PatternExpr,  # 表示模式匹配中的模式表达式
    PatternMatcherPass,  # 表示模式匹配的通行证
    register_graph_pattern,  # 注册图模式的函数
    RepeatedExpr,  # 表示模式匹配中的重复表达式
)
from .group_batch_fusion import is_node_meta_valid, POST_GRAD_FUSIONS, PRE_GRAD_FUSIONS  # 导入批次融合相关模块

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

_Arguments: TypeAlias = Tuple[torch.fx.node.Argument, ...]  # 定义参数列表类型别名
_TransformParam: TypeAlias = Tuple[  # 定义变换参数类型别名
    Optional[_Arguments],  # 可选的参数列表
    Optional[_Arguments],  # 可选的参数列表
    Optional[_Arguments],  # 可选的参数列表
    Optional[_Arguments],  # 可选的参数列表
]
_Range: TypeAlias = Tuple[int, int]  # 定义范围类型别名

PRE_GRAD_PATTERNS: Dict[str, PatternMatcherPass] = dict()  # 预梯度模式匹配字典
POST_GRAD_PATTERNS: Dict[str, PatternMatcherPass] = dict()  # 后梯度模式匹配字典

pre_grad_pass_names = [  # 预梯度通行证名称列表
    "normalization_pass",
    "remove_split_with_size_one_pass",
    "merge_getitem_cat_pass",
    "merge_stack_tahn_unbind_pass",
    "merge_splits_pass",
    "mutate_cat_pass",
    "split_cat_pass",
    "unbind_stack_pass",
]

post_grad_pass_names = [  # 后梯度通行证名称列表
    "normalization_aten_pass",
    "decompose_mm_pass",
    "unbind_stack_aten_pass",
    "shape_padding_multiplier",
]

for pass_name in pre_grad_pass_names:
    # 排除所有与批次融合相关的通行证
    # 它们不使用模式匹配器
    if pass_name in PRE_GRAD_FUSIONS:
        continue
    PRE_GRAD_PATTERNS[pass_name] = PatternMatcherPass(
        prevent_match_across_mutations=True,  # 防止跨突变进行匹配
        pass_name=pass_name,  # 设置通行证名称
    )

for pass_name in post_grad_pass_names:
    # 排除所有与批次融合相关的通行证
    # 它们不使用模式匹配器
    if pass_name in POST_GRAD_FUSIONS:
        continue
    POST_GRAD_PATTERNS[pass_name] = PatternMatcherPass(
        prevent_match_across_mutations=True,  # 防止跨突变进行匹配
        pass_name=pass_name,  # 设置通行证名称
    )


def construct_pattern_matcher_pass(pass_name: str):
    """
    根据通行证名称返回特定的模式匹配通行证对象。
    """
    if pass_name in PRE_GRAD_PATTERNS:
        return PRE_GRAD_PATTERNS[pass_name]  # 返回预梯度模式匹配通行证对象
    else:
        return POST_GRAD_PATTERNS[pass_name]  # 返回后梯度模式匹配通行证对象


def _get_split_args_default(split_node):
    """
    根据拆分节点返回默认的拆分参数。
    """
    input_kwarg = "tensor"  # 输入关键字参数
    split_size_kwarg = "split_size_or_sections"  # 拆分尺寸或片段关键字参数
    dim_kwarg = "dim"  # 维度关键字参数
    default_dim_value = 0  # 默认的维度值为0
    if split_node.op == "call_method":
        split_size_kwarg = "split_size"  # 如果是方法调用，则使用拆分尺寸参数
    return (
        get_arg_value(split_node, 0, input_kwarg),  # 获取输入参数的值
        get_arg_value(split_node, 1, split_size_kwarg),  # 获取拆分尺寸参数的值
        get_arg_value(split_node, 2, dim_kwarg) or default_dim_value,  # 获取维度参数的值或使用默认维度值
    )


def _get_dim(node: Any):
    """
    获取节点的维度信息。
    """
    assert isinstance(node, torch.fx.Node)  # 断言节点类型为torch.fx.Node
    if "dim" in node.kwargs:  # 如果节点的关键字参数中包含维度信息
        assert isinstance(node.kwargs["dim"], int)  # 断言维度值为整数类型
        return node.kwargs["dim"]  # 返回节点的维度值
    # 如果节点的目标函数是 torch.unbind
    if node.target == torch.unbind:
        # 如果参数列表长度为2
        if len(node.args) == 2:
            # 断言最后一个参数是整数类型，返回该参数作为维度值
            assert isinstance(node.args[-1], int)
            return node.args[-1]
        # 默认返回0，表示维度为0
        return 0  # defaults to dim=0
    
    # 如果节点的目标函数是 torch.split
    if node.target == torch.split:
        # 如果参数列表长度为3
        if len(node.args) == 3:
            # 断言最后一个参数是整数类型，返回该参数作为维度值
            assert isinstance(node.args[-1], int)
            return node.args[-1]
        # 默认返回0，表示维度为0
        return 0  # defaults to dim=0
    
    # 如果以上条件都不满足，则抛出断言错误，提示无法从节点的目标函数中提取维度信息
    raise AssertionError(
        f"Can't extract `dim` from {node.target} {node.args} {node.kwargs}"
    )
# noqa: W605
# ############The pattern to be optimized is#########
#         unbind (dim=0)
#       /   ...    \
# getitem      getitem   -> user=1
#    |            |
#  split         split  -> dim=1, user=1, split_section_size=1
#    |            |
#  getitem       getitem  -> user=1
#    \           /
#        cat (dim=1)  -> user=1
#          |

# ################After transformation#############
#          unbind (dim=0)
#        /    ...   \
#    getitem       getitem  -> user=1
#       \          /
#        cat (dim=1)  -> user=1
#         |

def normalize_split_base(
    match: Match,
    _get_split_args: Callable[
        [torch.fx.Node], Tuple[Optional[torch.fx.Node], Optional[Any], Optional[int]]
    ],
):
    """
    Normalize split with split_size into split_with_sizes, so that we only deal with one type of split in
    subsequent optimizations
    """
    split_node = match.nodes[0]  # 获取匹配中的第一个节点作为 split_node
    graph = match.graph  # 获取匹配的图对象
    split_input, split_size, split_dim = _get_split_args(split_node)  # 使用给定的函数获取 split_node 的参数

    # 如果无法找到 split 的参数，则记录日志并返回
    if split_input is None or split_dim is None or split_size is None:
        log.debug("couldn't find split args")
        return
    
    # 如果 split_node 的元数据中没有示例值，则记录日志并返回
    if "example_value" not in split_node.meta:
        log.debug("example value absent for node: %s", split_node)
        return
    
    # 确保 split_node 的示例值是列表或元组
    assert isinstance(split_node.meta["example_value"], (list, tuple))
    
    # 获取所有示例值中每个分割部分的大小
    split_sections = [t.size()[split_dim] for t in split_node.meta["example_value"]]

    # 如果任何分割部分是 torch.SymInt 类型，则暂时返回，不进行优化
    if any(isinstance(section, torch.SymInt) for section in split_sections):
        # TODO dynamic_shapes with assume_static_by_default=False fails while AOT Autograd tracing.
        return
    
    # 如果 split_dim 是负数，则将其标准化为非负数
    if split_dim < 0:
        split_dim += split_input.meta["example_value"].dim()

    # 准备新的参数和关键字参数
    new_args = (split_input, split_sections)
    new_kwargs = {"dim": split_dim}

    # 如果 split_node 的参数和关键字参数与新参数一致，并且操作为 "call_function"，则直接返回
    if (
        split_node.args == new_args
        and split_node.kwargs == new_kwargs
        and split_node.op == "call_function"
    ):
        return

    # 在 split_node 之后插入新的 split 节点，并更新相关的元数据
    with graph.inserting_after(split_node):
        new_split_node = graph.call_function(
            torch.split,
            args=new_args,
            kwargs=new_kwargs,
        )
    split_node.replace_all_uses_with(new_split_node)  # 替换所有使用 split_node 的地方为 new_split_node
    new_split_node.meta.update(split_node.meta)  # 更新新节点的元数据
    graph.erase_node(split_node)  # 删除原来的 split_node
    counters["inductor"]["normalization_pass"] += 1  # 计数器记录归一化步骤的执行次数


@register_graph_pattern(
    CallFunctionVarArgs(torch.split, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
@register_graph_pattern(
    CallMethodVarArgs("split", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_split_default(match: Match, *args, **kwargs):
    """
    Register and apply normalization patterns for torch.split and split method calls.
    """
    return normalize_split_base(match, _get_split_args_default)


@register_graph_pattern(
    CallFunctionVarArgs(torch.split, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("remove_split_with_size_one_pass"),
)
@register_graph_pattern(
    CallMethodVarnormalize_split_default(match: Match, *args, **kwargs):
    return normalize_split_base(match, _get_split_args_default)
    pass_dict=construct_pattern_matcher_pass("remove_split_with_size_one_pass"),
@register_graph_pattern(
    CallFunctionVarArgs(torch.cat, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_cat_default(match: Match, *args, **kwargs):
    # 导入必要的模块
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    # 获取匹配对象中的第一个节点（假设为torch.cat调用）
    cat_node = match.nodes[0]
    # 从match对象中获取图(graph)
    graph = match.graph
    # 从cat_node节点中获取参数"tensors"的值
    tensors = get_arg_value(cat_node, 0, "tensors")
    # 从cat_node节点中获取参数"dim"的值
    cat_dim = get_arg_value(cat_node, 1, "dim")
    # 如果cat_dim为None
    if cat_dim is None:
        # 尝试从cat_node的关键字参数中获取"axis"的值
        cat_axis = cat_node.kwargs.get("axis")
        # 如果cat_axis不为None，则将其赋值给cat_dim
        if cat_axis is not None:
            cat_dim = cat_axis
        else:
            # 否则默认将cat_dim设为0
            cat_dim = 0
    # 如果tensors或者cat_dim为None
    if tensors is None or cat_dim is None:
        # 记录调试信息，表示找不到cat操作的参数
        log.debug("couldn't find cat args")
        return
    # 断言tensors是列表或元组类型
    assert isinstance(tensors, (list, tuple))
    # 遍历tensors列表，包括cat_node本身
    for tensor in itertools.chain([cat_node], tensors):
        # 如果tensor的元数据中不存在"example_value"
        if "example_value" not in tensor.meta:
            # 记录调试信息，表示找不到tensor的示例值
            log.debug("example value absent for node: %s", tensor)
            return

    # 获取cat_node的示例值的维度
    ndim = cat_node.meta["example_value"].dim()

    # 定义一个函数判断是否是空张量
    def is_empty_tensor(x):
        # 特殊情况，torch.cat支持与空张量拼接
        x_shape = x.meta["example_value"].shape
        return len(x_shape) == 1 and guard_size_oblivious(x_shape[0] == 0)

    # 断言所有tensors中的张量与cat_node示例值的维度相同，或者是空张量
    assert all(
        ndim == x.meta["example_value"].dim() or is_empty_tensor(x) for x in tensors
    )

    # 如果cat_dim小于0，将其标准化到正数
    if cat_dim < 0:  # Normalize cat dim
        cat_dim += ndim

    # 构造新的参数和关键字参数
    new_args = (tensors,)
    new_kwargs = {"dim": cat_dim}
    # 如果cat_node的args、kwargs和操作类型与新构造的一致，则直接返回
    if (
        cat_node.args == new_args
        and cat_node.kwargs == new_kwargs
        and cat_node.op == "call_function"
    ):
        return

    # 在图中cat_node之后插入新的节点
    with graph.inserting_after(cat_node):
        # 创建一个新的cat操作节点
        new_cat_node = graph.call_function(
            torch.cat,
            args=new_args,
            kwargs=new_kwargs,
        )
    # 将原cat_node的所有使用处替换为新创建的节点new_cat_node
    cat_node.replace_all_uses_with(new_cat_node)
    # 将新节点new_cat_node的元数据更新为原cat_node的元数据
    new_cat_node.meta.update(cat_node.meta)
    # 在图中删除原cat_node节点
    graph.erase_node(cat_node)
    # 更新计数器，记录归一化处理的次数
    counters["inductor"]["normalization_pass"] += 1
# 注册一个图模式匹配器，用于匹配调用 torch.stack 的变长参数函数，用户为 MULTIPLE
# 以及使用名为 "normalization_pass" 的字典作为参数传递
@register_graph_pattern(
    CallFunctionVarArgs(torch.stack, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_stack_default(match: Match, *args, **kwargs):
    # 获取匹配中的第一个节点
    node = match.nodes[0]
    # 获取匹配的图对象
    graph = match.graph
    # 获取节点参数中的 "tensors"，如果不存在则返回 None
    tensors = get_arg_value(node, 0, "tensors")
    # 获取节点参数中的 "dim"，如果不存在则默认为 0
    dim = get_arg_value(node, 1, "dim") or 0
    
    # 如果 tensors 或 dim 为 None，则记录调试信息并返回
    if tensors is None or dim is None:
        log.debug("couldn't find stack args")
        return
    
    # 断言 tensors 是列表或元组
    assert isinstance(tensors, (list, tuple))

    # 检查是否有一些节点缺少 "example_value" 元数据
    for tensor in itertools.chain([node], tensors):
        if "example_value" not in tensor.meta:
            log.debug("example value absent for node: %s", tensor)
            return

    # 获取节点的 "example_value" 的维度信息
    ndim = node.meta["example_value"].dim()
    
    # 如果 dim 小于 0，则进行归一化处理
    if dim < 0:  # Normalize dim
        dim += ndim

    # 在节点后插入新节点
    with graph.inserting_after(node):
        # 创建一个调用 torch.stack 函数的新节点
        new_node = graph.call_function(
            node.target,
            args=(tensors,),
            kwargs={"dim": dim},
        )
    # 替换原节点所有的使用者为新节点
    node.replace_all_uses_with(new_node)
    # 更新新节点的元数据
    new_node.meta.update(node.meta)
    # 删除原节点
    graph.erase_node(node)
    # 更新计数器，记录归一化操作的次数
    counters["inductor"]["normalization_pass"] += 1


def find_next_users(split_node: torch.fx.Node) -> List[torch.fx.Node]:
    # 初始化一个空列表，用于存储下一级使用者节点
    next_users = []
    # 遍历 split_node 的所有用户节点
    for getitem_node in split_node.users.keys():
        # 遍历每个 getitem_node 的用户节点
        for getitem_user in getitem_node.users.keys():
            # 如果 getitem_user 不在 next_users 中，则添加进去
            if getitem_user not in next_users:
                next_users.append(getitem_user)
    # 返回所有下一级使用者节点的列表
    return next_users


# 注册一个图模式匹配器，用于匹配调用 "squeeze" 方法的变长参数函数，用户为 MULTIPLE
# 以及使用名为 "normalization_pass" 的字典作为参数传递
@register_graph_pattern(
    CallMethodVarArgs("squeeze", users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_pass"),
)
def normalize_squeeze_default(match: Match, *args, **kwargs):
    # 获取匹配中的第一个节点
    squeeze_node = match.nodes[0]
    # 获取 squeeze_node 参数中的第一个参数作为输入
    squeeze_input = get_arg_value(squeeze_node, 0)

    # 检查是否在 squeeze_node 的关键字参数中存在 "dim"
    if "dim" in squeeze_node.kwargs:
        # 如果存在，则确保 squeeze_node 的参数列表长度为 1
        assert len(squeeze_node.args) == 1
        # 将 dim 设置为 squeeze_node 的关键字参数中的 "dim" 值
        dim = squeeze_node.kwargs["dim"]
    elif len(squeeze_node.args) == 1:
        # 如果 squeeze_node 的参数列表长度为 1，则 dim 设为 None
        dim = None
    elif len(squeeze_node.args) == 2:
        # 如果 squeeze_node 的参数列表长度为 2，则 dim 设为第二个参数
        dim = squeeze_node.args[1]
    else:
        # 如果 squeeze_node 的参数列表长度大于 2，则 dim 设为第二个参数及之后的所有参数
        dim = squeeze_node.args[1:]

    # 如果 dim 是 Sequence 类型且长度为 1，则将 dim 设为其中的值
    if isinstance(dim, Sequence) and len(dim) == 1:
        dim = dim[0]

    # 在 squeeze_node 后插入新节点
    with match.graph.inserting_after(squeeze_node):
        # 根据 dim 的情况调用 torch.squeeze 方法创建新节点
        if dim is None:
            new_squeeze_node = match.graph.call_function(
                torch.squeeze, args=(squeeze_input,)
            )
        else:
            new_squeeze_node = match.graph.call_function(
                torch.squeeze, args=(squeeze_input,), kwargs={"dim": dim}
            )
    # 替换原 squeeze_node 所有的使用者为新节点
    squeeze_node.replace_all_uses_with(new_squeeze_node)
    # 删除原 squeeze_node
    match.graph.erase_node(squeeze_node)


class TorchSplit(CallFunction):
    """
    匹配调用 torch.split 的类，如果调用形式符合标准化要求，则确保所有 split 的用户是唯一的 getitem 节点。
    """
    # 使用 KeywordArg("dim") 作为 `dim` 的关键字参数来进行维度匹配检查
    super().__init__(func, arg, sizes, _users=MULTIPLE, dim=KeywordArg("dim"))

    # 执行对节点 `node` 的匹配操作，使用给定的上下文 `ctx`
    m = super()._match(node, ctx)
    if not m:
        return m
    
    # 获取节点 `node` 的第二个参数，即切割的段数 `split_sections`
    split_sections = node.args[1]
    if not isinstance(split_sections, (list, tuple)):
        return FailedMatch("split not normalized")
    
    # 检查所有使用此节点的操作是否都是唯一的 `getitem` 操作
    seen_idxs = set()
    for user in node.users:
        if not CallFunction(operator.getitem, Arg(), Arg()).match(user):
            # 理想情况下不应该发生，切割操作的使用者应始终是 `getitem`
            return FailedMatch(f"user of split not a getitem: {user}")
        if not isinstance(user.args[1], int):
            return FailedMatch("only integer getitems are handled")
        if user.args[1] in seen_idxs:
            return FailedMatch(f"duplicate getitem {user.args[1]}")
        if user.args[-1] < 0:  # type: ignore[operator]
            # 理想情况下不应该发生，索引应被规范为正数
            return FailedMatch("negative index")
        seen_idxs.add(user.args[1])
    
    return m
# 注册图模式，使用 TorchSplit 匹配模式的函数装饰器
@register_graph_pattern(
    # 使用 TorchSplit 匹配模式
    TorchSplit(
        # 使用 CallFunction 匹配操作符的函数调用
        CallFunction(
            operator.getitem,  # 调用 operator 模块的 getitem 函数
            # 使用 TorchSplit 匹配模式的第一个参数
            TorchSplit(
                KeywordArg("first_split_input"),  # 第一个关键字参数 first_split_input
                KeywordArg("first_split_sections"),  # 第二个关键字参数 first_split_sections
            ),
            Ignored(),  # 忽略的第三个参数
        ),
        KeywordArg("next_split_sections"),  # TorchSplit 匹配模式的第二个关键字参数
    ),
    # 使用 construct_pattern_matcher_pass 函数生成模式匹配器的通行证
    pass_dict=construct_pattern_matcher_pass("merge_splits_pass"),
)
# 定义合并分割的函数，接受多个参数，包括模式匹配对象 match 和多个关键字参数
def merge_splits(
    match: Match,  # 匹配对象
    first_split_input: torch.fx.Node,  # 第一个分割输入参数，类型为 torch.fx.Node
    first_split_sections: List[int],  # 第一个分割段列表参数，类型为整数列表
    next_split_sections: List[int],  # 下一个分割段列表参数，类型为整数列表
    # 注意：dim 参数由 TorchSplit 隐式传递，因为它在内部使用了一个带有 dim 的模式
    dim: int,  # 维度参数
):
    # 获取匹配的输出节点
    node = match.output_node()
    
    # 检查节点是否没有使用者，处理极端情况并跳过模式
    if len(node.users.keys()) == 0:
        return
    
    # 获取匹配的图对象
    graph = match.graph
    
    # 获取第一个分割的对象和下一个分割的索引
    first_split = node.args[0].args[0]  # 第一个分割对象
    next_split_index = node.args[0].args[1]  # 下一个分割索引
    
    # 复制第一个分割段列表
    new_split_sections = list(first_split_sections)
    
    # 替换新分割段列表中的特定段
    new_split_sections[next_split_index : next_split_index + 1] = next_split_sections
    
    # 获取第一个分割对象的维度
    first_split_dim = _get_dim(first_split)
    
    # 待移除的项目列表
    to_remove = []
    # 在图中的 first_split 节点之前插入操作
    with graph.inserting_before(first_split):
        # 添加新的分割节点
        new_split = graph.call_function(
            torch.split,  # 调用 torch.split 函数
            args=(first_split_input, new_split_sections),  # 传入参数
            kwargs={"dim": first_split_dim},  # 传入关键字参数
        )
        
        # 创建一个字典，将第一个分割节点的用户编号映射到其用户对象
        first_split_num_to_user = {
            user.args[1]: user for user in first_split.users.keys()  # 获取每个用户对象的第二个参数作为键
        }

        # 初始化新的分割编号为 0
        new_split_num = 0
        # 遍历第一个分割节点的每个分割编号
        for split_num in range(len(first_split_sections)):
            # 如果当前分割编号不在用户映射中
            if split_num not in first_split_num_to_user:
                new_split_num += 1
                continue
            
            # 获取当前分割编号对应的用户对象
            old_getitem = first_split_num_to_user[split_num]
            
            # 如果当前分割编号不等于下一个分割点的索引
            if split_num != next_split_index:
                # 更新用户对象的第一个参数为新的分割节点
                old_getitem.update_arg(0, new_split)
                # 更新用户对象的第二个参数为新的分割编号
                old_getitem.update_arg(1, new_split_num)
                new_split_num += 1
            else:
                # 创建一个字典，将下一个分割点的用户编号映射到其用户对象
                next_split_num_to_user = {
                    user.args[1]: user for user in node.users.keys()
                }
                
                # 不是所有的从分割节点得到的项都是必需的。
                # 我们使用用户数量来检查要合并的项。
                # 遍历下一个分割点的每个分割编号
                for next_split_num in range(len(node.users.keys())):
                    # 在新的分割节点之后插入操作
                    with graph.inserting_after(new_split):
                        # 调用操作符的 getitem 函数，传入参数为新的分割节点和新的分割编号
                        new_getitem = graph.call_function(
                            operator.getitem, args=(new_split, new_split_num)
                        )
                    new_split_num += 1
                    # 获取下一个分割点对应的用户对象
                    next_getitem = next_split_num_to_user[next_split_num]
                    # 更新新的获取项的元数据
                    new_getitem.meta.update(next_getitem.meta)
                    # 用新的获取项替换所有使用下一个获取项的地方
                    next_getitem.replace_all_uses_with(new_getitem)
                    # 将下一个获取项添加到要移除的列表中
                    to_remove.append(next_getitem)
                
                # 将当前节点、旧的获取项添加到要移除的列表中
                to_remove.append(node)
                to_remove.append(old_getitem)

        # 将第一个分割节点添加到要移除的列表中
        to_remove.append(first_split)  # type: ignore[arg-type]
    
    # 遍历要移除的节点列表，从图中擦除这些节点
    for node in to_remove:
        graph.erase_node(node)

    # 增加"inductor"类中"merge_splits_pass"计数器的值
    counters["inductor"]["merge_splits_pass"] += 1
    """
    Helper class to simplify split-cat pattern. In simple cases, both split and cat node can be removed in a "split->cat"
    pattern. However, there are various cases where they can't and we need to simplify split/ add transforms before cat.
    Some such cases are:
        1. Final node has additional args (not coming from the initial split)
        2. Shuffling of args between split/cat
        3. Some final nodes are non-(cat/stack)
        4. Split-dim != cat-dim (but equal split)

    Note that any combination of the above cases can happen.

    To deal with 1, 2, & 3 - we iterate over all users of split. And figure out common "ranges" that can be merged.
    Then, we simplify the split accordingly. In the best case, split can be entirely removed.

    To deal with 4, we add some transformations (unflatten + movedim) (See `get_transform_params`).

    Finally, depending on final node being cat or stack, unsqueeze/flatten needs to be added.
    """

    def simplify(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        split_sections: List[int],
    ):
        # Find the next users (i.e. users after the getitem)
        next_users = find_next_users(split_node)
        # Gather inputs of the next users. When inputs come from `split_node`, they are instead represented by
        # a tuple indicating the split ranges. See `get_user_input_list` for more details
        user_inputs_list = self.get_user_input_list(split_node, next_users)
        # Simplify the split_sections based on user_inputs_list. In simpler cases, len(simplified_split_ranges) == 1 and
        # we can simply replace the split node. Otherwise, we simplify it.
        simplified_split_ranges = self.get_simplified_split_ranges(
            split_sections, next_users, user_inputs_list
        )
        if not simplified_split_ranges:  # Simplification not possible
            return
        transform_params_list = self.get_transform_params(
            split_node, next_users, user_inputs_list
        )
        if not transform_params_list:
            return

        # Start actual replacement
        user_inputs_list_new = self.replace_split(
            graph, split_node, split_sections, user_inputs_list, simplified_split_ranges
        )
        self.replace_cat(
            graph, split_node, next_users, user_inputs_list_new, transform_params_list  # type: ignore[arg-type]
        )
        self.erase_old_nodes(graph, split_node, next_users)  # type: ignore[arg-type]
        counters["inductor"]["unbind_stack_pass"] += 1

    def get_user_input_list(
        self, split_node: torch.fx.Node, next_users: List[torch.fx.Node]
    ):
        """
        Collects input ranges for each user of `split_node`.

        Args:
        - split_node: The node representing the split operation.
        - next_users: List of nodes that use the output of `split_node`.

        Returns:
        - List of tuples representing the input ranges for each user node.
        """
        pass  # Method implementation would go here but is omitted in this example
    ) -> List[List[Union[torch.fx.Node, _Range]]]:
        """
        返回接下来用户节点的输入列表。外部列表表示用户节点，内部列表表示特定节点的输入。该列表可以包含以下内容之一：
          - 表示应合并为cat的get_items范围的元组（闭区间）
          - 表示“其他”输入的torch.fx.Node（不来自我们的split）
        """
        user_inputs_list: List[List[Union[torch.fx.Node, _Range]]] = []
        for user in next_users:
            if user.target in {torch.cat, torch.stack}:
                user_inputs_list.append(self.get_merged_user_inputs(split_node, user))
            else:
                user_inputs_list.append(self.get_non_cat_node_input(split_node, user))  # type: ignore[arg-type]
        return user_inputs_list

    def get_merged_user_inputs(
        self, split_node: torch.fx.Node, cat_node: torch.fx.Node
    ) -> List[Union[torch.fx.Node, _Range]]:
        """
        获取合并后的用户节点输入，以与`get_merged_user_inputs`相同的格式返回
        """
        user_inputs = get_arg_value(cat_node, 0, "tensors")
        simplified_user_inputs = []
        split_users = set(split_node.users.keys())
        for user_input in user_inputs:
            if user_input not in split_users:
                simplified_user_inputs.append(user_input)
            else:
                # 添加cat依赖的“getitem”号码
                simplified_user_inputs.append(user_input.args[1])
        return self.merge_consecutive_inputs(simplified_user_inputs)

    def get_non_cat_node_input(
        self, split_node: torch.fx.Node, node: torch.fx.Node
    ) -> List[_Range]:
        """
        获取与`get_merged_user_inputs`相同格式的非cat节点的输入
        """
        node_input = []
        split_users = set(split_node.users.keys())
        for node_arg in node.all_input_nodes:
            if node_arg in split_users:
                getitem_num = get_arg_value(node_arg, 1)
                node_input.append((getitem_num, getitem_num))
        return node_input

    def merge_consecutive_inputs(
        self, inputs: List[Union[torch.fx.Node, int]]
    ) -> List[List[Union[torch.fx.Node, _Range]]]:
        """
        合并连续的输入，返回一个列表，其中每个元素是一个列表，包含torch.fx.Node或_Range
        """
        merged_inputs: List[List[Union[torch.fx.Node, _Range]]] = []
        current_group: List[Union[torch.fx.Node, _Range]] = []
        for input_item in inputs:
            if isinstance(input_item, torch.fx.Node) or isinstance(input_item, _Range):
                current_group.append(input_item)
            else:
                if current_group:
                    merged_inputs.append(current_group)
                current_group = []
        if current_group:
            merged_inputs.append(current_group)
        return merged_inputs
    ) -> List[Union[torch.fx.Node, _Range]]:
        """
        合并连续的输入到用户节点。

        例如：
        [arg0, 0, 1, 2, arg1] -> [arg0, (0, 2), arg1]
        """
        merged_ranges = []  # 初始化空列表用于存储合并后的范围
        cur_range = None  # 初始化当前范围为 None
        for input_ in inputs:  # 遍历输入列表中的每个元素
            if isinstance(input_, int):  # 如果当前元素是整数
                if not cur_range:  # 如果当前范围为空
                    cur_range = [input_, input_]  # 创建新的范围列表
                elif input_ == cur_range[1] + 1:  # 如果当前整数可以扩展当前范围
                    cur_range[1] += 1  # 扩展当前范围的上限
                else:  # 如果当前整数不能与当前范围合并
                    merged_ranges.append(tuple(cur_range))  # 将当前范围转为元组并添加到合并范围列表
                    cur_range = [input_, input_]  # 创建新的范围列表
            else:  # 如果当前元素不是整数
                if cur_range:  # 如果存在当前范围
                    merged_ranges.append(tuple(cur_range))  # 将当前范围转为元组并添加到合并范围列表
                    cur_range = None  # 清空当前范围
                merged_ranges.append(input_)  # 将当前非整数元素直接添加到合并范围列表（类型标注忽略）
        if cur_range:  # 如果还有剩余的当前范围未处理
            merged_ranges.append(tuple(cur_range))  # 将剩余的当前范围添加到合并范围列表
        return merged_ranges  # 返回合并后的范围列表（类型标注忽略）

    def get_simplified_split_ranges(
        self,
        split_sections,
        next_users,
        user_inputs_list: List[List[Union[torch.fx.Node, _Range]]],
    ) -> Optional[List[_Range]]:
        """
        获取简化后的分割范围。

        将每个用户节点的输入合并后，根据分割节段和用户输入列表生成分割范围。
        """
        ranges = set()  # 使用集合存储所有的范围
        for user_node, user_inputs in zip(next_users, user_inputs_list):  # 遍历每个用户节点及其输入列表
            ranges |= {  # 合并当前用户输入中的所有元组形式的范围
                user_input
                for user_input in user_inputs
                if isinstance(user_input, tuple)
            }
        cumulative_sizes = [0] + torch.cumsum(torch.tensor(split_sections), 0).tolist()  # 计算累积分割节段的大小
        split_ranges = sorted(  # 对合并后的范围进行排序，并根据累积分割节段计算实际分割范围
            [(cumulative_sizes[r[0]], cumulative_sizes[r[1] + 1]) for r in ranges]
        )

        if not self.has_non_overlapping_ranges(  # 检查分割范围是否存在重叠
            split_ranges,
        ):  # 这不是一个严格的条件，但目前保留以保持简单性
            return None  # 如果范围重叠，返回 None

        split_ranges = self.fill_gaps(split_ranges, 0, cumulative_sizes[-1])  # 填补范围间的空隙
        if len(split_sections) == len(split_ranges):  # 如果无法进行进一步简化
            return None  # 返回 None

        counters["inductor"]["scmerge_split_sections_removed"] = len(
            split_sections
        ) - len(split_ranges)  # 更新计数器，记录移除的分割节段数
        return split_ranges  # 返回简化后的分割范围

    def has_non_overlapping_ranges(self, ranges: List[_Range]) -> bool:
        """
        检查范围列表中的范围是否互不重叠。

        如果存在重叠范围，则返回 False，否则返回 True。
        """
        for range_, next_range in zip(ranges, ranges[1:]):  # 遍历范围列表中的每对相邻范围
            if range_[1] > next_range[0]:  # 如果当前范围的结束大于下一个范围的开始
                return False  # 存在重叠范围，返回 False
        return True  # 所有范围均不重叠，返回 True

    def fill_gaps(self, ranges: List[_Range], min_: int, max_: int) -> List[_Range]:
        """
        填补给定范围列表中的空隙，使其覆盖指定的最小值到最大值范围。

        返回填充后的范围列表。
        """
        cur = min_  # 初始化当前位置为最小值
        filled_ranges = []  # 初始化空列表用于存储填充后的范围
        for a, b in ranges:  # 遍历给定范围列表中的每个范围
            if cur < a:  # 如果当前位置小于当前范围的起始值
                filled_ranges.append((cur, a))  # 添加当前位置到当前范围起始值之间的范围
            filled_ranges.append((a, b))  # 添加当前范围到填充范围列表中
            cur = b  # 更新当前位置为当前范围的结束值
        if filled_ranges[-1][1] < max_:  # 如果填充后的范围列表的最后一个范围仍小于最大值
            filled_ranges.append((filled_ranges[-1][1], max_))  # 添加最后一个范围到最大值的范围
        return filled_ranges  # 返回填充后的范围列表
    def get_transform_params(
        self,
        split_node: torch.fx.Node,
        next_users: List[torch.fx.Node],
        user_inputs_list: List[List[Union[torch.fx.Node, _Range]]],
    ) -> Optional[List[List[_TransformParam]]]:
        """
        Figure out what transforms are needed for each input to each cat node.

        We replace a split node with an unflatten followed by a movedim
        """
        # 获取分割维度
        split_dim = _get_dim(split_node)
        # 获取分割的片段数
        split_sections = split_node.args[1]
        # 初始化变换参数列表
        transform_params_list: List[List[_TransformParam]] = []

        # 遍历下游节点和对应的输入列表
        for user_node, user_inputs in zip(next_users, user_inputs_list):
            # 如果用户节点的目标不是 torch.cat 或 torch.stack，则不需要变换
            if user_node.target not in {torch.cat, torch.stack}:
                transform_params_list.append([])
                continue

            # 获取 cat 操作的维度参数
            cat_dim = get_arg_value(user_node, 1, "dim")
            # 初始化变换参数列表
            transform_params: List[_TransformParam] = []

            # 遍历用户输入
            for user_input in user_inputs:
                # 如果分割维度与 cat 操作的维度相同，并且用户节点目标是 torch.cat，则无需变换
                if split_dim == cat_dim and user_node.target == torch.cat:
                    # 不需要变换
                    transform_params.append((None, None, None, None))
                # 如果用户输入是元组，表示正在简化分割
                elif isinstance(user_input, tuple):
                    # 验证分割片段是否相等
                    subset_split_sections = split_sections[  # type: ignore[index]
                        user_input[0] : user_input[1] + 1
                    ]
                    # 所有的片段应该是相等的
                    if len(set(subset_split_sections)) != 1:
                        return None

                    num_splits = len(subset_split_sections)
                    # 设置 unflatten 参数
                    unflatten_params = (split_dim, (num_splits, -1))
                    # 设置 movedim 参数
                    movedim_params = (
                        (split_dim, cat_dim) if split_dim != cat_dim else None
                    )
                    transform_params.append(
                        (unflatten_params, movedim_params, None, None)
                    )
                # 如果用户节点目标是 torch.stack 或者分割维度不等于 cat 维度，需要对输入进行 unsqueeze
                elif (
                    user_node.target == torch.stack or split_dim != cat_dim
                ):
                    transform_params.append((None, None, (cat_dim,), None))
                else:
                    # 非分割输入，不需要变换
                    transform_params.append((None, None, None, None))
            transform_params_list.append(transform_params)
        return transform_params_list

    def replace_split(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        split_sections: List[int],
        user_inputs_list: List[List[Union[torch.fx.Node, _Range]]],
        split_ranges: List[_Range],
    ):
        # 替换分割节点
    ) -> List[List[torch.fx.Node]]:
        """
        Replace the split node. It can either remove the split node if len(split_ranges) == 1, or simplify it
        into a split with lesser sections if len(split_ranges) > 1.

        Returns the new `user_inputs_list`, with tuples replaced with new getitems from the newer split node.
        """
        # 获取分割节点的输入
        split_input = split_node.args[0]
        # 获取分割的维度
        split_dim = _get_dim(split_node)
        
        if len(split_ranges) == 1:  # 如果只有一个分割范围，则完全消除分割节点
            # 创建包含单个输入的列表
            split_items = [split_input]
        else:
            # 在分割节点之后插入新节点
            with graph.inserting_after(split_node):
                # 创建新的分割节点
                new_split = graph.call_function(
                    torch.split,
                    args=(
                        split_input,
                        [r[1] - r[0] for r in split_ranges],
                    ),
                    kwargs={"dim": split_dim},
                )
                # 更新新分割节点的元数据
                new_split.meta.update(split_node.meta)
                # 增加计数器指示分割节点的添加
                counters["inductor"]["scmerge_split_added"] += 1
            
            # 在新分割节点之后插入节点
            with graph.inserting_after(new_split):
                # 创建包含每个分段的新 getitem 调用的列表
                split_items = [
                    graph.call_function(operator.getitem, args=(new_split, i))
                    for i in range(len(split_ranges))
                ]
        
        # 现在分配正确的 getitem 到正确的输入
        cumulative_sizes = [0] + torch.cumsum(torch.tensor(split_sections), 0).tolist()
        new_user_inputs_list = []
        for user_inputs in user_inputs_list:
            new_user_inputs = []
            for user_input in user_inputs:
                if isinstance(user_input, tuple):
                    # 找到正确的新 getitem（在 split_items 中）
                    new_user_inputs.append(
                        split_items[
                            split_ranges.index(
                                (
                                    cumulative_sizes[user_input[0]],
                                    cumulative_sizes[user_input[1] + 1],
                                )
                            )
                        ]
                    )
                else:
                    new_user_inputs.append(user_input)
            new_user_inputs_list.append(new_user_inputs)
        
        # 返回更新后的用户输入列表
        return new_user_inputs_list  # type: ignore[return-value]

    def replace_cat(
        self,
        graph: torch.fx.GraphModule,
        split_node: torch.fx.Node,
        next_users: List[torch.fx.Node],
        user_inputs_list_new,
        transform_params_list: List[List[_TransformParam]],
    ):
        """
        Replace the concatenation node with a new implementation.
        This method is not fully provided here.
        """
        ...

    def erase_old_nodes(
        self,
        graph: torch.fx.GraphModule,
        split_node: torch.fx.Node,
        next_users: List[torch.fx.Node],
        ...
        ):
            # 将当前节点加入待移除列表
            to_remove = [split_node]
            # 增加统计计数器，记录移除分裂节点的次数
            counters["inductor"]["scmerge_split_removed"] += 1
            # 将分裂节点的用户键加入待移除列表
            to_remove.extend(split_node.users.keys())
            # 遍历下一步用户
            for next_user in next_users:
                # 如果下一步用户的目标不是 torch.cat 或 torch.stack，则跳过
                if next_user.target not in {torch.cat, torch.stack}:
                    continue
                # 增加统计计数器，记录移除 torch.cat 或 torch.stack 的次数
                counters["inductor"]["scmerge_cat_removed"] += 1
                # 将下一步用户加入待移除列表
                to_remove.append(next_user)
            # 反向遍历待移除列表中的节点并从图中擦除
            for node in reversed(to_remove):
                graph.erase_node(node)
class UnbindCatRemover(SplitCatSimplifier):
    """
    Helper class to merge Unbind->Cat/Stack. Many of the cases are similar to SplitCatSimplifier.

    Unbind can't be simplified like splits. So, we can only remove the unbind node. Other than this,
    other cases like multiple users, additional args, dim mismatch are similar to `SplitCatSimplifier`,
    hence we extend that class.
    """

    def remove_unbind(
        self,
        graph: torch.fx.Graph,
        unbind_node: torch.fx.Node,
    ):
        # 计算所有使用 unbind_node 的 getitem_node 中第二个参数的最大值并加1，得到 num_unbind
        num_unbind = (
            max(getitem_node.args[1] for getitem_node in unbind_node.users.keys()) + 1
        )
        # 创建一个长度为 num_unbind 的列表，每个元素为1，代表分割的段数
        split_sections = [1 for _ in range(num_unbind)]

        # 调用父类的 simplify 方法，简化 graph 中的 unbind_node，使用 split_sections
        super().simplify(graph, unbind_node, split_sections)

    def get_simplified_split_ranges(
        self,
        split_sections: List[int],
        next_users: List[torch.fx.Node],
        user_inputs_list: List[List[Union[torch.fx.Node, _Range]]],
    ) -> Optional[List[_Range]]:
        # 调用父类的方法获取简化后的分割范围
        simplified_split_ranges = super().get_simplified_split_ranges(
            split_sections, next_users, user_inputs_list
        )
        # 如果获取的简化分割范围为空或者长度不为1，则返回 None
        if not simplified_split_ranges or len(simplified_split_ranges) != 1:
            return None
        # 否则返回获取到的简化分割范围
        return simplified_split_ranges

    def get_transform_params(
        self,
        split_node: torch.fx.Node,
        next_users: List[torch.fx.Node],
        user_inputs_list: List[List[Union[torch.fx.Node, _Range]]],
    ):
        # 这个方法未完整，但假设要返回一些变换参数或处理逻辑
        pass
    ) -> Optional[List[List[_TransformParam]]]:
        """
        Figure out what transforms are needed for each input to each cat node.

        Here is the rough transforms we apply:

        x -> unbind -> stack => x -> movedim

        x -> unbind -> cat => x -> movedim -> flatten

        When cat/stack nodes have additional args:

             addn ---|              addn -> unsqueeze ---|
        x -> unbind -> stack  =>           x -> movedim  -> cat

             addn ---|                            addn ---|
        x -> unbind -> cat  =>   x -> movedim -> flatten  -> cat

        (Note application of these depends on the dims as well)


        """
        # 获取分离节点的维度
        split_dim = _get_dim(split_node)
        # 初始化存储转换参数的列表
        transform_params_list: List[List[_TransformParam]] = []
        # 遍历每个用户节点及其输入列表
        for user_node, user_inputs in zip(next_users, user_inputs_list):
            # 获取 cat 维度，如果未指定则默认为 0
            cat_dim = get_arg_value(user_node, 1, "dim") or 0
            # 初始化存储当前用户节点转换参数的列表
            transform_params: List[_TransformParam] = []
            # 遍历用户节点的输入
            for user_input in user_inputs:
                if isinstance(user_input, tuple):
                    # 当用户输入来自 unbind 函数时
                    # 确定 movedim 的参数
                    movedim_params = (
                        (split_dim, cat_dim) if split_dim != cat_dim else None
                    )
                    # 初始化 flatten 的参数为 None
                    flatten_params = None
                    # 如果用户节点的目标函数是 torch.cat
                    if user_node.target == torch.cat:
                        # 确定 flatten 的参数
                        flatten_params = (cat_dim, cat_dim + 1)
                    # 将当前转换参数添加到列表中
                    transform_params.append(
                        (None, movedim_params, None, flatten_params)
                    )
                elif (
                    user_node.target == torch.stack
                ):  # 对于不通过 unbind 的输入，需要对其进行 unsqueeze 处理后再进行 cat
                    # 添加 unsqueeze 的参数
                    transform_params.append((None, None, (cat_dim,), None))
                else:
                    # 对于非 unbind 输入，所有参数均为 None
                    transform_params.append((None, None, None, None))
            # 将当前用户节点的转换参数列表添加到总列表中
            transform_params_list.append(transform_params)
        # 返回所有用户节点的转换参数列表
        return transform_params_list
class GetItem(CallFunction):
    # GetItem 类继承自 CallFunction 类
    def __init__(self, arg, index, _users=1):
        # 初始化方法，调用父类 CallFunction 的构造函数，设置实例的属性
        super().__init__(operator.getitem, arg, index, _users=_users)

    def find_anchor_nodes(self, ctx: MatchContext, searched: Set[torch.fx.Node]):
        # 查找锚节点的方法，重写以不使用 ctx.pattern_to_node
        for pattern in self.flat_args_kwargs[0]:
            # 遍历参数列表的第一个元素
            if isinstance(pattern, PatternExpr):
                # 如果是 PatternExpr 类型的对象
                for other_node in pattern.find_anchor_nodes(ctx, searched):
                    # 递归查找其他节点的锚节点
                    if not isinstance(other_node, torch.fx.Node):
                        continue
                    for node in other_node.users:
                        # 遍历其他节点的用户节点
                        if node not in searched:
                            if self._match_fns(node):
                                # 如果匹配到了函数节点
                                yield node
                                searched.add(node)


@register_graph_pattern(
    RepeatedExpr(
        CallFunction(
            torch.squeeze,
            GetItem(
                TorchSplit(
                    KeywordArg("split_input"),
                    KeywordArg("split_sizes"),
                ),
                Ignored(),
            ),
            KeywordArg("dim"),
            _users=MULTIPLE,
        ),
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),
)
# 注册图模式匹配器，用于匹配重复的表达式结构
@register_graph_pattern(
    RepeatedExpr(
        CallFunction(
            torch.squeeze,
            GetItem(
                TorchSplit(
                    KeywordArg("split_input"),
                    KeywordArg("split_sizes"),
                ),
                Ignored(),
            ),
            dim=KeywordArg("dim"),
            _users=MULTIPLE,
        )
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),
)
# 注册另一个图模式匹配器，用于匹配重复的表达式结构
def merge_split_squeeze(
    match: Match, split_input: torch.fx.Node, split_sizes: List[int], dim: int
):
    # 合并、分割和压缩操作的函数定义
    graph = match.graph
    split = next(node for node in match.nodes if node.target == torch.split)
    # 查找目标为 torch.split 的节点
    if not all(s == 1 for s in split_sizes):
        return
    if isinstance(dim, Sequence):
        return
    # 如果维度参数是序列类型，则返回
    next_users = find_next_users(split)
    # 查找下一个用户节点
    if not all(node.target == torch.squeeze for node in next_users):
        return
    # 如果下一个用户节点的目标不是 torch.squeeze，则返回
    # 在图中的某个节点之前插入新节点
    with graph.inserting_before(match.output_node()):
        # 调用 torch.unbind 函数，解除 split_input 在指定维度 dim 上的绑定
        unbind = graph.call_function(
            torch.unbind, args=(split_input,), kwargs={"dim": dim}
        )
        # 遍历按照 item_index 排序的 getitem_node 列表
        for item_index, getitem_node in sorted(
            [
                (getitem_node.args[1], getitem_node)
                for getitem_node in split.users.keys()
            ]
        ):
            # 获取 getitem_node 的直接使用节点（squeeze）
            squeeze = next(iter(getitem_node.users.keys()))
            # 创建新的 getitem 节点，从 unbind 中获取指定 item_index 的元素
            new_get_item = graph.call_function(
                operator.getitem, args=(unbind, item_index)
            )
            # 将 squeeze 的所有使用替换为 new_get_item
            squeeze.replace_all_uses_with(new_get_item)
            # 更新 new_get_item 的元数据
            new_get_item.meta.update(squeeze.meta)
            # 在图中删除 squeeze 节点和 getitem_node 节点
            graph.erase_node(squeeze)
            graph.erase_node(getitem_node)
    # 在图中删除 split 节点
    graph.erase_node(split)
    # 增加 "inductor" 分类器中 "split_cat_pass" 计数器的计数值
    counters["inductor"]["split_cat_pass"] += 1
# 创建一个ListOf对象，包含一个GetItem对象
getitem_unbind = ListOf(
    # GetItem对象调用torch.unbind函数
    GetItem(
        # 调用torch.unbind函数，使用关键字参数'unbind_input'，维度为关键字参数'dim'
        CallFunction(
            torch.unbind,
            KeywordArg("unbind_input"),  # 关键字参数'unbind_input'
            dim=KeywordArg("dim"),       # 关键字参数'dim'
            _users=MULTIPLE,             # _users参数设为MULTIPLE
        ),
        Ignored(),                      # 忽略的参数
        _users=MULTIPLE,                # _users参数设为MULTIPLE
    ),
    partial=True,                       # 设定partial参数为True
)

# 注册一个图模式，匹配CallFunction调用torch.stack或torch.cat，使用getitem_unbind作为参数
@register_graph_pattern(
    CallFunction([torch.stack, torch.cat], getitem_unbind, Ignored(), _users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("unbind_stack_pass"),  # 传递pass_dict参数
)

# 注册一个图模式，匹配CallFunction调用torch.stack或torch.cat，使用getitem_unbind作为参数，且有一个忽略的维度参数
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], getitem_unbind, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=construct_pattern_matcher_pass("unbind_stack_pass"),  # 传递pass_dict参数
)

# 注册一个图模式，匹配CallFunction调用torch.stack或torch.cat，使用getitem_unbind作为参数，且有一个忽略的维度参数和忽略的张量参数
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], tensors=getitem_unbind, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=construct_pattern_matcher_pass("unbind_stack_pass"),  # 传递pass_dict参数
)
def merge_unbind_stack(match: Match, unbind_input: torch.fx.Node, dim: int):
    # 在匹配对象中查找目标为torch.unbind的节点，然后移除它
    unbind_node = next(node for node in match.nodes if node.target == torch.unbind)
    UnbindCatRemover().remove_unbind(match.graph, unbind_node)


# 创建一个ListOf对象，包含一个CallFunction对象
getitem_split = ListOf(
    CallFunction(
        operator.getitem,
        # 调用torch.split函数，使用关键字参数'split_sections'
        TorchSplit(
            Ignored(),                      # 忽略的参数
            KeywordArg("split_sections"),   # 关键字参数'split_sections'
        ),
        Ignored(),                          # 忽略的参数
        _users=MULTIPLE,                    # _users参数设为MULTIPLE
    ),
    partial=True,                           # 设定partial参数为True
)

# 注册一个图模式，匹配CallFunction调用torch.stack或torch.cat，使用getitem_split作为参数，且有一个忽略的维度参数
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat],
        tensors=getitem_split,
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),  # 传递pass_dict参数
)

# 注册一个图模式，匹配CallFunction调用torch.stack或torch.cat，使用getitem_split作为参数，且有一个忽略的维度参数
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat],
        getitem_split,
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),  # 传递pass_dict参数
)

# 注册一个图模式，匹配CallFunction调用torch.stack或torch.cat，使用getitem_split作为参数，且有一个忽略的参数
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat],
        getitem_split,
        Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=construct_pattern_matcher_pass("split_cat_pass"),  # 传递pass_dict参数
)
def simplify_split_cat(match: Match, split_sections: List[int], dim: int):
    # 如果split_sections不是列表或元组，则返回
    if not isinstance(split_sections, (list, tuple)):
        return
    # 在匹配对象中查找目标为torch.split的节点，然后简化它
    split_node = next(node for node in match.nodes if node.target == torch.split)
    SplitCatSimplifier().simplify(match.graph, split_node, split_sections)


# noqa: W605
# ############pattern to be optimized is#########

#                 split_node(dim=1)
#       /     \         ...       /         \
# getitem    getitem          getitem     getitem   -> user=1
#    \       /                     \       /
#      cat (user=mul, dim=1)           cat(user=mul, dim=1)
#       |            \                   |          \

# ################after transformation#############

#                 split_node(dim=1)
#       /              ...                  \
#     getitem                             getitem
#     |    \                              |     \
    # 检查节点的输入节点是否来自同一父节点
    prev_node = None
    # 遍历节点的第一个参数（假定是可迭代对象）
    for getitem in node.args[0]:  # type: ignore[union-attr]
        # 如果节点的目标不是 operator.getitem，则返回 False
        if getitem.target != operator.getitem:  # type: ignore[union-attr]
            return False
        # 如果是第一个节点，记录为 prev_node
        if prev_node is None:
            prev_node = getitem.args[0]  # type: ignore[union-attr]
        else:
            # 如果当前节点的输入节点不等于之前记录的 prev_node，则返回 False
            if getitem.args[0] != prev_node:
                return False
    # 如果所有节点的输入节点都来自同一父节点，则返回 True
    return True
def remove_zeros(split_sections: List[int]):
    """
    从列表中移除零并生成从原拆分节点到新拆分节点的索引映射字典
    """
    new_split_sections, index_mapping = [], {}  # 初始化新的拆分节点列表和索引映射字典
    idx = 0  # 初始化索引计数器
    for i in range(len(split_sections)):
        if split_sections[i] > 0:  # 如果拆分节点大于零
            new_split_sections.append(split_sections[i])  # 将非零拆分节点添加到新列表中
            index_mapping[i] = idx  # 建立原拆分节点到新拆分节点的索引映射
            idx += 1  # 索引计数器增加

    return new_split_sections, index_mapping  # 返回处理后的新拆分节点列表和索引映射字典


def is_sorted_and_consecutive(arr: List[int]) -> bool:
    """
    检查数组是否已排序且连续
    """
    if arr == sorted(arr):  # 如果数组已经排序
        # 检查相邻元素之间的差是否均为1
        return all(x[1] - x[0] == 1 for x in zip(arr, arr[1:]))
    else:
        return False  # 数组未排序或不连续


def calculate_fused_tensor_size(split_node: torch.fx.Node, indices: List[int]) -> int:
    """
    计算在给定索引处融合张量的大小
    """
    fused_tensor_size = 0  # 初始化融合张量的大小
    for i in range(len(split_node.args[1])):  # 遍历拆分节点的第二个参数的长度
        if i in indices:  # 如果索引在给定的索引列表中
            fused_tensor_size += split_node.args[1][i]  # 累加融合张量的大小
    return fused_tensor_size  # 返回融合张量的大小


@register_graph_pattern(
    CallFunction(
        torch.cat,
        getitem_split,
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=construct_pattern_matcher_pass("merge_getitem_cat_pass"),
)
def merge_getitem_cat(match: Match, split_sections: List[int], dim: int):
    """
    合并操作：在匹配的图中查找并优化 torch.cat 和 getitem 模式
    """
    if not isinstance(split_sections, (list, tuple)):  # 如果拆分节点不是列表或元组类型，即未归一化的拆分
        return
    graph = match.graph  # 获取匹配对象的图
    split_node = next(node for node in match.nodes if node.target == torch.split)  # 查找匹配中的拆分节点
    split_input, split_size, split_dim = _get_split_args_default(split_node)  # 获取拆分节点的输入、大小和维度信息
    # 如果 cat 和 split 操作的维度不同，直接返回
    next_users = find_next_users(split_node)  # 查找拆分节点之后的用户节点
    split_sections = list(split_sections)  # 创建拆分节点的副本以避免修改不可变列表
    # 如果split_sections不是list或tuple类型，表示未规范化的分割方式，直接返回
    if not isinstance(split_sections, (list, tuple)):  # Unnormalized split
        return
    
    # 获取匹配对象中的图(graph)
    graph = match.graph
    
    # 从匹配的节点中找到目标为torch.split的节点
    split_node = next(node for node in match.nodes if node.target == torch.split)
    
    # 调用函数获取torch.split的输入参数：split_input, split_size, split_dim
    split_input, split_size, split_dim = _get_split_args_default(split_node)
    
    # 如果cat操作和split操作的维度不同，则直接返回
    # Find the next users (i.e. users after the getitem)
    # 找到在getitem操作之后的下一个用户节点
    next_users = find_next_users(split_node)
    for cat_user in next_users:
        # 遍历每个使用 torch.cat 的用户节点
        if cat_user.target == torch.cat:
            # 获取 torch.cat 函数调用的维度参数，如果未指定则默认为0
            cat_dim = get_arg_value(cat_user, 1, "dim") or 0
            # 检查该 cat_user 的目标维度与 split_dim 是否相同，并且它们是否来自同一个父节点
            if split_dim != cat_dim or not has_same_parent_node(cat_user):
                # 如果不符合条件则跳过当前循环
                continue
            # 初始化索引列表和索引到 getitem 映射的字典
            indices, idx_to_getitem = [], {}
            # 遍历 cat_user 的第一个参数中的每个 getitem
            for getitem in cat_user.args[0]:  # type: ignore[union-attr]
                # 将每个 getitem 的索引添加到 indices 列表中
                indices.append(getitem.args[1])  # type: ignore[union-attr]
                # 将索引与 getitem 对象的映射保存到 idx_to_getitem 字典中
                idx_to_getitem[getitem.args[1]] = getitem  # type: ignore[union-attr]
            # 检查 indices 是否已排序且连续，以确保合并的 getitem 是连续的
            if not is_sorted_and_consecutive(indices):
                # 如果不连续则跳过当前循环
                continue
            # 情况1：cat 使用了 split 的所有 getitem
            if len(split_sections) == len(cat_user.args[0]):  # type: ignore[arg-type]
                # 替换所有使用 cat 节点结果的节点为 split 节点的输入
                cat_user.replace_all_uses_with(split_node.args[0])
                # 删除 cat 节点
                graph.erase_node(cat_user)
                # 增加统计计数器指示成功执行的次数
                counters["inductor"]["mutate_cat_pass"] += 1
            # 情况2：cat 使用了部分 split 的 getitem
            elif is_node_meta_valid(split_node.args[0]):  # type: ignore[arg-type]
                # 计算融合张量切片的起始大小
                start_fused_size = calculate_fused_tensor_size(
                    split_node, list(range(indices[0]))
                )
                # 计算融合张量切片的结束大小
                end_fused_size = start_fused_size + calculate_fused_tensor_size(
                    split_node, indices
                )
                # 构建切片列表，用于创建新的切片节点
                slice_list = []
                for i in range(len(split_node.args[0].meta["example_value"].shape)):  # type: ignore[union-attr]
                    if i != split_dim:
                        slice_list.append(slice(None, None, None))
                    else:
                        slice_list.append(slice(start_fused_size, end_fused_size, None))
                # 在 split_node 后插入新节点，执行切片操作
                with graph.inserting_after(split_node):
                    slice_node = graph.call_function(
                        operator.getitem,
                        args=(split_node.args[0], tuple(slice_list)),
                    )
                    # 替换所有使用 cat 结果的节点为新创建的切片节点
                    cat_user.replace_all_uses_with(slice_node)
                    # 更新新创建节点的元数据信息
                    slice_node.meta.update(cat_user.meta)

                # 删除原始的 cat 节点
                graph.erase_node(cat_user)
                # 增加统计计数器指示成功执行的次数
                counters["inductor"]["mutate_cat_pass"] += 1
@register_graph_pattern(
    CallFunction(
        torch.tanh,
        CallFunction(
            torch.stack,
            getitem_split,
            dim=Ignored(),
        ),
    ),
    pass_dict=construct_pattern_matcher_pass("merge_stack_tahn_unbind_pass"),
)
@register_graph_pattern(
    CallFunction(
        torch.tanh,
        CallFunction(
            torch.stack,
            tensors=getitem_split,
            dim=Ignored(),
        ),
    ),
    pass_dict=construct_pattern_matcher_pass("merge_stack_tahn_unbind_pass"),
)
@register_graph_pattern(
    CallFunction(
        torch.tanh,
        CallFunction(
            torch.stack,
            getitem_split,
            Ignored(),
        ),
    ),
    pass_dict=construct_pattern_matcher_pass("merge_stack_tahn_unbind_pass"),
)
# 定义函数 merge_stack_tahn_unbind，用于合并特定模式的计算图节点
def merge_stack_tahn_unbind(match: Match, split_sections: List[int], dim: int):
    if not isinstance(split_sections, (list, tuple)):  # 如果分割位置未被标准化，返回
        return
    graph = match.graph
    # 查找匹配中目标为 torch.split 的节点，即分割节点
    split_node = next(node for node in match.nodes if node.target == torch.split)
    # 获取 torch.split 函数的输入参数
    split_input, split_size, split_dim = _get_split_args_default(split_node)
    # 查找分割节点之后的下一个用户节点
    next_users = find_next_users(split_node)
    # 'immutable_list' 对象不支持修改，创建其副本
    split_sections = list(split_sections)


@register_graph_pattern(
    CallFunctionVarArgs(torch.ops.aten.cat.default, users=MULTIPLE),
    pass_dict=construct_pattern_matcher_pass("normalization_aten_pass"),
)
# 定义函数 normalize_cat_default_aten，用于规范化 torch.cat 默认操作
def normalize_cat_default_aten(match: Match, *args, **kwargs):
    cat_node = match.nodes[0]
    graph = match.graph
    # 获取 torch.cat 函数的输入参数 tensors
    tensors = get_arg_value(cat_node, 0, "tensors")
    # 获取 torch.cat 函数的输入参数 dim
    cat_dim = get_arg_value(cat_node, 1, "dim")
    # 如果未指定 dim，尝试从关键字参数 axis 中获取
    if cat_dim is None:
        cat_axis = cat_node.kwargs.get("axis")
        if cat_axis is not None:
            cat_dim = cat_axis
        else:
            cat_dim = 0
    # 如果 tensors 或者 cat_dim 为 None，则记录警告并返回
    if tensors is None or cat_dim is None:
        log.info("couldn't find cat args")
        return
    # 确保 tensors 是列表或元组类型
    assert isinstance(tensors, (list, tuple))
    # 检查每个 tensor 是否具有 "val" 属性
    for tensor in itertools.chain([cat_node], tensors):
        if "val" not in tensor.meta:
            log.warning("val absent for node: %s", tensor)
            return
    # 获取 cat_node 的张量维度信息
    ndim = cat_node.meta["val"].dim()
    def is_empty_tensor(x: torch.fx.Node) -> bool:
        # 判断是否是空张量的特殊情况，这里 torch.ops.aten.cat.default 支持对空张量进行连接
        x_shape = x.meta["val"].shape
        return len(x_shape) == 1 and x_shape[0] == 0

    # 确保所有张量的维度与指定的维度相等，或者是空张量
    assert all(ndim == x.meta["val"].dim() or is_empty_tensor(x) for x in tensors)

    if cat_dim < 0:  # 规范化连接维度
        cat_dim += ndim

    # 在 cat_node 后面插入新节点
    with graph.inserting_after(cat_node):
        # 创建一个新的函数调用节点，调用 torch.ops.aten.cat.default 函数
        new_cat_node = graph.call_function(
            torch.ops.aten.cat.default,
            args=(tensors,),
            kwargs={"dim": cat_dim},
        )
    # 用新的 cat 节点替换所有原来的 cat_node 的使用
    cat_node.replace_all_uses_with(new_cat_node)
    # 更新新 cat 节点的元数据
    new_cat_node.meta.update(cat_node.meta)
    # 从图中删除原始的 cat_node
    graph.erase_node(cat_node)
    # 增加计数器，记录执行了 normalization_aten_pass 正规化操作
    counters["inductor"]["normalization_aten_pass"] += 1
@register_graph_pattern(
    CallFunction(
        torch.ops.aten.cat,  # 注册一个图模式，匹配 torch.ops.aten.cat 调用
        ListOf(CallFunctionVarArgs(torch.ops.aten.unsqueeze)),  # 输入参数为 unsqueeze 操作的列表
        _users=MULTIPLE,  # 匹配多个用户
    ),
    pass_dict=construct_pattern_matcher_pass("unbind_stack_aten_pass"),  # 使用构造的模式匹配器传递字典参数
)
def merge_unbind_stack_aten(match: Match, *args, **kwargs):
    node = match.nodes[-1]  # 获取匹配中的最后一个节点
    graph = match.graph  # 获取匹配的图

    # pyre-fixme[6]
    unsqueeze_nodes = list(node.args[0])  # type: ignore[arg-type]  # 获取节点的第一个参数作为 unsqueeze 节点列表

    cat_dim = get_arg_value(node, 1, "dim")  # 获取节点的第二个参数作为 cat 操作的维度参数

    # 检查 unsqueeze 节点是否来自 select 节点
    if not all(
        get_arg_value(unsqueeze_node, 0, "input").target == torch.ops.aten.select
        for unsqueeze_node in unsqueeze_nodes
    ):
        return

    # 获取所有 select 节点
    select_nodes = [
        get_arg_value(unsqueeze_node, 0, "input") for unsqueeze_node in unsqueeze_nodes
    ]

    parent_of_select_node = get_arg_value(select_nodes[0], 0, "input")  # 获取 select 节点的父节点

    # 检查所有 select 节点的目标是否都是 torch.ops.aten.select
    if not all(
        select_node.target == torch.ops.aten.select for select_node in select_nodes
    ):
        return

    # 检查所有 select 节点是否来自同一个父节点
    if not all(
        get_arg_value(select_node, 0, "input") == parent_of_select_node
        for select_node in select_nodes
    ):
        return

    # 检查 unsqueeze 节点和 select 节点的数量是否一致
    if len(unsqueeze_nodes) != len(select_nodes):
        return

    # 检查所有 select 节点的维度参数是否都与 cat 操作的维度参数相同
    if not all(
        get_arg_value(select_node, 1, "dim") == cat_dim for select_node in select_nodes
    ):
        return

    # 检查所有 select 节点的索引是否连续且从0开始
    if get_arg_value(select_nodes[0], 2, "index") != 0 or not is_sorted_and_consecutive(
        [get_arg_value(select_node, 2, "index") for select_node in select_nodes]
    ):
        return

    # 检查 select 节点的父节点的用户数是否与 unsqueeze 节点列表长度相同
    if len(parent_of_select_node.users.keys()) != len(node.args[0]):  # type: ignore[arg-type]
        return

    # 替换节点的所有使用为其父节点
    node.replace_all_uses_with(parent_of_select_node)

    # 从图中删除节点和相关节点
    graph.erase_node(node)
    for unsqueeze_node in unsqueeze_nodes:
        graph.erase_node(unsqueeze_node)
    for select_node in select_nodes:
        if len(select_node.users) == 0:
            graph.erase_node(select_node)

    counters["inductor"]["unbind_stack_aten_pass"] += 1  # 增加统计计数器
```