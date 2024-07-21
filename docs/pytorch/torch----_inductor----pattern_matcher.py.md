# `.\pytorch\torch\_inductor\pattern_matcher.py`

```py
"""
# Inductor Pattern Matcher

The pattern matcher enables search/replace within an FX graph.

The main entrypoint to the pattern matcher is register_replacement(). Given a
search function and a replacement function this will register a replacement with
a pass (such as torch._inductor.fx_passes.joint_graph.patterns).

Internally the pattern matcher represents patterns as a graph (a DAG). Creating
new patterns manually as a graph is cumbersome and error-prone so the standard
way to create patterns (using register_replacement()) is to provide a search
function and a replacement function which is traced and converted into a graph.

Because the search functions are built somewhat generic (they tend to ignore
tensor sizes, for example) register_replacement() allows you to specify an
`extra_check` function which performs additional checks to verify that the
matched pattern fully matches before returning it.

## Precompiled Patterns

New patterns are added using register_replacement(). Patterns added in this way
can have a compile-time overhead because they need to be traced before
use. Patterns can be precompiled and added using gen_register_replacement()
instead. To do this you call gen_register_replacement() instead of
register_replacement(). The arguments are the same except for an additional
unique name which is used as a lookup key.

## Internals

The match DAG is represented by a graph of `PatternExpr` nodes. Each PatternExpr
implements a `_match` method which returns either a `Match` object for a
successful match or a `FailedMatch` object for a failure to match.
"""

# mypy: disallow-untyped-defs

from __future__ import annotations

# 引入 contextlib 模块，提供了上下文管理工具的支持
import contextlib

# 引入 dataclasses 模块，用于创建与处理数据类
import dataclasses

# 引入 functools 模块，提供了操作函数和可调用对象的功能
import functools

# 引入 importlib 模块，用于动态加载模块
import importlib

# 引入 inspect 模块，提供了函数和类的内省能力
import inspect

# 引入 itertools 模块，提供了操作迭代器的函数
import itertools

# 引入 logging 模块，用于记录日志信息
import logging

# 引入 operator 模块，提供了操作符和函数的功能
import operator

# 引入 os 模块，提供了与操作系统交互的功能
import os

# 引入 re 模块，提供了正则表达式的支持
import re

# 引入 textwrap 模块，提供了文本格式化和填充的功能
import textwrap

# 引入 typing 模块，用于类型提示
import typing

# 从 abc 模块中引入 ABC 和 abstractmethod 用于定义抽象基类和抽象方法
from abc import ABC, abstractmethod

# 从 collections 模块中引入 defaultdict 用于创建默认值为列表的字典
from collections import defaultdict

# 从 pathlib 模块中引入 Path 类，用于处理文件路径
from pathlib import Path

# 引入 typing 中的类型提示
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# 从 typing_extensions 中引入 Self 和 TypeGuard，用于类型提示
from typing_extensions import Self, TypeGuard

# 引入 torch 模块，主要深度学习框架
import torch

# 引入 torch._guards 模块，用于管理与保护内部状态
import torch._guards

# 引入 torch.fx 模块，提供了对 FX 图的支持
import torch.fx

# 引入 torch.utils._pytree 模块，处理 PyTree 数据结构
import torch.utils._pytree as pytree

# 从 torch._dispatch.python 模块中引入 enable_python_dispatcher 函数
from torch._dispatch.python import enable_python_dispatcher

# 引入 torch._dynamo.utils 模块中的 counters 函数，处理计数器
from torch._dynamo.utils import counters

# 引入 torch._inductor.config 模块中的 trace 函数，用于配置追踪
from torch._inductor.config import trace as trace_config

# 从 torch._prims_common 模块中引入 is_integer_dtype 函数，判断是否为整数类型
from torch._prims_common import is_integer_dtype

# 引入 torch.fx.experimental.proxy_tensor 模块，用于创建代理张量
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode

# 引入 torch.fx.experimental.symbolic_shapes 模块，用于符号形状处理
from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

# 引入 torch.fx.immutable_collections 模块，提供不可变集合的支持
from torch.fx.immutable_collections import immutable_dict, immutable_list

# 从 torch.fx.passes.graph_transform_observer 模块中引入 GraphTransformObserver 类
from torch.fx.passes.graph_transform_observer import GraphTransformObserver

# 从 .._functorch 模块中引入 config 配置
from .._functorch import config as functorch_config

# 从 .._functorch.aot_autograd 模块中引入 aot_function 和 make_boxed_func 函数
from .._functorch.aot_autograd import aot_function, make_boxed_func
# 导入所需模块
from .._functorch.partitioners import default_partition
from .._subclasses import FakeTensorMode
from ..fx import Transformer
from . import config
from .decomposition import select_decomp_table
from .lowering import fallback_node_due_to_unsupported_type

# 导入日志模块并获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 导入torch的ops模块中的aten和prims函数
aten = torch.ops.aten
prims = torch.ops.prims

# 定义类型别名
Constant = Any
NodeOrConstant = Union[Constant, torch.fx.Node]

# 定义函数类型协议SearchFn，包含一个字符串名称和__call__方法
class SearchFn(Protocol):
    __name__: str
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

# 定义函数类型协议ReplaceFn，包含一个__call__方法
class ReplaceFn(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

# 定义函数类型协议TraceFn，接受一个Union类型的fn参数并返回torch.fx.GraphModule对象
class TraceFn(Protocol):
    def __call__(
        self, fn: Union[SearchFn, ReplaceFn], *args: Any, **kwargs: Any
    ) -> torch.fx.GraphModule:
        ...

# 定义类型变量T
T = TypeVar("T")

# 定义FnsType类型别名，可以是torch.fx.node.Target或字符串类型
FnsType = Union[torch.fx.node.Target, str]

# 定义Multiple类，用作单例模式的标识
class Multiple:
    def __init__(self) -> None:
        # 确保真的是单例
        assert "MULTIPLE" not in globals() or self is MULTIPLE

# Sentinel常量，表示多个匹配项
MULTIPLE = Multiple()

# 定义Match类，表示成功匹配的模式
class Match:
    """
    Represents a successfully matched pattern.

    The `Match` object is returned to represent a successfully matched
    pattern. Included in the Match are the pattern that was matched, the graph
    nodes matched, and any args that were used during the matching.

    The args and kwargs are specific to the type of pattern that was matched and
    provide hints about what was matched.
    """

    def __init__(
        self,
        ctx: MatchContext,
        pattern: PatternExpr,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.pattern = pattern  # 匹配的模式表达式
        self.args = list(args or [])  # 用于匹配的参数列表
        self.kwargs = kwargs or {}  # 用于匹配的关键字参数
        self.nodes = []  # 匹配的图节点列表
        self.targets = {}  # 将CallFunction映射到node.target
        self.ctx = ctx  # 匹配的上下文环境
        self.replacement_graph = None  # 替换图，可选

    @property
    def graph(self) -> torch.fx.Graph:
        return self.ctx.graph  # 返回匹配的图对象

    def extend(self, other: Match) -> None:
        """
        Extend the current match with another match.

        Args:
            other (Match): Another Match object to extend with.

        Raises:
            FailedMatch: If there is a mismatch in kwargs between the two matches.
        """
        if self.kwargs:
            # 检查关键字参数是否匹配
            for key in set(self.kwargs.keys()) & set(other.kwargs.keys()):
                if self.kwargs[key] != other.kwargs[key]:
                    raise FailedMatch("kwarg mismatch: {}", key)
        self.args.extend(other.args)  # 扩展参数列表
        self.nodes.extend(other.nodes)  # 扩展节点列表
        self.kwargs.update(other.kwargs)  # 更新关键字参数
        self.targets.update(other.targets)  # 更新目标映射
    # 将参数用额外的列表包裹一层
    def bundle(self) -> Match:
        self.args = [tuple(self.args)] if self.args else []  # 如果参数存在，则将其转换为元组并放入列表中；否则置空列表
        return self  # 返回当前实例本身

    # 返回描述对象的字符串表示，包括参数和关键字参数
    def __repr__(self) -> str:
        return f"Match(..., {self.args}, {self.kwargs})"  # 返回带有参数和关键字参数的字符串表示形式

    # 从图中删除未使用且未被标记删除的节点
    def erase_nodes(self, graph: torch.fx.Graph) -> None:
        for n in reversed(self.nodes):  # 对于节点列表中的每个节点（反向遍历）
            if not n._erased and not n.users:  # 如果节点未被标记删除且没有使用者
                graph.erase_node(n)  # 在图中擦除该节点

    # 返回输出节点的列表，与输出模式映射，如果模式为空则返回None
    def output_nodes(self) -> List[Optional[torch.fx.Node]]:
        return [
            (self.ctx.pattern_to_node[p] if p is not None else None)  # 映射每个输出模式到相应的节点，如果模式为空则映射为None
            for p in self.ctx.outputs  # 对于上下文中的每个输出模式
        ]

    # 返回非空的输出节点，否则引发异常
    def output_node(self) -> torch.fx.Node:
        return next(p for p in self.output_nodes() if p)  # 返回第一个非空的输出节点，如果没有则引发异常

    # 使用替换图形替换当前模式条目
    def replace_with_graph(
        self, replacement_graph: torch.fx.Graph, args: Sequence[Any]
    ) -> None:
        ReplacementPatternEntry.replace_with_graph(  # 使用替换模式条目的静态方法
            self, self.ctx.graph, replacement_graph, args  # 传递当前模式条目、当前上下文的图形、替换图形和参数
        )

    # 根据示例替换当前模式条目
    def replace_by_example(
        self,
        replacement_fn: ReplaceFn,
        args: Sequence[Any],
        trace_fn: Optional[TraceFn] = None,
        run_dce: bool = True,
    ) -> None:
        from torch._inductor.virtualized import V  # 导入虚拟化模块中的V对象
        context = V.fake_mode if V.fake_mode is not None else contextlib.nullcontext  # 如果V.fake_mode不为None，则使用V.fake_mode；否则使用contextlib.nullcontext

        with context:  # 使用上下文管理器context
            if trace_fn is None:  # 如果跟踪函数为空
                trace_fn = functools.partial(fwd_only, run_dce=run_dce)  # 使用functools.partial创建部分应用的跟踪函数
            replacement = trace_fn(  # 使用跟踪函数替换
                replacement_fn, torch.fx.map_arg(args, lambda arg: arg.meta["val"])  # 使用替换函数和映射参数的元数据值
            )
            ReplacementPatternEntry.replace_with_graph(  # 使用替换模式条目的静态方法
                self,
                self.ctx.graph,
                replacement,
                args,
            )
class FailedMatch(RuntimeError):
    """
    表示匹配失败的异常类。

    `FailedMatch` 对象用于表示无法匹配某个模式的情况。
    """

    format_string: str

    def __init__(self, format_string: str, *args: Any, **kwargs: Any) -> None:
        """
        初始化方法，接受格式化字符串和可选的参数和关键字参数。

        如果格式化字符串超过 200 字符，将抛出 RuntimeError 异常。
        """
        self.format_string = format_string
        if len(format_string) > 200:
            raise RuntimeError(
                f"Format string too long - use lazy construction of strings instead. Format string is\n {format_string}"
            )
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        """
        返回格式化后的字符串。
        """
        return self.format_string.format(*self.args, **self.kwargs)

    def __bool__(self) -> bool:
        """
        始终返回 False。
        """
        return False


MatchResult = Union[Match, FailedMatch]


def is_match(m: MatchResult) -> TypeGuard[Match]:
    """
    TypeGuard 函数，用于识别 FailedMatch.__bool__ 作为 Match 的类型保护。
    """
    return bool(m)


class MatchContext:
    """
    在运行 PatternExpr._match() 时需要的内部状态类。
    """

    outputs: List[Optional[PatternExpr]]
    pattern_to_node: Dict[PatternExpr, Optional[torch.fx.Node]]
    graph: torch.fx.Graph
    exclusive_node_set: List[NodeOrConstant]

    def __init__(
        self,
        outputs: List[Optional[PatternExpr]],
        pattern_to_node: Optional[Dict[PatternExpr, torch.fx.Node]] = None,
        *,
        graph: torch.fx.Graph,
    ) -> None:
        """
        初始化方法，接受输出列表、模式到节点的映射字典和图对象作为参数。
        """
        self.outputs = outputs
        self.pattern_to_node = {} if pattern_to_node is None else dict(pattern_to_node)
        self.graph = graph
        self.exclusive_node_set = []

    def match(self, pattern: PatternExpr, node: NodeOrConstant) -> MatchResult:
        """
        包装器方法，用于检查模式中的重复节点。

        如果模式已经存在于 pattern_to_node 中，并且对应的节点与当前节点相同，则返回 Match 对象；
        否则返回 FailedMatch("repeated pattern differs")。
        """
        if pattern in self.pattern_to_node:
            if self.pattern_to_node[pattern] == node:
                return Match(self, pattern)  # 已经检查过该节点
            else:
                return FailedMatch("repeated pattern differs")
        m = pattern._match(node, self)
        assert pattern not in self.pattern_to_node
        self.pattern_to_node[pattern] = node if m else None
        return m

    def filter_multi_user_patterns(self) -> Dict[PatternExpr, torch.fx.Node]:
        """
        过滤出具有多个使用者的模式，并返回模式到节点的字典。
        """
        return {
            pattern: node
            for pattern, node in self.pattern_to_node.items()
            if pattern.has_multiple_users() and node is not None
        }


class PatternExpr(ABC):
    """
    模式类型的基类。
    """

    @abstractmethod
    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        """
        抽象方法，用于在给定上下文中匹配节点。
        """
        ...
    # 定义一个方法 `match`，用于在给定的节点上执行匹配操作，并返回匹配结果
    def match(self, node: torch.fx.Node) -> MatchResult:
        try:
            # 在匹配上下文中创建一个新的匹配对象，然后执行匹配操作
            return MatchContext([self], graph=node.graph).match(self, node)
        except FailedMatch as e:
            # 如果匹配失败，则返回一个 `FailedMatch` 异常对象
            return e

    # 定义一个方法 `has_multiple_users`，用于判断当前对象是否有多个使用者，始终返回 `False`
    def has_multiple_users(self) -> bool:
        return False

    # 重写 `__repr__` 方法，返回当前对象的类名字符串表示
    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    # 定义一个方法 `find_anchor_nodes`，用于在给定的匹配上下文中查找锚定节点，并返回一个生成器对象
    def find_anchor_nodes(
        self, ctx: MatchContext, searched: Set[torch.fx.Node]
    ) -> Generator[Optional[torch.fx.Node], None, None]:
        # 如果当前对象在匹配上下文的模式到节点映射中，则生成该节点
        if self in ctx.pattern_to_node:
            yield ctx.pattern_to_node[self]

    # 定义一个方法 `pattern_eq`，用于比较当前 `PatternExpr` 对象与另一个对象 `other` 是否相等
    def pattern_eq(self, other: Any) -> bool:
        """
        比较两个 `PatternExpr` 对象，如果它们结构上相同，则返回 `True`。
        注意，这不是执行模式匹配，而是比较模式结构（用于调试）。
        """
        return isinstance(other, self.__class__)
class Arg(PatternExpr):
    """
    Capture an arg which will become an input to the handler.  Args are
    passed in depth first order.
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        # 返回一个 MatchResult 对象，表示匹配成功，并将 node 作为 args 传递给 handler
        return Match(ctx, self, args=[node])  # matches anything


class Ignored(PatternExpr):
    """
    Match an arg, but don't pass it to handler
    """

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        # 返回一个 MatchResult 对象，表示匹配成功，但不传递任何参数给 handler
        return Match(ctx, self)  # matches anything

    def __repr__(self) -> str:
        # 返回一个字符串 "*"
        return "*"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        # 返回一个格式化的字符串 "Ignored()"
        return "Ignored()"


class KeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        # 返回一个格式化的字符串 "KeywordArg('name')"
        return f"KeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        # 返回一个 MatchResult 对象，表示匹配成功，并将 node 作为名为 self.name 的 kwarg 传递给 handler
        return Match(ctx, self, kwargs={self.name: node})  # matches anything

    def pattern_eq(self, other: Any) -> bool:
        # 比较当前对象与其他对象是否相等
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.name == other.name


class ExclusiveKeywordArg(PatternExpr):
    """
    Capture a kwarg which will become an input to the handler.
    """

    name: str

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def __repr__(self) -> str:
        # 返回一个格式化的字符串 "ExclusiveKeywordArg('name')"
        return f"ExclusiveKeywordArg({self.name!r})"

    def _match(self, node: NodeOrConstant, ctx: MatchContext) -> MatchResult:
        # 如果 node 在 ctx.exclusive_node_set 中已存在，则返回匹配失败的信息
        if node in ctx.exclusive_node_set:
            return FailedMatch("exclusive arg appears twice")

        # 将 node 添加到 ctx.exclusive_node_set 中，并返回一个 MatchResult 对象，
        # 表示匹配成功，并将 node 作为名为 self.name 的 kwarg 传递给 handler
        ctx.exclusive_node_set.append(node)
        return Match(ctx, self, kwargs={self.name: node})  # matches anything

    def pattern_eq(self, other: Any) -> bool:
        # 比较当前对象与其他对象是否相等
        other = typing.cast(Self, other)  # super makes sure this is true
        return super().pattern_eq(other) and self.name == other.name


class _TargetExpr(PatternExpr):
    """
    Base class for filtering match by node.target
    """

    fns: List[FnsType]
    fns_set: Set[FnsType]

    def __init__(
        self, fns: Union[FnsType, Sequence[FnsType]], users: Union[Multiple, int] = 1
    ) -> None:
        super().__init__()
        # 如果 fns 是单个函数类型或字符串，则将其转换为列表形式
        fns = [fns] if callable(fns) or isinstance(fns, str) else list(fns)
        # 对于 fn 中的每个函数，如果是 torch._ops.OpOverloadPacket 类型，则将其所有重载函数加入 fns 列表
        for fn in fns:
            if isinstance(fn, torch._ops.OpOverloadPacket):
                fns.extend(getattr(fn, overload) for overload in fn.overloads())

        # 初始化 fns 和 fns_set 属性
        self.fns = fns
        self.fns_set = set(fns)
        self.users = users

    @property
    @abstractmethod
    def op(self) -> str:
        # 抽象属性，子类需实现该属性
        ...
    # 返回此对象的字符串表示形式，用于显示对象信息
    def fns_repr(self) -> str:
        # 获取第一个函数的字符串表示形式
        first_repr = self.fns[0]
        # 如果第一个函数不是字符串，获取其名称
        if not isinstance(first_repr, str):
            first_repr = first_repr.__name__

        # 如果函数列表长度大于1，返回列表的简略表示形式
        if len(self.fns) > 1:
            return f"[{first_repr}, ...]"
        # 如果第一个函数是torch模块中的函数，则返回完整的torch路径
        elif self.fns[0] is getattr(torch, first_repr, None):
            return f"torch.{first_repr}"
        # 如果第一个函数是torch._ops.OpOverload的实例，返回其字符串表示形式
        elif isinstance(self.fns[0], torch._ops.OpOverload):
            return str(self.fns[0])
        else:
            return first_repr

    # 返回对象的字符串表示形式，包括函数列表和用户数（如果有）
    def __repr__(self) -> str:
        # 如果用户数是MULTIPLE，表示有多个用户
        if self.users is MULTIPLE:
            comma_users = ", MULTIPLE"
        # 如果用户数不等于1，表示有指定数量的用户
        elif self.users != 1:
            comma_users = f", {self.users})"
        else:
            comma_users = ""
        # 返回对象的字符串表示形式
        return f"{self.__class__.__name__}({self.fns_repr()}{comma_users})"

    # 判断对象是否有多个用户
    def has_multiple_users(self) -> bool:
        return isinstance(self.users, Multiple) or self.users > 1

    # 查找与给定匹配上下文和已搜索节点匹配的锚节点生成器，抛出未实现错误
    def find_anchor_nodes(
        self, ctx: MatchContext, searched: Set[torch.fx.Node]
    ) -> Generator[Optional[torch.fx.Node], None, None]:
        raise NotImplementedError

    # 判断给定节点是否与对象的操作和函数集合匹配
    def _match_fns(self, node: torch.fx.Node) -> bool:
        return (
            isinstance(node, torch.fx.Node)
            and node.op == self.op
            and extract_target(node) in self.fns_set
        )

    # 判断给定节点是否与对象的输出关联，或者对象有多个用户，或者节点的用户数与对象的用户数相同
    def _match_users(self, node: torch.fx.Node, ctx: MatchContext) -> bool:
        return (
            self in ctx.outputs
            or self.users is MULTIPLE
            or len(node.users) == self.users
        )

    # 比较对象是否与另一个对象相等，包括操作、函数列表和用户数
    def pattern_eq(self, other: Any) -> bool:
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            super().pattern_eq(other)
            and self.op == other.op
            and self.fns == other.fns
            and self.users == other.users
        )
# 定义一个类型别名 _SimpleSpec，表示一个任意类型的元组
_SimpleSpec = Tuple[Any, ...]

# 定义一个类 _TargetArgsExpr，继承自 _TargetExpr 类
class _TargetArgsExpr(_TargetExpr):
    """
    Base class for filtering match by node.{target,args,kwargs}
    """

    # 初始化方法
    def __init__(
        self,
        fns: Union[torch.fx.node.Target, str, Sequence[Any]],
        *args: Any,
        _users: Union[int, Multiple] = 1,
        **kwargs: Any,
    ) -> None:
        # 调用父类 _TargetExpr 的初始化方法
        super().__init__(fns, _users)
        # 将位置参数 args 转换为元组
        self.args = tuple(args)
        # 将关键字参数 kwargs 转换为字典
        self.kwargs = dict(kwargs)
        # 检查 args 和 kwargs 中是否有 dict、list 或 tuple 类型的元素
        if any(
            isinstance(x, (dict, list, tuple))
            for x in itertools.chain(args, kwargs.values())
        ):
            # 如果有，使用 pytree_flatten 方法进行扁平化处理
            self.flatten = self.pytree_flatten
        else:
            # 否则使用 simple_flatten 方法进行简单的扁平化处理
            self.flatten = self.simple_flatten
        # 对 args 和 kwargs 进行扁平化处理，得到扁平化后的参数列表和规范
        self.flat_args_kwargs = self.flatten(self.args, self.kwargs)

    # 静态方法，用于简单的扁平化处理
    @staticmethod
    def simple_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> Tuple[Sequence[Any], Union[_SimpleSpec, pytree.TreeSpec]]:
        # 将 args 和 kwargs 的值以及所有 kwargs 的键组成一个列表 values
        values = (*args, *kwargs.values())
        # 创建一个规范 spec，包含 args 的长度和 kwargs 的所有键
        spec = (len(args), *kwargs.keys())
        return values, spec

    # 静态方法，用于使用 pytree 进行扁平化处理
    @staticmethod
    def pytree_flatten(
        args: Sequence[Any], kwargs: Mapping[Any, Any]
    ) -> Tuple[Sequence[Any], Union[_SimpleSpec, pytree.TreeSpec]]:
        # 规范化 pytree 的规范 spec
        def norm_spec(s: pytree.TreeSpec) -> pytree.TreeSpec:
            if s.type is None:
                return s
            # 创建一个映射，将不可变列表映射为列表，元组映射为列表，不可变字典映射为字典
            mapping = {immutable_list: list, tuple: list, immutable_dict: dict}
            return pytree.TreeSpec(
                mapping.get(s.type, s.type),
                s.context,
                list(map(norm_spec, s.children_specs)),
            )

        # 使用 pytree 的 tree_flatten 方法扁平化 args 和 kwargs
        flat, spec = pytree.tree_flatten([args, kwargs])
        # 规范化 spec
        spec = norm_spec(spec)
        return flat, spec

    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        # 构建对象的详细描述信息列表 args
        args = [
            self.fns_repr(),  # 表示目标函数的字符串表示形式
            *map(repr, self.args),  # 对位置参数 args 的每个元素调用 repr 函数
            *[f"{k}={v}" for k, v in self.kwargs.items()],  # 对关键字参数 kwargs 的每个键值对构建字符串
        ]
        # 如果 users 属性为 MULTIPLE，则在 args 列表末尾添加 "_users=MULTIPLE"
        if self.users is MULTIPLE:
            args.append("_users=MULTIPLE")
        # 如果 users 属性不等于 1，则在 args 列表末尾添加 "_users=<users>"
        elif self.users != 1:
            args.append(f"_users={self.users}")
        # 返回对象的字符串表示形式
        return f"{self.__class__.__name__}({', '.join(args)})"

    # 返回对象在美化输出中的字符串表示形式
    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        # 构建美化输出中的详细描述信息列表 args
        args = [
            self.fns_repr(),  # 表示目标函数的字符串表示形式
            *(pp.pretty_print(x) for x in self.args),  # 对位置参数 args 的每个元素进行美化输出
            *[f"{k}={pp.pretty_print(v)}" for k, v in self.kwargs.items()],  # 对关键字参数 kwargs 的每个键值对进行美化输出
        ]
        # 如果 users 属性为 MULTIPLE，则在 args 列表末尾添加 "_users=MULTIPLE"
        if self.users is MULTIPLE:
            args.append("_users=MULTIPLE")
        # 如果 users 属性不等于 1，则在 args 列表末尾添加 "_users=<users>"
        elif self.users != 1:
            args.append(f"_users={self.users}")

        # 使用 joiner_str 连接 args 列表中的元素，形成美化输出的字符串表示形式
        joiner_str = ", "
        return f"{self.__class__.__name__}({joiner_str.join(args)})"
    # 定义一个方法用于匹配节点和模式，返回匹配结果对象
    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        # 如果节点的匹配函数返回False或者参数个数不匹配模式的参数个数，则匹配失败
        if not self._match_fns(node) or len(node.args) != len(self.args):
            return FailedMatch("function_mismatch: node={}, pattern={}", node, self)

        # 如果节点的用户数不符合预期（多于一个），则匹配失败
        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users {}", self)

        # 获取节点的位置参数和关键字参数
        _args = node.args
        _kwargs = node.kwargs
        # 如果节点的关键字参数个数小于模式的关键字参数个数
        if len(_kwargs) < len(self.kwargs):
            # 导入函数：用于规范化节点的函数及其参数
            from torch.fx.operator_schemas import normalize_function

            # 规范化节点的函数及其参数
            normalized_args_and_kwargs = normalize_function(
                node.target, node.args, node.kwargs
            )

            # 如果规范化结果为空，则匹配失败
            if normalized_args_and_kwargs is None:
                return FailedMatch("function_mismatch: node={}, pattern={}", node, self)
            else:
                _args, _kwargs = normalized_args_and_kwargs
                # 如果规范化后的参数个数和模式的参数个数匹配，并且关键字参数个数不小于模式的关键字参数个数
                if len(_args) == len(self.args) and len(_kwargs) >= len(self.kwargs):
                    # 筛选出匹配模式的关键字参数
                    _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}
                else:
                    return FailedMatch(
                        "function_mismatch: node={}, pattern={}", node, self
                    )
        else:
            # 筛选出匹配模式的关键字参数
            _kwargs = {i: _kwargs[i] for i in _kwargs if i in self.kwargs}

        # 将节点的位置参数和关键字参数展平
        node_items, node_spec = self.flatten(_args, _kwargs)
        # 获取模式对象的位置参数和关键字参数
        self_items, self_spec = self.flat_args_kwargs
        # 如果节点的参数结构与模式对象的参数结构不匹配，则匹配失败
        if node_spec != self_spec:
            return FailedMatch("args_structure {} {}", node_spec, self_spec)
        # 断言节点的位置参数个数与模式对象的位置参数个数相等
        assert len(node_items) == len(self_items)

        # 创建一个匹配对象
        m = Match(ctx, self)
        # 遍历模式对象的位置参数和节点的位置参数
        for i, pattern, child_node in zip(itertools.count(), self_items, node_items):
            # 如果模式对象的参数是PatternExpr类型
            if isinstance(pattern, PatternExpr):
                # 在上下文中匹配模式表达式和子节点
                child_match = ctx.match(pattern, child_node)
                # 如果匹配结果不符合预期，则返回匹配失败
                if not is_match(child_match):
                    return child_match
                # 将子匹配结果添加到当前匹配对象中
                m.extend(child_match)
            # 如果子节点是torch.fx.Node类型，或者子节点不等于模式对象的位置参数
            elif isinstance(child_node, torch.fx.Node) or child_node != pattern:
                return FailedMatch(
                    "constant_args: {} {!r}!={pattern!r}", node, child_node
                )
        # 将当前节点添加到匹配对象的节点列表中
        m.nodes.append(node)
        # 将当前节点的目标对象与模式对象关联
        m.targets[self] = node.target
        # 返回匹配对象
        return m

    # 定义一个方法用于查找锚点节点
    def find_anchor_nodes(
        self, ctx: MatchContext, searched: Set[torch.fx.Node]
    ) -> Generator[Optional[torch.fx.Node], None, None]:
        """
        当我们匹配具有多个输出的模式时使用。
        存在一个部分匹配（存储在 ctx 中），我们希望遍历
        此模式以找到与已匹配节点的连接。

        产生 `self._match` 可能喜欢的候选节点。
        """
        if self in ctx.pattern_to_node:
            # 如果当前节点已经在模式到节点的映射中，则生成对应的节点
            yield ctx.pattern_to_node[self]
            return

        for pattern in self.flat_args_kwargs[0]:
            if isinstance(pattern, PatternExpr):
                # 遍历模式表达式中的锚节点，查找符合条件的节点
                for other_node in pattern.find_anchor_nodes(ctx, searched):
                    if not isinstance(other_node, torch.fx.Node):
                        continue
                    # 遍历其他节点的使用者
                    for node in other_node.users:
                        if node not in searched:
                            # 如果节点尚未被搜索过且满足匹配条件，则生成该节点
                            if self._match_fns(node):
                                yield node
                                searched.add(node)

    def pattern_eq(self, other: Any) -> bool:
        # 使用 super 确保 other 是正确的类型
        other = typing.cast(Self, other)  # super makes sure this is true
        return (
            # 调用父类的 pattern_eq 方法比较结果
            super().pattern_eq(other)
            # 比较 flat_args_kwargs[1] 是否相等
            and self.flat_args_kwargs[1] == other.flat_args_kwargs[1]
            # 使用 zip 遍历并比较 flat_args_kwargs[0] 的每个元素
            and all(
                # 如果元素是 PatternExpr 类型则调用其 pattern_eq 方法，否则直接比较
                a.pattern_eq(b) if isinstance(a, PatternExpr) else a == b
                for a, b in zip(self.flat_args_kwargs[0], other.flat_args_kwargs[0])
            )
        )
class CallFunction(_TargetArgsExpr):
    """
    Matches a call_function node in the FX graphs: `fns[i](*args, **kwargs)`
    """

    op = "call_function"


class CallMethod(_TargetArgsExpr):
    """
    Matches a call_method node in the FX graphs: `fns[i].method(*args, **kwargs)`
    """

    op = "call_method"


class CallModule(_TargetArgsExpr):
    """
    Matches a call_module node in the FX graphs: `module(*args, **kwargs)`
    """

    op = "call_module"


class _TargetExprVarArgs(_TargetExpr):
    """
    Matches a call_function node with any arguments which are passed into the pattern
    """

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        # 检查节点是否与目标函数匹配
        if not self._match_fns(node):
            return FailedMatch("function_mismatch")

        # 检查节点是否有多个使用者
        if not self._match_users(node, ctx):
            return FailedMatch("multiple_users")

        # 创建匹配对象并添加节点到匹配列表
        m = Match(ctx, self)
        m.nodes.append(node)
        # 设置匹配的目标
        m.targets[self] = node.target
        # 扩展匹配的参数和关键字参数
        m.args.extend(node.args)
        m.kwargs.update(node.kwargs)
        return m


class CallFunctionVarArgs(_TargetExprVarArgs):
    op = "call_function"


class CallMethodVarArgs(_TargetExprVarArgs):
    op = "call_method"


class CallModuleVarArgs(_TargetExprVarArgs):
    op = "call_module"


class ListOf(PatternExpr):
    """
    Matches a repeated pattern
    """

    def __init__(self, pattern: PatternExpr, partial: bool = False) -> None:
        super().__init__()
        assert isinstance(pattern, PatternExpr)
        self.pattern = pattern
        self.partial = partial

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.pattern})"

    def _match(self, node: List[torch.fx.Node], ctx: MatchContext) -> MatchResult:  # type: ignore[override]
        # 检查节点是否为列表或元组且非空
        if not isinstance(node, (list, tuple)) or len(node) == 0:
            return FailedMatch("non_list")

        # 创建匹配对象
        m = Match(ctx, self)
        # 传播带有多个用户的模式，以确保不会重复访问相同的节点
        pattern_to_node = ctx.filter_multi_user_patterns()
        matched = False
        # 遍历列表中的节点
        for i, child_node in enumerate(node):
            # 创建子上下文
            child_ctx = MatchContext(
                ctx.outputs, pattern_to_node, graph=child_node.graph
            )
            # 在子上下文中匹配模式
            child_match = child_ctx.match(self.pattern, child_node)
            # 过滤带有多个用户的模式
            pattern_to_node = child_ctx.filter_multi_user_patterns()
            # 如果匹配不成功且不允许部分匹配，则返回失败信息
            if not is_match(child_match):
                if not self.partial:
                    return FailedMatch("list[{}]: {}", i, child_match)
                continue
            matched = True
            # 将成功匹配的结果添加到匹配对象中
            m.extend(child_match.bundle())
        # 如果没有成功匹配任何节点，则返回失败信息
        if not matched:
            return FailedMatch("list: no_match")
        return m.bundle()
    # 定义一个方法 `pattern_eq`，用于判断当前对象与另一个对象是否相等
    def pattern_eq(self, other: Any) -> bool:
        # 使用 `typing.cast` 将 `other` 强制转换为当前类的类型 `Self`
        other = typing.cast(Self, other)  # super makes sure this is true
        # 返回比较结果，需满足以下条件：
        # 1. 调用父类的 `pattern_eq` 方法，比较当前对象和另一个对象是否相等
        # 2. 比较当前对象的 `pattern` 属性与另一个对象的 `pattern` 属性是否相等
        # 3. 比较当前对象的 `partial` 属性与另一个对象的 `partial` 属性是否相等
        return (
            super().pattern_eq(other)
            and self.pattern.pattern_eq(other.pattern)
            and self.partial == other.partial
        )
class MultiOutputPattern(PatternExpr):
    outputs: List[Optional[PatternExpr]]

    def __init__(self, outputs: Sequence[Optional[PatternExpr]]) -> None:
        super().__init__()
        # 确保第一个输出是_TargetExpr类型
        assert isinstance(outputs[0], _TargetExpr)
        # 确保所有的输出要么是None，要么是PatternExpr类型
        assert all(x is None or isinstance(x, PatternExpr) for x in outputs), outputs
        # 将输出列表转换为普通列表
        self.outputs = list(outputs)
        # 将第一个输出的操作符保存为对象的操作符
        self.op = outputs[0].op

    @property
    def fns(self) -> Union[Callable[..., Any], str, Sequence[Any]]:
        # 此类型转换在__init__()中已经检查过
        output = typing.cast(_TargetExpr, self.outputs[0])
        # 返回第一个输出的fns属性
        return output.fns

    def __repr__(self) -> str:
        # 返回对象的字符串表示，包括所有输出
        return f"{self.__class__.__name__}({self.outputs})"

    def pretty_print(self, pp: PatternPrettyPrinter) -> str:
        # 使用PrettyPrinter对象打印每个输出的美观输出
        args = [pp.pretty_print(x) for x in self.outputs]
        joiner_str = f",\n{'  '}"
        str_out = f"{self.__class__.__name__}([{joiner_str.join(args)}"
        str_out = f"{str_out}\n])"
        return str_out

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        # 获取第一个输出作为_TargetExpr类型
        output = typing.cast(_TargetExpr, self.outputs[0])
        # 使用MatchContext对象进行匹配
        m = ctx.match(output, node)
        # 如果匹配不成功，则直接返回匹配结果
        if not is_match(m):
            return m

        # 对于除第一个之外的每个输出进行匹配
        for pattern in self.outputs[1:]:
            if pattern is None:
                continue
            # 使用锚点从模式中进行匹配
            child_match = self._match_from_anchors(pattern, ctx)
            # 如果子匹配不成功，则直接返回子匹配结果
            if not is_match(child_match):
                return child_match
            # 将子匹配结果扩展到主匹配结果中
            m.extend(child_match)

        return m

    def _match_from_anchors(
        self, pattern: PatternExpr, ctx: MatchContext
    ) -> MatchResult:
        # 保存先前的上下文状态
        prior = dict(ctx.pattern_to_node)
        # 默认情况下，匹配结果是失败的
        m: MatchResult = FailedMatch("no anchor found")
        # 遍历模式中的所有锚点节点
        for node in pattern.find_anchor_nodes(ctx, set()):
            # 使用MatchContext对象进行匹配
            m = ctx.match(pattern, node)
            # 如果匹配成功，则直接返回匹配结果
            if is_match(m):
                return m
            # 恢复任何部分匹配的状态
            ctx.pattern_to_node = dict(prior)
        return m

    def match(self, node: torch.fx.Node) -> MatchResult:
        try:
            # 创建MatchContext对象并使用其match方法进行匹配
            return MatchContext(self.outputs, graph=node.graph).match(self, node)
        except FailedMatch as e:
            # 如果匹配失败，则返回FailedMatch异常
            return e

    def pattern_eq(self, other: Any) -> bool:
        # 将other强制转换为Self类型（这里可能是MultiOutputPattern类型），确保类型匹配
        other = typing.cast(Self, other)  # super makes sure this is true
        # 检查父类的模式是否相等，并且检查每个输出是否相等
        return (
            super().pattern_eq(other)
            and len(self.outputs) == len(other.outputs)
            and all(
                a.pattern_eq(b) if isinstance(a, PatternExpr) else a == b
                for a, b in zip(self.outputs, other.outputs)
            )
        )


class RepeatedExpr(PatternExpr):
    """
    Checks for a repeated pattern. Useful for repeated operations after a node such as `split` or `unbind`
    """

    def __init__(self, inner_pattern: _TargetExpr) -> None:
        super().__init__()
        # 将内部模式保存为对象的内部模式
        self.inner_pattern = inner_pattern
        # 将内部模式的操作符保存为对象的操作符
        self.op = inner_pattern.op

    @property
    def fns(self) -> Sequence[FnsType]:
        # 返回内部模式的fns属性
        return self.inner_pattern.fns
    # 定义一个方法 `_match`，用于匹配给定节点 `node` 和匹配上下文 `ctx`，返回匹配结果 `MatchResult`
    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        # 调用匹配上下文的 `match` 方法，使用 `self.inner_pattern` 匹配 `node`，获取匹配结果 `m`
        m = ctx.match(self.inner_pattern, node)
        # 如果匹配结果 `m` 不满足匹配条件，直接返回 `m`
        if not is_match(m):
            return m
        # 从匹配上下文的 `pattern_to_node` 字典中移除 `self.inner_pattern`
        ctx.pattern_to_node.pop(
            self.inner_pattern,
        )
        # 检查所有锚点节点是否匹配模式 `self.inner_pattern`
        for anchor_node in self.inner_pattern.find_anchor_nodes(ctx, set()):
            # 创建一个新的匹配上下文，匹配模式 `self.inner_pattern` 和锚点节点 `anchor_node`
            anchor_m = MatchContext([self], graph=node.graph).match(
                self.inner_pattern, anchor_node
            )
            # 如果锚点节点的匹配结果 `anchor_m` 不满足匹配条件，直接返回 `anchor_m`
            if not is_match(anchor_m):
                return anchor_m
            # 将锚点节点的匹配结果 `anchor_m` 添加到主节点的匹配结果 `m` 中
            m.extend(anchor_m)
        # 返回整体的匹配结果 `m`
        return m

    # 定义一个方法 `pattern_eq`，用于判断当前对象与另一个对象 `other` 是否模式相等
    def pattern_eq(self, other: Any) -> bool:
        # 将 `other` 强制类型转换为 `Self` 类型（当前类的类型），确保其相等
        other = typing.cast(Self, other)  # super makes sure this is true
        # 调用父类的 `pattern_eq` 方法，比较当前对象与 `other` 内部模式是否相等
        return super().pattern_eq(other) and self.inner_pattern.pattern_eq(
            other.inner_pattern
        )
class PatternPrettyPrinter:
    """
    Serializes Patterns to executable python.
    XXX: currently only used and tested for fuse attention patterns. May not cover
    all patterns.
    """

    def __init__(self) -> None:
        # 初始化命名空间对象
        self.namespace = torch.fx.graph._Namespace()
        # 用于存储已记忆化对象的名称和其对应的序列化字符串
        self.memoized_objs_names: Dict[PatternExpr, str] = {}
        self.memoized_objs_pp: Dict[PatternExpr, str] = {}

    @staticmethod
    @functools.lru_cache(None)
    def run(obj: PatternExpr, output_name: str = "output") -> str:
        """
        Serializes obj to python code with obj written out to `output_name`
        """

        pp = PatternPrettyPrinter()
        # 断言对象具有 "pretty_print" 方法
        assert hasattr(obj, "pretty_print")
        # 调用对象的 pretty_print 方法，并将结果存储到 out_str 中
        out_str = obj.pretty_print(pp=pp)

        output = []
        # 遍历已记忆化对象的名称字典
        for key in pp.memoized_objs_names:
            # 将对象名称和其序列化字符串格式化为赋值语句，并添加到 output 列表中
            output.append(f"{pp.memoized_objs_names[key]} = {pp.memoized_objs_pp[key]}")

        # 将对象序列化字符串赋值语句添加到 output 列表中
        output.append(f"{output_name} = {out_str}")

        # 将 output 列表中的所有内容组合成一个字符串并返回
        return "\n".join(output)

    def pretty_print(self, obj: Any) -> str:
        # 如果 obj 是 _TargetArgsExpr 类型的实例
        if isinstance(obj, _TargetArgsExpr):
            # 如果 obj 已经被记忆化，则返回其对应的名称
            if memoized_name := self.memoized_objs_names.get(obj):
                return memoized_name
            else:
                # 否则，进行记忆化并返回其名称
                return self.memoize(obj)
        # 如果 obj 具有 "pretty_print" 方法，则调用该方法并返回结果
        if hasattr(obj, "pretty_print"):
            return obj.pretty_print(self)

        # 默认情况下，使用 repr 函数返回对象的字符串表示形式
        return repr(obj)

    def memoize(self, obj: _TargetArgsExpr) -> str:
        # 获取对象的字符串表示形式
        obj_str = obj.pretty_print(self)
        # 获取对象的函数表示形式
        obj_name = obj.fns_repr()
        # 去除函数表示形式中的特定前缀
        for prefix in ("aten.", "torch.", "prims."):
            obj_name = obj_name.replace(prefix, "")

        # 创建一个临时名称并将对象及其字符串表示形式存储到记忆化字典中
        tmp_name = self.namespace.create_name(obj_name, None)
        self.memoized_objs_names[obj] = tmp_name
        self.memoized_objs_pp[obj] = obj_str
        # 返回临时名称
        return tmp_name


class _PassDictsType(Protocol):
    def __getitem__(self, k: Tuple[str, torch.fx.node.Target]) -> List[PatternEntry]:
        ...


@dataclasses.dataclass
class PatternEntry:
    pattern: PatternExpr
    extra_check: Callable[[Match], bool]

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        # 抽象方法，子类实现，用于应用模式匹配后的操作
        raise NotImplementedError

    def register(
        self,
        pass_dicts: Union[_PassDictsType, Sequence[_PassDictsType]],
        target: Union[torch.fx.node.Target, None] = None,
        prepend: bool = False,
    ) -> None:
        # 如果 target 为 None，则注册模式的各个函数表达式
        if target is None:
            # 断言模式对象具有 "fns" 属性
            assert hasattr(self.pattern, "fns")
            # 遍历模式对象的函数表达式并依次注册
            for fn in self.pattern.fns:
                self.register(pass_dicts, fn, prepend=prepend)
        # 如果 pass_dicts 是 dict 或 PatternMatcherPass 类型的实例
        elif isinstance(pass_dicts, (dict, PatternMatcherPass)):
            # 断言模式对象具有 "op" 属性
            assert hasattr(self.pattern, "op")
            # 如果 prepend 为 True，则在 pass_dicts 中相应键值对的列表首部插入当前模式对象
            if prepend:
                pass_dicts[(self.pattern.op, target)].insert(0, self)
            else:
                # 否则，将当前模式对象追加到 pass_dicts 中相应键值对的列表末尾
                pass_dicts[(self.pattern.op, target)].append(self)
        else:
            # 将 pass_dicts 强制类型转换为 Sequence[_PassDictsType]，并对其进行遍历注册
            pass_dicts = typing.cast(Sequence[_PassDictsType], pass_dicts)
            for x in pass_dicts:
                self.register(x, target, prepend=prepend)
class LoweringPatternEntry(PatternEntry):
    handler: Callable[..., Any]  # 声明一个可接受任意参数的回调函数属性

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        # 使用 functools 包装 handler，并将 match 参数部分应用到 handler 中
        handler = functools.wraps(self.handler)(functools.partial(self.handler, match))
        # 在节点 node 前插入新的调用节点 replacement
        with graph.inserting_before(node):
            replacement = graph.call_function(handler, tuple(match.args), match.kwargs)
            replacement.meta.update(node.meta)  # 更新 replacement 的元数据
            node.replace_all_uses_with(replacement)  # 替换原节点 node 所有的使用者为 replacement
        assert match.nodes[-1] is node  # 断言最后一个节点是 node
        match.erase_nodes(graph)  # 在图中擦除匹配的节点


@dataclasses.dataclass
class GraphPatternEntry(PatternEntry):
    """
    A pattern that runs a function on the FX graph
    """

    handler: Callable[..., Any]  # 声明一个可接受任意参数的回调函数属性

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        # 在节点 node 前插入新的调用节点，直接调用 self.handler 函数
        with graph.inserting_before(node):
            self.handler(match, *match.args, **match.kwargs)  # 将 match 的参数传递给 handler 函数


@dataclasses.dataclass
class ReplacementPatternEntry(PatternEntry):
    normalize_args: Callable[..., List[Any]]  # 声明一个可接受任意参数的标准化参数函数属性

    @staticmethod
    def replace_with_graph(
        match: Match,
        graph: torch.fx.Graph,
        replacement_graph: Union[torch.fx.Graph, torch.fx.GraphModule],
        args: Sequence[torch.fx.Node],
    ) -> None:
        # 替换匹配的节点为 replacement_graph
        match.replace_with_graph(graph, replacement_graph, args)

    def apply(self, match: Match, graph: torch.fx.Graph, node: torch.fx.Node) -> None:
        assert match.replacement_graph is not None  # 断言 replacement_graph 不为空
        # 调用 replace_with_graph 方法，使用标准化参数后的参数替换节点
        self.replace_with_graph(
            match,
            graph,
            match.replacement_graph,
            self.normalize_args(*match.args, **match.kwargs),
        )


def _return_true(match: Match) -> bool:
    return True  # 返回布尔值 True


def log_trace_failure(search_fn: Callable[..., Any], e: RuntimeError) -> None:
    log.info(
        "Replacement pattern %s failed to apply due to shape mismatch: %s",
        search_fn.__name__,  # 记录搜索函数的名称
        e,  # 记录运行时错误 e 的信息
    )


def register_replacement(
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: Union[_PassDictsType, Sequence[_PassDictsType]],
    extra_check: Callable[[Match], bool] = _return_true,  # 默认为 _return_true 函数
    scalar_workaround: Union[Dict[str, Union[float, int]], None] = None,  # 可选的标量处理字典参数，默认为 None
    exclusive_arg_names: Sequence[str] = (),  # 独占参数名的序列，默认为空序列
    search_fn_pattern: Union[PatternExpr, None] = None,  # 可选的搜索函数模式，默认为 None
) -> bool:
    """
    Create a replacement rule based on example functions that get traced
    to create patterns.  This supports both training and inference when
    run on a joint forward+backward graph.

    Args:
        search_fn: traced to give original pattern  # 被追踪以提供原始模式的搜索函数
        replace_fn: traced to give replacement graph  # 被追踪以提供替换图的替换函数
        example_inputs: example inputs for initial trace  # 初始跟踪的示例输入
        trace_fn: fwd_only or joint_fwd_bwd  # fwd_only 或 joint_fwd_bwd 的跟踪函数
        pass_dict: dict of passes to register to  # 要注册的传递字典
        extra_check: additional check to run on match(using real shapes)  # 在匹配上运行的额外检查（使用真实形状）
    """
    argnames_static = [*inspect.signature(search_fn).parameters.keys()]  # 获取搜索函数的静态参数名列表
    # 定义函数，接收关键字参数并返回一个参数列表
    def normalize_args(**kwargs: Any) -> List[Any]:
        # 初始化空列表用于存放参数
        args = []
        # 遍历静态参数名列表
        for name in argnames_static:
            # 将对应的参数值从 kwargs 中取出并添加到 args 列表中
            args.append(kwargs.pop(name))
        # 遍历剩余的 kwargs 中的键值对
        for i in range(1, len(kwargs) + 1):
            # 如果以 "tangents_i" 形式的键不在 kwargs 中，跳出循环
            if f"tangents_{i}" not in kwargs:
                break
            # 将对应的参数值从 kwargs 中取出并添加到 args 列表中
            args.append(kwargs.pop(f"tangents_{i}"))
        # 断言确保 kwargs 已经处理完毕，否则抛出异常显示剩余的未处理参数
        assert not kwargs, f"leftover kwargs: {kwargs!r}"
        # 返回处理后的参数列表
        return args

    # 如果 trace_fn 是 joint_fwd_bwd 函数
    if trace_fn is joint_fwd_bwd:
        # 如果 Torch 的推断模式已启用，则返回 False，表示不进行训练图模式匹配
        if torch.is_inference_mode_enabled():
            return False

    # TODO: 重新审视 functionalize_rng_ops 以实现低内存的 dropout
    # 使用 functorch_config.patch 上下文管理器，禁用 functionalize_rng_ops
    with functorch_config.patch(functionalize_rng_ops=False):
        # 初始化 requires_grad 列表，检查 example_inputs 中每个元素是否是 Tensor 且需要梯度
        requires_grad: List[bool] = [
            isinstance(x, torch.Tensor) and x.requires_grad for x in example_inputs
        ]
        # 如果 search_fn_pattern 为 None，则根据其他参数生成模式
        if search_fn_pattern is None:
            # 生成模式，使用 gen_pattern 函数，传入搜索函数、示例输入、追踪函数等参数
            pattern = gen_pattern(
                search_fn,
                example_inputs,
                trace_fn,
                scalar_workaround,
                exclusive_arg_names,
            )
        else:
            # 否则，使用预定义的 search_fn_pattern 作为模式
            pattern = search_fn_pattern

        # 将生成的模式对象转换为可打印的字符串表示形式
        pattern_repr = PatternPrettyPrinter.run(pattern)
        # 断言确保该模式的字符串表示形式不在 _seen_patterns 集合中，避免重复注册相同的模式
        assert pattern_repr not in _seen_patterns
        # 将模式的字符串表示形式添加到 _seen_patterns 集合中，标记该模式已被注册
        _seen_patterns.add(pattern_repr)
        # 创建 ReplacementPatternEntry 对象，包含模式、额外检查函数和参数归一化函数
        pattern = ReplacementPatternEntry(
            pattern=pattern,
            extra_check=check_fn,
            normalize_args=normalize_args,
        )
        # 注册模式对象，传入 pass_dicts 参数
        pattern.register(pass_dicts)
        # 返回注册后的模式对象
        return pattern.pattern
# 初始化一个空集合，用于存储序列化的模式的唯一名称
_serialized_patterns: Set[str] = set()


def _serialize_pattern(
    unique_name: str,
    search_fn: SearchFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    scalar_workaround: Union[Dict[str, Union[float, int]], None],
) -> PatternExpr:
    # 定义一个内部函数，生成用于保存序列化模式的文件模板
    def get_file_template() -> str:
        # 自动化生成的文件消息，不要手动修改
        auto_generated_msg = textwrap.dedent(
            """\
            # This is an auto-generated file. Please do not modify it by hand.
            # To re-generate, run:
            # cd ~/pytorch && python torchgen/fuse/gen_patterns.py
            """
        )

        # 文件的模板，包括导入语句和模块引用
        file_template = textwrap.dedent(
            """\
            # mypy: ignore-errors

            # noqa: F401, E501
            {msg}
            import torch
            import torch._inductor

            aten = torch.ops.aten
            prims = torch.ops.prims

            """
        ).format(msg=auto_generated_msg)

        # 导入模式匹配器中的类和函数
        pattern_matcher_imports = []
        for name in dir(torch._inductor.pattern_matcher):
            attr = getattr(torch._inductor.pattern_matcher, name)
            if isinstance(attr, type) and issubclass(attr, (PatternExpr, _TargetExpr)):
                pattern_matcher_imports.append(name)

        formatted_imports = ",\n   ".join(pattern_matcher_imports)
        formatted_imports = f"from torch._inductor.pattern_matcher import (\n   {formatted_imports},\n)\n"
        return f"{file_template}{formatted_imports}"

    # 如果序列化模式的路径不存在，则抛出运行时错误
    if not SERIALIZED_PATTERN_PATH.is_dir():
        raise RuntimeError(
            f"Could not find serialized patterns directory at {SERIALIZED_PATTERN_PATH}"
        )

    # 获取搜索函数的名称作为模式名称
    pattern_name = search_fn.__name__

    # 从torch._functorch模块导入配置
    from torch._functorch import config as functorch_config

    # 使用函数调用上下文管理器，关闭functionalize_rng_ops选项
    with functorch_config.patch(functionalize_rng_ops=False):
        # 生成模式对象
        pattern = gen_pattern(search_fn, example_inputs, trace_fn, scalar_workaround)

    # 运行模式的美化打印器，生成序列化的模式字符串
    serialized_pattern = PatternPrettyPrinter.run(pattern, output_name=unique_name)

    # 如果模式名称不在已序列化的模式集合中，则将写入模式设置为覆盖模式文件，否则追加写入
    if pattern_name not in _serialized_patterns:
        write_mode = "w"
        _serialized_patterns.add(pattern_name)
    else:
        write_mode = "a"

    # 获取文件模板
    file_template = get_file_template()

    # 打开模式文件进行写入操作
    with open(SERIALIZED_PATTERN_PATH / f"{pattern_name}.py", write_mode) as f:
        if write_mode == "w":
            f.write(file_template)
        else:
            f.write("\n\n")
        f.write(serialized_pattern)
        f.write("\n")

    # 返回生成的模式对象
    return pattern


# 设置序列化模式文件的存储路径
SERIALIZED_PATTERN_PATH = Path(__file__).parent / "fx_passes" / "serialized_patterns"

# 已知的预编译模式列表，用于验证模式是否是最新的
_known_precompiled_patterns: List[
    Tuple[
        Any,
        Iterable[Any],
        Callable[[Callable[..., Any], Iterable[Any]], torch.fx.GraphModule],
        Any,
        PatternExpr,
    ]
] = []


def gen_register_replacement(
    unique_name: str,
    search_fn: SearchFn,
    replace_fn: ReplaceFn,
    example_inputs: Iterable[Any],
    trace_fn: TraceFn,
    pass_dicts: Union[_PassDictsType, Sequence[_PassDictsType]],
    extra_check: Callable[[Match], bool] = _return_true,
    scalar_workaround: Union[Dict[str, Union[float, int]], None] = None,
    exclusive_arg_names: Sequence[str] = (),
    skip_duplicates: bool = False,
# 定义函数，注册降阶模式匹配
def register_lowering_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = _return_true,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    注册一个 ATen 到 Inductor IR 替换模式的函数装饰器。
    装饰的函数将在运行时检查并执行模式匹配替换。

    Args:
    - pattern: 要注册的模式表达式。
    - extra_check: 可选参数，用于额外检查模式匹配的回调函数。
    - pass_dict: 包含传递字典的类型。
    - prepend: 可选参数，指示是否在前置模式匹配之前执行。

    Returns:
    - 装饰的函数，用于将模式注册到函数中。
    """
    function is saved and then called a lowering time allowing direct
    pattern to inductor IR conversion.
    """
    # 定义一个装饰器函数，接受一个处理函数作为参数，返回一个修饰过的处理函数
    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
        # 断言传入的处理函数是可调用的
        assert callable(handler)
        # 创建一个 LoweringPatternEntry 对象，并将其注册到 pass_dict 中
        LoweringPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        # 给处理函数添加一个属性 _inductor_lowering_function，表示它经过了下降模式转换
        handler._inductor_lowering_function = True  # type: ignore[attr-defined]
        # 返回经过修饰的处理函数
        return handler

    # 返回装饰器函数 decorator
    return decorator
# 注册一个图形模式，允许在FX图上运行函数，以进行自定义转换代码。
def register_graph_pattern(
    pattern: PatternExpr,
    extra_check: Callable[[Match], bool] = _return_true,
    *,
    pass_dict: _PassDictsType,
    prepend: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register a pattern that runs a function on the FX graph, allowing
    custom transformation code.
    """

    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
        assert callable(handler)
        # 创建图形模式条目并注册到指定的pass_dict中，可以选择在前面添加
        GraphPatternEntry(
            pattern=pattern, extra_check=extra_check, handler=handler
        ).register(pass_dict, prepend=prepend)
        return handler

    return decorator


# 检查节点是否为FX图中的起始节点
def is_start_of_fx_graph(graph: torch.fx.Graph, node: torch.fx.Node) -> bool:
    # FX图中的第一个节点
    return node is next(iter(graph.nodes))


# 正则表达式用于匹配特定的变异操作
_mutation_op_re = re.compile(r"_$|_[.]|(\b|_)(set|enter|exit|seed)(\b|_)")


# 检查节点是否为变异操作
def is_mutation_op(node: torch.fx.Node) -> bool:
    if node.op == "call_function":
        if _mutation_op_re.search(node.target.__name__):  # type: ignore[union-attr]
            return True
    elif node.op == "call_method":
        if _mutation_op_re.search(node.target):  # type: ignore[union-attr, arg-type]
            return True
    return node.kwargs.get("out") is not None


# 获取节点所属的变异区域ID
def get_mutation_region_id(graph: torch.fx.Graph, node: torch.fx.Node) -> int:
    n = node
    while "mutation_region_id" not in n.meta and not is_start_of_fx_graph(graph, n):
        n = n.prev
    # 获取节点的变异区域ID，如果不存在则默认为0
    mutation_region_id = n.meta.get("mutation_region_id", 0)
    while n is not node:
        n = n.next
        if is_mutation_op(n):
            mutation_region_id += 1
        n.meta["mutation_region_id"] = mutation_region_id
    return mutation_region_id


# 判断是否需要计算节点的变异区域ID
def should_compute_mutation_region_ids(graph: torch.fx.GraphModule) -> bool:
    return "mutation_region_id" not in next(iter(graph.nodes)).meta


# 计算所有节点的变异区域ID
def compute_mutation_region_ids(graph: torch.fx.GraphModule) -> None:
    mutation_region_id = 0
    for nd in graph.nodes:
        if is_mutation_op(nd):
            mutation_region_id += 1
        nd.meta["mutation_region_id"] = mutation_region_id


# 模式匹配器的类，用于管理模式和相关操作
class PatternMatcherPass:
    def __init__(
        self,
        prevent_match_across_mutations: bool = False,
        pass_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        # 使用默认字典存储模式列表，按照模式和FX节点目标分类
        self.patterns: DefaultDict[
            Tuple[str, torch.fx.node.Target], List[PatternEntry]
        ] = defaultdict(list)
        self.prevent_match_across_mutations = prevent_match_across_mutations
        self.pass_name = pass_name

    # 获取指定模式和FX节点目标的模式条目列表
    def __getitem__(self, item: Tuple[str, torch.fx.node.Target]) -> List[PatternEntry]:
        return self.patterns[item]

    # 清空所有模式
    def clear(self) -> None:
        self.patterns.clear()


# 用于未实现的占位函数
def _not_implemented(*args: Any, **kwargs: Any) -> NoReturn:
    raise NotImplementedError


# 将FX图或图形模块转换为模式的辅助函数
def fx_to_pattern(
    gm: Union[torch.fx.GraphModule, torch.fx.Graph],
    # 定义一个类型为 Sequence[Type[Any]] 的 ignore_types 变量，用于存储需要忽略的类型
    ignore_types: Sequence[Type[Any]] = (),
    # 定义一个类型为 Sequence[str] 的 argnames 变量，用于存储参数名列表
    argnames: Sequence[str] = (),
    # 定义一个类型为 Union[Dict[str, Union[float, int]], None] 的 scalar_workaround 变量，
    # 用于存储标量处理的特殊情况，可以是字典或者 None
    scalar_workaround: Union[Dict[str, Union[float, int]], None] = None,
    # 定义一个类型为 Sequence[str] 的 exclusive_arg_names 变量，用于存储排他性参数名列表
    exclusive_arg_names: Sequence[str] = (),
# 定义一个函数，将 FX 图转换为 PatternExpr 对象
def fx_to_pattern(
    gm: torch.fx.GraphModule,
    *,
    scalar_workaround: Optional[Dict[Any, Any]] = None,
    ignore_types: Set[Type] = {torch.Tensor},
    exclusive_arg_names: Set[str] = set(),
) -> PatternExpr:
    """
    Convert an FX graph into a PatternExpr.  This is useful for simple
    patterns that can only match single functions and fixed-length lists.
    """
    # scalar_workaround 是一个捕获 dropout_p 的临时解决方案
    # 参见 https://github.com/pytorch/pytorch/issues/97894
    scalar_workaround = scalar_workaround or {}
    # 创建一个逆映射，将值和键对调，用于后续处理
    inv_scalar_workaround = {v: k for k, v in scalar_workaround.items()}
    # 断言逆映射字典中键值对的数量和原字典相等
    assert len(inv_scalar_workaround) == len(scalar_workaround)

    # 定义一个处理参数的函数，根据参数类型返回相应的处理结果
    def process_arg(x: T) -> Union[T, KeywordArg, Ignored]:
        if isinstance(x, (float, int)) and x in inv_scalar_workaround:
            return KeywordArg(inv_scalar_workaround[x])
        if type(x) in ignore_types:
            return Ignored()
        if isinstance(x, list) and all(isinstance(y, Ignored) for y in x) and x:
            return Ignored()
        return x

    # 使用 itertools.count 创建一个计数器
    argnum = itertools.count()

    # 定义一个内部类 Converter，继承自 torch.fx.Interpreter
    class Converter(torch.fx.Interpreter):
        # 下面三个方法暂时未实现，用 _not_implemented 表示
        call_method = _not_implemented
        call_module = _not_implemented
        get_attr = _not_implemented

        # 定义 placeholder 方法，处理目标字符串、参数和关键字参数，返回关键字参数或独占关键字参数对象
        def placeholder(
            self, target: str, args: Sequence[Any], kwargs: Mapping[str, Any]
        ) -> Union[ExclusiveKeywordArg, KeywordArg]:
            n = next(argnum)
            if n < len(argnames):
                name = argnames[n]
            elif argnames:
                assert target.startswith("tangent")
                name = target
            else:
                target = re.sub(r"_\d+$", "", target)  # 解码参数名称
                name = target
            if name in exclusive_arg_names:
                return ExclusiveKeywordArg(name)
            else:
                return KeywordArg(name)

        # 定义 call_function 方法，处理目标字符串、参数和关键字参数，返回 PatternExpr 对象
        def call_function(
            self, target: str, args: Sequence[Any], kwargs: Mapping[str, Any]
        ) -> PatternExpr:
            # 对参数和关键字参数应用 process_arg 函数
            args, kwargs = pytree.tree_map(process_arg, (args, kwargs))
            if list in ignore_types:
                # 处理固定的张量大小，现在是 [Ignored(), Ignored(), ...]
                args = [process_arg(a) for a in args]
                kwargs = {k: process_arg(a) for k, a in kwargs.items()}
            return CallFunction(target, *args, **kwargs)

        # 定义 run_node 方法，运行给定节点并返回结果
        def run_node(self, n: torch.fx.Node) -> Any:
            rv = super().run_node(n)
            if n.op == "output" and isinstance(rv, tuple):
                assert len(rv) == len(n.args[0])  # 断言结果长度与参数一致
                for r, arg in zip(rv, n.args[0]):
                    r.users = len(arg.users)
            else:
                rv.users = len(n.users)
            return rv

    # 创建 Converter 类的实例，并运行它的 run 方法，得到转换后的 pattern
    pattern = Converter(gm).run()
    # 如果 pattern 不是 PatternExpr 对象，则返回 MultiOutputPattern 对象
    if not isinstance(pattern, PatternExpr):
        return MultiOutputPattern(pytree.tree_leaves(pattern))
    return pattern


# 定义装饰器 @torch.no_grad()，禁用梯度计算
@torch.no_grad()
# 定义函数 fwd_only，用于构建规范化的推断图，用于 fx_to_pattern 使用
def fwd_only(
    fn: Callable[..., Any], args: Sequence[Any], *, run_dce: bool = True
) -> torch.fx.GraphModule:
    """Build a normalized inference graph, for use with fx_to_pattern"""
    # 使用 enable_python_dispatcher() 启用 Python 调度器上下文
    with enable_python_dispatcher():
        # 使用 make_fx 函数创建一个 FunctionGraph 对象 gm，
        # 该函数基于给定的函数 fn、选择的分解表以及追踪模式 "real"，
        # 并使用 args 作为参数进行调用
        gm = make_fx(fn, select_decomp_table(), tracing_mode="real")(*args)
    
    # 从 fx_passes 模块导入 remove_noop_ops 函数
    from .fx_passes.post_grad import remove_noop_ops
    
    # 对 gm 对象的图形表示进行移除无操作的优化
    remove_noop_ops(gm.graph)
    
    # 如果 run_dce 为真，则执行图形表示中的死代码消除优化
    if run_dce:
        gm.graph.eliminate_dead_code()
    
    # 对 gm 对象进行重新编译
    gm.recompile()
    
    # 返回经过处理的 gm 对象
    return gm
@torch.enable_grad()
# 启用 Torch 梯度追踪环境的装饰器

def joint_fwd_bwd(fn: Callable[..., Any], args: Sequence[Any]) -> torch.fx.GraphModule:
    """Build a normalized training graph, for use with fx_to_pattern"""
    # 构建一个标准化的训练图，用于 fx_to_pattern 函数使用的注释

    gm: Optional[torch.fx.GraphModule] = None
    # 初始化一个可选的 Torch fx 图模块对象

    def record_joint_graph(
        joint_graph: torch.fx.GraphModule, inputs: Sequence[Any], **kwargs: Any
    ) -> Tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        nonlocal gm
        # 非本地的 gm 变量引用

        assert not gm
        # 断言确保 gm 不存在

        gm = clone_graph(joint_graph)
        # 克隆 joint_graph，并赋值给 gm

        return default_partition(joint_graph, inputs, **kwargs)
        # 返回默认分区函数应用于 joint_graph 的结果

    with torch._guards.tracing(None):
        # 在 Torch 的追踪上下文中禁用追踪

        aot_function(
            fn,
            lambda g, i: make_boxed_func(g),
            partition_fn=record_joint_graph,
            decompositions=select_decomp_table(),
            keep_inference_input_mutations=True,
            enable_log=False,
        )(*args)
        # 使用 aot_function 运行 fn 函数，并传入相关参数

    assert gm
    # 断言确保 gm 存在

    from .fx_passes.post_grad import remove_noop_ops
    # 从后向前的 Torch fx 后处理模块导入移除无操作的函数

    remove_noop_ops(gm.graph)
    # 移除 gm 图中的无操作节点

    from .fx_passes.joint_graph import pointless_view
    # 导入无意义视图处理函数

    matcher_pass = PatternMatcherPass()
    # 创建模式匹配器 Pass 对象

    pattern = CallFunction(
        torch.ops.aten.view.default, KeywordArg("arg"), KeywordArg("size")
    )
    # 创建调用 Torch.ops.aten.view.default 函数的模式

    GraphPatternEntry(
        pattern=pattern, handler=pointless_view, extra_check=_return_true
    ).register(matcher_pass.patterns)
    # 将模式、处理函数和额外检查函数注册到模式匹配器 Pass 的模式列表中

    matcher_pass.apply(gm.graph)  # type: ignore[arg-type]
    # 应用模式匹配器 Pass 到 gm 图中

    # remove in/out specs
    gm.graph._codegen = torch.fx.graph.CodeGen()
    # 重置 gm 图的代码生成器对象

    gm.graph.eliminate_dead_code()
    # 消除 gm 图中的死代码

    gm.recompile()
    # 重新编译 gm

    return gm
    # 返回 gm 图模块对象


def _args(n: torch.fx.Node) -> List[torch.fx.node.Argument]:
    # 返回一个 Torch fx 节点的参数列表

    args: List[torch.fx.node.Argument] = list()
    # 初始化一个空的 Torch fx 节点参数列表

    torch.fx.map_arg((n.args, n.kwargs), args.append)
    # 将节点 n 的参数映射到 args 列表中

    return args
    # 返回参数列表


def stable_topological_sort(graph: torch.fx.Graph) -> None:
    # 对 Torch fx 图进行稳定的拓扑排序

    # Nodes are in exactly one of these three collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))
    # 初始化待处理节点列表，以反向顺序存储图中的所有节点

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = set()
    # 初始化已处理节点集合

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)
    # 初始化等待字典，用于映射依赖关系到依赖于该依赖关系的节点列表

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    # 初始化游标，用于指示最后处理的节点，以便在其后添加新节点

    while pending:
        # 循环处理待处理节点列表
        node = pending.pop()
        # 弹出一个节点进行处理

        waiting_for = [x for x in _args(node) if x not in ready]
        # 获取当前节点等待的尚未处理完成的输入节点列表

        if waiting_for:
            # 如果存在未处理的输入节点
            waiting[waiting_for[-1]].append(node)
            # 将当前节点加入到最后一个待等待的输入节点依赖列表中
        else:
            ready.add(node)
            # 将当前节点标记为已处理

            if cursor and cursor.next is not node:
                cursor.append(node)
                # 如果存在游标并且游标的下一个节点不是当前节点，则将当前节点添加到游标的后面

            cursor = node
            # 更新游标为当前节点

            pending.extend(reversed(waiting.pop(node, ())))
            # 将等待当前节点完成的节点列表逆序添加到待处理列表中
    # 断言条件：确保等待队列为空并且就绪队列的长度等于图中节点的数量
    assert not waiting and len(ready) == len(graph.nodes)
def init_once_fakemode(fn: Callable[..., Any]) -> Callable[[], Any]:
    """Wrapper around lazy init functions in fx_passes/"""

    @functools.lru_cache(None)  # 使用 functools 提供的 lru_cache 进行结果缓存
    @functools.wraps(fn)  # 保留原始函数的元数据信息
    def lazy_init() -> Any:
        counters_ref = counters["inductor"].copy()  # 复制 counters 字典中 "inductor" 键对应的值

        with torch._guards.tracing(  # 使用 torch._guards.tracing 上下文管理器
            None
        ), maybe_disable_fake_tensor_mode(), FakeTensorMode():  # 调用 maybe_disable_fake_tensor_mode() 和 FakeTensorMode() 函数
            result = fn()  # 调用传入的 fn 函数获取结果

        # 清空在追踪过程中遇到的视图匹配
        counters["inductor"] = counters_ref  # 恢复 counters 中 "inductor" 键对应的值

        return result  # 返回函数 fn 的执行结果

    return lazy_init  # 返回 lazy_init 函数对象


def config_flag(name: str) -> Callable[[Match], Any]:
    """Function for extra_check to put pass behind a flag"""

    def flag_check(match: Match) -> Any:
        return getattr(config, name)  # 获取 config 对象中 name 参数对应的属性值

    return flag_check  # 返回 flag_check 函数对象


def clone_graph(input_graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
    class CopyGraph(Transformer):
        def run_node(self, old_node: torch.fx.Node) -> torch.fx.Node:
            new_node = super().run_node(old_node)  # 调用父类 Transformer 的 run_node 方法
            if isinstance(new_node, torch.fx.Proxy):  # 判断 new_node 是否为 torch.fx.Proxy 类型
                new_node.node.meta.update(old_node.meta)  # 更新 new_node 的元数据信息
                new_node.node.name = self.new_graph._graph_namespace.create_name(
                    old_node.name, None
                )  # 根据 old_node 的名称创建新节点名称
            return new_node  # 返回处理后的新节点对象

    return CopyGraph(input_graph).transform()  # 返回 CopyGraph 实例对 input_graph 进行转换后的结果


_seen_patterns: Set[str] = set()  # 创建一个空的集合 _seen_patterns 用于存储字符串模式


def get_arg_value(
    node: torch.fx.Node, arg_number: int, kwarg_name: Optional[str] = None
) -> Any:
    return (
        node.args[arg_number]  # 获取 node 的第 arg_number 个参数
        if len(node.args) > arg_number  # 如果 node 的参数数量大于 arg_number
        else node.kwargs.get(kwarg_name)  # 否则获取 node 的 kwargs 中 kwarg_name 对应的值（如果存在）
    )


def filter_nodes(nodes: Iterable[torch.fx.Node], fn: Any) -> List[torch.fx.Node]:
    fns = [fn]  # 将 fn 参数转换为列表 fns
    if isinstance(fn, torch._ops.OpOverloadPacket):  # 判断 fn 是否为 torch._ops.OpOverloadPacket 类型
        fns.extend([getattr(fn, overload) for overload in fn.overloads()])  # 如果是，则扩展 fns 列表

    return [node for node in nodes if node.target in fns]  # 返回节点列表 nodes 中，目标函数在 fns 中的节点


def extract_target(node: torch.fx.Node) -> torch.fx.node.Target:
    """For call_function and call_method, we directly use the target function;
    For call_module, the target is string, and we treat the module class
     as a function.
    """
    if node.op == "call_module":  # 如果节点操作是 "call_module"
        return getattr(node.graph.owning_module, node.target).__class__  # 返回节点所属模块的目标属性的类对象
    return node.target  # 否则返回节点的目标函数对象或字符串
```