# `.\pytorch\torch\fx\experimental\normalize.py`

```py
# mypy: allow-untyped-defs
# 导入所需的模块和类型
import operator  # 导入operator模块，用于操作符相关功能
from typing import Any, Callable, Dict, Tuple, Optional  # 导入类型提示相关模块

import torch  # 导入PyTorch库
import torch.fx  # 导入PyTorch的特定功能模块
import torch.fx as fx  # 导入PyTorch的特定功能模块，并使用别名fx
from torch.fx import Transformer, Proxy  # 从torch.fx模块导入Transformer和Proxy类
from torch.fx.node import Argument, Target, Node, map_aggregate  # 从torch.fx.node模块导入Argument、Target、Node和map_aggregate类
from torch.fx.operator_schemas import (  # 从torch.fx.operator_schemas模块导入指定函数
    normalize_module,
    normalize_function,
    create_type_hint,
)

from .schema_type_annotation import AnnotateTypesWithSchema  # 从相对路径导入模块schema_type_annotation中的AnnotateTypesWithSchema类


class NormalizeArgs(Transformer):
    """
    Normalize arguments to Python targets. This means that
    `args/kwargs` will be matched up to the module/functional's
    signature and rewritten to exclusively kwargs in positional order
    if `normalize_to_only_use_kwargs` is true. Also populates default
    values. Does not support positional-only parameters or varargs
    parameters (*args, **kwargs).

    If the nodes have 'type' metadata, it will use it to disambiguate
    overloads. Otherwise, it will throw an error.

    Example usage:
        m = torchvision.models.resnet18()
        traced = torch.fx.symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()
    """

    def __init__(
        self, module: torch.fx.GraphModule, normalize_to_only_use_kwargs: bool = True
    ):
        super().__init__(module)
        self.node_map: Dict[Proxy, Node] = {}  # 初始化节点映射字典，用于存储Proxy和Node之间的映射关系
        self.normalize_to_only_use_kwargs = normalize_to_only_use_kwargs  # 设置是否仅使用kwargs来归一化参数的标志

    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)  # 从环境中获取节点n的参数args和kwargs

        def get_type(arg):
            if isinstance(arg, fx.Node):
                return n.meta["type"] if "type" in n.meta else None  # 如果arg是fx.Node类型，则返回其metadata中的type信息
            return type(arg)  # 否则返回arg的类型

        arg_types = map_aggregate(n.args, get_type)  # 对节点n的参数args中的每个参数获取类型信息并聚合
        assert isinstance(arg_types, tuple)  # 确保arg_types是元组类型
        arg_types = tuple([create_type_hint(i) for i in arg_types])  # 使用每个参数的类型信息创建类型提示
        kwarg_types = {k: get_type(v) for k, v in kwargs.items()}  # 获取kwargs中每个参数的类型信息

        if n.op == "call_function":
            out = self.call_function(n.target, args, kwargs, arg_types, kwarg_types)  # 如果节点n是调用函数操作，则调用call_function方法
        else:
            out = super().run_node(n)  # 否则调用父类的run_node方法处理节点n

        if n.op != "output":
            self.node_map[out] = n  # 将输出out和节点n建立映射关系
            out.node.meta = n.meta  # 将输出out的节点的metadata设置为节点n的metadata
            out.node.type = n.type  # 将输出out的节点的类型设置为节点n的类型

        return out  # 返回处理后的输出

    def call_function(
        self,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Any],
        arg_types: Optional[Tuple[Any, ...]] = None,
        kwarg_types: Optional[Dict[str, Any]] = None,
    ):
        # 确保目标函数是可调用的
        assert callable(target)
        # 标准化函数参数和关键字参数，确保类型一致性
        new_args_and_kwargs = normalize_function(
            target,
            args,  # type: ignore[arg-type] - 忽略类型检查错误
            kwargs,
            arg_types,  # type: ignore[arg-type] - 忽略类型检查错误
            kwarg_types,
            self.normalize_to_only_use_kwargs,
        )
        # 如果成功标准化参数和关键字参数
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            # 创建函数调用的代理并返回
            return self.tracer.create_proxy(
                "call_function", target, new_args, new_kwargs
            )
        else:
            # 如果无法标准化，调用超类的函数
            return super().call_function(target, args, kwargs)

    def call_module(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ):
        # 确保目标是字符串
        assert isinstance(target, str)
        # 标准化模块参数和关键字参数，确保类型一致性
        new_args_and_kwargs = normalize_module(
            self.module,
            target,
            args,  # type: ignore[arg-type] - 忽略类型检查错误
            kwargs,
            self.normalize_to_only_use_kwargs,
        )
        # 如果成功标准化参数和关键字参数
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            # 调用超类的模块调用函数
            return super().call_module(target, new_args, new_kwargs)
        else:
            # 如果无法标准化，调用超类的模块调用函数
            return super().call_module(target, args, kwargs)
class NormalizeOperators(AnnotateTypesWithSchema):
    """
    Normalize callsites that are different ways of "spelling" the same
    invocation into a single, canonical call. Currently supports:

    1. Normalize operators (e.g. operator.add) to the `torch` ops they
       ultimately invoke (e.g. torch.add) when it is possible to statically
       reason that

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = NormalizeOperators(traced).transform()
    """

    # 字典，用于将 torch 操作映射到对应的 operator 操作
    binary_magic_method_remap: Dict[
        Callable[[Any, Any], Any], Callable[[Any, Any], Any]
    ] = {
        torch.add: operator.add,
        torch.mul: operator.mul,
        torch.sub: operator.sub,
        torch.div: operator.truediv,
        torch.floor_divide: operator.floordiv,
        torch.remainder: operator.mod,
        torch.eq: operator.eq,
        torch.ne: operator.ne,
        torch.lt: operator.lt,
        torch.le: operator.le,
        torch.gt: operator.gt,
        torch.ge: operator.ge,
    }

    def call_function(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ):
        # 根据张量的魔术方法规范化操作符
        # 参考链接: https://github.com/pytorch/pytorch/blob/28c5d90b679c6b38bf4183ec99f16d933c2f1bcd/tools/autograd/templates/python_variable_methods.cpp#L1137 # noqa: B950

        assert callable(target)

        # 如果目标函数在映射字典中
        if target in self.binary_magic_method_remap:
            # 如果参数数量不为2，则调用父类方法
            if len(args) != 2:
                return super().call_function(target, args, kwargs)
            lhs, rhs = args

            # 调用父类方法，使用映射后的操作符
            return super().call_function(
                target=self.binary_magic_method_remap[target],
                args=(lhs, rhs),
                kwargs={},
            )

        # 如果目标函数不在映射字典中，则调用父类方法
        return super().call_function(target, args, kwargs)
```