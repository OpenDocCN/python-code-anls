# `.\pytorch\torch\fx\passes\operator_support.py`

```py
# 声明一个静态类型检查的配置，允许未标注的函数定义
# Allow untyped definitions for mypy static type checker
# 导入抽象基类模块和类型提示模块
import abc
import typing as t

# 导入PyTorch相关模块
import torch
import torch.fx

# 导入兼容性模块中的compatibility函数
from torch.fx._compatibility import compatibility

# 导入本地模块：TensorMetadata类和一些工具函数
from .shape_prop import TensorMetadata
from .tools_common import get_node_target, CALLABLE_NODE_OPS

# __all__列表，包含模块中可以导出的公共名称
__all__ = ['OperatorSupportBase', 'OperatorSupport', 'create_op_support', 'chain', 'OpSupports', 'any_chain']

# 定义目标节点类型名称的别名，由get_node_target()函数返回
TargetTypeName = str

# 支持的节点参数数据类型的定义，用于给定节点，参见OperatorSupport类
SupportedArgumentDTypes = t.Optional[
    t.Tuple[
        t.Sequence[t.Sequence[torch.dtype]],  # 输入参数的数据类型列表
        t.Dict[str, t.Sequence[torch.dtype]], # 关键字参数的数据类型字典
    ]
]

# 支持信息字典类型的定义，映射目标节点类型名到支持的参数数据类型
SupportDict = t.Mapping[TargetTypeName, SupportedArgumentDTypes]


@compatibility(is_backward_compatible=False)
# OperatorSupportBase类，继承自abc.ABC，用于确定fx.Node是否被后端支持
class OperatorSupportBase(abc.ABC):
    """Interface for determining if a fx.Node is supported by a backend"""

    @abc.abstractmethod
    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        raise NotImplementedError


@compatibility(is_backward_compatible=False)
# OperatorSupport类，继承自OperatorSupportBase类
class OperatorSupport(OperatorSupportBase):
    """
    `_support_dict`将节点目标类型名映射到支持的输入数据类型。

    使用helper函数`get_node_target()`获取节点目标类型名。

    如果支持的输入数据类型为None，则表示支持任何数据类型；否则应该看到类似于
    (([dtypes], ...), {"name":[dtypes], ...}) 的元组结构。

    第一个元组 ([dtypes], ...) 表示节点args中支持的数据类型，第二个字典
    {"name": [dtypes], ...} 表示节点kwargs中支持的数据类型。

    对于args中的输入，如果我们不想检查它，可以放置None，例如 (None, [torch.float])
    表示我们不关心args中的第一个输入的类型。对于kwargs中的输入，如果未列出，将不进行检查。
    """

    _support_dict: SupportDict

    def __init__(
        self,
        support_dict: t.Optional[SupportDict] = None
    ):
        self._support_dict = support_dict or {}

    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        # 在这里实现了OperatorSupportBase中定义的抽象方法is_node_supported
    ) -> bool:
        """
        Args:
            `submodules`: 模块名称到模块对象的映射，可以通过调用 model.named_modules() 获取。

            `node`: 我们要确定其是否受支持的 Fx 节点。

        Returns:
            `is_supported`: 参数 `node` 是否受支持。
        """
        # 如果节点的操作不在 CALLABLE_NODE_OPS 中，则认为支持
        if node.op not in CALLABLE_NODE_OPS:
            return True

        # 获取节点目标
        target = get_node_target(submodules, node)

        # 如果目标不在 self._support_dict 中，表示我们根本不支持此操作
        if target not in self._support_dict:
            return False

        # 如果 self._support_dict 中对应目标的规则为 None，则表示接受任何 dtype
        if self._support_dict[target] is None:
            return True

        # 获取参数和关键字参数的 dtype 规则
        args_dtypes, kwargs_dtypes = self._support_dict[target]  # type: ignore[misc]

        # 检查参数的 dtype
        for i, dtypes in enumerate(args_dtypes):
            if len(node.args) <= i:
                break

            # None 表示我们不关心 args[i] 的 dtype
            if dtypes is None:
                continue

            # 如果参数不是节点，则不进行检查
            if not isinstance(node.args[i], torch.fx.Node):
                continue

            # 获取参数的 dtype
            arg_dtype = _get_arg_dtype(node.args[i])  # type: ignore[arg-type]
            if arg_dtype not in dtypes:
                return False

        # 检查关键字参数的 dtype
        for k, dtypes in kwargs_dtypes.items():
            if k not in node.kwargs:
                continue

            # 如果参数不是节点，则不进行检查
            if not isinstance(node.kwargs[k], torch.fx.Node):
                continue

            # 获取关键字参数的 dtype
            kwarg_dtype = _get_arg_dtype(node.kwargs[k])  # type: ignore[arg-type]
            if kwarg_dtype not in dtypes:
                return False

        # 如果所有检查通过，则认为支持
        return True
# ======================================================================
# Functional interfaces and utils for defining basic operator support logic
# and composing them into more complex ones
# ======================================================================

# 定义一个类型别名，表示判断节点是否支持的函数签名
IsNodeSupported = t.Callable[[t.Mapping[str, torch.nn.Module], torch.fx.Node], bool]

# 创建一个操作支持类的实例，将一个 `IsNodeSupported` 函数封装成 `OperatorSupportBase` 实例
@compatibility(is_backward_compatible=False)
def create_op_support(is_node_supported: IsNodeSupported) -> OperatorSupportBase:
    """Wraps a `IsNodeSupported` function into an `OperatorSupportBase` instance

    `IsNodeSupported` has the same call signature as
    `OperatorSupportBase.is_node_supported`
    """
    # 定义一个内部类，继承自 `OperatorSupportBase`
    class FunctionalOperatorSupport(OperatorSupportBase):
        # 实现 `is_node_supported` 方法，调用传入的 `is_node_supported` 函数
        def is_node_supported(
                self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
        ) -> bool:
            return is_node_supported(submodules, node)
    return FunctionalOperatorSupport()

# 定义一个函数，将一系列 `OperatorSupportBase` 实例组合成单个 `OperatorSupportBase` 实例
@compatibility(is_backward_compatible=False)
def chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns False if
    any of it reports False.
    """
    # 定义一个内部函数 `_chain`，用于组合多个操作支持实例的判断逻辑
    def _chain(submods, node) -> bool:
        return all(
            x.is_node_supported(submods, node)
            for x in op_support
        )
    return create_op_support(_chain)

# 定义一个函数，将一系列 `OperatorSupportBase` 实例组合成单个 `OperatorSupportBase` 实例
@compatibility(is_backward_compatible=False)
def any_chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns True if
    any of it reports True.
    """
    # 定义一个内部函数 `_any_chain`，用于组合多个操作支持实例的判断逻辑
    def _any_chain(submods, node) -> bool:
        return any(
            x.is_node_supported(submods, node)
            for x in op_support
        )
    return create_op_support(_any_chain)

# 定义一个类 `OpSupports`，用于组合原子的 `OperatorSupportBase` 实例以构建更复杂的操作支持逻辑
@compatibility(is_backward_compatible=False)
class OpSupports:
    """A set of atomic `OperatorSupportBase` instances that can be combined together
    to form more complex operator support logic.
    """
    # 类方法，返回一个 `OperatorSupportBase` 实例，用于检查节点是否支持，如果节点的任何输入参数是指定的 dtype，则不支持
    @classmethod
    def decline_if_input_dtype(cls, dtype: torch.dtype) -> OperatorSupportBase:
        """Report a node as non-supported, if any of its arguments is of dtype"""

        # 内部函数 `_decline_if_input_dtype`，判断节点的所有输入参数的 dtype 是否与指定的 dtype 相同
        def _decline_if_input_dtype(
            submodules: t.Mapping[str, torch.nn.Module],
            node: torch.fx.Node,
        ) -> bool:
            for arg in node.all_input_nodes:
                arg_dtype = _get_arg_dtype(arg)  # 获取参数的 dtype
                if arg_dtype == dtype:  # 如果参数的 dtype 与指定的 dtype 相同，则返回 False
                    return False
            return True  # 如果所有参数的 dtype 都不是指定的 dtype，则返回 True
        return create_op_support(_decline_if_input_dtype)
    # 定义一个类方法，用于创建一个操作支持基类的实例，检查节点名称是否在禁止集合中
    def decline_if_node_in_names(cls, disallow_set: t.Set[str]) -> OperatorSupportBase:
        """
        If a node has a name that is in the disallow set, reported it as non-supported.
        如果节点的名称在禁止集合中，则报告为不受支持的节点。
        """
        # 定义内部函数 _decline_if_node_in_names，用于检查节点名称是否在禁止集合中
        def _decline_if_node_in_names(
            submodules: t.Mapping[str, torch.nn.Module],
            node: torch.fx.Node,
        ) -> bool:
            # 如果节点名称在禁止集合中，则返回 False 表示不支持该节点
            if node.name in disallow_set:
                return False
            else:
                # 否则返回 True 表示支持该节点
                return True
        # 调用 create_op_support 函数，以 _decline_if_node_in_names 函数作为参数创建操作支持实例，并返回
        return create_op_support(_decline_if_node_in_names)
# 定义一个函数 _get_arg_dtype，用于从 torch.fx.Node 中获取参数的数据类型
def _get_arg_dtype(arg: torch.fx.Node) -> t.Any:
    # 断言参数 arg 是 torch.fx.Node 类型的对象
    assert isinstance(arg, torch.fx.Node)
    # 从参数 arg 的元数据中获取 "tensor_meta" 属性，类型可能是 TensorMetadata
    tensor_meta = arg.meta.get("tensor_meta")  # type: ignore[union-attr]
    # 如果 tensor_meta 是 TensorMetadata 类型的对象，则获取其 dtype 属性作为数据类型 dtype
    # 否则，从 arg 的元数据中直接获取 "type" 属性作为数据类型 dtype
    dtype = tensor_meta.dtype if isinstance(tensor_meta, TensorMetadata) else arg.meta["type"]
    # 返回确定的数据类型 dtype
    return dtype
```