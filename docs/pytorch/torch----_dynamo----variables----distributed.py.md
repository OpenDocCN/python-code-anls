# `.\pytorch\torch\_dynamo\variables\distributed.py`

```
# 忽略类型检查错误，通常用于类型检查器忽略特定的错误或警告
# 声明 functools 模块，用于高阶函数操作
# 声明 inspect 模块，用于对象内省（introspection）
# 声明 Dict 和 List 用于类型提示
import functools
import inspect
from typing import Dict, List

# 导入 PyTorch 库
import torch
# 导入 BackwardState 类，用于跟踪自动求导的反向状态
from ...fx.experimental._backward_state import BackwardState
# 从 .. 包中导入 compiled_autograd 和 variables 模块
from .. import compiled_autograd, variables
# 从 .._trace_wrapped_higher_order_op 模块导入 trace_wrapped 函数
from .._trace_wrapped_higher_order_op import trace_wrapped
# 从 ..exc 模块导入 unimplemented 异常
from ..exc import unimplemented
# 从 ..external_utils 模块导入 call_module_hooks_from_backward_state 函数
from ..external_utils import call_module_hooks_from_backward_state
# 从 ..guards 模块导入 GuardBuilder 和 install_guard 函数
from ..guards import GuardBuilder, install_guard
# 从 ..source 模块导入 AttrSource 类
from ..source import AttrSource
# 从 ..utils 模块导入 istype 函数
from ..utils import istype
# 从 .base 模块导入 VariableTracker 类
from .base import VariableTracker
# 从 .constant 模块导入 ConstantVariable 类

from .constant import ConstantVariable


class DistributedVariable(VariableTracker):
    """
    The base distributed variable that encapsulates common methods
    for the distributed objects (i.e. ProcessGroup, DeviceMesh, etc.).
    Concrete distributed objects could inherit this class and add object
    specific logic.

    i.e. It provides the check on the distributed package existance
    and hold the tracking value for the corresponding distributed object.
    """

    def __init__(self, value, **kwargs):
        # 调用父类 VariableTracker 的构造函数
        super().__init__(**kwargs)
        # 如果 torch.distributed 包不可用，抛出未实现异常
        if not DistributedVariable.is_available():
            unimplemented("torch.distributed package is not available!")
        # 初始化分布式变量的值
        self.value = value

    def python_type(self):
        # 返回变量值的 Python 类型
        return type(self.value)

    @staticmethod
    def is_available():
        # 检查 torch.distributed 包是否可用
        return torch.distributed.is_available()


def is_from_local(value):
    # 如果 torch.distributed 包不可用，返回 False
    if not DistributedVariable.is_available():
        return False
    # 从 torch.distributed._tensor 模块导入 DTensor 类
    from torch.distributed._tensor import DTensor
    # 检查 value 是否为 DTensor 类型的 from_local 方法
    return inspect.isfunction(value) and value is DTensor.from_local


def is_constant_pg_functions(value):
    # 如果 torch.distributed 包不可用，返回 False
    if not DistributedVariable.is_available():
        return False

    # 从 torch.distributed.distributed_c10d 模块导入若干常量处理组函数
    from torch.distributed.distributed_c10d import (
        _get_group_size_by_name,
        _get_group_tag,
        _rank_not_in_group,
        _resolve_group_name_by_ranks_and_tag,
        get_process_group_ranks,
    )

    # 定义常量处理组函数列表
    constant_processgroup_functions = [
        _get_group_size_by_name,
        _get_group_tag,
        _rank_not_in_group,
        get_process_group_ranks,
        _resolve_group_name_by_ranks_and_tag,
    ]

    # 检查 value 是否为常量处理组函数列表中的函数
    return inspect.isfunction(value) and value in constant_processgroup_functions


class WorldMetaClassVariable(DistributedVariable):
    """
    Tracks torch.distributed.GroupMember and torch.distributed.group, which are
    instances of the metaclass _WorldMeta.
    """

    @classmethod
    def is_group_member_type(cls, value):
        # 如果 torch.distributed 包不可用，返回 False
        if not cls.is_available():
            return False

        # 从 torch.distributed.distributed_c10d 模块导入 _WorldMeta 类
        from torch.distributed.distributed_c10d import _WorldMeta

        # 检查 value 是否为 _WorldMeta 类型
        return type(value) is _WorldMeta
    # 定义一个方法 `var_getattr`，接收参数 `self`, `tx`, `name`，返回类型为 `VariableTracker`
    def var_getattr(self, tx, name: str) -> VariableTracker:
        # 检查 `name` 是否等于 "WORLD"
        if name == "WORLD":
            # 如果是 "WORLD"，创建一个 `AttrSource` 对象，基于当前对象的 `source` 属性和 "WORLD" 成员
            source = AttrSource(base=self.source, member="WORLD")
            # 调用 `source` 对象的方法 `make_guard`，并生成一个 ID 匹配类型的保护器，安装它
            install_guard(source.make_guard(GuardBuilder.ID_MATCH))
            # 返回一个 `ProcessGroupVariable` 对象，使用 `self.value.WORLD` 作为其参数
            return ProcessGroupVariable(self.value.WORLD)
        # 如果 `name` 不是 "WORLD"，调用父类的 `var_getattr` 方法来处理
        return super().var_getattr(tx, name)
class PlacementClassVariable(DistributedVariable):
    @staticmethod
    def is_placement_type(value):
        # 检查是否分布式变量可用，因为 torch distributed 并非始终构建
        if not DistributedVariable.is_available():
            return False

        # 动态导入 Placement 类型
        from torch.distributed._tensor.placement_types import Placement

        # 检查 value 是否为类型且是 Placement 的子类
        return type(value) is type and issubclass(value, Placement)

    def as_python_constant(self):
        # 返回当前变量的 Python 常量值
        return self.value

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        if (
            inspect.getattr_static(self.value, "__new__", None) in (object.__new__,)
            and self.source
        ):
            # 注意：不需要追踪对放置类的变异，因为它们应该是不可变的。
            # 创建一个新对象
            new_obj = object.__new__(self.value)
            # 创建一个放置变量对象
            var = PlacementVariable(new_obj)
            if inspect.getattr_static(self.value, "__init__", None):
                # 调用对象的 __init__ 方法
                var.call_method(tx, "__init__", args, kwargs)
                return var

        # 调用父类的 call_function 方法
        return super().call_function(tx, args, kwargs)


class PlacementVariable(DistributedVariable):
    @staticmethod
    def is_placement(value):
        # 检查是否分布式变量可用，因为 torch distributed 并非始终构建
        if not DistributedVariable.is_available():
            return False

        # 动态导入 Placement 类型
        from torch.distributed._tensor.placement_types import Placement

        # 检查 value 是否是 Placement 类的实例
        return isinstance(value, Placement)

    def as_python_constant(self):
        # 返回当前变量的 Python 常量值
        return self.value

    def var_getattr(self, tx, name: str) -> VariableTracker:
        if name == "dim":
            # 如果属性名为 "dim"，则创建一个表示常量的变量追踪器
            return ConstantVariable.create(self.value.dim)
        # 调用父类的 var_getattr 方法
        return super().var_getattr(tx, name)
    
    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # TODO: 添加 call_method 方法的注释
    ) -> "VariableTracker":
        from . import ConstantVariable

        # 定义了仅允许动态追踪的常量折叠方法列表
        # __setattr__ 方法用于像 `Shard(dim)` 这样的情况以及方法。
        # 列表中的方法必须满足以下条件：
        #    1. 输入参数是常量，无需进行保护；
        #    2. 输出与其输入参数相关，是常量。
        constant_fold_functions = [
            "__init__",
            "__setattr__",
            "is_shard",
            "is_partial",
            "is_replicate",
        ]

        # 如果名称在常量折叠方法列表中
        if name in constant_fold_functions:
            try:
                # 获取值的类型
                value_type = type(self.value)
                # 确保值的类型没有自定义的 getattr 方法
                assert (
                    inspect.getattr_static(value_type, "__getattr__", None) is None
                ), "no custom getattr allowed!"
                # 获取指定名称的方法
                method = inspect.getattr_static(value_type, name)
            except AttributeError:
                method = None

            # 如果方法是 object.__init__
            if method is object.__init__:
                return ConstantVariable.create(None)

            # 将参数转换为其 Python 常量值
            args = [x.as_python_constant() for x in args]
            kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}

            # 如果方法是 __setattr__
            if name == "__setattr__":
                # 调用方法设置属性值
                method(self.value, *args, **kwargs)
                return self

            # 调用方法获取常量值
            constant_val = method(self.value, *args, **kwargs)
            return ConstantVariable.create(constant_val)

        # 如果名称不在常量折叠方法列表中，则调用父类的方法
        return super().call_method(tx, name, args, kwargs)
class DeviceMeshVariable(DistributedVariable):
    @staticmethod
    def is_device_mesh(value):
        # 如果无法依赖导入或访问 torch distributed，因为它并非始终构建。
        if not DistributedVariable.is_available():
            return False

        # 导入 DeviceMesh 类
        from torch.distributed.device_mesh import DeviceMesh

        # 检查 value 是否为 DeviceMesh 类型
        return istype(value, DeviceMesh)

    def as_python_constant(self):
        # 返回当前变量的 Python 常量值
        return self.value

    def var_getattr(self, tx, name: str) -> VariableTracker:
        # 如果属性名为 "ndim"
        if name == "ndim":
            # 创建一个表示值的常量变量并返回，此处是 value 的维度数
            return ConstantVariable.create(self.value.ndim)
        # 如果属性名为 "device_type"
        if name == "device_type":
            # 创建一个表示值的常量变量并返回，此处是 value 的设备类型
            return ConstantVariable.create(self.value.device_type)
        # 调用父类方法处理其他属性名
        return super().var_getattr(tx, name)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # 如果方法名为 "size"
        if name == "size":
            # 提取参数和关键字参数的 Python 常量值
            const_args = [x.as_python_constant() for x in args]
            const_kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
            # 调用 value 对象的 size 方法并返回其结果的常量变量
            return ConstantVariable.create(self.value.size(*const_args, **const_kwargs))
        # 如果方法名为 "get_coordinate"
        if name == "get_coordinate":
            # 创建一个表示值的常量变量并返回，此处是调用 value 的 get_coordinate 方法
            return ConstantVariable.create(self.value.get_coordinate())
        # 如果方法名为 "get_group"
        if name == "get_group":
            # 创建一个表示值的常量变量并返回，此处是调用 value 的 get_group 方法
            return ConstantVariable.create(self.value.get_group())
        # 如果方法名为 "_get_or_create_default_group"
        if name == "_get_or_create_default_group":
            # 创建一个 ProcessGroupVariable 对象来表示 value 的 _get_or_create_default_group 方法的结果
            return ProcessGroupVariable(self.value._get_or_create_default_group())
        # 调用父类方法处理其他方法名
        return super().call_method(tx, name, args, kwargs)


class ProcessGroupVariable(DistributedVariable):
    """
    We don't want a ProcessGroup object to end up in our output graph.

    But it's common for dynamo to intercept a PG that is then used to get info like
    rank() or world_size(), as well as passed to utility functions in distributed_c10d
    which desugar it into plain types like a ranklist and tag.

    For convenience and proper guarding, we construct a variable type.

    TODO: make it possible to use ProcessGroupVariable as input to simple functions
          like _expand_group without dynamo complaining about making a proxy for it.
          It is not a tensor-like type, and we don't want a proxy- but dynamo assumes
          torch library functions are dealing with tensor-like types and would have proxies
          for their args.
    TODO: should we make this inherit VT instead of UDOV? Do we want any of the default behaviors
          or just graph-break whenever one of our special cases is not hit?
    """

    def as_python_constant(self):
        # 返回当前变量的 Python 常量值
        return self.value

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # 这个类中的 call_method 方法主要用于处理特定方法的调用，并返回相应的常量变量或对象
        # 根据方法名不同，返回对应方法的结果或处理
    ) -> "VariableTracker":
        # 如果属性名为 "rank"，返回一个常量变量，其值为 self.value.rank() 的结果
        if name == "rank":
            return variables.ConstantVariable.create(self.value.rank())
        # 如果属性名为 "size"，返回一个常量变量，其值为 self.value.size() 的结果
        if name == "size":
            return variables.ConstantVariable.create(self.value.size())

        # 对于其它属性名，调用父类的方法处理
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name):
        # 如果属性名为 "group_name"，返回一个常量变量，其值为 self.value.group_name 的结果
        if name == "group_name":
            return variables.ConstantVariable.create(self.value.group_name)
        # 如果属性名为 "rank" 或 "size"，返回一个 lambda 变量，调用 self.call_method 方法
        if name in ["rank", "size"]:
            return variables.LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
            )
        # TODO 是否应该只是抛出未实现的异常？
        # 对于其它属性名，调用父类的 var_getattr 方法处理
        return super().var_getattr(tx, name)

    @staticmethod
    def is_process_group(value):
        # 我们不能依赖于导入或访问 torch distributed，因为它并不总是构建的。
        # 如果 DistributedVariable 不可用，返回 False
        if not DistributedVariable.is_available():
            return False
        # 导入 ProcessGroup 和 FakeProcessGroup 类
        from torch._C._distributed_c10d import ProcessGroup
        from torch.testing._internal.distributed.fake_pg import FakeProcessGroup

        # 检查 value 是否为 ProcessGroup 或 FakeProcessGroup 类型的实例
        return istype(value, (ProcessGroup, FakeProcessGroup))
class BackwardHookVariable(VariableTracker):
    """
    Handles torch.utils.hooks.BackwardHook for module-level backward
    hooks.
    """

    @staticmethod
    def create(
        tx,
        module: VariableTracker,
        user_hooks: VariableTracker,
        user_pre_hooks: VariableTracker,
    ):
        # 检查是否启用了编译后自动求导
        if not compiled_autograd.compiled_autograd_enabled:
            unimplemented("module-level backwards hooks require compiled autograd")

        def _in_graph_bw_hooks(bw_state: BackwardState):
            """
            Rather than installing the user hooks in the graph (which
            don't survive AotAutograd), we install hooks that will call
            trace_wrapped in the backward pass that CompiledAutograd
            can turn into actual hook calls.
            """
            # 创建并返回一个 BackwardHook 对象，用于在反向传播过程中调用模块钩子函数
            return torch.utils.hooks.BackwardHook(
                None,
                (
                    functools.partial(
                        trace_wrapped,
                        fn=call_module_hooks_from_backward_state,
                        bw_state=bw_state,
                        hooks_name=user_hooks_name,
                        module_name=module_name,
                    ),
                ),
                (
                    functools.partial(
                        trace_wrapped,
                        fn=call_module_hooks_from_backward_state,
                        bw_state=bw_state,
                        hooks_name=user_pre_hooks_name,
                        module_name=module_name,
                    ),
                ),
            )

        # 添加模块的反向状态钩子和用户定义的钩子
        module_name, bw_state_proxy = tx.output.add_backward_state_hook(module, "mod")
        user_pre_hooks_name, _ = tx.output.add_backward_state_hook(user_pre_hooks)
        user_hooks_name, _ = tx.output.add_backward_state_hook(user_hooks)
        # 创建一个代理对象，用于调用函数并传递到反向钩子中
        proxy = tx.output.create_proxy(
            "call_function",
            _in_graph_bw_hooks,
            (bw_state_proxy,),
            {},
        )
        # 设置节点的元数据，示例值为一个空的 BackwardHook 对象
        proxy.node.meta["example_value"] = torch.utils.hooks.BackwardHook(None, (), ())
        # 返回 BackwardHookVariable 对象
        return BackwardHookVariable(proxy, module, user_hooks, user_pre_hooks)

    def __init__(
        self,
        proxy: torch.fx.Proxy,
        module: VariableTracker,
        user_hooks: VariableTracker,
        user_pre_hooks: VariableTracker,
        **options,
    ):
        super().__init__(**options)
        self.proxy = proxy
        self.module = module
        self.user_hooks = user_hooks
        self.user_pre_hooks = user_pre_hooks

    def as_proxy(self):
        # 返回存储的代理对象
        return self.proxy

    def call_method(
        self,
        tx,
        name,
        args: List[VariableTracker],
        kwargs: Dict[str, VariableTracker],
    ) -> VariableTracker:
        # 如果调用的方法是设置输入或输出钩子之一，则调用相应的私有方法 _setup_hook
        if name in ("setup_input_hook", "setup_output_hook"):
            return self._setup_hook(tx, name, *args, **kwargs)
        # 否则调用父类的 call_method 方法
        return super().call_method(tx, name, args, kwargs)
    # 定义一个私有方法 `_setup_hook`，用于设置钩子
    def _setup_hook(self, tx, hook_method_name, args):
        # 导入包内的 `wrap_fx_proxy` 函数，用于包装特定的代理对象
        from .builder import wrap_fx_proxy

        # 返回经过包装的特定代理对象
        return wrap_fx_proxy(
            # 调用 `tx.output.create_proxy` 方法创建一个代理对象
            tx,
            tx.output.create_proxy(
                "call_method",  # 使用 "call_method" 类型创建代理对象
                hook_method_name,  # 指定要调用的钩子方法名
                (self.as_proxy(), args.as_proxy()),  # 将当前对象和参数对象转换为代理对象并传入
                {},  # 传递空字典作为附加参数
            ),
        )
```