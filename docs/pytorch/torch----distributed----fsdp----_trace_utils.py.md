# `.\pytorch\torch\distributed\fsdp\_trace_utils.py`

```py
# mypy: allow-untyped-defs
# 导入 functools 模块，用于高阶函数操作
import functools
# 从 contextlib 模块导入 contextmanager 装饰器，用于创建上下文管理器
from contextlib import contextmanager
# 从 dataclasses 模块导入 dataclass 装饰器和 field 函数，用于创建数据类
from dataclasses import dataclass, field
# 从 typing 模块导入需要的类型注解
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple

# 导入 PyTorch 库
import torch
# 从 torch.nn 模块导入 nn 模块，包含神经网络相关的类和函数
import torch.nn as nn


@dataclass
class TracingConfig:
    """
    This represents a symbolic tracing configuration.

    Args:
        tracer (torch.fx.Tracer): An instance of :class:`torch.fx.Tracer` to
            use for symbolic tracing. The default value is the native
            :class:`torch.fx.Tracer` constructed with default arguments.
            However, the user may want to pass a different value such as the
            ``HFTracer`` for models in the HuggingFace Transformers_ library.
            .. _Transformers: https://huggingface.co/docs/transformers/index
        concrete_args (Optional[Dict[str, Any]]): Concrete arguments that
            should not be treated as ``torch.fx.Proxy`` when tracing the
            module ``forward()``. Passing ``concrete_args`` allows partially
            specializing the forward, e.g. to remove control flow or data
            structures. This ``concrete_args`` here is the same argument used
            in :meth:`~torch.fx.Tracer.trace`.
    """

    # tracer 属性用于指定用于符号跟踪的 Tracer 实例，默认为 torch.fx.Tracer
    tracer: torch.fx.Tracer = field(default_factory=torch.fx.Tracer)
    # concrete_args 属性用于传递具体参数，在跟踪模块 forward() 时不作为 torch.fx.Proxy 处理
    concrete_args: Optional[Dict[str, Any]] = None


class _ParamUsageInfo(NamedTuple):
    """
    This is used for ``_ExecutionInfo.module_to_param_usage_infos`` to record
    execution information. The ``dict`` maps modules to a list of these
    ``_ParamUsageInfo`` instances, where each instance represents a group of
    parameters used together.

    Specifically, for each module key in the ``dict``, each instance of this
    class represents either:
    (1) the module and some sublist of its ``named_parameters()`` used
    together in execution (see ``_patched_create_proxy()``), or
    (2) a submodule and all of ``submodule.named_parameters()`` (see
    ``_patched_call_module()``).

    Type (1) corresponds to directly using parameters in ops without calling
    ``forward()``, and type (2) corresponds to calling ``forward()``. The
    mapped-to lists in the ``dict`` follow the execution order.
    """

    # module 属性表示 nn.Module 的实例，记录执行信息
    module: nn.Module
    # named_params 属性是一个列表，包含 (参数名, 参数值) 元组，记录一组一起使用的参数
    named_params: List[Tuple[str, nn.Parameter]]


class _ExecutionInfo:
    """
    This represents the execution order information from the forward pass.
    """

    # 该类用于记录前向传播的执行顺序信息
    pass
    # 初始化方法，用于创建一个新的对象实例
    def __init__(self, root_module: nn.Module) -> None:
        # 当前正在追踪的模块，初始化为传入的根模块
        self.curr_module: nn.Module = root_module
        # 存储模块执行顺序的列表，初始包含根模块
        self.module_forward_order: List[nn.Module] = [root_module]
        # 将每个模块映射到其参数使用信息列表的字典，初始时根模块对应空列表
        self.module_to_param_usage_infos: Dict[nn.Module, List[_ParamUsageInfo]] = {
            root_module: []
        }
        # 存储参数执行顺序的列表，初始为空
        self.param_forward_order: List[nn.Parameter] = []
        # 存储已访问过的参数的集合，用于快速成员检查，在追踪过程中使用
        self.visited_params: Set[nn.Parameter] = set()
# 定义一个内部类 `_ExecOrderTracer`，用于跟踪执行顺序
class _ExecOrderTracer:
    # 初始化方法，设置初始值为 None 的执行信息
    def __init__(self) -> None:
        self.exec_info: Optional[_ExecutionInfo] = None

    # 上下文管理器方法，用于注入跟踪器和根模块，并保存执行信息
    @contextmanager
    def patch_tracer(self, tracer: torch.fx.Tracer, root_module: nn.Module):
        self.exec_info = _ExecutionInfo(root_module)
        orig_call_module = tracer.call_module
        orig_create_proxy = tracer.create_proxy
        # 修改 tracer 的 call_module 方法，使用自定义的 _patched_call_module
        tracer.call_module = functools.partial(
            self._patched_call_module, orig_call_module, self.exec_info
        )
        # 创建参数代理时，修改 tracer 的 create_proxy 方法，使用 _patched_create_proxy
        fqn_to_param = dict(root_module.named_parameters())
        tracer.create_proxy = functools.partial(
            self._patched_create_proxy,
            orig_create_proxy,
            self.exec_info,
            fqn_to_param,
        )
        try:
            yield  # 执行上下文中的代码块
        finally:
            # 恢复原始的 tracer 方法
            tracer.call_module = orig_call_module
            tracer.create_proxy = orig_create_proxy

    # 用于替换 tracer 的 call_module 方法，记录模块的执行顺序和参数使用信息
    def _patched_call_module(
        self,
        call_module: Callable,
        exec_info: _ExecutionInfo,
        module: nn.Module,  # 当前模块
        forward: Callable,  # 模块的 forward 方法
        args: Tuple[Any, ...],  # forward 方法的位置参数
        kwargs: Dict[str, Any],  # forward 方法的关键字参数
    ) -> Any:
        """
        重写 `call_module` 方法，将执行信息保存到 `exec_info` 中。
        在符号跟踪期间，每个非根模块调用此方法。

        Args:
            call_module (Callable): 被重写的原始 `call_module` 方法。
            exec_info (_ExecutionInfo): 用于记录执行信息的对象。
            module (nn.Module): 与 `call_module` 对应的模块。
            forward (Callable): `module` 的 `forward()` 方法，用于当前 `call_module` 调用。
            args (Tuple[Any, ...]): `forward` 方法的位置参数。
            kwargs (Dict[str, Any]): `forward` 方法的关键字参数。

        Returns:
            与 `call_module` 相同的返回值。
        """
        exec_info.module_forward_order.append(module)  # 记录模块的执行顺序
        named_params = list(module.named_parameters())
        curr_module = exec_info.curr_module
        if named_params:
            assert (
                curr_module in exec_info.module_to_param_usage_infos
            ), "当前模块应已被修改的 `call_module` 处理过"
            exec_info.module_to_param_usage_infos[exec_info.curr_module].append(
                _ParamUsageInfo(module, named_params)
            )
        prev_curr_module = curr_module
        exec_info.curr_module = module
        exec_info.module_to_param_usage_infos[module] = []  # 初始化当前模块的参数使用信息
        output = call_module(module, forward, args, kwargs)  # 调用原始的 `call_module` 方法
        exec_info.curr_module = prev_curr_module  # 恢复当前模块
        return output
    # 定义一个方法 `_patched_create_proxy`，用于创建代理对象
    self,  # 方法的第一个参数通常表示对象实例本身
    create_proxy: Callable,  # 参数，指定了一个可调用对象 `create_proxy`，用于创建代理
    exec_info: _ExecutionInfo,  # 参数，包含执行信息的对象 `_ExecutionInfo`
    fqn_to_param: Dict[str, nn.Parameter],  # 参数，字典，将全限定名称映射到神经网络参数对象的字典

    # 以下是 `create_proxy` 方法期望的参数
    kind: str,  # 参数，表示对象类型的字符串
    target: torch.fx.node.Target,  # 参数，表示PyTorch FX节点的目标
    args: Tuple[Any, ...],  # 参数，元组，包含传递给 `create_proxy` 的位置参数
    kwargs: Dict[str, Any],  # 参数，字典，包含传递给 `create_proxy` 的关键字参数

    name: Optional[str] = None,  # 参数，可选参数，字符串，表示对象名称
    type_expr: Optional[Any] = None,  # 参数，可选参数，表示类型表达式的任意类型
    proxy_factory_fn: Optional[Callable[[torch.fx.Node], torch.fx.Proxy]] = None,
    # 参数，可选参数，一个可调用对象，用于创建 PyTorch FX 代理对象
```