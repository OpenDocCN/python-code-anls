# `.\pytorch\torch\_ops.py`

```py
# mypy: allow-untyped-defs
# 引入所需的模块和库
import contextlib  # 上下文管理器模块，用于创建和管理上下文
import ctypes  # 提供与 C 语言兼容的数据类型的库
import importlib  # 提供对模块的动态导入功能
import inspect  # 提供用于检查 Python 对象的类型、模块、类、函数等的工具
import sys  # 提供与 Python 解释器及其环境交互的功能
import types  # 定义 Python 中各种类型的标准类型对象的库
from typing import Any, Callable, Dict, List, Set, Type, Union  # 引入类型提示的必要类和模块

import torch._C  # 引入 torch C++ 扩展模块
import torch.utils._pytree as pytree  # 引入 torch 的 PyTree 模块
from torch import _utils_internal  # 引入 torch 内部工具函数模块
from torch._functorch.pyfunctorch import dispatch_functorch  # 引入 functorch 模块中的函数
from torch.utils._python_dispatch import TorchDispatchMode  # 引入 TorchDispatchMode 类

# Query `hasattr` only once.
# 只查询一次 `hasattr`，避免重复查询和提高效率

# 检查是否支持设置和获取动态链接库的打开标志
_SET_GLOBAL_FLAGS = hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags")


@contextlib.contextmanager
def dl_open_guard():
    """
    Context manager to set the RTLD_GLOBAL dynamic linker flag while we open a
    shared library to load custom operators.
    """
    # 如果当前环境不支持设置和获取动态链接库的标志，则直接退出上下文管理器
    if not _SET_GLOBAL_FLAGS:
        yield
        return
    # 记录当前的动态链接库打开标志
    old_flags = sys.getdlopenflags()
    # 设置动态链接库的打开标志，添加 RTLD_GLOBAL 标志
    sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
    try:
        yield
    finally:
        # 恢复原来的动态链接库打开标志
        sys.setdlopenflags(old_flags)


class OperatorBase:
    """
    Base class for OpOverload (which represents C++ ATen operators) and HigherOrderOperator
    (which represents Python-only operators that are unrepresentable in TorchScript).
    """
    # OperatorBase 类，作为 OpOverload 和 HigherOrderOperator 的基类
    pass  # 空的占位符，表示类定义结束
    # 构造函数初始化方法，用于创建一个分发缓存，将分发键映射到其对应的实现或Python可调用对象。
    # 分发键可以指向C++内核或Python可调用对象。这个缓存主要用于加速分发过程。
    def __init__(self):
        # 分发缓存用于存储分发键到实现或Python可调用对象的映射
        self._dispatch_cache: Dict[
            torch._C.DispatchKey, Union[torch._C.DispatchKey, Callable[..., Any]]
        ] = {}

        # py_kernels字典允许您覆盖特定分发键的行为，以调用自定义的Python函数，而不是常规的C++配置行为。
        # 这是Python分发器的主要用途之一，允许您从Python编程中修改分发器的行为。
        self.py_kernels: Dict[torch._C.DispatchKey, Callable[..., Any]] = {}

        # python_key_mode_table字典允许您为特定的TorchDispatchMode注册特定操作符的行为覆盖。
        # TorchDispatchMode可以被视为分发键的开放世界扩展，允许您注册它们的行为，类似于注册分发键。
        self.python_key_mode_table: Dict[
            Type[TorchDispatchMode], Callable[..., Any]
        ] = {}

        # functorch_table字典允许您覆盖functorch转换的行为，目前主要用于HigherOrderOperator。
        self.functorch_table = {}

    # __call__方法是一个占位符，用于在子类中实现具体的行为。
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    # 检查是否有针对特定分发键的Python内核可用
    def has_kernel_for_dispatch_key(self, k):
        return k in self.py_kernels

    # 检查是否有针对任何分发键的Python内核可用
    def has_kernel_for_any_dispatch_key(self, ks):
        for k in self.py_kernels:
            # 检查分发键是否不是别名且在给定的分发键集合中存在
            if not torch._C._dispatch_is_alias_key(k) and ks.has(k):
                return True
        return False
    # 定义一个装饰器函数 py_impl，接受一个参数 k
    def py_impl(self, k):
        # 定义内部函数 inner，接受一个参数 fn
        def inner(fn):
            # 如果 k 是类并且是 TorchDispatchMode 的子类
            if inspect.isclass(k) and issubclass(k, TorchDispatchMode):
                # 断言 k 不在 self.python_key_mode_table 中
                assert k not in self.python_key_mode_table
                # 将 fn 添加到 self.python_key_mode_table 中对应的 k 键下
                self.python_key_mode_table[k] = fn
                # 清空 _dispatch_cache 缓存
                self._dispatch_cache.clear()
                # 返回 fn
                return fn

            # 如果 k 是 torch._C._functorch.TransformType 的实例
            if isinstance(k, torch._C._functorch.TransformType):
                # 断言 k 不在 self.functorch_table 中
                assert k not in self.functorch_table
                # 将 fn 添加到 self.functorch_table 中对应的 k 键下
                self.functorch_table[k] = fn
                # 返回 fn
                return fn

            # 断言 k 是 torch._C.DispatchKey 的实例
            assert isinstance(k, torch._C.DispatchKey)
            # 断言 k 不等于 torch._C.DispatchKey.Python
            assert (
                k != torch._C.DispatchKey.Python
            ), "Please register a mode for the torch._C.DispatchKey.Python key instead."

            # 如果 k 已经存在于 self.py_kernels 中
            if k in self.py_kernels:
                # 抛出 RuntimeError，提示尝试覆盖在操作符 self.name() 上的 k 的 Python 实现
                raise RuntimeError(
                    f"Trying to override a python impl for {k} on operator {self.name()}"
                )
            # 将 fn 添加到 self.py_kernels 中对应的 k 键下
            self.py_kernels[k] = fn
            # 清空 _dispatch_cache 缓存
            self._dispatch_cache.clear()
            # 返回 fn
            return fn

        # 返回内部函数 inner
        return inner

    # 注册一个实现到三种变体的功能化中：
    # - DispatchKey.Functionalize
    # - functorch.TransformType.Functionalize
    # - FunctionalTensorMode
    # 示例：
    #   @py_functionalize_impl
    #   def functionalize_rule(ctx, inner_f, *args):
    #       args_unwrapped = ctx.unwrap_tensors(args)
    #       with ctx.redispatch_to_next():
    #           out = ctx.functionalize(inner_f)(*args_unwrapped)
    #           return ctx.wrap_tensors(out)
    def py_functionalize_impl(self, fn):
        # 导入 torch._subclasses.functional_tensor 下的三个功能化 API 类
        from torch._subclasses.functional_tensor import (
            CppFunctionalizeAPI as _CppFunctionalizeAPI,
            FunctorchFunctionalizeAPI as _FunctorchFunctionalizeAPI,
            PythonFunctionalizeAPI as _PythonFunctionalizeAPI,
        )

        # 定义功能化 DispatchKey 函数
        def functionalize_dk_fn(*args, **kwargs):
            # 调用 fn，传入 _CppFunctionalizeAPI() 实例和其他参数
            return fn(_CppFunctionalizeAPI(), *args, **kwargs)

        # 定义功能化 dispatch mode 函数
        def functionalize_dispatch_mode_fn(mode, *args, **kwargs):
            # 调用 fn，传入 _PythonFunctionalizeAPI(mode) 实例和其他参数
            return fn(_PythonFunctionalizeAPI(mode), *args, **kwargs)

        # 定义功能化 functorch 函数
        def functionalize_functorch_fn(interpreter, *args, **kwargs):
            # 调用 fn，传入 _FunctorchFunctionalizeAPI(interpreter) 实例和其他参数
            return fn(_FunctorchFunctionalizeAPI(interpreter), *args, **kwargs)

        # 将 functionalize_dk_fn 注册到 torch._C.DispatchKey.Functionalize 下
        self.py_impl(torch._C.DispatchKey.Functionalize)(functionalize_dk_fn)
        # 将 functionalize_dispatch_mode_fn 注册到 torch._subclasses.functional_tensor.FunctionalTensorMode 下
        self.py_impl(torch._subclasses.functional_tensor.FunctionalTensorMode)(
            functionalize_dispatch_mode_fn
        )
        # 将 functionalize_functorch_fn 注册到 torch._C._functorch.TransformType.Functionalize 下
        self.py_impl(torch._C._functorch.TransformType.Functionalize)(
            functionalize_functorch_fn
        )

        # 返回 fn
        return fn

    # 抛出 NotImplementedError 异常，提示子类需要实现 name 方法
    def name(self):
        raise NotImplementedError
# 引用torch._C._dispatch_is_included_in_alias函数
is_included_in_alias = torch._C._dispatch_is_included_in_alias

# 引用torch._C.DispatchKey作为DispatchKey的别名
DispatchKey = torch._C.DispatchKey

# 定义函数resolve_key，用于解析操作符和分派键
def resolve_key(op: OperatorBase, k: DispatchKey):  # type: ignore[valid-type]
    # 1. (直接) 操作符注册
    if op.has_kernel_for_dispatch_key(k):
        return k
    # 2.1 如果可用，使用CompositeExplicitAutogradNonFunctional内核
    cand = DispatchKey.CompositeExplicitAutogradNonFunctional
    if (
        k == DispatchKey.Undefined or is_included_in_alias(k, cand)
    ) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # 2.2 如果可用，使用CompositeExplicitAutograd内核
    cand = DispatchKey.CompositeExplicitAutograd
    if (
        k == DispatchKey.Undefined or is_included_in_alias(k, cand)
    ) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # 检查是否有后端内核存在，或者是否有CompositeExplicitAutograd内核可用
    has_backend_kernel = op.has_kernel_for_any_dispatch_key(
        torch._C._dispatch_get_backend_keyset_from_autograd(k)
    ) or op.has_kernel_for_dispatch_key(DispatchKey.CompositeExplicitAutograd)
    # 2.3 如果可用，使用CompositeImplicitAutogradNestedTensor内核
    cand = DispatchKey.CompositeImplicitAutogradNestedTensor
    if (
        (k != DispatchKey.Undefined and is_included_in_alias(k, cand))
        and op.has_kernel_for_dispatch_key(cand)
        and not has_backend_kernel
    ):
        return cand
    # 2.4 如果可用，使用CompositeImplicitAutograd内核
    cand = DispatchKey.CompositeImplicitAutograd
    if (
        k == DispatchKey.Undefined or is_included_in_alias(k, cand)
    ) and op.has_kernel_for_dispatch_key(cand):
        if k == DispatchKey.AutogradOther and op.has_kernel_for_any_dispatch_key(
            torch._C._dispatch_autogradother_backends
        ):
            raise RuntimeError("ambiguous autogradother kernel")
        elif not has_backend_kernel:
            return cand
    # 2.5 如果可用，使用DispatchKey::Autograd内核
    cand = DispatchKey.Autograd
    if is_included_in_alias(k, cand) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # 2.6 如果可用，使用DispatchKey::FuncTorchBatchedDecomposition内核
    cand = DispatchKey.FuncTorchBatchedDecomposition
    if is_included_in_alias(k, cand) and op.has_kernel_for_dispatch_key(cand):
        return cand
    # 后端回退
    if torch._C._dispatch_has_backend_fallback(k):
        # 分派键本身将隐式路由到后端回退
        # 这对于纯Python实现可能不是最佳选择
        return k
    # 抛出未实现的错误，指明找不到特定操作符和分派键的内核
    raise NotImplementedError(f"could not find kernel for {op} at dispatch key {k}")


# 字典，用于存储高阶操作符的字符串名称到HigherOrderOperator对象的映射
_higher_order_ops: Dict[str, "HigherOrderOperator"] = {}

# 默认的高阶操作符分派键列表
_HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS = [
    DispatchKey.PythonDispatcher,  # Python分派器的分派键
    DispatchKey.PythonTLSSnapshot,  # Python线程本地存储快照的分派键
    DispatchKey.ADInplaceOrView,  # 自动微分原位或视图的分派键
    DispatchKey.BackendSelect,  # 后端选择的分派键
    DispatchKey.AutocastCPU,  # CPU自动转换的分派键
]
    DispatchKey.AutocastCUDA,  # type: ignore[attr-defined]



    # 使用 DispatchKey.AutocastCUDA，但忽略其属性定义类型检查
class HigherOrderOperator(OperatorBase):
    # HigherOrderOperator 类，继承自 OperatorBase 类

    # The HigherOrderOperator will appear as torch.ops.higher_order.{name}
    #
    # 如果你正在创建一个新的 HigherOrderOperator，请不要更改默认设置。
    # 将运算符添加到全局 torch.ops 命名空间是一种不好的做法，因为可能会出现命名冲突。

    def __init__(self, name):
        # 构造函数，接收一个名称参数 name
        super().__init__()
        self._name = name

        # 将 self 的 __name__ 属性设置为 name
        self.__name__ = name
        # 将当前实例添加到 _higher_order_ops 字典中，键为 name
        _higher_order_ops[name] = self
        # 将 self 的 _ns 属性设置为 "higher_order"
        self._ns = "higher_order"

        # 对于普通的 HigherOrderOperator 实例，将其 __module__ 从 torch._ops 修改为 torch._ops.higher_order
        # 对于 HigherOrderOperator 的子类实例（例如自定义的 higher order 运算符），保持 __module__ 属性不变
        if self.__class__ is HigherOrderOperator:
            self_name_space = "." + self.namespace if self.namespace else ""
            self.__module__ = self.__module__ + self_name_space

        # 初始化 non_fallthrough_keys，包含完整的 dispatch key 集合
        self.non_fallthrough_keys = torch._C._dispatch_keyset_full()

        # 遍历 _HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS 中的每个 dispatch key
        # 并调用 fallthrough 方法处理
        for dispatch_key in _HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS:
            self.fallthrough(dispatch_key)

        # [NOTE] We have to register pre-dispatch key implementation
        # because sometimes HOP use aot-dispatch tracing to detect certaion
        # mutations. This is problematic when we are functionalizing HOP
        # during pre-dispatch because when the inner tracer starts, it will see
        # that PreDispatch key is still active. In that case, we just redispatch
        # it to next key. This is only safe to do when PreDispatch key stack has no
        # active modes.

    def py_impl(self, k):
        # 定义 py_impl 方法，用于处理特定的 dispatch key
        if isinstance(k, torch._C.DispatchKey) and not self.non_fallthrough_keys.has(k):
            self.non_fallthrough_keys = self.non_fallthrough_keys.add(k)
        # 调用父类的 py_impl 方法处理 k
        return super().py_impl(k)

    @property
    def namespace(self):
        # 返回当前实例的命名空间属性 _ns
        return self._ns

    def fallthrough(self, dispatch_key):
        # fallthrough 方法，处理传入的 dispatch_key，从 non_fallthrough_keys 中移除它
        self.non_fallthrough_keys = self.non_fallthrough_keys.remove(dispatch_key)

    def __call__(self, *args, **kwargs):
        # __call__ 方法，使实例对象可以像函数一样被调用

        # Dynamo 已经预先跟踪了 HigherOrderOp 的主体，因此不需要再次跟踪
        import torch._dynamo
        from torch._dynamo import disable

        @disable
        def wrapper():
            # 将 args 和 kwargs 转换为扁平化元组 flat_args
            flat_args = _to_flat_tuple(args, kwargs)
            # 如果存在 Torch 函数重载，则调用 handle_torch_function 处理
            if torch.overrides.has_torch_function(flat_args):
                return torch.overrides.handle_torch_function(
                    self, flat_args, *args, **kwargs
                )

            # 计算 dispatch key 集合 dispatch_key_set
            dispatch_key_set = _compute_keyset(args, kwargs, self.non_fallthrough_keys)
            # 调用 dispatch 方法并返回结果
            return self.dispatch(
                dispatch_key_set.highestPriorityTypeId(), *args, **kwargs
            )

        # 返回 wrapper 函数对象
        return wrapper()

    def __str__(self):
        # 返回当前实例的名称字符串表示形式
        return f"{self.name()}"
    # 定义一个方法 `name`，用于返回对象的 `_name` 属性的值
    def name(self):
        # 返回对象的 `_name` 属性的值
        return self._name
# 将参数和关键字参数展开为一个扁平的元组
def _to_flat_tuple(args, kwargs):
    return pytree.arg_tree_leaves(*args, **kwargs)


# 根据传入的参数和关键字参数获取张量，并调用 key_extractor 函数获取键集合
def _compute_keyset(args, kwargs, non_fallthrough_keys):
    tensors = _get_tensors(args, kwargs)
    return key_extractor(tensors, non_fallthrough_keys)


# 根据传入的参数和关键字参数获取所有张量
def _get_tensors(args, kwargs):
    flat_all = _to_flat_tuple(args, kwargs)
    # 从扁平化后的参数中筛选出张量
    tensor_args = [t for t in flat_all if isinstance(t, torch.Tensor)]
    return tuple(tensor_args)


# 从给定的张量集合和键掩码中提取关键键集合
# 注意：该实现需与 C++ 的分派键提取逻辑保持一致，位于 ATen/core/dispatch/DispatchKeyExtractor.h
def key_extractor(tensors, key_mask):
    # 初始化键集合
    key_set = torch._C._dispatch_tls_local_include_set()
    # 遍历张量集合，依次添加其分派键到键集合中
    for tensor in tensors:
        key_set = key_set | torch._C._dispatch_keys(tensor)
    # 从键集合中排除本地线程分派排除集
    key_set = key_set - torch._C._dispatch_tls_local_exclude_set()
    # 取键集合与键掩码的交集作为最终的键集合
    key_set = key_set & key_mask
    return key_set


# 用于 PreDispatch 的模式堆栈
# 总是包含三个键，优先考虑 FunctionalTensorMode 和 ProxyTorchDispatchMode
# 即第 0 个位置对应 ProxyTorchDispatchMode，第 1 个位置对应 FunctionalTensorMode
# SchemaCheckMode 是独立于其他两者的，仅在堆栈为空时有效
# SchemaCheckMode 用于测试目的，在具体输入上以急切模式运行，检查与别名或变异操作相关的不正确模式
class _ModeStackStateForPreDispatch:
    def __init__(self):
        # 基础模式列表，初始为两个 None
        self.__infra_modes = [None, None]
        # 模式检查模式，初始为 None
        self._schema_check_mode = None

    # 设置指定位置的模式
    def set(self, index, mode):
        assert index < len(self.__infra_modes)
        self.__infra_modes[index] = mode

    # 获取指定位置的模式
    def get(self, index):
        assert index < len(self.__infra_modes)
        return self.__infra_modes[index]

    # 返回当前设置的模式数量，包括非 None 的基础模式和模式检查模式
    def count(self):
        return len([i for i in self.__infra_modes if i is not None]) + int(
            self._schema_check_mode is not None
        )


# 单例模式，用于 PreDispatch 的模式堆栈状态
_mode_stack_state_for_pre_dispatch = _ModeStackStateForPreDispatch()


# 取消 PreDispatch 中的模式设置
# 参数 mode_key 为要取消的模式键，schema_check 用于指示是否进行模式检查
def unset_mode_pre_dispatch(mode_key, schema_check=False):
    # 获取当前的 PreDispatch 模式堆栈状态
    current_mode_stack_pre_dispatch = mode_stack_state_for_pre_dispatch()
    # 断言模式键为空或者为 Proxy 或 Functional 之一
    assert mode_key is None or mode_key in (
        torch._C._TorchDispatchModeKey.PROXY,
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
    )
    # 如果进行模式检查，则模式键必须为空
    if schema_check:
        assert mode_key is None

    # 内部函数，用于取消指定的模式
    def _unset_mode():
        if mode_key == torch._C._TorchDispatchModeKey.PROXY:
            # 取消 ProxyTorchDispatchMode
            current_mode = current_mode_stack_pre_dispatch.get(0)
            mode_stack_state_for_pre_dispatch().set(0, None)
            return current_mode
        elif mode_key == torch._C._TorchDispatchModeKey.FUNCTIONAL:
            # 取消 FunctionalTensorMode
            current_mode = current_mode_stack_pre_dispatch.get(1)
            mode_stack_state_for_pre_dispatch().set(1, None)
            return current_mode
        else:
            # 取消模式检查模式
            current_mode = mode_stack_state_for_pre_dispatch()._schema_check_mode
            mode_stack_state_for_pre_dispatch()._schema_check_mode = None
            return current_mode
    # 调用 _unset_mode 函数获取当前模式并赋值给 current_mode
    current_mode = _unset_mode()

    # 调用 _len_torch_dispatch_stack_pre_dispatch 函数获取预调度堆栈的长度，并赋值给 new_pre_dispatch_len
    new_pre_dispatch_len = _len_torch_dispatch_stack_pre_dispatch()

    # 当我们正在取消设置一个模式时，需要检查预调度键（PreDispatch key）上是否有活动的模式。
    # 如果没有活动的模式，我们需要从本地调度包含集合中移除 PreDispatch 键。
    if new_pre_dispatch_len == 0:
        # 调用 torch._C._dispatch_tls_set_dispatch_key_included 函数，
        # 将 PreDispatch 键从本地调度包含集合中移除
        torch._C._dispatch_tls_set_dispatch_key_included(
            torch._C.DispatchKey.PreDispatch, False
        )

    # 返回当前模式
    return current_mode
# 设置预调度模式
def _set_mode_pre_dispatch(mode):
    # 导入必要的模块和类
    from torch._subclasses.functional_tensor import FunctionalTensorMode
    from torch._subclasses.schema_check_mode import SchemaCheckMode
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

    # 断言 mode 是 FunctionalTensorMode、ProxyTorchDispatchMode 或 SchemaCheckMode 中的一种
    assert isinstance(
        mode,
        (
            FunctionalTensorMode,
            ProxyTorchDispatchMode,
            SchemaCheckMode,
        ),
    )

    # 获取之前的调度堆栈长度
    previous_mode_stack_len = _len_torch_dispatch_stack_pre_dispatch()

    # 根据 mode 的类型进行处理
    if isinstance(mode, SchemaCheckMode):
        # 如果 mode 是 SchemaCheckMode 类型
        current_mode = mode_stack_state_for_pre_dispatch()._schema_check_mode
        # 如果之前的调度堆栈长度大于 0，抛出异常
        if previous_mode_stack_len > 0:
            raise AssertionError(
                "SchemaCheckMode for pre-dispatch must be used exclusively, found other modes on the stack"
            )
        # 设置当前的 SchemaCheckMode
        mode_stack_state_for_pre_dispatch()._schema_check_mode = mode
    elif isinstance(mode, FunctionalTensorMode):
        # 如果 mode 是 FunctionalTensorMode 类型
        current_mode = mode_stack_state_for_pre_dispatch().get(1)
        # 断言当前模式为 None
        assert current_mode is None
        # 设置第一个位置的 FunctionalTensorMode
        mode_stack_state_for_pre_dispatch().set(1, mode)
    else:
        # 其他情况，包括 ProxyTorchDispatchMode
        current_mode = mode_stack_state_for_pre_dispatch().get(0)
        # 断言当前模式为 None
        assert current_mode is None
        # 设置第零个位置的 ProxyTorchDispatchMode
        mode_stack_state_for_pre_dispatch().set(0, mode)

    # 当设置模式时，需要检查 PreDispatch 键上是否还有活跃的模式。
    # 如果在设置此模式之前没有活动模式，则意味着 PreDispatch 键已经关闭，因此需要重新打开它。
    if previous_mode_stack_len == 0:
        torch._C._dispatch_tls_set_dispatch_key_included(
            torch._C.DispatchKey.PreDispatch, True
        )


# 从预调度堆栈中弹出模式
def _pop_mode_from_pre_dispatch():
    # 获取当前的模式堆栈状态
    mode_stack = mode_stack_state_for_pre_dispatch()
    # 获取预调度堆栈的长度
    pre_dispatch_len = _len_torch_dispatch_stack_pre_dispatch()

    # 如果预调度堆栈长度为 0，抛出异常
    if pre_dispatch_len == 0:
        raise AssertionError("Trying to pop empty mode stack")

    # 根据不同的模式类型进行处理
    if mode_stack._schema_check_mode is not None:
        return unset_mode_pre_dispatch(None, schema_check=True)
    if mode_stack.get(1) is not None:
        return unset_mode_pre_dispatch(torch._C._TorchDispatchModeKey.FUNCTIONAL)
    if mode_stack.get(0) is not None:
        return unset_mode_pre_dispatch(torch._C._TorchDispatchModeKey.PROXY)


# 获取预调度堆栈的长度
def _len_torch_dispatch_stack_pre_dispatch():
    return mode_stack_state_for_pre_dispatch().count()


# 获取指定调度模式的预调度模式
def _get_dispatch_mode_pre_dispatch(mode_key):
    # 断言 mode_key 是 PROXY 或 FUNCTIONAL 中的一个
    assert mode_key in (
        torch._C._TorchDispatchModeKey.PROXY,
        torch._C._TorchDispatchModeKey.FUNCTIONAL,
    )
    # 根据 mode_key 返回相应位置的模式
    if mode_key == torch._C._TorchDispatchModeKey.PROXY:
        return mode_stack_state_for_pre_dispatch().get(0)
    else:
        return mode_stack_state_for_pre_dispatch().get(1)


# 获取当前的预调度模式
def _get_current_dispatch_mode_pre_dispatch():
    # 如果当前的模式堆栈状态中有 _schema_check_mode，则返回它
    if mode_stack_state_for_pre_dispatch()._schema_check_mode is not None:
        return mode_stack_state_for_pre_dispatch()._schema_check_mode
    else:
        # 调用 mode_stack_state_for_pre_dispatch() 函数获取堆栈状态，并计算其元素个数
        stack_len = mode_stack_state_for_pre_dispatch().count()
        # 如果堆栈长度为2，返回索引为1的元素
        if stack_len == 2:
            return mode_stack_state_for_pre_dispatch().get(1)
        # 如果堆栈长度为1，返回索引为1的元素，如果该元素为None，则返回索引为0的元素
        if stack_len == 1:
            return (
                mode_stack_state_for_pre_dispatch().get(1)
                if mode_stack_state_for_pre_dispatch().get(1) is not None
                else mode_stack_state_for_pre_dispatch().get(0)
            )
    # 如果以上条件均不满足，返回None
    return None
# 返回全局变量 _mode_stack_state_for_pre_dispatch 的当前状态
def mode_stack_state_for_pre_dispatch():
    global _mode_stack_state_for_pre_dispatch
    return _mode_stack_state_for_pre_dispatch

# 初始化一个空的集合 cached_ops，用于存储 OpOverload 对象
cached_ops: Set["OpOverload"] = set()

# 向 cached_ops 集合中添加一个 OpOverload 对象
def add_cached_op(op_overload):
    global cached_ops
    cached_ops.add(op_overload)

# 清空 cached_ops 集合，移除所有缓存的 OpOverload 对象
def reset_cached_ops():
    global cached_ops
    cached_ops.clear()

# 返回 cached_ops 集合，包含当前所有缓存的 OpOverload 对象
def get_cached_ops():
    global cached_ops
    return cached_ops

# OpOverload 类继承自 OperatorBase 类，用于表示特定的运算符重载
class OpOverload(OperatorBase):
    def __init__(self, overloadpacket, op, op_dk, schema, tags):
        super().__init__()
        self._op = op
        self._op_dk = op_dk
        self._schema = schema
        self._overloadpacket = overloadpacket
        self._tags = tags
        # 设置默认的重载名称为 "default"，如果 schema 中没有指定重载名称，则使用空字符串
        self._overloadname = "default" if schema.overload_name == "" else schema.overload_name
        # 根据 schema 中的名称设置 OpOverload 的名称
        self._name = self._schema.name
        if schema.overload_name:
            self._name += "." + schema.overload_name
        # 根据 schema 和 overloadname 设置特定的名称和模块信息
        self.__name__ = f"{self._schema.name.split('::')[1]}.{self._overloadname}"
        self.__module__ = overloadpacket.__module__
        op.__module__ = overloadpacket.__module__
        self.__qualname__ = self._name
        self.__annotations__ = {}
        
        # 仅在需要时计算 OperatorHandle。不是所有 OpOverloads 都有 OperatorHandle（比如 TorchScript 中的一些）
        self._lazy_handle = None
        
        # 如果 OpOverload 是从 Python 中的 Library.def 创建的
        self._defined_in_python = self.__qualname__ in torch.library._defs
        
        # 根据 schema 中的参数信息，判断 OpOverload 是否是视图（view）
        is_write = None
        for a in self._schema.arguments:
            if a.alias_info is None:
                continue
            if is_write is None:
                is_write = a.alias_info.is_write
            else:
                # 如果存在可写的别名信息，则不是视图
                is_write = a.alias_info.is_write or is_write
        self.is_view = is_write is not None and not is_write
    
    # 返回 schema 的命名空间部分（第一部分）
    @property
    def _namespace(self):
        return self._schema.name.split("::")[0]
    
    # 返回 schema 的操作名部分（第二部分）
    @property
    def _opname(self):
        return self._schema.name.split("::")[1]
    
    # 返回 OpOverload 对象的操作句柄（handle），如果未初始化则进行初始化
    @property
    def _handle(self):
        if self._lazy_handle is None:
            self._lazy_handle = torch._C._dispatch_find_schema_or_throw(
                self._schema.name, self._schema.overload_name
            )
        return self._lazy_handle
    
    # 对象深拷贝方法，因为 OpOverload 对象是不可变的，所以返回自身
    def __deepcopy__(self, memo=None):
        return self
    # 返回一个格式化的字符串表示对象，描述操作重载的名称和重载的方法名
    def __repr__(self):
        return "<OpOverload(op='{}.{}', overload='{}')>".format(
            *self._schema.name.split("::"), self._overloadname
        )

    # 实现调用操作重载对象的方法，使用 `self_` 避免与可能命名为 "self" 的参数冲突，
    # 以便通过关键字参数调用所有的aten操作。
    def __call__(self_, *args, **kwargs):  # noqa: B902
        return self_._op(*args, **kwargs)

    # 重新分发操作至关联的对象，使用 `self_` 避免与可能命名为 "self" 的参数冲突，
    # 以便通过关键字参数重新分发所有的aten操作。
    def redispatch(self_, keyset, *args, **kwargs):  # noqa: B902
        return self_._handle.redispatch_boxed(keyset, *args, **kwargs)

    # 实现对象的哈希计算，基于操作对象的哈希值
    def __hash__(self):
        return hash(self._op)

    # 返回对象的字符串表示形式，格式为 `my_namespace.my_op_name.overload_name`
    def __str__(self):
        return "{}.{}.{}".format(*self._schema.name.split("::"), self._overloadname)

    # 检查是否存在特定分发键的内核，继承自父类实现或使用torch的C扩展函数检查
    def has_kernel_for_dispatch_key(self, k):
        return super().has_kernel_for_dispatch_key(
            k
        ) or torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), k)

    # 检查是否存在任何给定分发键的内核，使用torch的C扩展函数检查
    def has_kernel_for_any_dispatch_key(self, ks):
        return torch._C._dispatch_has_kernel_for_any_dispatch_key(
            self.name(), ks
        ) or super().has_kernel_for_any_dispatch_key(ks)

    # 返回对象的命名空间，即通过分隔符"::"分割后的第一部分
    @property
    def namespace(self):
        return self._schema.name.split("::")[0]

    # 对象的分解函数，根据不同的分发键执行不同的操作
    def decompose(self, *args, **kwargs):
        dk = torch._C.DispatchKey.CompositeImplicitAutograd
        if dk in self.py_kernels:
            # 注意：这个分支现在不太必要，因为我们可以在追踪之前应用Python的CompositeImplicitAutograd，
            # 利用Python分发器（同时利用自动求导公式）。但为了完整性而包含此分支。
            return self.py_kernels[dk](*args, **kwargs)
        elif torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), dk):
            return self._op_dk(dk, *args, **kwargs)
        else:
            return NotImplemented

    # 从分发缓存中移除指定的分发键，强制下次重新计算
    # 警告：如果向OpOverload的py_kernels注册了分发键，仅调用_del_dispatch并不足以应用更改，
    # 因为单个注册可能影响多个分发键（例如，注册Autograd会影响AutogradCPU）。del_dispatch仅在
    # 特别修改get_dispatch如何处理特定输入'key'时使用。
    def _uncache_dispatch(self, key):
        self._dispatch_cache.pop(key, None)

    # 实现Python分发器的预计算逻辑，返回对象的名称属性
    def name(self):
        return self._name

    # 返回对象的重载数据包属性
    @property
    def overloadpacket(self):
        return self._overloadpacket

    # 返回对象的操作属性
    @property
    def op(self):
        return self._op

    # 返回对象的标签属性
    @property
    def tags(self):
        return self._tags
    # 添加更多方法来公开有关输入和输出参数的信息的待办事项
# TorchBindOpOverload are those custom ops which have at least one overload's
# schema consists of torch.ScriptObject (i.e. custom class) input.
# TorchBindOpOverload will skip C++ dispatcher and purely dispatched in python
# when its inputs contain FakeScriptObject in a similar way as higher order ops.
class TorchBindOpOverload(OpOverload):
    # 定义方法 `_fallthrough_keys`，返回一个包含默认的 fallthrough keys 列表
    # 这些 key 对应于应该跳过 C++ 调度程序并在 Python 中纯粹调度的情况
    def _fallthrough_keys(self) -> List[DispatchKey]:
        # TODO: we should be calling the fallback for these, but a fallthrough is almost close
        # enough to the fallback in most cases that we care about.
        _DEFAULT_FALLTHROUGH_KEYS = [
            DispatchKey.Autograd,
            DispatchKey.AutogradCPU,
            DispatchKey.AutogradCUDA,
            DispatchKey.ADInplaceOrView,
            DispatchKey.BackendSelect,
            DispatchKey.PythonTLSSnapshot,
            DispatchKey.PythonDispatcher,
        ]

        # 定义内部函数 `_may_use_fallthrough_instead_of_fallback`，
        # 判断是否应该使用 fallthrough 而不是 fallback
        def _may_use_fallthrough_instead_of_fallback(key: DispatchKey):
            # 如果存在给定 key 的调度内核，则检查其是否为 fallthrough
            if torch._C._dispatch_has_kernel_for_dispatch_key(self.name(), key):
                return torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                    self.name(), key
                )

            # 如果不存在该 key 对应的 Python 内核，或者其为 fallthrough_kernel，则返回 True
            return (
                key not in self.py_kernels
                or self.py_kernels[key] is torch.library.fallthrough_kernel
            )

        # 返回筛选后的 fallthrough keys 列表
        return [
            key
            for key in _DEFAULT_FALLTHROUGH_KEYS
            if _may_use_fallthrough_instead_of_fallback(key)
        ]

    # 使用上下文管理器 `contextlib.contextmanager`，临时注册为 effectful 操作
    def _register_as_effectful_op_temporarily(self):
        # 导入所需模块和类
        from torch._higher_order_ops.effects import (
            _EffectType,
            _register_effectful_op,
            SIDE_EFFECTS,
        )

        try:
            # 如果当前实例不在 SIDE_EFFECTS 中，则注册为 effectful 操作，类型为 ORDERED
            if self not in SIDE_EFFECTS:
                _register_effectful_op(self, _EffectType.ORDERED)
            # 使用 yield 返回控制权，允许在此期间进行操作
            yield
        finally:
            # 最终处理，如果当前实例在 SIDE_EFFECTS 中，则从中删除
            if self in SIDE_EFFECTS:
                del SIDE_EFFECTS[self]

    # 使用 `self_` 避免与命名为 "self" 的参数发生冲突
    # 这样它们可以通过关键字参数进行调用。
    def __call__(self_, *args, **kwargs):  # noqa: B902
        # 如果必须在 Python 中调度，则调用 _must_dispatch_in_python 函数检查
        if _must_dispatch_in_python(args, kwargs):
            # 当任何输入是 FakeScriptObject 时，我们需要跳过 C++ 调度器，并通过 python_dispatcher 的 _get_dispatch 在 Python 中进行调度
            # 因为 C++ 调度器会检查模式，无法识别 FakeScriptObject。

            # 注意:
            # 1. 我们只暂时将 torchbind 操作注册为 effectful 操作，因为我们只希望 effect token 功能逻辑在追踪期间应用。
            #    否则，在追踪后，操作的急切执行行为可能会发生变化。
            # 2. 我们不希望在构造函数中为所有 torchbind 操作注册 effectful 操作，因为这可能会导致某些 autograd.profiler 操作的意外行为，
            #    例如 profiler._record_function_exit._RecordFunction。
            with self_._register_as_effectful_op_temporarily():
                return self_._dispatch_in_python(
                    args, kwargs, self_._fallthrough_keys()
                )
        
        # 如果不需要在 Python 中调度，则调用 self_._op 处理函数进行处理
        return self_._op(*args, **kwargs)

    def _dispatch_in_python(self, args, kwargs, fallthrough_keys):
        # 获取非 fallthrough keys 的完整 dispatch key 集合
        non_fallthrough_keys = torch._C._dispatch_keyset_full()
        for key in fallthrough_keys:
            non_fallthrough_keys = non_fallthrough_keys.remove(key)

        # 计算当前调度所需的 dispatch key
        dispatch_key_set = _compute_keyset(args, kwargs, non_fallthrough_keys)
        dispatch_key = dispatch_key_set.highestPriorityTypeId()

        # 获取与 dispatch key 相关的 handler 处理函数
        handler = (
            self._get_dispatch(dispatch_key)
            if dispatch_key not in self._dispatch_cache
            else self._dispatch_cache[dispatch_key]
        )

        # 如果 handler 是 DispatchKey 对象，则说明需要重新调度
        if isinstance(handler, DispatchKey):
            # fallthrough keys 可能在运行时通过 torch.library.impl 注册，因此需要将其添加到 fallthrough_keys 并重新调度
            if torch._C._dispatch_kernel_for_dispatch_key_is_fallthrough(
                self.name(), dispatch_key
            ):
                return self._dispatch_in_python(
                    args, kwargs, fallthrough_keys + [dispatch_key]
                )

            # 抛出运行时错误，说明 Torchbind 操作接收到了 FakeScriptObject 输入，但未找到对应的 Python 实现
            raise RuntimeError(
                f"Torchbind op {self} received a FakeScriptObject input when dispatching {handler}."
                f" but no python implementation is found."
                f" Please file an issue on this when you encounter this error."
                f" This error can happen when you export or compile the model."
                f" It can still happpen even if a C++ implementation for {dispatch_key}. "
                f" has been registered. That's because FakeScriptObject purely lives in python and cannot work "
                f" with a C++ implementation."
            )

        # 断言 handler 是可调用对象
        assert isinstance(handler, Callable)  # type: ignore[arg-type]
        # 调用 handler 处理函数并返回结果
        return handler(*args, **kwargs)
# 判断参数 args 和 kwargs 中是否存在任何一个对象是 torch._library.fake_class_registry.FakeScriptObject 类的实例
def _must_dispatch_in_python(args, kwargs):
    return pytree.tree_any(
        lambda obj: isinstance(
            obj, torch._library.fake_class_registry.FakeScriptObject
        ),
        (args, kwargs),
    )


# 检查给定的 torch.FunctionSchema 对象中是否存在任何一个参数类型为 torch.ClassType 的参数
def _has_script_object_arg(schema: torch.FunctionSchema) -> bool:
    return any(isinstance(arg.type, torch.ClassType) for arg in schema.arguments)


# OpOverloadPacket 类包含指向未解析运算符的指针，该运算符不对应特定的运算符
# 可通过属性查询获取 OpOverload 对象。
class OpOverloadPacket:
    def __init__(self, qualified_op_name, op_name, op, overload_names):
        # 这些属性通过下面定义的属性访问器在对象上可访问，但是是不可变的
        self._qualified_op_name = qualified_op_name
        self.__name__ = op_name
        self._op = op
        self._overload_names = overload_names
        self._dir = []  # 初始化一个空列表 _dir
        # 检查是否存在与 self._schemas 中的任何 schema 的参数类型为 torch.ClassType 的参数
        self._has_torchbind_op_overload = any(
            _has_script_object_arg(schema) for schema in self._schemas.values()
        )

    # 由于 OpOverloadPacket 对象是不可变的，所以这是一个空操作。
    def __deepcopy__(self, memo=None):
        return self

    # 返回 OpOverloadPacket 对象的字符串表示形式，格式为 "<OpOverloadPacket(op='{}.{}')>"
    def __repr__(self):
        return "<OpOverloadPacket(op='{}.{}')>".format(
            *self._qualified_op_name.split("::")
        )

    # 返回 OpOverloadPacket 对象的字符串表示形式，格式为 "{}.{}"
    def __str__(self):
        return "{}.{}".format(*self._qualified_op_name.split("::"))

    # 返回 OpOverloadPacket 对象的哈希值，使用 self._op 的哈希值
    def __hash__(self):
        return hash(self._op)

    # 返回 OpOverloadPacket 对象的 _op 属性
    @property
    def op(self):
        return self._op

    # 返回 OpOverloadPacket 对象的 _schemas 属性，其中包含以 self._overload_names 为键的字典，
    # 每个值是调用 torch._C._get_schema 方法得到的结果
    @property
    def _schemas(self):
        return {
            overload_name: torch._C._get_schema(self._qualified_op_name, overload_name)
            for overload_name in self._overload_names
        }
    # 当访问不存在的属性时，__getattr__ 方法被调用。在这里，如果 key 是 "__file__"，返回固定字符串 "torch.ops"
    if key == "__file__":
        return "torch.ops"

    # 确保查询存在于 self._op 而不是 opoverloadpacket 上的 dunder 属性时，不会不必要地调用 `_get_operation_overload`（这是一个昂贵的操作）。
    # 这样做是为了防止潜在的减速。如果存在其他像 `__name__` 这样的属性，只存在于 self._op 而不是 opoverloadpacket 上，可以扩展这个列表。
    # 这是可以接受的，因为我们保证一个 aten 操作的重载名不会以 '__' 开头。
    try:
        if key.startswith("__"):
            return getattr(self._op, key)
    except AttributeError:
        # 为了一致性，因为抛出一个带有对象名称不同于属性查询执行对象的错误消息的属性错误似乎很奇怪。
        raise AttributeError(
            f"'{str(self)}' 不能有以 '__' 开头的重载名，而底层操作 {str(self._op)} 也没有属性 {key}。"
        ) from None

    try:
        # 这是可以接受的，因为我们保证一个 aten 操作的重载名不会是 'default'
        use_key = "" if key == "default" else key
        # TODO: 禁止访问 JIT 注册的重载
        # 获取操作重载、操作描述符和标签
        op_, op_dk_, tags = torch._C._get_operation_overload(
            self._qualified_op_name, use_key
        )
        # 获取操作的模式
        schema = torch._C._get_schema(self._qualified_op_name, use_key)
        # 如果模式不含有脚本对象参数，则创建 OpOverload 对象；否则创建 TorchBindOpOverload 对象
        overload = (
            OpOverload(self, op_, op_dk_, schema, tags)
            if not _has_script_object_arg(schema)
            else TorchBindOpOverload(self, op_, op_, schema, tags)
        )
        # 缓存重载对象
        setattr(self, key, overload)
        self._dir.append(key)
        return overload
    except RuntimeError:
        # 抛出属性错误，指出底层操作没有重载名为 'key' 的重载
        raise AttributeError(
            f"'{str(self)}' 的底层操作没有重载名 '{key}'"
        ) from None
    # 定义一个特殊方法 __call__，使得实例对象可被调用，处理参数和关键字参数
    def __call__(self_, *args, **kwargs):  # noqa: B902
        # 使用 `self_` 以避免与 aten 操作中的参数命名冲突，其中命名为 "self"
        # 这样，所有 aten 操作都可以通过关键字参数调用。

        # 重载 __call__ 方法以确保 torch.ops.foo.bar() 依然可以从 JIT 中调用
        # 我们将函数指针保存在 OpOverloadPacket 的 `op` 属性上以便在此处访问。

        # 直接调用 OverloadPacket 将进入 C++，会检查模式(schema)，当输入包含 FakeScriptObject 时可能导致错误，
        # 因此我们在此拦截它并调用 TorchBindOpverload 代替。
        if self_._has_torchbind_op_overload and _must_dispatch_in_python(args, kwargs):
            return _call_overload_packet_from_python(self_, args, kwargs)
        # 否则调用内部保存的 `_op` 方法，传入参数和关键字参数
        return self_._op(*args, **(kwargs or {}))

    # TODO: use this to make a __dir__
    # 返回一个列表，包含当前对象的所有重载方法的名称，若无则返回 "default"
    def overloads(self):
        return [n if n else "default" for n in self._overload_names]
# Note - this mirrors the logic of the cpp_function defined in jit/python/init.cpp
# _jit_get_operations, which calls _get_operation_for_overload_or_packet.
# 定义一个函数 _call_overload_packet_from_python，用于处理 OpOverloadPacket 类型的操作
def _call_overload_packet_from_python(op: OpOverloadPacket, args, kwargs):
    # Re-use the torch function handling logic in cpp
    # 调用 torch._C._maybe_call_torch_function_for_op_packet 函数处理 torch 函数调用逻辑
    torch_function_called, ret = torch._C._maybe_call_torch_function_for_op_packet(
        op, *args, **kwargs
    )

    # 如果 torch 函数已经被调用，则直接返回其结果
    if torch_function_called:
        return ret

    # The following mirrors getOpWithStack.
    # In cpp, we do a schema matching for the arguments, and call ToIValue to
    # to check whether the arguments are valid. But need to do similar things here
    # and check the schema whether the FakeScriptObject is the corresponding fake class
    # of the actual class used in schema.
    # 下面的代码与 getOpWithStack 类似。
    # 在 cpp 中，我们对参数进行模式匹配，并调用 ToIValue 来检查参数是否有效。
    # 在这里，需要进行类似的操作，并检查模式是否与 FakeScriptObject 相对应，是否是模式中使用的实际类的伪类。
    exceptions = {}
    found_op = None
    for overload_name in op.overloads():
        op_overload = getattr(op, overload_name)
        try:
            # 调用 torch._C._check_schema_allow_fake_script_object 函数检查模式是否匹配
            _ = torch._C._check_schema_allow_fake_script_object(
                op_overload._schema, *args, **kwargs
            )
            found_op = op_overload
            break
        except RuntimeError as e:
            exceptions[overload_name] = e

    # 如果找到了匹配的操作，则调用该操作
    if found_op:
        return found_op(*args, **kwargs)

    # 如果没有找到匹配的操作，抛出异常
    err_msg = (
        f"Fail to match any TorchBindOverload of {op} with following exceptions:\n"
    )
    for i, (key, msg) in enumerate(exceptions.items()):
        err_msg += f"Overload name {key}:\n {msg}\n"
    raise RuntimeError(err_msg)


# Resolution of torch.fn is different from torch.ops.aten.fn
# torch.fn uses the Python argparser, matches with the
# appropriate schema, and calls into the unboxed version of the method
# torch.ops.aten.fn resolution is done via the mechanism defined in JIT.
# JIT creates a stack of all the overloads and then tries to match the
# correct one at runtime and always calls into the boxed version of the method
# Autograd codegen creates VariableType, TracerType,
# inplace or view type and python bindings.
# Aten codegen generates tensor methods for the tensor class.

# _OpNamespace is a subclass of ModuleType because the torch script
# allows attribute lookups on modules only. Since we want torch.ops.foo.bar()
# to work from script, we need to ensure ops and foo are modules

# _OpNamespace 是 ModuleType 的子类，因为 torch 脚本只允许在模块上进行属性查找。
# 由于我们希望在脚本中使用 torch.ops.foo.bar()，我们需要确保 ops 和 foo 是模块。
class _OpNamespace(types.ModuleType):
    """
    An op namespace to dynamically bind Operators into Python.

    Say a user has created a custom Operator called "my_namespace::my_op". To
    call this op, the user will write torch.ops.my_namespace.my_op(...).
    At startup, this operation will not yet be bound into Python. Instead, the
    following sequence of magic tricks will occur:
    1. `torch.ops.my_namespace` will invoke the `__getattr__` magic method
       on the `torch.ops` object, which will create a new `_OpNamespace`
       object called `my_namespace` and set it as an attribute on the `ops`
       object.
    """
    def __init__(self, name):
        # 调用父类构造函数初始化，构造函数名为 "torch.ops." + name
        super().__init__("torch.ops." + name)
        # 设置对象的名称属性为传入的 name 参数
        self.name = name
        # 初始化对象的私有属性 _dir 为空列表
        self._dir = []

    def __iter__(self):
        # 返回对象的 _dir 属性的迭代器
        return iter(self._dir)

    def __getattr__(self, op_name):
        # 如果 op_name 为 "__file__"，返回字符串 "torch.ops"
        if op_name == "__file__":
            return "torch.ops"
        # 如果 op_name 在 ["__origin__", "__self__"] 中，则抛出 AttributeError 异常
        elif op_name in ["__origin__", "__self__"]:
            raise AttributeError(
                f"Invalid attribute '{op_name}' for '_OpNamespace' '{self.name}'"
            )

        # 设置命名空间名称为 self.name
        namespace_name = self.name
        # 构造限定操作名称，格式为 "命名空间名称::操作名称"
        qualified_op_name = f"{namespace_name}::{op_name}"
        # 构造模块名称，格式为 self.__module__ + "." + 命名空间名称
        module_name = self.__module__ + "." + namespace_name

        try:
            # 调用 _get_packet 函数获取操作及其重载列表
            op, overload_names = _get_packet(qualified_op_name, module_name)
            # 如果操作对象为 None，则抛出 AttributeError 异常
            if op is None:
                raise AttributeError(
                    f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
                )
        except RuntimeError as e:
            # 将 RuntimeError 转换为 AttributeError 异常
            raise AttributeError(
                f"'_OpNamespace' '{self.name}' object has no attribute '{op_name}'"
            ) from e

        # 设置操作对象的模块属性为 module_name
        op.__module__ = module_name
        # 构造 OpOverloadPacket 对象，封装操作及其重载列表
        opoverloadpacket = OpOverloadPacket(
            qualified_op_name, op_name, op, overload_names
        )
        # 设置 OpOverloadPacket 对象的模块属性为 self.__module__ + "." + 命名空间名称
        opoverloadpacket.__module__ = self.__module__ + "." + namespace_name
        # 缓存操作对象，确保每个操作对应唯一的 OpOverloadPacket 对象
        setattr(self, op_name, opoverloadpacket)
        # 将操作名称 op_name 添加到对象的 _dir 属性中
        self._dir.append(op_name)
        # 返回 OpOverloadPacket 对象
        return opoverloadpacket
# 定义一个函数 `_get_packet`，用于获取指定限定名称和操作模块的操作及其重载名称列表
def _get_packet(qualname, op_module):
    # 调用 Torch 的底层函数 `_jit_get_operation` 获取指定限定名称的操作及其重载名称列表
    op, overload_names = torch._C._jit_get_operation(qualname)
    if op is not None:
        # 如果操作不为空，则注册该内置操作 `op` 到指定的限定名称 `qualname` 中
        torch.jit._builtins._register_builtin(op, qualname)
        # 设置操作 `op` 的模块为 `op_module`
        op.__module__ = op_module
    # 返回获取的操作 `op` 和重载名称列表 `overload_names`
    return op, overload_names


# 定义一个函数 `_refresh_packet`，用于更新给定 packet 的操作和重载名称列表
def _refresh_packet(packet):
    # 使用 `_get_packet` 函数获取 packet 的限定操作名称和操作模块的操作及其重载名称列表
    op, overload_names = _get_packet(packet._qualified_op_name, packet._op.__module__)
    # 断言操作 `op` 不为空
    assert op is not None
    # 更新 packet 的操作 `_op` 和重载名称列表 `_overload_names`
    packet._op = op
    packet._overload_names = overload_names


# 定义一个类 `_PyOpNamespace`，继承自 `_OpNamespace` 类
class _PyOpNamespace(_OpNamespace):
    def __init__(self, name, ops):
        # 调用父类 `_OpNamespace` 的构造函数，并传入命名空间名称 `name`
        super().__init__(name)
        # 初始化 `_ops` 属性为给定的操作字典 `ops`
        self._ops = ops

    def __getattr__(self, name):
        # 覆盖父类的 `__getattr__` 方法，从 `_ops` 字典中获取名称为 `name` 的操作 `op`
        op = self._ops.get(name, None)
        if op is None:
            # 如果操作 `op` 为空，则抛出属性错误异常
            raise AttributeError(
                f"'_PyOpNamespace' '{self.name}' object has no attribute '{name}'"
            )
        # 将获取到的操作 `op` 缓存到当前对象中，并返回该操作 `op`
        setattr(self, name, op)
        return op


# 定义一个类 `_Ops`，继承自 `types.ModuleType` 类
class _Ops(types.ModuleType):
    __file__ = "_ops.py"

    def __init__(self):
        # 调用父类 `types.ModuleType` 的构造函数，指定模块名称为 "torch.ops"
        super().__init__("torch.ops")
        # 初始化 `loaded_libraries` 属性为空集合，用于存储已加载的库
        self.loaded_libraries = set()
        # 初始化 `_higher_order_op_namespace` 属性为 `_PyOpNamespace` 对象，
        # 表示高阶操作的命名空间，并传入命名空间名称和高阶操作字典 `_higher_order_ops`
        self._higher_order_op_namespace = _PyOpNamespace(
            "torch.ops.higher_order", _higher_order_ops
        )
        # 初始化 `_dir` 属性为空列表，用于存储模块中的属性名称
        self._dir = []

    def __getattr__(self, name):
        # 检查名称是否为 "higher_order"，如果是，则返回 `_higher_order_op_namespace` 属性
        if name == "higher_order":
            return self._higher_order_op_namespace

        # 如果不是 "higher_order"，则创建一个新的 `_OpNamespace` 对象 `namespace`
        namespace = _OpNamespace(name)
        # 将新创建的命名空间对象 `namespace` 设置为当前对象的属性 `name`
        setattr(self, name, namespace)
        # 将名称 `name` 添加到 `_dir` 列表中，表示该属性已被创建
        self._dir.append(name)
        # 返回新创建的命名空间对象 `namespace`
        return namespace

    def __iter__(self):
        # 返回 `_dir` 列表的迭代器，用于迭代模块中已创建的属性名称
        return iter(self._dir)

    def import_module(self, module):
        """
        Imports a Python module that has torch.library registrations.

        Generally, to extend PyTorch with custom operators, a user will
        create a Python module whose import triggers registration of
        the custom operators via a torch.ops.load_library call or a call
        to one or more torch.library.* APIs.

        It is unexpected for Python modules to have side effects, so some
        linters and formatters will complain. Use this API to import Python
        modules that contain these torch.library side effects.

        Args:
            module (str): The name of the Python module to import

        """
        # 使用 `importlib.import_module` 导入指定名称的 Python 模块
        importlib.import_module(module)
    # 如果当前是在使用 PyTorch Deploy 运行时环境中，则直接返回，不执行加载操作
    if torch._running_with_deploy():
        return

    # 解析给定路径，获取标准化的库文件路径
    path = _utils_internal.resolve_library_path(path)

    # 使用 dl_open_guard 上下文管理器保护加载操作，确保资源安全释放
    with dl_open_guard():
        # 使用 ctypes.CDLL 导入指定路径的共享库文件到当前进程中
        # 这会执行共享库的静态全局初始化代码，用于向 JIT 注册自定义操作符
        ctypes.CDLL(path)

    # 将加载的库文件路径添加到 self.loaded_libraries 属性中的集合中
    self.loaded_libraries.add(path)
# 创建一个名为 "ops" 的变量，赋值为 _Ops 类的实例，即一个操作类的实例对象
ops: _Ops = _Ops()
```