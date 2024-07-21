# `.\pytorch\torch\distributed\fsdp\wrap.py`

```py
# Import necessary modules and types from standard libraries and torch
import contextlib  # 上下文管理工具模块
import copy  # 复制模块
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import (  # 导入类型提示相关的各种类型
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import torch.nn as nn  # 导入神经网络模块中的神经网络类


__all__ = [  # 定义可以通过 `from module import *` 导入的内容列表
    "always_wrap_policy",
    "lambda_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "size_based_auto_wrap_policy",
    "enable_wrap",
    "wrap",
    "CustomPolicy",
    "ModuleWrapPolicy",
]


# NOTE: We intentionally keep this function simple and isolate the complexity
# to `fn` to enable using this function generically. We may move this to a
# non-FSDP-specific folder and/or make it public in the future.
def _post_order_apply(
    root_module: nn.Module,
    fn: Callable[[nn.Module], Optional[nn.Module]],
):
    """
    This applies ``fn`` to every module in the module tree of ``root_module``
    following a post-order traversal. If ``fn`` returns an :class:`nn.Module`,
    then this replaces the original module with the newly returned one in the
    tree. Otherwise, ``fn`` should return ``None``, in which case the module is
    not changed.
    """
    # Track visited modules to avoid visiting shared modules multiple times
    visited_modules: Set[nn.Module] = {root_module}

    def _post_order_apply_inner(
        module: nn.Module,
        module_name: str,
        parent_module: Optional[nn.Module],
    ):
        """
        Recursively traverse the module tree starting from 'module' and apply 'fn' to each module.
        Replace the module with the returned module from 'fn' if it's not None.
        """
        for child_module_name, child_module in module.named_children():
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                _post_order_apply_inner(child_module, child_module_name, module)
        optional_module = fn(module)
        if optional_module is not None:
            assert isinstance(parent_module, nn.Module), (
                "Non-root modules should have their parent module set but got "
                f"{parent_module} for {module}"
            )
            assert module_name, (
                "Non-root modules should have their module name set but got "
                f"an empty module name for {module}"
            )
            assert isinstance(
                optional_module, nn.Module
            ), f"fn should return None or an nn.Module but got {optional_module}"
            setattr(parent_module, module_name, optional_module)

    _post_order_apply_inner(root_module, "", None)


def _construct_wrap_fn(
    root_module: nn.Module,
    target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]],
    fsdp_fn: Callable,
) -> Callable[[nn.Module], Optional[nn.Module]]:
    """
    This constructs the "wrap" function to pass to :func:`_post_order_apply`
    based on ``target_module_to_kwargs``, which should be constructed from the
    wrapping policy.
    """
    # 定义一个函数 fn，接受一个类型为 nn.Module 的参数 module，并返回一个可选的 nn.Module 对象
    def fn(module: nn.Module) -> Optional[nn.Module]:
        # 明确避免包装根模块，因为对于 FSDP（Fully Sharded Data Parallelism），根模块由调用者处理
        # 检查 module 是否在 target_module_to_kwargs 中，并且 module 不是 root_module
        if module in target_module_to_kwargs and module is not root_module:
            # 如果满足条件，从 target_module_to_kwargs 中获取对应的参数 kwargs
            kwargs = target_module_to_kwargs[module]
            # 调用 fsdp_fn 函数，传递 module 和对应的 kwargs 参数，返回结果
            return fsdp_fn(module, **kwargs)
        # 如果不满足条件，则返回 None
        return None
    
    # 返回定义的函数 fn
    return fn
def _run_mixed_precision_override_policy(
    root_module: nn.Module,
    module_classes: Iterable[Type[nn.Module]],
    ignored_modules: Set[nn.Module],
    root_kwargs: Dict[str, Any],
    target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]],
):
    # 将 module_classes 转换为不重复的元组
    module_classes_tuple = tuple(set(module_classes))
    # 遍历 root_module 的所有子模块
    for module in root_module.modules():
        # 如果当前模块在被忽略的模块集合中，则跳过
        if module in ignored_modules:
            continue
        # 如果当前模块属于 module_classes_tuple 中的某种类型
        elif isinstance(module, module_classes_tuple):
            # 此策略将覆盖任何现有的策略
            if module not in target_module_to_kwargs:
                # 如果尚未指定策略，从 root_kwargs 继承策略
                target_module_to_kwargs[module] = root_kwargs
            # 将 mixed_precision 参数设置为 None
            target_module_to_kwargs[module]["mixed_precision"] = None
    # 返回包含模块到参数字典映射的目标字典
    return target_module_to_kwargs


def always_wrap_policy(*args, **kwargs) -> bool:
    """
    A simple recursive wrap policy that always returns ``True``. This means
    that every submodule is wrapped by the wrapper class in
    :func:`_recursive_wrap`.
    """
    return True


class _Policy(ABC):
    """
    This defines an abstract base class that represents a policy for applying
    a module-level API.
    """

    @abstractmethod
    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: Set[nn.Module],
        root_kwargs: Dict[str, Any],
    ) -> Dict[nn.Module, Dict[str, Any]]:
        """
        This should return a dict ``target_module_to_kwargs`` that maps from
        each target module to wrap to its kwargs.
        """
        ...


def _module_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    module_classes: Set[Type[nn.Module]],
) -> bool:
    """
    This auto wrap policy wraps every module that is an instance of any type in
    ``module_classes`` as its own FSDP instance. The root module given by
    ``module`` is always wrapped as an FSDP instance regardless. Since the
    wrapping proceeds bottom up, each FSDP instance manages the parameters in
    its subtree excluding any already managed by a child FSDP instance.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.
        module_classes (Set[Type[nn.Module]]): Set of module classes that are
            wrapped as FSDP instances.

    Returns:
        ``True`` if ``recurse=True``, and whether ``module`` should be wrapped
        if ``recurse=False``.
    """
    if recurse:
        return True  # 总是递归
    # 如果 module 是 module_classes 中任一类型的实例，则返回 True，表示应该包装该 module
    return isinstance(module, tuple(module_classes))


class ModuleWrapPolicy(_Policy):
    """
    This policy applies to every module of the specified module classes,
    # 初始化方法，接受一个模块类的可迭代对象作为参数
    def __init__(self, module_classes: Iterable[Type[nn.Module]]):
        # 将模块类的可迭代对象转换为集合
        module_classes_set = set(module_classes)
        # 将模块类集合赋值给实例变量 _module_classes
        self._module_classes = module_classes_set
        # 将模块类集合的字符串表示赋值给实例变量 _module_classes_str
        self._module_classes_str = str(module_classes_set)

    # 策略执行方法，返回目标模块到参数字典的映射
    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: Set[nn.Module],
        root_kwargs: Dict[str, Any],
    ) -> Dict[nn.Module, Dict[str, Any]]:
        # 将实例变量 _module_classes 转换为元组
        module_classes = tuple(self._module_classes)
        # 存储目标模块到参数字典的映射
        target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]] = {}
        # 遍历根模块的所有子模块
        for module in root_module.modules():
            # 如果模块在忽略的模块集合中，则跳过
            if module in ignored_modules:
                continue
            # 如果模块属于指定的模块类
            elif isinstance(module, module_classes):
                # 浅拷贝根参数字典，以避免跨模块的耦合更改
                target_module_to_kwargs[module] = copy.copy(root_kwargs)
        # 返回目标模块到参数字典的映射
        return target_module_to_kwargs

    # 对象可调用方法，用于应用模块包装策略
    def __call__(self, module, recurse, *args, **kwargs):
        # 调用模块包装策略函数，并传递实例变量 _module_classes 作为参数
        return _module_wrap_policy(
            module, recurse, nonwrapped_numel=-1, module_classes=self._module_classes
        )

    # 返回对象的字符串表示，包括模块类集合的字符串表示
    def __repr__(self) -> str:
        # 调用父类的 __repr__() 方法，添加模块类集合的字符串表示
        return super().__repr__() + f"({self._module_classes_str})"
# 定义了一个自定义的策略类 CustomPolicy，继承自 _Policy 类
class CustomPolicy(_Policy):
    
    """
    This policy takes in a lambda function that maps a given ``nn.Module`` to
    either ``False``, ``True``, or a kwarg dictionary.
    这个策略接受一个 lambda 函数，将给定的 nn.Module 映射到 False、True 或一个关键字参数字典。
    
    - If the function returns ``False`` or an empty dictionary, then the module
      does not have the API applied.
    如果函数返回 False 或空字典，则模块不应用 API。
      
    - If the function returns ``True``, then the module has the API applied
      with the root's kwargs.
    如果函数返回 True，则模块应用了 API，使用根节点的关键字参数。
      
    - If the function returns a non-empty dictionary, then the module has the
      API applied, and the dictionary overrides the root's kwargs.
    如果函数返回非空字典，则模块应用了 API，并且该字典覆盖了根节点的关键字参数。
    """

    def __init__(self, lambda_fn: Callable[[nn.Module], Union[bool, Dict[str, Any]]]):
        # 初始化方法，接受一个 lambda 函数作为参数
        self._lambda_fn = lambda_fn

    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: Set[nn.Module],
        root_kwargs: Dict[str, Any],
    ) -> Dict[nn.Module, Dict[str, Any]]:
        # 执行策略的方法，返回一个字典，将模块映射到关键字参数字典
        target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]] = {}
        
        # 遍历根模块下的所有模块
        for module in root_module.modules():
            # 如果模块在被忽略的模块集合中，则跳过
            if module in ignored_modules:
                continue
            
            # 调用传入的 lambda 函数，将结果存储在 res 中
            res = self._lambda_fn(module)
            
            # 检查 lambda 函数的返回值类型，应为 bool 或 dict
            if not isinstance(res, (dict, bool)):
                raise ValueError(
                    "The lambda_fn passed to CustomPolicy should return "
                    f"False/True or a kwarg dict, but it returned {res}"
                )
            
            # 如果 res 为 False，则继续下一个模块
            if not res:
                continue
            
            # 复制根节点的关键字参数
            kwargs = copy.copy(root_kwargs)
            
            # 如果 res 是字典类型，则用 res 覆盖 kwargs 中的对应项
            if isinstance(res, dict):
                # 用 lambda 函数指定的关键字参数覆盖根节点的关键字参数
                kwargs.update(res)
            
            # 将模块及其对应的关键字参数字典加入目标映射字典中
            target_module_to_kwargs[module] = kwargs
        
        # 返回模块到关键字参数字典的映射结果
        return target_module_to_kwargs


def lambda_auto_wrap_policy(
    module: nn.Module, recurse: bool, nonwrapped_numel: int, lambda_fn: Callable
) -> bool:
    """
    A convenient auto wrap policy to wrap submodules based on an arbitrary user
    function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
    a `wrapper_cls` unit.

    Return if a module should be wrapped during auto wrapping.

    The first three parameters are required by :func:`_recursive_wrap`.
    
    便捷的自动包装策略，根据用户定义的函数包装子模块。
    如果 `lambda_fn(submodule) == True`，则子模块将被作为 `wrapper_cls` 单元进行包装。
    
    返回在自动包装期间是否应包装模块。
    
    前三个参数是 :func:`_recursive_wrap` 所需的参数。
    """
    Args:
        module (nn.Module): 当前被考虑的模块。
        recurse (bool): 如果为 ``False``，则此函数必须决定是否将 ``module`` 包装为 FSDP 实例。如果为 ``True``，则函数仍然在模块树中作为 DFS 的一部分递归。
        nonwrapped_numel (int): 尚未包装的参数 numel。

        lambda_fn (Callable[[nn.Module], bool]): 如果此函数返回 ``True``，则将包装此模块。
    """
    # 如果递归标志为真，始终递归处理
    if recurse:
        return True  # 总是递归处理
    # 否则，根据 lambda_fn 决定是否包装当前模块
    return lambda_fn(module)
# 定义一个函数，确定是否应该自动包装模块为 FSDP 实例的策略
def size_based_auto_wrap_policy(
    module: nn.Module,
    recurse: bool,
    nonwrapped_numel: int,
    # 额外的自定义参数
    min_num_params: int = int(1e8),
    force_leaf_modules: Optional[Set[Type[nn.Module]]] = None,
    exclude_wrap_modules: Optional[Set[Type[nn.Module]]] = None,
) -> bool:
    """
    A size-based auto wrap policy.

    Args:
        module (nn.Module): 当前正在考虑的模块。
        recurse (bool): 如果为 ``False``，则该函数必须决定是否将 ``module`` 包装为 FSDP 实例。如果为
            ``True``，则函数仍在深度优先搜索中递归遍历模块树。
        nonwrapped_numel (int): 尚未包装的参数的数量。

        min_num_params (int): 控制模块何时准备包装的大小阈值。以 numel 为单位。
        force_leaf_modules (Set[Type[nn.Module]]): 保持为叶子节点的模块类型集合，即它们的子模块永远不会被包装。
        exclude_wrap_modules (Set[Type[nn.Module]]): 在包装时应该排除的模块类型集合。

    Returns:
        bool: 是否应该包装 ``module``。
    """
    # 如果 force_leaf_modules 为 None，则使用默认的 FORCE_LEAF_MODULES 属性
    force_leaf_modules = (
        size_based_auto_wrap_policy.FORCE_LEAF_MODULES  # type: ignore[attr-defined]
        if force_leaf_modules is None
        else force_leaf_modules
    )
    # 如果 exclude_wrap_modules 为 None，则使用默认的 EXCLUDE_WRAP_MODULES 属性
    exclude_wrap_modules = (
        size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES  # type: ignore[attr-defined]
        if exclude_wrap_modules is None
        else exclude_wrap_modules
    )
    # 将 `min_num_params` 参数保留为 BC（兼容性）考虑，它表示在触发包装之前的最小非包装 *numel*。
    min_nonwrapped_numel = min_num_params
    # 判断非包装的 *numel* 是否大于或等于设定的最小非包装 *numel*
    is_large = nonwrapped_numel >= min_nonwrapped_numel
    if recurse:
        # 如果需要递归，则判断模块大小是否足够大，并且不在 `force_leaf_modules` 列表中才递归。
        return is_large and not isinstance(module, tuple(force_leaf_modules))
    else:
        # 如果不需要递归，判断是否应该进行包装。
        return is_large and not isinstance(module, tuple(exclude_wrap_modules))
# 将这些默认设置应用于 size_based_auto_wrap_policy 函数。方便导入时使用。
size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {nn.ModuleList, nn.ModuleDict}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {nn.MultiheadAttention}  # type: ignore[attr-defined]

# 定义一个上下文管理器 enable_wrap，用于为模块应用包装器
@contextlib.contextmanager
def enable_wrap(
    *, wrapper_cls: Any, **wrapper_kwargs: Any
) -> Generator[None, None, None]:
    """
    Context manager to wrap modules using a wrapper.

    Useful for when you'd like to apply the same configuration arguments to all
    child modules that you wrap. A particularly important use case is wrapping
    large layers so that they get sharded (in-place) during initialization, to
    avoid running out of system memory. Large layers can indicate that they
    should be sharded via the ``wrap`` annotation and this context manager can
    provide the exact configuration for these nested instances.

    Usage::

        with enable_wrap(wrapper_cls, **params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        wrapper_cls:
            Class that `wrap` annotation will `wrap` modules with, such as
            `FullyShardedDataParallel`.
        **wrapper_kwargs:
            Configuration settings that will be passed to all ``wrap``
            instances inside the context
    """
    kwargs = {
        "wrapper_cls": wrapper_cls,
        **wrapper_kwargs,
    }
    # 进入 _ConfigAutoWrap 上下文，传入参数 kwargs
    with _ConfigAutoWrap(**kwargs):
        yield


# 定义 wrap 函数，用于注明一个模块应该被包装
def wrap(module: nn.Module, **wrap_overrides: Any) -> nn.Module:
    """
    Annotate that a module should be wrapped. Annotated modules will only be
    wrapped if inside of an :func:`enable_wrap` context manager. This allows
    a module to be initialized both with and without a wrapper without code
    change.

    The class that this function wraps the passed in ``nn.Module`` with is the
    passed in ``wrapper_cls`` argument into ``enable_wrap``. Both
    ``enable_wrap`` and ``wrap`` can take in kwargs specifying how to construct
    the ``wrapper_cls`` instance. In the case of duplicate kwargs in
    ``enable_wrap`` and ``wrap``, the argument passed into ``wrap`` will be
    respected.

    Usage::

        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        module (nn.Module): module to wrap (if in :func:`enable_wrap` context)
        **wrap_overrides: configuration overrides that will take priority over
            the values provided by the :func:`enable_wrap` context
    """
    # 如果当前在自动包装的上下文中
    if _ConfigAutoWrap.in_autowrap_context:
        # 确保 _ConfigAutoWrap.wrapper_cls 不为空
        assert _ConfigAutoWrap.wrapper_cls is not None

        # 合并 wrap_overrides 和 _ConfigAutoWrap.kwargs，wrap_overrides 优先
        wrap_overrides = {**_ConfigAutoWrap.kwargs, **wrap_overrides}
        # 调用 _wrap 函数，将 module 包装为 _ConfigAutoWrap.wrapper_cls 类型的模块
        return _wrap(
            module,
            _ConfigAutoWrap.wrapper_cls,
            **wrap_overrides,
        )
    # 返回当前循环迭代的模块对象
    return module
def _wrap(module: nn.Module, wrapper_cls: Callable, **kwargs) -> nn.Module:
    assert wrapper_cls is not None
    # 检查传入的 wrapper_cls 参数不为空

    if hasattr(module, "_wrap_overrides"):
        # 如果 module 有 _wrap_overrides 属性，强制使用这些属性覆盖 FSDP 配置给这个 module
        # 目前仅用于在 auto_wrapping 时禁用 BatchNorm 的混合精度
        overrides = {**kwargs, **module._wrap_overrides}  # 合并 kwargs 和 module._wrap_overrides
        return wrapper_cls(module, **overrides)

    return wrapper_cls(module, **kwargs)
    # 如果 module 没有 _wrap_overrides 属性，则直接使用传入的参数创建 wrapper_cls 对象并返回


def _recursive_wrap(
    module: nn.Module,
    auto_wrap_policy: Callable,
    wrapper_cls: Callable,
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
    only_wrap_children: bool = False,
    **kwargs: Any,
) -> Tuple[nn.Module, int]:
    """
    Wraps submodules of ``module`` for which ``auto_wrap_policy`` returns
    ``True`` with ``wrapper_cls``.

    Args:
        module (nn.Module): Module to recursively wrap.
        auto_wrap_policy (Callable): A callable representing a policy that
            determines which modules to recursively wrap with ``wrapper_cls``.
        ignored_modules (Set[torch.nn.Module]): Modules to ignore when
            wrapping.
        ignored_params (Set[torch.nn.Parameter]): Parameters to ignore when
            wrapping; these should be the parameters contained in the modules
            in ``ignored_modules``.
        only_wrap_children (bool): Flag indicating whether to wrap only children modules.
        **kwargs (Any): Additional keyword arguments passed to wrapper_cls.

    Returns:
        (nn.Module, int):
            ``module`` after wrapping and the numel recursively wrapped.
    """
    assert auto_wrap_policy is not None, "Must specify auto_wrap_policy."
    assert wrapper_cls is not None, "Must specify wrapper_cls"

    # 确保没有已经被包装的子模块
    for _, child in module.named_modules():
        if child in ignored_modules:
            continue
        try:
            assert not isinstance(child, cast(type, wrapper_cls))
        except TypeError:
            # 如果 wrapper_cls 是函数而不是类类型，则跳过上述检查
            pass

    # 统计所有参数的 numel，假设它们都没有被包装过
    nonwrapped_numel = sum(
        p.numel() for p in module.parameters() if p not in ignored_params
    )

    assert auto_wrap_policy is not None
    # 根据自动包装策略检查模块是否需要进行包装
    if auto_wrap_policy(module=module, recurse=True, nonwrapped_numel=nonwrapped_numel):
        # 初始化已包装参数的计数
        total_wrapped_numel = 0
        # 遍历模块的子模块，递归地进行必要的包装操作
        for name, child in module.named_children():
            # 如果子模块在被忽略的列表中，则跳过
            if child in ignored_modules:
                continue
            # 递归调用 _recursive_wrap 函数，对子模块进行包装，并获取已包装参数的数量
            wrapped_child, num_wrapped_params = _recursive_wrap(
                module=child,
                auto_wrap_policy=auto_wrap_policy,
                wrapper_cls=wrapper_cls,
                ignored_modules=ignored_modules,
                ignored_params=ignored_params,
                **kwargs,
            )
            # 将包装后的子模块设置为当前模块的属性
            setattr(module, name, wrapped_child)
            # 更新已包装参数的总数
            total_wrapped_numel += num_wrapped_params
        
        # 计算剩余未包装参数的数量
        remainder = nonwrapped_numel - total_wrapped_numel
        # 如果不仅包装子模块，并且根据策略当前模块需要被包装，则执行以下操作
        if not only_wrap_children and auto_wrap_policy(
            module=module, recurse=False, nonwrapped_numel=remainder
        ):
            # 返回当前模块经过包装后的结果以及未包装的参数数量
            return _wrap(module, wrapper_cls, **kwargs), nonwrapped_numel
        else:
            # 否则返回当前模块和已包装的参数总数
            return module, total_wrapped_numel
    # 如果根据策略不需要对模块进行包装，则直接返回当前模块和已包装的参数总数为 0
    return module, 0
class _ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """

    in_autowrap_context: bool = False  # Context flag
    wrapper_cls: Optional[Callable] = None  # The wrapper class
    kwargs: Dict[str, Any] = {}  # Wrapper's args

    def __init__(self, **kwargs: Dict[str, Any]):
        # Initialize with keyword arguments passed to the instance
        self.kwargs = kwargs

    @staticmethod
    def enable_autowrap_context(kwargs: Any) -> None:
        # Static method to enable autowrap context
        if _ConfigAutoWrap.in_autowrap_context:
            # Check if already in autowrap context and raise error if nested autowrap is attempted
            raise NotImplementedError(
                "You are already within an autowrap context and we currently do not supported nested autowrap."
            )
        _ConfigAutoWrap.in_autowrap_context = True
        # Get and save the wrapper class for the context
        assert (
            "wrapper_cls" in kwargs.keys()
        ), "Expected to pass in wrapper_cls arg into _ConfigAutoWrap."
        _ConfigAutoWrap.wrapper_cls = cast(Callable, kwargs["wrapper_cls"])
        del kwargs["wrapper_cls"]
        # Save the remaining keyword arguments
        _ConfigAutoWrap.kwargs = kwargs

    @staticmethod
    def disable_autowrap_context() -> None:
        # Static method to disable autowrap context and reset variables
        _ConfigAutoWrap.in_autowrap_context = False
        _ConfigAutoWrap.wrapper_cls = None
        _ConfigAutoWrap.kwargs = {}

    def __enter__(self) -> None:
        # Context manager entry point, enables autowrap context
        self.enable_autowrap_context(self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Context manager exit point, disables autowrap context
        self.disable_autowrap_context()
```