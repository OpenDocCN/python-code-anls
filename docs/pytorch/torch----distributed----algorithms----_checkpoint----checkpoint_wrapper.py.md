# `.\pytorch\torch\distributed\algorithms\_checkpoint\checkpoint_wrapper.py`

```
# mypy: allow-untyped-defs
# 引入警告模块，用于在运行时产生警告信息
import warnings
# 从枚举模块中导入auto函数和Enum基类
from enum import auto, Enum
# 导入functools模块中的partial函数，用于创建偏函数
from functools import partial
# 导入类型提示模块中的类型定义
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch的图自动求导模块中的save_on_cpu函数
from torch.autograd.graph import save_on_cpu
# 导入PyTorch的分布式工具模块中的一些函数
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
# 导入PyTorch的工具模块中的checkpoint函数
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint

# 定义常量_CHECKPOINT_WRAPPED_MODULE，表示被检查点包装的模块名称
_CHECKPOINT_WRAPPED_MODULE = "_checkpoint_wrapped_module"
# 定义常量_CHECKPOINT_PREFIX，表示检查点前缀
_CHECKPOINT_PREFIX = _CHECKPOINT_WRAPPED_MODULE + "."

# 定义枚举类型CheckpointImpl，包含REENTRANT和NO_REENTRANT两个枚举值
class CheckpointImpl(Enum):
    REENTRANT = auto()  # 自动分配枚举值
    NO_REENTRANT = auto()  # 自动分配枚举值

# 定义ActivationWrapper类，继承自torch.nn.Module
class ActivationWrapper(torch.nn.Module):
    """
    激活检查点和激活卸载的基类。

    不应直接实例化。
    """

    def __init__(self, mod):
        super().__init__()
        self._checkpoint_wrapped_module = mod  # 初始化检查点包装的模块
        # 注册状态字典钩子以移除前缀，允许加载到非检查点包装的模块中
        self._register_state_dict_hook(self._post_state_dict_hook)
        # 注册加载状态字典前钩子，允许加载回检查点包装的模块
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )

    def forward(self, *args, **kwargs):
        raise ValueError("Subclasses should implement forward().")  # 抛出数值错误异常

    def __getattr__(self, name: str) -> Any:
        """将缺失的属性转发到包装的模块。"""
        try:
            return super().__getattr__(name)  # 委托给nn.Module的逻辑
        except AttributeError:
            return getattr(self._checkpoint_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """在模块是nn.Sequential的情况下，转发索引调用。"""
        return self._checkpoint_wrapped_module.__getitem__(key)  # 忽略类型检查[操作员]

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        重写:meth:`named_parameters()`以拦截参数名称。

        删除所有出现的_CHECKPOINT_PREFIX。
        """
        for param_name, param in super().named_parameters(*args, **kwargs):
            yield param_name.replace(_CHECKPOINT_PREFIX, ""), param

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> Dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this FSDP module is executed.

        For ``checkpoint_wrapper``, it will strip checkpoint-wrapped module prefix,
        so that this module can be loaded into non-checkpointed modules.
        It would still be able to be loaded into checkpoint-wrapped modules as this class,
        adds the prefix back before loading the state_dict.
        """
        # 使用 _replace_by_prefix 函数，去除 state_dict 中的带有 _CHECKPOINT_PREFIX 的模块前缀
        _replace_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}", prefix)
        # 返回修改后的 state_dict
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()`` is called.

        For ``checkpoint_wrapper``, it will add back the module
        prefix so that non-checkpointed modules can be loaded into
        checkpoint_wrapper modules properly.
        """
        # 使用 _replace_by_prefix 函数，添加 _CHECKPOINT_PREFIX 到 state_dict 中每个键的前缀
        _replace_by_prefix(state_dict, prefix, prefix + f"{_CHECKPOINT_PREFIX}")
class OffloadWrapper(ActivationWrapper):
    # 继承自 ActivationWrapper 的 OffloadWrapper 类
    def __init__(self, mod):
        # 调用父类 ActivationWrapper 的构造函数
        super().__init__(mod)

    def forward(self, *args, **kwargs):
        # 在 CPU 上保存中间结果
        with save_on_cpu(pin_memory=True):
            # 调用 _checkpoint_wrapped_module 方法进行前向传播
            return self._checkpoint_wrapped_module(*args, **kwargs)


class CheckpointWrapper(ActivationWrapper):
    """
    An ``nn.Module`` that wraps another ``nn.Module`` with checkpointing.

    Note that this module is not meant to be used directly but instead,
    it is to be used through the ``checkpoint_wrapper`` function.
    """

    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
        checkpoint_fn=None,
        **checkpoint_fn_kwargs,
    ):
        # 调用父类 ActivationWrapper 的构造函数
        super().__init__(mod)
        # 设置检查点实现方式
        self.checkpoint_impl = checkpoint_impl
        if checkpoint_fn is None:
            # 如果未指定 checkpoint_fn，则使用 torch.utils.checkpoint
            self.checkpoint_fn = partial(
                torch_utils_checkpoint,
                use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
                **checkpoint_fn_kwargs,
            )
        else:
            # 构建用户指定的 checkpoint_fn 函数
            self.checkpoint_fn = partial(
                checkpoint_fn,
                **checkpoint_fn_kwargs,
            )

    def forward(self, *args, **kwargs):
        # 支持关键字参数用于可重入检查点
        if self.checkpoint_impl == CheckpointImpl.REENTRANT and kwargs != {}:
            # 打包参数和关键字参数
            flat_args, kwarg_keys = _pack_kwargs(*args, **kwargs)

            # 定义一个仅接受打包参数的函数
            def my_function(*inputs):
                # 解包为原始的 args 和 kwargs，运行原始模块
                unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
                return self._checkpoint_wrapped_module(
                    *unpacked_args, **unpacked_kwargs
                )

            # 将仅接受打包参数的函数传递给可重入检查点 API
            return self.checkpoint_fn(
                my_function,
                *flat_args,
            )
        else:
            return self.checkpoint_fn(
                self._checkpoint_wrapped_module, *args, **kwargs
            )


def offload_wrapper(module: torch.nn.Module) -> torch.nn.Module:
    """
    Wrap a module for activation offloading to CPU.

    Offloads intermediate activations to the CPU for modules wrapped with this function.
    Wrappers with activation offload can be composed with ones that do recomputation-based
    """
    # 返回一个将中间激活值迁移到 CPU 的模块
    checkpoint to trade off increased compute versus increased CPU
    memory usage and additional H2D transfers.
    ```
    # 这个函数是用来实现模型的checkpointing，以在增加计算量的同时减少CPU内存使用和额外的H2D传输。

    """
    Usage::
        offloaded_module = offload_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
    Returns:
        (nn.Module):
            Wrapped module
    """
    # 返回经过OffloadWrapper封装后的模块，用于实现checkpointing机制。
    return OffloadWrapper(module)
    ```
# 定义一个函数，用于为神经网络模块添加激活检查点功能
def apply_activation_checkpointing(
    model,  # 输入参数：神经网络模型
    checkpoint_wrapper_fn=checkpoint_wrapper,  # 检查点包装器函数，默认为checkpoint_wrapper
    check_fn=lambda _: True,  # 检查函数，决定是否对模块进行检查点包装，默认始终返回True
    auto_wrap_policy: Optional[Callable[[nn.Module, bool, int], bool]] = None,  # 自动包装策略，用于控制是否自动包装模块
):
    """
    Apply :func:`checkpoint_wrapper` to modules within `model` based on a user-defined configuration.

    For each module within `model`, the `check_fn` is used to decide
    whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

    Note::
        This function modifies `model` in place and replaces appropriate layers with
        their checkpoint-wrapped modules.
    Note::
        This function will not wrap the overall root module. If this is needed, please directly use
        :func:`checkpoint_wrapper` or :func:`offload_wrapper`.
    """
    # TODO: 在函数内部导入，以避免 FSDP 和 checkpoint_wrapper 之间的循环导入问题。
    # 这可以在 wrap() API 从 FSDP 代码中解耦之后解决。
    from torch.distributed.fsdp._wrap_utils import _construct_wrap_fn, _post_order_apply
    from torch.distributed.fsdp.wrap import (
        _Policy,
        _recursive_wrap,
        lambda_auto_wrap_policy,
    )

    # 根据 auto_wrap_policy 是否为空来确定最终的策略
    policy = (
        auto_wrap_policy
        if auto_wrap_policy is not None
        else partial(lambda_auto_wrap_policy, lambda_fn=check_fn)
    )
    
    # 如果 policy 不是可调用对象，则抛出异常
    if not callable(policy):
        if not isinstance(policy, _Policy):
            raise ValueError(
                f"Expected {policy} to be callable or be a pre-defined wrap policy"
            )
        
        # 运行指定的策略，并获取目标模块到参数字典的映射
        target_module_to_kwargs = policy._run_policy(
            model, ignored_modules=set(), root_kwargs={}
        )
        
        # 构造包装函数，用于将目标模块包装起来
        wrap_fn = _construct_wrap_fn(
            model, target_module_to_kwargs, checkpoint_wrapper_fn
        )
        
        # 对模型应用后序遍历的方式应用包装函数
        _post_order_apply(model, wrap_fn)
        return

    # 递归地对模型应用自动包装策略
    _recursive_wrap(
        module=model,
        auto_wrap_policy=policy,  # type: ignore[arg-type]
        wrapper_cls=checkpoint_wrapper_fn,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True,
    )
```