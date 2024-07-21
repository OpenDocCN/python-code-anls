# `.\pytorch\torch\distributed\_composable\checkpoint_activation.py`

```
# 设置 mypy 选项，允许未经类型定义的函数
# 从 contextlib 中导入 contextmanager 和 nullcontext
# 从 typing 中导入 Any、ContextManager、Dict、Optional、Tuple
from contextlib import contextmanager, nullcontext
from typing import Any, ContextManager, Dict, Optional, Tuple

# 导入 torch 库
import torch
# 从 torch.nn 中导入 nn 模块
import torch.nn as nn
# 从 torch.utils.checkpoint 中导入 _checkpoint_without_reentrant_generator 和 _DEFAULT_DETERMINISM_MODE
from torch.utils.checkpoint import (
    _checkpoint_without_reentrant_generator,
    _DEFAULT_DETERMINISM_MODE,
)

# 从当前包中导入 contract 模块
from .contract import contract

# 定义一个上下文管理器函数 _no_hook，用于禁用由 checkpoint 安装的钩子，以避免在反向重计算期间意外递归
@contextmanager
def _no_hook(module: nn.Module, user_ctx: Optional[ContextManager] = None):
    r"""
    Disable hooks installed by checkpoint to avoid unintentional recursion
    during backward recomputation.
    """
    # 如果提供了用户上下文管理器，则使用它；否则使用 nullcontext
    with user_ctx if user_ctx else nullcontext():
        # 获取模块当前的钩子状态
        orig_enable_hook = checkpoint.state(module).enable_hook
        # 禁用当前模块的钩子
        checkpoint.state(module).enable_hook = False
        try:
            # 执行 yield 语句
            yield
        finally:
            # 恢复模块原始的钩子状态
            checkpoint.state(module).enable_hook = orig_enable_hook

# 定义函数 checkpoint，实现可组合的激活检查点 API
@contract()
def checkpoint(module: nn.Module, **kwargs) -> nn.Module:
    r"""
    This is a composable activation checkpointing API. Unlike functional
    activation checkpointing APIs, this one does not require changing model
    source code. Unlike ``nn.Module`` wrapper activation checkpointing APIs,
    this one does not modify model structure or fully-qualified names either.
    Under the hood, it registers activation checkpointing logic as pre- and
    post-forward hooks. Hence, this API can be easily applied to any model or
    sub-modules in the model.

    Args:
        module (nn.Module): the target model or sub-module to apply activation
            checkpointing.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> model = MyModel()
        >>> checkpoint(model.l1)  # apply activation checkpointing only to l1
        >>> model(torch.zeros(2, 10)).sum().backward()

    """
    # 记录 API 使用情况
    torch._C._log_api_usage_once("torch.distributed.checkpoint")

    # 获取 use_reentrant 参数，如果设置为 True，抛出 NotImplementedError
    use_reentrant = kwargs.pop("use_reentrant", False)
    if use_reentrant:
        raise NotImplementedError(
            "use_reentrant=True is not supported in composable checkpoint. "
            "Please use torch.utils.checkpoint.checkpoint instead."
        )
    # 获取 preserve_rng_state 参数，默认为 True，保留 RNG 状态
    preserve_rng_state = kwargs.pop("preserve_rng_state", True)
    # 获取 context_fn 参数，用于用户定义的上下文函数
    user_context_fns = kwargs.pop("context_fn", None)
    # 获取 determinism_check 参数，检测确定性模式，默认为 _DEFAULT_DETERMINISM_MODE
    determinism_check = kwargs.pop("determinism_check", _DEFAULT_DETERMINISM_MODE)
    # 获取 debug 参数，默认为 False，用于调试模式
    debug = kwargs.pop("debug", False)

    # 如果还有未处理的 kwargs 参数，抛出 ValueError
    if kwargs:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    # 定义 forward_pre_hook 函数，作为前向传播前的钩子函数
    def forward_pre_hook(
        module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        # 如果启用了模块的检查点钩子
        if checkpoint.state(module).enable_hook:

            def context_fns():
                # 如果用户定义了上下文函数
                if user_context_fns is not None:
                    # 调用用户定义的上下文函数，并获取其返回的两个上下文对象
                    ctx1, ctx2 = user_context_fns()
                    # 返回一个空的上下文对象和使用 _no_hook 函数处理后的第二个上下文对象
                    return ctx1, _no_hook(module, ctx2)
                else:
                    # 返回一个空的上下文对象和使用 _no_hook 函数处理后的上下文对象
                    return nullcontext(), _no_hook(module)

            # 将非重入生成器检查点绑定到模块的状态中
            checkpoint.state(
                module
            )._ac_generator = _checkpoint_without_reentrant_generator(
                module,
                preserve_rng_state,
                context_fns,
                determinism_check,
                debug,
                *args,
                **kwargs,
            )
            # 启动非重入生成器
            next(checkpoint.state(module)._ac_generator)

    # 前向钩子函数，用于处理模块的前向传播
    def forward_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
        # 如果启用了模块的检查点钩子
        if checkpoint.state(module).enable_hook:
            try:
                # 启动非重入生成器
                next(checkpoint.state(module)._ac_generator)
            except StopIteration:
                pass
            else:
                # 如果生成器没有耗尽，则抛出运行时错误
                raise RuntimeError(
                    "Expected non-reentrant activation checkpoint generator to be exhausted, but it was not!"
                )

        # 确保不再持有生成器。即使在前向传播中发生异常，always_call=True 也能确保清除这一状态。
        checkpoint.state(module)._ac_generator = None

    # 启用模块的检查点钩子
    checkpoint.state(module).enable_hook = True
    # 注册前向传播预钩子函数
    module.register_forward_pre_hook(forward_pre_hook, with_kwargs=True)
    # 注册前向传播钩子函数，优先级最高，并且总是调用
    module.register_forward_hook(forward_hook, prepend=True, always_call=True)
    # 返回修改后的模块对象
    return module
```