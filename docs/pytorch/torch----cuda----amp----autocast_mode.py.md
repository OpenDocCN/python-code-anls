# `.\pytorch\torch\cuda\amp\autocast_mode.py`

```
# 指定允许未类型化的函数定义
# 导入 functools 和 typing 模块中的 Any 类型
import functools
from typing import Any
from typing_extensions import deprecated

# 导入 torch 库
import torch

# 定义公开的符号列表
__all__ = ["autocast", "custom_fwd", "custom_bwd"]

# 定义 autocast 类，继承自 torch.amp.autocast_mode.autocast
class autocast(torch.amp.autocast_mode.autocast):
    r"""See :class:`torch.autocast`.

    ``torch.cuda.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cuda", args...)`` instead.
    """

    # 标记初始化函数为过时
    @deprecated(
        "`torch.cuda.amp.autocast(args...)` is deprecated. "
        "Please use `torch.amp.autocast('cuda', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_enabled: bool = True,
    ):
        # 如果处于脚本化状态，则设置成员变量并返回
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cuda"
            self.fast_dtype = dtype
            return
        # 否则调用父类的初始化方法
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    # 进入上下文管理器时调用的方法
    def __enter__(self):
        # 如果处于脚本化状态，则直接返回自身
        if torch._jit_internal.is_scripting():
            return self
        # 否则调用父类的方法
        return super().__enter__()

    # 退出上下文管理器时调用的方法
    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        # 如果处于脚本化状态，则直接返回
        if torch._jit_internal.is_scripting():
            return
        # 否则调用父类的方法
        return super().__exit__(exc_type, exc_val, exc_tb)

    # 对象被调用时调用的方法
    def __call__(self, func):
        # 如果处于脚本化状态，则直接返回 func
        if torch._jit_internal.is_scripting():
            return func
        # 否则调用父类的方法
        return super().__call__(func)


# 保留用于向后兼容的函数定义
@deprecated(
    "`torch.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated. "
    "Please use `torch.amp.autocast_mode._cast(value, 'cuda', dtype)` instead.",
    category=FutureWarning,
)
def _cast(value, dtype):
    # 调用 torch.amp.autocast_mode._cast 函数，将 value 强制转换为指定的 dtype 类型
    return torch.amp.autocast_mode._cast(value, "cuda", dtype)


# 标记为过时的函数定义
@deprecated(
    "`torch.cuda.amp.custom_fwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``torch.cuda.amp.custom_fwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_fwd(args..., device_type='cuda')`` instead.
    """
    # 返回一个 functools.partial 对象，用于调用 torch.amp.custom_fwd 函数，指定 device_type='cuda'
    return functools.partial(torch.amp.custom_fwd, device_type="cuda")(
        fwd=fwd, cast_inputs=cast_inputs
    )


# 标记为过时的函数定义
@deprecated(
    "`torch.cuda.amp.custom_bwd(args...)` is deprecated. "
    "Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd):
    """
    ``torch.cuda.amp.custom_bwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_bwd(args..., device_type='cuda')`` instead.
    """
    # 返回一个 functools.partial 对象，用于调用 torch.amp.custom_bwd 函数，指定 device_type='cuda'
    return functools.partial(torch.amp.custom_bwd, device_type="cuda")(bwd)
```