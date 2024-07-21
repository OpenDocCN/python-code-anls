# `.\pytorch\torch\cpu\amp\autocast_mode.py`

```
# mypy: allow-untyped-defs
# 导入必要的类型定义模块
from typing import Any
from typing_extensions import deprecated

# 导入 PyTorch 库
import torch

# 定义公开的接口列表
__all__ = ["autocast"]

# 自定义类 autocast，继承自 torch.amp.autocast_mode.autocast
class autocast(torch.amp.autocast_mode.autocast):
    """
    See :class:`torch.autocast`.
    ``torch.cpu.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cpu", args...)`` instead.
    """

    # 使用装饰器标记，表示该函数已被废弃
    @deprecated(
        "`torch.cpu.amp.autocast(args...)` is deprecated. "
        "Please use `torch.amp.autocast('cpu', args...)` instead.",
        category=FutureWarning,
    )
    # 初始化函数，设置自动类型转换的参数
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        cache_enabled: bool = True,
    ):
        # 如果正在进行 Torch 脚本化，则直接设置基本属性
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cpu"
            self.fast_dtype = dtype
            return
        # 否则调用父类的初始化方法来设置参数
        super().__init__(
            "cpu", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    # 进入上下文管理器时的操作
    def __enter__(self):
        # 如果正在进行 Torch 脚本化，则直接返回自身
        if torch._jit_internal.is_scripting():
            return self
        # 否则调用父类的 __enter__ 方法
        return super().__enter__()

    # 退出上下文管理器时的操作
    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        # 如果正在进行 Torch 脚本化，则直接返回
        if torch._jit_internal.is_scripting():
            return
        # 否则调用父类的 __exit__ 方法
        return super().__exit__(exc_type, exc_val, exc_tb)

    # 对象被调用时的操作，用于装饰函数
    def __call__(self, func):
        # 如果正在进行 Torch 脚本化，则直接返回原始函数
        if torch._jit_internal.is_scripting():
            return func
        # 否则调用父类的 __call__ 方法
        return super().__call__(func)
```