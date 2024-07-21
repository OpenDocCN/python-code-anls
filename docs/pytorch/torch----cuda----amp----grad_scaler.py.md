# `.\pytorch\torch\cuda\amp\grad_scaler.py`

```
from typing_extensions import deprecated  # 导入 deprecated 装饰器

import torch  # 导入 PyTorch 库

# We need to keep this unused import for BC reasons
from torch.amp.grad_scaler import OptState  # noqa: F401  # 保留此未使用的导入，用于向后兼容性

__all__ = ["GradScaler"]  # 定义模块的公开接口列表，只包含 GradScaler 类


class GradScaler(torch.amp.GradScaler):
    r"""
    See :class:`torch.amp.GradScaler`.
    ``torch.cuda.amp.GradScaler(args...)`` is deprecated. Please use ``torch.amp.GradScaler("cuda", args...)`` instead.
    """

    @deprecated(
        "`torch.cuda.amp.GradScaler(args...)` is deprecated. "
        "Please use `torch.amp.GradScaler('cuda', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        init_scale: float = 2.0**16,  # 初始化缩放因子，默认为 2 的 16 次方
        growth_factor: float = 2.0,  # 生长因子，默认为 2.0
        backoff_factor: float = 0.5,  # 后退因子，默认为 0.5
        growth_interval: int = 2000,  # 生长间隔，默认为 2000
        enabled: bool = True,  # 是否启用，默认为 True
    ) -> None:
        super().__init__(
            "cuda",  # 使用 'cuda' 设备初始化父类 GradScaler
            init_scale=init_scale,  # 设置初始缩放因子
            growth_factor=growth_factor,  # 设置生长因子
            backoff_factor=backoff_factor,  # 设置后退因子
            growth_interval=growth_interval,  # 设置生长间隔
            enabled=enabled,  # 设置是否启用
        )
```