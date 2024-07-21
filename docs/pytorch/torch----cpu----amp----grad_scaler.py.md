# `.\pytorch\torch\cpu\amp\grad_scaler.py`

```
# 导入 deprecated 模块中的 deprecated 装饰器
from typing_extensions import deprecated

# 导入 torch 库
import torch

# 声明 __all__ 变量，指定 GradScaler 类为公开接口
__all__ = ["GradScaler"]

# 定义 GradScaler 类，继承自 torch.amp.GradScaler
class GradScaler(torch.amp.GradScaler):
    r"""
    See :class:`torch.amp.GradScaler`.
    ``torch.cpu.amp.GradScaler(args...)`` is deprecated. Please use ``torch.amp.GradScaler("cpu", args...)`` instead.
    """

    # 使用 deprecated 装饰器标记初始化方法
    @deprecated(
        "`torch.cpu.amp.GradScaler(args...)` is deprecated. "
        "Please use `torch.amp.GradScaler('cpu', args...)` instead.",
        category=FutureWarning,
    )
    # 初始化方法，接受多个参数并设置默认值，不返回任何值
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        # 调用父类的初始化方法，传入参数和 "cpu" 作为设备参数
        super().__init__(
            "cpu",
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
```