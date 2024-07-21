# `.\pytorch\torch\cuda\amp\__init__.py`

```py
# 导入自定义模块中的特定函数和类
from .autocast_mode import autocast, custom_bwd, custom_fwd
from .common import amp_definitely_not_available
from .grad_scaler import GradScaler

# 定义导出的符号列表，这些符号可以被外部模块访问
__all__ = [
    "amp_definitely_not_available",  # 禁用 AMP 的标志
    "autocast",                       # 自动类型转换装饰器
    "custom_bwd",                     # 自定义的反向传播函数
    "custom_fwd",                     # 自定义的前向传播函数
    "GradScaler",                     # 梯度缩放器类
]
```