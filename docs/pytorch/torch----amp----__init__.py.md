# `.\pytorch\torch\amp\__init__.py`

```
# 从自动类型转换模块中导入以下函数和变量：
# - _enter_autocast: 进入自动类型转换模式的函数
# - _exit_autocast: 退出自动类型转换模式的函数
# - autocast: 自动类型转换装饰器或上下文管理器
# - custom_bwd: 自定义反向传播函数
# - custom_fwd: 自定义前向传播函数
# - is_autocast_available: 检查当前环境是否支持自动类型转换的函数
from .autocast_mode import (
    _enter_autocast,
    _exit_autocast,
    autocast,
    custom_bwd,
    custom_fwd,
    is_autocast_available,
)

# 从梯度缩放器模块中导入 GradScaler 类
from .grad_scaler import GradScaler
```