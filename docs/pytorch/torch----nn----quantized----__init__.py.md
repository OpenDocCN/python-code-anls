# `.\pytorch\torch\nn\quantized\__init__.py`

```
# 导入动态模块，禁止检查 F403
from . import dynamic  # noqa: F403

# 导入函数模块，禁止检查 F403
from . import functional  # noqa: F403

# 导入模块模块，禁止检查 F403
from . import modules  # noqa: F403

# 从模块导入所有内容，禁止检查 F403
from .modules import *  # noqa: F403

# 从模块导入 MaxPool2d 类
from .modules import MaxPool2d

# 定义 __all__ 列表，包含所有公开的类和函数名
__all__ = [
    'BatchNorm2d',          # 二维批量归一化
    'BatchNorm3d',          # 三维批量归一化
    'Conv1d',               # 一维卷积
    'Conv2d',               # 二维卷积
    'Conv3d',               # 三维卷积
    'ConvTranspose1d',      # 一维转置卷积
    'ConvTranspose2d',      # 二维转置卷积
    'ConvTranspose3d',      # 三维转置卷积
    'DeQuantize',           # 取消量化
    'Dropout',              # 随机失活
    'ELU',                  # 指数线性单元
    'Embedding',            # 嵌入层
    'EmbeddingBag',         # 嵌入包
    'GroupNorm',            # 分组归一化
    'Hardswish',            # 硬切线激活函数
    'InstanceNorm1d',       # 一维实例归一化
    'InstanceNorm2d',       # 二维实例归一化
    'InstanceNorm3d',       # 三维实例归一化
    'LayerNorm',            # 层归一化
    'LeakyReLU',            # 泄漏整流线性单元
    'Linear',               # 线性层
    'LSTM',                 # 长短期记忆网络
    'MultiheadAttention',   # 多头注意力
    'PReLU',                # 参数化整流线性单元
    'Quantize',             # 量化
    'ReLU6',                # ReLU6 激活函数
    'Sigmoid',              # sigmoid 激活函数
    'Softmax',              # softmax 激活函数
    # 封装模块
    'FloatFunctional',      # 浮点数功能模块
    'FXFloatFunctional',    # FX 浮点数功能模块
    'QFunctional',          # 量化功能模块
]
```