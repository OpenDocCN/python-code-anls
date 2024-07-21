# `.\pytorch\torch\ao\nn\quantized\__init__.py`

```
# 导入本地的 functional 模块
from . import functional
# 导入所有模块，禁止警告 F403
from .modules import *  # noqa: F403
# 从 modules 模块中单独导入 MaxPool2d 类
from .modules import MaxPool2d

# 定义一个公开的符号列表，包含所有需要公开的模块和类名
__all__ = [
    'BatchNorm2d',         # 批归一化在二维数据上的应用
    'BatchNorm3d',         # 批归一化在三维数据上的应用
    'Conv1d',              # 一维卷积层
    'Conv2d',              # 二维卷积层
    'Conv3d',              # 三维卷积层
    'ConvTranspose1d',     # 一维转置卷积层（反卷积）
    'ConvTranspose2d',     # 二维转置卷积层（反卷积）
    'ConvTranspose3d',     # 三维转置卷积层（反卷积）
    'DeQuantize',          # 量化反转操作
    'ELU',                 # ELU 激活函数
    'Embedding',           # 嵌入层，用于将离散数据映射到连续向量空间
    'EmbeddingBag',        # 带加权嵌入的池化操作
    'GroupNorm',           # 分组归一化
    'Hardswish',           # Hardswish 激活函数
    'InstanceNorm1d',      # 一维实例归一化
    'InstanceNorm2d',      # 二维实例归一化
    'InstanceNorm3d',      # 三维实例归一化
    'LayerNorm',           # 层归一化
    'LeakyReLU',           # Leaky ReLU 激活函数
    'Linear',              # 线性层（全连接层）
    'LSTM',                # 长短期记忆网络
    'MultiheadAttention',  # 多头注意力机制
    'Quantize',            # 量化操作
    'ReLU6',               # ReLU6 激活函数
    'Sigmoid',             # Sigmoid 激活函数
    'Softmax',             # Softmax 激活函数
    'Dropout',             # 随机失活层
    'PReLU',               # 参数化 ReLU 激活函数
    # 封装模块
    'FloatFunctional',     # 浮点功能封装
    'FXFloatFunctional',   # FXFloat 功能封装
    'QFunctional',         # 量化功能封装
]
```