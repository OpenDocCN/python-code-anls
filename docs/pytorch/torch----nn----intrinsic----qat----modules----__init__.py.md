# `.\pytorch\torch\nn\intrinsic\qat\modules\__init__.py`

```py
# 从当前目录下的 linear_relu 模块中导入 LinearReLU 类
from .linear_relu import LinearReLU
# 从当前目录下的 linear_fused 模块中导入 LinearBn1d 类
from .linear_fused import LinearBn1d
# 从当前目录下的 conv_fused 模块中导入以下类和函数：
# ConvBn1d, ConvBn2d, ConvBn3d, ConvBnReLU1d, ConvBnReLU2d, ConvBnReLU3d,
# ConvReLU1d, ConvReLU2d, ConvReLU3d, update_bn_stats, freeze_bn_stats
from .conv_fused import (
    ConvBn1d,
    ConvBn2d,
    ConvBn3d,
    ConvBnReLU1d,
    ConvBnReLU2d,
    ConvBnReLU3d,
    ConvReLU1d,
    ConvReLU2d,
    ConvReLU3d,
    update_bn_stats,
    freeze_bn_stats,
)

# __all__ 是一个列表，定义了模块中所有公开的对象（类、函数、变量等）
__all__ = [
    "LinearReLU",         # LinearReLU 类
    "LinearBn1d",         # LinearBn1d 类
    "ConvReLU1d",         # ConvReLU1d 类
    "ConvReLU2d",         # ConvReLU2d 类
    "ConvReLU3d",         # ConvReLU3d 类
    "ConvBn1d",           # ConvBn1d 类
    "ConvBn2d",           # ConvBn2d 类
    "ConvBn3d",           # ConvBn3d 类
    "ConvBnReLU1d",       # ConvBnReLU1d 类
    "ConvBnReLU2d",       # ConvBnReLU2d 类
    "ConvBnReLU3d",       # ConvBnReLU3d 类
    "update_bn_stats",    # update_bn_stats 函数
    "freeze_bn_stats",    # freeze_bn_stats 函数
]
```