# `.\pytorch\torch\ao\nn\intrinsic\qat\modules\__init__.py`

```
# 导入自定义模块中的特定类和函数，包括 LinearReLU 类
# 这些模块在当前目录的 linear_relu.py 文件中定义
from .linear_relu import LinearReLU

# 导入自定义模块中的特定类，包括 LinearBn1d 类
# 这些模块在当前目录的 linear_fused.py 文件中定义
from .linear_fused import LinearBn1d

# 导入自定义模块中的特定类，包括以下几种类型的类
# 这些模块在当前目录的 conv_fused.py 文件中定义
from .conv_fused import (
    ConvBn1d,        # 一维卷积层与批归一化结合并带有ReLU激活函数
    ConvBn2d,        # 二维卷积层与批归一化结合并带有ReLU激活函数
    ConvBn3d,        # 三维卷积层与批归一化结合并带有ReLU激活函数
    ConvBnReLU1d,    # 一维卷积层与批归一化结合并带有ReLU激活函数
    ConvBnReLU2d,    # 二维卷积层与批归一化结合并带有ReLU激活函数
    ConvBnReLU3d,    # 三维卷积层与批归一化结合并带有ReLU激活函数
    ConvReLU1d,      # 一维卷积层并带有ReLU激活函数
    ConvReLU2d,      # 二维卷积层并带有ReLU激活函数
    ConvReLU3d,      # 三维卷积层并带有ReLU激活函数
    update_bn_stats, # 更新批归一化统计数据的函数
    freeze_bn_stats, # 冻结批归一化统计数据的函数
)

# 将这些类和函数加入到 __all__ 列表中，使它们能够被 from module import * 导入
__all__ = [
    "LinearReLU",
    "LinearBn1d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "ConvBn1d",
    "ConvBn2d",
    "ConvBn3d",
    "ConvBnReLU1d",
    "ConvBnReLU2d",
    "ConvBnReLU3d",
    "update_bn_stats",
    "freeze_bn_stats",
]
```