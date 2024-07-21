# `.\pytorch\torch\nn\intrinsic\qat\modules\conv_fused.py`

```
# flake8: noqa: F401
r"""Intrinsic QAT Modules.

This file is in the process of migration to `torch/ao/nn/intrinsic/qat`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/intrinsic/qat/modules`,
while adding an import statement here.
"""

# 定义模块导出的列表，包括各种 QAT 模块和工具函数
__all__ = [
    # Modules
    'ConvBn1d',          # 一维卷积+批归一化
    'ConvBnReLU1d',      # 一维卷积+批归一化+ReLU激活
    'ConvReLU1d',        # 一维卷积+ReLU激活
    'ConvBn2d',          # 二维卷积+批归一化
    'ConvBnReLU2d',      # 二维卷积+批归一化+ReLU激活
    'ConvReLU2d',        # 二维卷积+ReLU激活
    'ConvBn3d',          # 三维卷积+批归一化
    'ConvBnReLU3d',      # 三维卷积+批归一化+ReLU激活
    'ConvReLU3d',        # 三维卷积+ReLU激活
    # Utilities
    'freeze_bn_stats',   # 冻结批归一化统计信息
    'update_bn_stats',   # 更新批归一化统计信息
]

# 导入各模块和工具函数
from torch.ao.nn.intrinsic.qat import ConvBn1d
from torch.ao.nn.intrinsic.qat import ConvBnReLU1d
from torch.ao.nn.intrinsic.qat import ConvReLU1d
from torch.ao.nn.intrinsic.qat import ConvBn2d
from torch.ao.nn.intrinsic.qat import ConvBnReLU2d
from torch.ao.nn.intrinsic.qat import ConvReLU2d
from torch.ao.nn.intrinsic.qat import ConvBn3d
from torch.ao.nn.intrinsic.qat import ConvBnReLU3d
from torch.ao.nn.intrinsic.qat import ConvReLU3d
from torch.ao.nn.intrinsic.qat import freeze_bn_stats
from torch.ao.nn.intrinsic.qat import update_bn_stats
```