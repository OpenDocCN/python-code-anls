# `.\pytorch\torch\nn\quantized\modules\conv.py`

```py
# flake8: noqa: F401
# 禁止 flake8 检查 F401 错误，即未使用的导入模块
r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""
# 文档字符串，说明此文件是量化模块的一部分，正在迁移至 `torch/ao/nn/quantized` 目录，当前仍在此处保留以保证兼容性

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']
# 导出的模块列表，包含此文件中导出的类名

from torch.ao.nn.quantized.modules.conv import _reverse_repeat_padding
# 导入 _reverse_repeat_padding 函数，用于卷积模块的反向重复填充

from torch.ao.nn.quantized.modules.conv import Conv1d
from torch.ao.nn.quantized.modules.conv import Conv2d
from torch.ao.nn.quantized.modules.conv import Conv3d
# 导入量化卷积模块的不同维度的类 Conv1d, Conv2d, Conv3d

from torch.ao.nn.quantized.modules.conv import ConvTranspose1d
from torch.ao.nn.quantized.modules.conv import ConvTranspose2d
from torch.ao.nn.quantized.modules.conv import ConvTranspose3d
# 导入量化转置卷积模块的不同维度的类 ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
```