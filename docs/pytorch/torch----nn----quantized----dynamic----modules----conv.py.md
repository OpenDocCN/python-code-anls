# `.\pytorch\torch\nn\quantized\dynamic\modules\conv.py`

```py
# flake8: noqa: F401
r"""Quantized Dynamic Modules.

This file is in the process of migration to `torch/ao/nn/quantized/dynamic`,
and is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/dynamic/modules`,
while adding an import statement here.
"""

__all__ = ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']
# 定义该模块中公开的符号列表，用于模块导入时的限定符

from torch.ao.nn.quantized.dynamic.modules.conv import Conv1d
# 从动态量化卷积模块中导入 Conv1d 类
from torch.ao.nn.quantized.dynamic.modules.conv import Conv2d
# 从动态量化卷积模块中导入 Conv2d 类
from torch.ao.nn.quantized.dynamic.modules.conv import Conv3d
# 从动态量化卷积模块中导入 Conv3d 类
from torch.ao.nn.quantized.dynamic.modules.conv import ConvTranspose1d
# 从动态量化卷积模块中导入 ConvTranspose1d 类
from torch.ao.nn.quantized.dynamic.modules.conv import ConvTranspose2d
# 从动态量化卷积模块中导入 ConvTranspose2d 类
from torch.ao.nn.quantized.dynamic.modules.conv import ConvTranspose3d
# 从动态量化卷积模块中导入 ConvTranspose3d 类
```