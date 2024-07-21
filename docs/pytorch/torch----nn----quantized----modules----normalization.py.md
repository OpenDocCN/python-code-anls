# `.\pytorch\torch\nn\quantized\modules\normalization.py`

```
# flake8: noqa: F401
# 禁用 flake8 对 F401 错误的检查，表示未使用的导入

r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""
# 文档字符串，说明此文件包含量化模块的导入声明，并指出其正在迁移的过程中保持兼容性

__all__ = ['LayerNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d']
# 定义了导出的所有公共接口，包括 LayerNorm、GroupNorm 和 InstanceNorm1d 到 InstanceNorm3d

# 以下导入语句从 quantized.modules.normalization 子模块中导入不同的规范化模块
from torch.ao.nn.quantized.modules.normalization import LayerNorm
from torch.ao.nn.quantized.modules.normalization import GroupNorm
from torch.ao.nn.quantized.modules.normalization import InstanceNorm1d
from torch.ao.nn.quantized.modules.normalization import InstanceNorm2d
from torch.ao.nn.quantized.modules.normalization import InstanceNorm3d
```