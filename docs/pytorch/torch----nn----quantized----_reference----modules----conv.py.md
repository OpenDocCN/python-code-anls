# `.\pytorch\torch\nn\quantized\_reference\modules\conv.py`

```py
# flake8: noqa: F401
r"""Quantized Reference Modules.

This module is in the process of migration to
`torch/ao/nn/quantized/reference`, and is kept here for
compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/reference`,
while adding an import statement here.
"""

# 从 torch.ao.nn.quantized.reference.modules.conv 模块中导入以下类：
# - _ConvNd
# - Conv1d
# - Conv2d
# - Conv3d
# - _ConvTransposeNd
# - ConvTranspose1d
# - ConvTranspose2d
# - ConvTranspose3d
from torch.ao.nn.quantized.reference.modules.conv import _ConvNd
from torch.ao.nn.quantized.reference.modules.conv import Conv1d
from torch.ao.nn.quantized.reference.modules.conv import Conv2d
from torch.ao.nn.quantized.reference.modules.conv import Conv3d
from torch.ao.nn.quantized.reference.modules.conv import _ConvTransposeNd
from torch.ao.nn.quantized.reference.modules.conv import ConvTranspose1d
from torch.ao.nn.quantized.reference.modules.conv import ConvTranspose2d
from torch.ao.nn.quantized.reference.modules.conv import ConvTranspose3d
```