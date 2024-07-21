# `.\pytorch\torch\nn\quantized\modules\linear.py`

```
# flake8: noqa: F401
r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""

# 导出的模块列表，仅包括 'LinearPackedParams' 和 'Linear'
__all__ = ['LinearPackedParams', 'Linear']

# 导入 quantized 模块中的 Linear 和 LinearPackedParams 类
from torch.ao.nn.quantized.modules.linear import Linear
from torch.ao.nn.quantized.modules.linear import LinearPackedParams
```