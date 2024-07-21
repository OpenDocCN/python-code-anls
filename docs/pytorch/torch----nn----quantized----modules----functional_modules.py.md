# `.\pytorch\torch\nn\quantized\modules\functional_modules.py`

```py
# flake8: noqa: F401
r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""

# 定义导出的模块列表，只包括以下三个模块
__all__ = ['FloatFunctional', 'FXFloatFunctional', 'QFunctional']

# 从功能模块中导入以下三个类
from torch.ao.nn.quantized.modules.functional_modules import FloatFunctional
from torch.ao.nn.quantized.modules.functional_modules import FXFloatFunctional
from torch.ao.nn.quantized.modules.functional_modules import QFunctional
```