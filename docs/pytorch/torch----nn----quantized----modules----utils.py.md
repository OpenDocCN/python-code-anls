# `.\pytorch\torch\nn\quantized\modules\utils.py`

```
# flake8: noqa: F401
r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""

# 从 torch.ao.nn.quantized.modules.utils 导入 _ntuple_from_first 函数
from torch.ao.nn.quantized.modules.utils import _ntuple_from_first
# 从 torch.ao.nn.quantized.modules.utils 导入 _pair_from_first 函数
from torch.ao.nn.quantized.modules.utils import _pair_from_first
# 从 torch.ao.nn.quantized.modules.utils 导入 _quantize_weight 函数
from torch.ao.nn.quantized.modules.utils import _quantize_weight
# 从 torch.ao.nn.quantized.modules.utils 导入 _hide_packed_params_repr 函数
from torch.ao.nn.quantized.modules.utils import _hide_packed_params_repr
# 从 torch.ao.nn.quantized.modules.utils 导入 WeightedQuantizedModule 类
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule
```