# `.\pytorch\torch\nn\quantized\_reference\modules\utils.py`

```
# flake8: noqa: F401
r"""Quantized Reference Modules.

This module is in the process of migration to
`torch/ao/nn/quantized/reference`, and is kept here for
compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/reference`,
while adding an import statement here.
"""
# 从torch/ao/nn/quantized/reference/modules/utils导入_quantize_weight函数
from torch.ao.nn.quantized.reference.modules.utils import _quantize_weight
# 从torch/ao/nn/quantized/reference/modules/utils导入_quantize_and_dequantize_weight函数
from torch.ao.nn.quantized.reference.modules.utils import _quantize_and_dequantize_weight
# 从torch/ao/nn/quantized/reference/modules/utils导入_save_weight_qparams函数
from torch.ao.nn.quantized.reference.modules.utils import _save_weight_qparams
# 从torch/ao/nn/quantized/reference/modules/utils导入_get_weight_qparam_keys函数
from torch.ao.nn.quantized.reference.modules.utils import _get_weight_qparam_keys
# 从torch/ao/nn/quantized/reference/modules/utils导入ReferenceQuantizedModule类
from torch.ao.nn.quantized.reference.modules.utils import ReferenceQuantizedModule
```