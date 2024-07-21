# `.\pytorch\torch\quantization\quant_type.py`

```
# 引入禁用了 flake8 检查的标志，以避免 F401 错误（未使用导入）
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quant_type.py`, while adding an import statement
here.
"""

# 从 torch/ao/quantization/quant_type.py 导入 _get_quant_type_to_str 和 QuantType 类
from torch.ao.quantization.quant_type import _get_quant_type_to_str, QuantType
```