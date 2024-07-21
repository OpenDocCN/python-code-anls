# `.\pytorch\torch\nn\quantized\modules\dropout.py`

```py
# 禁止 flake8 检查导入模块时的 F401 错误，用于忽略未使用的导入警告
r"""Quantized Modules.

This file is in the process of migration to `torch/ao/nn/quantized`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/quantized/modules`,
while adding an import statement here.
"""
# 指定在导入时需要导出的符号列表
__all__ = ['Dropout']

# 从 torch/ao/nn/quantized/modules/dropout 模块导入 Dropout 类
from torch.ao.nn.quantized.modules.dropout import Dropout
```