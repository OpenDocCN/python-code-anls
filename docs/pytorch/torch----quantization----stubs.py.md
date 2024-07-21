# `.\pytorch\torch\quantization\stubs.py`

```py
# 禁用 flake8 对 F401 错误的检查，允许未使用的导入存在
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/stubs.py`, while adding an import statement
here.
"""

# 从 torch/ao/quantization/stubs.py 导入 DeQuantStub、QuantStub 和 QuantWrapper 类
from torch.ao.quantization.stubs import DeQuantStub, QuantStub, QuantWrapper
```