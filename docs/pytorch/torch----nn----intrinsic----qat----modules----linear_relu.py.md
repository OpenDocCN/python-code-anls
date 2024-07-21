# `.\pytorch\torch\nn\intrinsic\qat\modules\linear_relu.py`

```
# flake8: noqa: F401
r"""Intrinsic QAT Modules.

This file is in the process of migration to `torch/ao/nn/intrinsic/qat`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/intrinsic/qat/modules`,
while adding an import statement here.
"""

# 定义 __all__ 列表，包含了当前模块公开的所有接口
__all__ = [
    'LinearReLU',
]

# 从 torch.ao.nn.intrinsic.qat 模块中导入 LinearReLU 类
from torch.ao.nn.intrinsic.qat import LinearReLU
```