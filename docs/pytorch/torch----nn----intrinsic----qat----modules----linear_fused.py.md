# `.\pytorch\torch\nn\intrinsic\qat\modules\linear_fused.py`

```
# flake8: noqa: F401
r"""Intrinsic QAT Modules.

This file is in the process of migration to `torch/ao/nn/intrinsic/qat`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/intrinsic/qat/modules`,
while adding an import statement here.
"""

# 定义 __all__ 列表，包含此模块中公开的所有符号名称
__all__ = [
    'LinearBn1d',  # 将 LinearBn1d 添加到 __all__ 列表中，表示它是公开的接口之一
]

# 从 torch.ao.nn.intrinsic.qat 模块导入 LinearBn1d 类
from torch.ao.nn.intrinsic.qat import LinearBn1d
```