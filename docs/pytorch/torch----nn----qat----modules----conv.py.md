# `.\pytorch\torch\nn\qat\modules\conv.py`

```py
# flake8: noqa: F401
r"""QAT Modules.

This file is in the process of migration to `torch/ao/nn/qat`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate file under the `torch/ao/nn/qat/modules`,
while adding an import statement here.
"""

# 导入 QAT 模块的卷积层定义：Conv1d, Conv2d, Conv3d
from torch.ao.nn.qat.modules.conv import Conv1d
from torch.ao.nn.qat.modules.conv import Conv2d
from torch.ao.nn.qat.modules.conv import Conv3d
```