# `.\pytorch\torch\quantization\fx\__init__.py`

```py
# flake8: noqa: F401
"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""

# 从 torch.ao.quantization.fx.convert 模块导入 convert 函数
from torch.ao.quantization.fx.convert import convert
# 从 torch.ao.quantization.fx.fuse 模块导入 fuse 函数
from torch.ao.quantization.fx.fuse import fuse

# 省略一些目前不太可能使用的文件，例如 newly added lower_to_fbgemm 等
# 从 torch.ao.quantization.fx.prepare 模块导入 prepare 函数
from torch.ao.quantization.fx.prepare import prepare
```