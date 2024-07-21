# `.\pytorch\torch\quantization\fuse_modules.py`

```
# flake8: noqa: F401
"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/fuse_modules.py`, while adding an import statement
here.
"""

# TODO: These functions are not used outside the `fuse_modules.py`
#       Keeping here for now, need to remove them later.

# 从torch/ao/quantization/fuse_modules.py导入以下函数，用于模块融合
from torch.ao.quantization.fuse_modules import (
    _fuse_modules,       # 导入模块融合函数
    _get_module,         # 导入获取模块函数
    _set_module,         # 导入设置模块函数
    fuse_known_modules,  # 导入已知模块融合函数
    fuse_modules,        # 导入模块融合函数
    get_fuser_method,    # 导入获取融合方法函数
)

# 为了向后兼容性而导入
from torch.ao.quantization.fuser_method_mappings import fuse_conv_bn, fuse_conv_bn_relu
```