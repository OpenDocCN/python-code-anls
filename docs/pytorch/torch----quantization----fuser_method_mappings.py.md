# `.\pytorch\torch\quantization\fuser_method_mappings.py`

```py
# flake8: noqa: F401
"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/fuser_method_mappings.py`, while adding an import statement
here.
"""
# 从 torch/ao/quantization/fuser_method_mappings.py 中导入所需的函数和变量
from torch.ao.quantization.fuser_method_mappings import (
    _DEFAULT_OP_LIST_TO_FUSER_METHOD,  # 导入默认操作列表到融合方法的映射
    fuse_conv_bn,                     # 导入融合卷积和批归一化的函数
    fuse_conv_bn_relu,                # 导入融合卷积、批归一化和ReLU激活的函数
    fuse_linear_bn,                   # 导入融合线性层和批归一化的函数
    get_fuser_method,                 # 导入获取融合方法的函数
)
```