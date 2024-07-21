# `.\pytorch\torch\quantization\utils.py`

```py
# flake8: noqa: F401
"""
Utils shared by different modes of quantization (eager/graph)

This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/utils.py`, while adding an import statement
here.
"""

# 从 torch.ao.quantization.utils 中导入以下函数和变量，用于量化模型的实用工具函数
from torch.ao.quantization.utils import (
    activation_dtype,                        # 获取激活量化的数据类型
    activation_is_int8_quantized,            # 检查激活是否以 int8 进行量化
    activation_is_statically_quantized,      # 检查激活是否以静态方式进行量化
    calculate_qmin_qmax,                     # 计算量化的最小值和最大值
    check_min_max_valid,                     # 检查最小值和最大值是否有效
    get_combined_dict,                       # 获取组合字典
    get_qconfig_dtypes,                      # 获取量化配置的数据类型
    get_qparam_dict,                         # 获取量化参数的字典
    get_quant_type,                          # 获取量化类型
    get_swapped_custom_module_class,         # 获取交换的自定义模块类
    getattr_from_fqn,                        # 从完全限定名获取属性
    is_per_channel,                          # 检查是否按通道量化
    is_per_tensor,                           # 检查是否按张量量化
    weight_dtype,                            # 获取权重的数据类型
    weight_is_quantized,                     # 检查权重是否量化
    weight_is_statically_quantized,          # 检查权重是否以静态方式进行量化
)
```