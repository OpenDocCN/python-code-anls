# `.\pytorch\torch\quantization\quantization_mappings.py`

```
# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quantization_mappings.py`, while adding an import statement
here.
"""

# 从 torch/ao/quantization/quantization_mappings.py 导入以下函数和变量，用于量化映射和配置
from torch.ao.quantization.quantization_mappings import (
    _get_special_act_post_process,                # 导入特殊激活后处理函数
    _has_special_act_post_process,                # 导入检查是否有特殊激活后处理函数的函数
    _INCLUDE_QCONFIG_PROPAGATE_LIST,              # 导入包含量化配置传播列表的变量
    DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,        # 导入默认动态量化模块映射
    DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS, # 导入默认浮点到量化操作符映射
    DEFAULT_MODULE_TO_ACT_POST_PROCESS,           # 导入默认模块到激活后处理映射
    DEFAULT_QAT_MODULE_MAPPINGS,                  # 导入默认量化训练模块映射
    DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS,# 导入默认参考静态量化模块映射
    DEFAULT_STATIC_QUANT_MODULE_MAPPINGS,         # 导入默认静态量化模块映射
    get_default_compare_output_module_list,       # 导入获取默认比较输出模块列表的函数
    get_default_dynamic_quant_module_mappings,    # 导入获取默认动态量化模块映射的函数
    get_default_float_to_quantized_operator_mappings, # 导入获取默认浮点到量化操作符映射的函数
    get_default_qat_module_mappings,              # 导入获取默认量化训练模块映射的函数
    get_default_qconfig_propagation_list,         # 导入获取默认量化配置传播列表的函数
    get_default_static_quant_module_mappings,     # 导入获取默认静态量化模块映射的函数
    get_dynamic_quant_module_class,               # 导入获取动态量化模块类的函数
    get_quantized_operator,                       # 导入获取量化操作符的函数
    get_static_quant_module_class,                # 导入获取静态量化模块类的函数
    no_observer_set,                              # 导入未设置观察器的常量
)
```