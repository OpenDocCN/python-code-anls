# `.\pytorch\torch\quantization\fx\_equalize.py`

```
# 导入模块和函数，用于 AO（Activation Observation）量化的均衡处理
"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""
from torch.ao.quantization.fx._equalize import (
    _convert_equalization_ref,                  # 导入函数_convert_equalization_ref
    _InputEqualizationObserver,                 # 导入类_InputEqualizationObserver
    _WeightEqualizationObserver,                # 导入类_WeightEqualizationObserver
    calculate_equalization_scale,               # 导入函数calculate_equalization_scale
    clear_weight_quant_obs_node,                # 导入函数clear_weight_quant_obs_node
    convert_eq_obs,                             # 导入函数convert_eq_obs
    CUSTOM_MODULE_SUPP_LIST,                    # 导入变量CUSTOM_MODULE_SUPP_LIST
    custom_module_supports_equalization,        # 导入函数custom_module_supports_equalization
    default_equalization_qconfig,               # 导入函数default_equalization_qconfig
    EqualizationQConfig,                        # 导入类EqualizationQConfig
    fused_module_supports_equalization,         # 导入函数fused_module_supports_equalization
    get_equalization_qconfig_dict,              # 导入函数get_equalization_qconfig_dict
    get_layer_sqnr_dict,                        # 导入函数get_layer_sqnr_dict
    get_op_node_and_weight_eq_obs,              # 导入函数get_op_node_and_weight_eq_obs
    input_equalization_observer,                # 导入函数input_equalization_observer
    is_equalization_observer,                   # 导入函数is_equalization_observer
    maybe_get_next_equalization_scale,          # 导入函数maybe_get_next_equalization_scale
    maybe_get_next_input_eq_obs,                # 导入函数maybe_get_next_input_eq_obs
    maybe_get_weight_eq_obs_node,               # 导入函数maybe_get_weight_eq_obs_node
    nn_module_supports_equalization,            # 导入函数nn_module_supports_equalization
    node_supports_equalization,                 # 导入函数node_supports_equalization
    remove_node,                                # 导入函数remove_node
    reshape_scale,                              # 导入函数reshape_scale
    scale_input_observer,                       # 导入函数scale_input_observer
    scale_weight_functional,                    # 导入函数scale_weight_functional
    scale_weight_node,                          # 导入函数scale_weight_node
    update_obs_for_equalization,                # 导入函数update_obs_for_equalization
    weight_equalization_observer,               # 导入函数weight_equalization_observer
)
```