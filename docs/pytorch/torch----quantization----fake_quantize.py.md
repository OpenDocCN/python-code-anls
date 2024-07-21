# `.\pytorch\torch\quantization\fake_quantize.py`

```
# flake8: noqa: F401
"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/fake_quantize.py`, while adding an import statement
here.
"""

# 从 torch.ao.quantization.fake_quantize 模块中导入以下函数和类
from torch.ao.quantization.fake_quantize import (
    _is_fake_quant_script_module,                        # 导入 _is_fake_quant_script_module 函数
    _is_per_channel,                                    # 导入 _is_per_channel 函数
    _is_per_tensor,                                     # 导入 _is_per_tensor 函数
    _is_symmetric_quant,                                # 导入 _is_symmetric_quant 函数
    default_fake_quant,                                 # 导入 default_fake_quant 函数
    default_fixed_qparams_range_0to1_fake_quant,        # 导入 default_fixed_qparams_range_0to1_fake_quant 函数
    default_fixed_qparams_range_neg1to1_fake_quant,     # 导入 default_fixed_qparams_range_neg1to1_fake_quant 函数
    default_fused_act_fake_quant,                       # 导入 default_fused_act_fake_quant 函数
    default_fused_per_channel_wt_fake_quant,            # 导入 default_fused_per_channel_wt_fake_quant 函数
    default_fused_wt_fake_quant,                        # 导入 default_fused_wt_fake_quant 函数
    default_histogram_fake_quant,                       # 导入 default_histogram_fake_quant 函数
    default_per_channel_weight_fake_quant,              # 导入 default_per_channel_weight_fake_quant 函数
    default_weight_fake_quant,                          # 导入 default_weight_fake_quant 函数
    disable_fake_quant,                                 # 导入 disable_fake_quant 函数
    disable_observer,                                   # 导入 disable_observer 函数
    enable_fake_quant,                                  # 导入 enable_fake_quant 函数
    enable_observer,                                    # 导入 enable_observer 函数
    FakeQuantize,                                       # 导入 FakeQuantize 类
    FakeQuantizeBase,                                   # 导入 FakeQuantizeBase 类
    FixedQParamsFakeQuantize,                           # 导入 FixedQParamsFakeQuantize 类
    FusedMovingAvgObsFakeQuantize,                      # 导入 FusedMovingAvgObsFakeQuantize 类
)
```