# `.\pytorch\torch\quantization\quantize.py`

```
# flake8: noqa: F401
r"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
`torch/ao/quantization/quantize.py`, while adding an import statement
here.
"""

# 从 torch.ao.quantization.quantize 模块导入以下函数和类，用于量化和量化相关操作
from torch.ao.quantization.quantize import (
    _add_observer_,  # 添加观察器函数
    _convert,  # 转换函数
    _get_observer_dict,  # 获取观察器字典函数
    _get_unique_devices_,  # 获取唯一设备函数
    _is_activation_post_process,  # 判断是否为激活后处理函数
    _observer_forward_hook,  # 观察器前向钩子函数
    _propagate_qconfig_helper,  # 传播量化配置助手函数
    _register_activation_post_process_hook,  # 注册激活后处理钩子函数
    _remove_activation_post_process,  # 移除激活后处理函数
    _remove_qconfig,  # 移除量化配置函数
    add_quant_dequant,  # 添加量化-反量化函数
    convert,  # 转换函数（与上面的 _convert 可能有重名）
    prepare,  # 准备函数
    prepare_qat,  # 准备量化感知训练函数
    propagate_qconfig_,  # 传播量化配置函数
    quantize,  # 量化函数
    quantize_dynamic,  # 动态量化函数
    quantize_qat,  # 量化感知训练函数
    swap_module,  # 模块交换函数
)
```