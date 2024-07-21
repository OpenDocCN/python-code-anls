# `.\pytorch\torch\quantization\fx\pattern_utils.py`

```
# flake8: noqa: F401
"""
This file is in the process of migration to `torch/ao/quantization`, and
is kept here for compatibility while the migration process is ongoing.
If you are adding a new entry/functionality, please, add it to the
appropriate files under `torch/ao/quantization/fx/`, while adding an import statement
here.
"""

# 从torch.ao.quantization.fx.pattern_utils模块中导入以下函数和类
from torch.ao.quantization.fx.pattern_utils import (
    _register_fusion_pattern,
    _register_quant_pattern,
    get_default_fusion_patterns,
    get_default_output_activation_post_process_map,
    get_default_quant_patterns,
    QuantizeHandler,
)

# 设置QuantizeHandler的模块名为_NAMESPACE
QuantizeHandler.__module__ = _NAMESPACE

# 设置以下函数的模块名为"torch.ao.quantization.fx.pattern_utils"
_register_fusion_pattern.__module__ = "torch.ao.quantization.fx.pattern_utils"
get_default_fusion_patterns.__module__ = "torch.ao.quantization.fx.pattern_utils"
_register_quant_pattern.__module__ = "torch.ao.quantization.fx.pattern_utils"
get_default_quant_patterns.__module__ = "torch.ao.quantization.fx.pattern_utils"
get_default_output_activation_post_process_map.__module__ = (
    "torch.ao.quantization.fx.pattern_utils"
)

# 下面的__all__列表声明了在此模块中公开的所有名称
# 暂时被注释掉，如果需要公开以下名称，请取消注释
# __all__ = [
#     "QuantizeHandler",
#     "_register_fusion_pattern",
#     "get_default_fusion_patterns",
#     "_register_quant_pattern",
#     "get_default_quant_patterns",
#     "get_default_output_activation_post_process_map",
# ]
```