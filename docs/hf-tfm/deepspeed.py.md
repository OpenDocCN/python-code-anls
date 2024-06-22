# `.\transformers\deepspeed.py`

```py
# 导入警告模块，用于发出警告信息
import warnings

# 发出关于 transformers.deepspeed 模块即将被移除的警告
warnings.warn(
    "transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations",
    FutureWarning,
)

# 导入 integrations/deepspeed 中的对象，以确保向后兼容性
from .integrations.deepspeed import (  # noqa
    HfDeepSpeedConfig,  # 导入 HfDeepSpeedConfig 类
    HfTrainerDeepSpeedConfig,  # 导入 HfTrainerDeepSpeedConfig 类
    deepspeed_config,  # 导入 deepspeed_config 函数
    deepspeed_init,  # 导入 deepspeed_init 函数
    deepspeed_load_checkpoint,  # 导入 deepspeed_load_checkpoint 函数
    deepspeed_optim_sched,  # 导入 deepspeed_optim_sched 函数
    is_deepspeed_available,  # 导入 is_deepspeed_available 函数
    is_deepspeed_zero3_enabled,  # 导入 is_deepspeed_zero3_enabled 函数
    set_hf_deepspeed_config,  # 导入 set_hf_deepspeed_config 函数
    unset_hf_deepspeed_config,  # 导入 unset_hf_deepspeed_config 函数
)
```