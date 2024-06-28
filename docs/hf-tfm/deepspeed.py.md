# `.\deepspeed.py`

```
# 引入警告模块，用于向用户显示有关未来变更或不推荐使用的信息
import warnings

# 发出警告，提示用户 transformers.deepspeed 模块已被弃用，并且将在未来的版本中移除
warnings.warn(
    "transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations",
    FutureWarning,
)

# 导入用于向后兼容的模块，确保所有对象可以在 integrations/deepspeed 中找到
from .integrations.deepspeed import (  # noqa
    HfDeepSpeedConfig,                  # 导入 HfDeepSpeedConfig 类
    HfTrainerDeepSpeedConfig,           # 导入 HfTrainerDeepSpeedConfig 类
    deepspeed_config,                   # 导入 deepspeed_config 函数
    deepspeed_init,                     # 导入 deepspeed_init 函数
    deepspeed_load_checkpoint,          # 导入 deepspeed_load_checkpoint 函数
    deepspeed_optim_sched,              # 导入 deepspeed_optim_sched 函数
    is_deepspeed_available,             # 导入 is_deepspeed_available 函数
    is_deepspeed_zero3_enabled,         # 导入 is_deepspeed_zero3_enabled 函数
    set_hf_deepspeed_config,            # 导入 set_hf_deepspeed_config 函数
    unset_hf_deepspeed_config,          # 导入 unset_hf_deepspeed_config 函数
)
```