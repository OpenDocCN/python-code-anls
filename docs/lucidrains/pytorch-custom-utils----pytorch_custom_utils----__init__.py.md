# `.\lucidrains\pytorch-custom-utils\pytorch_custom_utils\__init__.py`

```py
# 从 pytorch_custom_utils.module_device 模块中导入 module_device 和 autocast_device 函数
from pytorch_custom_utils.module_device import (
    module_device,
    autocast_device
)

# 从 pytorch_custom_utils.save_load 模块中导入 save_load 函数
from pytorch_custom_utils.save_load import save_load

# 从 pytorch_custom_utils.total_parameters 模块中导入 total_parameters 函数
from pytorch_custom_utils.total_parameters import total_parameters

# 从 pytorch_custom_utils.get_adam_optimizer 模块中导入 get_adam_optimizer 函数
from pytorch_custom_utils.get_adam_optimizer import get_adam_optimizer

# 从 pytorch_custom_utils.optimizer_scheduler_warmup 模块中导入 OptimizerWithWarmupSchedule 类
from pytorch_custom_utils.optimizer_scheduler_warmup import OptimizerWithWarmupSchedule

# 从 pytorch_custom_utils.accelerate_utils 模块中导入 add_wandb_tracker_contextmanager 和 auto_unwrap_model 函数
from pytorch_custom_utils.accelerate_utils import (
    add_wandb_tracker_contextmanager,
    auto_unwrap_model
)
```