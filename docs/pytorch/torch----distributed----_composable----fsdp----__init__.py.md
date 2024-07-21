# `.\pytorch\torch\distributed\_composable\fsdp\__init__.py`

```
# 从._fsdp_api模块中导入CPUOffloadPolicy、MixedPrecisionPolicy和OffloadPolicy类
# 从fully_shard模块中导入FSDPModule类、fully_shard函数和register_fsdp_forward_method函数
from ._fsdp_api import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy
from .fully_shard import FSDPModule, fully_shard, register_fsdp_forward_method
```