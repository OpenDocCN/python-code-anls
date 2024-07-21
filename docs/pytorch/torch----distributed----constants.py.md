# `.\pytorch\torch\distributed\constants.py`

```
# 从 datetime 模块导入 timedelta 类型
from datetime import timedelta
# 从 typing 模块导入 Optional 类型
from typing import Optional
# 从 torch._C._distributed_c10d 模块导入 _DEFAULT_PG_TIMEOUT 常量
from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT

# 定义公开接口列表，包含两个元素 default_pg_timeout 和 default_pg_nccl_timeout
__all__ = ["default_pg_timeout", "default_pg_nccl_timeout"]

# 默认的进程组超时时间，仅适用于非 nccl 后端
# 这是为了与 THD 兼容性尝试，使用一个非常高的默认超时时间，因为 THD 没有超时设置。
default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT

# 单独的 PGNCCL 超时时间，主要因为在 C++ 层面一直都这样，
# 但直到最近，Python 层面上所有后端都使用了同一个默认超时值。
# 以后，如果可以在 C++ 层面上达成一致，我们可以考虑将它们合并在一起。
# （仅当 TORCH_NCCL_BLOCKING_WAIT 或 TORCH_NCCL_ASYNC_ERROR_HANDLING 设置为 1 时有效）。
try:
    # 尝试从 torch._C._distributed_c10d 模块导入 _DEFAULT_PG_NCCL_TIMEOUT 常量
    from torch._C._distributed_c10d import _DEFAULT_PG_NCCL_TIMEOUT

    # 定义一个可选的 timedelta 类型的 default_pg_nccl_timeout 变量，赋值为 _DEFAULT_PG_NCCL_TIMEOUT
    default_pg_nccl_timeout: Optional[timedelta] = _DEFAULT_PG_NCCL_TIMEOUT
except ImportError:
    # 如果导入错误，则表示 C++ 没有编译支持 NCCL，此时默认的 nccl 超时值不可用。
    # 如果有人在这种状态下尝试使用 nccl，应该会出错。
    default_pg_nccl_timeout = None
```