# `.\pytorch\torch\distributed\rpc\constants.py`

```py
# 导入需要的模块和类型
from datetime import timedelta
from typing import List

# 从 torch._C._distributed_rpc 模块导入以下常量

# 默认的 RPC 调用超时时间（秒）
DEFAULT_RPC_TIMEOUT_SEC: float = _DEFAULT_RPC_TIMEOUT_SEC
# 默认的初始化方法
DEFAULT_INIT_METHOD: str = _DEFAULT_INIT_METHOD
# 默认的关闭超时时间设为 0 秒
DEFAULT_SHUTDOWN_TIMEOUT: float = 0

# 对于 TensorPipeAgent

# 默认的工作线程数
DEFAULT_NUM_WORKER_THREADS: int = _DEFAULT_NUM_WORKER_THREADS

# 确保在底层进程组没有任何操作时不会超时的时间段
DEFAULT_PROCESS_GROUP_TIMEOUT: timedelta = timedelta(milliseconds=2**31 - 1)

# 表示 RPC 调用没有设置超时，应使用默认值
UNSET_RPC_TIMEOUT: float = _UNSET_RPC_TIMEOUT

# 声明 __all__ 列表为空，表示没有导出任何符号
__all__: List[str] = []
```