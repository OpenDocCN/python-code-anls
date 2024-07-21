# `.\pytorch\torch\utils\tensorboard\__init__.py`

```py
# 导入 `tensorboard` 模块，用于与 TensorBoard 相关的功能
import tensorboard

# 从 `torch._vendor.packaging.version` 模块中导入 `Version` 类，用于版本比较
from torch._vendor.packaging.version import Version

# 检查 `tensorboard` 模块是否具有 `__version__` 属性，或其版本是否低于 1.15
if not hasattr(tensorboard, "__version__") or Version(
    tensorboard.__version__
) < Version("1.15"):
    # 如果条件不满足，抛出 ImportError 异常，要求至少安装 TensorBoard 版本 1.15
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

# 清理环境：删除引入的 `Version` 类和 `tensorboard` 模块，避免全局污染和内存占用
del Version
del tensorboard

# 从当前包中导入 `FileWriter` 和 `SummaryWriter` 类，忽略未使用的 `F401` 警告
from .writer import FileWriter, SummaryWriter  # noqa: F401

# 从 `tensorboard.summary.writer.record_writer` 模块导入 `RecordWriter` 类，忽略未使用的 `F401` 警告
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: F401
```