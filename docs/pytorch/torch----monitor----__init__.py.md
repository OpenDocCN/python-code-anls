# `.\pytorch\torch\monitor\__init__.py`

```py
# 从 torch._C._monitor 模块导入所有内容，禁止 pylint 报错 F403
from torch._C._monitor import *  # noqa: F403

# 导入 TYPE_CHECKING 类型提示
from typing import TYPE_CHECKING

# 如果 TYPE_CHECKING 为真，则导入 SummaryWriter 类型提示
if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

# 定义常量 STAT_EVENT，表示监控事件为 "torch.monitor.Stat"
STAT_EVENT = "torch.monitor.Stat"

class TensorboardEventHandler:
    """
    TensorboardEventHandler 是一个事件处理器，将已知事件写入提供的 SummaryWriter 中。

    目前仅支持 ``torch.monitor.Stat`` 事件，这些事件被记录为标量。

    示例:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_MONITOR)
        >>> # xdoctest: +REQUIRES(module:tensorboard)
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> from torch.monitor import TensorboardEventHandler, register_event_handler
        >>> writer = SummaryWriter("log_dir")
        >>> register_event_handler(TensorboardEventHandler(writer))
    """
    
    def __init__(self, writer: "SummaryWriter") -> None:
        """
        初始化 ``TensorboardEventHandler``。

        参数:
            writer: 用于写入事件的 SummaryWriter 对象。
        """
        self._writer = writer

    def __call__(self, event: Event) -> None:
        """
        将事件写入 SummaryWriter。

        如果事件名称为 STAT_EVENT，则遍历事件数据项，将其作为标量写入 SummaryWriter。

        参数:
            event: 要处理的事件对象。
        """
        if event.name == STAT_EVENT:
            for k, v in event.data.items():
                self._writer.add_scalar(k, v, walltime=event.timestamp.timestamp())
```