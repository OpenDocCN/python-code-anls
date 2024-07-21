# `.\pytorch\torch\distributed\fsdp\_limiter_utils.py`

```
import collections  # 导入collections模块，用于创建双端队列
from typing import Deque, Optional  # 导入类型提示，用于声明Deque和Optional类型

import torch  # 导入torch模块，假设在此处需要使用torch.cuda.Event

class _FreeEventQueue:
    """
    This tracks all pending frees corresponding to inflight all-gathers. The
    queueing pattern is iterative enqueues with a single dequeue per iteration
    once the limit ``_max_num_inflight_all_gathers`` is reached.
    """

    def __init__(self) -> None:
        self._queue: Deque[torch.cuda.Event] = collections.deque()  # 初始化一个空的双端队列，用于存储torch.cuda.Event对象
        self._max_num_inflight_all_gathers = 2  # empirically chosen，设置最大的同时进行的all-gather操作数量

    def enqueue(self, free_event: torch.cuda.Event) -> None:
        """Enqueues a free event."""
        self._queue.append(free_event)  # 将给定的torch.cuda.Event对象添加到队列尾部

    def dequeue_if_needed(self) -> Optional[torch.cuda.Event]:
        """Dequeues a single event if the limit is reached."""
        if len(self._queue) >= self._max_num_inflight_all_gathers:
            return self._dequeue()  # 如果队列长度达到限制，执行出队操作
        return None  # 否则返回None表示未执行出队操作

    def _dequeue(self) -> Optional[torch.cuda.Event]:
        """Dequeues a free event if possible."""
        if self._queue:
            event = self._queue.popleft()  # 从队列左侧取出一个torch.cuda.Event对象
            return event  # 返回取出的事件对象
        return None  # 如果队列为空，返回None表示未取出事件
```