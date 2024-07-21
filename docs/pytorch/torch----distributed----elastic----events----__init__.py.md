# `.\pytorch\torch\distributed\elastic\events\__init__.py`

```py
#!/usr/bin/env/python3
"""
Module contains events processing mechanisms that are integrated with the standard python logging.

Example of usage:

::

  from torch.distributed.elastic import events
  event = events.Event(name="test_event", source=events.EventSource.WORKER, metadata={...})
  events.get_logging_handler(destination="console").info(event)

"""

import inspect  # 导入inspect模块，用于获取有关活动对象的信息
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，用于操作系统相关功能
import socket  # 导入socket模块，用于网络通信
import traceback  # 导入traceback模块，用于提取和格式化异常信息
from typing import Dict, Optional  # 导入类型提示模块，用于静态类型检查

from torch.distributed.elastic.events.handlers import get_logging_handler  # 从torch.distributed.elastic.events.handlers导入get_logging_handler函数

from .api import (  # 从当前目录下的api模块导入以下标识符
    Event,
    EventMetadataValue,
    EventSource,
    NodeState,
    RdzvEvent,
)

_events_loggers: Dict[str, logging.Logger] = {}  # 初始化一个字典用于存储日志记录器对象


def _get_or_create_logger(destination: str = "null") -> logging.Logger:
    """
    Construct python logger based on the destination type or extends if provided.

    Available destination could be found in ``handlers.py`` file.
    The constructed logger does not propagate messages to the upper level loggers,
    e.g. root logger. This makes sure that a single event can be processed once.

    Args:
        destination: The string representation of the event handler.
            Available handlers found in ``handlers`` module
    """
    global _events_loggers

    if destination not in _events_loggers:
        _events_logger = logging.getLogger(f"torchelastic-events-{destination}")
        _events_logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))  # 设置日志记录器的日志级别，默认为INFO
        # Do not propagate message to the root logger
        _events_logger.propagate = False  # 设置日志记录器不向上层日志记录器传递消息

        logging_handler = get_logging_handler(destination)  # 获取指定目的地的日志处理程序
        _events_logger.addHandler(logging_handler)  # 将日志处理程序添加到日志记录器中

        # Add the logger to the global dictionary
        _events_loggers[destination] = _events_logger  # 将日志记录器对象存储到全局字典中

    return _events_loggers[destination]


def record(event: Event, destination: str = "null") -> None:
    """
    Records the event into the specified destination logger.

    Args:
        event: The event object to record.
        destination: The string representation of the destination logger.
    """
    _get_or_create_logger(destination).info(event.serialize())  # 使用指定的日志记录器记录事件的序列化信息


def record_rdzv_event(event: RdzvEvent) -> None:
    """
    Records the rendezvous event into the "dynamic_rendezvous" logger.

    Args:
        event: The RdzvEvent object to record.
    """
    _get_or_create_logger("dynamic_rendezvous").info(event.serialize())  # 使用"dynamic_rendezvous"日志记录器记录RdzvEvent对象的序列化信息


def construct_and_record_rdzv_event(
    run_id: str,
    message: str,
    node_state: NodeState,
    name: str = "",
    hostname: str = "",
    pid: Optional[int] = None,
    master_endpoint: str = "",
    local_id: Optional[int] = None,
    rank: Optional[int] = None,
) -> None:
    """
    Initialize rendezvous event object and record its operations.

    Args:
        run_id: The ID of the current run.
        message: The message to record with the event.
        node_state: The state of the node.
        name: Optional name of the event.
        hostname: Optional hostname of the node.
        pid: Optional process ID of the node.
        master_endpoint: Optional endpoint of the master node.
        local_id: Optional local ID of the node.
        rank: Optional rank of the node in distributed setup.
    """
    # 初始化RdzvEvent对象
    event = RdzvEvent(
        run_id=run_id,
        message=message,
        node_state=node_state,
        name=name,
        hostname=hostname,
        pid=pid,
        master_endpoint=master_endpoint,
        local_id=local_id,
        rank=rank,
    )
    _get_or_create_logger("dynamic_rendezvous").info(event.serialize())  # 使用"dynamic_rendezvous"日志记录器记录RdzvEvent对象的序列化信息
    # 如果日志处理器是 logging.NullHandler，表示不需要执行额外的计算，直接返回
    if isinstance(get_logging_handler("dynamic_rendezvous"), logging.NullHandler):
        return

    # 设置参数的默认值
    if not hostname:
        # 如果未提供主机名，获取当前主机的全限定域名
        hostname = socket.getfqdn()
    if not pid:
        # 如果未提供进程ID，获取当前进程的ID
        pid = os.getpid()

    # 确定调用此函数的文件名
    callstack = inspect.stack()
    filename = "no_file"
    if len(callstack) > 1:
        stack_depth_1 = callstack[1]
        # 获取调用栈的文件名（不含路径）
        filename = os.path.basename(stack_depth_1.filename)
        if not name:
            # 如果未提供事件名，使用调用栈的函数名作为事件名
            name = stack_depth_1.function

    # 删除调用栈变量，以避免影响 Python 的垃圾回收
    del callstack

    # 如果节点状态为 FAILED，设置错误跟踪信息
    if node_state == NodeState.FAILED:
        # 获取异常的堆栈信息
        error_trace = traceback.format_exc()
    else:
        error_trace = ""

    # 初始化事件对象
    event = RdzvEvent(
        name=f"{filename}:{name}",
        run_id=run_id,
        message=message,
        hostname=hostname,
        pid=pid,
        node_state=node_state,
        master_endpoint=master_endpoint,
        rank=rank,
        local_id=local_id,
        error_trace=error_trace,
    )

    # 最后，记录事件
    record_rdzv_event(event)
```