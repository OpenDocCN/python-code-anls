# `.\pytorch\torch\distributed\c10d_logger.py`

```py
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 导入必要的库
import functools  # 导入 functools 库，用于装饰器
import logging  # 导入 logging 库，用于日志记录
import time  # 导入 time 库，用于时间相关操作
from typing import Any, Callable, Dict, List, Tuple, TypeVar  # 导入类型提示相关的模块
from typing_extensions import ParamSpec  # 导入 ParamSpec 类型提示扩展

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入 PyTorch 分布式训练的库
from torch.distributed.logging_handlers import _log_handlers  # 导入分布式日志处理器


__all__: List[str] = []  # 初始化一个空列表，用于定义模块的公开接口

_DEFAULT_DESTINATION = "default"  # 定义默认日志目的地为 "default"


def _get_or_create_logger(destination: str = _DEFAULT_DESTINATION) -> logging.Logger:
    # 根据指定的目的地获取或创建一个 logger 对象
    logging_handler, log_handler_name = _get_logging_handler(destination)
    logger = logging.getLogger(f"c10d-{log_handler_name}")
    logger.setLevel(logging.DEBUG)  # 设置日志级别为 DEBUG
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)  # 设置日志格式
    logger.propagate = False  # 禁止 logger 传播日志
    logger.addHandler(logging_handler)  # 添加日志处理器到 logger
    return logger  # 返回创建的 logger 对象


def _get_logging_handler(
    destination: str = _DEFAULT_DESTINATION,
) -> Tuple[logging.Handler, str]:
    # 根据指定的日志目的地获取相应的日志处理器和处理器名称
    log_handler = _log_handlers[destination]
    log_handler_name = type(log_handler).__name__
    return (log_handler, log_handler_name)  # 返回日志处理器和处理器名称的元组


global _c10d_logger  # 声明全局变量 _c10d_logger
_c10d_logger = _get_or_create_logger()  # 初始化全局 logger 对象


def _get_msg_dict(func_name, *args, **kwargs) -> Dict[str, Any]:
    # 根据函数名称、参数和关键字参数生成消息字典
    if dist.is_initialized():  # 如果分布式已初始化
        group = kwargs.get("group") or kwargs.get("process_group")
        msg_dict = {
            "func_name": f"{func_name}",
            "args": f"{args}, {kwargs}",
            "pg_name": f"{dist._get_process_group_name(kwargs.get('pg'))}",  # type: ignore[arg-type]
            "backend": f"{dist.get_backend(group)}",
            "world_size": f"{dist.get_world_size()}",
            "group_size": f"{dist.get_world_size(group)}",
            "global_rank": f"{dist.get_rank()}",
            "local_rank": f"{dist.get_rank(group)}",
        }
        if msg_dict["backend"] == "nccl":  # 如果后端是 nccl
            nccl_version = torch.cuda.nccl.version()  # 获取 nccl 版本信息
            msg_dict["nccl_version"] = ".".join(str(v) for v in nccl_version)  # 添加 nccl 版本到消息字典
    else:
        msg_dict = {  # 如果分布式未初始化，则仅记录函数名称、参数和关键字参数
            "func_name": f"{func_name}",
            "args": f"{args}, {kwargs}",
        }
    return msg_dict  # 返回生成的消息字典


_T = TypeVar("_T")  # 声明一个类型变量 _T
_P = ParamSpec("_P")  # 声明一个 ParamSpec 参数规范


def _exception_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    # 异常日志记录装饰器函数
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return func(*args, **kwargs)  # 调用被装饰函数
        except Exception as error:  # 捕获异常
            msg_dict = _get_msg_dict(func.__name__, *args, **kwargs)  # 获取异常信息字典
            msg_dict["error"] = f"{error}"  # 将异常信息添加到字典
            _c10d_logger.debug(msg_dict)  # 使用全局 logger 记录 DEBUG 级别的异常信息
            raise  # 抛出异常

    return wrapper  # 返回装饰后的函数


def _time_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    # 时间日志记录装饰器函数
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        # 记录进入函数执行前的时间戳（纳秒级）
        t1 = time.time_ns()
        # 调用被装饰的函数，并获取其返回值
        func_return = func(*args, **kwargs)
        # 计算函数执行所花费的时间（纳秒级）
        time_spent = time.time_ns() - t1

        # 获取函数执行相关的日志信息字典
        msg_dict = _get_msg_dict(func.__name__, *args, **kwargs)
        # 将函数执行时间加入日志信息字典中
        msg_dict["time_spent"] = f"{time_spent}ns"
        # 使用调试级别的日志记录器记录日志信息字典
        _c10d_logger.debug(msg_dict)

        # 返回被装饰函数的返回值
        return func_return

    return wrapper
```