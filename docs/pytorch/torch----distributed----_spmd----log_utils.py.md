# `.\pytorch\torch\distributed\_spmd\log_utils.py`

```py
# 导入日志模块
import logging
# 导入日志配置模块
import logging.config
# 导入操作系统相关模块
import os
# 导入类型提示模块，用于类型检查
from typing import Optional

# 导入分布式操作模块
import torch.distributed as dist

# 日志配置字典，定义了不同日志格式、处理器和记录器
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "spmd_format": {"format": "%(name)s: [%(levelname)s] %(message)s"},
        "graph_opt_format": {"format": "%(name)s: [%(levelname)s] %(message)s"},
    },
    "handlers": {
        "spmd_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "spmd_format",
            "stream": "ext://sys.stdout",
        },
        "graph_opt_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "graph_opt_format",
            "stream": "ext://sys.stdout",
        },
        "null_console": {
            "class": "logging.NullHandler",
        },
    },
    "loggers": {
        "spmd_exp": {
            "level": "DEBUG",
            "handlers": ["spmd_console"],
            "propagate": False,
        },
        "graph_opt": {
            "level": "DEBUG",
            "handlers": ["graph_opt_console"],
            "propagate": False,
        },
        "null_logger": {
            "handlers": ["null_console"],
            "propagate": False,
        },
        # TODO(anj): Add loggers for MPMD
    },
    "disable_existing_loggers": False,
}

# 根据日志类型获取相应的日志记录器对象
def get_logger(log_type: str) -> Optional[logging.Logger]:
    # 导入内部配置模块，用于获取配置信息
    from torch.distributed._spmd import config

    # 如果当前环境中不存在 PYTEST_CURRENT_TEST 变量
    if "PYTEST_CURRENT_TEST" not in os.environ:
        # 使用日志配置字典配置日志
        logging.config.dictConfig(LOGGING_CONFIG)
        # 获取可用的日志记录器列表
        avail_loggers = list(LOGGING_CONFIG["loggers"].keys())  # type: ignore[attr-defined]
        # 断言所请求的日志类型在可用的日志记录器列表中
        assert (
            log_type in avail_loggers
        ), f"Unable to find {log_type} in the available list of loggers {avail_loggers}"

        # 如果分布式环境未初始化，则返回相应的日志记录器
        if not dist.is_initialized():
            return logging.getLogger(log_type)

        # 如果当前进程的 rank 为 0
        if dist.get_rank() == 0:
            # 获取日志记录器对象
            logger = logging.getLogger(log_type)
            # 设置日志记录器的日志级别为配置文件中的级别
            logger.setLevel(config.log_level)
            # 如果配置中指定了日志文件名，则创建文件处理器并添加到日志记录器中
            if config.log_file_name is not None:
                log_file = logging.FileHandler(config.log_file_name)
                log_file.setLevel(config.log_level)
                logger.addHandler(log_file)
        else:
            # 如果当前进程的 rank 不为 0，则返回空日志记录器
            logger = logging.getLogger("null_logger")

        # 返回获取到的日志记录器对象
        return logger

    # 如果是在 pytest 测试环境中，则返回空日志记录器
    return logging.getLogger("null_logger")
```