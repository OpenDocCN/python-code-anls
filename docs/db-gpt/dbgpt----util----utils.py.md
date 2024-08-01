# `.\DB-GPT-src\dbgpt\util\utils.py`

```py
# 引入 asyncio 模块，支持异步编程
import asyncio
# 引入 logging 模块，用于记录日志
import logging
# 引入 logging.handlers 模块，提供日志处理器
import logging.handlers
# 引入 os 模块，提供与操作系统交互的功能
import os
# 引入 sys 模块，提供与解释器交互的功能
import sys
# 引入 typing 模块，提供类型相关的支持
from typing import Any, List, Optional, cast

# 从 dbgpt.configs.model_config 模块导入 LOGDIR 常量
from dbgpt.configs.model_config import LOGDIR

# 尝试导入 termcolor 模块，如果失败定义 colored 函数为原样输出
try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

# 定义服务器错误消息的常量
server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)

# 初始化全局变量 handler 为 None
handler = None


# 返回环境变量 DBGPT_LOG_LEVEL，如果不存在则默认为 "INFO"
def _get_logging_level() -> str:
    return os.getenv("DBGPT_LOG_LEVEL", "INFO")


# 设置日志级别
def setup_logging_level(
    logging_level: Optional[str] = None, logger_name: Optional[str] = None
):
    # 如果未提供 logging_level，则使用环境变量中的日志级别
    if not logging_level:
        logging_level = _get_logging_level()
    # 将字符串形式的日志级别转换为对应的整数值
    if type(logging_level) is str:
        logging_level = logging.getLevelName(logging_level.upper())
    # 如果提供了 logger_name，则设置该名称的日志记录器的级别
    if logger_name:
        logger = logging.getLogger(logger_name)
        logger.setLevel(cast(str, logging_level))
    else:
        # 否则，设置根日志记录器的基本配置
        logging.basicConfig(level=logging_level, encoding="utf-8")


# 设置日志记录器的配置
def setup_logging(
    logger_name: str,
    logging_level: Optional[str] = None,
    logger_filename: Optional[str] = None,
    redirect_stdio: bool = False,
):
    # 如果未提供 logging_level，则使用环境变量中的日志级别
    if not logging_level:
        logging_level = _get_logging_level()
    # 构建日志记录器
    logger = _build_logger(logger_name, logging_level, logger_filename, redirect_stdio)
    try:
        # 尝试导入 coloredlogs 模块，如果成功则设置日志记录器的颜色输出
        import coloredlogs

        color_level = logging_level if logging_level else "INFO"
        coloredlogs.install(level=color_level, logger=logger)
    except ImportError:
        pass


# 获取 GPU 的内存使用情况
def get_gpu_memory(max_gpus=None):
    import torch

    gpu_memory = []
    # 获取 GPU 数量，如果指定了 max_gpus 则取其与实际 GPU 数量的较小值
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )
    # 遍历每个 GPU
    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)  # 总内存
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # 已分配内存
            available_memory = total_memory - allocated_memory  # 可用内存
            gpu_memory.append(available_memory)  # 将可用内存添加到列表中
    return gpu_memory  # 返回 GPU 内存列表


# 构建日志记录器
def _build_logger(
    logger_name,
    logging_level: Optional[str] = None,
    logger_filename: Optional[str] = None,
    redirect_stdio: bool = False,
):
    global handler

    # 定义日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 如果根记录器没有处理器，则设置根记录器的日志级别
    if not logging.getLogger().handlers:
        setup_logging_level(logging_level=logging_level)
    # 设置根记录器的格式为定义好的 formatter
    logging.getLogger().handlers[0].setFormatter(formatter)

    # 添加一个文件处理器供所有记录器使用
    # 如果未提供 handler 且有指定日志文件名，则进行日志设置
    if handler is None and logger_filename:
        # 确保 LOGDIR 目录存在，如果不存在则创建
        os.makedirs(LOGDIR, exist_ok=True)
        # 构造完整的日志文件路径
        filename = os.path.join(LOGDIR, logger_filename)
        # 创建 TimedRotatingFileHandler 处理器，按天切割日志文件，使用 UTC 时间
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True
        )
        # 设置日志输出格式
        handler.setFormatter(formatter)

        # 如果设置了日志级别，则应用于 handler
        if logging_level is not None:
            handler.setLevel(logging_level)
        
        # 将 handler 添加到根日志器中
        logging.getLogger().addHandler(handler)
        
        # 遍历所有日志器，将 handler 添加到每个 Logger 对象中
        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)
                item.propagate = True
                logging.getLogger(name).debug(f"Added handler to logger: {name}")
            else:
                logging.getLogger(name).debug(f"Skipping non-logger: {name}")

        # 如果需要重定向标准输出和标准错误输出到日志
        if redirect_stdio:
            # 创建输出到标准输出的处理器
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(formatter)
            # 创建输出到标准错误的处理器
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(formatter)

            # 获取根日志器对象
            root_logger = logging.getLogger()
            # 将 stdout_handler 和 stderr_handler 添加到根日志器
            root_logger.addHandler(stdout_handler)
            root_logger.addHandler(stderr_handler)
            logging.getLogger().debug("Added stdout and stderr handlers to root logger")

    # 获取指定名称的 Logger 对象
    logger = logging.getLogger(logger_name)

    # 设置指定日志器的日志级别和记录器名称
    setup_logging_level(logging_level=logging_level, logger_name=logger_name)

    # 调试信息：打印指定日志器的所有处理器
    logging.getLogger(logger_name).debug(
        f"Logger {logger_name} handlers: {logger.handlers}"
    )
    # 调试信息：打印全局 handler 的信息
    logging.getLogger(logger_name).debug(f"Global handler: {handler}")

    # 返回指定名称的 Logger 对象
    return logger
def get_or_create_event_loop() -> asyncio.BaseEventLoop:
    loop = None
    try:
        # 尝试获取当前的事件循环
        loop = asyncio.get_event_loop()
        assert loop is not None
        return cast(asyncio.BaseEventLoop, loop)
    except RuntimeError as e:
        # 捕获事件循环相关的异常
        if not "no running event loop" in str(e) and not "no current event loop" in str(e):
            raise e
        # 如果没有运行的事件循环，则创建一个新的事件循环
        logging.warning("Cant not get running event loop, create new event loop now")
    return cast(asyncio.BaseEventLoop, asyncio.get_event_loop_policy().new_event_loop())


def logging_str_to_uvicorn_level(log_level_str):
    level_str_mapping = {
        "CRITICAL": "critical",
        "ERROR": "error",
        "WARNING": "warning",
        "INFO": "info",
        "DEBUG": "debug",
        "NOTSET": "info",
    }
    # 将日志级别字符串转换为uvicorn日志级别字符串，如果没有匹配则默认为"info"
    return level_str_mapping.get(log_level_str.upper(), "info")


class EndpointFilter(logging.Filter):
    """Disable access log on certain endpoint

    source: https://github.com/encode/starlette/issues/864#issuecomment-1254987630
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        # 初始化EndpointFilter，设置要禁用日志的路径
        self._path = path

    def filter(self, record: logging.LogRecord) -> bool:
        # 根据日志记录的消息判断是否包含要禁用日志的路径
        return record.getMessage().find(self._path) == -1


def setup_http_service_logging(exclude_paths: Optional[List[str]] = None):
    """Setup http service logging

    Now just disable some logs

    Args:
        exclude_paths (List[str]): The paths to disable log
    """
    if not exclude_paths:
        # 如果没有传入要排除的路径，则默认不显示心跳日志
        exclude_paths = ["/api/controller/heartbeat", "/api/health"]
    uvicorn_logger = logging.getLogger("uvicorn.access")
    if uvicorn_logger:
        # 添加EndpointFilter来禁用指定路径的访问日志
        for path in exclude_paths:
            uvicorn_logger.addFilter(EndpointFilter(path=path))
    httpx_logger = logging.getLogger("httpx")
    if httpx_logger:
        # 设置httpx日志记录器的日志级别为WARNING
        httpx_logger.setLevel(logging.WARNING)
```