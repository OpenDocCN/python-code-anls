# `.\PaddleOCR\StyleText\utils\logging.py`

```
# 导入必要的库
import os
import sys
import logging
import functools
import paddle.distributed as dist

# 初始化记录器字典
logger_initialized = {}

# 使用 functools.lru_cache() 装饰器缓存结果，初始化并获取一个记录器
def get_logger(name='srnet', log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    # 获取指定名称的记录器
    logger = logging.getLogger(name)
    # 如果记录器已经初始化过，则直接返回记录器
    if name in logger_initialized:
        return logger
    # 遍历已初始化的记录器，如果指定名称以已初始化的记录器名称开头，则直接返回记录器
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # 设置日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    # 创建一个输出到标准输出的 StreamHandler
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    # 设置日志处理器的格式
    stream_handler.setFormatter(formatter)
    # 将流处理器添加到日志记录器中
    logger.addHandler(stream_handler)
    # 如果日志文件不为空且当前进程的排名为0
    if log_file is not None and dist.get_rank() == 0:
        # 获取日志文件的文件夹路径
        log_file_folder = os.path.split(log_file)[0]
        # 如果文件夹不存在则创建
        os.makedirs(log_file_folder, exist_ok=True)
        # 创建文件处理器，追加模式
        file_handler = logging.FileHandler(log_file, 'a')
        # 设置文件处理器的格式
        file_handler.setFormatter(formatter)
        # 将文件处理器添加到日志记录器中
        logger.addHandler(file_handler)
    # 如果当前进程的排名为0
    if dist.get_rank() == 0:
        # 设置日志记录器的日志级别
        logger.setLevel(log_level)
    else:
        # 设置日志记录器的日志级别为 ERROR
        logger.setLevel(logging.ERROR)
    # 标记该日志记录器已初始化
    logger_initialized[name] = True
    # 返回日志记录器
    return logger
```