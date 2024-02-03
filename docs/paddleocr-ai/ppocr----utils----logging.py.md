# `.\PaddleOCR\ppocr\utils\logging.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按"原样"分发，不附带任何担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
"""
# 引用来源
# https://github.com/WenmuZhou/PytorchOCR/blob/master/torchocr/utils/logging.py

# 导入必要的库
import os
import sys
import logging
import functools
import paddle.distributed as dist

# 初始化记录器字典
logger_initialized = {}

# 使用 functools.lru_cache() 装饰器缓存结果
@functools.lru_cache()
# 初始化并获取一个名为 'ppocr' 的记录器
def get_logger(name='ppocr', log_file=None, log_level=logging.DEBUG):
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
    # 获取名为 'name' 的记录器
    logger = logging.getLogger(name)
    # 如果记录器已经初始化过，则直接返回记录器
    if name in logger_initialized:
        return logger
    # 遍历已初始化的记录器
    for logger_name in logger_initialized:
        # 如果 'name' 以已初始化的记录器名字开头，则返回记录器
        if name.startswith(logger_name):
            return logger
    # 创建日志格式化器，定义日志输出格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")
    
    # 创建输出到标准输出流的日志处理器
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    # 将输出到标准输出流的日志处理器添加到日志记录器中
    logger.addHandler(stream_handler)
    
    # 如果指定了日志文件且当前进程为主进程，则创建输出到文件的日志处理器
    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 如果当前进程为主进程，则设置日志记录器的日志级别为指定的日志级别，否则设置为 ERROR 级别
    if dist.get_rank() == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    
    # 标记该日志记录器已经初始化
    logger_initialized[name] = True
    
    # 禁止日志传播到父记录器
    logger.propagate = False
    
    # 返回配置好的日志记录器
    return logger
```