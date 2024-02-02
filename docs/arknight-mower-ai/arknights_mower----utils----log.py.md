# `arknights-mower\arknights_mower\utils\log.py`

```py
# 导入日志模块
import logging
# 导入操作系统模块
import os
# 导入系统模块
import sys
# 导入线程模块
import threading
# 导入时间模块
import time
# 导入日志处理模块
from logging.handlers import RotatingFileHandler
# 导入路径模块
from pathlib import Path
# 导入彩色日志模块
import colorlog
# 导入配置模块
from . import config

# 定义日志的基本格式
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(relativepath)s:%(lineno)d - %(funcName)s - %(message)s'
# 定义彩色日志的格式
COLOR_FORMAT = '%(log_color)s%(asctime)s - %(levelname)s - %(relativepath)s:%(lineno)d - %(funcName)s - %(message)s'
# 定义日期格式
DATE_FORMAT = None
# 创建基本格式的日志处理器
basic_formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
# 创建彩色日志格式的日志处理器
color_formatter = colorlog.ColoredFormatter(COLOR_FORMAT, DATE_FORMAT)

# 创建路径过滤器类
class PackagePathFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True

# 创建最大级别过滤器类
class MaxFilter(object):
    def __init__(self, max_level: int) -> None:
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno <= self.max_level:
            return True

# 创建日志处理器类
class Handler(logging.StreamHandler):
    def __init__(self, pipe):
        logging.StreamHandler.__init__(self)
        self.pipe = pipe

    def emit(self, record):
        record = f'{record.message}'
        self.pipe.send({'type':'log','data':record})

# 创建控制台日志处理器
chlr = logging.StreamHandler(stream=sys.stdout)
chlr.setFormatter(color_formatter)
chlr.setLevel('INFO')
chlr.addFilter(MaxFilter(logging.INFO))
chlr.addFilter(PackagePathFilter())

# 创建错误日志处理器
ehlr = logging.StreamHandler(stream=sys.stderr)
ehlr.setFormatter(color_formatter)
ehlr.setLevel('WARNING')
ehlr.addFilter(PackagePathFilter())

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
# 添加控制台日志处理器
logger.addHandler(chlr)
# 添加错误日志处理器
logger.addHandler(ehlr)

# 初始化日志文件处理器
def init_fhlr(pipe=None) -> None:
    """ initialize log file """
    # 如果日志文件路径为空，则直接返回
    if config.LOGFILE_PATH is None:
        return
    # 创建日志文件夹
    folder = Path(config.LOGFILE_PATH)
    folder.mkdir(exist_ok=True, parents=True)
    # 创建日志文件处理器
    fhlr = RotatingFileHandler(
        folder.joinpath('runtime.log'),
        encoding='utf8',
        maxBytes=10 * 1024 * 1024,
        backupCount=config.LOGFILE_AMOUNT,
    )
    fhlr.setFormatter(basic_formatter)
    fhlr.setLevel('DEBUG')
    fhlr.addFilter(PackagePathFilter())
    # 将日志文件处理器添加到日志记录器中
    logger.addHandler(fhlr)
    # 如果有管道，则创建处理器
    if pipe is not None:
        wh = Handler(pipe)
        wh.setLevel(logging.INFO)
        logger.addHandler(wh)

# 设置调试模式
def set_debug_mode() -> None:
    """ set debud mode on """
    # 如果调试模式开启
    if config.DEBUG_MODE:
        # 记录调试模式日志
        logger.info(f'Start debug mode, log is stored in {config.LOGFILE_PATH}')
        # 初始化日志文件处理器
        init_fhlr()

# 保存截图
def save_screenshot(img: bytes, subdir: str = '') -> None:
    """ save screenshot """
    # 如果截图路径为空，则直接返回
    if config.SCREENSHOT_PATH is None:
        return
    # 创建截图文件夹
    folder = Path(config.SCREENSHOT_PATH).joinpath(subdir)
    folder.mkdir(exist_ok=True, parents=True)
    # 如果子目录不为'-1'且文件数量超过最大限制，则删除多余的截图
    if subdir != '-1' and len(list(folder.iterdir())) > config.SCREENSHOT_MAXNUM:
        screenshots = list(folder.iterdir())
        screenshots = sorted(screenshots, key=lambda x: x.name)
        for x in screenshots[: -config.SCREENSHOT_MAXNUM]:
            logger.debug(f'remove screenshot: {x.name}')
            x.unlink()
    # 生成截图文件名
    filename = time.strftime('%Y%m%d%H%M%S.png', time.localtime())
    # 写入截图文件
    with folder.joinpath(filename).open('wb') as f:
        f.write(img)
    logger.debug(f'save screenshot: {filename}')

# 日志同步线程类
class log_sync(threading.Thread):
    """ recv output from subprocess """

    def __init__(self, process: str, pipe: int) -> None:
        self.process = process
        self.pipe = os.fdopen(pipe)
        super().__init__(daemon=True)

    def __del__(self) -> None:
        self.pipe.close()
    # 定义一个方法，用于执行某些操作，没有返回值
    def run(self) -> None:
        # 循环执行以下操作
        while True:
            # 从管道中读取一行数据，并去除首尾的空白字符
            line = self.pipe.readline().strip()
            # 记录调试信息，包括进程名称和读取的行数据
            logger.debug(f'{self.process}: {line}')
```