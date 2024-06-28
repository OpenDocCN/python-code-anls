# `.\utils\logging.py`

```py
# 设置脚本编码格式为 UTF-8
# Copyright 2020 Optuna, Hugging Face
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则禁止使用该文件
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“原样”分发的软件
# 不附带任何明示或暗示的保证或条件
# 请参阅许可证以了解特定语言的详情
""" Logging utilities."""

# 导入必要的库和模块
import functools
import logging
import os
import sys
import threading
from logging import (
    CRITICAL,  # NOQA
    DEBUG,  # NOQA
    ERROR,  # NOQA
    FATAL,  # NOQA
    INFO,  # NOQA
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # NOQA
)
from logging import captureWarnings as _captureWarnings
from typing import Optional

# 导入 Hugging Face Hub 工具函数
import huggingface_hub.utils as hf_hub_utils
# 导入 tqdm 库的自动模式作为 tqdm_lib
from tqdm import auto as tqdm_lib

# 创建线程锁
_lock = threading.Lock()
# 默认处理器初始化为 None
_default_handler: Optional[logging.Handler] = None

# 日志级别映射字典
log_levels = {
    "detail": logging.DEBUG,  # 还会打印文件名和行号
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# 默认日志级别设置为 WARNING
_default_log_level = logging.WARNING

# 检查是否禁用了进度条
_tqdm_active = not hf_hub_utils.are_progress_bars_disabled()

def _get_default_logging_level():
    """
    如果 TRANSFORMERS_VERBOSITY 环境变量设置为有效选择之一，则返回其作为新的默认级别。
    如果未设置，则回退到 `_default_log_level`
    """
    # 获取环境变量 TRANSFORMERS_VERBOSITY
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        # 如果环境变量在日志级别字典中，则返回对应的日志级别
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            # 否则发出警告
            logging.getLogger().warning(
                f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    # 默认返回 `_default_log_level`
    return _default_log_level

def _get_library_name() -> str:
    # 返回当前模块的名称的第一部分作为库名称
    return __name__.split(".")[0]

def _get_library_root_logger() -> logging.Logger:
    # 返回指定名称的根日志记录器
    return logging.getLogger(_get_library_name())

def _configure_library_root_logger() -> None:
    global _default_handler
    # 使用全局锁 `_lock`，确保线程安全地执行以下代码块
    with _lock:
        # 如果 `_default_handler` 已经设置，则说明日志已经配置过，直接返回
        if _default_handler:
            return
        
        # 如果 `_default_handler` 未设置，则创建一个将日志输出到标准错误流 `sys.stderr` 的流处理器
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        
        # 根据 https://github.com/pyinstaller/pyinstaller/issues/7334#issuecomment-1357447176 设置默认值
        # 如果标准错误流 `sys.stderr` 为 None，则将其重定向到 `/dev/null`
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        # 将流处理器的 flush 方法设置为和 `sys.stderr` 的 flush 方法一致
        _default_handler.flush = sys.stderr.flush

        # 获取库的根日志记录器 `_get_library_root_logger()`，并向其添加 `_default_handler` 处理器
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        
        # 设置库的根日志记录器的日志级别为默认日志级别 `_get_default_logging_level()`
        library_root_logger.setLevel(_get_default_logging_level())
        
        # 如果环境变量 `TRANSFORMERS_VERBOSITY` 的值为 "detail"，则配置日志格式化器为包含路径名和行号的详细格式
        if os.getenv("TRANSFORMERS_VERBOSITY", None) == "detail":
            formatter = logging.Formatter("[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s")
            _default_handler.setFormatter(formatter)

        # 禁止库的根日志记录器向上传播日志消息
        library_root_logger.propagate = False
# 重设库根日志记录器的方法，没有返回值
def _reset_library_root_logger() -> None:
    # 使用全局锁保证线程安全操作
    with _lock:
        # 如果默认处理器 `_default_handler` 不存在，直接返回
        if not _default_handler:
            return
        
        # 获取库根日志记录器对象
        library_root_logger = _get_library_root_logger()
        # 从库根日志记录器中移除 `_default_handler` 处理器
        library_root_logger.removeHandler(_default_handler)
        # 设置库根日志记录器的日志级别为 `NOTSET`，表示不设定特定的级别
        library_root_logger.setLevel(logging.NOTSET)
        # 将 `_default_handler` 置为 `None`
        _default_handler = None


# 返回日志级别字典 `log_levels`
def get_log_levels_dict():
    return log_levels


# 启用/禁用警告捕获的函数
def captureWarnings(capture):
    """
    调用 logging 库中的 `captureWarnings` 方法，用于管理 `warnings` 库发出的警告。

    详细信息请参阅：
    https://docs.python.org/3/library/logging.html#integration-with-the-warnings-module

    所有警告将通过 `py.warnings` 记录器记录。

    注意：如果 `py.warnings` 记录器不存在处理器，则此方法还会添加一个处理器，并更新该记录器的日志级别为库的根日志记录器级别。
    """
    # 获取名为 `py.warnings` 的日志记录器对象
    logger = get_logger("py.warnings")

    # 如果 `logger` 没有处理器，则添加 `_default_handler`
    if not logger.handlers:
        logger.addHandler(_default_handler)

    # 设置 `logger` 的日志级别为库的根日志记录器的级别
    logger.setLevel(_get_library_root_logger().level)

    # 调用内部方法 `_captureWarnings` 启用/禁用警告捕获
    _captureWarnings(capture)


# 获取指定名称的日志记录器对象
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    返回指定名称的日志记录器。

    除非您正在编写自定义的 transformers 模块，否则不应直接访问此函数。
    """
    # 如果 `name` 为 `None`，则使用 `_get_library_name()` 返回的名称
    if name is None:
        name = _get_library_name()

    # 配置库的根日志记录器
    _configure_library_root_logger()
    # 返回指定名称的日志记录器对象
    return logging.getLogger(name)


# 获取当前 🤗 Transformers 根日志记录器的日志级别作为整数返回
def get_verbosity() -> int:
    """
    返回 🤗 Transformers 根日志记录器的当前日志级别作为整数。

    返回值:
        `int`: 日志级别。

    <Tip>

    🤗 Transformers 有以下日志级别:

    - 50: `transformers.logging.CRITICAL` 或 `transformers.logging.FATAL`
    - 40: `transformers.logging.ERROR`
    - 30: `transformers.logging.WARNING` 或 `transformers.logging.WARN`
    - 20: `transformers.logging.INFO`
    - 10: `transformers.logging.DEBUG`

    </Tip>"""
    # 配置库的根日志记录器
    _configure_library_root_logger()
    # 返回根日志记录器的有效日志级别
    return _get_library_root_logger().getEffectiveLevel()


# 设置 🤗 Transformers 根日志记录器的日志级别
def set_verbosity(verbosity: int) -> None:
    """
    设置 🤗 Transformers 根日志记录器的日志级别。

    参数:
        verbosity (`int`):
            日志级别，例如：

            - `transformers.logging.CRITICAL` 或 `transformers.logging.FATAL`
            - `transformers.logging.ERROR`
            - `transformers.logging.WARNING` 或 `transformers.logging.WARN`
            - `transformers.logging.INFO`
            - `transformers.logging.DEBUG`
    """
    # 配置库的根日志记录器
    _configure_library_root_logger()
    # 设置根日志记录器的日志级别
    _get_library_root_logger().setLevel(verbosity)


# 设置日志级别为 `INFO`
def set_verbosity_info():
    """将日志级别设置为 `INFO`。"""
    return set_verbosity(INFO)


# 设置日志级别为 `WARNING`
def set_verbosity_warning():
    """将日志级别设置为 `WARNING`。"""
    return set_verbosity(WARNING)


# 设置日志级别为 `DEBUG`
def set_verbosity_debug():
    """将日志级别设置为 `DEBUG`。"""
    return set_verbosity(DEBUG)
    # 调用一个名为 set_verbosity 的函数，并将常量 DEBUG 作为参数传递给它
    return set_verbosity(DEBUG)
# 设置日志的详细程度为错误级别（ERROR）
def set_verbosity_error():
    """Set the verbosity to the `ERROR` level."""
    return set_verbosity(ERROR)


# 禁用 HuggingFace Transformers 根记录器的默认处理程序
def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()  # 配置库的根记录器

    assert _default_handler is not None  # 确保默认处理程序不为 None
    _get_library_root_logger().removeHandler(_default_handler)  # 从根记录器中移除默认处理程序


# 启用 HuggingFace Transformers 根记录器的默认处理程序
def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()  # 配置库的根记录器

    assert _default_handler is not None  # 确保默认处理程序不为 None
    _get_library_root_logger().addHandler(_default_handler)  # 向根记录器添加默认处理程序


# 向 HuggingFace Transformers 根记录器添加处理程序
def add_handler(handler: logging.Handler) -> None:
    """adds a handler to the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()  # 配置库的根记录器

    assert handler is not None  # 确保处理程序不为 None
    _get_library_root_logger().addHandler(handler)  # 向根记录器添加处理程序


# 从 HuggingFace Transformers 根记录器移除处理程序
def remove_handler(handler: logging.Handler) -> None:
    """removes given handler from the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()  # 配置库的根记录器

    assert handler is not None and handler not in _get_library_root_logger().handlers  # 确保处理程序不为 None，且不在根记录器的处理程序列表中
    _get_library_root_logger().removeHandler(handler)  # 从根记录器中移除处理程序


# 禁用库日志输出的传播
def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """

    _configure_library_root_logger()  # 配置库的根记录器
    _get_library_root_logger().propagate = False  # 将根记录器的传播设置为 False


# 启用库日志输出的传播
def enable_propagation() -> None:
    """
    Enable propagation of the library log outputs. Please disable the HuggingFace Transformers's default handler to
    prevent double logging if the root logger has been configured.
    """

    _configure_library_root_logger()  # 配置库的根记录器
    _get_library_root_logger().propagate = True  # 将根记录器的传播设置为 True


# 启用明确的格式化方式用于每个 HuggingFace Transformers 的记录器
def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers  # 获取根记录器的所有处理程序

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)  # 为每个处理程序设置指定的格式化方式


# 重置 HuggingFace Transformers 记录器的格式化方式
def reset_format() -> None:
    """
    Resets the formatting for HuggingFace Transformers's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers  # 获取根记录器的所有处理程序

    for handler in handlers:
        handler.setFormatter(None)  # 将每个处理程序的格式化方式重置为 None


# 提供警告建议，类似于 logger.warning()，但如果环境变量 TRANSFORMERS_NO_ADVISORY_WARNINGS=1 设置为真，则不打印该警告
def warning_advice(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but if env var TRANSFORMERS_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    no_advisory_warnings = os.getenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", False)  # 获取环境变量 TRANSFORMERS_NO_ADVISORY_WARNINGS 的值
    if no_advisory_warnings:
        return  # 如果设置了环境变量不显示警告，则返回
    self.warning(*args, **kwargs)  # 否则调用 logger 的 warning 方法输出警告
# 将警告建议函数绑定到 Logger 对象的 warning_advice 属性上
logging.Logger.warning_advice = warning_advice

# 使用 functools.lru_cache(None) 装饰器定义一个函数，使其能够缓存结果
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    # 调用 logger 的 warning 方法，传递相同的参数和关键字参数
    self.warning(*args, **kwargs)

# 将 warning_once 函数绑定到 Logger 对象的 warning_once 属性上
logging.Logger.warning_once = warning_once

# 定义一个名为 EmptyTqdm 的类，作为 tqdm 的替代品，不执行任何操作
class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        # 如果有参数 args，则将第一个参数作为迭代器存储在 _iterator 属性中，否则置为 None
        self._iterator = args[0] if args else None

    def __iter__(self):
        # 返回 _iterator 的迭代器
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""
        # 定义一个空函数 empty_fn，忽略所有传入的参数和关键字参数，并返回空值
        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        # 返回空函数 empty_fn
        return empty_fn

    def __enter__(self):
        # 返回自身实例，用于支持上下文管理器
        return self

    def __exit__(self, type_, value, traceback):
        # 返回 None，表示不处理任何异常
        return

# 定义一个名为 _tqdm_cls 的类
class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        # 如果 _tqdm_active 为真，则调用 tqdm_lib.tqdm 创建 tqdm 进度条并返回，否则返回 EmptyTqdm 实例
        if _tqdm_active:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        # 设置 _lock 为 None
        self._lock = None
        # 如果 _tqdm_active 为真，则调用 tqdm_lib.tqdm.set_lock 设置锁，并返回其结果

    def get_lock(self):
        # 如果 _tqdm_active 为真，则调用 tqdm_lib.tqdm.get_lock 返回锁对象

# 创建一个 tqdm 实例，并赋值给 tqdm 变量
tqdm = _tqdm_cls()

# 定义一个函数 is_progress_bar_enabled，返回一个布尔值，指示 tqdm 进度条是否启用
def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    global _tqdm_active
    return bool(_tqdm_active)

# 定义一个函数 enable_progress_bar，启用 tqdm 进度条
def enable_progress_bar():
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True
    hf_hub_utils.enable_progress_bars()

# 定义一个函数 disable_progress_bar，禁用 tqdm 进度条
def disable_progress_bar():
    """Disable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False
    hf_hub_utils.disable_progress_bars()
```