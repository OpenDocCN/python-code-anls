# `.\transformers\utils\logging.py`

```
# 设置文件编码为 utf-8
# 版权声明，指明 Optuna 和 Hugging Face 的版权信息
# 根据 Apache 许可证 2.0 版本，对文件进行许可，要求遵守许可证规定
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
# 日志工具

# 导入必要的库
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

# 导入 Hugging Face Hub 工具库
import huggingface_hub.utils as hf_hub_utils
# 导入 tqdm 库
from tqdm import auto as tqdm_lib

# 创建线程锁
_lock = threading.Lock()
# 初始化默认处理程序为 None
_default_handler: Optional[logging.Handler] = None

# 定义日志级别字典
log_levels = {
    "detail": logging.DEBUG,  # 将打印文件名和行号
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# 设置默认日志级别为 WARNING
_default_log_level = logging.WARNING

# 默认情况下启用 tqdm 进度条
_tqdm_active = True

# 获取默认日志级别
def _get_default_logging_level():
    """
    如果 TRANSFORMERS_VERBOSITY 环境变量设置为有效选项之一，则将其作为新的默认级别返回。
    如果没有设置，则回退到 `_default_log_level`
    """
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level

# 获取库名称
def _get_library_name() -> str:
    return __name__.split(".")[0]

# 获取库的根记录器
def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())

# 配置库的根记录器
def _configure_library_root_logger() -> None:
    global _default_handler
    # 使用锁确保线程安全
    with _lock:
        # 如果已经配置了默认处理程序，则直接返回，避免重复配置
        if _default_handler:
            return
        # 将 sys.stderr 设置为流，并作为默认处理程序
        _default_handler = logging.StreamHandler()
        # 根据 https://github.com/pyinstaller/pyinstaller/issues/7334#issuecomment-1357447176 设置默认值
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        # 将默认处理程序的 flush 方法设置为 sys.stderr 的 flush 方法
        _default_handler.flush = sys.stderr.flush

        # 将默认配置应用于库的根记录器
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        # 如果日志级别为 debug，则为便于调试，将 pathname 和 lineno 添加到格式化程序中
        if os.getenv("TRANSFORMERS_VERBOSITY", None) == "detail":
            formatter = logging.Formatter("[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s")
            _default_handler.setFormatter(formatter)

        # 禁止根记录器传播日志消息
        library_root_logger.propagate = False
# 重置库的根日志记录器
def _reset_library_root_logger() -> None:
    # 声明全局变量_default_handler
    global _default_handler

    # 使用锁确保线程安全
    with _lock:
        # 如果_default_handler不存在，则直接返回
        if not _default_handler:
            return

        # 获取库的根日志记录器
        library_root_logger = _get_library_root_logger()
        # 移除默认处理程序_default_handler
        library_root_logger.removeHandler(_default_handler)
        # 设置日志记录器的级别为NOTSET
        library_root_logger.setLevel(logging.NOTSET)
        # 将_default_handler设为None


# 返回日志级别字典
def get_log_levels_dict():
    return log_levels


# 捕获警告
def captureWarnings(capture):
    """
    调用logging库的`captureWarnings`方法来启用对`warnings`库发出的警告进行管理。

    了解更多关于此方法的信息：
    https://docs.python.org/3/library/logging.html#integration-with-the-warnings-module

    所有警告将通过`py.warnings`日志记录器记录。

    注意：如果日志记录器没有处理程序，此方法还会为该日志记录器添加处理程序，并更新该日志记录器的日志级别为库的根日志记录器。
    """
    # 获取名为"py.warnings"的日志记录器
    logger = get_logger("py.warnings")

    # 如果日志记录器没有处理程序，则添加默认处理程序_default_handler
    if not logger.handlers:
        logger.addHandler(_default_handler)

    # 设置日志记录器的级别为库的根日志记录器的级别
    logger.setLevel(_get_library_root_logger().level)

    # 调用_captureWarnings函数
    _captureWarnings(capture)


# 获取指定名称的日志记录器
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    返回指定名称的日志记录器。

    除非您正在编写自定义transformers模块，否则不应直接访问此函数。
    """

    # 如果名称为None，则使用库的名称
    if name is None:
        name = _get_library_name()

    # 配置库的根日志记录器
    _configure_library_root_logger()
    return logging.getLogger(name)


# 获取当前🤗 Transformers根日志记录器的级别
def get_verbosity() -> int:
    """
    返回🤗 Transformers根日志记录器的当前级别。

    返回值:
        `int`: 日志级别。

    <提示>

    🤗 Transformers有以下日志级别：

    - 50: `transformers.logging.CRITICAL`或`transformers.logging.FATAL`
    - 40: `transformers.logging.ERROR`
    - 30: `transformers.logging.WARNING`或`transformers.logging.WARN`
    - 20: `transformers.logging.INFO`
    - 10: `transformers.logging.DEBUG`

    </提示>"""

    # 配置库的根日志记录器
    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


# 设置🤗 Transformers根日志记录器的级别
def set_verbosity(verbosity: int) -> None:
    """
    设置🤗 Transformers根日志记录器的详细级别。

    参数:
        verbosity (`int`):
            日志级别，例如：

            - `transformers.logging.CRITICAL`或`transformers.logging.FATAL`
            - `transformers.logging.ERROR`
            - `transformers.logging.WARNING`或`transformers.logging.WARN`
            - `transformers.logging.INFO`
            - `transformers.logging.DEBUG`
    """

    # 配置库的根日志记录器
    _configure_library_root_logger()
    # 设置根日志记录器的级别为verbosity


# 将详细级别设置为INFO
def set_verbosity_info():
    """将详细级别设置为`INFO`。"""
    return set_verbosity(INFO)


# 将详细级别设置为WARNING
def set_verbosity_warning():
    """将详细级别设置为`WARNING`。"""
    return set_verbosity(WARNING)


# 将详细级别设置为DEBUG
def set_verbosity_debug():
    """将详细级别设置为`DEBUG`。"""
    # 设置日志级别为 DEBUG，并返回设置结果
    return set_verbosity(DEBUG)
# 将日志级别设置为 `ERROR`
def set_verbosity_error():
    return set_verbosity(ERROR)


# 禁用 HuggingFace Transformers 根记录器的默认处理程序
def disable_default_handler() -> None:
    _configure_library_root_logger()
    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


# 启用 HuggingFace Transformers 根记录器的默认处理程序
def enable_default_handler() -> None:
    _configure_library_root_logger()
    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


# 向 HuggingFace Transformers 根记录器添加处理程序
def add_handler(handler: logging.Handler) -> None:
    _configure_library_root_logger()
    assert handler is not None
    _get_library_root_logger().addHandler(handler)


# 从 HuggingFace Transformers 根记录器中移除给定的处理程序
def remove_handler(handler: logging.Handler) -> None:
    _configure_library_root_logger()
    assert handler is not None and handler not in _get_library_root_logger().handlers
    _get_library_root_logger().removeHandler(handler)


# 禁用库日志输出的传播。请注意，默认情况下禁用日志传播。
def disable_propagation() -> None:
    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


# 启用库日志输出的传播。如果根记录器已配置，请禁用 HuggingFace Transformers 的默认处理程序以防止重复记录。
def enable_propagation() -> None:
    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


# 启用每个 HuggingFace Transformers 记录器的显式格式化
def enable_explicit_format() -> None:
    handlers = _get_library_root_logger().handlers
    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)


# 重置 HuggingFace Transformers 记录器的格式化
def reset_format() -> None:
    handlers = _get_library_root_logger().handlers
    for handler in handlers:
        handler.setFormatter(None)


# 与 `logger.warning()` 相同，但如果设置了环境变量 TRANSFORMERS_NO_ADVISORY_WARNINGS=1，则不会打印此警告
def warning_advice(self, *args, **kwargs):
    no_advisory_warnings = os.getenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", False)
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)
# 将 warning_advice 函数赋值给 Logger 类的 warning_advice 属性
logging.Logger.warning_advice = warning_advice

# 使用 functools.lru_cache 装饰器，创建一个缓存装饰函数，用于缓存 warning_once 函数的结果
@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    # 调用 logger 的 warning 方法，传入参数 args 和 kwargs
    self.warning(*args, **kwargs)

# 将 warning_once 函数赋值给 Logger 类的 warning_once 属性
logging.Logger.warning_once = warning_once

# 定义一个名为 EmptyTqdm 的类，用于模拟一个不执行任何操作的 tqdm 对象
class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    # 初始化方法，接收任意参数，但不使用
    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        # 将第一个参数作为迭代器保存在 _iterator 属性中
        self._iterator = args[0] if args else None

    # 返回迭代器的迭代器方法
    def __iter__(self):
        return iter(self._iterator)

    # 获取属性的方法，返回一个空函数
    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        return empty_fn

    # 进入上下文管理器时调用的方法
    def __enter__(self):
        return self

    # 退出上下文管理器时调用的方法
    def __exit__(self, type_, value, traceback):
        return

# 定义一个名为 _tqdm_cls 的类
class _tqdm_cls:
    # 调用实例时调用的方法
    def __call__(self, *args, **kwargs):
        # 如果 _tqdm_active 为真，则返回 tqdm_lib.tqdm(*args, **kwargs)，否则返回 EmptyTqdm(*args, **kwargs)
        if _tqdm_active:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    # 设置锁的方法
    def set_lock(self, *args, **kwargs):
        # 将 _lock 属性设��为 None
        self._lock = None
        # 如果 _tqdm_active 为真，则返回 tqdm_lib.tqdm.set_lock(*args, **kwargs)
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    # 获取锁的方法
    def get_lock(self):
        # 如果 _tqdm_active 为真，则返回 tqdm_lib.tqdm.get_lock()
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()

# 将 _tqdm_cls 实例化后赋值给 tqdm 变量
tqdm = _tqdm_cls()

# 返回一个布尔值，指示是否启用了 tqdm 进度条
def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    # 获取全局变量 _tqdm_active 的布尔值
    global _tqdm_active
    return bool(_tqdm_active)

# 启用 tqdm 进度条的方法
def enable_progress_bar():
    """Enable tqdm progress bar."""
    # 设置全局变量 _tqdm_active 为真
    global _tqdm_active
    _tqdm_active = True
    # 调用 hf_hub_utils.enable_progress_bars() 方法
    hf_hub_utils.enable_progress_bars()

# 禁用 tqdm 进度条的方法
def disable_progress_bar():
    """Disable tqdm progress bar."""
    # 设置全局变量 _tqdm_active 为假
    global _tqdm_active
    _tqdm_active = False
    # 调用 hf_hub_utils.disable_progress_bars() 方法
    hf_hub_utils.disable_progress_bars()
```