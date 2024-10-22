# `.\diffusers\utils\logging.py`

```
# 指定文件编码为 UTF-8
# coding=utf-8
# 版权声明，标明版权所有者和年份
# Copyright 2024 Optuna, Hugging Face
#
# 根据 Apache License 2.0 版本许可本文件的使用
# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件在未遵守许可证的情况下不可使用
# you may not use this file except in compliance with the License.
# 可在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 在适用的情况下，许可证下的软件以“原样”提供，不提供任何明示或暗示的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
"""记录工具函数的模块。"""

# 导入 logging 模块以实现日志记录功能
import logging
# 导入 os 模块以进行操作系统交互
import os
# 导入 sys 模块以访问系统特定参数和功能
import sys
# 导入 threading 模块以实现线程支持
import threading
# 从 logging 模块导入不同的日志级别常量
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
# 导入 Dict 和 Optional 类型以进行类型注解
from typing import Dict, Optional

# 从 tqdm 库导入自动选择的进度条支持
from tqdm import auto as tqdm_lib

# 创建一个线程锁以确保线程安全
_lock = threading.Lock()
# 定义一个默认的日志处理程序，初始为 None
_default_handler: Optional[logging.Handler] = None

# 定义日志级别的字典，映射字符串到 logging 的级别
log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# 设置默认日志级别为 WARNING
_default_log_level = logging.WARNING

# 标志表示进度条是否处于活动状态
_tqdm_active = True

# 定义获取默认日志级别的函数
def _get_default_logging_level() -> int:
    """
    如果环境变量 DIFFUSERS_VERBOSITY 设置为有效选项，则返回该值作为新的默认级别。
    如果没有设置，则返回 `_default_log_level`。
    """
    # 获取环境变量 DIFFUSERS_VERBOSITY 的值
    env_level_str = os.getenv("DIFFUSERS_VERBOSITY", None)
    # 如果环境变量存在
    if env_level_str:
        # 检查环境变量值是否在日志级别字典中
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            # 如果值无效，记录警告信息
            logging.getLogger().warning(
                f"Unknown option DIFFUSERS_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    # 返回默认日志级别
    return _default_log_level

# 定义获取库名称的函数
def _get_library_name() -> str:
    # 返回模块名称的第一个部分作为库名称
    return __name__.split(".")[0]

# 定义获取库根日志记录器的函数
def _get_library_root_logger() -> logging.Logger:
    # 返回库名称对应的日志记录器
    return logging.getLogger(_get_library_name())

# 定义配置库根日志记录器的函数
def _configure_library_root_logger() -> None:
    global _default_handler

    # 使用线程锁来确保线程安全
    with _lock:
        # 如果默认处理程序已存在，返回
        if _default_handler:
            # 该库已配置根日志记录器
            return
        # 创建一个流处理程序，输出到标准错误
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.

        # 检查 sys.stderr 是否存在
        if sys.stderr:  # only if sys.stderr exists, e.g. when not using pythonw in windows
            # 设置 flush 方法为 sys.stderr 的 flush 方法
            _default_handler.flush = sys.stderr.flush

        # 应用默认配置到库根日志记录器
        library_root_logger = _get_library_root_logger()
        # 添加默认处理程序到库根日志记录器
        library_root_logger.addHandler(_default_handler)
        # 设置库根日志记录器的日志级别
        library_root_logger.setLevel(_get_default_logging_level())
        # 禁用日志记录器的传播
        library_root_logger.propagate = False

# 定义重置库根日志记录器的函数
def _reset_library_root_logger() -> None:
    global _default_handler
    # 使用锁确保线程安全，防止竞争条件
        with _lock:
            # 如果没有默认处理器，则直接返回
            if not _default_handler:
                return
    
            # 获取库的根日志记录器
            library_root_logger = _get_library_root_logger()
            # 从根日志记录器中移除默认处理器
            library_root_logger.removeHandler(_default_handler)
            # 将根日志记录器的日志级别设置为 NOTSET，表示接受所有级别的日志
            library_root_logger.setLevel(logging.NOTSET)
            # 将默认处理器设置为 None，表示不再使用默认处理器
            _default_handler = None
# 获取日志级别字典
def get_log_levels_dict() -> Dict[str, int]:
    # 返回全局日志级别字典
    return log_levels


# 获取指定名称的日志记录器
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    返回具有指定名称的日志记录器。

    该函数不应直接访问，除非您正在编写自定义的 diffusers 模块。
    """

    # 如果未提供名称，则获取库名称
    if name is None:
        name = _get_library_name()

    # 配置库根日志记录器
    _configure_library_root_logger()
    # 返回指定名称的日志记录器
    return logging.getLogger(name)


# 获取当前日志级别
def get_verbosity() -> int:
    """
    返回 🤗 Diffusers 根日志记录器的当前级别作为 `int`。

    返回：
        `int`:
            日志级别整数，可以是以下之一：

            - `50`: `diffusers.logging.CRITICAL` 或 `diffusers.logging.FATAL`
            - `40`: `diffusers.logging.ERROR`
            - `30`: `diffusers.logging.WARNING` 或 `diffusers.logging.WARN`
            - `20`: `diffusers.logging.INFO`
            - `10`: `diffusers.logging.DEBUG`
    """

    # 配置库根日志记录器
    _configure_library_root_logger()
    # 返回根日志记录器的有效级别
    return _get_library_root_logger().getEffectiveLevel()


# 设置日志级别
def set_verbosity(verbosity: int) -> None:
    """
    设置 🤗 Diffusers 根日志记录器的详细程度。

    参数：
        verbosity (`int`):
            日志级别，可以是以下之一：

            - `diffusers.logging.CRITICAL` 或 `diffusers.logging.FATAL`
            - `diffusers.logging.ERROR`
            - `diffusers.logging.WARNING` 或 `diffusers.logging.WARN`
            - `diffusers.logging.INFO`
            - `diffusers.logging.DEBUG`
    """

    # 配置库根日志记录器
    _configure_library_root_logger()
    # 设置根日志记录器的级别
    _get_library_root_logger().setLevel(verbosity)


# 设置日志级别为 INFO
def set_verbosity_info() -> None:
    """将详细程度设置为 `INFO` 级别。"""
    # 调用设置详细程度的函数
    return set_verbosity(INFO)


# 设置日志级别为 WARNING
def set_verbosity_warning() -> None:
    """将详细程度设置为 `WARNING` 级别。"""
    # 调用设置详细程度的函数
    return set_verbosity(WARNING)


# 设置日志级别为 DEBUG
def set_verbosity_debug() -> None:
    """将详细程度设置为 `DEBUG` 级别。"""
    # 调用设置详细程度的函数
    return set_verbosity(DEBUG)


# 设置日志级别为 ERROR
def set_verbosity_error() -> None:
    """将详细程度设置为 `ERROR` 级别。"""
    # 调用设置详细程度的函数
    return set_verbosity(ERROR)


# 禁用默认处理程序
def disable_default_handler() -> None:
    """禁用 🤗 Diffusers 根日志记录器的默认处理程序。"""

    # 配置库根日志记录器
    _configure_library_root_logger()

    # 确保默认处理程序存在
    assert _default_handler is not None
    # 从根日志记录器中移除默认处理程序
    _get_library_root_logger().removeHandler(_default_handler)


# 启用默认处理程序
def enable_default_handler() -> None:
    """启用 🤗 Diffusers 根日志记录器的默认处理程序。"""

    # 配置库根日志记录器
    _configure_library_root_logger()

    # 确保默认处理程序存在
    assert _default_handler is not None
    # 将默认处理程序添加到根日志记录器
    _get_library_root_logger().addHandler(_default_handler)


# 添加处理程序到日志记录器
def add_handler(handler: logging.Handler) -> None:
    """将处理程序添加到 HuggingFace Diffusers 根日志记录器。"""

    # 配置库根日志记录器
    _configure_library_root_logger()

    # 确保处理程序存在
    assert handler is not None
    # 将处理程序添加到根日志记录器
    _get_library_root_logger().addHandler(handler)


# 从日志记录器移除处理程序
def remove_handler(handler: logging.Handler) -> None:
    """从 HuggingFace Diffusers 根日志记录器移除给定的处理程序。"""

    # 配置库根日志记录器
    _configure_library_root_logger()
    # 确保处理器不为空，并且在库根日志记录器的处理器列表中
    assert handler is not None and handler in _get_library_root_logger().handlers
    # 从库根日志记录器中移除指定的处理器
    _get_library_root_logger().removeHandler(handler)
# 定义一个函数，用于禁用库的日志输出传播
def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """
    # 配置库的根日志记录器
    _configure_library_root_logger()
    # 设置根日志记录器的传播属性为 False，禁用日志传播
    _get_library_root_logger().propagate = False


# 定义一个函数，用于启用库的日志输出传播
def enable_propagation() -> None:
    """
    Enable propagation of the library log outputs. Please disable the HuggingFace Diffusers' default handler to prevent
    double logging if the root logger has been configured.
    """
    # 配置库的根日志记录器
    _configure_library_root_logger()
    # 设置根日志记录器的传播属性为 True，启用日志传播
    _get_library_root_logger().propagate = True


# 定义一个函数，用于启用明确的日志格式
def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every 🤗 Diffusers' logger. The explicit formatter is as follows:
    ```py
    [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    # 获取根日志记录器的所有处理器
    handlers = _get_library_root_logger().handlers

    # 遍历每个处理器，设置其格式化器
    for handler in handlers:
        # 创建一个新的格式化器
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        # 将格式化器设置到处理器上
        handler.setFormatter(formatter)


# 定义一个函数，用于重置日志格式
def reset_format() -> None:
    """
    Resets the formatting for 🤗 Diffusers' loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    # 获取根日志记录器的所有处理器
    handlers = _get_library_root_logger().handlers

    # 遍历每个处理器，重置其格式化器
    for handler in handlers:
        # 将处理器的格式化器设置为 None，重置格式
        handler.setFormatter(None)


# 定义一个方法，用于发出警告信息
def warning_advice(self, *args, **kwargs) -> None:
    """
    This method is identical to `logger.warning()`, but if env var DIFFUSERS_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    # 检查环境变量是否设置为不发出建议警告
    no_advisory_warnings = os.getenv("DIFFUSERS_NO_ADVISORY_WARNINGS", False)
    # 如果设置了环境变量，则直接返回，不发出警告
    if no_advisory_warnings:
        return
    # 调用日志记录器的警告方法
    self.warning(*args, **kwargs)


# 将自定义的警告方法绑定到日志记录器
logging.Logger.warning_advice = warning_advice


# 定义一个空的 tqdm 类，用于替代真实的进度条
class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    # 初始化方法，接收可变参数
    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        # 如果有参数，保存第一个参数为迭代器
        self._iterator = args[0] if args else None

    # 定义迭代器方法，返回迭代器
    def __iter__(self):
        return iter(self._iterator)

    # 定义属性访问方法，返回一个空函数
    def __getattr__(self, _):
        """Return empty function."""

        # 返回一个空函数
        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        return empty_fn

    # 定义上下文管理器的进入方法
    def __enter__(self):
        return self

    # 定义上下文管理器的退出方法
    def __exit__(self, type_, value, traceback):
        return


# 定义一个自定义的 tqdm 类
class _tqdm_cls:
    # 定义调用方法
    def __call__(self, *args, **kwargs):
        # 检查 tqdm 是否处于激活状态
        if _tqdm_active:
            # 返回激活状态下的 tqdm 实例
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            # 返回空的 tqdm 实例
            return EmptyTqdm(*args, **kwargs)

    # 定义设置锁的方法
    def set_lock(self, *args, **kwargs):
        # 将锁设置为 None
        self._lock = None
        # 如果 tqdm 处于激活状态，设置锁
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    # 定义获取锁的方法
    def get_lock(self):
        # 如果 tqdm 处于激活状态，获取锁
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()


# 创建一个 _tqdm_cls 的实例
tqdm = _tqdm_cls()


# 定义一个函数，检查进度条是否启用
def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    global _tqdm_active
    # 返回进度条激活状态的布尔值
    return bool(_tqdm_active)
# 定义一个启用进度条的函数，不返回任何值
def enable_progress_bar() -> None:
    # 函数文档字符串，说明该函数的作用是启用 tqdm 进度条
    """Enable tqdm progress bar."""
    # 声明全局变量 _tqdm_active
    global _tqdm_active
    # 将全局变量 _tqdm_active 设置为 True，表示进度条处于启用状态
    _tqdm_active = True


# 定义一个禁用进度条的函数，不返回任何值
def disable_progress_bar() -> None:
    # 函数文档字符串，说明该函数的作用是禁用 tqdm 进度条
    """Disable tqdm progress bar."""
    # 声明全局变量 _tqdm_active
    global _tqdm_active
    # 将全局变量 _tqdm_active 设置为 False，表示进度条处于禁用状态
    _tqdm_active = False
```