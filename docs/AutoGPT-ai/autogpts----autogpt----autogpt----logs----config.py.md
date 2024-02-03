# `.\AutoGPT\autogpts\autogpt\autogpt\logs\config.py`

```py
"""Logging module for Auto-GPT."""
# 导入必要的模块和库
from __future__ import annotations

import enum
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from auto_gpt_plugin_template import AutoGPTPluginTemplate
from openai._base_client import log as openai_logger

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.speech import TTSConfig

# 导入自定义模块
from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.runner.client_lib.logging import BelowLevelFilter

# 导入自定义格式化和处理模块
from .formatters import AutoGptFormatter, StructuredLoggingFormatter
from .handlers import TTSHandler, TypingConsoleHandler

# 定义日志文件路径
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = "activity.log"
DEBUG_LOG_FILE = "debug.log"
ERROR_LOG_FILE = "error.log"

# 定义简单和调试日志格式
SIMPLE_LOG_FORMAT = "%(asctime)s %(levelname)s  %(title)s%(message)s"
DEBUG_LOG_FORMAT = (
    "%(asctime)s %(levelname)s %(filename)s:%(lineno)d" "  %(title)s%(message)s"
)

# 定义特定日志记录器名称
SPEECH_OUTPUT_LOGGER = "VOICE"
USER_FRIENDLY_OUTPUT_LOGGER = "USER_FRIENDLY_OUTPUT"

# 初始化插件列表
_chat_plugins: list[AutoGPTPluginTemplate] = []

# 定义日志格式名称枚举
class LogFormatName(str, enum.Enum):
    SIMPLE = "simple"
    DEBUG = "debug"
    STRUCTURED = "structured_google_cloud"

# 定义不同日志格式对应的格式字符串
TEXT_LOG_FORMAT_MAP = {
    LogFormatName.DEBUG: DEBUG_LOG_FORMAT,
    LogFormatName.SIMPLE: SIMPLE_LOG_FORMAT,
}

# 日志配置类，继承自系统配置类
class LoggingConfig(SystemConfiguration):
    # 日志级别，默认为INFO
    level: int = UserConfigurable(
        default=logging.INFO,
        from_env=lambda: logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")),
    )

    # 控制台输出配置
    log_format: LogFormatName = UserConfigurable(
        default=LogFormatName.SIMPLE,
        from_env=lambda: LogFormatName(os.getenv("LOG_FORMAT", "simple")),
    )
    plain_console_output: bool = UserConfigurable(
        default=False,
        from_env=lambda: os.getenv("PLAIN_OUTPUT", "False") == "True",
    )

    # 文件输出配置
    log_dir: Path = LOG_DIR
    # 定义一个可选的日志格式名称变量，初始值为 LogFormatName.SIMPLE
    log_file_format: Optional[LogFormatName] = UserConfigurable(
        # 设置默认值为 LogFormatName.SIMPLE，并从环境变量中获取 LOG_FILE_FORMAT 或 LOG_FORMAT 的值
        default=LogFormatName.SIMPLE,
        from_env=lambda: LogFormatName(
            os.getenv("LOG_FILE_FORMAT", os.getenv("LOG_FORMAT", "simple"))
        ),
    )
# 配置本地日志模块的函数
def configure_logging(
    # 设置日志级别，默认为 INFO
    level: int = logging.INFO,
    # 设置日志目录，默认为 LOG_DIR
    log_dir: Path = LOG_DIR,
    # 设置日志格式，可选参数，默认为 None
    log_format: Optional[LogFormatName] = None,
    # 设置日志文件格式，可选参数，默认为 None
    log_file_format: Optional[LogFormatName] = None,
    # 是否在控制台输出简单文本，默认为 False
    plain_console_output: bool = False,
    # TTS 配置，可选参数，默认为 None
    tts_config: Optional[TTSConfig] = None,
) -> None:
    """Configure the native logging module.

    Should be usable as `configure_logging(**config.logging.dict())`, where
    `config.logging` is a `LoggingConfig` object.
    """

    # 根据日志级别自动调整默认日志格式
    log_format = log_format or (
        LogFormatName.SIMPLE if level != logging.DEBUG else LogFormatName.DEBUG
    )
    log_file_format = log_file_format or log_format

    # 是否使用结构化日志
    structured_logging = log_format == LogFormatName.STRUCTURED

    if structured_logging:
        plain_console_output = True
        log_file_format = None

    # 如果日志目录不存在，则创建
    if not log_dir.exists():
        log_dir.mkdir()

    # 日志处理器列表
    log_handlers: list[logging.Handler] = []

    if log_format in (LogFormatName.DEBUG, LogFormatName.SIMPLE):
        # 根据日志格式选择控制台输出格式
        console_format_template = TEXT_LOG_FORMAT_MAP[log_format]
        console_formatter = AutoGptFormatter(console_format_template)
    else:
        console_formatter = StructuredLoggingFormatter()
        console_format_template = SIMPLE_LOG_FORMAT

    # 控制台输出处理器
    stdout = logging.StreamHandler(stream=sys.stdout)
    stdout.setLevel(level)
    stdout.addFilter(BelowLevelFilter(logging.WARNING))
    stdout.setFormatter(console_formatter)
    stderr = logging.StreamHandler()
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(console_formatter)
    log_handlers += [stdout, stderr]

    # 模拟打字效果的控制台输出处理器
    typing_console_handler = TypingConsoleHandler(stream=sys.stdout)
    typing_console_handler.setLevel(logging.INFO)
    typing_console_handler.setFormatter(console_formatter)

    # 用户友好输出日志器（文本 + 语音）
    # 获取用户友好输出的日志记录器对象
    user_friendly_output_logger = logging.getLogger(USER_FRIENDLY_OUTPUT_LOGGER)
    # 设置用户友好输出的日志级别为 INFO
    user_friendly_output_logger.setLevel(logging.INFO)
    # 添加控制台处理器到用户友好输出的日志记录器
    user_friendly_output_logger.addHandler(
        typing_console_handler if not plain_console_output else stdout
    )
    # 如果存在 TTS 配置，则添加 TTS 处理器到用户友好输出的日志记录器
    if tts_config:
        user_friendly_output_logger.addHandler(TTSHandler(tts_config))
    # 添加标准错误输出处理器到用户友好输出的日志记录器
    user_friendly_output_logger.addHandler(stderr)
    # 禁止用户友好输出的日志记录器向上传播
    user_friendly_output_logger.propagate = False

    # 文件输出处理器
    if log_file_format is not None:
        # 如果日志级别小于 ERROR，则使用指定格式的日志模板
        if level < logging.ERROR:
            file_output_format_template = TEXT_LOG_FORMAT_MAP[log_file_format]
            # 创建自动生成的日志格式化器
            file_output_formatter = AutoGptFormatter(
                file_output_format_template, no_color=True
            )

            # INFO 日志文件处理器
            activity_log_handler = logging.FileHandler(log_dir / LOG_FILE, "a", "utf-8")
            activity_log_handler.setLevel(level)
            activity_log_handler.setFormatter(file_output_formatter)
            log_handlers += [activity_log_handler]
            user_friendly_output_logger.addHandler(activity_log_handler)

        # ERROR 日志文件处理器
        error_log_handler = logging.FileHandler(log_dir / ERROR_LOG_FILE, "a", "utf-8")
        error_log_handler.setLevel(logging.ERROR)
        error_log_handler.setFormatter(
            AutoGptFormatter(DEBUG_LOG_FORMAT, no_color=True)
        )
        log_handlers += [error_log_handler]
        user_friendly_output_logger.addHandler(error_log_handler)

    # 配置根日志记录器
    logging.basicConfig(
        format=console_format_template,
        level=level,
        handlers=log_handlers,
    )

    # 语音输出日志记录器
    speech_output_logger = logging.getLogger(SPEECH_OUTPUT_LOGGER)
    # 设置语音输出日志级别为 INFO
    speech_output_logger.setLevel(logging.INFO)
    # 如果存在 TTS 配置，则添加 TTS 处理器到语音输出日志记录器
    if tts_config:
        speech_output_logger.addHandler(TTSHandler(tts_config))
    # 禁止语音输出日志记录器向上传播
    speech_output_logger.propagate = False

    # 具有更好格式的 JSON 日志记录器
    # 创建名为"JSON_LOGGER"的日志记录器对象
    json_logger = logging.getLogger("JSON_LOGGER")
    # 设置"JSON_LOGGER"的日志级别为DEBUG
    json_logger.setLevel(logging.DEBUG)
    # 禁止"JSON_LOGGER"的日志传播到父记录器
    json_logger.propagate = False
    
    # 禁止OpenAI库的debug级别日志记录
    openai_logger.setLevel(logging.WARNING)
# 配置聊天插件以供日志模块使用
def configure_chat_plugins(config: Config) -> None:
    """Configure chat plugins for use by the logging module"""

    # 获取当前模块的日志记录器
    logger = logging.getLogger(__name__)

    # 如果启用了聊天消息功能
    if config.chat_messages_enabled:
        # 如果已经存在聊天插件列表，则清空
        if _chat_plugins:
            _chat_plugins.clear()

        # 遍历配置中的插件列表
        for plugin in config.plugins:
            # 检查插件是否具有处理报告的能力
            if hasattr(plugin, "can_handle_report") and plugin.can_handle_report():
                # 将插件添加到日志记录器中
                logger.debug(f"Loaded plugin into logger: {plugin.__class__.__name__}")
                _chat_plugins.append(plugin)
```