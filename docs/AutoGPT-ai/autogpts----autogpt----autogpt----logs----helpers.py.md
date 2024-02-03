# `.\AutoGPT\autogpts\autogpt\autogpt\logs\helpers.py`

```py
# 导入 logging 模块
import logging
# 导入 Any 和 Optional 类型提示
from typing import Any, Optional
# 导入 colorama 模块中的 Fore 类
from colorama import Fore
# 从当前目录下的 config 模块中导入 SPEECH_OUTPUT_LOGGER, USER_FRIENDLY_OUTPUT_LOGGER 和 _chat_plugins
from .config import SPEECH_OUTPUT_LOGGER, USER_FRIENDLY_OUTPUT_LOGGER, _chat_plugins

# 定义一个函数，以用户友好的方式向用户输出消息
def user_friendly_output(
    message: str,
    level: int = logging.INFO,
    title: str = "",
    title_color: str = "",
    preserve_message_color: bool = False,
) -> None:
    """Outputs a message to the user in a user-friendly way.

    This function outputs on up to two channels:
    1. The console, in typewriter style
    2. Text To Speech, if configured
    """
    # 获取 USER_FRIENDLY_OUTPUT_LOGGER 对应的 logger
    logger = logging.getLogger(USER_FRIENDLY_OUTPUT_LOGGER)

    # 如果存在 _chat_plugins
    if _chat_plugins:
        # 遍历 _chat_plugins
        for plugin in _chat_plugins:
            # 调用 plugin 的 report 方法，输出消息
            plugin.report(f"{title}: {message}")

    # 使用 logger 输出消息
    logger.log(
        level,
        message,
        extra={
            "title": title,
            "title_color": title_color,
            "preserve_color": preserve_message_color,
        },
    )

# 定义一个函数，打印属性的标题和值
def print_attribute(
    title: str, value: Any, title_color: str = Fore.GREEN, value_color: str = ""
) -> None:
    # 获取默认 logger
    logger = logging.getLogger()
    # 使用 logger 输出属性值
    logger.info(
        str(value),
        extra={
            "title": f"{title.rstrip(':')}:",
            "title_color": title_color,
            "color": value_color,
        },
    )

# 定义一个函数，请求用户双重检查配置
def request_user_double_check(additionalText: Optional[str] = None) -> None:
    # 如果没有提供 additionalText，则使用默认提示信息
    if not additionalText:
        additionalText = (
            "Please ensure you've setup and configured everything correctly. "
            "Read https://docs.agpt.co/autogpt/setup/ to double check. "
            "You can also create a github issue or join the discord and ask there!"
        )

    # 调用 user_friendly_output 函数输出消息
    user_friendly_output(
        additionalText,
        level=logging.WARN,
        title="DOUBLE CHECK CONFIGURATION",
        preserve_message_color=True,
    )

# 定义一个函数，以语音的方式输出消息
def speak(message: str, level: int = logging.INFO) -> None:
    # 获取 SPEECH_OUTPUT_LOGGER 对应的 logger，并输出消息
    logging.getLogger(SPEECH_OUTPUT_LOGGER).log(level, message)
```