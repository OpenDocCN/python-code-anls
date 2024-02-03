# `.\AutoGPT\autogpts\forge\forge\sdk\forge_log.py`

```
# 导入必要的模块
import json
import logging
import logging.config
import logging.handlers
import os
import queue

# 根据环境变量设置是否启用 JSON 日志
JSON_LOGGING = os.environ.get("JSON_LOGGING", "false").lower() == "true"

# 定义自定义日志级别 CHAT
CHAT = 29
logging.addLevelName(CHAT, "CHAT")

# ANSI 控制码，用于控制终端输出颜色和样式
RESET_SEQ: str = "\033[0m"
COLOR_SEQ: str = "\033[1;%dm"
BOLD_SEQ: str = "\033[1m"
UNDERLINE_SEQ: str = "\033[04m"

# ANSI 控制码，定义不同颜色
ORANGE: str = "\033[33m"
YELLOW: str = "\033[93m"
WHITE: str = "\33[37m"
BLUE: str = "\033[34m"
LIGHT_BLUE: str = "\033[94m"
RED: str = "\033[91m"
GREY: str = "\33[90m"
GREEN: str = "\033[92m"

# 表情符号对应不同日志级别
EMOJIS: dict[str, str] = {
    "DEBUG": "🐛",
    "INFO": "📝",
    "CHAT": "💬",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "💥",
}

# 不同日志级别对应的颜色
KEYWORD_COLORS: dict[str, str] = {
    "DEBUG": WHITE,
    "INFO": LIGHT_BLUE,
    "CHAT": GREEN,
    "WARNING": YELLOW,
    "ERROR": ORANGE,
    "CRITICAL": RED,
}

# 自定义日志格式化类，将日志记录转换为 JSON 格式
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(record.__dict__)

# 格式化消息，支持语法高亮显示关键字
def formatter_message(message: str, use_color: bool = True) -> str:
    """
    Syntax highlight certain keywords
    """
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

# 格式化消息中的指定单词，添加颜色和样式
def format_word(
    message: str, word: str, color_seq: str, bold: bool = False, underline: bool = False
) -> str:
    """
    Surround the fiven word with a sequence
    """
    replacer = color_seq + word + RESET_SEQ
    if underline:
        replacer = UNDERLINE_SEQ + replacer
    if bold:
        replacer = BOLD_SEQ + replacer
    return message.replace(word, replacer)

# 控制台日志格式化类，用于给日志级别着色
class ConsoleFormatter(logging.Formatter):
    """
    This Formatted simply colors in the levelname i.e 'INFO', 'DEBUG'
    """

    def __init__(
        self, fmt: str, datefmt: str = None, style: str = "%", use_color: bool = True
    ):
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color
    # 格式化日志记录，突出显示特定关键字
    def format(self, record: logging.LogRecord) -> str:
        """
        Format and highlight certain keywords
        """
        # 将记录对象赋值给变量rec
        rec = record
        # 获取记录的日志级别名称
        levelname = rec.levelname
        # 如果使用颜色且日志级别在关键字颜色字典中
        if self.use_color and levelname in KEYWORD_COLORS:
            # 获取关键字对应的颜色，并将日志级别名称着色
            levelname_color = KEYWORD_COLORS[levelname] + levelname + RESET_SEQ
            rec.levelname = levelname_color
        # 格式化记录的名称，左对齐并添加灰色
        rec.name = f"{GREY}{rec.name:<15}{RESET_SEQ}"
        # 格式化记录的消息，添加关键字对应的颜色和表情符号
        rec.msg = (
            KEYWORD_COLORS[levelname] + EMOJIS[levelname] + "  " + rec.msg + RESET_SEQ
        )
        # 返回格式化后的日志记录
        return logging.Formatter.format(self, rec)
class ForgeLogger(logging.Logger):
    """
    This adds extra logging functions such as logger.trade and also
    sets the logger to use the custom formatter
    """

    # 定义控制台输出格式
    CONSOLE_FORMAT: str = (
        "[%(asctime)s] [$BOLD%(name)-15s$RESET] [%(levelname)-8s]\t%(message)s"
    )
    # 定义日志格式
    FORMAT: str = "%(asctime)s %(name)-15s %(levelname)-8s %(message)s"
    # 根据控制台格式生成带颜色的格式
    COLOR_FORMAT: str = formatter_message(CONSOLE_FORMAT, True)
    # 定义 JSON 格式
    JSON_FORMAT: str = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

    def __init__(self, name: str, logLevel: str = "DEBUG"):
        # 调用父类构造函数初始化 Logger
        logging.Logger.__init__(self, name, logLevel)

        # 创建队列处理器
        queue_handler = logging.handlers.QueueHandler(queue.Queue(-1))
        # 创建 JSON 格式化器
        json_formatter = logging.Formatter(self.JSON_FORMAT)
        # 设置队列处理器的格式化器
        queue_handler.setFormatter(json_formatter)
        # 将队列处理器添加到 Logger
        self.addHandler(queue_handler)

        # 根据 JSON_LOGGING 变量选择控制台格式化器
        if JSON_LOGGING:
            console_formatter = JsonFormatter()
        else:
            console_formatter = ConsoleFormatter(self.COLOR_FORMAT)
        # 创建控制台处理器
        console = logging.StreamHandler()
        # 设置控制台处理器的格式化器
        console.setFormatter(console_formatter)
        # 将控制台处理器添加到 Logger
        self.addHandler(console)
    # 定义一个方法用于处理聊天消息，接收角色、OpenAI 的响应、消息列表以及其他参数
    def chat(self, role: str, openai_repsonse: dict, messages=None, *args, **kws):
        """
        Parse the content, log the message and extract the usage into prometheus metrics
        """
        # 定义角色对应的表情符号
        role_emojis = {
            "system": "🖥️",
            "user": "👤",
            "assistant": "🤖",
            "function": "⚙️",
        }
        # 如果日志级别为 CHAT
        if self.isEnabledFor(CHAT):
            # 如果有消息列表
            if messages:
                # 遍历消息列表中的每条消息
                for message in messages:
                    # 记录日志，包括角色对应的表情符号和消息内容
                    self._log(
                        CHAT,
                        f"{role_emojis.get(message['role'], '🔵')}: {message['content']}",
                    )
            else:
                # 解析 OpenAI 的响应
                response = json.loads(openai_repsonse)
                # 记录日志，包括角色对应的表情符号和 OpenAI 响应中的消息内容
                self._log(
                    CHAT,
                    f"{role_emojis.get(role, '🔵')}: {response['choices'][0]['message']['content']}",
                )
class QueueLogger(logging.Logger):
    """
    Custom logger class with queue
    """

    def __init__(self, name: str, level: int = logging.NOTSET):
        # 调用父类的初始化方法，设置日志器的名称和级别
        super().__init__(name, level)
        # 创建一个队列处理器，将日志消息放入队列中
        queue_handler = logging.handlers.QueueHandler(queue.Queue(-1))
        # 将队列处理器添加到日志器中
        self.addHandler(queue_handler)


logging_config: dict = dict(
    version=1,
    formatters={
        "console": {
            "()": ConsoleFormatter,
            "format": ForgeLogger.COLOR_FORMAT,
        },
    },
    handlers={
        "h": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "level": logging.INFO,
        },
    },
    root={
        "handlers": ["h"],
        "level": logging.INFO,
    },
    loggers={
        "autogpt": {
            "handlers": ["h"],
            "level": logging.INFO,
            "propagate": False,
        },
    },
)


def setup_logger():
    """
    Setup the logger with the specified format
    """
    # 使用 logging 模块的 dictConfig 方法配置日志记录器
    logging.config.dictConfig(logging_config)
```