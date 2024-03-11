# `.\PokeLLMon\poke_env\environment\weather.py`

```py
"""This module defines the Weather class, which represents a in-battle weather.
"""
# 导入日志记录模块
import logging
# 导入枚举模块
from enum import Enum, auto, unique

# 定义 Weather 枚举类
@unique
class Weather(Enum):
    """Enumeration, represent a non null weather in a battle."""

    # 枚举值
    UNKNOWN = auto()
    DESOLATELAND = auto()
    DELTASTREAM = auto()
    HAIL = auto()
    PRIMORDIALSEA = auto()
    RAINDANCE = auto()
    SANDSTORM = auto()
    SNOW = auto()
    SUNNYDAY = auto()

    # 返回对象的字符串表示
    def __str__(self) -> str:
        return f"{self.name} (weather) object"

    # 根据 Showdown 消息返回对应的 Weather 对象
    @staticmethod
    def from_showdown_message(message: str):
        """Returns the Weather object corresponding to the message.

        :param message: The message to convert.
        :type message: str
        :return: The corresponding Weather object.
        :rtype: Weather
        """
        # 处理消息字符串
        message = message.replace("move: ", "")
        message = message.replace(" ", "_")
        message = message.replace("-", "_")

        try:
            # 尝试从枚举中获取对应的 Weather 对象
            return Weather[message.upper()]
        except KeyError:
            # 如果未找到对应的 Weather 对象，则记录警告并返回 UNKNOWN
            logging.getLogger("poke-env").warning(
                "Unexpected weather '%s' received. Weather.UNKNOWN will be used "
                "instead. If this is unexpected, please open an issue at "
                "https://github.com/hsahovic/poke-env/issues/ along with this error "
                "message and a description of your program.",
                message,
            )
            return Weather.UNKNOWN
```