# `.\PokeLLMon\poke_env\environment\side_condition.py`

```py
"""This module defines the SideCondition class, which represents a in-battle side
condition.
"""
# 导入日志记录模块
import logging
# 导入枚举模块
from enum import Enum, auto, unique

# 定义一个枚举类，表示战斗中的一侧状态
@unique
class SideCondition(Enum):
    """Enumeration, represent a in-battle side condition."""

    UNKNOWN = auto()
    AURORA_VEIL = auto()
    FIRE_PLEDGE = auto()
    G_MAX_CANNONADE = auto()
    G_MAX_STEELSURGE = auto()
    G_MAX_VINE_LASH = auto()
    G_MAX_VOLCALITH = auto()
    G_MAX_WILDFIRE = auto()
    GRASS_PLEDGE = auto()
    LIGHT_SCREEN = auto()
    LUCKY_CHANT = auto()
    MIST = auto()
    REFLECT = auto()
    SAFEGUARD = auto()
    SPIKES = auto()
    STEALTH_ROCK = auto()
    STICKY_WEB = auto()
    TAILWIND = auto()
    TOXIC_SPIKES = auto()
    WATER_PLEDGE = auto()

    # 返回对象的字符串表示形式
    def __str__(self) -> str:
        return f"{self.name} (side condition) object"

    @staticmethod
    def from_showdown_message(message: str):
        """Returns the SideCondition object corresponding to the message.

        :param message: The message to convert.
        :type message: str
        :return: The corresponding SideCondition object.
        :rtype: SideCondition
        """
        # 替换消息中的特定字符
        message = message.replace("move: ", "")
        message = message.replace(" ", "_")
        message = message.replace("-", "_")

        try:
            # 尝试返回对应的 SideCondition 对象
            return SideCondition[message.upper()]
        except KeyError:
            # 如果未找到对应的对象，则记录警告信息
            logging.getLogger("poke-env").warning(
                "Unexpected side condition '%s' received. SideCondition.UNKNOWN will be"
                " used instead. If this is unexpected, please open an issue at "
                "https://github.com/hsahovic/poke-env/issues/ along with this error "
                "message and a description of your program.",
                message,
            )
            return SideCondition.UNKNOWN


# SideCondition -> Max useful stack level
# 定义可叠加状态及其最大叠加层数的字典
STACKABLE_CONDITIONS = {SideCondition.SPIKES: 3, SideCondition.TOXIC_SPIKES: 2}
```