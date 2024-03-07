# `.\PokeLLMon\poke_env\__init__.py`

```
"""poke_env module init.
"""
# 导入日志模块
import logging
# 导入环境模块
import poke_env.environment as environment
# 导入异常模块
import poke_env.exceptions as exceptions
# 导入玩家模块
import poke_env.player as player
# 导入 PS 客户端模块
import poke_env.ps_client as ps_client
# 导入统计模块
import poke_env.stats as stats
# 导入队伍构建模块
import poke_env.teambuilder as teambuilder
# 导入生成数据和转换为 ID 字符串的函数
from poke_env.data import gen_data, to_id_str
# 导入 Showdown 异常
from poke_env.exceptions import ShowdownException
# 导入账户配置
from poke_env.ps_client import AccountConfiguration
# 导入服务器配置
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
# 导入计算原始统计的函数
from poke_env.stats import compute_raw_stats

# 获取 logger 对象
__logger = logging.getLogger("poke-env")
# 创建流处理器
__stream_handler = logging.StreamHandler()
# 创建格式化器
__formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# 设置格式化器
__stream_handler.setFormatter(__formatter)
# 将流处理器添加到 logger
__logger.addHandler(__stream_handler)
# 添加日志级别名称
logging.addLevelName(25, "PS_ERROR")

# 导出的模块列表
__all__ = [
    "AccountConfiguration",
    "LocalhostServerConfiguration",
    "ServerConfiguration",
    "ShowdownException",
    "ShowdownServerConfiguration",
    "compute_raw_stats",
    "environment",
    "exceptions",
    "gen_data",
    "player",
    "ps_client",
    "stats",
    "teambuilder",
    "to_id_str",
]
```