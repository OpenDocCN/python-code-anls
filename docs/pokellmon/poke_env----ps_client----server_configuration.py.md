# `.\PokeLLMon\poke_env\ps_client\server_configuration.py`

```
# 该模块包含与服务器配置相关的对象
from typing import NamedTuple

# 定义一个名为ServerConfiguration的命名元组，表示服务器配置对象，包含两个条目：服务器URL和认证端点URL
class ServerConfiguration(NamedTuple):
    server_url: str  # 服务器URL
    authentication_url: str  # 认证端点URL

# 使用本地主机和smogon的认证端点创建一个名为LocalhostServerConfiguration的ServerConfiguration对象
LocalhostServerConfiguration = ServerConfiguration(
    "localhost:8000", "https://play.pokemonshowdown.com/action.php?"
)

# 使用smogon的服务器和认证端点创建一个名为ShowdownServerConfiguration的ServerConfiguration对象
ShowdownServerConfiguration = ServerConfiguration(
    "sim.smogon.com:8000", "https://play.pokemonshowdown.com/action.php?"
)
```