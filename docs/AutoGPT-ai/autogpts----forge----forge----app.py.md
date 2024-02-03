# `.\AutoGPT\autogpts\forge\forge\app.py`

```py
# 导入 os 模块
import os

# 从 forge.agent 模块中导入 ForgeAgent 类
from forge.agent import ForgeAgent
# 从 forge.sdk 模块中导入 LocalWorkspace 类
from forge.sdk import LocalWorkspace

# 从当前目录下的 db 模块中导入 ForgeDatabase 类
from .db import ForgeDatabase

# 从环境变量中获取数据库名称
database_name = os.getenv("DATABASE_STRING")
# 根据环境变量中的代理工作空间路径创建本地工作空间对象
workspace = LocalWorkspace(os.getenv("AGENT_WORKSPACE"))
# 创建 ForgeDatabase 对象，传入数据库名称和关闭调试模式
database = ForgeDatabase(database_name, debug_enabled=False)
# 创建 ForgeAgent 对象，传入数据库和工作空间对象
agent = ForgeAgent(database=database, workspace=workspace)

# 从代理对象中获取代理应用程序对象
app = agent.get_agent_app()
```