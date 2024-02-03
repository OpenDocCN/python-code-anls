# `.\AutoGPT\autogpts\autogpt\autogpt\agent_manager\agent_manager.py`

```py
# 导入必要的模块和类型提示
from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

# 如果是类型检查阶段，则导入AgentSettings
if TYPE_CHECKING:
    from autogpt.agents.agent import AgentSettings

# 导入AgentFileManager类
from autogpt.agents.utils.agent_file_manager import AgentFileManager

# 定义AgentManager类
class AgentManager:
    # 初始化方法，传入应用数据目录
    def __init__(self, app_data_dir: Path):
        # 设置agents_dir为应用数据目录下的"agents"目录
        self.agents_dir = app_data_dir / "agents"
        # 如果"agents"目录不存在，则创建该目录
        if not self.agents_dir.exists():
            self.agents_dir.mkdir()

    # 静态方法，生成agent的唯一ID
    @staticmethod
    def generate_id(agent_name: str) -> str:
        # 生成唯一ID
        unique_id = str(uuid.uuid4())[:8]
        return f"{agent_name}-{unique_id}"

    # 列出所有agent的名称
    def list_agents(self) -> list[str]:
        return [
            dir.name
            for dir in self.agents_dir.iterdir()
            if dir.is_dir() and AgentFileManager(dir).state_file_path.exists()
        ]

    # 获取agent的目录路径
    def get_agent_dir(self, agent_id: str, must_exist: bool = False) -> Path:
        assert len(agent_id) > 0
        agent_dir = self.agents_dir / agent_id
        # 如果必须存在且目录不存在，则抛出FileNotFoundError
        if must_exist and not agent_dir.exists():
            raise FileNotFoundError(f"No agent with ID '{agent_id}'")
        return agent_dir

    # 检索agent的状态
    def retrieve_state(self, agent_id: str) -> AgentSettings:
        # 导入AgentSettings类
        from autogpt.agents.agent import AgentSettings

        # 获取agent的目录路径
        agent_dir = self.get_agent_dir(agent_id, True)
        # 获取agent的状态文件路径
        state_file = AgentFileManager(agent_dir).state_file_path
        # 如果状态文件不存在，则抛出FileNotFoundError
        if not state_file.exists():
            raise FileNotFoundError(f"Agent with ID '{agent_id}' has no state.json")

        # 从状态文件加载AgentSettings对象
        state = AgentSettings.load_from_json_file(state_file)
        # 设置AgentSettings对象的agent_data_dir属性为agent的目录路径
        state.agent_data_dir = agent_dir
        return state
```