# `.\AutoGPT\autogpts\autogpt\tests\integration\agent_factory.py`

```py
# 导入 pytest 模块
import pytest

# 导入 Agent、AgentConfiguration、AgentSettings 类
from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
# 导入 AIProfile、Config 类
from autogpt.config import AIProfile, Config
# 导入 get_memory 函数
from autogpt.memory.vector import get_memory
# 导入 CommandRegistry 类
from autogpt.models.command_registry import CommandRegistry

# 定义 fixture，用于设置 memory_backend 为 "json_file"，并清空内存
@pytest.fixture
def memory_json_file(config: Config):
    # 保存原始的 memory_backend
    was_memory_backend = config.memory_backend

    # 设置 memory_backend 为 "json_file"
    config.memory_backend = "json_file"
    # 获取内存对象
    memory = get_memory(config)
    # 清空内存
    memory.clear()
    # 返回内存对象
    yield memory

    # 恢复原始的 memory_backend
    config.memory_backend = was_memory_backend

# 定义 fixture，创建一个 dummy agent
@pytest.fixture
def dummy_agent(config: Config, llm_provider, memory_json_file):
    # 创建一个 CommandRegistry 对象
    command_registry = CommandRegistry()

    # 创建一个 AIProfile 对象
    ai_profile = AIProfile(
        ai_name="Dummy Agent",
        ai_role="Dummy Role",
        ai_goals=[
            "Dummy Task",
        ],
    )

    # 复制 Agent 的默认设置
    agent_prompt_config = Agent.default_settings.prompt_config.copy(deep=True)
    # 根据配置设置是否使用 functions api
    agent_prompt_config.use_functions_api = config.openai_functions
    # 创建 AgentSettings 对象
    agent_settings = AgentSettings(
        name=Agent.default_settings.name,
        description=Agent.default_settings.description,
        ai_profile=ai_profile,
        config=AgentConfiguration(
            fast_llm=config.fast_llm,
            smart_llm=config.smart_llm,
            use_functions_api=config.openai_functions,
            plugins=config.plugins,
        ),
        prompt_config=agent_prompt_config,
        history=Agent.default_settings.history.copy(deep=True),
    )

    # 创建 Agent 对象
    agent = Agent(
        settings=agent_settings,
        llm_provider=llm_provider,
        command_registry=command_registry,
        legacy_config=config,
    )

    # 返回 Agent 对象
    return agent
```