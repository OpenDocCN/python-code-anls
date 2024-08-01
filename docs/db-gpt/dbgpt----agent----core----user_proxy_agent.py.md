# `.\DB-GPT-src\dbgpt\agent\core\user_proxy_agent.py`

```py
"""A proxy agent for the user."""
# 引入必要的模块和类
from .base_agent import ConversableAgent
from .profile import ProfileConfig

# 定义一个用户代理类，继承自ConversableAgent类
class UserProxyAgent(ConversableAgent):
    """A proxy agent for the user.

    That can execute code and provide feedback to the other agents.
    """

    # 用户代理的配置文件，包括名称、角色和描述信息
    profile: ProfileConfig = ProfileConfig(
        name="User",
        role="Human",
        description=(
            "A human admin. Interact with the planner to discuss the plan. "
            "Plan execution needs to be approved by this admin."
        ),
    )

    # 标识该代理是否是人类用户
    is_human: bool = True
```