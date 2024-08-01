# `.\DB-GPT-src\dbgpt\agent\core\base_team.py`

```py
"""
Base classes for managing a group of agents in a team chat.
"""

# 引入日志模块
import logging
# 引入类型提示模块
from typing import Any, Dict, List, Optional, Tuple, Union
# 引入 Pydantic 相关模块
from dbgpt._private.pydantic import BaseModel, ConfigDict, Field
# 引入自定义模块
from .action.base import ActionOutput
from .agent import Agent, AgentMessage
from .base_agent import ConversableAgent
from .profile import ProfileConfig

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 定义一个私有函数，将内容转换为字符串格式
def _content_str(content: Union[str, List, None]) -> str:
    """Convert content into a string format.

    This function processes content that may be a string, a list of mixed text and
    image URLs, or None, and converts it into a string. Text is directly appended to
    the result string, while image URLs are represented by a placeholder image token.
    If the content is None, an empty string is returned.

    Args:
        content (Union[str, List, None]): The content to be processed. Can be a
            string, a list of dictionaries representing text and image URLs, or None.

    Returns:
        str: A string representation of the input content. Image URLs are replaced with
            an image token.

    Raises:
        TypeError: If content is not of type None, str, or list.
        TypeError: If content list contains elements that are not dictionaries.
        ValueError: If an unknown type is encountered in the content dictionaries.

    Note:
    - The function expects each dictionary in the list to have a "type" key that is
        either "text" or "image_url".
        For "text" type, the "text" key's value is appended to the result.
        For "image_url", an image token is appended.
    - This function is useful for handling content that may include both text and image
        references, especially in contexts where images need to be represented as
        placeholders.
    """
    # 如果 content 为 None，则返回空字符串
    if content is None:
        return ""
    # 如果 content 是字符串，则直接返回
    if isinstance(content, str):
        return content
    # 如果 content 不是列表，则抛出类型错误
    if not isinstance(content, list):
        raise TypeError(f"content must be None, str, or list, but got {type(content)}")

    rst = ""
    # 遍历 content 中的每个元素
    for item in content:
        # 如果元素不是字典，则抛出类型错误
        if not isinstance(item, dict):
            raise TypeError(
                "Wrong content format: every element should be dict if the content is "
                "a list."
            )
        # 断言字典中必须有 'type' 键
        assert (
            "type" in item
        ), "Wrong content format. Missing 'type' key in content's dict."
        # 根据 'type' 键的值进行处理
        if item["type"] == "text":
            rst += item["text"]
        elif item["type"] == "image_url":
            rst += "<image>"
        else:
            # 如果 'type' 键的值不是 'text' 或者 'image_url'，则抛出值错误
            raise ValueError(
                f"Wrong content format: unknown type {item['type']} within the content"
            )
    return rst


# 定义一个团队类，用于管理团队聊天中的代理
class Team(BaseModel):
    """Team class for managing a group of agents in a team chat."""

    # 模型配置字典
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 代理列表，默认为空列表
    agents: List[Agent] = Field(default_factory=list)
    # 消息列表，默认为空列表
    messages: List[Dict] = Field(default_factory=list)
    # 最大回合数，默认为 100
    max_round: int = 100
    # 是否为团队，默认为 True
    is_team: bool = True

    def __init__(self, **kwargs):
        """Create a new Team instance."""
        super().__init__(**kwargs)
    # 将传入的一组代理（Agent）添加到当前对象的代理列表中
    def hire(self, agents: List[Agent]):
        """Hire roles to cooperate."""
        self.agents.extend(agents)

    # 返回当前对象代理列表中所有代理的角色名组成的列表
    @property
    def agent_names(self) -> List[str]:
        """Return the names of the agents in the group chat."""
        return [agent.role for agent in self.agents]

    # 根据给定的名字查找并返回对应的代理对象
    def agent_by_name(self, name: str) -> Agent:
        """Return the agent with a given name."""
        return self.agents[self.agent_names.index(name)]

    # 从一组代理中选择下一个发言者，并返回该代理及可能的目标上下文
    async def select_speaker(
        self,
        last_speaker: Agent,
        selector: Agent,
        now_goal_context: Optional[str] = None,
        pre_allocated: Optional[str] = None,
    ) -> Tuple[Agent, Optional[str]]:
        """Select the next speaker in the group chat."""
        raise NotImplementedError

    # 清空当前对象中的消息列表，重置群聊状态
    def reset(self):
        """Reset the group chat."""
        self.messages.clear()

    # 将一个消息字典追加到当前对象的消息列表中
    def append(self, message: Dict):
        """Append a message to the group chat.

        We cast the content to str here so that it can be managed by text-based
        model.
        """
        # 将消息内容转换为字符串，以便文本模型可以处理
        message["content"] = _content_str(message["content"])
        self.messages.append(message)
# ManagerAgent 类，继承自 ConversableAgent 和 Team
class ManagerAgent(ConversableAgent, Team):
    """Manager Agent class."""

    # 模型配置，允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 管理代理的配置文件
    profile: ProfileConfig = ProfileConfig(
        name="ManagerAgent",
        profile="TeamManager",
        goal="manage all hired intelligent agents to complete mission objectives",
        constraints=[],
        desc="manage all hired intelligent agents to complete mission objectives",
    )

    # 是否为团队类型
    is_team: bool = True

    # 最大重试次数，管理代理不需要重试异常，因为实际执行已经重试
    max_retry_count: int = 1

    def __init__(self, **kwargs):
        """Create a new ManagerAgent instance."""
        # 初始化函数，调用 ConversableAgent 和 Team 的初始化函数
        ConversableAgent.__init__(self, **kwargs)
        Team.__init__(self, **kwargs)

    async def thinking(
        self, messages: List[AgentMessage], prompt: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """Think and reason about the current task goal."""
        # 思考函数，根据当前任务目标进行思考和推理
        if messages is None or len(messages) <= 0:
            return None, None
        else:
            message = messages[-1]
            self.messages.append(message.to_llm_message())
            return message.content, None

    async def _load_thinking_messages(
        self,
        received_message: AgentMessage,
        sender: Agent,
        rely_messages: Optional[List[AgentMessage]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[AgentMessage]:
        """Load messages for thinking."""
        # 加载用于思考的消息
        return [AgentMessage(content=received_message.content)]

    async def act(
        self,
        message: Optional[str],
        sender: Optional[Agent] = None,
        reviewer: Optional[Agent] = None,
        **kwargs,
    ) -> Optional[ActionOutput]:
        """Perform actions based on the received message."""
        # 根据接收到的消息执行动作
        return None
```