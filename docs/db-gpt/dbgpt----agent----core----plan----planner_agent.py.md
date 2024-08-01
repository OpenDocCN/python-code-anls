# `.\DB-GPT-src\dbgpt\agent\core\plan\planner_agent.py`

```py
"""Planner Agent."""

from typing import Any, Dict, List, Optional  # 导入必要的模块和类

from dbgpt._private.pydantic import Field  # 导入字段类

from ...resource.pack import ResourcePack  # 导入资源包类
from ..agent import AgentMessage  # 导入代理消息类
from ..base_agent import ConversableAgent  # 导入可对话代理类
from ..plan.plan_action import PlanAction  # 导入计划动作类
from ..profile import DynConfig, ProfileConfig  # 导入动态配置和配置文件类


class PlannerAgent(ConversableAgent):
    """Planner Agent.

    Planner agent, realizing task goal planning decomposition through LLM.
    """

    agents: List[ConversableAgent] = Field(default_factory=list)  # 初始化代理列表字段

    user: help me build a sales report summarizing our key metrics and trends
    assistants: [
        {{
            "serial_number": "1",
            "agent": "DataScientist",
            "content": "Retrieve total sales, average sales, and number of transactions grouped by "product_category"'.",
            "rely": ""
        }},
        {{
            "serial_number": "2",
            "agent": "DataScientist",
            "content": "Retrieve monthly sales and transaction number trends.",
            "rely": ""
        }},
        {{
            "serial_number": "3",
            "agent": "Reporter",
            "content": "Integrate analytical data into the format required to build sales reports.",
            "rely": "1,2"
        }}
    ]""",  # noqa: E501
            category="agent",
            key="dbgpt_agent_plan_planner_agent_profile_examples",
        ),  # 定义用户问题及助手要执行的任务列表
    )  # noqa: E501

    _goal_zh: str = (
        "理解下面每个智能体(agent)和他们的能力，使用给出的资源，通过协调智能体来解决"
        "用户问题。 请发挥你LLM的知识和理解能力，理解用户问题的意图和目标，生成一个可以在没有用户帮助"
        "下，由智能体协作完成目标的任务计划。"
    )  # 定义中文版的任务目标描述

    _expand_prompt_zh: str = "可用智能体(agent):\n {{ agents }}"  # 定义中文版的智能体展示提示

    _constraints_zh: List[str] = [
        "任务计划的每个步骤都应该是为了推进解决用户目标而存在，不要生成无意义的任务步骤，确保每个步骤内目标明确内容完整。",
        "关注任务计划每个步骤的依赖关系和逻辑，被依赖步骤要考虑被依赖的数据，是否能基于当前目标得到，如果不能请在目标中提示要生成被依赖数据。",
        "每个步骤都是一个独立可完成的目标，一定要确保逻辑和信息完整，不要出现类似:"
        "'Analyze the retrieved issues data'这样目标不明确，不知道具体要分析啥内容的步骤",
        "请确保只使用上面提到的智能体，并且可以只使用其中需要的部分，严格根据描述能力和限制分配给合适的步骤，每个智能体都可以重复使用。",
        "根据用户目标的实际需要使用提供的资源来协助生成计划步骤，不要使用不需要的资源。",
        "每个步骤最好只使用一种资源完成一个子目标，如果当前目标可以分解为同类型的多个子任务，可以生成相互不依赖的并行任务。",
        "数据资源可以被合适的智能体加载使用，不用考虑数据资源的加载链接问题",
        "尽量合并有顺序依赖的连续相同步骤,如果用户目标无拆分必要，可以生成内容为用户目标的单步任务。",
        "仔细检查计划，确保计划完整的包含了用户问题所涉及的所有信息，并且最终能完成目标，确认每个步骤是否包含了需要用到的资源信息,如URL、资源名等. ",
    ]  # 定义中文版的约束条件列表

    _desc_zh: str = "你是一个任务规划专家！可以协调智能体，分配资源完成复杂的任务目标。"  # 定义中文版的描述信息

    def __init__(self, **kwargs):
        """Create a new PlannerAgent instance."""
        super().__init__(**kwargs)  # 调用父类的初始化方法
        self._init_actions([PlanAction])  # 初始化计划动作

    def _init_reply_message(self, received_message: AgentMessage):
        reply_message = super()._init_reply_message(received_message)  # 调用父类的消息初始化方法
        reply_message.context = {
            "agents": "\n".join([f"- {item.role}:{item.desc}" for item in self.agents]),  # 格式化并设置回复消息的上下文
        }
        return reply_message  # 返回回复消息对象
    # 将给定的代理列表绑定到规划代理上
    def bind_agents(self, agents: List[ConversableAgent]) -> ConversableAgent:
        """Bind the agents to the planner agent."""
        # 将参数中的代理列表赋值给当前对象的 agents 属性
        self.agents = agents
        # 初始化一个空列表，用于存储每个代理的资源
        resources = []
        # 遍历每个代理，如果代理有资源，则将其资源添加到 resources 列表中
        for agent in self.agents:
            if agent.resource:
                resources.append(agent.resource)
        # 将收集到的资源列表封装成 ResourcePack 对象，赋值给当前对象的 resource 属性
        self.resource = ResourcePack(resources)
        # 返回当前对象自身，用于方法链式调用
        return self

    # 异步生成资源变量的方法，可选参数为问题描述字符串，返回包含生成的资源变量的字典
    async def generate_resource_variables(
        self, question: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate the resource variables."""
        # 初始化一个可选字符串变量，用于存储输出模式的值，默认为 None
        out_schema: Optional[str] = None
        # 如果当前对象的 actions 属性存在且非空，则将第一个 action 的输出模式赋值给 out_schema
        if self.actions and len(self.actions) > 0:
            out_schema = self.actions[0].ai_out_schema
        # 返回包含 out_schema 变量的字典
        return {"out_schema": out_schema}

    # 准备 act 方法的参数字典，返回包含上下文和记忆计划的字典
    def prepare_act_param(self) -> Dict[str, Any]:
        """Prepare the parameters for the act method."""
        # 返回包含当前对象的非空代理上下文和记忆计划的字典
        return {
            "context": self.not_null_agent_context,
            "plans_memory": self.memory.plans_memory,
        }
```