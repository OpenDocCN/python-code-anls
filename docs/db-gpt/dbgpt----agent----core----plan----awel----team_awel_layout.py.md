# `.\DB-GPT-src\dbgpt\agent\core\plan\awel\team_awel_layout.py`

```py
"""The manager of the team for the AWEL layout."""

import logging  # 导入日志模块
from abc import ABC, abstractmethod  # 导入ABC和abstractmethod装饰器
from typing import Optional, cast  # 导入类型提示相关的工具

from dbgpt._private.config import Config  # 导入Config类
from dbgpt._private.pydantic import (  # 导入pydantic相关工具
    BaseModel,  # 基础模型类
    ConfigDict,  # 配置字典类型
    Field,  # 字段定义
    model_to_dict,  # 将模型转换为字典的函数
    validator,  # 验证器装饰器
)
from dbgpt.core.awel import DAG  # 导入DAG类
from dbgpt.core.awel.dag.dag_manager import DAGManager  # 导入DAGManager类

from ...action.base import ActionOutput  # 导入ActionOutput类
from ...agent import Agent, AgentGenerateContext, AgentMessage  # 导入Agent相关类
from ...base_team import ManagerAgent  # 导入ManagerAgent类
from ...profile import DynConfig, ProfileConfig  # 导入配置相关类
from .agent_operator import AWELAgentOperator, WrappedAgentOperator  # 导入AgentOperator相关类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class AWELTeamContext(BaseModel):
    """The context of the team for the AWEL layout."""

    dag_id: str = Field(
        ...,
        description="The unique id of dag",
        examples=["flow_dag_testflow_66d8e9d6-f32e-4540-a5bd-ea0648145d0e"],
    )
    uid: str = Field(
        default=None,
        description="The unique id of flow",
        examples=["66d8e9d6-f32e-4540-a5bd-ea0648145d0e"],
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of dag",
    )
    label: Optional[str] = Field(
        default=None,
        description="The label of dag",
    )
    version: Optional[str] = Field(
        default=None,
        description="The version of dag",
    )
    description: Optional[str] = Field(
        default=None,
        description="The description of dag",
    )
    editable: bool = Field(
        default=False,
        description="is the dag is editable",
        examples=[True, False],
    )
    state: Optional[str] = Field(
        default=None,
        description="The state of dag",
    )
    user_name: Optional[str] = Field(
        default=None,
        description="The owner of current dag",
    )
    sys_code: Optional[str] = Field(
        default=None,
        description="The system code of current dag",
    )
    flow_category: Optional[str] = Field(
        default="common",
        description="The flow category of current dag",
    )

    def to_dict(self):
        """Convert the object to a dictionary."""
        return model_to_dict(self)


class AWELBaseManager(ManagerAgent, ABC):
    """AWEL base manager."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 定义一个配置字典属性

    profile: ProfileConfig = ProfileConfig(
        name="AWELBaseManager",
        role=DynConfig(
            "PlanManager", category="agent", key="dbgpt_agent_plan_awel_profile_name"
        ),
        goal=DynConfig(
            "Promote and solve user problems according to the process arranged "
            "by AWEL.",
            category="agent",
            key="dbgpt_agent_plan_awel_profile_goal",
        ),
        desc=DynConfig(
            "Promote and solve user problems according to the process arranged "
            "by AWEL.",
            category="agent",
            key="dbgpt_agent_plan_awel_profile_desc",
        ),
    )
    )

    async def _a_process_received_message(self, message: AgentMessage, sender: Agent):
        """Process the received message."""
        pass


    @abstractmethod
    def get_dag(self) -> DAG:
        """Get the DAG of the manager."""


    async def act(
        self,
        message: Optional[str],
        sender: Optional[Agent] = None,
        reviewer: Optional[Agent] = None,
        **kwargs,
    ) -> Optional[ActionOutput]:
        """Perform the action."""
        try:
            # 获取管理器的DAG（有向无环图）
            agent_dag = self.get_dag()
            # 获取DAG的叶节点，类型为AWELAgentOperator，并强制类型转换
            last_node: AWELAgentOperator = cast(
                AWELAgentOperator, agent_dag.leaf_nodes[0]
            )

            # 创建起始消息上下文，包括消息内容、发送者、审核者、内存状态、代理上下文和LLM客户端
            start_message_context: AgentGenerateContext = AgentGenerateContext(
                message=AgentMessage(content=message, current_goal=message),
                sender=sender,
                reviewer=reviewer,
                memory=self.memory.structure_clone(),
                agent_context=self.agent_context,
                llm_client=self.not_null_llm_config.llm_client,
            )
            # 调用最后一个节点，生成最终的生成上下文
            final_generate_context: AgentGenerateContext = await last_node.call(
                call_data=start_message_context
            )
            # 获取最后一条依赖消息
            last_message = final_generate_context.rely_messages[-1]

            # 获取最后一个代理
            last_agent = await last_node.get_agent(final_generate_context)
            # 如果有轮次索引，设置最后一个代理的连续自动回复计数器
            if final_generate_context.round_index is not None:
                last_agent.consecutive_auto_reply_counter = (
                    final_generate_context.round_index
                )
            # 如果没有发送者，则抛出值错误异常
            if not sender:
                raise ValueError("sender is required!")
            # 发送最后一条消息给代理
            await last_agent.send(
                last_message, sender, start_message_context.reviewer, False
            )

            # 查看消息，初始化为None
            view_message: Optional[str] = None
            # 如果最后一条消息有动作报告，则获取其中的“view”部分
            if last_message.action_report:
                view_message = last_message.action_report.get("view", None)

            # 返回动作输出对象，包括内容和视图消息
            return ActionOutput(
                content=last_message.content,
                view=view_message,
            )
        except Exception as e:
            # 记录异常日志并返回执行失败的动作输出对象
            logger.exception(f"DAG run failed!{str(e)}")

            return ActionOutput(
                is_exe_success=False,
                content=f"Failed to complete goal! {str(e)}",
            )
class WrappedAWELLayoutManager(AWELBaseManager):
    """The manager of the team for the AWEL layout.

    Receives a DAG or builds a DAG from the agents.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dag: Optional[DAG] = Field(None, description="The DAG of the manager")

    def get_dag(self) -> DAG:
        """Get the DAG of the manager."""
        # 如果已经存在 DAG 对象，则直接返回
        if self.dag:
            return self.dag
        # 从上下文中获取对话 ID
        conv_id = self.not_null_agent_context.conv_id
        last_node: Optional[WrappedAgentOperator] = None
        # 使用 DAG 对象构建上下文相关的命名
        with DAG(
            f"layout_agents_{self.not_null_agent_context.gpts_app_name}_{conv_id}"
        ) as dag:
            # 遍历管理器中的代理
            for agent in self.agents:
                now_node = WrappedAgentOperator(agent=agent)
                if not last_node:
                    last_node = now_node
                else:
                    last_node >> now_node
                    last_node = now_node
        # 将构建好的 DAG 对象保存到实例中
        self.dag = dag
        return dag

    async def act(
        self,
        message: Optional[str],
        sender: Optional[Agent] = None,
        reviewer: Optional[Agent] = None,
        **kwargs,
    ) -> Optional[ActionOutput]:
        """Perform the action."""
        try:
            # 获取当前的 DAG 对象
            dag = self.get_dag()
            # 获取 DAG 中的最后一个节点作为起始节点
            last_node: WrappedAgentOperator = cast(
                WrappedAgentOperator, dag.leaf_nodes[0]
            )
            # 构建起始消息的上下文
            start_message_context: AgentGenerateContext = AgentGenerateContext(
                message=AgentMessage(content=message, current_goal=message),
                sender=self,
                reviewer=reviewer,
            )
            # 调用最后一个节点的方法进行消息传递
            final_generate_context: AgentGenerateContext = await last_node.call(
                call_data=start_message_context
            )
            # 获取最后一条消息
            last_message = final_generate_context.rely_messages[-1]

            # 获取最后一个代理
            last_agent = last_node.agent
            # 向最后一个代理发送消息
            await last_agent.send(
                last_message,
                self,
                start_message_context.reviewer,
                False,
            )

            # 查看消息动作报告中的视图消息
            view_message: Optional[str] = None
            if last_message.action_report:
                view_message = last_message.action_report.get("view", None)

            # 返回动作的输出结果
            return ActionOutput(
                content=last_message.content,
                view=view_message,
            )
        except Exception as e:
            # 记录异常情况
            logger.exception(f"DAG run failed!{str(e)}")

            # 返回执行失败的输出结果
            return ActionOutput(
                is_exe_success=False,
                content=f"Failed to complete goal! {str(e)}",
            )


class DefaultAWELLayoutManager(AWELBaseManager):
    """The manager of the team for the AWEL layout."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dag: AWELTeamContext = Field(...)

    @validator("dag")
    def check_dag(cls, value):
        """Check the DAG of the manager."""
        # 验证 DAG 对象不能为空
        assert value is not None and value != "", "dag must not be empty"
        return value
    # 定义一个方法，返回当前管理器的DAG（有向无环图）
    def get_dag(self) -> DAG:
        """Get the DAG of the manager."""
        # 创建配置对象
        cfg = Config()
        # 获取DAG管理器的单例实例，使用系统应用的配置作为参数
        _dag_manager = DAGManager.get_instance(cfg.SYSTEM_APP)  # type: ignore
        # 根据当前实例的dag.uid属性获取特定别名的DAG对象
        agent_dag: Optional[DAG] = _dag_manager.get_dag(alias_name=self.dag.uid)
        # 如果agent_dag为None，抛出数值错误异常，指明配置的流程未找到
        if agent_dag is None:
            raise ValueError(f"The configured flow cannot be found! [{self.dag.name}]")
        # 返回获取到的DAG对象
        return agent_dag
```