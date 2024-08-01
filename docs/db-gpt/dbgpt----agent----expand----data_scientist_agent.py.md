# `.\DB-GPT-src\dbgpt\agent\expand\data_scientist_agent.py`

```py
"""Data Scientist Agent."""

# 导入所需模块和类
import json
import logging
from typing import List, Optional, Tuple, cast

# 导入Agent相关模块和类
from ..core.action.base import ActionOutput
from ..core.agent import AgentMessage
from ..core.base_agent import ConversableAgent
from ..core.profile import DynConfig, ProfileConfig
from ..resource.database import DBResource
from .actions.chart_action import ChartAction

# 设置日志记录器
logger = logging.getLogger(__name__)


# 定义数据科学家代理类，继承自ConversableAgent
class DataScientistAgent(ConversableAgent):
    """Data Scientist Agent."""

    # 配置代理的个人资料
    profile: ProfileConfig = ProfileConfig(
        # 代理的名称
        name=DynConfig(
            "Edgar",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_name",
        ),
        # 代理的角色
        role=DynConfig(
            "DataScientist",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_role",
        ),
        # 代理的目标任务描述
        goal=DynConfig(
            "Use correct {{ dialect }} SQL to analyze and solve tasks based on the data"
            " structure information of the database given in the resource.",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_goal",
        ),
        # 代理的工作约束
        constraints=DynConfig(
            [
                "Please check the generated SQL carefully. Please strictly abide by "
                "the data structure definition given. It is prohibited to use "
                "non-existent fields and data values. Do not use fields from table A "
                "to table B. You can perform multi-table related queries.",
                "If the data and fields that need to be analyzed in the target are in "
                "different tables, it is recommended to use multi-table correlation "
                "queries first, and pay attention to the correlation between multiple "
                "table structures.",
                "It is forbidden to construct data by yourself as a query condition. "
                "If you want to query a specific field, if the value of the field is "
                "provided, then you can perform a group statistical query on the "
                "field.",
                "Please select an appropriate one from the supported display methods "
                "for data display. If no suitable display type is found, "
                "table display is used by default. Supported display types: \n"
                "{{ display_type }}",
            ],
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_constraints",
        ),
        # 代理的详细描述
        desc=DynConfig(
            "Use database resources to conduct data analysis, analyze SQL, and provide "
            "recommended rendering methods.",
            category="agent",
            key="dbgpt_agent_expand_dashboard_assistant_agent_profile_desc",
        ),
    )

    # 最大重试次数设定
    max_retry_count: int = 5
    def __init__(self, **kwargs):
        """
        Create a new DataScientistAgent instance.
        """
        # 调用父类的初始化方法，传入所有参数
        super().__init__(**kwargs)
        # 调用内部方法 _init_actions，初始化操作列表，只包含 ChartAction 类
        self._init_actions([ChartAction])

    def _init_reply_message(self, received_message: AgentMessage) -> AgentMessage:
        """
        Initialize the reply message based on the received message.
        """
        # 调用父类的 _init_reply_message 方法，获取基础的回复消息
        reply_message = super()._init_reply_message(received_message)
        # 设置回复消息的上下文信息
        reply_message.context = {
            "display_type": self.actions[0].render_prompt(),  # 设置显示类型，调用第一个操作的 render_prompt 方法
            "dialect": self.database.dialect,  # 设置数据库方言信息
        }
        return reply_message

    @property
    def database(self) -> DBResource:
        """
        Get the database resource.
        """
        # 从资源中获取数据库资源列表
        dbs: List[DBResource] = DBResource.from_resource(self.resource)
        if not dbs:
            # 如果资源为空，抛出数值错误异常，指示不支持的资源类型
            raise ValueError(
                f"Resource type {self.actions[0].resource_need} is not supported."
            )
        # 返回第一个数据库资源对象
        return dbs[0]

    async def correctness_check(
        self, message: AgentMessage
    ) -> Tuple[bool, Optional[str]]:
        """定义函数的返回类型为一个布尔值和一个可选的字符串。"""
        # 获取消息的动作报告
        action_reply = message.action_report
        # 如果动作回复为空，则返回错误消息
        if action_reply is None:
            return (
                False,
                f"No executable analysis SQL is generated,{message.content}.",
            )
        # 将动作回复对象转换为 ActionOutput 类型
        action_out = cast(ActionOutput, ActionOutput.from_dict(action_reply))
        # 如果执行不成功，则返回错误消息
        if not action_out.is_exe_success:
            return (
                False,
                f"Please check your answer, {action_out.content}.",
            )
        # 将动作回复对象的内容解析为 JSON 格式
        action_reply_obj = json.loads(action_out.content)
        # 获取动作回复对象中的 SQL 语句，如果不存在则返回错误消息
        sql = action_reply_obj.get("sql", None)
        if not sql:
            return (
                False,
                "Please check your answer, the sql information that needs to be "
                "generated is not found.",
            )
        try:
            # 如果资源值为空，则返回错误消息
            if not action_out.resource_value:
                return (
                    False,
                    "Please check your answer, the data resource information is not "
                    "found.",
                )

            # 使用数据库对象执行 SQL 查询
            columns, values = await self.database.query(
                sql=sql,
                db=action_out.resource_value,
            )
            # 如果查询结果为空或长度小于等于 0，则返回错误消息
            if not values or len(values) <= 0:
                return (
                    False,
                    "Please check your answer, the current SQL cannot find the data to "
                    "determine whether filtered field values or inappropriate filter "
                    "conditions are used.",
                )
            else:
                # 记录日志，显示检查成功并返回数据行数
                logger.info(
                    f"reply check success! There are {len(values)} rows of data"
                )
                return True, None
        except Exception as e:
            # 记录异常日志并返回 SQL 执行错误消息
            logger.exception(f"DataScientist check exception！{str(e)}")
            return (
                False,
                f"SQL execution error, please re-read the historical information to "
                f"fix this SQL. The error message is as follows:{str(e)}",
            )
```