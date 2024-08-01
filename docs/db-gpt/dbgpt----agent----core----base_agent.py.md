# `.\DB-GPT-src\dbgpt\agent\core\base_agent.py`

```py
# 引入异步 I/O 操作的标准库
import asyncio
# 引入处理 JSON 数据的标准库
import json
# 引入日志记录功能的标准库
import logging
# 从并发库中引入执行器和线程池执行器
from concurrent.futures import Executor, ThreadPoolExecutor
# 从 typing 模块中引入类型提示的相关类和函数
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, cast

# 从内部模块 dbgpt._private.pydantic 中引入 ConfigDict 和 Field 类
from dbgpt._private.pydantic import ConfigDict, Field
# 从 dbgpt.core 模块中引入 LLMClient 和 ModelMessageRoleType 类
from dbgpt.core import LLMClient, ModelMessageRoleType
# 从 dbgpt.util.error_types 模块中引入 LLMChatError 类
from dbgpt.util.error_types import LLMChatError
# 从 dbgpt.util.executor_utils 模块中引入 blocking_func_to_async 函数
from dbgpt.util.executor_utils import blocking_func_to_async
# 从 dbgpt.util.tracer 模块中引入 SpanType 和 root_tracer 函数
from dbgpt.util.tracer import SpanType, root_tracer
# 从 dbgpt.util.utils 模块中引入 colored 函数

# 从上一级目录的 resource.base 模块中引入 Resource 类
from ..resource.base import Resource
# 从上一级目录的 util.llm.llm 模块中引入 LLMConfig 和 LLMStrategyType 类
from ..util.llm.llm import LLMConfig, LLMStrategyType
# 从上一级目录的 util.llm.llm_client 模块中引入 AIWrapper 类
from ..util.llm.llm_client import AIWrapper
# 从当前目录的 action.base 模块中引入 Action 和 ActionOutput 类
from .action.base import Action, ActionOutput
# 从当前目录的 agent 模块中引入 Agent, AgentContext, AgentMessage 和 AgentReviewInfo 类
from .agent import Agent, AgentContext, AgentMessage, AgentReviewInfo
# 从当前目录的 memory.agent_memory 模块中引入 AgentMemory 类
from .memory.agent_memory import AgentMemory
# 从当前目录的 memory.gpts.base 模块中引入 GptsMessage 类
from .memory.gpts.base import GptsMessage
# 从当前目录的 memory.gpts.gpts_memory 模块中引入 GptsMemory 类
from .memory.gpts.gpts_memory import GptsMemory
# 从当前目录的 profile.base 模块中引入 ProfileConfig 类
from .profile.base import ProfileConfig
# 从当前目录的 role 模块中引入 Role 类

# 设置当前模块的日志记录器
logger = logging.getLogger(__name__)


class ConversableAgent(Role, Agent):
    """ConversableAgent is an agent that can communicate with other agents."""

    # 定义模型配置属性，允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 定义代理上下文属性，可选的代理上下文对象
    agent_context: Optional[AgentContext] = Field(None, description="Agent context")
    # 定义动作列表属性，存储动作对象的列表，默认为空列表
    actions: List[Action] = Field(default_factory=list)
    # 定义资源属性，可选的资源对象
    resource: Optional[Resource] = Field(None, description="Resource")
    # 定义语言模型配置属性，LLM 配置对象，默认为 None
    llm_config: Optional[LLMConfig] = None
    # 定义最大重试次数属性，设定为 3
    max_retry_count: int = 3
    # 定义连续自动回复计数器属性，设定为 0
    consecutive_auto_reply_counter: int = 0
    # 定义语言模型客户端属性，LLM 客户端对象，默认为 None
    llm_client: Optional[AIWrapper] = None
    # 定义执行器属性，使用线程池执行器，最大工作线程数为 1
    executor: Executor = Field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1),
        description="Executor for running tasks",
    )

    def __init__(self, **kwargs):
        """Create a new agent."""
        # 调用 Role 类的初始化方法，传入所有关键字参数
        Role.__init__(self, **kwargs)
        # 调用 Agent 类的初始化方法，传入所有关键字参数
        Agent.__init__(self)
    def check_available(self) -> None:
        """Check if the agent is available.

        Raises:
            ValueError: If the agent is not available.
        """
        # 调用身份验证方法
        self.identity_check()
        
        # 检查运行上下文是否存在
        if self.agent_context is None:
            raise ValueError(
                f"{self.name}[{self.role}] Missing context in which agent is "
                f"running!"
            )

        # 检查是否有可执行动作
        if self.actions and len(self.actions) > 0:
            for action in self.actions:
                # 检查是否需要资源，并且资源是否可用
                if action.resource_need and (
                    not self.resource
                    or not self.resource.get_resource_by_type(action.resource_need)
                ):
                    raise ValueError(
                        f"{self.name}[{self.role}] Missing resources required for "
                        "runtime！"
                    )
        else:
            # 如果不是人工智能或团队，则需要动作模块
            if not self.is_human and not self.is_team:
                raise ValueError(
                    f"This agent {self.name}[{self.role}] is missing action modules."
                )
        
        # 检查是否是人工智能模型，并且检查语言模型配置和服务是否可用
        if not self.is_human and (
            self.llm_config is None or self.llm_config.llm_client is None
        ):
            raise ValueError(
                f"{self.name}[{self.role}] Model configuration is missing or model "
                "service is unavailable！"
            )

    @property
    def not_null_agent_context(self) -> AgentContext:
        """Get the agent context.

        Returns:
            AgentContext: The agent context.

        Raises:
            ValueError: If the agent context is not initialized.
        """
        # 如果代理上下文未初始化，则引发异常
        if not self.agent_context:
            raise ValueError("Agent context is not initialized！")
        return self.agent_context

    @property
    def not_null_llm_config(self) -> LLMConfig:
        """Get the LLM config."""
        # 如果语言模型配置未初始化，则引发异常
        if not self.llm_config:
            raise ValueError("LLM config is not initialized！")
        return self.llm_config

    @property
    def not_null_llm_client(self) -> LLMClient:
        """Get the LLM client."""
        # 获取语言模型客户端，如果未初始化，则引发异常
        llm_client = self.not_null_llm_config.llm_client
        if not llm_client:
            raise ValueError("LLM client is not initialized！")
        return llm_client

    async def blocking_func_to_async(
        self, func: Callable[..., Any], *args, **kwargs
    ) -> Any:
        """Run a potentially blocking function within an executor."""
        # 如果函数不是异步函数，则通过阻塞函数转换为异步执行
        if not asyncio.iscoroutinefunction(func):
            return await blocking_func_to_async(self.executor, func, *args, **kwargs)
        return await func(*args, **kwargs)

    async def preload_resource(self) -> None:
        """Preload resources before agent initialization."""
        # 如果存在资源，则预加载资源（异步执行）
        if self.resource:
            await self.blocking_func_to_async(self.resource.preload_resource)
    async def build(self) -> "ConversableAgent":
        """Build the agent."""
        # 预加载资源
        await self.preload_resource()
        # 检查代理是否可用
        self.check_available()
        # 获取非空的代理上下文中的语言
        _language = self.not_null_agent_context.language
        if _language:
            self.language = _language

        # 初始化资源加载器
        for action in self.actions:
            action.init_resource(self.resource)

        # 初始化LLM服务器
        if not self.is_human:
            # 检查LLM配置和LLM客户端是否已初始化
            if not self.llm_config or not self.llm_config.llm_client:
                raise ValueError("LLM client is not initialized！")
            # 创建LLM客户端包装器
            self.llm_client = AIWrapper(llm_client=self.llm_config.llm_client)
            # 初始化内存
            self.memory.initialize(
                self.name,
                self.llm_config.llm_client,
                importance_scorer=self.memory_importance_scorer,
                insight_extractor=self.memory_insight_extractor,
            )
            # 克隆内存结构
            self.memory = self.memory.structure_clone()
        return self

    def bind(self, target: Any) -> "ConversableAgent":
        """Bind the resources to the agent."""
        # 根据目标类型绑定相应资源
        if isinstance(target, LLMConfig):
            self.llm_config = target
        elif isinstance(target, GptsMemory):
            raise ValueError("GptsMemory is not supported!")
        elif isinstance(target, AgentContext):
            self.agent_context = target
        elif isinstance(target, Resource):
            self.resource = target
        elif isinstance(target, AgentMemory):
            self.memory = target
        elif isinstance(target, ProfileConfig):
            self.profile = target
        elif isinstance(target, type) and issubclass(target, Action):
            self.actions.append(target())
        return self

    async def send(
        self,
        message: AgentMessage,
        recipient: Agent,
        reviewer: Optional[Agent] = None,
        request_reply: Optional[bool] = True,
        is_recovery: Optional[bool] = False,
    ) -> None:
        """Send a message to recipient agent."""
        # 使用根跟踪器开始一个跟踪 span，记录发送的消息相关信息
        with root_tracer.start_span(
            "agent.send",
            metadata={
                "sender": self.name,
                "recipient": recipient.name,
                "reviewer": reviewer.name if reviewer else None,
                "agent_message": json.dumps(message.to_dict(), ensure_ascii=False),
                "request_reply": request_reply,
                "is_recovery": is_recovery,
                "conv_uid": self.not_null_agent_context.conv_id,
            },
        ):
            # 异步发送消息给接收者代理
            await recipient.receive(
                message=message,
                sender=self,
                reviewer=reviewer,
                request_reply=request_reply,
                is_recovery=is_recovery,
            )
    # 异步方法：接收来自另一个代理的消息
    async def receive(
        self,
        message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        request_reply: Optional[bool] = None,
        silent: Optional[bool] = False,
        is_recovery: Optional[bool] = False,
    ) -> None:
        """Receive a message from another agent."""
        # 开始一个新的跟踪span，用于追踪接收消息的过程
        with root_tracer.start_span(
            "agent.receive",
            metadata={
                "sender": sender.name,
                "recipient": self.name,
                "reviewer": reviewer.name if reviewer else None,
                "agent_message": json.dumps(message.to_dict(), ensure_ascii=False),
                "request_reply": request_reply,
                "silent": silent,
                "is_recovery": is_recovery,
                "conv_uid": self.not_null_agent_context.conv_id,
                "is_human": self.is_human,
            },
        ):
            # 调用内部方法处理接收到的消息
            await self._a_process_received_message(message, sender)
            # 如果不需要回复或未指定回复请求，则直接返回
            if request_reply is False or request_reply is None:
                return
            # 如果代理是人类代理，暂时不生成回复
            if self.is_human:
                # 不为人类代理生成回复
                return

            # 如果自动回复计数小于最大聊天轮次限制
            if (
                self.consecutive_auto_reply_counter
                <= self.not_null_agent_context.max_chat_round
            ):
                # 生成回复消息
                reply = await self.generate_reply(
                    received_message=message, sender=sender, reviewer=reviewer
                )
                # 如果生成的回复不为空，则发送回复消息给发送者
                if reply is not None:
                    await self.send(reply, sender)
            else:
                # 记录日志：当前轮次超过最大聊天轮次限制
                logger.info(
                    f"Current round {self.consecutive_auto_reply_counter} "
                    f"exceeds the maximum chat round "
                    f"{self.not_null_agent_context.max_chat_round}!"
                )

    # 准备用于 act 方法的参数字典
    def prepare_act_param(self) -> Dict[str, Any]:
        """Prepare the parameters for the act method."""
        return {}

    # 异步方法：生成回复消息
    async def generate_reply(
        self,
        received_message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        rely_messages: Optional[List[AgentMessage]] = None,
        **kwargs,
    ):
        ...

    # 异步方法：思考过程，接受消息列表和可选的提示字符串作为参数
    async def thinking(
        self, messages: List[AgentMessage], prompt: Optional[str] = None
    ):
        ...
    ) -> Tuple[Optional[str], Optional[str]]:
        """定义一个异步方法act，用于执行代理的行为。

        Args:
            message(Optional[str]): 要处理的消息内容
            sender(Optional[Agent]): 发送者代理对象
            reviewer(Optional[Agent]): 审核者代理对象
            **kwargs: 其他关键字参数

        Returns:
            Tuple[Optional[str], Optional[str]]: 返回一个包含两个可选字符串的元组，表示处理结果和错误信息或状态
        """
        last_model = None
        last_err = None
        retry_count = 0
        llm_messages = [message.to_llm_message() for message in messages]
        # LLM 推理自动重试3次，以减少由速度限制和网络稳定性引起的中断概率
        while retry_count < 3:
            llm_model = await self._a_select_llm_model(last_model)
            try:
                if prompt:
                    llm_messages = _new_system_message(prompt) + llm_messages

                if not self.llm_client:
                    raise ValueError("LLM client is not initialized!")
                response = await self.llm_client.create(
                    context=llm_messages[-1].pop("context", None),
                    messages=llm_messages,
                    llm_model=llm_model,
                    max_new_tokens=self.not_null_agent_context.max_new_tokens,
                    temperature=self.not_null_agent_context.temperature,
                    verbose=self.not_null_agent_context.verbose,
                )
                return response, llm_model
            except LLMChatError as e:
                logger.error(f"model:{llm_model} generate Failed!{str(e)}")
                retry_count += 1
                last_model = llm_model
                last_err = str(e)
                await asyncio.sleep(10)

        if last_err:
            raise ValueError(last_err)
        else:
            raise ValueError("LLM model inference failed!")
    ) -> Optional[ActionOutput]:
        """Perform actions."""
        # 初始化最后一个输出为 None
        last_out: Optional[ActionOutput] = None
        # 遍历所有动作
        for i, action in enumerate(self.actions):
            # 选择动作所需资源
            if action.resource_need and self.resource:
                need_resources = self.resource.get_resource_by_type(
                    action.resource_need
                )
            else:
                need_resources = []

            # 如果消息为空，则抛出数值错误
            if not message:
                raise ValueError("The message content is empty!")

            # 创建一个新的跟踪 span
            with root_tracer.start_span(
                "agent.act.run",
                metadata={
                    "message": message,
                    "sender": sender.name if sender else None,
                    "recipient": self.name,
                    "reviewer": reviewer.name if reviewer else None,
                    "need_resource": need_resources[0].name if need_resources else None,
                    "rely_action_out": last_out.to_dict() if last_out else None,
                    "conv_uid": self.not_null_agent_context.conv_id,
                    "action_index": i,
                    "total_action": len(self.actions),
                },
            ) as span:
                # 执行动作并更新最后一个输出
                last_out = await action.run(
                    ai_message=message,
                    resource=None,
                    rely_action_out=last_out,
                    **kwargs,
                )
                # 将动作输出添加到 span 的元数据中
                span.metadata["action_out"] = last_out.to_dict() if last_out else None
        # 返回最后一个输出
        return last_out

    async def correctness_check(
        self, message: AgentMessage
    ) -> Tuple[bool, Optional[str]]:
        """Verify the correctness of the results."""
        # 验证结果的正确性
        return True, None

    async def verify(
        self,
        message: AgentMessage,
        sender: Agent,
        reviewer: Optional[Agent] = None,
        **kwargs,
    ) -> Tuple[bool, Optional[str]]:
        """Verify the current execution results."""
        # 检查批准结果
        if message.review_info and not message.review_info.approve:
            return False, message.review_info.comments

        # 检查动作运行结果
        action_output: Optional[ActionOutput] = ActionOutput.from_dict(
            message.action_report
        )
        if action_output:
            if not action_output.is_exe_success:
                return False, action_output.content
            elif not action_output.content or len(action_output.content.strip()) < 1:
                return (
                    False,
                    "The current execution result is empty. Please rethink the "
                    "question and background and generate a new answer.. ",
                )

        # 验证代理输出的正确性
        return await self.correctness_check(message)

    async def initiate_chat(
        self,
        recipient: Agent,
        reviewer: Optional[Agent] = None,
        message: Optional[str] = None,
    ):
        """Initiate a chat with another agent.

        Args:
            recipient (Agent): The recipient agent.
            reviewer (Agent): The reviewer agent.
            message (str): The message to send.
        """
        # 创建一个代理消息对象，用于发送和当前目标
        agent_message = AgentMessage(content=message, current_goal=message)
        
        # 使用分布式追踪系统的根跟踪器开始一个新的跟踪 span，标记为“agent.initiate_chat”，并附带元数据
        with root_tracer.start_span(
            "agent.initiate_chat",
            span_type=SpanType.AGENT,
            metadata={
                "sender": self.name,
                "recipient": recipient.name,
                "reviewer": reviewer.name if reviewer else None,
                "agent_message": json.dumps(
                    agent_message.to_dict(), ensure_ascii=False
                ),
                "conv_uid": self.not_null_agent_context.conv_id,
            },
        ):
            # 使用代理消息对象向特定的接收者发送消息，并可能涉及审核者进行请求回复
            await self.send(
                agent_message,
                recipient,
                reviewer,
                request_reply=True,
            )

    #######################################################################
    # Private Function Begin
    #######################################################################

    def _init_actions(self, actions: List[Type[Action]]):
        # 初始化动作列表为空
        self.actions = []
        
        # 遍历给定的动作列表，如果动作是 Action 类的子类，则将其实例化并加入到动作列表中
        for idx, action in enumerate(actions):
            if issubclass(action, Action):
                self.actions.append(action())

    async def _a_append_message(
        self, message: AgentMessage, role, sender: Agent
    ) -> bool:
        # 将 sender 转换为 ConversableAgent 类型
        new_sender = cast(ConversableAgent, sender)
        # 更新连续自动回复计数器
        self.consecutive_auto_reply_counter = (
            new_sender.consecutive_auto_reply_counter + 1
        )
        # 将消息对象转换为字典格式
        message_dict = message.to_dict()
        # 从消息字典中提取需要的字段，构建 oai_message 字典
        oai_message = {
            k: message_dict[k]
            for k in (
                "content",
                "function_call",
                "name",
                "context",
                "action_report",
                "review_info",
                "current_goal",
                "model_name",
            )
            if k in message_dict
        }

        # 创建 GptsMessage 对象
        gpts_message: GptsMessage = GptsMessage(
            conv_id=self.not_null_agent_context.conv_id,  # 对话 ID
            sender=sender.role,  # 发送者角色
            receiver=self.role,  # 接收者角色
            role=role,  # 角色
            rounds=self.consecutive_auto_reply_counter,  # 连续自动回复轮数
            current_goal=oai_message.get("current_goal", None),  # 当前目标
            content=oai_message.get("content", None),  # 内容
            context=(
                json.dumps(oai_message["context"], ensure_ascii=False)  # 上下文信息转为 JSON
                if "context" in oai_message
                else None
            ),
            review_info=(
                json.dumps(oai_message["review_info"], ensure_ascii=False)  # 审查信息转为 JSON
                if "review_info" in oai_message
                else None
            ),
            action_report=(
                json.dumps(oai_message["action_report"], ensure_ascii=False)  # 行动报告转为 JSON
                if "action_report" in oai_message
                else None
            ),
            model_name=oai_message.get("model_name", None),  # 模型名称
        )

        # 使用 root_tracer 开始一个新的跟踪 span 来保存消息到内存中
        with root_tracer.start_span(
            "agent.save_message_to_memory",
            metadata={
                "gpts_message": gpts_message.to_dict(),  # 将 gpts_message 转为字典并作为元数据
                "conv_uid": self.not_null_agent_context.conv_id,  # 对话 ID
            },
        ):
            # 将 gpts_message 添加到内存中的消息存储
            self.memory.message_memory.append(gpts_message)
            return True  # 返回 True 表示消息保存成功
    # 打印接收到的消息及相关信息
    def _print_received_message(self, message: AgentMessage, sender: Agent):
        # 打印分隔线
        print("\n", "-" * 80, flush=True, sep="")
        # 确定要打印的名称或角色
        _print_name = self.name if self.name else self.role
        # 打印消息发送者的名称或角色
        print(
            colored(
                sender.name if sender.name else sender.role,
                "yellow",
            ),
            "(to",
            f"{_print_name})-[{message.model_name or ''}]:\n",
            flush=True,
        )

        # 将消息内容转换成 JSON 格式字符串并打印
        content = json.dumps(message.content, ensure_ascii=False)
        if content is not None:
            print(content, flush=True)

        # 如果有审阅信息，打印审阅结果及评论
        review_info = message.review_info
        if review_info:
            name = sender.name if sender.name else sender.role
            pass_msg = "Pass" if review_info.approve else "Reject"
            review_msg = f"{pass_msg}({review_info.comments})"
            approve_print = f">>>>>>>>{name} Review info: \n{review_msg}"
            print(colored(approve_print, "green"), flush=True)

        # 如果有操作报告，打印操作执行结果及内容
        action_report = message.action_report
        if action_report:
            name = sender.name if sender.name else sender.role
            action_msg = (
                "execution succeeded"
                if action_report["is_exe_success"]
                else "execution failed"
            )
            action_report_msg = f"{action_msg},\n{action_report['content']}"
            action_print = f">>>>>>>>{name} Action report: \n{action_report_msg}"
            print(colored(action_print, "blue"), flush=True)

        # 打印分隔线
        print("\n", "-" * 80, flush=True, sep="")

    # 异步处理接收到的消息
    async def _a_process_received_message(self, message: AgentMessage, sender: Agent):
        # 将消息附加到适当的地方，验证消息有效性
        valid = await self._a_append_message(message, None, sender)
        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion"
                " message. Either content or function_call must be provided."
            )

        # 调用同步方法打印接收到的消息及相关信息
        self._print_received_message(message, sender)

    # 生成资源变量的方法
    async def generate_resource_variables(
        self, question: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate the resource variables."""
        resource_prompt = None
        # 如果存在资源，获取相应的提示信息
        if self.resource:
            resource_prompt = await self.resource.get_prompt(
                lang=self.language, question=question
            )

        out_schema: Optional[str] = ""
        # 如果存在动作列表并且不为空，获取第一个动作的输出模式
        if self.actions and len(self.actions) > 0:
            out_schema = self.actions[0].ai_out_schema
        # 返回生成的资源变量字典
        return {"resource_prompt": resource_prompt, "out_schema": out_schema}

    # 获取被排除的模型列表
    def _excluded_models(
        self,
        all_models: List[str],
        order_llms: Optional[List[str]] = None,
        excluded_models: Optional[List[str]] = None,
    ):
        # 如果没有指定排序的语言模型列表，则初始化为空列表
        if not order_llms:
            order_llms = []
        # 如果没有指定要排除的模型列表，则初始化为空列表
        if not excluded_models:
            excluded_models = []
        # 可以使用的语言模型列表
        can_uses = []
        # 如果有指定的排序语言模型列表并且列表长度大于0，则按顺序处理
        if order_llms and len(order_llms) > 0:
            # 遍历排序后的语言模型列表
            for llm_name in order_llms:
                # 如果语言模型在所有模型列表中且未在排除列表中，则添加到可用列表中
                if llm_name in all_models and (
                    not excluded_models or llm_name not in excluded_models
                ):
                    can_uses.append(llm_name)
        else:
            # 没有指定排序语言模型列表，则遍历所有语言模型列表
            for llm_name in all_models:
                # 如果语言模型不在排除列表中，则添加到可用列表中
                if not excluded_models or llm_name not in excluded_models:
                    can_uses.append(llm_name)

        # 返回可用的语言模型列表
        return can_uses

    async def _a_select_llm_model(
        self, excluded_models: Optional[List[str]] = None
    ) -> str:
        # 记录选择语言模型的操作日志，包括排除的模型列表
        logger.info(f"_a_select_llm_model:{excluded_models}")
        try:
            # 异步获取所有非空语言模型客户端的模型信息
            all_models = await self.not_null_llm_client.models()
            # 提取所有模型的名称列表
            all_model_names = [item.model for item in all_models]
            # TODO 目前只实现了两种策略，优先级和默认。
            # 如果配置的语言模型策略为优先级
            if self.not_null_llm_config.llm_strategy == LLMStrategyType.Priority:
                priority: List[str] = []
                # 获取策略上下文
                strategy_context = self.not_null_llm_config.strategy_context
                if strategy_context is not None:
                    # 将策略上下文解析为优先级列表
                    priority = json.loads(strategy_context)  # type: ignore
                # 根据优先级和排除的模型列表选择可以使用的模型
                can_uses = self._excluded_models(
                    all_model_names, priority, excluded_models
                )
            else:
                # 如果未配置优先级策略，则直接根据所有模型和排除的模型列表选择可以使用的模型
                can_uses = self._excluded_models(all_model_names, None, excluded_models)
            # 如果有可用模型并且列表长度大于0，则返回第一个可用模型
            if can_uses and len(can_uses) > 0:
                return can_uses[0]
            else:
                # 如果没有可用模型，则抛出数值错误
                raise ValueError("No model service available!")
        except Exception as e:
            # 捕获所有异常，并记录错误日志
            logger.error(f"{self.role} get next llm failed!{str(e)}")
            # 抛出数值错误，指示分配模型服务失败
            raise ValueError(f"Failed to allocate model service,{str(e)}!")

    def _init_reply_message(self, received_message: AgentMessage) -> AgentMessage:
        """Create a new message from the received message.

        Initialize a new message from the received message

        Args:
            received_message(AgentMessage): The received message

        Returns:
            AgentMessage: A new message
        """
        # 从接收到的消息中创建一个新消息对象并返回
        return AgentMessage(
            content=received_message.content,
            current_goal=received_message.current_goal,
        )

    def _convert_to_ai_message(
        self, gpts_messages: List[GptsMessage]
    ) -> List[AgentMessage]:
        oai_messages: List[AgentMessage] = []
        # 基于当前的代理，所有接收到的消息被认为是用户发来的，所有发送出去的消息被认为是助手发出的。
        for item in gpts_messages:
            # 如果消息对象有角色信息，则使用该角色
            if item.role:
                role = item.role
            else:
                # 如果消息对象没有角色信息，则根据接收者和发送者判断角色类型
                if item.receiver == self.role:
                    role = ModelMessageRoleType.HUMAN
                elif item.sender == self.role:
                    role = ModelMessageRoleType.AI
                else:
                    continue  # 如果既非接收者也非发送者，跳过当前消息的处理

            # 消息内容的处理，优先使用执行结果的转换，否则使用模型输出的结果
            content = item.content
            if item.action_report:
                # 解析动作报告为 ActionOutput 对象
                action_out = ActionOutput.from_dict(json.loads(item.action_report))
                if (
                    action_out is not None
                    and action_out.is_exe_success
                    and action_out.content is not None
                ):
                    content = action_out.content
            # 构造 AgentMessage 对象并添加到 oai_messages 列表中
            oai_messages.append(
                AgentMessage(
                    content=content,
                    role=role,
                    context=(
                        json.loads(item.context) if item.context is not None else None
                    ),
                )
            )
        return oai_messages

    async def _load_thinking_messages(
        self,
        received_message: AgentMessage,
        sender: Agent,
        rely_messages: Optional[List[AgentMessage]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[AgentMessage]:
        # 从接收到的消息中获取观察结果
        observation = received_message.content
        # 如果观察结果为空，则抛出数值错误异常
        if not observation:
            raise ValueError("The received message content is empty!")
        # 使用观察结果读取记忆信息
        memories = await self.read_memories(observation)
        # 初始化回复消息字符串
        reply_message_str = ""
        # 如果上下文为空，则初始化为空字典
        if context is None:
            context = {}
        # 如果有依赖消息
        if rely_messages:
            # 复制依赖消息列表中的每个消息
            copied_rely_messages = [m.copy() for m in rely_messages]
            # 遍历复制的依赖消息列表
            for message in copied_rely_messages:
                # 从消息的动作报告中创建动作输出对象，可能为空
                action_report: Optional[ActionOutput] = ActionOutput.from_dict(
                    message.action_report
                )
                # 如果动作报告不为空
                if action_report:
                    # TODO: 进行原地修改，需要优化
                    # 将消息内容替换为动作报告中的内容
                    message.content = action_report.content
                # 如果消息的名称不是当前角色
                if message.name != self.role:
                    # TODO，使用名称
                    # 依赖消息不是来自当前代理
                    # 如果消息角色是人类
                    if message.role == ModelMessageRoleType.HUMAN:
                        # 添加问题类型的回复消息字符串
                        reply_message_str += f"Question: {message.content}\n"
                    # 如果消息角色是AI
                    elif message.role == ModelMessageRoleType.AI:
                        # 添加观察类型的回复消息字符串
                        reply_message_str += f"Observation: {message.content}\n"
        # 如果有回复消息字符串
        if reply_message_str:
            # 将回复消息添加到记忆中
            memories += "\n" + reply_message_str

        # 构建系统提示语
        system_prompt = await self.build_prompt(
            question=observation,
            is_system=True,
            most_recent_memories=memories,
            **context,
        )
        # 构建用户提示语
        user_prompt = await self.build_prompt(
            question=observation,
            is_system=False,
            most_recent_memories=memories,
            **context,
        )

        # 初始化代理消息列表
        agent_messages = []
        # 如果系统提示语不为空
        if system_prompt:
            # 将系统提示语添加为代理消息
            agent_messages.append(
                AgentMessage(
                    content=system_prompt,
                    role=ModelMessageRoleType.SYSTEM,
                )
            )
        # 如果用户提示语不为空
        if user_prompt:
            # 将用户提示语添加为代理消息
            agent_messages.append(
                AgentMessage(
                    content=user_prompt,
                    role=ModelMessageRoleType.HUMAN,
                )
            )

        # 返回代理消息列表
        return agent_messages

    def _old_load_thinking_messages(
        self,
        received_message: AgentMessage,
        sender: Agent,
        rely_messages: Optional[List[AgentMessage]] = None,
        ) -> List[AgentMessage]:
        # 获取当前接收到的消息的当前目标
        current_goal = received_message.current_goal

        # 将集体记忆中的信息转换并适应当前 Agent 可用的上下文记忆

        # 使用根跟踪器开始一个命名为 "agent._load_thinking_messages" 的 span
        with root_tracer.start_span(
            "agent._load_thinking_messages",
            metadata={
                "sender": sender.name,
                "recipient": self.name,
                "conv_uid": self.not_null_agent_context.conv_id,
                "current_goal": current_goal,
            },
        ) as span:
            # 从记忆中获取历史信息的消息
            memory_messages = self.memory.message_memory.get_between_agents(
                self.not_null_agent_context.conv_id,
                self.role,
                sender.role,
                current_goal,
            )
            # 将 memory_messages 转换为字典形式存储在 span 的元数据中
            span.metadata["memory_messages"] = [
                message.to_dict() for message in memory_messages
            ]
        
        # 将历史消息转换为 AI 消息格式
        current_goal_messages = self._convert_to_ai_message(memory_messages)

        # 当没有目标和上下文时，使用当前接收到的消息作为目标问题
        if current_goal_messages is None or len(current_goal_messages) <= 0:
            received_message.role = ModelMessageRoleType.HUMAN
            current_goal_messages = [received_message]

        # 转发消息
        cut_messages = []
        if rely_messages:
            # 当直接依赖历史消息时，使用执行结果内容作为依赖
            for rely_message in rely_messages:
                action_report: Optional[ActionOutput] = ActionOutput.from_dict(
                    rely_message.action_report
                )
                if action_report:
                    # TODO: 在原地修改，需要优化
                    rely_message.content = action_report.content

            cut_messages.extend(rely_messages)

        # TODO: 根据令牌预算分配基于历史信息
        if len(current_goal_messages) < 5:
            cut_messages.extend(current_goal_messages)
        else:
            # 暂时使用默认的最小历史消息记录大小
            # 使用前两轮消息了解初始目标
            cut_messages.extend(current_goal_messages[:2])
            # 使用最后三轮通信信息确保当前思维了解最后通信的发生和需要做什么
            cut_messages.extend(current_goal_messages[-3:])
        return cut_messages
# 返回包含系统消息的字典列表，每个字典包含内容和角色信息
def _new_system_message(content):
    """Return the system message."""
    return [{"content": content, "role": ModelMessageRoleType.SYSTEM}]


# 检查列表中的所有元素是否都属于指定类型
def _is_list_of_type(lst: List[Any], type_cls: type) -> bool:
    """Check if all elements in the list are instances of the specified type."""
    return all(isinstance(item, type_cls) for item in lst)
```