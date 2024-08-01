# `.\DB-GPT-src\dbgpt\serve\agent\agents\controller.py`

```py
# 引入异步 I/O 模块
import asyncio
# 引入 JSON 模块
import json
# 引入日志记录模块
import logging
# 引入 UUID 模块
import uuid
# 引入抽象基类 ABC
from abc import ABC
# 引入类型提示模块
from typing import Any, Dict, List, Optional, Type

# 引入 FastAPI 中的 APIRouter 和 Body
from fastapi import APIRouter, Body
# 引入 FastAPI 中的 StreamingResponse
from fastapi.responses import StreamingResponse

# 引入 dbgpt 应用的配置模块 Config
from dbgpt._private.config import Config
# 引入 dbgpt 应用中的 Agent 和 AgentContext 类
from dbgpt.agent.core.agent import Agent, AgentContext
# 引入 dbgpt 应用中的 Agent 管理函数 get_agent_manager
from dbgpt.agent.core.agent_manage import get_agent_manager
# 引入 dbgpt 应用中的 ConversableAgent 抽象基类
from dbgpt.agent.core.base_agent import ConversableAgent
# 引入 dbgpt 应用中的 AgentMemory 类
from dbgpt.agent.core.memory.agent_memory import AgentMemory
# 引入 dbgpt 应用中的 GptsMemory 类
from dbgpt.agent.core.memory.gpts.gpts_memory import GptsMemory
# 引入 dbgpt 应用中的计划管理相关模块
from dbgpt.agent.core.plan import AutoPlanChatManager, DefaultAWELLayoutManager
# 引入 dbgpt 应用中的 schema 模块中的 Status 类
from dbgpt.agent.core.schema import Status
# 引入 dbgpt 应用中的 UserProxyAgent 类
from dbgpt.agent.core.user_proxy_agent import UserProxyAgent
# 引入 dbgpt 应用中的资源管理相关模块
from dbgpt.agent.resource.base import Resource
from dbgpt.agent.resource.manage import get_resource_manager
# 引入 dbgpt 应用中的 LL 模块相关配置和策略
from dbgpt.agent.util.llm.llm import LLMConfig, LLMStrategyType
# 引入 dbgpt 应用中的 API ViewModel 模块中的 Result 类
from dbgpt.app.openapi.api_view_model import Result
# 引入 dbgpt 应用中的场景模块 ChatScene 类
from dbgpt.app.scene.base import ChatScene
# 引入 dbgpt 应用中的组件相关模块
from dbgpt.component import BaseComponent, ComponentType, SystemApp
# 引入 dbgpt 应用中的消息存储接口 StorageConversation 类
from dbgpt.core.interface.message import StorageConversation
# 引入 dbgpt 应用中的集群管理模块
from dbgpt.model.cluster import WorkerManagerFactory
# 引入 dbgpt 应用中的默认 LLM 客户端模块
from dbgpt.model.cluster.client import DefaultLLMClient
# 引入 dbgpt 应用中的服务模块
from dbgpt.serve.agent.model import PagenationFilter, PluginHubFilter
# 引入 dbgpt 应用中的对话服务模块 Serve 类别名 ConversationServe
from dbgpt.serve.conversation.serve import Serve as ConversationServe
# 引入 dbgpt 应用中的 JSON 序列化工具模块
from dbgpt.util.json_utils import serialize
# 引入 dbgpt 应用中的追踪工具模块
from dbgpt.util.tracer import root_tracer

# 引入当前模块中的数据库操作相关模块
from ..db.gpts_app import GptsApp, GptsAppDao, GptsAppQuery
from ..db.gpts_conversations_db import GptsConversationsDao, GptsConversationsEntity
from ..db.gpts_manage_db import GptsInstanceEntity
# 引入当前模块中的团队模式基类
from ..team.base import TeamMode
# 引入当前模块中的与数据库 GPTS 内存相关模块
from .db_gpts_memory import MetaDbGptsMessageMemory, MetaDbGptsPlansMemory

# 实例化配置对象 CFG
CFG = Config()

# 创建 APIRouter 对象
router = APIRouter()
# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)

# 定义一个函数 _build_conversation，用于构建对话存储对象
def _build_conversation(
    conv_id: str,
    select_param: Dict[str, Any],
    model_name: str,
    summary: str,
    conv_serve: ConversationServe,
    user_name: Optional[str] = "",
    sys_code: Optional[str] = "",
) -> StorageConversation:
    return StorageConversation(
        conv_uid=conv_id,
        chat_mode=ChatScene.ChatAgent.value(),
        user_name=user_name,
        sys_code=sys_code,
        model_name=model_name,
        summary=summary,
        param_type="DbGpts",
        param_value=select_param,
        conv_storage=conv_serve.conv_storage,
        message_storage=conv_serve.message_storage,
    )

# 定义一个类 MultiAgents，继承自 BaseComponent 和 ABC 抽象基类
class MultiAgents(BaseComponent, ABC):
    # 类属性 name，表示组件类型为 MULTI_AGENTS
    name = ComponentType.MULTI_AGENTS

    # 初始化应用方法，接受一个系统应用对象 system_app
    def init_app(self, system_app: SystemApp):
        # 将当前路由器对象 router 加入到系统应用中，API 路径为 "/api"，标签为 ["Multi-Agents"]
        system_app.app.include_router(router, prefix="/api", tags=["Multi-Agents"])
    # 初始化方法，设置了GptsConversationsDao实例和GptsAppDao实例，并创建了GptsMemory实例。
    # 同时初始化了一个空的agent_memory_map字典，并调用父类的初始化方法。
    def __init__(self):
        self.gpts_conversations = GptsConversationsDao()

        self.gpts_app = GptsAppDao()
        self.memory = GptsMemory(
            plans_memory=MetaDbGptsPlansMemory(),
            message_memory=MetaDbGptsMessageMemory(),
        )
        self.agent_memory_map = {}
        super().__init__()

    # 根据会话ID和调试GPTS名称获取或构建一个AgentMemory实例。
    # 如果已存在相同的内存键(memory_key)，直接返回已有的AgentMemory实例。
    # 否则，根据给定的embedding配置，创建一个新的AgentMemory实例，并保存在agent_memory_map中。
    def get_or_build_agent_memory(self, conv_id: str, dbgpts_name: str) -> AgentMemory:
        from dbgpt.agent.core.memory.agent_memory import (
            AgentMemory,
            AgentMemoryFragment,
        )
        from dbgpt.agent.core.memory.hybrid import HybridMemory
        from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG
        from dbgpt.rag.embedding.embedding_factory import EmbeddingFactory

        memory_key = f"{dbgpts_name}_{conv_id}"
        if memory_key in self.agent_memory_map:
            return self.agent_memory_map[memory_key]

        embedding_factory = EmbeddingFactory.get_instance(CFG.SYSTEM_APP)
        embedding_fn = embedding_factory.create(
            model_name=EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]
        )
        vstore_name = f"_chroma_agent_memory_{dbgpts_name}_{conv_id}"
        
        # 使用HybridMemory从Chroma存储中创建AgentMemory实例，并关联到全局GptsMemory。
        memory = HybridMemory[AgentMemoryFragment].from_chroma(
            vstore_name=vstore_name,
            embeddings=embedding_fn,
        )
        agent_memory = AgentMemory(memory, gpts_memory=self.memory)
        self.agent_memory_map[memory_key] = agent_memory
        return agent_memory

    # 向GptsInstanceEntity表中添加一个实体。
    def gpts_create(self, entity: GptsInstanceEntity):
        self.gpts_intance.add(entity)

    # 根据用户代码和系统代码获取GPTS应用列表。
    def get_dbgpts(
        self, user_code: str = None, sys_code: str = None
    ) -> Optional[List[GptsApp]]:
        apps = self.gpts_app.app_list(
            GptsAppQuery(user_code=user_code, sys_code=sys_code)
        ).app_list
        return apps

    # 异步方法，用于与Agent进行对话。
    # 接收Agent会话ID(agent_conv_id)、GPTS名称(gpts_name)、用户查询(user_query)等参数。
    # 可选参数包括用户代码(user_code)和系统代码(sys_code)，以及可选的AgentMemory实例(agent_memory)。
    async def agent_chat(
        self,
        agent_conv_id: str,
        gpts_name: str,
        user_query: str,
        user_code: str = None,
        sys_code: str = None,
        agent_memory: Optional[AgentMemory] = None,
    ):
        # 省略具体的异步对话逻辑，不在这段代码中显示。
        pass
    ):
        # 根据 gpts_name 获取 GptsApp 对象的详细信息
        gpt_app: GptsApp = self.gpts_app.app_detail(gpts_name)

        # 根据 agent_conv_id 获取或创建 GptsConversationsEntity 对象
        gpts_conversation = self.gpts_conversations.get_by_conv_id(agent_conv_id)
        is_retry_chat = True
        
        # 如果 gpts_conversation 不存在，则表示不是重试聊天，需要添加新的会话记录
        if not gpts_conversation:
            is_retry_chat = False
            self.gpts_conversations.add(
                GptsConversationsEntity(
                    conv_id=agent_conv_id,
                    user_goal=user_query,
                    gpts_name=gpts_name,
                    team_mode=gpt_app.team_mode,
                    state=Status.RUNNING.value,
                    max_auto_reply_round=0,
                    auto_reply_count=0,
                    user_code=user_code,
                    sys_code=sys_code,
                )
            )

        # 创建一个异步任务，调用 multi_agents.agent_team_chat_new 方法进行多代理团队聊天
        task = asyncio.create_task(
            multi_agents.agent_team_chat_new(
                user_query,
                agent_conv_id,
                gpt_app,
                is_retry_chat,
                agent_memory,
                span_id=root_tracer.get_current_span_id(),
            )
        )

        # 异步迭代 multi_agents.chat_messages(agent_conv_id) 方法返回的消息块
        async for chunk in multi_agents.chat_messages(agent_conv_id):
            if chunk:
                try:
                    # 将 chunk 序列化为 JSON 格式字符串
                    chunk = json.dumps(
                        {"vis": chunk}, default=serialize, ensure_ascii=False
                    )
                    # 如果 chunk 为 None 或长度小于等于 0，则跳过处理下一个消息块
                    if chunk is None or len(chunk) <= 0:
                        continue
                    # 构造响应数据字符串，格式为 "data:{chunk}\n\n"
                    resp = f"data:{chunk}\n\n"
                    # 返回任务对象和响应数据
                    yield task, resp
                except Exception as e:
                    # 记录异常日志并返回异常信息
                    logger.exception(f"get messages {gpts_name} Exception!" + str(e))
                    yield f"data: {str(e)}\n\n"

        # 向客户端发送标记为完成的消息
        yield task, f'data:{json.dumps({"vis": "[DONE]"}, default=serialize, ensure_ascii=False)} \n\n'

    # 定义异步方法 app_agent_chat，用于处理与应用代理的用户交互
    async def app_agent_chat(
        self,
        conv_uid: str,
        gpts_name: str,
        user_query: str,
        user_code: str = None,
        sys_code: str = None,
        ):
            # 记录信息到日志，标记为app_agent_chat，包括GPT模型名称、用户查询、会话UID
            logger.info(f"app_agent_chat:{gpts_name},{user_query},{conv_uid}")

            # 获取ConversationServe的单例对象，用于处理对话服务
            conv_serve = ConversationServe.get_instance(CFG.SYSTEM_APP)
            
            # 构建一个StorageConversation对象来存储当前会话的信息
            current_message: StorageConversation = _build_conversation(
                conv_id=conv_uid,
                select_param=gpts_name,
                summary=user_query,
                model_name="",
                conv_serve=conv_serve,
            )
            
            # 将当前会话信息保存到存储中
            current_message.save_to_storage()
            
            # 开始一个新的对话轮次
            current_message.start_new_round()
            
            # 将用户的查询消息添加到当前会话中
            current_message.add_user_message(user_query)
            
            # 生成代理会话ID，格式为会话UID_聊天序号
            agent_conv_id = conv_uid + "_" + str(current_message.chat_order)
            
            agent_task = None
            
            # 尝试开始与多个代理进行异步聊天
            try:
                # 获取或构建代理的记忆数据
                agent_memory = self.get_or_build_agent_memory(conv_uid, gpts_name)
                
                # 通过多个代理进行异步聊天，每个代理任务返回一个chunk
                async for task, chunk in multi_agents.agent_chat(
                    agent_conv_id,
                    gpts_name,
                    user_query,
                    user_code,
                    sys_code,
                    agent_memory,
                ):
                    agent_task = task
                    # 通过生成器返回聊天消息的chunk
                    yield chunk
            
            except asyncio.CancelledError:
                # 处理客户端断开连接的情况
                print("Client disconnected")
                if agent_task:
                    # 如果有代理任务，取消代理任务
                    logger.info(f"Chat to App {gpts_name}:{agent_conv_id} Cancel!")
                    agent_task.cancel()
            
            except Exception as e:
                # 处理其他异常情况，并记录到日志中
                logger.exception(f"Chat to App {gpts_name} Failed!" + str(e))
                raise
            
            finally:
                # 最终处理步骤，记录最终的代理聊天信息
                logger.info(f"save agent chat info！{conv_uid}")
                
                # 获取稳定的消息，并将其添加到当前会话中
                final_message = await self.stable_message(agent_conv_id)
                if final_message:
                    current_message.add_view_message(final_message)
                
                # 结束当前对话轮次
                current_message.end_current_round()
                
                # 将当前会话信息保存到存储中
                current_message.save_to_storage()

    async def agent_team_chat_new(
        self,
        user_query: str,
        conv_uid: str,
        gpts_app: GptsApp,
        is_retry_chat: bool = False,
        agent_memory: Optional[AgentMemory] = None,
        span_id: Optional[str] = None,
        ):
        # 定义一个空列表，用于存储员工对象
        employees: List[Agent] = []
        # 获取资源管理器对象
        rm = get_resource_manager()
        # 创建一个代理上下文对象，包括对话ID、应用名称和语言信息
        context: AgentContext = AgentContext(
            conv_id=conv_uid,
            gpts_app_name=gpts_app.app_name,
            language=gpts_app.language,
        )

        # 初始化LLM提供者
        ### 初始化聊天参数
        # 从系统应用配置中获取工作管理器工厂组件，并创建工作管理器
        worker_manager = CFG.SYSTEM_APP.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        # 使用默认的LLM客户端创建LLM提供者对象，支持自动消息转换
        self.llm_provider = DefaultLLMClient(worker_manager, auto_convert_message=True)

        # 定义一个可选的资源对象
        depend_resource: Optional[Resource] = None
        # 遍历GPTS应用的详情记录
        for record in gpts_app.details:
            # 根据记录中的代理名称获取对应的代理管理器类
            cls: Type[ConversableAgent] = get_agent_manager().get_by_name(
                record.agent_name
            )
            # 创建LLM配置对象，包括LLM客户端、LLM策略类型和策略上下文
            llm_config = LLMConfig(
                llm_client=self.llm_provider,
                llm_strategy=LLMStrategyType(record.llm_strategy),
                strategy_context=record.llm_strategy_value,
            )
            # 根据记录中的资源信息构建资源对象
            depend_resource = rm.build_resource(record.resources, version="v1")

            # 创建代理对象，并绑定上下文、LLM配置、资源和代理内存
            agent = (
                await cls()
                .bind(context)
                .bind(llm_config)
                .bind(depend_resource)
                .bind(agent_memory)
                .build()
            )
            # 将创建的代理对象添加到员工列表中
            employees.append(agent)

        # 根据GPTS应用的团队模式创建接收者对象
        team_mode = TeamMode(gpts_app.team_mode)
        if team_mode == TeamMode.SINGLE_AGENT:
            recipient = employees[0]
        else:
            # 根据团队模式选择不同的聊天管理器
            llm_config = LLMConfig(llm_client=self.llm_provider)
            if TeamMode.AUTO_PLAN == team_mode:
                manager = AutoPlanChatManager()
            elif TeamMode.AWEL_LAYOUT == team_mode:
                manager = DefaultAWELLayoutManager(dag=gpts_app.team_context)
            else:
                # 抛出错误，指示未知的代理团队模式
                raise ValueError(f"Unknown Agent Team Mode!{team_mode}")
            # 创建选定的聊天管理器，并绑定上下文、LLM配置和代理内存
            manager = (
                await manager.bind(context).bind(llm_config).bind(agent_memory).build()
            )
            # 将员工列表添加到聊天管理器中
            manager.hire(employees)
            recipient = manager

        # 创建用户代理对象，绑定上下文和代理内存
        user_proxy: UserProxyAgent = (
            await UserProxyAgent().bind(context).bind(agent_memory).build()
        )
        # 如果是重试聊天，则更新对话状态为运行中
        if is_retry_chat:
            self.gpts_conversations.update(conv_uid, Status.RUNNING.value)

        try:
            # 使用根跟踪器开始一个跨度，命名为"dbgpt.serve.agent.run_agent"，并指定父跨度ID
            with root_tracer.start_span(
                "dbgpt.serve.agent.run_agent", parent_span_id=span_id
            ):
                # 初始化用户代理的聊天，指定接收者和用户查询消息
                await user_proxy.initiate_chat(
                    recipient=recipient,
                    message=user_query,
                )
        except Exception as e:
            # 捕获异常，并记录错误日志，更新对话状态为失败
            logger.error(f"chat abnormal termination！{str(e)}", e)
            self.gpts_conversations.update(conv_uid, Status.FAILED.value)

        # 更新对话状态为完成
        self.gpts_conversations.update(conv_uid, Status.COMPLETE.value)
        # 返回对话ID
        return conv_uid

    async def chat_messages(
        self, conv_id: str, user_code: str = None, system_app: str = None
    ):
        # 初始化 is_complete 标志为 False
        is_complete = False
        # 进入无限循环
        while True:
            # 从 self.gpts_conversations 根据 conv_id 获取对应的会话对象 gpts_conv
            gpts_conv = self.gpts_conversations.get_by_conv_id(conv_id)
            # 如果找到了对应的会话对象 gpts_conv
            if gpts_conv:
                # 根据会话对象的状态判断会话是否已完成
                is_complete = (
                    True
                    if gpts_conv.state
                    in [
                        Status.COMPLETE.value,
                        Status.WAITING.value,
                        Status.FAILED.value,
                    ]
                    else False
                )
            # 调用 self.memory.one_chat_completions_v2 方法获取一个聊天消息，并将消息返回
            message = await self.memory.one_chat_completions_v2(conv_id)
            # 返回消息
            yield message

            # 如果会话已完成，则退出循环
            if is_complete:
                break
            else:
                # 如果会话未完成，则等待 2 秒钟
                await asyncio.sleep(2)

    async def stable_message(
        self, conv_id: str, user_code: str = None, system_app: str = None
    ):
        # 从 self.gpts_conversations 根据 conv_id 获取对应的会话对象 gpts_conv
        gpts_conv = self.gpts_conversations.get_by_conv_id(conv_id)
        # 如果找到了对应的会话对象 gpts_conv
        if gpts_conv:
            # 根据会话对象的状态判断会话是否已完成
            is_complete = (
                True
                if gpts_conv.state
                in [Status.COMPLETE.value, Status.WAITING.value, Status.FAILED.value]
                else False
            )
            # 如果会话已完成，则直接返回从 self.memory.one_chat_completions_v2 获取的聊天消息
            if is_complete:
                return await self.memory.one_chat_completions_v2(conv_id)
            else:
                # 如果会话未完成，则不执行任何操作
                pass
                # 如果需要抛出异常，可以取消下面的注释
                # raise ValueError(
                #     "The conversation has not been completed yet, so we cannot directly obtain information."
                # )
        else:
            # 如果未找到对应的会话记录，则抛出异常
            raise ValueError("No conversation record found!")

    # 返回通过 user_code 和 system_app 参数获取的会话列表
    def gpts_conv_list(self, user_code: str = None, system_app: str = None):
        return self.gpts_conversations.get_convs(user_code, system_app)
# 创建多代理对象
multi_agents = MultiAgents()

# 定义一个 POST 路由处理函数，用于列出代理信息
@router.post("/v1/dbgpts/agents/list", response_model=Result[Dict[str, str]])
async def agents_list():
    # 记录日志信息，表示函数被调用
    logger.info("agents_list!")
    try:
        # 获取所有代理信息
        agents = get_agent_manager().all_agents()
        # 返回成功结果及代理信息
        return Result.succ(agents)
    except Exception as e:
        # 捕获异常，返回失败结果并记录异常信息
        return Result.failed(code="E30001", msg=str(e))

# 定义一个 GET 路由处理函数，用于获取调试点应用信息
@router.get("/v1/dbgpts/list", response_model=Result[List[GptsApp]])
async def get_dbgpts(user_code: str = None, sys_code: str = None):
    # 记录日志信息，表示函数被调用，并打印用户和系统代码信息
    logger.info(f"get_dbgpts:{user_code},{sys_code}")
    try:
        # 返回成功结果及调试点应用信息列表
        return Result.succ(multi_agents.get_dbgpts())
    except Exception as e:
        # 捕获异常，记录异常信息到日志，并返回失败结果
        logger.error(f"get_dbgpts failed:{str(e)}")
        return Result.failed(msg=str(e), code="E300003")

# 定义一个 POST 路由处理函数，用于获取调试点聊天完成情况
@router.post("/v1/dbgpts/chat/completions", response_model=Result[str])
async def dbgpts_completions(
    gpts_name: str,
    user_query: str,
    conv_id: str = None,
    user_code: str = None,
    sys_code: str = None,
):
    # 记录日志信息，表示函数被调用，并打印 GPT 名称、用户查询和会话 ID 信息
    logger.info(f"dbgpts_completions:{gpts_name},{user_query},{conv_id}")
    if conv_id is None:
        # 如果会话 ID 未提供，则生成一个新的会话 ID
        conv_id = str(uuid.uuid1())

    # 定义响应头信息
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
    }
    # 返回流式响应，调用多代理对象的代理聊天方法
    return StreamingResponse(
        multi_agents.agent_chat(
            agent_conv_id=conv_id,
            gpts_name=gpts_name,
            user_query=user_query,
            user_code=user_code,
            sys_code=sys_code,
        ),
        headers=headers,
        media_type="text/plain",
    )

# 定义一个 POST 路由处理函数，用于取消调试点聊天
@router.post("/v1/dbgpts/chat/cancel", response_model=Result[str])
async def dbgpts_chat_cancel(
    conv_id: str = None, user_code: str = None, sys_code: str = None
):
    # 该函数暂未实现，仅占位符，不执行任何操作
    pass

# 定义一个 POST 路由处理函数，用于提供调试点聊天的反馈
@router.post("/v1/dbgpts/chat/feedback", response_model=Result[str])
async def dbgpts_chat_feedback(filter: PagenationFilter[PluginHubFilter] = Body()):
    # 该函数暂未实现，仅占位符，不执行任何操作
    pass
```