# `.\DB-GPT-src\dbgpt\app\openapi\api_v1\api_v1.py`

```py
# 异步编程模块 asyncio 的导入
import asyncio
# 日志模块 logging 的导入
import logging
# 操作系统相关功能的导入
import os
# 生成唯一标识符 uuid 的导入
import uuid
# 并发执行的执行器 Executor 的导入
from concurrent.futures import Executor
# 类型提示模块中的 List, Optional, cast 的导入
from typing import List, Optional, cast

# 异步文件操作 aiofiles 的导入
import aiofiles
# FastAPI 框架相关的导入：APIRouter, Body, Depends, File, UploadFile
from fastapi import APIRouter, Body, Depends, File, UploadFile
# FastAPI 返回类型相关的导入：StreamingResponse
from fastapi.responses import StreamingResponse

# 私有配置模块的导入
from dbgpt._private.config import Config
# Pydantic 相关模块的导入：model_to_dict, model_to_json
from dbgpt._private.pydantic import model_to_dict, model_to_json
# 知识请求相关模块的导入
from dbgpt.app.knowledge.request.request import KnowledgeSpaceRequest
# 知识服务相关模块的导入
from dbgpt.app.knowledge.service import KnowledgeService
# API 视图模型相关模块的导入：ChatSceneVo, ConversationVo, MessageVo, Result
from dbgpt.app.openapi.api_view_model import (
    ChatSceneVo,
    ConversationVo,
    MessageVo,
    Result,
)
# 场景相关模块的导入：BaseChat, ChatFactory, ChatScene
from dbgpt.app.scene import BaseChat, ChatFactory, ChatScene
# 组件类型相关模块的导入
from dbgpt.component import ComponentType
# 配置相关的导入：TAG_KEY_KNOWLEDGE_CHAT_DOMAIN_TYPE
from dbgpt.configs import TAG_KEY_KNOWLEDGE_CHAT_DOMAIN_TYPE
# 模型配置相关的导入：KNOWLEDGE_UPLOAD_ROOT_PATH
from dbgpt.configs.model_config import KNOWLEDGE_UPLOAD_ROOT_PATH
# AWEL 核心模块的导入：BaseOperator, CommonLLMHttpRequestBody
from dbgpt.core.awel import BaseOperator, CommonLLMHttpRequestBody
# DAG 管理模块的导入：DAGManager
from dbgpt.core.awel.dag.dag_manager import DAGManager
# 聊天工具相关模块的导入：safe_chat_stream_with_dag_task
from dbgpt.core.awel.util.chat_util import safe_chat_stream_with_dag_task
# API 模型相关模块的导入：ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse, DeltaMessage
from dbgpt.core.schema.api import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)
# 数据库连接信息相关模块的导入：DBConfig, DbTypeInfo
from dbgpt.datasource.db_conn_info import DBConfig, DbTypeInfo
# 基础模型相关的导入：FlatSupportedModel
from dbgpt.model.base import FlatSupportedModel
# 集群模型相关的导入：BaseModelController, WorkerManager, WorkerManagerFactory
from dbgpt.model.cluster import BaseModelController, WorkerManager, WorkerManagerFactory
# 摘要客户端相关的导入：DBSummaryClient
from dbgpt.rag.summary.db_summary_client import DBSummaryClient
# 多代理控制器相关的导入：multi_agents
from dbgpt.serve.agent.agents.controller import multi_agents
# 流服务相关的导入：FlowService
from dbgpt.serve.flow.service.service import Service as FlowService
# 执行器工具相关的导入：DefaultExecutorFactory, ExecutorFactory, blocking_func_to_async
from dbgpt.util.executor_utils import (
    DefaultExecutorFactory,
    ExecutorFactory,
    blocking_func_to_async,
)
# 跟踪器相关的导入：SpanType, root_tracer
from dbgpt.util.tracer import SpanType, root_tracer

# 创建 FastAPI 路由器实例
router = APIRouter()
# 读取配置模块中的配置信息
CFG = Config()
# 创建聊天工厂实例
CHAT_FACTORY = ChatFactory()
# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)
# 创建知识服务实例
knowledge_service = KnowledgeService()

# 全局信号量变量初始化为 None
model_semaphore = None
# 全局计数器初始化为 0
global_counter = 0


# 从会话中获取用户消息的函数
def __get_conv_user_message(conversations: dict):
    messages = conversations["messages"]
    for item in messages:
        if item["type"] == "human":
            return item["data"]["content"]
    return ""


# 创建新对话的函数，返回 ConversationVo 对象
def __new_conversation(chat_mode, user_name: str, sys_code: str) -> ConversationVo:
    unique_id = uuid.uuid1()
    return ConversationVo(
        conv_uid=str(unique_id),
        chat_mode=chat_mode,
        user_name=user_name,
        sys_code=sys_code,
    )


# 获取数据库列表的函数
def get_db_list():
    # 从配置管理器中获取本地数据库列表
    dbs = CFG.local_db_manager.get_db_list()
    # 初始化数据库参数列表
    db_params = []
    # 遍历数据库列表，构建数据库参数字典并添加到列表中
    for item in dbs:
        params: dict = {}
        params.update({"param": item["db_name"]})
        params.update({"type": item["db_type"]})
        db_params.append(params)
    return db_params


# 获取数据库列表信息的函数
def get_db_list_info():
    # 从配置管理器中获取本地数据库列表
    dbs = CFG.local_db_manager.get_db_list()
    # 初始化数据库信息参数字典
    params: dict = {}
    # 遍历数据库列表，如果数据库有注释信息，则添加到参数字典中
    for item in dbs:
        comment = item["comment"]
        if comment is not None and len(comment) > 0:
            params.update({item["db_name"]: comment})
    return params


# 知识列表信息函数，暂未实现
def knowledge_list_info():
    # 这里是一个占位函数，用于获取知识列表信息，但具体实现尚未提供
    pass
    # 创建一个空的字典 params，用于存储知识空间的名称和描述信息
    params: dict = {}
    
    # 创建一个 KnowledgeSpaceRequest 对象，用于发起获取知识空间的请求
    request = KnowledgeSpaceRequest()
    
    # 调用 knowledge_service 的 get_knowledge_space 方法，获取知识空间列表
    spaces = knowledge_service.get_knowledge_space(request)
    
    # 遍历获取到的每个知识空间对象
    for space in spaces:
        # 更新 params 字典，将当前知识空间的名称作为键，描述信息作为值
        params.update({space.name: space.desc})
    
    # 返回包含所有知识空间名称和描述信息的字典 params
    return params
# 返回知识空间列表
def knowledge_list():
    """return knowledge space list"""
    # 创建知识空间请求对象
    request = KnowledgeSpaceRequest()
    # 获取知识空间服务返回的知识空间列表
    spaces = knowledge_service.get_knowledge_space(request)
    # 初始化空间列表
    space_list = []
    # 遍历每个知识空间对象
    for space in spaces:
        # 初始化参数字典
        params: dict = {}
        # 添加空间名称作为参数
        params.update({"param": space.name})
        # 添加类型参数为 "space"
        params.update({"type": "space"})
        # 将参数字典添加到空间列表中
        space_list.append(params)
    # 返回空间列表
    return space_list


# 获取模型控制器对象
def get_model_controller() -> BaseModelController:
    controller = CFG.SYSTEM_APP.get_component(
        ComponentType.MODEL_CONTROLLER, BaseModelController
    )
    return controller


# 获取工作管理器对象
def get_worker_manager() -> WorkerManager:
    worker_manager = CFG.SYSTEM_APP.get_component(
        ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
    ).create()
    return worker_manager


# 获取全局默认的 DAG 管理器对象
def get_dag_manager() -> DAGManager:
    """Get the global default DAGManager"""
    return DAGManager.get_instance(CFG.SYSTEM_APP)


# 获取聊天流程服务对象
def get_chat_flow() -> FlowService:
    """Get Chat Flow Service."""
    return FlowService.get_instance(CFG.SYSTEM_APP)


# 获取全局默认的执行器对象
def get_executor() -> Executor:
    """Get the global default executor"""
    return CFG.SYSTEM_APP.get_component(
        ComponentType.EXECUTOR_DEFAULT,
        ExecutorFactory,
        or_register_component=DefaultExecutorFactory,
    ).create()


# 定义路由，处理 GET 请求 "/v1/chat/db/list"，返回数据库配置列表
@router.get("/v1/chat/db/list", response_model=Result[List[DBConfig]])
async def db_connect_list():
    return Result.succ(CFG.local_db_manager.get_db_list())


# 定义路由，处理 POST 请求 "/v1/chat/db/add"，添加数据库配置
@router.post("/v1/chat/db/add", response_model=Result[bool])
async def db_connect_add(db_config: DBConfig = Body()):
    return Result.succ(CFG.local_db_manager.add_db(db_config))


# 定义路由，处理 POST 请求 "/v1/chat/db/edit"，编辑数据库配置
@router.post("/v1/chat/db/edit", response_model=Result[bool])
async def db_connect_edit(db_config: DBConfig = Body()):
    return Result.succ(CFG.local_db_manager.edit_db(db_config))


# 定义路由，处理 POST 请求 "/v1/chat/db/delete"，删除数据库配置
@router.post("/v1/chat/db/delete", response_model=Result[bool])
async def db_connect_delete(db_name: str = None):
    # 删除指定数据库配置的数据库概要文件
    CFG.local_db_manager.db_summary_client.delete_db_profile(db_name)
    # 删除数据库配置并返回操作结果
    return Result.succ(CFG.local_db_manager.delete_db(db_name))


# 定义路由，处理 POST 请求 "/v1/chat/db/refresh"，刷新数据库配置
@router.post("/v1/chat/db/refresh", response_model=Result[bool])
async def db_connect_refresh(db_config: DBConfig = Body()):
    # 删除指定数据库配置的数据库概要文件
    CFG.local_db_manager.db_summary_client.delete_db_profile(db_config.db_name)
    # 异步更新数据库摘要嵌入
    success = await CFG.local_db_manager.async_db_summary_embedding(
        db_config.db_name, db_config.db_type
    )
    # 返回操作成功与否的结果
    return Result.succ(success)


# 异步执行数据库摘要嵌入
async def async_db_summary_embedding(db_name, db_type):
    db_summary_client = DBSummaryClient(system_app=CFG.SYSTEM_APP)
    db_summary_client.db_summary_embedding(db_name, db_type)


# 定义路由，处理 POST 请求 "/v1/chat/db/test/connect"，测试数据库连接
@router.post("/v1/chat/db/test/connect", response_model=Result[bool])
async def test_connect(db_config: DBConfig = Body()):
    try:
        # TODO 将同步调用改为异步调用
        CFG.local_db_manager.test_connect(db_config)
        # 返回测试连接成功的结果
        return Result.succ(True)
    except Exception as e:
        # 返回测试连接失败的详细信息
        return Result.failed(code="E1001", msg=str(e))
# 定义一个异步接口，用于处理数据库摘要的提交请求，返回布尔类型的结果
@router.post("/v1/chat/db/summary", response_model=Result[bool])
async def db_summary(db_name: str, db_type: str):
    # TODO Change the synchronous call to the asynchronous call
    # 异步调用处理数据库摘要的嵌入
    async_db_summary_embedding(db_name, db_type)
    # 返回处理结果成功的响应
    return Result.succ(True)


# 定义一个异步接口，用于获取数据库支持的类型信息列表
@router.get("/v1/chat/db/support/type", response_model=Result[List[DbTypeInfo]])
async def db_support_types():
    # 获取本地数据库管理器中已完成支持的所有类型
    support_types = CFG.local_db_manager.get_all_completed_types()
    db_type_infos = []
    # 遍历每种支持的类型，创建相应的类型信息对象并加入列表
    for type in support_types:
        db_type_infos.append(
            DbTypeInfo(db_type=type.value(), is_file_db=type.is_file_db())
        )
    # 返回包含类型信息对象列表的成功响应结果
    return Result[DbTypeInfo].succ(db_type_infos)


# 定义一个异步接口，用于获取对话场景列表
@router.post("/v1/chat/dialogue/scenes", response_model=Result[List[ChatSceneVo]])
async def dialogue_scenes():
    scene_vos: List[ChatSceneVo] = []
    # 定义新的对话场景模式列表
    new_modes: List[ChatScene] = [
        ChatScene.ChatWithDbExecute,
        ChatScene.ChatWithDbQA,
        ChatScene.ChatExcel,
        ChatScene.ChatKnowledge,
        ChatScene.ChatDashboard,
        ChatScene.ChatAgent,
    ]
    # 遍历新的对话场景模式列表，创建对应的场景视图对象并加入列表
    for scene in new_modes:
        scene_vo = ChatSceneVo(
            chat_scene=scene.value(),
            scene_name=scene.scene_name(),
            scene_describe=scene.describe(),
            param_title=",".join(scene.param_types()),
            show_disable=scene.show_disable(),
        )
        scene_vos.append(scene_vo)
    # 返回包含对话场景视图对象列表的成功响应结果
    return Result.succ(scene_vos)


# 定义一个异步接口，用于获取不同对话模式下的参数列表或字典
@router.post("/v1/chat/mode/params/list", response_model=Result[dict | list])
async def params_list(chat_mode: str = ChatScene.ChatNormal.value()):
    if ChatScene.ChatWithDbQA.value() == chat_mode:
        return Result.succ(get_db_list())
    elif ChatScene.ChatWithDbExecute.value() == chat_mode:
        return Result.succ(get_db_list())
    elif ChatScene.ChatDashboard.value() == chat_mode:
        return Result.succ(get_db_list())
    elif ChatScene.ChatKnowledge.value() == chat_mode:
        return Result.succ(knowledge_list())
    elif ChatScene.ChatKnowledge.ExtractRefineSummary.value() == chat_mode:
        return Result.succ(knowledge_list())
    else:
        # 默认情况下返回空结果
        return Result.succ(None)


# 定义一个异步接口，用于加载特定对话模式下的参数文件
@router.post("/v1/chat/mode/params/file/load")
async def params_load(
    conv_uid: str,
    chat_mode: str,
    model_name: str,
    user_name: Optional[str] = None,
    sys_code: Optional[str] = None,
    doc_file: UploadFile = File(...),
):
    # 记录参数加载操作的日志信息
    logger.info(f"params_load: {conv_uid},{chat_mode},{model_name}")
    try:
        if doc_file:
            # 如果有上传的文档文件，保存上传的文件到指定目录
            upload_dir = os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, chat_mode)
            # 创建上传目录，如果目录已存在则不进行任何操作
            os.makedirs(upload_dir, exist_ok=True)
            # 构建上传文件的完整路径
            upload_path = os.path.join(upload_dir, doc_file.filename)
            # 使用 aiofiles 异步打开文件，以二进制写入模式
            async with aiofiles.open(upload_path, "wb") as f:
                # 异步读取上传的文件内容，并写入到目标文件中
                await f.write(await doc_file.read())

            # 准备对话参数对象
            dialogue = ConversationVo(
                conv_uid=conv_uid,
                chat_mode=chat_mode,
                select_param=doc_file.filename,
                model_name=model_name,
                user_name=user_name,
                sys_code=sys_code,
            )
            # 获取对应的聊天实例对象
            chat: BaseChat = await get_chat_instance(dialogue)
            # 准备聊天环境
            resp = await chat.prepare()

        # 刷新消息历史记录
        return Result.succ(get_hist_messages(conv_uid))
    except Exception as e:
        # 捕获并记录任何异常，将异常信息记录到日志中
        logger.error("excel load error!", e)
        # 返回一个失败的结果对象，包含错误码和错误信息
        return Result.failed(code="E000X", msg=f"File Load Error {str(e)}")
@router.post("/v1/chat/prepare")
async def chat_prepare(dialogue: ConversationVo = Body()):
    # 执行日志记录，记录函数调用和传入参数
    logger.info(f"chat_prepare:{dialogue}")
    ## 检查对话的 conv_uid 是否存在
    # 调用 get_chat_instance 函数获取 BaseChat 实例
    chat: BaseChat = await get_chat_instance(dialogue)
    # 如果对话历史消息存在，则返回成功的结果
    if chat.has_history_messages():
        return Result.succ(None)
    # 否则调用 chat.prepare() 准备聊天
    resp = await chat.prepare()
    # 返回成功的结果和聊天准备的响应
    return Result.succ(resp)


@router.post("/v1/chat/completions")
async def chat_completions(
    dialogue: ConversationVo = Body(),
    flow_service: FlowService = Depends(get_chat_flow),
):
    # 执行日志记录，记录函数调用和传入参数中的聊天模式、选择参数和模型名称
    logger.info(
        f"chat_completions:{dialogue.chat_mode},{dialogue.select_param},{dialogue.model_name}"
    )
    # 定义响应头部信息，指定内容类型为事件流，无缓存，保持长连接，分块传输编码
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
    }
    # 解析对话的领域类型
    domain_type = _parse_domain_type(dialogue)
    # 如果对话模式为 ChatScene.ChatAgent.value()，则调用 multi_agents.app_agent_chat 处理多代理人聊天
    if dialogue.chat_mode == ChatScene.ChatAgent.value():
        return StreamingResponse(
            multi_agents.app_agent_chat(
                conv_uid=dialogue.conv_uid,
                gpts_name=dialogue.select_param,
                user_query=dialogue.user_input,
                user_code=dialogue.user_name,
                sys_code=dialogue.sys_code,
            ),
            headers=headers,
            media_type="text/event-stream",
        )
    # 如果对话的聊天模式为 ChatFlow 值时执行以下代码块
    elif dialogue.chat_mode == ChatScene.ChatFlow.value():
        # 创建一个 CommonLLMHttpRequestBody 对象，用于流式传输请求
        flow_req = CommonLLMHttpRequestBody(
            model=dialogue.model_name,  # 设置模型名称
            messages=dialogue.user_input,  # 设置用户输入的消息
            stream=True,  # 启用流式传输
            # context=flow_ctx,  # （注释掉的）上下文信息
            # temperature=  # （注释掉的）温度参数
            # max_new_tokens=  # （注释掉的）生成的最大新标记数量
            # enable_vis=  # （注释掉的）是否启用可视化
            conv_uid=dialogue.conv_uid,  # 对话的唯一标识符
            span_id=root_tracer.get_current_span_id(),  # 获取当前跨度的标识符
            chat_mode=dialogue.chat_mode,  # 对话的模式
            chat_param=dialogue.select_param,  # 选择的参数
            user_name=dialogue.user_name,  # 用户名
            sys_code=dialogue.sys_code,  # 系统代码
            incremental=dialogue.incremental,  # 增量标志
        )
        # 返回流式响应，调用 flow_service 的 chat_stream_flow_str 方法
        return StreamingResponse(
            flow_service.chat_stream_flow_str(dialogue.select_param, flow_req),
            headers=headers,
            media_type="text/event-stream",  # 媒体类型为事件流
        )
    # 如果 domain_type 不为空且不为 "Normal" 时执行以下代码块
    elif domain_type is not None and domain_type != "Normal":
        # 返回带有特定领域流的流式响应
        return StreamingResponse(
            chat_with_domain_flow(dialogue, domain_type),
            headers=headers,
            media_type="text/event-stream",  # 媒体类型为事件流
        )
    else:
        # 使用 root_tracer 开始一个名为 "get_chat_instance" 的新跨度
        with root_tracer.start_span(
            "get_chat_instance",
            span_type=SpanType.CHAT,  # 跨度类型为 CHAT
            metadata=model_to_dict(dialogue),  # 将对话模型转换为字典形式的元数据
        ):
            # 异步获取对话的基本聊天实例
            chat: BaseChat = await get_chat_instance(dialogue)

        # 如果 chat 的 prompt_template 的 stream_out 属性为 False 时执行以下代码块
        if not chat.prompt_template.stream_out:
            # 返回不带流的流式响应，使用 no_stream_generator 生成器
            return StreamingResponse(
                no_stream_generator(chat),
                headers=headers,
                media_type="text/event-stream",  # 媒体类型为事件流
            )
        else:
            # 返回带有流的流式响应，使用 stream_generator 生成器
            return StreamingResponse(
                stream_generator(chat, dialogue.incremental, dialogue.model_name),
                headers=headers,
                media_type="text/plain",  # 媒体类型为纯文本
            )
@router.get("/v1/model/types")
async def model_types(controller: BaseModelController = Depends(get_model_controller)):
    # 记录请求日志
    logger.info(f"/controller/model/types")
    try:
        # 初始化一个空集合，用于存储模型类型
        types = set()
        # 从控制器获取所有健康的模型实例
        models = await controller.get_all_instances(healthy_only=True)
        # 遍历每个模型实例
        for model in models:
            # 解析模型名称，获取工作名称和工作类型
            worker_name, worker_type = model.model_name.split("@")
            # 如果工作类型为 "llm"，将工作名称添加到类型集合中
            if worker_type == "llm":
                types.add(worker_name)
        # 返回成功的结果，包含模型类型集合的列表
        return Result.succ(list(types))

    except Exception as e:
        # 如果发生异常，返回失败的结果，包含错误代码和消息
        return Result.failed(code="E000X", msg=f"controller model types error {e}")


@router.get("/v1/model/supports")
async def model_supports(worker_manager: WorkerManager = Depends(get_worker_manager)):
    # 记录请求日志
    logger.info(f"/controller/model/supports")
    try:
        # 获取支持的模型列表
        models = await worker_manager.supported_models()
        # 返回成功的结果，包含平铺后的支持模型信息
        return Result.succ(FlatSupportedModel.from_supports(models))
    except Exception as e:
        # 如果发生异常，返回失败的结果，包含错误代码和消息
        return Result.failed(code="E000X", msg=f"Fetch supportd models error {e}")


async def no_stream_generator(chat):
    # 创建一个新的跟踪 span，用于跟踪此函数执行
    with root_tracer.start_span("no_stream_generator"):
        # 调用 chat 对象的 nostream_call 方法，获取消息
        msg = await chat.nostream_call()
        # 生成器函数，每次产生一条数据流响应
        yield f"data: {msg}\n\n"


async def stream_generator(chat, incremental: bool, model_name: str):
    """Generate streaming responses

    Our goal is to generate an openai-compatible streaming responses.
    Currently, the incremental response is compatible, and the full response will be transformed in the future.

    Args:
        chat (BaseChat): Chat instance.
        incremental (bool): Used to control whether the content is returned incrementally or in full each time.
        model_name (str): The model name

    Yields:
        str: Streaming responses in a specific format
    """
    # 创建一个新的跟踪 span，用于跟踪此函数执行
    span = root_tracer.start_span("stream_generator")
    # 初始化错误消息
    msg = "[LLM_ERROR]: llm server has no output, maybe your prompt template is wrong."

    # 初始化前一次响应内容
    previous_response = ""
    # 异步迭代 chat 对象的 stream_call 方法，获取数据块
    async for chunk in chat.stream_call():
        # 如果有数据块
        if chunk:
            # 处理数据块中的特殊字符
            msg = chunk.replace("\ufffd", "")
            # 如果是增量响应
            if incremental:
                # 计算增量输出内容
                incremental_output = msg[len(previous_response):]
                # 构建响应数据结构
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=incremental_output),
                )
                chunk = ChatCompletionStreamResponse(
                    id=chat.chat_session_id, choices=[choice_data], model=model_name
                )
                # 将响应数据转换为 JSON 格式
                json_chunk = model_to_json(
                    chunk, exclude_unset=True, ensure_ascii=False
                )
                # 产生流响应数据
                yield f"data: {json_chunk}\n\n"
            else:
                # TODO 生成兼容 OpenAI 的流响应
                msg = msg.replace("\n", "\\n")
                yield f"data:{msg}\n\n"
            # 更新前一次响应内容
            previous_response = msg
            # 等待一小段时间，模拟异步操作
            await asyncio.sleep(0.02)
    # 如果是增量响应模式，最后产生一个结束标志
    if incremental:
        yield "data: [DONE]\n\n"
    # 结束当前跟踪 span
    span.end()
# 根据消息字典和其他参数创建一个 MessageVo 对象并返回
def message2Vo(message: dict, order, model_name) -> MessageVo:
    return MessageVo(
        role=message["type"],            # 设置 MessageVo 对象的 role 属性为消息字典中的 "type" 字段值
        context=message["data"]["content"],  # 设置 MessageVo 对象的 context 属性为消息字典中的 "data" 字段的 "content" 字段值
        order=order,                     # 设置 MessageVo 对象的 order 属性为传入的 order 参数
        model_name=model_name,           # 设置 MessageVo 对象的 model_name 属性为传入的 model_name 参数
    )


# 解析对话对象的领域类型，如果是知识对话，则返回对应的领域类型；否则返回 None
def _parse_domain_type(dialogue: ConversationVo) -> Optional[str]:
    if dialogue.chat_mode == ChatScene.ChatKnowledge.value():
        # 在知识对话模式中支持的操作
        space_name = dialogue.select_param   # 获取对话对象的 select_param 属性作为空间名称
        spaces = knowledge_service.get_knowledge_space(
            KnowledgeSpaceRequest(name=space_name)  # 根据空间名称创建知识空间请求对象并获取知识空间信息
        )
        if len(spaces) == 0:
            return Result.failed(
                code="E000X", msg=f"Knowledge space {space_name} not found"  # 如果找不到指定名称的知识空间，则返回失败结果
            )
        if spaces[0].domain_type:
            return spaces[0].domain_type   # 返回找到的第一个知识空间的领域类型
    else:
        return None   # 如果不是知识对话模式，则返回 None


# 在特定领域流程下与对话进行交互
async def chat_with_domain_flow(dialogue: ConversationVo, domain_type: str):
    """Chat with domain flow"""
    dag_manager = get_dag_manager()  # 获取 DAG 管理器实例
    dags = dag_manager.get_dags_by_tag(TAG_KEY_KNOWLEDGE_CHAT_DOMAIN_TYPE, domain_type)  # 根据领域类型获取相关的 DAG 列表
    if not dags or not dags[0].leaf_nodes:
        raise ValueError(f"Cant find the DAG for domain type {domain_type}")  # 如果找不到对应领域类型的 DAG，则抛出异常

    end_task = cast(BaseOperator, dags[0].leaf_nodes[0])  # 获取 DAG 的第一个叶节点作为结束任务

    space = dialogue.select_param  # 获取对话对象的 select_param 属性作为空间名称
    connector_manager = CFG.local_db_manager  # 获取本地数据库管理器实例
    # TODO: Some flow maybe not connector
    db_list = [item["db_name"] for item in connector_manager.get_db_list()]  # 获取数据库列表
    db_names = [item for item in db_list if space in item]  # 过滤出包含空间名称的数据库名称列表
    if len(db_names) == 0:
        raise ValueError(f"fin repost dbname {space}_fin_report not found.")  # 如果找不到符合条件的数据库名称，则抛出异常

    flow_ctx = {"space": space, "db_name": db_names[0]}  # 设置流程上下文信息，包括空间名称和数据库名称
    request = CommonLLMHttpRequestBody(
        model=dialogue.model_name,      # 设置请求体的模型名称为对话对象的 model_name 属性
        messages=dialogue.user_input,   # 设置请求体的消息内容为对话对象的 user_input 属性
        stream=True,                    # 设置请求为流式传输
        extra=flow_ctx,                 # 设置额外的流程上下文信息
        conv_uid=dialogue.conv_uid,     # 设置会话 UID
        span_id=root_tracer.get_current_span_id(),  # 设置跟踪的当前 span ID
        chat_mode=dialogue.chat_mode,   # 设置对话的聊天模式
        chat_param=dialogue.select_param,  # 设置对话的选择参数
        user_name=dialogue.user_name,   # 设置用户名称
        sys_code=dialogue.sys_code,     # 设置系统代码
        incremental=dialogue.incremental,  # 设置增量标志
    )

    async for output in safe_chat_stream_with_dag_task(end_task, request, False):
        text = output.text
        if text:
            text = text.replace("\n", "\\n")  # 替换文本中的换行符为转义字符
        if output.error_code != 0:
            yield f"data:[SERVER_ERROR]{text}\n\n"  # 如果输出有错误码，则生成带有错误信息的数据流
            break
        else:
            yield f"data:{text}\n\n"  # 如果输出正常，则生成正常的数据流
```