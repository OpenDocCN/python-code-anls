# `.\DB-GPT-src\dbgpt\app\openapi\api_v2.py`

```py
# 导入所需模块和库
import json  # 导入用于 JSON 操作的模块
import re  # 导入正则表达式模块
import time  # 导入时间模块
import uuid  # 导入生成 UUID 的模块
from typing import AsyncIterator, Optional  # 导入类型提示相关模块

from fastapi import (  # 导入 FastAPI 框架的 APIRouter, Body, Depends, HTTPException 等
    APIRouter,
    Body,
    Depends,
    HTTPException,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # 导入 FastAPI 的安全认证相关模块
from starlette.responses import JSONResponse, StreamingResponse  # 导入 Starlette 的响应模块

from dbgpt._private.pydantic import model_to_dict, model_to_json  # 导入 dbgpt 内部的模块
from dbgpt.app.openapi.api_v1.api_v1 import (  # 导入 dbgpt 的 API v1 相关模块和函数
    CHAT_FACTORY,
    __new_conversation,
    get_chat_flow,
    get_chat_instance,
    get_executor,
    stream_generator,
)
from dbgpt.app.scene import BaseChat, ChatScene  # 导入场景相关的基础聊天和聊天场景模块
from dbgpt.client.schema import ChatCompletionRequestBody, ChatMode  # 导入聊天完成请求体和聊天模式相关的模块
from dbgpt.component import logger  # 导入日志记录器模块
from dbgpt.core.awel import CommonLLMHttpRequestBody  # 导入通用的 LLM HTTP 请求体模块
from dbgpt.core.schema.api import (  # 导入 dbgpt 核心的 API 相关的模块和类
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    UsageInfo,
)
from dbgpt.model.cluster.apiserver.api import APISettings  # 导入 API 设置模块
from dbgpt.serve.agent.agents.controller import multi_agents  # 导入多代理控制器模块
from dbgpt.serve.flow.api.endpoints import get_service  # 导入获取服务端点模块
from dbgpt.serve.flow.service.service import Service as FlowService  # 导入服务类别名为 FlowService
from dbgpt.util.executor_utils import blocking_func_to_async  # 导入执行器工具函数转异步的模块
from dbgpt.util.tracer import SpanType, root_tracer  # 导入追踪器相关的模块和类

router = APIRouter()  # 创建 FastAPI 的 API 路由对象
api_settings = APISettings()  # 创建 API 设置对象
get_bearer_token = HTTPBearer(auto_error=False)  # 创建获取 Bearer Token 的对象


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
    service=Depends(get_service),
) -> Optional[str]:
    """检查 API 密钥是否有效
    Args:
        auth (Optional[HTTPAuthorizationCredentials]): Bearer Token 对象.
        service (Service): 流服务对象.
    Raises:
        HTTPException: 如果请求无效.
    """
    if service.config.api_keys:
        # 将配置中的 API 密钥分割并清理空格，形成 API 密钥列表
        api_keys = [key.strip() for key in service.config.api_keys.split(",")]
        # 如果认证对象为空或者认证 token 不在 API 密钥列表中，则抛出 HTTP 异常
        if auth is None or (token := auth.credentials) not in api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token  # 返回有效的 API 密钥 token
    else:
        return None  # 如果没有配置 API 密钥，则返回 None


@router.post("/v2/chat/completions", dependencies=[Depends(check_api_key)])
async def chat_completions(
    request: ChatCompletionRequestBody = Body(),
):
    """处理 V2 版本的聊天完成请求
    Args:
        request (ChatCompletionRequestBody): 聊天请求体对象.
    """
    logger.info(
        f"chat_completions:{request.chat_mode},{request.chat_param},{request.model}"
    )
    headers = {
        "Content-Type": "text/event-stream",  # 设置响应头的Content-Type为text/event-stream，表示服务器端推送事件流
        "Cache-Control": "no-cache",  # 设置缓存控制为no-cache，禁用缓存
        "Connection": "keep-alive",  # 设置连接方式为keep-alive，保持长连接
        "Transfer-Encoding": "chunked",  # 设置传输编码为chunked，分块传输数据
    }
    # 检查聊天请求的有效性
    check_chat_request(request)
    if request.conv_uid is None:
        request.conv_uid = str(uuid.uuid4())  # 如果会话ID为空，则生成一个UUID作为会话ID
    if request.chat_mode == ChatMode.CHAT_APP.value:
        if request.stream is False:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "chat app now not support no stream",  # 抛出HTTP异常，表明当前聊天应用不支持非流式传输
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_request_error",
                    }
                },
            )
        return StreamingResponse(
            chat_app_stream_wrapper(
                request=request,
            ),
            headers=headers,  # 返回基于流的响应，使用预定义的headers
            media_type="text/event-stream",  # 响应的媒体类型为text/event-stream，表示服务器端推送事件流
        )
    elif request.chat_mode == ChatMode.CHAT_AWEL_FLOW.value:
        if not request.stream:
            return await chat_flow_wrapper(request)  # 如果不是流式传输，则调用chat_flow_wrapper处理请求
        else:
            return StreamingResponse(
                chat_flow_stream_wrapper(request),
                headers=headers,  # 返回基于流的响应，使用预定义的headers
                media_type="text/event-stream",  # 响应的媒体类型为text/event-stream，表示服务器端推送事件流
            )
    elif (
        request.chat_mode is None
        or request.chat_mode == ChatMode.CHAT_NORMAL.value
        or request.chat_mode == ChatMode.CHAT_KNOWLEDGE.value
        or request.chat_mode == ChatMode.CHAT_DATA.value
    ):
        with root_tracer.start_span(
            "get_chat_instance",  # 使用分布式跟踪器开始一个名为get_chat_instance的跟踪Span
            span_type=SpanType.CHAT,  # 设置Span的类型为CHAT
            metadata=model_to_dict(request),  # 将请求的数据模型转换为元数据
        ):
            chat: BaseChat = await get_chat_instance(request)  # 异步获取聊天实例

        if not request.stream:
            return await no_stream_wrapper(request, chat)  # 如果不是流式传输，则调用no_stream_wrapper处理请求
        else:
            return StreamingResponse(
                stream_generator(chat, request.incremental, request.model),  # 返回基于流的响应，使用自定义的流生成器
                headers=headers,  # 使用预定义的headers
                media_type="text/plain",  # 响应的媒体类型为text/plain，表示纯文本响应
            )
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "chat mode now only support chat_normal, chat_app, chat_flow, chat_knowledge, chat_data",  # 抛出HTTP异常，表明目前仅支持指定的聊天模式
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_chat_mode",
                }
            },
        )
async def get_chat_instance(dialogue: ChatCompletionRequestBody = Body()) -> BaseChat:
    """
    Get chat instance
    Args:
        dialogue (OpenAPIChatCompletionRequest): The chat request.
    """
    # 记录日志，显示获取聊天实例的对话信息
    logger.info(f"get_chat_instance:{dialogue}")
    
    # 如果对话模式未指定，则使用默认的普通聊天模式
    if not dialogue.chat_mode:
        dialogue.chat_mode = ChatScene.ChatNormal.value()
    
    # 如果对话没有会话 UID，创建新的会话对象并获取其 UID
    if not dialogue.conv_uid:
        conv_vo = __new_conversation(
            dialogue.chat_mode, dialogue.user_name, dialogue.sys_code
        )
        dialogue.conv_uid = conv_vo.conv_uid
    
    # 如果对话模式为特定的数据聊天模式，则设置为带数据库执行的聊天模式
    if dialogue.chat_mode == "chat_data":
        dialogue.chat_mode = ChatScene.ChatWithDbExecute.value()
    
    # 检查对话模式是否为有效模式，如果不是则抛出异常
    if not ChatScene.is_valid_mode(dialogue.chat_mode):
        raise StopAsyncIteration(f"Unsupported Chat Mode,{dialogue.chat_mode}!")
    
    # 构建聊天参数字典
    chat_param = {
        "chat_session_id": dialogue.conv_uid,
        "user_name": dialogue.user_name,
        "sys_code": dialogue.sys_code,
        "current_user_input": dialogue.messages,
        "select_param": dialogue.chat_param,
        "model_name": dialogue.model,
    }
    
    # 使用异步函数将阻塞操作转换为异步操作，获取聊天实例
    chat: BaseChat = await blocking_func_to_async(
        get_executor(),
        CHAT_FACTORY.get_implementation,
        dialogue.chat_mode,
        **{"chat_param": chat_param},
    )
    
    # 返回获取的聊天实例
    return chat


async def no_stream_wrapper(
    request: ChatCompletionRequestBody, chat: BaseChat
) -> ChatCompletionResponse:
    """
    no stream wrapper
    Args:
        request (OpenAPIChatCompletionRequest): request
        chat (BaseChat): chat
    """
    # 使用根跟踪器开始一个命名为"no_stream_generator"的跟踪
    with root_tracer.start_span("no_stream_generator"):
        # 调用聊天实例的非流式调用方法
        response = await chat.nostream_call()
        
        # 处理响应消息中的特殊字符，如替换特殊编码字符和 HTML 实体
        msg = response.replace("\ufffd", "").replace("&quot;", '"')
        
        # 构建选择数据对象，包含角色和消息内容
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=msg),
        )
        
        # 创建用法信息对象
        usage = UsageInfo()
        
        # 返回聊天完成响应对象，包含会话 UID、选择数据、模型名称和用法信息
        return ChatCompletionResponse(
            id=request.conv_uid, choices=[choice_data], model=request.model, usage=usage
        )


async def chat_app_stream_wrapper(request: ChatCompletionRequestBody = None):
    """chat app stream
    Args:
        request (OpenAPIChatCompletionRequest): request
        token (APIToken): token
    """
    # 异步迭代多代理应用程序的聊天方法，生成器模式
    async for output in multi_agents.app_agent_chat(
        conv_uid=request.conv_uid,
        gpts_name=request.chat_param,
        user_query=request.messages,
        user_code=request.user_name,
        sys_code=request.sys_code,
        # 此处缺少了函数的闭合括号，应该在调用此函数时补全
        # 示例中的代码应该在此处添加：
        # )
        # 可以参照下面的示例：
        # )
    # 迭代器函数，生成器函数，用于处理输出流的数据
    ):
        # 在输出中查找以"data:"开头，后跟一个包含JSON数据的对象的正则匹配
        match = re.search(r"data:\s*({.*})", output)
        # 如果找到匹配项
        if match:
            # 从匹配结果中获取JSON字符串
            json_str = match.group(1)
            # 将JSON字符串解析为Python对象
            vis = json.loads(json_str)
            # 获取解析后对象中"vis"键对应的值
            vis_content = vis.get("vis", None)
            # 如果"vis"值不等于"[DONE]"
            if vis_content != "[DONE]":
                # 创建一个ChatCompletionResponseStreamChoice对象，表示助理的选择数据
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant", content=vis.get("vis", None)),
                )
                # 创建一个ChatCompletionStreamResponse对象，表示聊天完成的响应流
                chunk = ChatCompletionStreamResponse(
                    id=request.conv_uid,
                    choices=[choice_data],
                    model=request.model,
                    created=int(time.time()),
                )
                # 将chunk对象转换为JSON格式的字符串
                json_content = model_to_json(
                    chunk, exclude_unset=True, ensure_ascii=False
                )
                # 构建包含JSON数据的数据流内容
                content = f"data: {json_content}\n\n"
                # 生成器函数返回内容
                yield content
    # 最后一个输出，表示处理结束，发送"[DONE]"标记
    yield "data: [DONE]\n\n"
# 异步函数，用于封装聊天流程的请求
async def chat_flow_wrapper(request: ChatCompletionRequestBody):
    # 获取聊天流服务对象
    flow_service = get_chat_flow()
    # 将请求对象转换为字典格式，并创建通用的 HTTP 请求体对象
    flow_req = CommonLLMHttpRequestBody(**model_to_dict(request))
    # 获取聊天流的唯一标识
    flow_uid = request.chat_param
    # 调用安全聊天流处理函数，等待处理结果
    output = await flow_service.safe_chat_flow(flow_uid, flow_req)
    
    # 如果处理失败，则返回包含错误信息的 JSON 响应
    if not output.success:
        return JSONResponse(
            model_to_dict(ErrorResponse(message=output.text, code=output.error_code)),
            status_code=400,
        )
    else:
        # 构建聊天完成的选择数据
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content=output.text),
        )
        # 如果输出中包含使用情况信息，则转换为 UsageInfo 对象；否则创建空的 UsageInfo 对象
        if output.usage:
            usage = UsageInfo(**output.usage)
        else:
            usage = UsageInfo()
        
        # 返回聊天完成的响应对象，包括会话 UID、选择数据、模型信息和使用情况信息
        return ChatCompletionResponse(
            id=request.conv_uid, choices=[choice_data], model=request.model, usage=usage
        )


# 异步生成器函数，用于封装聊天流的流式处理请求
async def chat_flow_stream_wrapper(
    request: ChatCompletionRequestBody,
) -> AsyncIterator[str]:
    """chat app stream
    Args:
        request (OpenAPIChatCompletionRequest): request
    """
    # 获取聊天流服务对象
    flow_service = get_chat_flow()
    # 将请求对象转换为字典格式，并创建通用的 HTTP 请求体对象
    flow_req = CommonLLMHttpRequestBody(**model_to_dict(request))
    # 获取聊天流的唯一标识

    # 使用异步迭代处理聊天流输出，生成流式输出
    async for output in flow_service.chat_stream_openai(flow_uid, flow_req):
        yield output


# 函数，用于检查聊天请求的有效性
def check_chat_request(request: ChatCompletionRequestBody = Body()):
    """
    Check the chat request
    Args:
        request (ChatCompletionRequestBody): The chat request.
    Raises:
        HTTPException: If the request is invalid.
    """
    # 如果聊天模式不为正常模式且未提供聊天参数，则抛出 HTTP 异常
    if request.chat_mode and request.chat_mode != ChatScene.ChatNormal.value():
        if request.chat_param is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "chart param is None",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_chat_param",
                    }
                },
            )
    # 如果未提供模型信息，则抛出 HTTP 异常
    if request.model is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "model is None",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_model",
                }
            },
        )
    # 如果未提供消息内容，则抛出 HTTP 异常
    if request.messages is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "messages is None",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_messages",
                }
            },
        )
```