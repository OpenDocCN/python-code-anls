# `.\chatglm4-finetune\basic_demo\glm_server.py`

```py
# 导入时间模块，用于时间处理
import time
# 从 asyncio 日志中导入 logger，通常用于异步日志记录
from asyncio.log import logger
# 导入正则表达式模块，用于字符串匹配和操作
import re
# 导入系统模块，提供对 Python 解释器使用或维护的一些变量和函数的访问
import sys
# 导入 Uvicorn，作为 ASGI 服务器运行 FastAPI 应用
import uvicorn
# 导入垃圾回收模块，管理内存使用
import gc
# 导入 JSON 模块，用于处理 JSON 数据
import json
# 导入 PyTorch 库，支持深度学习模型的构建与训练
import torch
# 导入随机模块，提供随机数生成
import random
# 导入字符串模块，处理字符串操作
import string
# 从 vllm 库中导入相关的参数和引擎类
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
# 从 FastAPI 导入应用程序类和异常处理
from fastapi import FastAPI, HTTPException, Response
# 导入 CORS 中间件，处理跨域资源共享
from fastapi.middleware.cors import CORSMiddleware
# 导入异步上下文管理器
from contextlib import asynccontextmanager
# 导入类型注释相关工具
from typing import List, Literal, Optional, Union
# 导入 Pydantic 基类和字段定义工具
from pydantic import BaseModel, Field
# 从 transformers 库中导入相关工具，用于处理自然语言处理模型
from transformers import AutoTokenizer, LogitsProcessor
# 从 SSE Starlette 库中导入事件源响应
from sse_starlette.sse import EventSourceResponse

# 设置默认的事件源响应心跳间隔为 1000 毫秒
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

# 定义最大模型长度为 8192
MAX_MODEL_LENGTH = 8192 

# 定义异步上下文管理器，处理应用的生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 生成上下文，允许在此处执行初始化
    yield
    # 如果可用，清空 CUDA 的缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 执行 CUDA 进程间通信的收集
        torch.cuda.ipc_collect()

# 创建 FastAPI 应用，传入生命周期管理器
app = FastAPI(lifespan=lifespan)

# 添加中间件以处理跨域请求
app.add_middleware(
    CORSMiddleware,
    # 允许所有来源的请求
    allow_origins=["*"],
    # 允许凭据
    allow_credentials=True,
    # 允许所有 HTTP 方法
    allow_methods=["*"],
    # 允许所有请求头
    allow_headers=["*"],
)

# 定义生成唯一 ID 的函数，接受前缀和长度参数
def generate_id(prefix: str, k=29) -> str:
    # 随机生成指定长度的后缀，由字母和数字组成
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    # 返回组合的前缀和后缀
    return f"{prefix}{suffix}"

# 定义模型卡类，表示模型的信息
class ModelCard(BaseModel):
    # 模型 ID，默认为空字符串
    id: str = ""
    # 对象类型，默认为 "model"
    object: str = "model"
    # 创建时间，默认为当前时间戳
    created: int = Field(default_factory=lambda: int(time.time()))
    # 模型拥有者，默认为 "owner"
    owned_by: str = "owner"
    # 根模型的 ID，可选
    root: Optional[str] = None
    # 父模型的 ID，可选
    parent: Optional[str] = None
    # 权限信息，可选
    permission: Optional[list] = None

# 定义模型列表类，表示多个模型的信息
class ModelList(BaseModel):
    # 对象类型，默认为 "list"
    object: str = "list"
    # 包含的模型卡数据，默认为一个包含 "glm-4" 的列表
    data: List[ModelCard] = ["glm-4"]

# 定义函数调用类，表示函数名称和参数
class FunctionCall(BaseModel):
    # 函数名称，可选
    name: Optional[str] = None
    # 函数参数，可选
    arguments: Optional[str] = None

# 定义工具调用的函数参数类
class ChoiceDeltaToolCallFunction(BaseModel):
    # 函数名称，可选
    name: Optional[str] = None
    # 函数参数，可选
    arguments: Optional[str] = None

# 定义使用信息类，记录提示和完成的令牌数量
class UsageInfo(BaseModel):
    # 提示令牌数量，默认为 0
    prompt_tokens: int = 0
    # 总令牌数量，默认为 0
    total_tokens: int = 0
    # 完成令牌数量，可选，默认为 0
    completion_tokens: Optional[int] = 0

# 定义聊天完成消息工具调用类
class ChatCompletionMessageToolCall(BaseModel):
    # 消息索引，可选，默认为 0
    index: Optional[int] = 0
    # 消息 ID，可选
    id: Optional[str] = None
    # 函数调用对象
    function: FunctionCall
    # 消息类型，可选，默认为 "function"
    type: Optional[Literal["function"]] = 'function'

# 定义聊天消息类，表示用户与助手之间的消息
class ChatMessage(BaseModel):
    # “function” 字段解释：
    # 使用较老的OpenAI API版本需要注意在这里添加 function 字段并在 process_messages函数中添加相应角色转换逻辑为 observation
    # 消息角色，取值包括 "user"、"assistant"、"system" 和 "tool"
    role: Literal["user", "assistant", "system", "tool"]
    # 消息内容，可选
    content: Optional[str] = None
    # 函数调用，可选
    function_call: Optional[ChoiceDeltaToolCallFunction] = None
    # 工具调用列表，可选
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

# 定义增量消息类，表示消息的变化
class DeltaMessage(BaseModel):
    # 消息角色，可选，取值包括 "user"、"assistant" 和 "system"
    role: Optional[Literal["user", "assistant", "system"]] = None
    # 消息内容，可选
    content: Optional[str] = None
    # 函数调用，可选
    function_call: Optional[ChoiceDeltaToolCallFunction] = None
    # 工具调用列表，可选
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

# 定义聊天完成响应选择类
class ChatCompletionResponseChoice(BaseModel):
    # 消息索引
    index: int
    # 消息内容
    message: ChatMessage
    # 完成原因，取值包括 "stop"、"length" 和 "tool_calls"
    finish_reason: Literal["stop", "length", "tool_calls"]

# 定义聊天完成响应流选择类
class ChatCompletionResponseStreamChoice(BaseModel):
    # 增量消息
    delta: DeltaMessage
    # 完成原因，取值包括 "stop"、"length" 和 "tool_calls"，可选
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    # 消息索引
    index: int
# 定义一个聊天完成响应类，继承自 BaseModel
class ChatCompletionResponse(BaseModel):
    # 模型名称
    model: str
    # 唯一 ID，使用默认工厂函数生成
    id: Optional[str] = Field(default_factory=lambda: generate_id('chatcmpl-', 29))
    # 对象类型，限制为特定字符串
    object: Literal["chat.completion", "chat.completion.chunk"]
    # 可选项，包含响应选择的列表
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    # 创建时间戳，使用默认工厂函数生成
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    # 系统指纹，使用默认工厂函数生成
    system_fingerprint: Optional[str] = Field(default_factory=lambda: generate_id('fp_', 9))
    # 可选的使用信息
    usage: Optional[UsageInfo] = None


# 定义一个聊天完成请求类，继承自 BaseModel
class ChatCompletionRequest(BaseModel):
    # 模型名称
    model: str
    # 消息列表
    messages: List[ChatMessage]
    # 温度参数，默认为 0.8
    temperature: Optional[float] = 0.8
    # top_p 参数，默认为 0.8
    top_p: Optional[float] = 0.8
    # 最大 token 数，默认为 None
    max_tokens: Optional[int] = None
    # 流式输出标志，默认为 False
    stream: Optional[bool] = False
    # 可选工具，可能是字典或字典列表
    tools: Optional[Union[dict, List[dict]]] = None
    # 可选工具选择，可能是字符串或字典
    tool_choice: Optional[Union[str, dict]] = None
    # 重复惩罚参数，默认为 1.1
    repetition_penalty: Optional[float] = 1.1


# 定义一个无效得分日志处理器，继承自 LogitsProcessor
class InvalidScoreLogitsProcessor(LogitsProcessor):
    # 重载调用方法以处理得分
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # 检查得分是否为 NaN 或无穷大
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            # 将得分置为零
            scores.zero_()
            # 设置特定索引的得分值
            scores[..., 5] = 5e4
        # 返回处理后的得分
        return scores


# 定义一个处理响应的函数
def process_response(output: str, tools: dict | List[dict] = None, use_tool: bool = False) -> Union[str, dict]:
    # 去除输出的多余空白并按行分割
    lines = output.strip().split("\n")
    # 初始化 JSON 参数
    arguments_json = None
    # 定义特殊工具列表
    special_tools = ["cogview", "simple_browser"]
    # 将工具提取为集合，若工具参数为 None 则为空集合
    tools = {tool['function']['name'] for tool in tools} if tools else {}

    # 这是一个简单的工具比较函数，不能保证拦截所有非工具输出的结果，比如参数未对齐等特殊情况。
    ##TODO 如果你希望做更多判断，可以在这里进行逻辑完善。
    # 检查行数是否大于等于2且第二行以"{"开头
        if len(lines) >= 2 and lines[1].startswith("{"):
            # 获取第一行并去除首尾空白，作为函数名
            function_name = lines[0].strip()
            # 将第二行及之后的内容合并为一个字符串，去除首尾空白
            arguments = "\n".join(lines[1:]).strip()
            # 检查函数名是否在工具或特殊工具列表中
            if function_name in tools or function_name in special_tools:
                try:
                    # 尝试将参数字符串解析为 JSON 格式
                    arguments_json = json.loads(arguments)
                    # 标记为工具调用
                    is_tool_call = True
                except json.JSONDecodeError:
                    # 如果解析失败，检查函数名是否在特殊工具中
                    is_tool_call = function_name in special_tools
    
                # 如果确认是工具调用且允许使用工具
                if is_tool_call and use_tool:
                    # 创建内容字典，包含函数名和参数
                    content = {
                        "name": function_name,
                        "arguments": json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments,
                                                ensure_ascii=False)
                    }
                    # 特殊处理 "simple_browser" 函数
                    if function_name == "simple_browser":
                        # 定义正则表达式用于匹配搜索模式
                        search_pattern = re.compile(r'search\("(.+?)"\s*,\s*recency_days\s*=\s*(\d+)\)')
                        # 尝试在参数中匹配搜索模式
                        match = search_pattern.match(arguments)
                        if match:
                            # 如果匹配成功，更新内容字典中的参数
                            content["arguments"] = json.dumps({
                                "query": match.group(1),
                                "recency_days": int(match.group(2))
                            }, ensure_ascii=False)
                    # 特殊处理 "cogview" 函数
                    elif function_name == "cogview":
                        # 更新内容字典中的参数为提示文本
                        content["arguments"] = json.dumps({
                            "prompt": arguments
                        }, ensure_ascii=False)
    
                    # 返回内容字典
                    return content
        # 返回处理后的输出，去除首尾空白
        return output.strip()
# 定义一个异步函数，用于生成流式输出
@torch.inference_mode()
async def generate_stream_glm4(params):
    # 从参数中提取消息
    messages = params["messages"]
    # 从参数中提取工具
    tools = params["tools"]
    # 从参数中提取工具选择
    tool_choice = params["tool_choice"]
    # 获取温度参数，默认为1.0
    temperature = float(params.get("temperature", 1.0))
    # 获取重复惩罚参数，默认为1.0
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    # 获取top_p参数，默认为1.0
    top_p = float(params.get("top_p", 1.0))
    # 获取最大新标记数，默认为8192
    max_new_tokens = int(params.get("max_tokens", 8192))

    # 处理消息并根据工具和选择进行调整
    messages = process_messages(messages, tools=tools, tool_choice=tool_choice)
    # 应用聊天模板，将消息转化为输入格式
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # 创建参数字典，包含生成设置
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "repetition_penalty": repetition_penalty,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop_token_ids": [151329, 151336, 151338],
        "ignore_eos": False,
        "max_tokens": max_new_tokens,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    # 将参数字典转换为SamplingParams对象
    sampling_params = SamplingParams(**params_dict)
    # 异步生成输出，遍历生成的结果
    async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
        # 计算输出的标记长度
        output_len = len(output.outputs[0].token_ids)
        # 计算输入的标记长度
        input_len = len(output.prompt_token_ids)
        # 构建返回结果字典
        ret = {
            "text": output.outputs[0].text,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": output_len + input_len
            },
            "finish_reason": output.outputs[0].finish_reason,
        }
        # 生成结果
        yield ret
    # 垃圾回收，释放内存
    gc.collect()
    # 清空CUDA缓存
    torch.cuda.empty_cache()


# 定义一个函数处理消息，带有可选工具和工具选择
def process_messages(messages, tools=None, tool_choice="none"):
    # 将原始消息存储到变量中
    _messages = messages
    # 创建一个空的处理消息列表
    processed_messages = []
    # 标记消息是否包含系统角色
    msg_has_sys = False

    # 定义一个过滤工具的函数
    def filter_tools(tool_choice, tools):
        # 从工具选择中获取函数名称
        function_name = tool_choice.get('function', {}).get('name', None)
        # 如果没有函数名称，返回空列表
        if not function_name:
            return []
        # 过滤工具，仅保留与函数名称匹配的工具
        filtered_tools = [
            tool for tool in tools
            if tool.get('function', {}).get('name') == function_name
        ]
        return filtered_tools

    # 如果工具选择不为“none”
    if tool_choice != "none":
        # 如果工具选择是字典，进行过滤
        if isinstance(tool_choice, dict):
            tools = filter_tools(tool_choice, tools)
        # 如果有过滤后的工具，添加系统角色消息
        if tools:
            processed_messages.append(
                {
                    "role": "system",
                    "content": None,
                    "tools": tools
                }
            )
            msg_has_sys = True

    # 如果工具选择是字典且存在工具，添加助手角色消息
    if isinstance(tool_choice, dict) and tools:
        processed_messages.append(
            {
                "role": "assistant",
                "metadata": tool_choice["function"]["name"],
                "content": ""
            }
        )
    # 遍历消息列表 _messages 中的每条消息
    for m in _messages:
        # 获取消息的角色、内容和函数调用信息
        role, content, func_call = m.role, m.content, m.function_call
        # 获取消息中的工具调用，如果没有则为 None
        tool_calls = getattr(m, 'tool_calls', None)

        # 如果消息的角色是 "function"
        if role == "function":
            # 将处理后的观察结果添加到列表中，包含角色和内容
            processed_messages.append(
                {
                    "role": "observation",  # 角色设为 "observation"
                    "content": content      # 内容为消息的内容
                }
            )
        # 如果消息的角色是 "tool"
        elif role == "tool":
            # 将处理后的观察结果添加到列表中，包含角色、内容和函数调用标志
            processed_messages.append(
                {
                    "role": "observation",  # 角色设为 "observation"
                    "content": content,     # 内容为消息的内容
                    "function_call": True   # 表示这是一个函数调用
                }
            )
        # 如果消息的角色是 "assistant"
        elif role == "assistant":
            # 如果存在工具调用
            if tool_calls:
                # 遍历每个工具调用
                for tool_call in tool_calls:
                    # 将工具调用的处理结果添加到列表中
                    processed_messages.append(
                        {
                            "role": "assistant",    # 角色设为 "assistant"
                            "metadata": tool_call.function.name,  # 函数名作为元数据
                            "content": tool_call.function.arguments  # 函数参数作为内容
                        }
                    )
            # 如果没有工具调用
            else:
                # 将内容按换行符分割为多个响应
                for response in content.split("\n"):
                    # 如果响应包含换行符
                    if "\n" in response:
                        # 将响应分为元数据和子内容，最多分割一次
                        metadata, sub_content = response.split("\n", maxsplit=1)
                    else:
                        # 如果没有换行符，则元数据为空，子内容为响应
                        metadata, sub_content = "", response
                    # 将处理结果添加到列表中
                    processed_messages.append(
                        {
                            "role": role,           # 角色设为当前消息的角色
                            "metadata": metadata,   # 元数据为解析得到的元数据
                            "content": sub_content.strip()  # 内容为去除前后空格的子内容
                        }
                    )
        # 处理其他角色
        else:
            # 如果角色是 "system" 且 msg_has_sys 为 True
            if role == "system" and msg_has_sys:
                msg_has_sys = False  # 标记系统消息已处理
                continue  # 跳过当前循环，继续下一条消息
            # 添加处理后的消息到列表中
            processed_messages.append({"role": role, "content": content})

    # 如果没有工具或选择的工具为 "none"
    if not tools or tool_choice == "none":
        # 再次遍历消息列表 _messages
        for m in _messages:
            # 如果消息的角色是 'system'
            if m.role == 'system':
                # 将系统消息插入到处理结果的开头
                processed_messages.insert(0, {"role": m.role, "content": m.content})
                break  # 找到后跳出循环
    # 返回处理后的消息列表
    return processed_messages
# 定义健康检查的 HTTP GET 路由
@app.get("/health")
# 定义异步处理健康检查请求的函数，返回 Response 对象
async def health() -> Response:
    """Health check."""  # 函数的文档字符串，描述该函数的用途
    # 返回状态码为 200 的响应，表示服务正常
    return Response(status_code=200)


# 定义获取模型列表的 HTTP GET 路由
@app.get("/v1/models", response_model=ModelList)
# 定义异步处理获取模型列表请求的函数
async def list_models():
    # 创建一个模型卡对象，ID 为 "glm-4"
    model_card = ModelCard(id="glm-4")
    # 返回一个包含模型卡的 ModelList 对象
    return ModelList(data=[model_card])


# 定义创建聊天完成的 HTTP POST 路由
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
# 定义异步处理聊天完成请求的函数，接受 ChatCompletionRequest 类型的请求体
async def create_chat_completion(request: ChatCompletionRequest):
    # 检查消息列表是否为空或最后一条消息角色为 "assistant"
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        # 如果条件不满足，抛出 HTTP 400 异常，表示请求无效
        raise HTTPException(status_code=400, detail="Invalid request")

    # 创建生成参数字典，包含来自请求的各项参数
    gen_params = dict(
        messages=request.messages,  # 消息列表
        temperature=request.temperature,  # 温度参数
        top_p=request.top_p,  # top-p 采样参数
        max_tokens=request.max_tokens or 1024,  # 最大 token 数，默认为 1024
        echo=False,  # 是否回显输入
        stream=request.stream,  # 是否使用流式输出
        repetition_penalty=request.repetition_penalty,  # 重复惩罚参数
        tools=request.tools,  # 可用工具
        tool_choice=request.tool_choice,  # 工具选择
    )
    # 记录生成参数的调试日志
    logger.debug(f"==== request ====\n{gen_params}")

    # 如果请求使用流式输出
    if request.stream:
        # 调用 predict_stream 函数生成预测流生成器
        predict_stream_generator = predict_stream(request.model, gen_params)
        # 获取预测流中的第一个输出
        output = await anext(predict_stream_generator)
        # 如果有输出，返回事件源响应
        if output:
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
        # 记录第一个结果的调试日志
        logger.debug(f"First result output：\n{output}")

        # 初始化函数调用变量
        function_call = None
        # 如果有输出并且请求使用工具
        if output and request.tools:
            try:
                # 处理响应以获取工具调用
                function_call = process_response(output, request.tools, use_tool=True)
            except:
                # 如果解析工具调用失败，记录警告日志
                logger.warning("Failed to parse tool call")

        # 如果函数调用是字典类型
        if isinstance(function_call, dict):
            # 将字典转换为 ChoiceDeltaToolCallFunction 对象
            function_call = ChoiceDeltaToolCallFunction(**function_call)
            # 解析输出文本并生成
            generate = parse_output_text(request.model, output, function_call=function_call)
            # 返回事件源响应
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            # 如果没有有效的函数调用，返回事件源响应
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
    
    # 初始化响应变量
    response = ""
    # 异步生成 GLM4 的流式输出
    async for response in generate_stream_glm4(gen_params):
        pass  # 持续生成直到结束

    # 如果响应文本以换行符开始，去掉第一个换行符
    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    # 去掉响应文本的前后空白
    response["text"] = response["text"].strip()

    # 创建使用信息对象
    usage = UsageInfo()

    # 初始化函数调用和结束原因变量
    function_call, finish_reason = None, "stop"
    # 如果请求使用工具
    tool_calls = None
    if request.tools:
        try:
            # 处理响应以获取工具调用
            function_call = process_response(response["text"], request.tools, use_tool=True)
        except Exception as e:
            # 如果解析工具调用失败，记录警告日志
            logger.warning(f"Failed to parse tool call: {e}")
    # 检查 function_call 是否为字典类型
        if isinstance(function_call, dict):
            # 设置完成原因为工具调用
            finish_reason = "tool_calls"
            # 使用提供的字典参数创建 ChoiceDeltaToolCallFunction 实例
            function_call_response = ChoiceDeltaToolCallFunction(**function_call)
            # 创建 FunctionCall 实例，传入函数名和参数
            function_call_instance = FunctionCall(
                name=function_call_response.name,
                arguments=function_call_response.arguments
            )
            # 创建工具调用列表，包括生成的 FunctionCall 实例
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=generate_id('call_', 24),  # 生成唯一 ID
                    function=function_call_instance,  # 传入函数调用实例
                    type="function")]
    
        # 创建 ChatMessage 实例，内容为响应文本或空，视 tool_calls 是否存在而定
        message = ChatMessage(
            role="assistant",
            content=None if tool_calls else response["text"],  # 如果有工具调用，内容为 None
            function_call=None,  # 没有函数调用
            tool_calls=tool_calls,  # 包含工具调用列表
        )
    
        # 记录调试信息，输出消息内容
        logger.debug(f"==== message ====\n{message}")
    
        # 创建 ChatCompletionResponseChoice 实例，包含索引、消息和完成原因
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,  # 传入之前创建的消息
            finish_reason=finish_reason,  # 传入完成原因
        )
        # 使用模型验证响应中的使用信息
        task_usage = UsageInfo.model_validate(response["usage"])
        # 遍历使用信息，将其添加到使用统计中
        for usage_key, usage_value in task_usage.model_dump().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    
        # 返回 ChatCompletionResponse 实例，包含模型、选择和使用信息
        return ChatCompletionResponse(
            model=request.model,  # 请求的模型
            choices=[choice_data],  # 选择列表包含一个选择数据
            object="chat.completion",  # 对象类型
            usage=usage  # 包含使用信息
        )
# 异步函数，预测流式输出
async def predict_stream(model_id, gen_params):
    # 初始化输出为空字符串
    output = ""
    # 标记是否为函数调用
    is_function_call = False
    # 标记是否已经发送第一个数据块
    has_send_first_chunk = False
    # 获取当前时间戳
    created_time = int(time.time())
    # 初始化函数名称为 None
    function_name = None
    # 生成响应 ID，前缀为 'chatcmpl-'，长度为 29
    response_id = generate_id('chatcmpl-', 29)
    # 生成系统指纹，前缀为 'fp_'，长度为 9
    system_fingerprint = generate_id('fp_', 9)
    # 从生成参数中提取工具名称，若无工具则为空集合
    tools = {tool['function']['name'] for tool in gen_params['tools']} if gen_params['tools'] else {}
    # 初始化增量文本为空
    delta_text = ""
    # 如果是函数调用，返回相应格式的响应
    if is_function_call:
        yield ChatCompletionResponse(
            model=model_id,
            id=response_id,
            system_fingerprint=system_fingerprint,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        content=None,
                        role=None,
                        function_call=None,
                    ),
                    finish_reason="tool_calls"  # 结束原因为工具调用
                )],
            created=created_time,
            object="chat.completion.chunk",  # 指定对象类型为聊天完成块
            usage=None
        ).model_dump_json(exclude_unset=True)  # 转换为 JSON 格式并排除未设置的字段
    # 如果增量文本不为空，则处理增量文本
    elif delta_text != "":
        # 创建增量消息
        message = DeltaMessage(
            content="",
            role="assistant",
            function_call=None,
        )
        # 创建选择数据
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=message,
            finish_reason=None  # 结束原因为空
        )
        # 创建响应块
        chunk = ChatCompletionResponse(
            model=model_id,
            id=response_id,
            choices=[choice_data],
            created=created_time,
            system_fingerprint=system_fingerprint,
            object="chat.completion.chunk"
        )
        yield chunk.model_dump_json(exclude_unset=True)  # 返回响应块的 JSON 格式
        
        # 设置结束原因为 'stop'
        finish_reason = 'stop'
        # 创建增量消息，包含增量文本
        message = DeltaMessage(
            content=delta_text,
            role="assistant",
            function_call=None,
        )
        # 清空增量文本
        delta_text = ""
        # 创建新的选择数据
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=message,
            finish_reason=finish_reason  # 设置结束原因
        )
        # 创建响应块
        chunk = ChatCompletionResponse(
            model=model_id,
            id=response_id,
            choices=[choice_data],
            created=created_time,
            system_fingerprint=system_fingerprint,
            object="chat.completion.chunk"
        )
        yield chunk.model_dump_json(exclude_unset=True)  # 返回响应块的 JSON 格式
        yield '[DONE]'  # 返回完成标识
    else:
        yield '[DONE]'  # 如果没有增量文本，直接返回完成标识

# 异步函数，解析输出文本
async def parse_output_text(model_id: str, value: str, function_call: ChoiceDeltaToolCallFunction = None):
    # 创建增量消息，角色为助手，内容为提供的值
    delta = DeltaMessage(role="assistant", content=value)
    # 如果提供了函数调用，则将其赋值给增量消息
    if function_call is not None:
        delta.function_call = function_call

    # 创建选择数据
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=delta,
        finish_reason=None  # 结束原因为空
    )
    # 创建响应块
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk"  # 指定对象类型为聊天完成块
    )
    # 生成器返回 JSON 格式的字符串，包含模型数据，排除未设置的字段
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
        # 生成器返回 '[DONE]' 字符串，表示处理完成
        yield '[DONE]'
# 当脚本作为主程序运行时执行以下代码
if __name__ == "__main__":
    # 从命令行参数获取模型路径
    MODEL_PATH = sys.argv[1]
    # 加载预训练的分词器，信任远程代码
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # 设置异步引擎的参数，包括模型路径和分词器路径
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        # 设置张量并行的显卡数量，默认为1
        tensor_parallel_size=1,
        # 指定数据类型为半精度
        dtype="half",
        # 信任远程代码
        trust_remote_code=True,
        # 设置 GPU 显存占用比例，根据显卡显存大小调整
        gpu_memory_utilization=0.9,
        # 强制启用 eager 执行
        enforce_eager=True,
        # 禁用 Ray 工作线程
        worker_use_ray=False,
        # 禁用日志请求
        disable_log_requests=True,
        # 设置模型的最大长度
        max_model_len=MAX_MODEL_LENGTH,
    )
    # 从引擎参数创建异步 LLM 引擎
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # 启动 Uvicorn 服务器，监听所有 IP 地址，端口为 8000，使用 1 个工作进程
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
```