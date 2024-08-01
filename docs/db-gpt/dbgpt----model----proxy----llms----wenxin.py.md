# `.\DB-GPT-src\dbgpt\model\proxy\llms\wenxin.py`

```py
import json  # 导入处理 JSON 数据的模块
import logging  # 导入日志记录模块
import os  # 导入操作系统相关功能的模块
from concurrent.futures import Executor  # 导入并发执行任务的 Executor 类
from typing import Iterator, List, Optional  # 引入类型提示，用于类型注解

import requests  # 导入处理 HTTP 请求的模块
from cachetools import TTLCache, cached  # 导入缓存相关工具

from dbgpt.core import (  # 导入调试 GPT 模型所需的核心功能
    MessageConverter,
    ModelMessage,
    ModelMessageRoleType,
    ModelOutput,
    ModelRequest,
    ModelRequestContext,
)
from dbgpt.model.parameter import ProxyModelParameters  # 导入代理模型参数类
from dbgpt.model.proxy.base import ProxyLLMClient  # 导入代理长文本模型客户端基类
from dbgpt.model.proxy.llms.proxy_model import ProxyModel  # 导入代理模型

# 百度云的模型版本映射表
MODEL_VERSION_MAPPING = {
    "ERNIE-Bot-4.0": "completions_pro",
    "ERNIE-Bot-8K": "ernie_bot_8k",
    "ERNIE-Bot": "completions",
    "ERNIE-Bot-turbo": "eb-instant",
}

_DEFAULT_MODEL = "ERNIE-Bot"  # 设置默认的模型为 ERNIE-Bot

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


@cached(TTLCache(1, 1800))  # 使用缓存装饰器缓存函数结果，有效期为1800秒
def _build_access_token(api_key: str, secret_key: str) -> str:
    """
    Generate Access token according AK, SK
    根据 AK 和 SK 生成访问令牌
    """

    url = "https://aip.baidubce.com/oauth/2.0/token"  # 百度 AI 平台获取访问令牌的 URL
    params = {
        "grant_type": "client_credentials",  # 授权类型为客户端凭证模式
        "client_id": api_key,  # 客户端 ID
        "client_secret": secret_key,  # 客户端密钥
    }

    res = requests.get(url=url, params=params)  # 发起 GET 请求获取访问令牌

    if res.status_code == 200:  # 如果请求成功
        return res.json().get("access_token")  # 返回获取到的访问令牌


def _to_wenxin_messages(messages: List[ModelMessage]):
    """Convert messages to wenxin compatible format

    将消息转换为与稳信兼容的格式
    See https://cloud.baidu.com/doc/WENXINWORKSHOP/s/jlil56u11
    查看文档了解稳信兼容格式的详细信息
    """
    wenxin_messages = []  # 初始化稳信消息列表
    system_messages = []  # 初始化系统消息列表
    for message in messages:  # 遍历输入的消息列表
        if message.role == ModelMessageRoleType.HUMAN:  # 如果消息角色为人类
            wenxin_messages.append({"role": "user", "content": message.content})  # 添加用户角色消息到稳信消息列表
        elif message.role == ModelMessageRoleType.SYSTEM:  # 如果消息角色为系统
            system_messages.append(message.content)  # 添加系统消息到系统消息列表
        elif message.role == ModelMessageRoleType.AI:  # 如果消息角色为 AI
            wenxin_messages.append({"role": "assistant", "content": message.content})  # 添加助手角色消息到稳信消息列表
        else:
            pass
    if len(system_messages) > 1:  # 如果系统消息数量大于1
        raise ValueError("Wenxin only support one system message")  # 抛出异常，稳信仅支持一个系统消息
    str_system_message = system_messages[0] if len(system_messages) > 0 else ""  # 获取系统消息的字符串表示，如果不存在则为空字符串
    return wenxin_messages, str_system_message  # 返回转换后的稳信消息列表和系统消息字符串


def wenxin_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    client: WenxinLLMClient = model.proxy_llm_client  # 获取代理长文本模型的客户端
    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))  # 创建模型请求上下文对象
    request = ModelRequest.build_request(
        client.default_model,  # 使用客户端默认模型
        messages=params["messages"],  # 使用传入的消息列表
        temperature=params.get("temperature"),  # 获取温度参数
        context=context,  # 使用创建的请求上下文
        max_new_tokens=params.get("max_new_tokens"),  # 获取最大新令牌数参数
    )
    for r in client.sync_generate_stream(request):  # 使用客户端同步生成流请求
        yield r  # 返回生成的结果


class WenxinLLMClient(ProxyLLMClient):
    # 稳信长文本模型客户端类，继承自代理长文本模型客户端基类
    # 初始化函数，用于实例化 WenxinLLMClient 类的对象
    def __init__(
        self,
        model: Optional[str] = None,  # 模型名称，默认为 None
        api_key: Optional[str] = None,  # API 密钥，默认为 None
        api_secret: Optional[str] = None,  # API 密钥，默认为 None
        model_version: Optional[str] = None,  # 模型版本，默认为 None
        model_alias: Optional[str] = "wenxin_proxyllm",  # 模型别名，默认为 "wenxin_proxyllm"
        context_length: Optional[int] = 8192,  # 上下文长度，默认为 8192
        executor: Optional[Executor] = None,  # 执行器对象，默认为 None
    ):
        # 如果 model 为空，则使用默认模型 _DEFAULT_MODEL
        if not model:
            model = _DEFAULT_MODEL
        # 如果 api_key 为空，则尝试从环境变量 WEN_XIN_API_KEY 中获取
        if not api_key:
            api_key = os.getenv("WEN_XIN_API_KEY")
        # 如果 api_secret 为空，则尝试从环境变量 WEN_XIN_API_SECRET 中获取
        if not api_secret:
            api_secret = os.getenv("WEN_XIN_API_SECRET")
        # 如果 model_version 为空
        if not model_version:
            # 如果 model 不为空，则尝试从 MODEL_VERSION_MAPPING 中获取对应的版本号
            if model:
                model_version = MODEL_VERSION_MAPPING.get(model)
            # 否则，尝试从环境变量 WEN_XIN_MODEL_VERSION 中获取
            else:
                model_version = os.getenv("WEN_XIN_MODEL_VERSION")
        # 如果 api_key 为空，则抛出 ValueError 异常
        if not api_key:
            raise ValueError("api_key can't be empty")
        # 如果 api_secret 为空，则抛出 ValueError 异常
        if not api_secret:
            raise ValueError("api_secret can't be empty")
        # 如果 model_version 为空，则抛出 ValueError 异常
        if not model_version:
            raise ValueError("model_version can't be empty")
        
        # 设置对象的属性值
        self._model = model
        self._api_key = api_key
        self._api_secret = api_secret
        self._model_version = model_version

        # 调用父类的初始化方法，设置模型名称和上下文长度，并指定执行器对象
        super().__init__(
            model_names=[model, model_alias],
            context_length=context_length,
            executor=executor,
        )

    # 类方法，用于创建一个新的 WenxinLLMClient 客户端对象
    @classmethod
    def new_client(
        cls,
        model_params: ProxyModelParameters,  # 代理模型参数对象
        default_executor: Optional[Executor] = None,  # 默认的执行器对象，默认为 None
    ) -> "WenxinLLMClient":
        # 使用给定的代理模型参数创建一个新的 WenxinLLMClient 对象
        return cls(
            model=model_params.proxyllm_backend,  # 模型名称
            api_key=model_params.proxy_api_key,  # API 密钥
            api_secret=model_params.proxy_api_secret,  # API 密钥
            model_version=model_params.proxy_api_version,  # 模型版本
            model_alias=model_params.model_name,  # 模型别名
            context_length=model_params.max_context_size,  # 上下文长度
            executor=default_executor,  # 执行器对象
        )

    # 属性方法，返回默认的模型名称
    @property
    def default_model(self) -> str:
        return self._model

    # 同步生成流的方法，用于向模型请求生成文本
    def sync_generate_stream(
        self,
        request: ModelRequest,  # 模型请求对象
        message_converter: Optional[MessageConverter] = None,  # 消息转换器对象，默认为 None
    # 定义一个生成器函数，返回一个迭代器，每次生成一个 ModelOutput 对象
    ) -> Iterator[ModelOutput]:
        # 将请求消息转换为本地消息格式
        request = self.local_covert_message(request, message_converter)

        try:
            # 构建访问令牌
            access_token = _build_access_token(self._api_key, self._api_secret)

            # 设置请求头
            headers = {"Content-Type": "application/json", "Accept": "application/json"}

            # 构建代理服务器的 URL
            proxy_server_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{self._model_version}?access_token={access_token}"

            # 如果没有访问令牌，则抛出运行时错误
            if not access_token:
                raise RuntimeError(
                    "Failed to get access token. please set the correct api_key and secret key."
                )

            # 将请求消息转换为温馨消息格式
            history, system_message = _to_wenxin_messages(request.get_messages())
            # 构建请求的 payload
            payload = {
                "messages": history,
                "system": system_message,
                "temperature": request.temperature,
                "stream": True,
            }

            text = ""
            # 发送 POST 请求到代理服务器
            res = requests.post(
                proxy_server_url, headers=headers, json=payload, stream=True
            )
            # 记录日志
            logger.info(
                f"Send request to {proxy_server_url} with real model {self._model}, model version {self._model_version}"
            )
            # 遍历响应的每一行数据
            for line in res.iter_lines():
                if line:
                    if not line.startswith(b"data: "):
                        # 如果行不以"data: "开头，则解码错误消息并生成 ModelOutput 对象
                        error_message = line.decode("utf-8")
                        yield ModelOutput(text=error_message, error_code=1)
                    else:
                        # 解析 JSON 数据并提取内容，生成 ModelOutput 对象
                        json_data = line.split(b": ", 1)[1]
                        decoded_line = json_data.decode("utf-8")
                        if decoded_line.lower() != "[DONE]".lower():
                            obj = json.loads(json_data)
                            if obj["result"] is not None:
                                content = obj["result"]
                                text += content
                        yield ModelOutput(text=text, error_code=0)
        except Exception as e:
            # 如果发生异常，则返回包含错误信息的 ModelOutput 对象
            return ModelOutput(
                text=f"**LLMServer Generate Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )
```