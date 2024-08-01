# `.\DB-GPT-src\dbgpt\model\proxy\llms\gemini.py`

```py
import os
from concurrent.futures import Executor
from typing import Any, Dict, Iterator, List, Optional, Tuple

from dbgpt.core import (
    MessageConverter,
    ModelMessage,
    ModelOutput,
    ModelRequest,
    ModelRequestContext,
)
from dbgpt.core.interface.message import parse_model_messages
from dbgpt.model.parameter import ProxyModelParameters
from dbgpt.model.proxy.base import ProxyLLMClient
from dbgpt.model.proxy.llms.proxy_model import ProxyModel

# 设置默认的Gemini模型
GEMINI_DEFAULT_MODEL = "gemini-pro"

# 安全设置列表，包含不同类别的安全阈值
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# 生成Gemini流的函数
def gemini_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    # 获取模型参数
    model_params = model.get_params()
    print(f"Model: {model}, model_params: {model_params}")

    # 获取Gemini LLM客户端
    client: GeminiLLMClient = model.proxy_llm_client
    
    # 创建请求上下文
    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))
    
    # 构建模型请求
    request = ModelRequest.build_request(
        client.default_model,
        messages=params["messages"],
        temperature=params.get("temperature"),
        context=context,
        max_new_tokens=params.get("max_new_tokens"),
    )
    
    # 使用客户端同步生成Gemini流
    for r in client.sync_generate_stream(request):
        yield r

# 将消息转换为Gemini格式的函数
def _transform_to_gemini_messages(
    messages: List[ModelMessage],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Transform messages to gemini format

    See https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_python.ipynb

    Args:
        messages (List[ModelMessage]): messages

    Returns:
        Tuple[str, List[Dict[str, Any]]]: user_prompt, gemini_hist

    Examples:
        .. code-block:: python

            messages = [
                ModelMessage(role="human", content="Hello"),
                ModelMessage(role="ai", content="Hi there!"),
                ModelMessage(role="human", content="How are you?"),
            ]
            user_prompt, gemini_hist = _transform_to_gemini_messages(messages)
            assert user_prompt == "How are you?"
            assert gemini_hist == [
                {"role": "user", "parts": {"text": "Hello"}},
                {"role": "model", "parts": {"text": "Hi there!"}},
            ]
    """
    # 解析模型消息
    user_prompt, system_messages, history_messages = parse_model_messages(messages)
    
    # 如果存在系统消息，则抛出异常，Gemini不支持系统角色
    if system_messages:
        raise ValueError("Gemini does not support system role")
    
    # 初始化Gemini历史消息列表
    gemini_hist = []
    # 如果历史消息列表不为空
    if history_messages:
        # 遍历历史消息列表中的每一对用户消息和模型消息
        for user_message, model_message in history_messages:
            # 将用户消息和角色信息添加到 gemini_hist 列表中
            gemini_hist.append({"role": "user", "parts": {"text": user_message}})
            # 将模型消息和角色信息添加到 gemini_hist 列表中
            gemini_hist.append({"role": "model", "parts": {"text": model_message}})
    # 返回用户提示和 gemini_hist 列表
    return user_prompt, gemini_hist
# 创建 GeminiLLMClient 类，继承自 ProxyLLMClient 类
class GeminiLLMClient(ProxyLLMClient):
    
    # 初始化方法，设置类的初始属性
    def __init__(
        self,
        model: Optional[str] = None,                 # 模型名称，可选参数，默认为 None
        api_key: Optional[str] = None,               # API 密钥，可选参数，默认为 None
        api_base: Optional[str] = None,              # API 基础地址，可选参数，默认为 None
        model_alias: Optional[str] = "gemini_proxyllm",  # 模型别名，可选参数，默认为 "gemini_proxyllm"
        context_length: Optional[int] = 8192,        # 上下文长度，可选参数，默认为 8192
        executor: Optional[Executor] = None,         # 执行器，可选参数，默认为 None
    ):
        try:
            import google.generativeai as genai   # 尝试导入 google.generativeai 库
        except ImportError as exc:
            # 如果导入失败，抛出 ValueError 异常并提示安装对应的库
            raise ValueError(
                "Could not import python package: generativeai "
                "Please install dashscope by command `pip install google-generativeai"
            ) from exc
        
        # 如果未指定模型，则使用默认模型 GEMINI_DEFAULT_MODEL
        if not model:
            model = GEMINI_DEFAULT_MODEL
        
        # 设置对象的 API 密钥属性，如果未提供则从环境变量 GEMINI_PROXY_API_KEY 获取
        self._api_key = api_key if api_key else os.getenv("GEMINI_PROXY_API_KEY")
        # 设置对象的 API 基础地址属性，如果未提供则从环境变量 GEMINI_PROXY_API_BASE 获取
        self._api_base = api_base if api_base else os.getenv("GEMINI_PROXY_API_BASE")
        self._model = model   # 设置对象的模型属性
        
        # 如果 API 密钥为空，则抛出 RuntimeError
        if not self._api_key:
            raise RuntimeError("api_key can't be empty")
        
        # 如果指定了 API 基础地址，则配置 genai 的客户端选项
        if self._api_base:
            from google.api_core import client_options
            
            # 创建客户端选项对象并配置 API 终端地址
            client_opts = client_options.ClientOptions(api_endpoint=self._api_base)
            # 配置 genai 库的 API 密钥、传输方式和客户端选项
            genai.configure(
                api_key=self._api_key, transport="rest", client_options=client_opts
            )
        else:
            # 否则仅配置 genai 库的 API 密钥
            genai.configure(api_key=self._api_key)
        
        # 调用父类的初始化方法，传递模型名称列表、上下文长度和执行器
        super().__init__(
            model_names=[model, model_alias],    # 模型名称列表包括当前模型和模型别名
            context_length=context_length,       # 上下文长度
            executor=executor,                   # 执行器
        )

    # 类方法，用于创建新的 GeminiLLMClient 实例
    @classmethod
    def new_client(
        cls,
        model_params: ProxyModelParameters,            # 代理模型参数对象
        default_executor: Optional[Executor] = None,   # 默认执行器，可选参数，默认为 None
    ) -> "GeminiLLMClient":
        # 使用给定的模型参数创建并返回 GeminiLLMClient 实例
        return cls(
            model=model_params.proxyllm_backend,       # 模型名称从代理模型参数中获取
            api_key=model_params.proxy_api_key,        # API 密钥从代理模型参数中获取
            api_base=model_params.proxy_api_base,      # API 基础地址从代理模型参数中获取
            model_alias=model_params.model_name,       # 模型别名从代理模型参数中获取
            context_length=model_params.max_context_size,  # 上下文长度从代理模型参数中获取
            executor=default_executor,                 # 执行器
        )

    # 属性装饰器，返回默认模型的名称
    @property
    def default_model(self) -> str:
        return self._model

    # 同步生成流的方法，接受模型请求对象和消息转换器作为参数
    def sync_generate_stream(
        self,
        request: ModelRequest,                           # 模型请求对象
        message_converter: Optional[MessageConverter] = None,  # 消息转换器，可选参数，默认为 None
        # 下面是方法的具体实现，未提供完整代码
    ) -> Iterator[ModelOutput]:
        # 对传入的请求消息进行本地转换处理，使用指定的消息转换器
        request = self.local_covert_message(request, message_converter)
        try:
            # 尝试导入 Google Generative AI 库
            import google.generativeai as genai

            # 配置生成模型的参数
            generation_config = {
                "temperature": request.temperature,    # 温度参数，控制生成文本的多样性
                "top_p": 1,                            # 基于概率的文本生成控制参数
                "top_k": 1,                            # 基于排序的文本生成控制参数
                "max_output_tokens": request.max_new_tokens,  # 生成文本的最大输出标记数
            }
            # 创建生成模型对象
            model = genai.GenerativeModel(
                model_name=self._model,                # 使用的模型名称
                generation_config=generation_config,    # 生成配置参数
                safety_settings=safety_settings,        # 安全设置参数
            )
            # 将请求消息转换为 Gemini 格式的消息和历史消息
            user_prompt, gemini_hist = _transform_to_gemini_messages(request.messages)
            # 使用模型开始对话，提供历史消息
            chat = model.start_chat(history=gemini_hist)
            # 发送用户提示消息到对话中，使用流式发送
            response = chat.send_message(user_prompt, stream=True)
            text = ""
            # 遍历响应消息的每个片段
            for chunk in response:
                text += chunk.text                     # 将每个片段的文本内容添加到 text 变量中
                yield ModelOutput(text=text, error_code=0)  # 返回生成的文本片段作为 ModelOutput 对象
        except Exception as e:
            # 捕获异常并返回带有错误信息的 ModelOutput 对象
            return ModelOutput(
                text=f"**LLMServer Generate Error, Please CheckErrorInfo.**: {e}",  # 错误信息的格式化输出
                error_code=1,                        # 错误码指示
            )
```