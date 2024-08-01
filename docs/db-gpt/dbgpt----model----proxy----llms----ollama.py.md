# `.\DB-GPT-src\dbgpt\model\proxy\llms\ollama.py`

```py
# 导入日志模块
import logging
# 导入并发执行器接口
from concurrent.futures import Executor
# 导入类型提示模块
from typing import Iterator, Optional

# 导入调试器相关模块
from dbgpt.core import MessageConverter, ModelOutput, ModelRequest, ModelRequestContext
from dbgpt.model.parameter import ProxyModelParameters
from dbgpt.model.proxy.base import ProxyLLMClient
from dbgpt.model.proxy.llms.proxy_model import ProxyModel

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def ollama_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=4096
):
    # 获取模型的代理LLM客户端
    client: OllamaLLMClient = model.proxy_llm_client
    # 构建模型请求的上下文
    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))
    # 构建模型请求
    request = ModelRequest.build_request(
        client.default_model,
        messages=params["messages"],
        temperature=params.get("temperature"),
        context=context,
        max_new_tokens=params.get("max_new_tokens"),
    )
    # 使用客户端同步生成流数据
    for r in client.sync_generate_stream(request):
        yield r


class OllamaLLMClient(ProxyLLMClient):
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        model_alias: Optional[str] = "ollama_proxyllm",
        context_length: Optional[int] = 4096,
        executor: Optional[Executor] = None,
    ):
        # 如果未提供模型，则默认使用"llama2"
        if not model:
            model = "llama2"
        # 如果未提供主机，则默认使用"http://localhost:11434"
        if not host:
            host = "http://localhost:11434"
        # 设置模型和主机属性
        self._model = model
        self._host = host

        # 调用父类的构造方法初始化
        super().__init__(
            model_names=[model, model_alias],
            context_length=context_length,
            executor=executor,
        )

    @classmethod
    def new_client(
        cls,
        model_params: ProxyModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "OllamaLLMClient":
        # 创建一个新的客户端实例
        return cls(
            model=model_params.proxyllm_backend,
            host=model_params.proxy_server_url,
            model_alias=model_params.model_name,
            context_length=model_params.max_context_size,
            executor=default_executor,
        )

    @property
    def default_model(self) -> str:
        # 返回默认模型名称
        return self._model

    def sync_generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> Iterator[ModelOutput]:
        try:
            import ollama  # 导入 ollama 包
            from ollama import Client  # 导入 ollama 包中的 Client 类
        except ImportError as e:
            # 如果导入失败，抛出 ValueError 异常，提示安装 ollama 包
            raise ValueError(
                "Could not import python package: ollama "
                "Please install ollama by command `pip install ollama"
            ) from e
        # 调用本地方法 local_covert_message 处理请求中的消息，使用指定的消息转换器
        request = self.local_covert_message(request, message_converter)
        # 将请求对象转换为常见消息对象
        messages = request.to_common_messages()

        # 设置要使用的模型，如果请求中指定了模型则使用请求中的，否则使用对象自身的模型
        model = request.model or self._model
        # 创建与指定主机的 ollama 客户端连接
        client = Client(self._host)
        try:
            # 发起与 ollama 服务的实时聊天流
            stream = client.chat(
                model=model,
                messages=messages,
                stream=True,
            )
            # 初始化内容字符串
            content = ""
            # 遍历流中的每个消息块
            for chunk in stream:
                # 将每个消息块的内容添加到 content 中
                content = content + chunk["message"]["content"]
                # 生成一个 ModelOutput 对象，包含当前累积的内容和错误码 0
                yield ModelOutput(text=content, error_code=0)
        except ollama.ResponseError as e:
            # 如果 ollama 返回了错误，生成一个包含错误信息的 ModelOutput 对象，并设置错误码为 -1
            return ModelOutput(
                text=f"**Ollama Response Error, Please CheckErrorInfo.**: {e}",
                error_code=-1,
            )
```