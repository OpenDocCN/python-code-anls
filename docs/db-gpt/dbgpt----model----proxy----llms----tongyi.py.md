# `.\DB-GPT-src\dbgpt\model\proxy\llms\tongyi.py`

```py
# 导入日志模块
import logging
# 导入并发执行器接口
from concurrent.futures import Executor
# 导入类型提示工具
from typing import Iterator, Optional

# 导入调试工具中心相关模块
from dbgpt.core import MessageConverter, ModelOutput, ModelRequest, ModelRequestContext
# 导入代理模型参数模块
from dbgpt.model.parameter import ProxyModelParameters
# 导入代理LLM客户端基础模块
from dbgpt.model.proxy.base import ProxyLLMClient
# 导入代理LLM模型相关模块
from dbgpt.model.proxy.llms.proxy_model import ProxyModel

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# 定义函数 tongyi_generate_stream，用于生成同义LLM客户端的数据流
def tongyi_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    # 获取同义LLM客户端对象
    client: TongyiLLMClient = model.proxy_llm_client
    # 创建模型请求上下文，指定为数据流模式，同时传入用户名称
    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))
    # 构建模型请求对象
    request = ModelRequest.build_request(
        client.default_model,  # 使用默认模型
        messages=params["messages"],  # 传入消息内容
        temperature=params.get("temperature"),  # 可选的温度参数
        context=context,  # 使用上述定义的请求上下文
        max_new_tokens=params.get("max_new_tokens"),  # 可选的最大新token数量
    )
    # 使用同义LLM客户端同步生成数据流，通过迭代器返回结果
    for r in client.sync_generate_stream(request):
        yield r


# 定义 TongyiLLMClient 类，继承自 ProxyLLMClient 类
class TongyiLLMClient(ProxyLLMClient):
    # 构造函数，初始化同义LLM客户端
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_region: Optional[str] = None,
        model_alias: Optional[str] = "tongyi_proxyllm",
        context_length: Optional[int] = 4096,
        executor: Optional[Executor] = None,
    ):
        try:
            # 尝试导入 dashscope 库及其相关模块
            import dashscope
            from dashscope import Generation
        except ImportError as exc:
            # 如果导入失败，抛出异常并提示安装 dashscope 库
            raise ValueError(
                "Could not import python package: dashscope "
                "Please install dashscope by command `pip install dashscope"
            ) from exc
        # 如果未指定模型，则使用默认的 Generation 模型
        if not model:
            model = Generation.Models.qwen_turbo
        # 如果提供了 API 密钥，设置 dashscope 的 API 密钥
        if api_key:
            dashscope.api_key = api_key
        # 如果提供了 API 区域信息，设置 dashscope 的 API 区域
        if api_region:
            dashscope.api_region = api_region
        # 保存模型名称到实例变量 _model
        self._model = model

        # 调用父类的构造函数，初始化代理LLM客户端
        super().__init__(
            model_names=[model, model_alias],  # 指定模型名称列表
            context_length=context_length,  # 指定上下文长度
            executor=executor,  # 可选的执行器
        )

    # 类方法，创建一个新的 TongyiLLMClient 客户端实例
    @classmethod
    def new_client(
        cls,
        model_params: ProxyModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "TongyiLLMClient":
        return cls(
            model=model_params.proxyllm_backend,  # 使用代理LLM后端模型
            api_key=model_params.proxy_api_key,  # 使用代理API密钥
            model_alias=model_params.model_name,  # 使用模型名称别名
            context_length=model_params.max_context_size,  # 使用最大上下文大小
            executor=default_executor,  # 可选的执行器
        )

    # 属性方法，返回默认的模型名称
    @property
    def default_model(self) -> str:
        return self._model

    # 方法，使用同义LLM客户端对象同步生成数据流
    def sync_generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> Iterator[ModelOutput]:
        # 导入 Generation 类从 dashscope 模块
        from dashscope import Generation
        
        # 使用本地 covert_message 方法处理请求和消息转换器
        request = self.local_covert_message(request, message_converter)
        
        # 将请求对象转换为通用消息
        messages = request.to_common_messages()
        
        # 获取模型，如果未指定模型则使用默认的 self._model
        model = request.model or self._model
        
        # 尝试执行生成操作
        try:
            # 创建 Generation 实例
            gen = Generation()
            
            # 调用 Generation 实例的 call 方法，传入模型、消息、参数设置等
            res = gen.call(
                model,
                messages=messages,
                top_p=0.8,
                stream=True,
                result_format="message",
            )
            
            # 遍历生成的结果
            for r in res:
                if r:
                    # 检查返回结果的状态码，如果为 200，则提取生成的内容
                    if r["status_code"] == 200:
                        content = r["output"]["choices"][0]["message"].get("content")
                        yield ModelOutput(text=content, error_code=0)
                    else:
                        # 如果状态码不为 200，则返回错误代码和消息内容
                        content = r["code"] + ":" + r["message"]
                        yield ModelOutput(text=content, error_code=-1)
        
        # 捕获可能发生的异常
        except Exception as e:
            # 返回包含错误信息的 ModelOutput 对象
            return ModelOutput(
                text=f"**LLMServer Generate Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )
```