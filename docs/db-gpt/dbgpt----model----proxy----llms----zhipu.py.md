# `.\DB-GPT-src\dbgpt\model\proxy\llms\zhipu.py`

```py
# 导入必要的模块
import os
from concurrent.futures import Executor
from typing import Iterator, Optional

# 导入自定义模块
from dbgpt.core import MessageConverter, ModelOutput, ModelRequest, ModelRequestContext
from dbgpt.model.parameter import ProxyModelParameters
from dbgpt.model.proxy.base import ProxyLLMClient
from dbgpt.model.proxy.llms.proxy_model import ProxyModel

# 定义默认的模型名称
CHATGLM_DEFAULT_MODEL = "chatglm_pro"

# 生成智扑 AI 流的函数
def zhipu_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    """Zhipu ai, see: https://open.bigmodel.cn/dev/api#overview"""
    # 获取模型参数
    model_params = model.get_params()
    print(f"Model: {model}, model_params: {model_params}")

    # 创建智扑 AI 客户端
    client: ZhipuLLMClient = model.proxy_llm_client
    context = ModelRequestContext(stream=True, user_name=params.get("user_name"))
    # 构建模型请求
    request = ModelRequest.build_request(
        client.default_model,
        messages=params["messages"],
        temperature=params.get("temperature"),
        context=context,
        max_new_tokens=params.get("max_new_tokens"),
    )
    # 生成流式数据
    for r in client.sync_generate_stream(request):
        yield r

# 智扑 AI 客户端类
class ZhipuLLMClient(ProxyLLMClient):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model_alias: Optional[str] = "zhipu_proxyllm",
        context_length: Optional[int] = 8192,
        executor: Optional[Executor] = None,
    ):
        # 导入智扑 AI 模块
        try:
            from zhipuai import ZhipuAI
        except ImportError as exc:
            if (
                "No module named" in str(exc)
                or "cannot find module" in str(exc).lower()
            ):
                raise ValueError(
                    "The python package 'zhipuai' is not installed. "
                    "Please install it by running `pip install zhipuai`."
                ) from exc
            else:
                raise ValueError(
                    "Could not import python package: zhipuai "
                    "This may be due to a version that is too low. "
                    "Please upgrade the zhipuai package by running `pip install --upgrade zhipuai`."
                ) from exc
        # 设置默认模型和 API 密钥
        if not model:
            model = CHATGLM_DEFAULT_MODEL
        if not api_key:
            # 兼容 DB-GPT 的配置
            api_key = os.getenv("ZHIPU_PROXY_API_KEY")

        # 初始化智扑 AI 客户端
        self._model = model
        self.client = ZhipuAI(api_key=api_key, base_url=api_base)

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
    # 定义类方法，返回一个名为 "ZhipuLLMClient" 的实例对象
    def create_llm_client(
        model_params: ModelParams,
        default_executor: Executor = default_executor,
    ) -> "ZhipuLLMClient":
        # 使用给定的 model_params 参数和默认执行器创建 ZhipuLLMClient 实例
        return cls(
            model=model_params.proxyllm_backend,
            api_key=model_params.proxy_api_key,
            model_alias=model_params.model_name,
            context_length=model_params.max_context_size,
            executor=default_executor,
        )

    @property
    def default_model(self) -> str:
        # 返回对象的私有属性 _model，作为默认模型的名称字符串
        return self._model

    def sync_generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> Iterator[ModelOutput]:
        # 将传入的 request 对象转换为本地消息格式，使用指定的消息转换器
        request = self.local_covert_message(request, message_converter)

        # 从转换后的 request 中提取普通消息，不支持系统角色
        messages = request.to_common_messages(support_system_role=False)

        # 确定要使用的模型，如果 request 中有指定模型则使用其值，否则使用对象的默认模型
        model = request.model or self._model
        try:
            # 向服务端发送聊天完成请求，返回一个流式响应
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                stream=True,  # 声明返回结果是一个流式的响应
            )
            partial_text = ""
            for chunk in response:
                # 从每个响应块中获取第一个选择项的内容增量，累加到 partial_text 中
                delta_content = chunk.choices[0].delta.content
                partial_text += delta_content
                # 生成一个包含 partial_text 和错误码为 0 的 ModelOutput 对象
                yield ModelOutput(text=partial_text, error_code=0)
        except Exception as e:
            # 如果发生异常，则返回一个包含错误信息和错误码为 1 的 ModelOutput 对象
            return ModelOutput(
                text=f"**LLMServer Generate Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )
```