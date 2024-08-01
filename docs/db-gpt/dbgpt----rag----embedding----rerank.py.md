# `.\DB-GPT-src\dbgpt\rag\embedding\rerank.py`

```py
# 导入所需的模块
from typing import Any, Dict, List, Optional, cast
# 导入 aiohttp 和 numpy 模块
import aiohttp
import numpy as np
import requests
# 导入自定义模块
from dbgpt._private.pydantic import EXTRA_FORBID, BaseModel, ConfigDict, Field
from dbgpt.core import RerankEmbeddings
from dbgpt.util.tracer import DBGPT_TRACER_SPAN_ID, root_tracer

# 定义一个类，继承自 BaseModel 和 RerankEmbeddings
class CrossEncoderRerankEmbeddings(BaseModel, RerankEmbeddings):
    """CrossEncoder Rerank Embeddings."""

    # 定义模型配置
    model_config = ConfigDict(extra=EXTRA_FORBID, protected_namespaces=())

    # 定义属性 client
    client: Any  #: :meta private:
    # 定义属性 model_name，默认值为 "BAAI/bge-reranker-base"
    model_name: str = "BAAI/bge-reranker-base"
    # 定义属性 max_length，默认值为 None
    max_length: Optional[int] = None
    # 定义属性 model_kwargs，默认值为一个空字典
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    # 初始化方法
    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        # 尝试导入 sentence_transformers 模块，如果导入失败则抛出 ImportError
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "please `pip install sentence-transformers`",
            )

        # 设置属性 client 为 CrossEncoder 对象
        kwargs["client"] = CrossEncoder(
            kwargs.get("model_name", "BAAI/bge-reranker-base"),
            max_length=kwargs.get("max_length"),  # type: ignore
            **(kwargs.get("model_kwargs") or {}),
        )
        super().__init__(**kwargs)

    # 定义方法 predict，用于预测候选项的排名分数
    def predict(self, query: str, candidates: List[str]) -> List[float]:
        """Predict the rank scores of the candidates.

        Args:
            query: The query text.
            candidates: The list of candidate texts.

        Returns:
            List[float]: The rank scores of the candidates.
        """
        # 导入 sentence_transformers 模块
        from sentence_transformers import CrossEncoder

        # 将查询文本和候选项组成列表
        query_content_pairs = [[query, candidate] for candidate in candidates]
        # 获取属性 client，并将其转换为 CrossEncoder 类型
        _model = cast(CrossEncoder, self.client)
        # 使用模型预测候选项的排名分数
        rank_scores = _model.predict(sentences=query_content_pairs)
        # 如果排名分数是 numpy 数组，则转换为列表
        if isinstance(rank_scores, np.ndarray):
            rank_scores = rank_scores.tolist()
        return rank_scores  # type: ignore


# 定义另一个类，继承自 BaseModel 和 RerankEmbeddings
class OpenAPIRerankEmbeddings(BaseModel, RerankEmbeddings):
    """OpenAPI Rerank Embeddings."""

    # 定义模型配置
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    # 定义属性 api_url，默认值为 "http://localhost:8100/v1/beta/relevance"
    api_url: str = Field(
        default="http://localhost:8100/v1/beta/relevance",
        description="The URL of the embeddings API.",
    )
    # 定义属性 api_key，默认值为 None
    api_key: Optional[str] = Field(
        default=None, description="The API key for the embeddings API."
    )
    # 定义属性 model_name，默认值为 "bge-reranker-base"
    model_name: str = Field(
        default="bge-reranker-base", description="The name of the model to use."
    )
    # 定义属性 timeout，默认值为 60
    timeout: int = Field(
        default=60, description="The timeout for the request in seconds."
    )
    # 定义属性 pass_trace_id，默认值为 True
    pass_trace_id: bool = Field(
        default=True, description="Whether to pass the trace ID to the API."
    )
    # 定义属性 session，默认值为 None
    session: Optional[requests.Session] = None
    def __init__(self, **kwargs):
        """Initialize the OpenAPIEmbeddings."""
        # 尝试导入 requests 库，如果失败则抛出错误提醒安装
        try:
            import requests
        except ImportError:
            raise ValueError(
                "The requests python package is not installed. "
                "Please install it with `pip install requests`"
            )
        
        # 检查 kwargs 中是否包含 session 参数，如果没有则创建一个新的 requests.Session 对象
        if "session" not in kwargs:  # noqa: SIM401
            session = requests.Session()
        else:
            session = kwargs["session"]
        
        # 如果传入了 api_key，则在 session 的 headers 中添加 Authorization 头部
        api_key = kwargs.get("api_key")
        if api_key:
            session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        # 更新 kwargs 中的 session 参数为新创建或传入的 session 对象
        kwargs["session"] = session
        
        # 调用父类的初始化方法，传入更新后的 kwargs 参数
        super().__init__(**kwargs)

    def predict(self, query: str, candidates: List[str]) -> List[float]:
        """Predict the rank scores of the candidates.

        Args:
            query: The query text.
            candidates: The list of candidate texts.

        Returns:
            List[float]: The rank scores of the candidates.
        """
        # 如果 candidates 列表为空，则直接返回空列表
        if not candidates:
            return []
        
        # 初始化 headers 字典为空
        headers = {}
        
        # 获取当前的跟踪 ID
        current_span_id = root_tracer.get_current_span_id()
        
        # 如果 self.pass_trace_id 为真且 current_span_id 存在，则设置 headers 中的跟踪 ID
        if self.pass_trace_id and current_span_id:
            headers[DBGPT_TRACER_SPAN_ID] = current_span_id
        
        # 构造 POST 请求的 JSON 数据
        data = {"model": self.model_name, "query": query, "documents": candidates}
        
        # 发起 POST 请求到指定的 API URL，使用 session 对象发送请求，设置超时时间和 headers
        response = self.session.post(  # type: ignore
            self.api_url, json=data, timeout=self.timeout, headers=headers
        )
        
        # 检查响应的状态，如果不是成功状态则抛出异常
        response.raise_for_status()
        
        # 返回响应的 JSON 数据中的 "data" 字段内容
        return response.json()["data"]

    async def apredict(self, query: str, candidates: List[str]) -> List[float]:
        """Predict the rank scores of the candidates asynchronously."""
        # 初始化 headers 字典，包含 Authorization 头部信息
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # 获取当前的跟踪 ID
        current_span_id = root_tracer.get_current_span_id()
        
        # 如果 self.pass_trace_id 为真且 current_span_id 存在，则设置 headers 中的跟踪 ID
        if self.pass_trace_id and current_span_id:
            headers[DBGPT_TRACER_SPAN_ID] = current_span_id
        
        # 使用 aiohttp 创建异步的 HTTP 客户端会话对象
        async with aiohttp.ClientSession(
            headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            # 构造 POST 请求的 JSON 数据
            data = {"model": self.model_name, "query": query, "documents": candidates}
            
            # 使用异步会话对象发送 POST 请求到指定的 API URL
            async with session.post(self.api_url, json=data) as resp:
                # 检查响应的状态，如果不是成功状态则抛出异常
                resp.raise_for_status()
                
                # 解析响应的 JSON 数据
                response_data = await resp.json()
                
                # 如果响应数据中不存在 "data" 字段，则抛出运行时异常
                if "data" not in response_data:
                    raise RuntimeError(response_data["detail"])
                
                # 返回响应数据中的 "data" 字段内容
                return response_data["data"]
```