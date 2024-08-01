# `.\DB-GPT-src\dbgpt\rag\embedding\embeddings.py`

```py
"""Embedding implementations."""

# 引入必要的类型声明
from typing import Any, Dict, List, Optional

# 引入异步HTTP请求库和同步HTTP请求库
import aiohttp
import requests

# 引入从私有模块中导入的类和对象
from dbgpt._private.pydantic import EXTRA_FORBID, BaseModel, ConfigDict, Field
from dbgpt.core import Embeddings
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource
from dbgpt.util.i18n_utils import _
from dbgpt.util.tracer import DBGPT_TRACER_SPAN_ID, root_tracer

# 默认的模型名称和说明文本常量
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-large"
DEFAULT_BGE_MODEL = "BAAI/bge-large-en"
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

# 注册资源使用的装饰器，注册了一个HuggingFace嵌入模型资源
@register_resource(
    _("HuggingFace Embeddings"),  # 资源名称的国际化字符串
    "huggingface_embeddings",     # 资源标识符
    category=ResourceCategory.EMBEDDINGS,  # 资源分类
    description=_("HuggingFace sentence_transformers embedding models."),  # 资源描述
    parameters=[
        Parameter.build_from(
            _("Model Name"),       # 参数名称的国际化字符串
            "model_name",          # 参数标识符
            str,                   # 参数类型为字符串
            optional=True,         # 可选参数
            default=DEFAULT_MODEL_NAME,  # 默认值为预设的模型名称
            description=_("Model name to use."),  # 参数描述
        ),
        # TODO, support more parameters
    ],
)
class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Refer to `Langchain Embeddings <https://github.com/langchain-ai/langchain/tree/
    master/libs/langchain/langchain/embeddings>`_.

    Example:
        .. code-block:: python

            from dbgpt.rag.embedding import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
    """

    model_config = ConfigDict(extra=EXTRA_FORBID, protected_namespaces=())

    client: Any  #: :meta private: 客户端对象，私有标记
    model_name: str = DEFAULT_MODEL_NAME  # 使用的模型名称，默认为预设的模型名称
    cache_folder: Optional[str] = Field(None, description="Path of the cache folder.")  # 缓存文件夹的路径，可以为空
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)  # 传递给模型的关键字参数，默认为空字典
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)  # 调用模型`encode`方法时的关键字参数，默认为空字典
    multi_process: bool = False  # 是否在多个GPU上运行`encode()`方法，默认为False，单GPU运行
    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        try:
            import sentence_transformers  # 尝试导入 sentence_transformers 库

        except ImportError as exc:
            # 如果导入失败，抛出 ImportError 异常，提醒用户安装 sentence-transformers 库
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        # 使用传入的参数初始化 client，其中包括一个 SentenceTransformer 对象
        kwargs["client"] = sentence_transformers.SentenceTransformer(
            kwargs.get("model_name") or DEFAULT_MODEL_NAME,  # 使用给定的模型名称或默认模型名称
            cache_folder=kwargs.get("cache_folder"),  # 设置缓存文件夹路径
            **(kwargs.get("model_kwargs") or {}),  # 使用额外的模型参数
        )
        super().__init__(**kwargs)  # 调用父类的初始化方法，传递参数 kwargs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import sentence_transformers  # 导入 sentence_transformers 库

        texts = list(map(lambda x: x.replace("\n", " "), texts))  # 将每个文本中的换行符替换为空格
        if self.multi_process:
            pool = self.client.start_multi_process_pool()  # 如果启用多进程，则启动多进程池
            embeddings = self.client.encode_multi_process(texts, pool)  # 使用多进程编码文本得到 embeddings
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)  # 停止多进程池
        else:
            embeddings = self.client.encode(texts, **self.encode_kwargs)  # 使用单进程编码文本得到 embeddings

        return embeddings.tolist()  # 将 embeddings 转换为列表格式并返回

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]  # 对单个文本进行嵌入并返回其 embeddings
# 注册资源，指定资源名称、类别、描述和参数
@register_resource(
    _("HuggingFace Instructor Embeddings"),
    "huggingface_instructor_embeddings",
    category=ResourceCategory.EMBEDDINGS,
    description=_("HuggingFace Instructor embeddings."),
    parameters=[
        # 定义参数：模型名称
        Parameter.build_from(
            _("Model Name"),
            "model_name",
            str,
            optional=True,
            default=DEFAULT_INSTRUCT_MODEL,
            description=_("Model name to use."),
        ),
        # 定义参数：嵌入指令
        Parameter.build_from(
            _("Embed Instruction"),
            "embed_instruction",
            str,
            optional=True,
            default=DEFAULT_EMBED_INSTRUCTION,
            description=_("Instruction to use for embedding documents."),
        ),
        # 定义参数：查询指令
        Parameter.build_from(
            _("Query Instruction"),
            "query_instruction",
            str,
            optional=True,
            default=DEFAULT_QUERY_INSTRUCTION,
            description=_("Instruction to use for embedding query."),
        ),
    ],
)
# 定义类：HuggingFaceInstructEmbeddings，继承自BaseModel和Embeddings
class HuggingFaceInstructEmbeddings(BaseModel, Embeddings):
    """Wrapper around sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers``
    and ``InstructorEmbedding`` python packages installed.

    Example:
        .. code-block:: python

            from dbgpt.rag.embeddings import HuggingFaceInstructEmbeddings

            model_name = "hkunlp/instructor-large"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            hf = HuggingFaceInstructEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
    """

    # 定义模型配置
    model_config = ConfigDict(extra=EXTRA_FORBID, protected_namespaces=())

    client: Any  #: :meta private:
    model_name: str = DEFAULT_INSTRUCT_MODEL
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME
    environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    embed_instruction: str = DEFAULT_EMBED_INSTRUCTION
    """Instruction to use for embedding documents."""
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    """Instruction to use for embedding query."""
    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        # 尝试导入InstructorEmbedding模块中的INSTRUCTOR类
        try:
            from InstructorEmbedding import INSTRUCTOR
            # 设置关键字参数中的client，使用INSTRUCTOR类创建对象并赋值给client
            kwargs["client"] = INSTRUCTOR(
                kwargs.get("model_name"),
                cache_folder=kwargs.get("cache_folder"),
                **kwargs.get("model_kwargs"),
            )
        # 如果导入失败，则抛出ImportError并附加详细信息
        except ImportError as e:
            raise ImportError("Dependencies for InstructorEmbedding not found.") from e

        # 调用父类的初始化方法，传递所有关键字参数
        super().__init__(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace instruct model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # 创建包含每个文本与embed_instruction组成的列表
        instruction_pairs = [[self.embed_instruction, text] for text in texts]
        # 使用self.client对象对instruction_pairs进行编码，传递额外的编码关键字参数
        embeddings = self.client.encode(instruction_pairs, **self.encode_kwargs)
        # 将结果转换为列表格式并返回
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace instruct model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        # 创建包含查询指令和文本的列表
        instruction_pair = [self.query_instruction, text]
        # 使用self.client对象对instruction_pair进行编码，传递额外的编码关键字参数，并取出第一个结果
        embedding = self.client.encode([instruction_pair], **self.encode_kwargs)[0]
        # 将结果转换为列表格式并返回
        return embedding.tolist()
# TODO: Support AWEL flow
class HuggingFaceBgeEmbeddings(BaseModel, Embeddings):
    """HuggingFace BGE sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    refer to `Langchain Embeddings <https://github.com/langchain-ai/langchain/tree/
    master/libs/langchain/langchain/embeddings>`_.

    Example:
        .. code-block:: python

            from dbgpt.rag.embeddings import HuggingFaceBgeEmbeddings

            model_name = "BAAI/bge-large-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            hf = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
    """

    model_config = ConfigDict(extra=EXTRA_FORBID, protected_namespaces=())

    client: Any  #: :meta private:
    model_name: str = DEFAULT_BGE_MODEL
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN
    """Instruction to use for embedding query."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        try:
            import sentence_transformers  # 尝试导入 sentence_transformers 包

        except ImportError as exc:
            # 如果导入失败，抛出 ImportError 异常并提示安装 sentence_transformers 包
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence_transformers`."
            ) from exc

        # 根据传入的参数初始化 client 属性，使用 sentence_transformers.SentenceTransformer 创建对象
        kwargs["client"] = sentence_transformers.SentenceTransformer(
            kwargs.get("model_name"),
            cache_folder=kwargs.get("cache_folder"),
            **(kwargs.get("model_kwargs") or {}),
        )

        super().__init__(**kwargs)

        # 如果模型名字包含 "-zh"，则修改查询指令为中文版本的默认值
        if "-zh" in self.model_name:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # 将每个文本中的换行符替换为空格
        texts = [t.replace("\n", " ") for t in texts]
        # 使用 client 对象编码文本列表，返回嵌入向量列表
        embeddings = self.client.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()
    def embed_query(self, text: str) -> List[float]:
        """计算使用 HuggingFace transformer 模型的查询嵌入。

        Args:
            text: 要嵌入的文本。

        Returns:
            文本的嵌入向量。
        """
        # 将文本中的换行符替换为空格，处理文本格式
        text = text.replace("\n", " ")
        # 使用预先初始化的客户端对象编码文本与查询指令，生成嵌入向量
        embedding = self.client.encode(
            self.query_instruction + text, **self.encode_kwargs
        )
        # 将嵌入向量转换为列表形式并返回
        return embedding.tolist()
@register_resource(
    _("HuggingFace Inference API Embeddings"),  # 注册资源，显示名称为"HuggingFace Inference API Embeddings"
    "huggingface_inference_api_embeddings",    # 资源标识符为"huggingface_inference_api_embeddings"
    category=ResourceCategory.EMBEDDINGS,      # 资源类别为嵌入类别
    description=_("HuggingFace Inference API embeddings."),  # 资源描述为"HuggingFace Inference API embeddings."
    parameters=[                              # 定义资源的参数列表
        Parameter.build_from(
            _("API Key"),                     # 参数名为"API Key"
            "api_key",                        # 参数标识符为"api_key"
            str,                              # 参数类型为字符串
            description=_("Your API key for the HuggingFace Inference API."),  # 参数描述为"HuggingFace Inference API 的 API 密钥"
        ),
        Parameter.build_from(
            _("Model Name"),                   # 参数名为"Model Name"
            "model_name",                     # 参数标识符为"model_name"
            str,                              # 参数类型为字符串
            optional=True,                    # 参数可选
            default="sentence-transformers/all-MiniLM-L6-v2",  # 默认值为"sentence-transformers/all-MiniLM-L6-v2"
            description=_("The name of the model to use for text embeddings."),  # 参数描述为"用于文本嵌入的模型名称"
        ),
    ],
)
class HuggingFaceInferenceAPIEmbeddings(BaseModel, Embeddings):
    """Embed texts using the HuggingFace API.

    Requires a HuggingFace Inference API key and a model name.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())  # 定义模型配置，允许任意类型，保护命名空间为空

    api_key: str  # 类属性，存储 HuggingFace Inference API 的 API 密钥
    """Your API key for the HuggingFace Inference API."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # 类属性，默认为"sentence-transformers/all-MiniLM-L6-v2"
    """The name of the model to use for text embeddings."""

    @property
    def _api_url(self) -> str:
        return (
            "https://api-inference.huggingface.co"   # 返回 HuggingFace API 的推理 API URL
            "/pipeline"
            "/feature-extraction"
            f"/{self.model_name}"                    # 加上模型名称组成完整的 API URL
        )

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}  # 返回 HTTP 请求头部，包含授权信息

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
                corresponds to a single input text.

        Example:
            .. code-block:: python

                from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

                hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
                    api_key="your_api_key",
                    model_name="sentence-transformers/all-MiniLM-l6-v2",
                )
                texts = ["Hello, world!", "How are you?"]
                hf_embeddings.embed_documents(texts)
        """
        response = requests.post(
            self._api_url,
            headers=self._headers,
            json={
                "inputs": texts,
                "options": {"wait_for_model": True, "use_cache": True},
            },
        )
        return response.json()  # 返回 HTTP 响应的 JSON 内容作为嵌入的结果

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]  # 调用 embed_documents 方法来计算文本的嵌入并返回

def _handle_request_result(res: requests.Response) -> List[List[float]]:
    """Parse the result from a request.
    
    此函数未完全定义，注释暂时留空
    """
    Args:
        res: The response from the request.  # 参数：请求的响应对象

    Returns:
        List[List[float]]: The embeddings.  # 返回类型：嵌入向量的列表

    Raises:
        RuntimeError: If the response is not successful.  # 异常：如果响应不成功，则引发运行时错误
    """
    # 检查响应的状态，如果不是成功状态会抛出异常
    res.raise_for_status()
    # 解析响应的 JSON 数据
    resp = res.json()
    # 如果响应中没有"data"字段，则抛出异常，异常内容为详细信息中的描述
    if "data" not in resp:
        raise RuntimeError(resp["detail"])
    # 获取嵌入向量数据
    embeddings = resp["data"]
    # 根据嵌入向量中的"index"字段排序嵌入向量
    sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])  # type: ignore
    # 返回排序后的嵌入向量列表
    return [result["embedding"] for result in sorted_embeddings]
@register_resource(
    _("Jina AI Embeddings"),  # 注册资源，名称为"Jina AI Embeddings"
    "jina_embeddings",  # 资源标识符为"jina_embeddings"
    category=ResourceCategory.EMBEDDINGS,  # 资源类别为嵌入类别
    description=_("Jina AI embeddings."),  # 描述为"Jina AI embeddings."
    parameters=[  # 参数列表开始
        Parameter.build_from(
            _("API Key"),  # 参数名为"API Key"
            "api_key",  # 参数标识符为"api_key"
            str,  # 参数类型为字符串
            description=_("Your API key for the Jina AI API."),  # 描述为"Your API key for the Jina AI API."
        ),
        Parameter.build_from(
            _("Model Name"),  # 参数名为"Model Name"
            "model_name",  # 参数标识符为"model_name"
            str,  # 参数类型为字符串
            optional=True,  # 参数可选
            default="jina-embeddings-v2-base-en",  # 默认值为"jina-embeddings-v2-base-en"
            description=_("The name of the model to use for text embeddings."),  # 描述为"The name of the model to use for text embeddings."
        ),
    ],  # 参数列表结束
)
class JinaEmbeddings(BaseModel, Embeddings):  # 定义类 JinaEmbeddings，继承自 BaseModel 和 Embeddings
    """Jina AI embeddings.

    This class is used to get embeddings for a list of texts using the Jina AI API.
    It requires an API key and a model name. The default model name is
    "jina-embeddings-v2-base-en".
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())  # 定义模型配置，允许任意类型和保护命名空间

    api_url: Any  #: :meta private:  # API URL，私有属性
    session: Any  #: :meta private:  # 会话对象，私有属性
    api_key: str  # API key，字符串类型
    """API key for the Jina AI API.."""  # API key 的文档字符串
    model_name: str = "jina-embeddings-v2-base-en"  # 模型名称，默认为"jina-embeddings-v2-base-en"
    """The name of the model to use for text embeddings. Defaults to
    "jina-embeddings-v2-base-en"."""  # 模型名称的文档字符串

    def __init__(self, **kwargs):  # 初始化方法
        """Create a new JinaEmbeddings instance."""
        try:  # 尝试导入 requests 库
            import requests
        except ImportError:  # 如果导入失败，抛出 ValueError
            raise ValueError(
                "The requests python package is not installed. Please install it with "
                "`pip install requests`"
            )
        if "api_url" not in kwargs:  # 如果参数中没有提供 api_url
            kwargs["api_url"] = "https://api.jina.ai/v1/embeddings"  # 使用默认的 API URL
        if "session" not in kwargs:  # 如果参数中没有提供 session
            session = requests.Session()  # 创建一个新的会话对象
        else:
            session = kwargs["session"]  # 否则使用参数中传入的会话对象
        api_key = kwargs.get("api_key")  # 获取参数中的 API key
        if api_key:  # 如果 API key 存在
            session.headers.update(  # 更新会话对象的头部信息
                {
                    "Authorization": f"Bearer {api_key}",  # 设置 Authorization 头部
                    "Accept-Encoding": "identity",  # 设置 Accept-Encoding 头部
                }
            )
        kwargs["session"] = session  # 更新参数中的会话对象

        super().__init__(**kwargs)  # 调用父类的初始化方法

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # 嵌入文档方法
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
                corresponds to a single input text.
        """
        # Call Jina AI Embedding API 调用 Jina AI 嵌入 API
        resp = self.session.post(  # 发起 POST 请求
            self.api_url, json={"input": texts, "model": self.model_name}  # 发送的 JSON 数据包括输入文本和模型名称
        )
        return _handle_request_result(resp)  # 处理请求结果并返回
    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a Jina AI embedding model.

        Args:
            text: The text to embed. 传入需要嵌入的文本

        Returns:
            Embeddings for the text. 返回文本的嵌入向量
        """
        # 调用 embed_documents 方法来计算文本的嵌入向量，返回第一个文档的结果
        return self.embed_documents([text])[0]
@register_resource(
    # 注册资源，显示在界面上的名称为"OpenAPI Embeddings"
    _("OpenAPI Embeddings"),
    # 资源标识为"openapi_embeddings"
    "openapi_embeddings",
    # 资源类别为嵌入类型
    category=ResourceCategory.EMBEDDINGS,
    # 资源描述为"OpenAPI embeddings."
    description=_("OpenAPI embeddings."),
    # 定义参数列表
    parameters=[
        # API URL 参数，类型为字符串，默认为本地地址
        Parameter.build_from(
            _("API URL"),
            "api_url",
            str,
            optional=True,
            default="http://localhost:8100/api/v1/embeddings",
            description=_("The URL of the embeddings API."),
        ),
        # API Key 参数，类型为字符串，默认为None
        Parameter.build_from(
            _("API Key"),
            "api_key",
            str,
            optional=True,
            default=None,
            description=_("Your API key for the Open API."),
        ),
        # Model Name 参数，类型为字符串，默认为"text2vec"
        Parameter.build_from(
            _("Model Name"),
            "model_name",
            str,
            optional=True,
            default="text2vec",
            description=_("The name of the model to use for text embeddings."),
        ),
        # Timeout 参数，类型为整数，默认为60秒
        Parameter.build_from(
            _("Timeout"),
            "timeout",
            int,
            optional=True,
            default=60,
            description=_("The timeout for the request in seconds."),
        ),
    ],
)
class OpenAPIEmbeddings(BaseModel, Embeddings):
    """The OpenAPI embeddings.

    This class is used to get embeddings for a list of texts using the API.
    This API is compatible with the OpenAI Embedding API.
    """
    # 定义一个包含配置项的模型配置类，允许任意类型和保护命名空间为空
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
    
    # 定义 API URL 字段，默认为本地 API 服务的嵌入 API 地址
    api_url: str = Field(
        default="http://localhost:8100/api/v1/embeddings",
        description="The URL of the embeddings API.",
    )
    
    # 定义 API 密钥字段，默认为 None，用于访问嵌入 API 的身份验证
    api_key: Optional[str] = Field(
        default=None, description="The API key for the embeddings API."
    )
    
    # 定义模型名称字段，默认为 "text2vec"，指定要使用的模型名称
    model_name: str = Field(
        default="text2vec", description="The name of the model to use."
    )
    
    # 定义超时时间字段，默认为 60 秒，指定 API 请求的超时时间
    timeout: int = Field(
        default=60, description="The timeout for the request in seconds."
    )
    
    # 定义是否传递追踪 ID 到 API 的布尔字段，默认为 True，决定是否将追踪 ID 传递给 API
    pass_trace_id: bool = Field(
        default=True, description="Whether to pass the trace ID to the API."
    )
    
    # 可选的请求会话对象，默认为 None，在执行请求时可用于维护会话状态
    session: Optional[requests.Session] = None
    # 初始化 OpenAPIEmbeddings 类
    def __init__(self, **kwargs):
        """Initialize the OpenAPIEmbeddings."""
        # 尝试导入 requests 模块，如果导入失败则抛出异常
        try:
            import requests
        except ImportError:
            raise ValueError(
                "The requests python package is not installed. "
                "Please install it with `pip install requests`"
            )
        # 检查 kwargs 中是否包含 session，如果没有则创建一个 requests.Session() 对象
        if "session" not in kwargs:  # noqa: SIM401
            session = requests.Session()
        else:
            session = kwargs["session"]
        # 如果存在 api_key，则更新 session 的 headers
        api_key = kwargs.get("api_key")
        if api_key:
            session.headers.update({"Authorization": f"Bearer {api_key}"})
        kwargs["session"] = session
        # 调用父类的初始化方法
        super().__init__(**kwargs)

    # 获取一组文本的嵌入向量
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
                corresponds to a single input text.
        """
        # 调用 OpenAI Embedding API
        headers = {}
        current_span_id = root_tracer.get_current_span_id()
        if self.pass_trace_id and current_span_id:
            # 如果 pass_trace_id 为真且存在当前 span_id，则设置 trace ID
            headers[DBGPT_TRACER_SPAN_ID] = current_span_id
        # 发送 POST 请求获取嵌入向量
        res = self.session.post(  # type: ignore
            self.api_url,
            json={"input": texts, "model": self.model_name},
            timeout=self.timeout,
            headers=headers,
        )
        # 处理请求结果并返回
        return _handle_request_result(res)

    # 使用 OpenAPI 嵌入模型计算查询的嵌入向量
    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a OpenAPI embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        # 调用 embed_documents 方法获取文本的嵌入向量，并返回第一个元素
        return self.embed_documents([text])[0]
    # 异步函数，用于将文本列表嵌入到模型中并返回嵌入向量列表
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: A list of texts to get embeddings for.

        Returns:
            List[List[float]]: Embedded texts as List[List[float]], where each inner
                List[float] corresponds to a single input text.
        """
        # 设置请求头，包含认证信息
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # 获取当前跟踪ID
        current_span_id = root_tracer.get_current_span_id()
        # 如果需要传递跟踪ID并且当前跟踪ID可用，则设置跟踪ID
        if self.pass_trace_id and current_span_id:
            headers[DBGPT_TRACER_SPAN_ID] = current_span_id
        # 使用 aiohttp 客户端会话进行异步请求
        async with aiohttp.ClientSession(
            headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
        ) as session:
            # 发起 POST 请求，发送文本和模型名称作为 JSON 数据
            async with session.post(
                self.api_url, json={"input": texts, "model": self.model_name}
            ) as resp:
                # 检查响应状态，如果不是 200 OK，则抛出异常
                resp.raise_for_status()
                # 解析响应内容为 JSON
                data = await resp.json()
                # 如果响应中不包含 "data" 键，则抛出 RuntimeError 异常
                if "data" not in data:
                    raise RuntimeError(data["detail"])
                # 获取嵌入向量数据
                embeddings = data["data"]
                # 根据 "index" 键对嵌入向量进行排序
                sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])
                # 返回排序后的嵌入向量列表
                return [result["embedding"] for result in sorted_embeddings]

    # 异步函数，用于将查询文本嵌入到模型中并返回嵌入向量
    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        # 调用 aembed_documents 方法将单个文本作为列表传递，并等待返回结果
        embeddings = await self.aembed_documents([text])
        # 返回嵌入向量列表的第一个元素（即单个查询文本的嵌入向量）
        return embeddings[0]
class OllamaEmbeddings(BaseModel, Embeddings):
    """Ollama proxy embeddings.

    This class is used to get embeddings for a list of texts using the Ollama API.
    It requires a proxy server url `api_url` and a model name `model_name`.
    The default model name is "llama2".
    """

    # 定义一个配置字典，允许任意类型，没有受保护的命名空间
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    # API URL，默认为本地地址
    api_url: str = Field(
        default="http://localhost:11434",
        description="The URL of the embeddings API.",
    )

    # 模型名称，默认为 "llama2"
    model_name: str = Field(
        default="llama2", description="The name of the model to use."
    )

    def __init__(self, **kwargs):
        """Initialize the OllamaEmbeddings."""
        super().__init__(**kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
            corresponds to a single input text.
        """
        # 返回文本列表的嵌入向量列表，使用embed_query方法实现
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a OpenAPI embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        try:
            import ollama
            from ollama import Client
        except ImportError as e:
            raise ValueError(
                "Could not import python package: ollama "
                "Please install ollama by command `pip install ollama"
            ) from e
        try:
            # 调用 Ollama API 获取文本的嵌入向量
            return (
                Client(self.api_url).embeddings(model=self.model_name, prompt=text)
            )["embedding"]
        except ollama.ResponseError as e:
            # 处理 Ollama API 响应错误
            raise ValueError(f"**Ollama Response Error, Please CheckErrorInfo.**: {e}")

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: A list of texts to get embeddings for.

        Returns:
            List[List[float]]: Embedded texts as List[List[float]], where each inner
                List[float] corresponds to a single input text.
        """
        embeddings = []
        for text in texts:
            # 异步调用aembed_query方法获取文本的嵌入向量
            embedding = await self.aembed_query(text)
            embeddings.append(embedding)
        return embeddings
    # 异步函数，用于查询文本的嵌入向量
    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        # 尝试导入 ollama 库和 AsyncClient 类
        try:
            import ollama
            from ollama import AsyncClient
        except ImportError:
            # 如果导入失败，抛出异常并提醒用户安装 ollama 库
            raise ValueError(
                "The ollama python package is not installed. "
                "Please install it with `pip install ollama`"
            )
        
        # 使用 ollama 的 AsyncClient 异步请求获取文本的嵌入向量
        try:
            embedding = await AsyncClient(host=self.api_url).embeddings(
                model=self.model_name, prompt=text
            )
            # 返回嵌入向量
            return embedding["embedding"]
        except ollama.ResponseError as e:
            # 如果发生 ollama 的 ResponseError 异常，抛出详细错误信息
            raise ValueError(f"**Ollama Response Error, Please CheckErrorInfo.**: {e}")
class TongYiEmbeddings(BaseModel, Embeddings):
    """The tongyi embeddings.

    import dashscope  # 导入dashscope库
    from http import HTTPStatus  # 导入HTTPStatus枚举
    from dashscope import TextEmbedding  # 导入TextEmbedding类

    dashscope.api_key = ''  # 设置dashscope库的API密钥为空字符串

    def embed_with_list_of_str():
        resp = TextEmbedding.call(
            model=TextEmbedding.Models.text_embedding_v1,
            # 最多支持10条，每条最长支持2048tokens
            input=['风急天高猿啸哀', '渚清沙白鸟飞回', '无边落木萧萧下', '不尽长江滚滚来']
        )
        if resp.status_code == HTTPStatus.OK:
            print(resp)  # 打印响应内容
        else:
            print(resp)  # 打印响应内容

    if __name__ == '__main__':
        embed_with_list_of_str()  # 调用embed_with_list_of_str函数

    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())  # 定义model_config属性
    api_key: Optional[str] = Field(
        default=None, description="The API key for the embeddings API."  # 定义api_key属性
    )
    model_name: str = Field(
        default="text-embedding-v1", description="The name of the model to use."  # 定义model_name属性
    )

    def __init__(self, **kwargs):
        """Initialize the OpenAPIEmbeddings."""
        try:
            import dashscope  # type: ignore  # 尝试导入dashscope库
        except ImportError as exc:
            raise ValueError(
                "Could not import python package: dashscope "
                "Please install dashscope by command `pip install dashscope"
            ) from exc
        dashscope.TextEmbedding.api_key = kwargs.get("api_key")  # 设置dashscope库中TextEmbedding类的api_key属性
        super().__init__(**kwargs)
        self._api_key = kwargs.get("api_key")  # 初始化对象的_api_key属性

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get the embeddings for a list of texts.

        Args:
            texts (Documents): A list of texts to get embeddings for.

        Returns:
            Embedded texts as List[List[float]], where each inner List[float]
                corresponds to a single input text.
        """
        from dashscope import TextEmbedding  # 导入TextEmbedding类（局部导入）

        # 最多支持10条，每条最长支持2048tokens
        resp = TextEmbedding.call(
            model=self.model_name, input=texts, api_key=self._api_key
        )
        if "output" not in resp:
            raise RuntimeError(resp["message"])  # 如果响应中没有output字段，抛出运行时错误

        embeddings = resp["output"]["embeddings"]  # 从响应中获取嵌入向量列表
        sorted_embeddings = sorted(embeddings, key=lambda e: e["text_index"])  # 根据text_index排序嵌入向量列表

        return [result["embedding"] for result in sorted_embeddings]  # 返回排序后的嵌入向量列表

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a OpenAPI embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]  # 调用embed_documents方法获取文本的嵌入向量并返回第一个元素（即单个文本的嵌入向量）
```