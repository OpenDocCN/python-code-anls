# `.\graphrag\graphrag\config\create_graphrag_config.py`

```py
# 版权声明和许可信息
# 2024 年由 Microsoft Corporation 发布，根据 MIT 许可证授权

"""从环境变量加载默认配置的参数化设置。"""

# 导入必要的库
import os
from enum import Enum
from pathlib import Path
from typing import cast

# 导入外部库和模块
from datashaper import AsyncType
from environs import Env
from pydantic import TypeAdapter

# 导入默认配置
import graphrag.config.defaults as defs

# 导入枚举类型
from .enums import (
    CacheType,
    InputFileType,
    InputType,
    LLMType,
    ReportingType,
    StorageType,
    TextEmbeddingTarget,
)

# 导入环境读取模块和自定义错误模块
from .environment_reader import EnvironmentReader
from .errors import (
    ApiKeyMissingError,
    AzureApiBaseMissingError,
    AzureDeploymentNameMissingError,
)

# 导入输入模型
from .input_models import (
    GraphRagConfigInput,
    LLMConfigInput,
)

# 导入数据模型
from .models import (
    CacheConfig,
    ChunkingConfig,
    ClaimExtractionConfig,
    ClusterGraphConfig,
    CommunityReportsConfig,
    EmbedGraphConfig,
    EntityExtractionConfig,
    GlobalSearchConfig,
    GraphRagConfig,
    InputConfig,
    LLMParameters,
    LocalSearchConfig,
    ParallelizationParameters,
    ReportingConfig,
    SnapshotsConfig,
    StorageConfig,
    SummarizeDescriptionsConfig,
    TextEmbeddingConfig,
    UmapConfig,
)

# 导入 dotenv 读取函数
from .read_dotenv import read_dotenv

# 定义输入模型验证器
InputModelValidator = TypeAdapter(GraphRagConfigInput)

# 创建函数 create_graphrag_config，用于生成 GraphRagConfig 对象
def create_graphrag_config(
    values: GraphRagConfigInput | None = None, root_dir: str | None = None
) -> GraphRagConfig:
    """从字典加载配置参数。

    Args:
        values: 包含配置参数的字典，可以为空。
        root_dir: 根目录的路径字符串，可以为空，默认为当前工作目录的路径。

    Returns:
        GraphRagConfig: 包含配置的 GraphRagConfig 对象。
    """
    # 如果 values 为空，则设为一个空字典
    values = values or {}
    # 如果 root_dir 为空，则设为当前工作目录的路径字符串
    root_dir = root_dir or str(Path.cwd())
    # 创建 Env 对象以读取环境变量
    env = _make_env(root_dir)
    # 替换 tokens（未定义的函数 _token_replace(cast(dict, values))）

    # 使用 InputModelValidator 验证 values 中的 Python 对象，严格模式
    InputModelValidator.validate_python(values, strict=True)

    # 创建 EnvironmentReader 对象以读取环境变量
    reader = EnvironmentReader(env)

    # 定义 hydrate_async_type 函数，用于填充异步类型
    def hydrate_async_type(input: LLMConfigInput, base: AsyncType) -> AsyncType:
        # 获取 input 中的异步模式值
        value = input.get(Fragment.async_mode)
        # 如果有值，则返回 AsyncType 对象，否则返回 base
        return AsyncType(value) if value else base

    # 定义 hydrate_llm_params 函数，用于填充 LLM 参数
    def hydrate_llm_params(
        config: LLMConfigInput, base: LLMParameters
        # 以下部分缺失，未完整注释
    # 定义一个函数，接受两个参数：config 和 base，返回一个LLMParameters对象
    ) -> LLMParameters:
        # 使用reader来访问LLM配置
        with reader.use(config.get("llm")):
            # 从Fragment.type读取LLM类型字符串
            llm_type = reader.str(Fragment.type)
            # 如果llm_type不为空，则将其转换为LLMType对象；否则使用base.type
            llm_type = LLMType(llm_type) if llm_type else base.type
            # 从Fragment.api_key读取API密钥字符串；如果为空，则使用base.api_key
            api_key = reader.str(Fragment.api_key) or base.api_key
            # 从Fragment.api_base读取API基础地址字符串；如果为空，则使用base.api_base
            api_base = reader.str(Fragment.api_base) or base.api_base
            # 从Fragment.cognitive_services_endpoint读取认知服务终端点字符串；如果为空，则使用base.cognitive_services_endpoint
            cognitive_services_endpoint = (
                reader.str(Fragment.cognitive_services_endpoint)
                or base.cognitive_services_endpoint
            )
            # 从Fragment.deployment_name读取部署名称字符串；如果为空，则使用base.deployment_name
            deployment_name = (
                reader.str(Fragment.deployment_name) or base.deployment_name
            )

            # 如果api_key为空并且llm_type不是Azure类型，则抛出ApiKeyMissingError异常
            if api_key is None and not _is_azure(llm_type):
                raise ApiKeyMissingError
            # 如果llm_type是Azure类型，并且api_base为空，则抛出AzureApiBaseMissingError异常
            if _is_azure(llm_type):
                if api_base is None:
                    raise AzureApiBaseMissingError
                # 如果deployment_name为空，则抛出AzureDeploymentNameMissingError异常
                if deployment_name is None:
                    raise AzureDeploymentNameMissingError

            # 从Fragment.sleep_recommendation读取睡眠推荐布尔值；如果为空，则使用base.sleep_on_rate_limit_recommendation
            sleep_on_rate_limit = reader.bool(Fragment.sleep_recommendation)
            # 如果sleep_on_rate_limit为空，则将其设置为base.sleep_on_rate_limit_recommendation

            if sleep_on_rate_limit is None:
                sleep_on_rate_limit = base.sleep_on_rate_limit_recommendation

            # 返回一个LLMParameters对象，包含从reader获取的各种配置参数
            return LLMParameters(
                api_key=api_key,
                type=llm_type,
                api_base=api_base,
                api_version=reader.str(Fragment.api_version) or base.api_version,
                organization=reader.str("organization") or base.organization,
                proxy=reader.str("proxy") or base.proxy,
                model=reader.str("model") or base.model,
                max_tokens=reader.int(Fragment.max_tokens) or base.max_tokens,
                temperature=reader.float(Fragment.temperature) or base.temperature,
                top_p=reader.float(Fragment.top_p) or base.top_p,
                n=reader.int(Fragment.n) or base.n,
                model_supports_json=reader.bool(Fragment.model_supports_json)
                or base.model_supports_json,
                request_timeout=reader.float(Fragment.request_timeout)
                or base.request_timeout,
                cognitive_services_endpoint=cognitive_services_endpoint,
                deployment_name=deployment_name,
                tokens_per_minute=reader.int("tokens_per_minute", Fragment.tpm)
                or base.tokens_per_minute,
                requests_per_minute=reader.int("requests_per_minute", Fragment.rpm)
                or base.requests_per_minute,
                max_retries=reader.int(Fragment.max_retries) or base.max_retries,
                max_retry_wait=reader.float(Fragment.max_retry_wait)
                or base.max_retry_wait,
                sleep_on_rate_limit_recommendation=sleep_on_rate_limit,
                concurrent_requests=reader.int(Fragment.concurrent_requests)
                or base.concurrent_requests,
            )
    # 定义一个函数，用于根据给定的配置和基础参数生成并返回并行化参数对象
    def hydrate_parallelization_params(
        config: LLMConfigInput, base: ParallelizationParameters
    ) -> ParallelizationParameters:
        # 使用配置中的读取器对象来处理并行化相关配置
        with reader.use(config.get("parallelization")):
            # 创建并返回新的并行化参数对象，其中包括线程数和间隔时间
            return ParallelizationParameters(
                num_threads=reader.int("num_threads", Fragment.thread_count)
                or base.num_threads,  # 设置线程数，默认使用基础参数中的值
                stagger=reader.float("stagger", Fragment.thread_stagger)
                or base.stagger,  # 设置线程间隔时间，默认使用基础参数中的值
            )

    # 设置备用的 OpenAI API 密钥，优先选择环境变量中的 OPENAI_API_KEY 或 AZURE_OPENAI_API_KEY
    fallback_oai_key = env("OPENAI_API_KEY", env("AZURE_OPENAI_API_KEY", None))
    # 设置备用的 OpenAI 组织 ID，优先选择环境变量中的 OPENAI_ORG_ID
    fallback_oai_org = env("OPENAI_ORG_ID", None)
    # 设置备用的 OpenAI 基础 URL，优先选择环境变量中的 OPENAI_BASE_URL
    fallback_oai_base = env("OPENAI_BASE_URL", None)
    # 设置备用的 OpenAI API 版本，优先选择环境变量中的 OPENAI_API_VERSION
    fallback_oai_version = env("OPENAI_API_VERSION", None)

    # 返回一个 GraphRagConfig 对象，包含各种配置和模型
    return GraphRagConfig(
        root_dir=root_dir,  # 根目录路径
        llm=llm_model,  # LLM 模型对象
        parallelization=llm_parallelization_model,  # 并行化参数模型对象
        async_mode=async_mode,  # 异步模式配置
        embeddings=embeddings_model,  # 嵌入模型对象
        embed_graph=embed_graph_model,  # 图嵌入模型对象
        reporting=reporting_model,  # 报告模型对象
        storage=storage_model,  # 存储模型对象
        cache=cache_model,  # 缓存模型对象
        input=input_model,  # 输入模型对象
        chunks=chunks_model,  # 块模型对象
        snapshots=snapshots_model,  # 快照模型对象
        entity_extraction=entity_extraction_model,  # 实体提取模型对象
        claim_extraction=claim_extraction_model,  # 主张提取模型对象
        community_reports=community_reports_model,  # 社区报告模型对象
        summarize_descriptions=summarize_descriptions_model,  # 总结描述模型对象
        umap=umap_model,  # UMAP 模型对象
        cluster_graph=cluster_graph_model,  # 聚类图模型对象
        encoding_model=encoding_model,  # 编码模型对象
        skip_workflows=skip_workflows,  # 是否跳过工作流配置
        local_search=local_search_model,  # 本地搜索模型对象
        global_search=global_search_model,  # 全局搜索模型对象
    )
# 定义枚举类 Fragment，表示配置片段，继承自 str 和 Enum
class Fragment(str, Enum):
    """Configuration Fragments."""

    # 定义枚举成员，每个成员表示一个配置项，值为配置项的名称
    api_base = "API_BASE"
    api_key = "API_KEY"
    api_version = "API_VERSION"
    api_organization = "API_ORGANIZATION"
    api_proxy = "API_PROXY"
    async_mode = "ASYNC_MODE"
    base_dir = "BASE_DIR"
    cognitive_services_endpoint = "COGNITIVE_SERVICES_ENDPOINT"
    concurrent_requests = "CONCURRENT_REQUESTS"
    conn_string = "CONNECTION_STRING"
    container_name = "CONTAINER_NAME"
    deployment_name = "DEPLOYMENT_NAME"
    description = "DESCRIPTION"
    enabled = "ENABLED"
    encoding = "ENCODING"
    encoding_model = "ENCODING_MODEL"
    file_type = "FILE_TYPE"
    max_gleanings = "MAX_GLEANINGS"
    max_length = "MAX_LENGTH"
    max_retries = "MAX_RETRIES"
    max_retry_wait = "MAX_RETRY_WAIT"
    max_tokens = "MAX_TOKENS"
    temperature = "TEMPERATURE"
    top_p = "TOP_P"
    n = "N"
    model = "MODEL"
    model_supports_json = "MODEL_SUPPORTS_JSON"
    prompt_file = "PROMPT_FILE"
    request_timeout = "REQUEST_TIMEOUT"
    rpm = "REQUESTS_PER_MINUTE"
    sleep_recommendation = "SLEEP_ON_RATE_LIMIT_RECOMMENDATION"
    storage_account_blob_url = "STORAGE_ACCOUNT_BLOB_URL"
    thread_count = "THREAD_COUNT"
    thread_stagger = "THREAD_STAGGER"
    tpm = "TOKENS_PER_MINUTE"
    type = "TYPE"

# 定义枚举类 Section，表示配置部分，继承自 str 和 Enum
class Section(str, Enum):
    """Configuration Sections."""

    # 定义枚举成员，每个成员表示一个配置部分，值为配置部分的名称
    base = "BASE"
    cache = "CACHE"
    chunk = "CHUNK"
    claim_extraction = "CLAIM_EXTRACTION"
    community_reports = "COMMUNITY_REPORTS"
    embedding = "EMBEDDING"
    entity_extraction = "ENTITY_EXTRACTION"
    graphrag = "GRAPHRAG"
    input = "INPUT"
    llm = "LLM"
    node2vec = "NODE2VEC"
    reporting = "REPORTING"
    snapshot = "SNAPSHOT"
    storage = "STORAGE"
    summarize_descriptions = "SUMMARIZE_DESCRIPTIONS"
    umap = "UMAP"
    local_search = "LOCAL_SEARCH"
    global_search = "GLOBAL_SEARCH"

# 定义函数 _is_azure，判断给定的 LLMType 是否为 Azure OpenAI 类型
def _is_azure(llm_type: LLMType | None) -> bool:
    return (
        llm_type == LLMType.AzureOpenAI
        or llm_type == LLMType.AzureOpenAIChat
        or llm_type == LLMType.AzureOpenAIEmbedding
    )

# 定义函数 _make_env，根据指定的根目录创建环境对象 Env，并返回该对象
def _make_env(root_dir: str) -> Env:
    # 调用 read_dotenv 函数读取根目录中的环境变量文件
    read_dotenv(root_dir)
    # 创建 Env 对象，设置 expand_vars=True 表示扩展环境变量
    env = Env(expand_vars=True)
    # 从环境变量文件中读取环境变量
    env.read_env()
    return env

# 定义函数 _token_replace，替换字典对象中的环境变量令牌
def _token_replace(data: dict):
    """Replace env-var tokens in a dictionary object."""
    # 遍历字典中的键值对
    for key, value in data.items():
        # 如果值是字典类型，则递归调用 _token_replace 函数
        if isinstance(value, dict):
            _token_replace(value)
        # 如果值是字符串类型，则使用 os.path.expandvars 替换其中的环境变量令牌
        elif isinstance(value, str):
            data[key] = os.path.expandvars(value)
```