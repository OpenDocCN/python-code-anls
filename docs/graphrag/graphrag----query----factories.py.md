# `.\graphrag\graphrag\query\factories.py`

```py
# 导入所需模块和库
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# 导入配置和模型定义
from graphrag.config import (
    GraphRagConfig,
    LLMType,
)
from graphrag.model import (
    CommunityReport,
    Covariate,
    Entity,
    Relationship,
    TextUnit,
)
# 导入实体向量存储的关键部分
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
# 导入 OpenAI 的聊天功能
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
# 导入 OpenAI 的嵌入功能
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
# 导入 OpenAI 的 API 类型定义
from graphrag.query.llm.oai.typing import OpenaiApiType
# 导入全局社区上下文搜索
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
# 导入全局搜索功能
from graphrag.query.structured_search.global_search.search import GlobalSearch
# 导入混合上下文的本地搜索功能
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
# 导入本地搜索功能
from graphrag.query.structured_search.local_search.search import LocalSearch
# 导入基础向量存储功能
from graphrag.vector_stores import BaseVectorStore


# 定义函数：获取与配置相关的 LLM 客户端
def get_llm(config: GraphRagConfig) -> ChatOpenAI:
    """Get the LLM client."""
    # 判断是否为 Azure 客户端
    is_azure_client = (
        config.llm.type == LLMType.AzureOpenAIChat
        or config.llm.type == LLMType.AzureOpenAI
    )
    # 获取调试信息中的 LLM 密钥
    debug_llm_key = config.llm.api_key or ""
    # 构建调试信息字典，包含模型信息和已处理的 API 密钥
    llm_debug_info = {
        **config.llm.model_dump(),
        "api_key": f"REDACTED,len={len(debug_llm_key)}",
    }
    # 如果未指定认知服务的终结点，则使用默认值
    if config.llm.cognitive_services_endpoint is None:
        cognitive_services_endpoint = "https://cognitiveservices.azure.com/.default"
    else:
        cognitive_services_endpoint = config.llm.cognitive_services_endpoint
    # 输出调试信息到控制台，T201 表示不检查行尾空格
    print(f"creating llm client with {llm_debug_info}")  # noqa T201
    # 返回 ChatOpenAI 类的实例化对象，使用配置参数
    return ChatOpenAI(
        api_key=config.llm.api_key,
        azure_ad_token_provider=(
            get_bearer_token_provider(
                DefaultAzureCredential(), cognitive_services_endpoint
            )
            if is_azure_client and not config.llm.api_key
            else None
        ),
        api_base=config.llm.api_base,
        model=config.llm.model,
        api_type=OpenaiApiType.AzureOpenAI if is_azure_client else OpenaiApiType.OpenAI,
        deployment_name=config.llm.deployment_name,
        api_version=config.llm.api_version,
        max_retries=config.llm.max_retries,
    )


# 定义函数：获取与配置相关的文本嵌入器 LLM 客户端
def get_text_embedder(config: GraphRagConfig) -> OpenAIEmbedding:
    """Get the LLM client for embeddings."""
    # 判断是否为 Azure 客户端
    is_azure_client = config.embeddings.llm.type == LLMType.AzureOpenAIEmbedding
    # 获取调试信息中的嵌入 API 密钥
    debug_embedding_api_key = config.embeddings.llm.api_key or ""
    # 构建调试信息字典，包含模型信息和已处理的 API 密钥
    llm_debug_info = {
        **config.embeddings.llm.model_dump(),
        "api_key": f"REDACTED,len={len(debug_embedding_api_key)}",
    }
    # 如果未指定认知服务的终结点，则使用默认值
    if config.embeddings.llm.cognitive_services_endpoint is None:
        cognitive_services_endpoint = "https://cognitiveservices.azure.com/.default"
    else:
        # 如果不是 Azure 客户端，则从配置中获取嵌入式模型的 Cognitive Services 终结点
        cognitive_services_endpoint = config.embeddings.llm.cognitive_services_endpoint
    # 打印调试信息，显示创建嵌入式 LLM 客户端
    print(f"creating embedding llm client with {llm_debug_info}")  # noqa T201
    # 返回一个 OpenAIEmbedding 实例，包括以下参数：
    return OpenAIEmbedding(
        # 使用配置中的 API 密钥
        api_key=config.embeddings.llm.api_key,
        # 如果是 Azure 客户端且未设置 API 密钥，则使用默认的 Azure 凭据获取令牌提供程序
        azure_ad_token_provider=(
            get_bearer_token_provider(
                DefaultAzureCredential(), cognitive_services_endpoint
            )
            if is_azure_client and not config.embeddings.llm.api_key
            else None
        ),
        # API 的基本地址，从配置中获取
        api_base=config.embeddings.llm.api_base,
        # 根据客户端类型确定 API 类型，如果是 Azure 客户端则使用 AzureOpenAI，否则使用 OpenAI
        api_type=OpenaiApiType.AzureOpenAI if is_azure_client else OpenaiApiType.OpenAI,
        # 模型名称，从配置中获取
        model=config.embeddings.llm.model,
        # 部署名称，从配置中获取
        deployment_name=config.embeddings.llm.deployment_name,
        # API 版本，从配置中获取
        api_version=config.embeddings.llm.api_version,
        # 最大重试次数，从配置中获取
        max_retries=config.embeddings.llm.max_retries,
    )
# 根据给定配置、社区报告、文本单元、实体、关系、协变量、响应类型和描述嵌入存储创建本地搜索引擎对象
def get_local_search_engine(
    config: GraphRagConfig,
    reports: list[CommunityReport],
    text_units: list[TextUnit],
    entities: list[Entity],
    relationships: list[Relationship],
    covariates: dict[str, list[Covariate]],
    response_type: str,
    description_embedding_store: BaseVectorStore,
) -> LocalSearch:
    """Create a local search engine based on data + configuration."""
    # 获取语言模型（LLM）对象，根据给定配置
    llm = get_llm(config)
    # 获取文本嵌入器对象，根据给定配置
    text_embedder = get_text_embedder(config)
    # 获取令牌编码器对象，根据给定编码模型
    token_encoder = tiktoken.get_encoding(config.encoding_model)

    # 获取本地搜索的配置信息
    ls_config = config.local_search

    # 返回一个本地搜索引擎对象，包括语言模型、上下文构建器、令牌编码器和其他参数
    return LocalSearch(
        llm=llm,
        context_builder=LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates=covariates,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,  # 如果向量存储使用实体标题作为 ID，请将此设置为 EntityVectorStoreKey.TITLE
            text_embedder=text_embedder,
            token_encoder=token_encoder,
        ),
        token_encoder=token_encoder,
        llm_params={
            "max_tokens": ls_config.llm_max_tokens,  # 根据模型的令牌限制进行设置（如果使用具有 8k 限制的模型，则设置为 1000=1500 可能是一个不错的选择）
            "temperature": ls_config.temperature,
            "top_p": ls_config.top_p,
            "n": ls_config.n,
        },
        context_builder_params={
            "text_unit_prop": ls_config.text_unit_prop,
            "community_prop": ls_config.community_prop,
            "conversation_history_max_turns": ls_config.conversation_history_max_turns,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": ls_config.top_k_entities,
            "top_k_relationships": ls_config.top_k_relationships,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # 如果向量存储使用实体标题作为 ID，请将此设置为 EntityVectorStoreKey.TITLE
            "max_tokens": ls_config.max_tokens,  # 根据模型的令牌限制进行设置（如果使用具有 8k 限制的模型，则设置为 5000 可能是一个不错的选择）
        },
        response_type=response_type,
    )


# 根据给定配置、社区报告、实体和响应类型创建全局搜索引擎对象
def get_global_search_engine(
    config: GraphRagConfig,
    reports: list[CommunityReport],
    entities: list[Entity],
    response_type: str,
):
    """Create a global search engine based on data + configuration."""
    # 获取令牌编码器对象，根据给定编码模型
    token_encoder = tiktoken.get_encoding(config.encoding_model)
    # 获取全局搜索的配置信息
    gs_config = config.global_search
    # 返回一个 GlobalSearch 对象，用于全局搜索
    return GlobalSearch(
        # 获取语言模型的配置并传递给 GlobalSearch 的 llm 参数
        llm=get_llm(config),
        # 构建全局社区上下文，包括社区报告、实体和令牌编码器
        context_builder=GlobalCommunityContext(
            community_reports=reports, entities=entities, token_encoder=token_encoder
        ),
        # 将令牌编码器传递给 GlobalSearch 的 token_encoder 参数
        token_encoder=token_encoder,
        # 设置最大数据令牌数，从 gs_config 中获取
        max_data_tokens=gs_config.data_max_tokens,
        # 设置映射语言模型参数，包括最大令牌数、温度、top-p 和 n
        map_llm_params={
            "max_tokens": gs_config.map_max_tokens,
            "temperature": gs_config.temperature,
            "top_p": gs_config.top_p,
            "n": gs_config.n,
        },
        # 设置减少语言模型参数，同样包括最大令牌数、温度、top-p 和 n
        reduce_llm_params={
            "max_tokens": gs_config.reduce_max_tokens,
            "temperature": gs_config.temperature,
            "top_p": gs_config.top_p,
            "n": gs_config.n,
        },
        # 禁用通用知识的使用
        allow_general_knowledge=False,
        # 设置 JSON 模式为 False
        json_mode=False,
        # 设置上下文构建器的详细参数，包括社区摘要使用、数据洗牌、包含社区排名等
        context_builder_params={
            "use_community_summary": False,
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": gs_config.max_tokens,
            "context_name": "Reports",
        },
        # 设置并发协程数，从 gs_config 中获取
        concurrent_coroutines=gs_config.concurrency,
        # 设置响应类型，由外部传入的参数决定
        response_type=response_type,
    )
```