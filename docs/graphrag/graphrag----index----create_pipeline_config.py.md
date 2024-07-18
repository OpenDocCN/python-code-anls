# `.\graphrag\graphrag\index\create_pipeline_config.py`

```py
# 导入必要的模块和类
import json
import logging
from pathlib import Path

from graphrag.config.enums import (
    CacheType,
    InputFileType,
    ReportingType,
    StorageType,
    TextEmbeddingTarget,
)
from graphrag.config.models import (
    GraphRagConfig,
    TextEmbeddingConfig,
)
from graphrag.index.config.cache import (
    PipelineBlobCacheConfig,
    PipelineCacheConfigTypes,
    PipelineFileCacheConfig,
    PipelineMemoryCacheConfig,
    PipelineNoneCacheConfig,
)
from graphrag.index.config.input import (
    PipelineCSVInputConfig,
    PipelineInputConfigTypes,
    PipelineTextInputConfig,
)
from graphrag.index.config.pipeline import (
    PipelineConfig,
)
from graphrag.index.config.reporting import (
    PipelineBlobReportingConfig,
    PipelineConsoleReportingConfig,
    PipelineFileReportingConfig,
    PipelineReportingConfigTypes,
)
from graphrag.index.config.storage import (
    PipelineBlobStorageConfig,
    PipelineFileStorageConfig,
    PipelineMemoryStorageConfig,
    PipelineStorageConfigTypes,
)
from graphrag.index.config.workflow import (
    PipelineWorkflowReference,
)
from graphrag.index.workflows.default_workflows import (
    create_base_documents,
    create_base_entity_graph,
    create_base_extracted_entities,
    create_base_text_units,
    create_final_communities,
    create_final_community_reports,
    create_final_covariates,
    create_final_documents,
    create_final_entities,
    create_final_nodes,
    create_final_relationships,
    create_final_text_units,
    create_summarized_entities,
    join_text_units_to_covariate_ids,
    join_text_units_to_entity_ids,
    join_text_units_to_relationship_ids,
)

# 设置日志记录器
log = logging.getLogger(__name__)

# 定义各种文本嵌入的名称常量
entity_name_embedding = "entity.name"
entity_description_embedding = "entity.description"
relationship_description_embedding = "relationship.description"
document_raw_content_embedding = "document.raw_content"
community_title_embedding = "community.title"
community_summary_embedding = "community.summary"
community_full_content_embedding = "community.full_content"
text_unit_text_embedding = "text_unit.text"

# 创建包含所有文本嵌入名称的集合
all_embeddings: set[str] = {
    entity_name_embedding,
    entity_description_embedding,
    relationship_description_embedding,
    document_raw_content_embedding,
    community_title_embedding,
    community_summary_embedding,
    community_full_content_embedding,
    text_unit_text_embedding,
}

# 创建包含必需文本嵌入名称的集合
required_embeddings: set[str] = {entity_description_embedding}

# 定义内置文档属性的集合
builtin_document_attributes: set[str] = {
    "id",
    "source",
    "text",
    "title",
    "timestamp",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
}

def create_pipeline_config(settings: GraphRagConfig, verbose=False) -> PipelineConfig:
    """获取用于流程的默认配置"""
    # 从设置中获取流程配置对象
    # 如果 verbose 参数为 True，则生成详细的日志记录
    # 返回配置对象
    # 如果 verbose 参数为真，则记录当前的 LLM 设置信息
    if verbose:
        _log_llm_settings(settings)

    # 确定应跳过的工作流程列表
    skip_workflows = _determine_skip_workflows(settings)
    
    # 获取嵌入字段的配置信息
    embedded_fields = _get_embedded_fields(settings)
    
    # 判断是否启用协变量
    covariates_enabled = (
        settings.claim_extraction.enabled
        and create_final_covariates not in skip_workflows
    )

    # 构造 PipelineConfig 对象，包括根目录、输入配置、报告配置、存储配置、缓存配置和多个工作流配置
    result = PipelineConfig(
        root_dir=settings.root_dir,
        input=_get_pipeline_input_config(settings),
        reporting=_get_reporting_config(settings),
        storage=_get_storage_config(settings),
        cache=_get_cache_config(settings),
        workflows=[
            *_document_workflows(settings, embedded_fields),
            *_text_unit_workflows(settings, covariates_enabled, embedded_fields),
            *_graph_workflows(settings, embedded_fields),
            *_community_workflows(settings, covariates_enabled, embedded_fields),
            *(_covariate_workflows(settings) if covariates_enabled else []),
        ],
    )

    # 记录日志，说明跳过的工作流程列表
    log.info("skipping workflows %s", ",".join(skip_workflows))
    
    # 从结果的工作流程列表中移除应该跳过的工作流程
    result.workflows = [w for w in result.workflows if w.name not in skip_workflows]
    
    # 返回最终的 PipelineConfig 对象作为结果
    return result
# 获取嵌入字段的集合，根据给定的 GraphRagConfig 设置
def _get_embedded_fields(settings: GraphRagConfig) -> set[str]:
    # 根据嵌入目标类型进行匹配
    match settings.embeddings.target:
        # 如果目标为 TextEmbeddingTarget.all，则返回除了设置中跳过的嵌入字段外的所有嵌入字段
        case TextEmbeddingTarget.all:
            return all_embeddings - {*settings.embeddings.skip}
        # 如果目标为 TextEmbeddingTarget.required，则返回所需的嵌入字段
        case TextEmbeddingTarget.required:
            return required_embeddings
        # 其他情况，抛出异常，显示未知的嵌入目标类型
        case _:
            msg = f"Unknown embeddings target: {settings.embeddings.target}"
            raise ValueError(msg)


# 确定需要跳过的工作流列表，根据给定的 GraphRagConfig 设置
def _determine_skip_workflows(settings: GraphRagConfig) -> list[str]:
    # 获取设置中的跳过工作流列表
    skip_workflows = settings.skip_workflows
    # 如果设置中包含 create_final_covariates 但不包含 join_text_units_to_covariate_ids
    if (
        create_final_covariates in skip_workflows
        and join_text_units_to_covariate_ids not in skip_workflows
    ):
        # 添加 join_text_units_to_covariate_ids 到跳过工作流列表中
        skip_workflows.append(join_text_units_to_covariate_ids)
    return skip_workflows


# 记录 LLM 设置信息，输出到日志
def _log_llm_settings(settings: GraphRagConfig) -> None:
    # 记录 LLM 配置信息到日志，隐藏敏感信息 "api_key"
    log.info(
        "Using LLM Config %s",
        json.dumps(
            {**settings.entity_extraction.llm.model_dump(), "api_key": "*****"},
            indent=4,
        ),
    )
    # 记录嵌入配置信息到日志，隐藏敏感信息 "api_key"
    log.info(
        "Using Embeddings Config %s",
        json.dumps(
            {**settings.embeddings.llm.model_dump(), "api_key": "*****"}, indent=4
        ),
    )


# 构建文档工作流列表，根据给定的 GraphRagConfig 设置和嵌入字段集合
def _document_workflows(
    settings: GraphRagConfig, embedded_fields: set[str]
) -> list[PipelineWorkflowReference]:
    # 检查是否需要跳过文档原始内容嵌入
    skip_document_raw_content_embedding = (
        document_raw_content_embedding not in embedded_fields
    )
    # 返回文档工作流的列表
    return [
        # 创建基础文档的工作流引用
        PipelineWorkflowReference(
            name=create_base_documents,
            config={
                # 设置文档属性列，排除内置文档属性列
                "document_attribute_columns": list(
                    {*(settings.input.document_attribute_columns)}
                    - builtin_document_attributes
                )
            },
        ),
        # 创建最终文档的工作流引用
        PipelineWorkflowReference(
            name=create_final_documents,
            config={
                # 设置文档原始内容嵌入的配置
                "document_raw_content_embed": _get_embedding_settings(
                    settings.embeddings, "document_raw_content"
                ),
                # 是否跳过文档原始内容嵌入
                "skip_raw_content_embedding": skip_document_raw_content_embedding,
            },
        ),
    ]


# 构建文本单元工作流列表，根据给定的 GraphRagConfig 设置、协变量启用状态和嵌入字段集合
def _text_unit_workflows(
    settings: GraphRagConfig,
    covariates_enabled: bool,
    embedded_fields: set[str],
) -> list[PipelineWorkflowReference]:
    # 检查是否需要跳过文本单元嵌入
    skip_text_unit_embedding = text_unit_text_embedding not in embedded_fields
    # 返回一个包含多个 PipelineWorkflowReference 实例的列表，用于描述数据处理流程
    return [
        # 创建基本文本单元的处理流程参考
        PipelineWorkflowReference(
            name=create_base_text_units,
            config={
                "chunk_by": settings.chunks.group_by_columns,  # 设置按列分块的策略
                "text_chunk": {"strategy": settings.chunks.resolved_strategy()},  # 设置文本块的处理策略
            },
        ),
        # 连接文本单元到实体 ID 的处理流程参考
        PipelineWorkflowReference(
            name=join_text_units_to_entity_ids,
        ),
        # 连接文本单元到关系 ID 的处理流程参考
        PipelineWorkflowReference(
            name=join_text_units_to_relationship_ids,
        ),
        # 如果启用了协变量，将连接文本单元到协变量 ID 的处理流程参考加入列表
        *(
            [
                PipelineWorkflowReference(
                    name=join_text_units_to_covariate_ids,
                )
            ]
            if covariates_enabled
            else []  # 如果未启用协变量，返回空列表
        ),
        # 创建最终文本单元的处理流程参考
        PipelineWorkflowReference(
            name=create_final_text_units,
            config={
                "text_unit_text_embed": _get_embedding_settings(
                    settings.embeddings, "text_unit_text"
                ),  # 获取文本单元文本嵌入的设置
                "covariates_enabled": covariates_enabled,  # 传递协变量启用状态
                "skip_text_unit_embedding": skip_text_unit_embedding,  # 是否跳过文本单元嵌入
            },
        ),
    ]
def _get_embedding_settings(settings: TextEmbeddingConfig, embedding_name: str) -> dict:
    # 获取文本嵌入设置中的向量存储设置
    vector_store_settings = settings.vector_store
    # 如果向量存储设置为空，返回仅包含解析后策略的字典
    if vector_store_settings is None:
        return {"strategy": settings.resolved_strategy()}

    #
    # 如果程序执行到这里，说明settings.vector_store已定义，并且这种嵌入有特定的设置。
    # settings.vector_store.base包含连接信息，或可能未定义
    # settings.vector_store.<vector_name>包含此嵌入的特定设置
    #
    
    strategy = settings.resolved_strategy()  # 获取默认策略
    strategy.update({
        "vector_store": vector_store_settings
    })  # 使用向量存储设置更新默认策略
    # 这确保向量存储配置是策略的一部分，而不是全局配置的一部分
    return {
        "strategy": strategy,
        "embedding_name": embedding_name,
    }


def _graph_workflows(
    settings: GraphRagConfig, embedded_fields: set[str]
) -> list[PipelineWorkflowReference]:
    skip_entity_name_embedding = entity_name_embedding not in embedded_fields
    skip_entity_description_embedding = (
        entity_description_embedding not in embedded_fields
    )
    skip_relationship_description_embedding = (
        relationship_description_embedding not in embedded_fields
    )
    # 返回一个包含多个 PipelineWorkflowReference 实例的列表，每个实例代表一个工作流引用
    return [
        # 创建基本实体提取的工作流引用
        PipelineWorkflowReference(
            name=create_base_extracted_entities,
            config={
                "graphml_snapshot": settings.snapshots.graphml,  # 使用设置中的图形快照
                "raw_entity_snapshot": settings.snapshots.raw_entities,  # 使用设置中的原始实体快照
                "entity_extract": {
                    **settings.entity_extraction.parallelization.model_dump(),  # 并行化模型转储设置
                    "async_mode": settings.entity_extraction.async_mode,  # 使用异步模式设置
                    "strategy": settings.entity_extraction.resolved_strategy(  # 解析实体提取策略
                        settings.root_dir, settings.encoding_model
                    ),
                    "entity_types": settings.entity_extraction.entity_types,  # 使用设置中的实体类型
                },
            },
        ),
        # 创建摘要实体的工作流引用
        PipelineWorkflowReference(
            name=create_summarized_entities,
            config={
                "graphml_snapshot": settings.snapshots.graphml,  # 使用设置中的图形快照
                "summarize_descriptions": {
                    **settings.summarize_descriptions.parallelization.model_dump(),  # 并行化模型转储设置
                    "async_mode": settings.summarize_descriptions.async_mode,  # 使用异步模式设置
                    "strategy": settings.summarize_descriptions.resolved_strategy(  # 解析摘要描述策略
                        settings.root_dir,
                    ),
                },
            },
        ),
        # 创建基本实体图的工作流引用
        PipelineWorkflowReference(
            name=create_base_entity_graph,
            config={
                "graphml_snapshot": settings.snapshots.graphml,  # 使用设置中的图形快照
                "embed_graph_enabled": settings.embed_graph.enabled,  # 使用嵌入图启用设置
                "cluster_graph": {
                    "strategy": settings.cluster_graph.resolved_strategy()  # 解析集群图策略
                },
                "embed_graph": {"strategy": settings.embed_graph.resolved_strategy()},  # 解析嵌入图策略
            },
        ),
        # 创建最终实体的工作流引用
        PipelineWorkflowReference(
            name=create_final_entities,
            config={
                "entity_name_embed": _get_embedding_settings(  # 获取实体名称嵌入设置
                    settings.embeddings, "entity_name"
                ),
                "entity_name_description_embed": _get_embedding_settings(  # 获取实体名称描述嵌入设置
                    settings.embeddings, "entity_name_description"
                ),
                "skip_name_embedding": skip_entity_name_embedding,  # 跳过实体名称嵌入设置
                "skip_description_embedding": skip_entity_description_embedding,  # 跳过实体描述嵌入设置
            },
        ),
        # 创建最终关系的工作流引用
        PipelineWorkflowReference(
            name=create_final_relationships,
            config={
                "relationship_description_embed": _get_embedding_settings(  # 获取关系描述嵌入设置
                    settings.embeddings, "relationship_description"
                ),
                "skip_description_embedding": skip_relationship_description_embedding,  # 跳过关系描述嵌入设置
            },
        ),
        # 创建最终节点的工作流引用
        PipelineWorkflowReference(
            name=create_final_nodes,
            config={
                "layout_graph_enabled": settings.umap.enabled,  # 使用 UMAP 布局图启用设置
                "snapshot_top_level_nodes": settings.snapshots.top_level_nodes,  # 使用设置中的顶级节点快照
            },
        ),
    ]
# 定义一个函数 _community_workflows，用于生成社区工作流列表
# 参数包括 settings: GraphRagConfig 对象，covariates_enabled: bool 类型的变量，embedded_fields: set[str] 类型的集合
# 返回一个列表，列表中的元素是 PipelineWorkflowReference 对象的实例
def _community_workflows(
    settings: GraphRagConfig, covariates_enabled: bool, embedded_fields: set[str]
) -> list[PipelineWorkflowReference]:
    # 检查 community_title_embedding 是否在 embedded_fields 中，得到布尔值 skip_community_title_embedding
    skip_community_title_embedding = community_title_embedding not in embedded_fields
    # 检查 community_summary_embedding 是否在 embedded_fields 中，得到布尔值 skip_community_summary_embedding
    skip_community_summary_embedding = (
        community_summary_embedding not in embedded_fields
    )
    # 检查 community_full_content_embedding 是否在 embedded_fields 中，得到布尔值 skip_community_full_content_embedding
    skip_community_full_content_embedding = (
        community_full_content_embedding not in embedded_fields
    )
    # 返回一个列表，包含两个 PipelineWorkflowReference 对象的实例
    return [
        PipelineWorkflowReference(name=create_final_communities),
        PipelineWorkflowReference(
            name=create_final_community_reports,
            config={
                # 设置 "covariates_enabled" 键的值为 covariates_enabled 变量的值
                "covariates_enabled": covariates_enabled,
                # 设置 "skip_title_embedding" 键的值为 skip_community_title_embedding 变量的值
                "skip_title_embedding": skip_community_title_embedding,
                # 设置 "skip_summary_embedding" 键的值为 skip_community_summary_embedding 变量的值
                "skip_summary_embedding": skip_community_summary_embedding,
                # 设置 "skip_full_content_embedding" 键的值为 skip_community_full_content_embedding 变量的值
                "skip_full_content_embedding": skip_community_full_content_embedding,
                # 设置 "create_community_reports" 键的值为一个字典，包含多个配置项
                "create_community_reports": {
                    **settings.community_reports.parallelization.model_dump(),
                    # 设置 "async_mode" 键的值为 settings.community_reports.async_mode 的值
                    "async_mode": settings.community_reports.async_mode,
                    # 设置 "strategy" 键的值为 settings.community_reports.resolved_strategy(settings.root_dir) 的返回值
                    "strategy": settings.community_reports.resolved_strategy(
                        settings.root_dir
                    ),
                },
                # 设置 "community_report_full_content_embed" 键的值为 _get_embedding_settings(settings.embeddings, "community_report_full_content") 的返回值
                "community_report_full_content_embed": _get_embedding_settings(
                    settings.embeddings, "community_report_full_content"
                ),
                # 设置 "community_report_summary_embed" 键的值为 _get_embedding_settings(settings.embeddings, "community_report_summary") 的返回值
                "community_report_summary_embed": _get_embedding_settings(
                    settings.embeddings, "community_report_summary"
                ),
                # 设置 "community_report_title_embed" 键的值为 _get_embedding_settings(settings.embeddings, "community_report_title") 的返回值
                "community_report_title_embed": _get_embedding_settings(
                    settings.embeddings, "community_report_title"
                ),
            },
        ),
    ]


# 定义一个函数 _covariate_workflows，用于生成协变量工作流列表
# 参数包括 settings: GraphRagConfig 对象
# 返回一个列表，列表中的元素是 PipelineWorkflowReference 对象的实例
def _covariate_workflows(
    settings: GraphRagConfig,
) -> list[PipelineWorkflowReference]:
    # 返回一个列表，包含一个 PipelineWorkflowReference 对象的实例
    return [
        PipelineWorkflowReference(
            name=create_final_covariates,
            config={
                # 设置 "claim_extract" 键的值为一个字典，包含多个配置项
                "claim_extract": {
                    **settings.claim_extraction.parallelization.model_dump(),
                    # 设置 "strategy" 键的值为 settings.claim_extraction.resolved_strategy(settings.root_dir) 的返回值
                    "strategy": settings.claim_extraction.resolved_strategy(
                        settings.root_dir
                    ),
                },
            },
        )
    ]


# 定义一个函数 _get_pipeline_input_config，用于获取管道输入配置
# 参数包括 settings: GraphRagConfig 对象
# 返回一个 PipelineInputConfigTypes 类型的值，表示管道的输入配置类型
def _get_pipeline_input_config(
    settings: GraphRagConfig,
) -> PipelineInputConfigTypes:
    # 获取 settings.input 的 file_type 属性的值，赋给变量 file_type
    file_type = settings.input.file_type
    # 根据文件类型选择合适的输入配置对象并返回

    # 当文件类型为 CSV 时，返回 CSV 输入配置对象
    case InputFileType.csv:
        return PipelineCSVInputConfig(
            base_dir=settings.input.base_dir,  # 设置基础目录
            file_pattern=settings.input.file_pattern,  # 文件匹配模式
            encoding=settings.input.encoding,  # 文件编码方式
            source_column=settings.input.source_column,  # 源列名称
            timestamp_column=settings.input.timestamp_column,  # 时间戳列名称
            timestamp_format=settings.input.timestamp_format,  # 时间戳格式
            text_column=settings.input.text_column,  # 文本列名称
            title_column=settings.input.title_column,  # 标题列名称
            type=settings.input.type,  # 文件类型
            connection_string=settings.input.connection_string,  # 连接字符串
            storage_account_blob_url=settings.input.storage_account_blob_url,  # 存储账户 Blob URL
            container_name=settings.input.container_name,  # 容器名称
        )

    # 当文件类型为文本时，返回文本输入配置对象
    case InputFileType.text:
        return PipelineTextInputConfig(
            base_dir=settings.input.base_dir,  # 设置基础目录
            file_pattern=settings.input.file_pattern,  # 文件匹配模式
            encoding=settings.input.encoding,  # 文件编码方式
            type=settings.input.type,  # 文件类型
            connection_string=settings.input.connection_string,  # 连接字符串
            storage_account_blob_url=settings.input.storage_account_blob_url,  # 存储账户 Blob URL
            container_name=settings.input.container_name,  # 容器名称
        )

    # 如果文件类型未知，则抛出值错误并附带错误消息
    case _:
        msg = f"Unknown input type: {file_type}"  # 错误消息
        raise ValueError(msg)  # 抛出值错误异常
# 从配置中获取报告配置，返回相应的报告配置类型
def _get_reporting_config(
    settings: GraphRagConfig,
) -> PipelineReportingConfigTypes:
    """Get the reporting config from the settings."""
    # 根据报告类型进行匹配
    match settings.reporting.type:
        # 如果报告类型是文件
        case ReportingType.file:
            # 相对于根目录的路径
            return PipelineFileReportingConfig(base_dir=settings.reporting.base_dir)
        # 如果报告类型是 Blob 存储
        case ReportingType.blob:
            # 获取 Blob 报告需要的连接字符串和存储帐户 Blob URL
            connection_string = settings.reporting.connection_string
            storage_account_blob_url = settings.reporting.storage_account_blob_url
            container_name = settings.reporting.container_name
            # 如果容器名为空，抛出异常
            if container_name is None:
                msg = "Container name must be provided for blob reporting."
                raise ValueError(msg)
            # 如果连接字符串和存储帐户 Blob URL 都为空，抛出异常
            if connection_string is None and storage_account_blob_url is None:
                msg = "Connection string or storage account blob url must be provided for blob reporting."
                raise ValueError(msg)
            # 返回 Blob 报告配置对象
            return PipelineBlobReportingConfig(
                connection_string=connection_string,
                container_name=container_name,
                base_dir=settings.reporting.base_dir,
                storage_account_blob_url=storage_account_blob_url,
            )
        # 如果报告类型是控制台
        case ReportingType.console:
            # 返回控制台报告配置对象
            return PipelineConsoleReportingConfig()
        # 对于其他未匹配到的报告类型
        case _:
            # 相对于根目录的路径
            return PipelineFileReportingConfig(base_dir=settings.reporting.base_dir)


# 获取存储配置类型从设置中
def _get_storage_config(
    settings: GraphRagConfig,
) -> PipelineStorageConfigTypes:
    """Get the storage type from the settings."""
    # 从设置中获取根目录路径
    root_dir = settings.root_dir
    # 根据设置中的存储类型进行不同的处理
    match settings.storage.type:
        case StorageType.memory:
            # 如果存储类型是内存，则返回内存存储配置对象
            return PipelineMemoryStorageConfig()
        case StorageType.file:
            # 如果存储类型是文件
            # 相对于根目录的基础目录
            base_dir = settings.storage.base_dir
            if base_dir is None:
                msg = "Base directory must be provided for file storage."
                raise ValueError(msg)
            # 构建文件存储配置对象，基础目录是根目录下的指定路径
            return PipelineFileStorageConfig(base_dir=str(Path(root_dir) / base_dir))
        case StorageType.blob:
            # 如果存储类型是 Blob
            connection_string = settings.storage.connection_string
            storage_account_blob_url = settings.storage.storage_account_blob_url
            container_name = settings.storage.container_name
            if container_name is None:
                msg = "Container name must be provided for blob storage."
                raise ValueError(msg)
            if connection_string is None and storage_account_blob_url is None:
                msg = "Connection string or storage account blob url must be provided for blob storage."
                raise ValueError(msg)
            # 构建 Blob 存储配置对象，包括连接字符串、容器名、基础目录等信息
            return PipelineBlobStorageConfig(
                connection_string=connection_string,
                container_name=container_name,
                base_dir=settings.storage.base_dir,
                storage_account_blob_url=storage_account_blob_url,
            )
        case _:
            # 如果存储类型未知，则按文件存储处理
            # 相对于根目录的基础目录
            base_dir = settings.storage.base_dir
            if base_dir is None:
                msg = "Base directory must be provided for file storage."
                raise ValueError(msg)
            # 构建文件存储配置对象，基础目录是根目录下的指定路径
            return PipelineFileStorageConfig(base_dir=str(Path(root_dir) / base_dir))
def _get_cache_config(
    settings: GraphRagConfig,
) -> PipelineCacheConfigTypes:
    """Get the cache type from the settings."""
    # 根据设置获取缓存类型
    match settings.cache.type:
        case CacheType.memory:
            # 如果缓存类型是内存，则返回内存缓存配置对象
            return PipelineMemoryCacheConfig()
        case CacheType.file:
            # 如果缓存类型是文件，则返回文件缓存配置对象，并基于根目录
            return PipelineFileCacheConfig(base_dir=settings.cache.base_dir)
        case CacheType.none:
            # 如果缓存类型是无缓存，则返回无缓存配置对象
            return PipelineNoneCacheConfig()
        case CacheType.blob:
            # 如果缓存类型是 blob 存储，则获取相关配置参数
            connection_string = settings.cache.connection_string
            storage_account_blob_url = settings.cache.storage_account_blob_url
            container_name = settings.cache.container_name
            if container_name is None:
                # 如果未提供容器名，则抛出异常
                msg = "Container name must be provided for blob cache."
                raise ValueError(msg)
            if connection_string is None and storage_account_blob_url is None:
                # 如果未提供连接字符串或存储账户 blob URL，则抛出异常
                msg = "Connection string or storage account blob url must be provided for blob cache."
                raise ValueError(msg)
            # 返回 blob 缓存配置对象，并设置相关参数
            return PipelineBlobCacheConfig(
                connection_string=connection_string,
                container_name=container_name,
                base_dir=settings.cache.base_dir,
                storage_account_blob_url=storage_account_blob_url,
            )
        case _:  # 处理其它未知的缓存类型，默认使用文件缓存，并基于当前目录
            return PipelineFileCacheConfig(base_dir="./cache")
```