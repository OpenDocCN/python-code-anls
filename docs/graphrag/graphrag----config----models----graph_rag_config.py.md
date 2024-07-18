# `.\graphrag\graphrag\config\models\graph_rag_config.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

# 从 devtools 中导入 pformat 工具，用于格式化输出
from devtools import pformat
# 从 pydantic 中导入 Field，用于定义字段的元数据
from pydantic import Field

# 从 graphrag.config.defaults 中导入 defs，用于默认配置
import graphrag.config.defaults as defs

# 以下是各个子模块的配置类的导入

# 导入缓存配置类
from .cache_config import CacheConfig
# 导入分块配置类
from .chunking_config import ChunkingConfig
# 导入声明提取配置类
from .claim_extraction_config import ClaimExtractionConfig
# 导入集群图配置类
from .cluster_graph_config import ClusterGraphConfig
# 导入社区报告配置类
from .community_reports_config import CommunityReportsConfig
# 导入图嵌入配置类
from .embed_graph_config import EmbedGraphConfig
# 导入实体提取配置类
from .entity_extraction_config import EntityExtractionConfig
# 导入全局搜索配置类
from .global_search_config import GlobalSearchConfig
# 导入输入配置类
from .input_config import InputConfig
# 导入LLM配置类
from .llm_config import LLMConfig
# 导入本地搜索配置类
from .local_search_config import LocalSearchConfig
# 导入报告配置类
from .reporting_config import ReportingConfig
# 导入快照配置类
from .snapshots_config import SnapshotsConfig
# 导入存储配置类
from .storage_config import StorageConfig
# 导入描述摘要配置类
from .summarize_descriptions_config import SummarizeDescriptionsConfig
# 导入文本嵌入配置类
from .text_embedding_config import TextEmbeddingConfig
# 导入UMAP配置类
from .umap_config import UmapConfig

# GraphRagConfig 类继承自 LLMConfig，表示默认配置的参数化设置
class GraphRagConfig(LLMConfig):
    """Base class for the Default-Configuration parameterization settings."""

    # 获取对象的字符串表示形式
    def __repr__(self) -> str:
        """Get a string representation."""
        return pformat(self, highlight=False)

    # 获取对象的字符串表示形式
    def __str__(self):
        """Get a string representation."""
        return self.model_dump_json(indent=4)

    # 根目录的配置，表示配置的根目录路径
    root_dir: str = Field(
        description="The root directory for the configuration.", default=None
    )

    # 报告配置，表示报告的配置项
    reporting: ReportingConfig = Field(
        description="The reporting configuration.", default=ReportingConfig()
    )
    """The reporting configuration."""

    # 存储配置，表示存储的配置项
    storage: StorageConfig = Field(
        description="The storage configuration.", default=StorageConfig()
    )
    """The storage configuration."""

    # 缓存配置，表示缓存的配置项
    cache: CacheConfig = Field(
        description="The cache configuration.", default=CacheConfig()
    )
    """The cache configuration."""

    # 输入配置，表示输入的配置项
    input: InputConfig = Field(
        description="The input configuration.", default=InputConfig()
    )
    """The input configuration."""

    # 图嵌入配置，表示图嵌入的配置项
    embed_graph: EmbedGraphConfig = Field(
        description="Graph embedding configuration.",
        default=EmbedGraphConfig(),
    )
    """Graph Embedding configuration."""

    # 嵌入配置，表示要使用的嵌入LLM配置
    embeddings: TextEmbeddingConfig = Field(
        description="The embeddings LLM configuration to use.",
        default=TextEmbeddingConfig(),
    )
    """The embeddings LLM configuration to use."""

    # 分块配置，表示要使用的分块配置
    chunks: ChunkingConfig = Field(
        description="The chunking configuration to use.",
        default=ChunkingConfig(),
    )
    """The chunking configuration to use."""

    # 快照配置，表示要使用的快照配置
    snapshots: SnapshotsConfig = Field(
        description="The snapshots configuration to use.",
        default=SnapshotsConfig(),
    )
    """The snapshots configuration to use."""
    entity_extraction: EntityExtractionConfig = Field(
        description="The entity extraction configuration to use.",
        default=EntityExtractionConfig(),
    )
    """定义实体抽取配置，并设置默认配置为EntityExtractionConfig()实例。"""

    summarize_descriptions: SummarizeDescriptionsConfig = Field(
        description="The description summarization configuration to use.",
        default=SummarizeDescriptionsConfig(),
    )
    """定义描述摘要配置，并设置默认配置为SummarizeDescriptionsConfig()实例。"""

    community_reports: CommunityReportsConfig = Field(
        description="The community reports configuration to use.",
        default=CommunityReportsConfig(),
    )
    """定义社区报告配置，并设置默认配置为CommunityReportsConfig()实例。"""

    claim_extraction: ClaimExtractionConfig = Field(
        description="The claim extraction configuration to use.",
        default=ClaimExtractionConfig(
            enabled=defs.CLAIM_EXTRACTION_ENABLED,
        ),
    )
    """定义索赔提取配置，并设置默认配置为ClaimExtractionConfig()实例。
    这里的默认值中，启用了defs.CLAIM_EXTRACTION_ENABLED定义的索赔提取功能。"""

    cluster_graph: ClusterGraphConfig = Field(
        description="The cluster graph configuration to use.",
        default=ClusterGraphConfig(),
    )
    """定义集群图配置，并设置默认配置为ClusterGraphConfig()实例。"""

    umap: UmapConfig = Field(
        description="The UMAP configuration to use.", default=UmapConfig()
    )
    """定义UMAP配置，并设置默认配置为UmapConfig()实例。"""

    local_search: LocalSearchConfig = Field(
        description="The local search configuration.", default=LocalSearchConfig()
    )
    """定义本地搜索配置，并设置默认配置为LocalSearchConfig()实例。"""

    global_search: GlobalSearchConfig = Field(
        description="The global search configuration.", default=GlobalSearchConfig()
    )
    """定义全局搜索配置，并设置默认配置为GlobalSearchConfig()实例。"""

    encoding_model: str = Field(
        description="The encoding model to use.", default=defs.ENCODING_MODEL
    )
    """定义编码模型配置，并设置默认配置为defs.ENCODING_MODEL定义的编码模型。"""

    skip_workflows: list[str] = Field(
        description="The workflows to skip, usually for testing reasons.", default=[]
    )
    """定义跳过工作流配置，并设置默认为空列表，通常用于测试目的。"""
```