# `.\graphrag\graphrag\config\defaults.py`

```py
"""Common default configuration values."""

# 从 datashaper 模块导入 AsyncType 类型
from datashaper import AsyncType

# 从当前目录的 enums 模块导入以下枚举类型
from .enums import (
    CacheType,
    InputFileType,
    InputType,
    LLMType,
    ReportingType,
    StorageType,
    TextEmbeddingTarget,
)

# 异步模式设定为使用线程
ASYNC_MODE = AsyncType.Threaded

# 编码模型设定为 "cl100k_base"
ENCODING_MODEL = "cl100k_base"

#
# LLM Parameters
#

# 语言模型类型设定为 OpenAIChat
LLM_TYPE = LLMType.OpenAIChat

# 语言模型使用的具体模型名称为 "gpt-4-turbo-preview"
LLM_MODEL = "gpt-4-turbo-preview"

# 语言模型的最大 token 数量设定为 4000
LLM_MAX_TOKENS = 4000

# 语言模型的温度设定为 0，即完全确定性的输出
LLM_TEMPERATURE = 0

# 语言模型的 top-p 参数设定为 1，保留所有可能的 token
LLM_TOP_P = 1

# 每个请求的语言模型实例数设定为 1
LLM_N = 1

# 单个请求的超时时间设定为 180.0 秒
LLM_REQUEST_TIMEOUT = 180.0

# 每分钟的 token 数设定为 0，表示无限制
LLM_TOKENS_PER_MINUTE = 0

# 每分钟的请求数设定为 0，表示无限制
LLM_REQUESTS_PER_MINUTE = 0

# 最大重试次数设定为 10
LLM_MAX_RETRIES = 10

# 每次重试之间的最大等待时间设定为 10.0 秒
LLM_MAX_RETRY_WAIT = 10.0

# 在达到速率限制时是否休眠，设定为 True
LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION = True

# 并发请求的最大数目设定为 25
LLM_CONCURRENT_REQUESTS = 25

#
# Text Embedding Parameters
#

# 文本嵌入的类型设定为 OpenAIEmbedding
EMBEDDING_TYPE = LLMType.OpenAIEmbedding

# 文本嵌入的具体模型名称为 "text-embedding-3-small"
EMBEDDING_MODEL = "text-embedding-3-small"

# 文本嵌入的批次大小设定为 16
EMBEDDING_BATCH_SIZE = 16

# 单个文本嵌入批次的最大 token 数量设定为 8191
EMBEDDING_BATCH_MAX_TOKENS = 8191

# 文本嵌入的目标类型设定为必需
EMBEDDING_TARGET = TextEmbeddingTarget.required

# 缓存类型设定为文件
CACHE_TYPE = CacheType.file

# 缓存的基础目录设定为 "cache"
CACHE_BASE_DIR = "cache"

# 分块处理时的块大小设定为 1200
CHUNK_SIZE = 1200

# 分块处理时的重叠大小设定为 100
CHUNK_OVERLAP = 100

# 分块处理时根据的分组列设定为 ["id"]
CHUNK_GROUP_BY_COLUMNS = ["id"]

# 声明描述的内容，用于信息发现
CLAIM_DESCRIPTION = (
    "Any claims or facts that could be relevant to information discovery."
)

# 每个请求最多获取的信息量设定为 1
CLAIM_MAX_GLEANINGS = 1

# 是否启用实体抽取设定为 False
CLAIM_EXTRACTION_ENABLED = False

# 最大群集大小设定为 10
MAX_CLUSTER_SIZE = 10

# 社区报告的最大长度设定为 2000
COMMUNITY_REPORT_MAX_LENGTH = 2000

# 社区报告的最大输入长度设定为 8000
COMMUNITY_REPORT_MAX_INPUT_LENGTH = 8000

# 实体抽取的实体类型设定为 ["organization", "person", "geo", "event"]
ENTITY_EXTRACTION_ENTITY_TYPES = ["organization", "person", "geo", "event"]

# 实体抽取每次最多获取的信息量设定为 1
ENTITY_EXTRACTION_MAX_GLEANINGS = 1

# 输入文件类型设定为文本
INPUT_FILE_TYPE = InputFileType.text

# 输入的数据类型设定为文件
INPUT_TYPE = InputType.file

# 输入文件的基础目录设定为 "input"
INPUT_BASE_DIR = "input"

# 输入文件的编码设定为 "utf-8"
INPUT_FILE_ENCODING = "utf-8"

# 文本输入的列名设定为 "text"
INPUT_TEXT_COLUMN = "text"

# 输入的 CSV 文件匹配模式设定为 ".*\.csv$"
INPUT_CSV_PATTERN = ".*\\.csv$"

# 输入的文本文件匹配模式设定为 ".*\.txt$"
INPUT_TEXT_PATTERN = ".*\\.txt$"

# 并行处理时的间隔时间设定为 0.3 秒
PARALLELIZATION_STAGGER = 0.3

# 并行处理时的线程数设定为 50
PARALLELIZATION_NUM_THREADS = 50

# 是否启用 Node2Vec 算法设定为 False
NODE2VEC_ENABLED = False

# 每个 Node2Vec 实例的步行次数设定为 10
NODE2VEC_NUM_WALKS = 10

# 每次步行的长度设定为 40
NODE2VEC_WALK_LENGTH = 40

# Node2Vec 的窗口大小设定为 2
NODE2VEC_WINDOW_SIZE = 2

# Node2Vec 的迭代次数设定为 3
NODE2VEC_ITERATIONS = 3

# Node2Vec 的随机种子设定为 597832
NODE2VEC_RANDOM_SEED = 597832

# 报告的输出类型设定为文件
REPORTING_TYPE = ReportingType.file

# 报告的基础目录设定为 "output/${timestamp}/reports"
REPORTING_BASE_DIR = "output/${timestamp}/reports"

# 是否输出图形文件的 GraphML 格式设定为 False
SNAPSHOTS_GRAPHML = False

# 是否输出原始实体数据文件设定为 False
SNAPSHOTS_RAW_ENTITIES = False

# 是否输出顶层节点数据文件设定为 False
SNAPSHOTS_TOP_LEVEL_NODES = False

# 存储的基础目录设定为 "output/${timestamp}/artifacts"
STORAGE_BASE_DIR = "output/${timestamp}/artifacts"

# 存储类型设定为文件
STORAGE_TYPE = StorageType.file

# 描述文本总结的最大长度设定为 500
SUMMARIZE_DESCRIPTIONS_MAX_LENGTH = 500

# 是否启用 UMAP 算法设定为 False
UMAP_ENABLED = False

#
# Local Search
#

# 本地搜索时，文本单元比例设定为 0.5
LOCAL_SEARCH_TEXT_UNIT_PROP = 0.5

# 本地搜索时，社区比例设定为 0.1
LOCAL_SEARCH_COMMUNITY_PROP = 0.1

# 本地搜索时，对话历史的最大转数设定为 5
LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS = 5

# 本地搜索时，最常映射的实体数设定为 10
LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES = 10

# 本地搜索时，最常关联的关系数设定为 10
LOCAL_SEARCH_TOP_K_RELATIONSHIPS = 10

# 本地搜索时，最大 token 数量设定为 12000
LOCAL_SEARCH_MAX_TOKENS = 12_000

# 本地搜索时，语言模型的温度设定为 0
LOCAL_SEARCH_LLM_TEMPERATURE = 0

# 本地搜索时，语言模型的 top-p 参数设定为 1
LOCAL_SEARCH_LLM_TOP_P = 1

# 本地搜索时，每次请求的语言模型的实例数设定为 1
LOCAL_SEARCH_LLM_N = 1

# 本地搜索时，语言模型的最大 token 数量设定为 2000
LOCAL_SEARCH_LLM_MAX_TOKENS = 2000

#
# Global Search
#

# 全局搜索时，语言模型的温度设定为 0
GLOBAL_SEARCH_LLM_TEMPERATURE = 0

# 全局搜索时，语言模型的 top-p
```