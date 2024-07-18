# `.\graphrag\graphrag\index\init_content.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Content for the init CLI command."""

# 导入默认配置模块
import graphrag.config.defaults as defs

# 初始化 YAML 字符串，配置初始化命令的参数和数值
INIT_YAML = f"""
encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ${{GRAPHRAG_API_KEY}}
  type: {defs.LLM_TYPE.value} # or azure_openai_chat
  model: {defs.LLM_MODEL}
  model_supports_json: true # recommended if this is available for your model.
  # max_tokens: {defs.LLM_MAX_TOKENS}
  # request_timeout: {defs.LLM_REQUEST_TIMEOUT}
  # api_base: https://<instance>.openai.azure.com
  # api_version: 2024-02-15-preview
  # organization: <organization_id>
  # deployment_name: <azure_model_deployment_name>
  # tokens_per_minute: 150_000 # set a leaky bucket throttle
  # requests_per_minute: 10_000 # set a leaky bucket throttle
  # max_retries: {defs.LLM_MAX_RETRIES}
  # max_retry_wait: {defs.LLM_MAX_RETRY_WAIT}
  # sleep_on_rate_limit_recommendation: true # whether to sleep when azure suggests wait-times
  # concurrent_requests: {defs.LLM_CONCURRENT_REQUESTS} # the number of parallel inflight requests that may be made
  # temperature: {defs.LLM_TEMPERATURE} # temperature for sampling
  # top_p: {defs.LLM_TOP_P} # top-p sampling
  # n: {defs.LLM_N} # Number of completions to generate

parallelization:
  stagger: {defs.PARALLELIZATION_STAGGER}
  # num_threads: {defs.PARALLELIZATION_NUM_THREADS} # the number of threads to use for parallel processing

async_mode: {defs.ASYNC_MODE.value} # or asyncio

embeddings:
  ## parallelization: override the global parallelization settings for embeddings
  async_mode: {defs.ASYNC_MODE.value} # or asyncio
  llm:
    api_key: ${{GRAPHRAG_API_KEY}}
    type: {defs.EMBEDDING_TYPE.value} # or azure_openai_embedding
    model: {defs.EMBEDDING_MODEL}
    # api_base: https://<instance>.openai.azure.com
    # api_version: 2024-02-15-preview
    # organization: <organization_id>
    # deployment_name: <azure_model_deployment_name>
    # tokens_per_minute: 150_000 # set a leaky bucket throttle
    # requests_per_minute: 10_000 # set a leaky bucket throttle
    # max_retries: {defs.LLM_MAX_RETRIES}
    # max_retry_wait: {defs.LLM_MAX_RETRY_WAIT}
    # sleep_on_rate_limit_recommendation: true # whether to sleep when azure suggests wait-times
    # concurrent_requests: {defs.LLM_CONCURRENT_REQUESTS} # the number of parallel inflight requests that may be made
    # batch_size: {defs.EMBEDDING_BATCH_SIZE} # the number of documents to send in a single request
    # batch_max_tokens: {defs.EMBEDDING_BATCH_MAX_TOKENS} # the maximum number of tokens to send in a single request
    # target: {defs.EMBEDDING_TARGET.value} # or optional
  

chunks:
  size: {defs.CHUNK_SIZE}
  overlap: {defs.CHUNK_OVERLAP}
  group_by_columns: [{",".join(defs.CHUNK_GROUP_BY_COLUMNS)}] # by default, we don't allow chunks to cross documents
input:
  type: {defs.INPUT_TYPE.value} # 定义输入类型，可以是预定义常量或者 blob
  file_type: {defs.INPUT_FILE_TYPE.value} # 定义文件类型，可以是预定义常量或者 csv
  base_dir: "{defs.INPUT_BASE_DIR}" # 指定基础目录路径，使用预定义常量
  file_encoding: {defs.INPUT_FILE_ENCODING} # 定义文件编码方式，使用预定义常量
  file_pattern: ".*\\\\.txt$" # 定义文件名匹配模式，限定为以 .txt 结尾的文件

cache:
  type: {defs.CACHE_TYPE.value} # 指定缓存类型，可以是预定义常量或者 blob
  base_dir: "{defs.CACHE_BASE_DIR}" # 指定缓存基础目录路径，使用预定义常量
  # connection_string: <azure_blob_storage_connection_string> # Azure Blob 存储连接字符串（注释掉，可能是未使用的设置项）
  # container_name: <azure_blob_storage_container_name> # Azure Blob 存储容器名（注释掉，可能是未使用的设置项）

storage:
  type: {defs.STORAGE_TYPE.value} # 指定存储类型，可以是预定义常量或者 blob
  base_dir: "{defs.STORAGE_BASE_DIR}" # 指定存储基础目录路径，使用预定义常量
  # connection_string: <azure_blob_storage_connection_string> # Azure Blob 存储连接字符串（注释掉，可能是未使用的设置项）
  # container_name: <azure_blob_storage_container_name> # Azure Blob 存储容器名（注释掉，可能是未使用的设置项）

reporting:
  type: {defs.REPORTING_TYPE.value} # 指定报告输出类型，可以是预定义常量如 console 或 blob
  base_dir: "{defs.REPORTING_BASE_DIR}" # 指定报告输出基础目录路径，使用预定义常量
  # connection_string: <azure_blob_storage_connection_string> # Azure Blob 存储连接字符串（注释掉，可能是未使用的设置项）
  # container_name: <azure_blob_storage_container_name> # Azure Blob 存储容器名（注释掉，可能是未使用的设置项）

entity_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/entity_extraction.txt" # 指定实体提取的提示文件路径
  entity_types: [{",".join(defs.ENTITY_EXTRACTION_ENTITY_TYPES)}] # 定义实体类型列表，使用预定义常量
  max_gleanings: {defs.ENTITY_EXTRACTION_MAX_GLEANINGS} # 定义最大 gleanings 数量，使用预定义常量

summarize_descriptions:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/summarize_descriptions.txt" # 指定描述总结的提示文件路径
  max_length: {defs.SUMMARIZE_DESCRIPTIONS_MAX_LENGTH} # 定义描述总结的最大长度，使用预定义常量

claim_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  # enabled: true
  prompt: "prompts/claim_extraction.txt" # 指定索赔提取的提示文件路径
  description: "{defs.CLAIM_DESCRIPTION}" # 定义索赔描述，使用预定义常量
  max_gleanings: {defs.CLAIM_MAX_GLEANINGS} # 定义最大 gleanings 数量，使用预定义常量

community_reports:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/community_report.txt" # 指定社区报告的提示文件路径
  max_length: {defs.COMMUNITY_REPORT_MAX_LENGTH} # 定义社区报告的最大长度，使用预定义常量
  max_input_length: {defs.COMMUNITY_REPORT_MAX_INPUT_LENGTH} # 定义社区报告输入的最大长度，使用预定义常量

cluster_graph:
  max_cluster_size: {defs.MAX_CLUSTER_SIZE} # 定义集群图的最大集群大小，使用预定义常量

embed_graph:
  enabled: false # 是否启用图嵌入，如果为 true，将为节点生成 node2vec 嵌入（当前设置为 false）
  # num_walks: {defs.NODE2VEC_NUM_WALKS} # node2vec 参数：步行次数，使用预定义常量
  # walk_length: {defs.NODE2VEC_WALK_LENGTH} # node2vec 参数：步行长度，使用预定义常量
  # window_size: {defs.NODE2VEC_WINDOW_SIZE} # node2vec 参数：窗口大小，使用预定义常量
  # iterations: {defs.NODE2VEC_ITERATIONS} # node2vec 参数：迭代次数，使用预定义常量
  # random_seed: {defs.NODE2VEC_RANDOM_SEED} # node2vec 参数：随机种子，使用预定义常量

umap:
  enabled: false # 是否启用 UMAP 嵌入，当前设置为 false

snapshots:
  graphml: false # 是否生成 graphml 快照，当前设置为 false
  raw_entities: false # 是否生成原始实体快照，当前设置为 false
  top_level_nodes: false # 是否生成顶级节点快照，当前设置为 false
local_search:
  # text_unit_prop: {defs.LOCAL_SEARCH_TEXT_UNIT_PROP}  # 搜索本地文本单元属性的配置值
  # community_prop: {defs.LOCAL_SEARCH_COMMUNITY_PROP}  # 搜索本地社区属性的配置值
  # conversation_history_max_turns: {defs.LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS}  # 搜索本地对话历史的最大轮数配置值
  # top_k_mapped_entities: {defs.LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES}  # 搜索本地映射实体的前 k 个配置值
  # top_k_relationships: {defs.LOCAL_SEARCH_TOP_K_RELATIONSHIPS}  # 搜索本地关系的前 k 个配置值
  # llm_temperature: {defs.LOCAL_SEARCH_LLM_TEMPERATURE} # temperature for sampling  # 用于局部搜索的语言模型温度参数
  # llm_top_p: {defs.LOCAL_SEARCH_LLM_TOP_P} # top-p sampling  # 用于局部搜索的语言模型 top-p 抽样参数
  # llm_n: {defs.LOCAL_SEARCH_LLM_N} # Number of completions to generate  # 生成的局部搜索完成数
  # max_tokens: {defs.LOCAL_SEARCH_MAX_TOKENS}  # 局部搜索的最大 token 数

global_search:
  # llm_temperature: {defs.GLOBAL_SEARCH_LLM_TEMPERATURE} # temperature for sampling  # 用于全局搜索的语言模型温度参数
  # llm_top_p: {defs.GLOBAL_SEARCH_LLM_TOP_P} # top-p sampling  # 用于全局搜索的语言模型 top-p 抽样参数
  # llm_n: {defs.GLOBAL_SEARCH_LLM_N} # Number of completions to generate  # 生成的全局搜索完成数
  # max_tokens: {defs.GLOBAL_SEARCH_MAX_TOKENS}  # 全局搜索的最大 token 数
  # data_max_tokens: {defs.GLOBAL_SEARCH_DATA_MAX_TOKENS}  # 全局搜索数据部分的最大 token 数
  # map_max_tokens: {defs.GLOBAL_SEARCH_MAP_MAX_TOKENS}  # 全局搜索映射部分的最大 token 数
  # reduce_max_tokens: {defs.GLOBAL_SEARCH_REDUCE_MAX_TOKENS}  # 全局搜索减少部分的最大 token 数
  # concurrency: {defs.GLOBAL_SEARCH_CONCURRENCY}  # 全局搜索的并发数设置


注释：


INIT_DOTENV = """
GRAPHRAG_API_KEY=<API_KEY>
"""
```