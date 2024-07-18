# `.\graphrag\tests\unit\config\test_default_config.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入所需的模块
import json
import os
import re
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest
import yaml

# 导入默认配置模块
import graphrag.config.defaults as defs

# 导入配置相关模块和类
from graphrag.config import (
    ApiKeyMissingError,
    AzureApiBaseMissingError,
    AzureDeploymentNameMissingError,
    CacheConfig,
    CacheConfigInput,
    CacheType,
    ChunkingConfig,
    ChunkingConfigInput,
    ClaimExtractionConfig,
    ClaimExtractionConfigInput,
    ClusterGraphConfig,
    ClusterGraphConfigInput,
    CommunityReportsConfig,
    CommunityReportsConfigInput,
    EmbedGraphConfig,
    EmbedGraphConfigInput,
    EntityExtractionConfig,
    EntityExtractionConfigInput,
    GlobalSearchConfig,
    GraphRagConfig,
    GraphRagConfigInput,
    InputConfig,
    InputConfigInput,
    InputFileType,
    InputType,
    LLMParameters,
    LLMParametersInput,
    LocalSearchConfig,
    ParallelizationParameters,
    ReportingConfig,
    ReportingConfigInput,
    ReportingType,
    SnapshotsConfig,
    SnapshotsConfigInput,
    StorageConfig,
    StorageConfigInput,
    StorageType,
    SummarizeDescriptionsConfig,
    SummarizeDescriptionsConfigInput,
    TextEmbeddingConfig,
    TextEmbeddingConfigInput,
    UmapConfig,
    UmapConfigInput,
    create_graphrag_config,
)

# 导入索引相关模块和类
from graphrag.index import (
    PipelineConfig,
    PipelineCSVInputConfig,
    PipelineFileCacheConfig,
    PipelineFileReportingConfig,
    PipelineFileStorageConfig,
    PipelineInputConfig,
    PipelineTextInputConfig,
    PipelineWorkflowReference,
    create_pipeline_config,
)

# 获取当前文件所在目录
current_dir = os.path.dirname(__file__)

# 定义所有环境变量的映射字典
ALL_ENV_VARS = {
    "GRAPHRAG_API_BASE": "http://some/base",
    "GRAPHRAG_API_KEY": "test",
    "GRAPHRAG_API_ORGANIZATION": "test_org",
    "GRAPHRAG_API_PROXY": "http://some/proxy",
    "GRAPHRAG_API_VERSION": "v1234",
    "GRAPHRAG_ASYNC_MODE": "asyncio",
    "GRAPHRAG_CACHE_STORAGE_ACCOUNT_BLOB_URL": "cache_account_blob_url",
    "GRAPHRAG_CACHE_BASE_DIR": "/some/cache/dir",
    "GRAPHRAG_CACHE_CONNECTION_STRING": "test_cs1",
    "GRAPHRAG_CACHE_CONTAINER_NAME": "test_cn1",
    "GRAPHRAG_CACHE_TYPE": "blob",
    "GRAPHRAG_CHUNK_BY_COLUMNS": "a,b",
    "GRAPHRAG_CHUNK_OVERLAP": "12",
    "GRAPHRAG_CHUNK_SIZE": "500",
    "GRAPHRAG_CLAIM_EXTRACTION_ENABLED": "True",
    "GRAPHRAG_CLAIM_EXTRACTION_DESCRIPTION": "test 123",
    "GRAPHRAG_CLAIM_EXTRACTION_MAX_GLEANINGS": "5000",
    "GRAPHRAG_CLAIM_EXTRACTION_PROMPT_FILE": "tests/unit/config/prompt-a.txt",
    "GRAPHRAG_COMMUNITY_REPORTS_MAX_LENGTH": "23456",
    "GRAPHRAG_COMMUNITY_REPORTS_PROMPT_FILE": "tests/unit/config/prompt-b.txt",
    "GRAPHRAG_EMBEDDING_BATCH_MAX_TOKENS": "17",
    "GRAPHRAG_EMBEDDING_BATCH_SIZE": "1000000",
    "GRAPHRAG_EMBEDDING_CONCURRENT_REQUESTS": "12",
    "GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME": "model-deployment-name",
}
    # 程序配置：定义了多个环境变量和其对应的值，用于控制程序的行为和设置
    
    "GRAPHRAG_EMBEDDING_MAX_RETRIES": "3",  # 最大重试次数
    "GRAPHRAG_EMBEDDING_MAX_RETRY_WAIT": "0.1123",  # 每次重试之间的最大等待时间
    "GRAPHRAG_EMBEDDING_MODEL": "text-embedding-2",  # 文本嵌入模型
    "GRAPHRAG_EMBEDDING_REQUESTS_PER_MINUTE": "500",  # 每分钟的请求限制
    "GRAPHRAG_EMBEDDING_SKIP": "a1,b1,c1",  # 跳过的项列表
    "GRAPHRAG_EMBEDDING_SLEEP_ON_RATE_LIMIT_RECOMMENDATION": "False",  # 是否在达到速率限制时睡眠
    "GRAPHRAG_EMBEDDING_TARGET": "all",  # 嵌入目标
    "GRAPHRAG_EMBEDDING_THREAD_COUNT": "2345",  # 线程计数
    "GRAPHRAG_EMBEDDING_THREAD_STAGGER": "0.456",  # 线程错开时间
    "GRAPHRAG_EMBEDDING_TOKENS_PER_MINUTE": "7000",  # 每分钟的令牌数
    "GRAPHRAG_EMBEDDING_TYPE": "azure_openai_embedding",  # 嵌入类型
    "GRAPHRAG_ENCODING_MODEL": "test123",  # 编码模型
    "GRAPHRAG_INPUT_STORAGE_ACCOUNT_BLOB_URL": "input_account_blob_url",  # 输入存储帐户 Blob URL
    "GRAPHRAG_ENTITY_EXTRACTION_ENTITY_TYPES": "cat,dog,elephant",  # 实体类型
    "GRAPHRAG_ENTITY_EXTRACTION_MAX_GLEANINGS": "112",  # 最大收集数
    "GRAPHRAG_ENTITY_EXTRACTION_PROMPT_FILE": "tests/unit/config/prompt-c.txt",  # 提示文件路径
    "GRAPHRAG_INPUT_BASE_DIR": "/some/input/dir",  # 输入基础目录
    "GRAPHRAG_INPUT_CONNECTION_STRING": "input_cs",  # 输入连接字符串
    "GRAPHRAG_INPUT_CONTAINER_NAME": "input_cn",  # 输入容器名称
    "GRAPHRAG_INPUT_DOCUMENT_ATTRIBUTE_COLUMNS": "test1,test2",  # 文档属性列
    "GRAPHRAG_INPUT_ENCODING": "utf-16",  # 输入编码
    "GRAPHRAG_INPUT_FILE_PATTERN": ".*\\test\\.txt$",  # 文件模式匹配
    "GRAPHRAG_INPUT_SOURCE_COLUMN": "test_source",  # 输入源列
    "GRAPHRAG_INPUT_TYPE": "blob",  # 输入类型
    "GRAPHRAG_INPUT_TEXT_COLUMN": "test_text",  # 文本列
    "GRAPHRAG_INPUT_TIMESTAMP_COLUMN": "test_timestamp",  # 时间戳列
    "GRAPHRAG_INPUT_TIMESTAMP_FORMAT": "test_format",  # 时间戳格式
    "GRAPHRAG_INPUT_TITLE_COLUMN": "test_title",  # 标题列
    "GRAPHRAG_INPUT_FILE_TYPE": "text",  # 文件类型
    "GRAPHRAG_LLM_CONCURRENT_REQUESTS": "12",  # 并发请求数
    "GRAPHRAG_LLM_DEPLOYMENT_NAME": "model-deployment-name-x",  # 部署名称
    "GRAPHRAG_LLM_MAX_RETRIES": "312",  # 最大重试次数
    "GRAPHRAG_LLM_MAX_RETRY_WAIT": "0.1122",  # 每次重试之间的最大等待时间
    "GRAPHRAG_LLM_MAX_TOKENS": "15000",  # 最大令牌数
    "GRAPHRAG_LLM_MODEL_SUPPORTS_JSON": "true",  # 模型是否支持 JSON
    "GRAPHRAG_LLM_MODEL": "test-llm",  # 语言模型
    "GRAPHRAG_LLM_N": "1",  # N 值
    "GRAPHRAG_LLM_REQUEST_TIMEOUT": "12.7",  # 请求超时时间
    "GRAPHRAG_LLM_REQUESTS_PER_MINUTE": "900",  # 每分钟的请求限制
    "GRAPHRAG_LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION": "False",  # 是否在达到速率限制时睡眠
    "GRAPHRAG_LLM_THREAD_COUNT": "987",  # 线程计数
    "GRAPHRAG_LLM_THREAD_STAGGER": "0.123",  # 线程错开时间
    "GRAPHRAG_LLM_TOKENS_PER_MINUTE": "8000",  # 每分钟的令牌数
    "GRAPHRAG_LLM_TYPE": "azure_openai_chat",  # 模型类型
    "GRAPHRAG_MAX_CLUSTER_SIZE": "123",  # 最大集群大小
    "GRAPHRAG_NODE2VEC_ENABLED": "true",  # 是否启用 Node2Vec
    "GRAPHRAG_NODE2VEC_ITERATIONS": "878787",  # Node2Vec 迭代次数
    "GRAPHRAG_NODE2VEC_NUM_WALKS": "5000000",  # Node2Vec 的行走次数
    "GRAPHRAG_NODE2VEC_RANDOM_SEED": "010101",  # Node2Vec 的随机种子
    "GRAPHRAG_NODE2VEC_WALK_LENGTH": "555111",  # Node2Vec 的行走长度
    "GRAPHRAG_NODE2VEC_WINDOW_SIZE": "12345",  # Node2Vec 的窗口大小
    "GRAPHRAG_REPORTING_STORAGE_ACCOUNT_BLOB_URL": "reporting_account_blob_url",  # 报告存储帐户 Blob URL
    "GRAPHRAG_REPORTING_BASE_DIR": "/some/reporting/dir",  # 报告基础目录
    "GRAPHRAG_REPORTING_CONNECTION_STRING": "test_cs2",  # 报告连接字符串
    "GRAPHRAG_REPORTING_CONTAINER_NAME": "test_cn2",  # 报告容器名称
    "GRAPHRAG_REPORTING_TYPE": "blob",  # 报告类型
    "GRAPHRAG_SKIP_WORKFLOWS": "a,b,c",  # 跳过的工作流列表
    "GRAPHRAG_SNAPSHOT_GRAPHML": "true",  # 是否生成 GraphML 快照
    "GRAPHRAG_SNAPSHOT_RAW_ENTITIES": "true",  # 是否生成原始实体快照
    "GRAPHRAG_SNAPSHOT_TOP_LEVEL_NODES": "true",  # 是否生成顶级节点快照
    # 定义环境变量，存储与存储帐户 Blob URL 相关的键
    "GRAPHRAG_STORAGE_STORAGE_ACCOUNT_BLOB_URL": "storage_account_blob_url",
    # 定义基础目录路径
    "GRAPHRAG_STORAGE_BASE_DIR": "/some/storage/dir",
    # 定义连接字符串，用于存储服务的测试用途
    "GRAPHRAG_STORAGE_CONNECTION_STRING": "test_cs",
    # 定义存储容器的名称，用于存储服务的测试用途
    "GRAPHRAG_STORAGE_CONTAINER_NAME": "test_cn",
    # 定义存储类型为 Blob 存储
    "GRAPHRAG_STORAGE_TYPE": "blob",
    # 定义摘要描述的最大长度
    "GRAPHRAG_SUMMARIZE_DESCRIPTIONS_MAX_LENGTH": "12345",
    # 定义摘要描述提示文件的路径
    "GRAPHRAG_SUMMARIZE_DESCRIPTIONS_PROMPT_FILE": "tests/unit/config/prompt-d.txt",
    # 定义语言模型生成文本时的温度参数
    "GRAPHRAG_LLM_TEMPERATURE": "0.0",
    # 定义语言模型生成文本时的 top-p 参数
    "GRAPHRAG_LLM_TOP_P": "1.0",
    # 启用 UMAP 功能的标志位
    "GRAPHRAG_UMAP_ENABLED": "true",
    # 定义本地搜索的文本单元属性值
    "GRAPHRAG_LOCAL_SEARCH_TEXT_UNIT_PROP": "0.713",
    # 定义本地搜索的社区属性值
    "GRAPHRAG_LOCAL_SEARCH_COMMUNITY_PROP": "0.1234",
    # 定义本地搜索时语言模型的温度参数
    "GRAPHRAG_LOCAL_SEARCH_LLM_TEMPERATURE": "0.1",
    # 定义本地搜索时语言模型的 top-p 参数
    "GRAPHRAG_LOCAL_SEARCH_LLM_TOP_P": "0.9",
    # 定义本地搜索时语言模型的 n 参数
    "GRAPHRAG_LOCAL_SEARCH_LLM_N": "2",
    # 定义本地搜索时语言模型的最大 tokens 数量
    "GRAPHRAG_LOCAL_SEARCH_LLM_MAX_TOKENS": "12",
    # 定义本地搜索时关系的 top-k 值
    "GRAPHRAG_LOCAL_SEARCH_TOP_K_RELATIONSHIPS": "15",
    # 定义本地搜索时实体的 top-k 值
    "GRAPHRAG_LOCAL_SEARCH_TOP_K_ENTITIES": "14",
    # 定义本地搜索时对话历史的最大轮数
    "GRAPHRAG_LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS": "2",
    # 定义本地搜索的最大 tokens 数量
    "GRAPHRAG_LOCAL_SEARCH_MAX_TOKENS": "142435",
    # 定义全局搜索时语言模型的温度参数
    "GRAPHRAG_GLOBAL_SEARCH_LLM_TEMPERATURE": "0.1",
    # 定义全局搜索时语言模型的 top-p 参数
    "GRAPHRAG_GLOBAL_SEARCH_LLM_TOP_P": "0.9",
    # 定义全局搜索时语言模型的 n 参数
    "GRAPHRAG_GLOBAL_SEARCH_LLM_N": "2",
    # 定义全局搜索的最大 tokens 数量
    "GRAPHRAG_GLOBAL_SEARCH_MAX_TOKENS": "5123",
    # 定义全局搜索数据的最大 tokens 数量
    "GRAPHRAG_GLOBAL_SEARCH_DATA_MAX_TOKENS": "123",
    # 定义全局搜索地图的最大 tokens 数量
    "GRAPHRAG_GLOBAL_SEARCH_MAP_MAX_TOKENS": "4123",
    # 定义全局搜索的并发度
    "GRAPHRAG_GLOBAL_SEARCH_CONCURRENCY": "7",
    # 定义全局搜索减少 tokens 的最大数量
    "GRAPHRAG_GLOBAL_SEARCH_REDUCE_MAX_TOKENS": "15432",
}

# 定义一个单元测试类 TestDefaultConfig，继承自 unittest.TestCase
class TestDefaultConfig(unittest.TestCase):

    # 定义测试方法 test_clear_warnings，用于清除未使用的导入警告
    def test_clear_warnings(self):
        """Just clearing unused import warnings"""
        # 断言各个配置对象是否不为 None
        assert CacheConfig is not None
        assert ChunkingConfig is not None
        assert ClaimExtractionConfig is not None
        assert ClusterGraphConfig is not None
        assert CommunityReportsConfig is not None
        assert EmbedGraphConfig is not None
        assert EntityExtractionConfig is not None
        assert GlobalSearchConfig is not None
        assert GraphRagConfig is not None
        assert InputConfig is not None
        assert LLMParameters is not None
        assert LocalSearchConfig is not None
        assert ParallelizationParameters is not None
        assert ReportingConfig is not None
        assert SnapshotsConfig is not None
        assert StorageConfig is not None
        assert SummarizeDescriptionsConfig is not None
        assert TextEmbeddingConfig is not None
        assert UmapConfig is not None
        assert PipelineConfig is not None
        assert PipelineFileReportingConfig is not None
        assert PipelineFileStorageConfig is not None
        assert PipelineInputConfig is not None
        assert PipelineFileCacheConfig is not None
        assert PipelineWorkflowReference is not None

    # 使用 mock.patch.dict 方法设置环境变量，测试方法 test_string_repr
    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=True)
    def test_string_repr(self):
        # __str__ 方法的返回值可以被解析为 JSON
        config = create_graphrag_config()
        string_repr = str(config)
        assert string_repr is not None
        assert json.loads(string_repr) is not None

        # __repr__ 方法的返回值可以被 eval() 解析
        repr_str = config.__repr__()
        # TODO: 在 datashaper 枚举中添加 __repr__ 方法
        repr_str = repr_str.replace("async_mode=<AsyncType.Threaded: 'threaded'>,", "")
        assert eval(repr_str) is not None

        # Pipeline config 的 __str__ 方法的返回值可以被解析为 JSON
        pipeline_config = create_pipeline_config(config)
        string_repr = str(pipeline_config)
        assert string_repr is not None
        assert json.loads(string_repr) is not None

        # Pipeline config 的 __repr__ 方法的返回值可以被 eval() 解析
        repr_str = pipeline_config.__repr__()
        # TODO: 在 datashaper 枚举中添加 __repr__ 方法
        repr_str = repr_str.replace(
            "'async_mode': <AsyncType.Threaded: 'threaded'>,", ""
        )
        assert eval(repr_str) is not None

    # 使用 mock.patch.dict 方法设置空的环境变量，测试方法 test_default_config_with_no_env_vars_throws
    @mock.patch.dict(os.environ, {}, clear=True)
    def test_default_config_with_no_env_vars_throws(self):
        with pytest.raises(ApiKeyMissingError):
            # 这应该抛出一个错误，因为缺少 API 密钥
            create_pipeline_config(create_graphrag_config())

    # 使用 mock.patch.dict 方法设置包含 GRAPHRAG_API_KEY 的环境变量，测试方法 test_default_config_with_api_key_passes
    @mock.patch.dict(os.environ, {"GRAPHRAG_API_KEY": "test"}, clear=True)
    def test_default_config_with_api_key_passes(self):
        # 不应该抛出异常
        config = create_pipeline_config(create_graphrag_config())
        assert config is not None
    @mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=True)
    # 使用 mock.patch 装饰器将环境变量中的 OPENAI_API_KEY 设置为 "test"，并清除之前的设置
    def test_default_config_with_oai_key_passes_envvar(self):
        # 不会抛出异常
        # 创建默认的管道配置，并验证返回的配置对象不为空
        config = create_pipeline_config(create_graphrag_config())
        assert config is not None
    
    def test_default_config_with_oai_key_passes_obj(self):
        # 不会抛出异常
        # 创建包含指定 LLG 参数的 GraphRag 配置，并验证返回的配置对象不为空
        config = create_pipeline_config(
            create_graphrag_config({"llm": {"api_key": "test"}})
        )
        assert config is not None
    
    @mock.patch.dict(
        os.environ,
        {"GRAPHRAG_API_KEY": "test", "GRAPHRAG_LLM_TYPE": "azure_openai_chat"},
        clear=True,
    )
    # 使用 mock.patch 装饰器将环境变量中的 GRAPHRAG_API_KEY 和 GRAPHRAG_LLM_TYPE 设置为指定值，并清除之前的设置
    def test_throws_if_azure_is_used_without_api_base_envvar(self):
        # 预期引发 AzureApiBaseMissingError 异常
        with pytest.raises(AzureApiBaseMissingError):
            create_graphrag_config()
    
    @mock.patch.dict(os.environ, {"GRAPHRAG_API_KEY": "test"}, clear=True)
    # 使用 mock.patch 装饰器将环境变量中的 GRAPHRAG_API_KEY 设置为 "test"，并清除之前的设置
    def test_throws_if_azure_is_used_without_api_base_obj(self):
        # 预期引发 AzureApiBaseMissingError 异常
        # 创建包含指定 LLG 参数的 GraphRag 配置对象，其中指定了 Azure LLM 的类型
        with pytest.raises(AzureApiBaseMissingError):
            create_graphrag_config(
                GraphRagConfigInput(llm=LLMParametersInput(type="azure_openai_chat"))
            )
    
    @mock.patch.dict(
        os.environ,
        {
            "GRAPHRAG_API_KEY": "test",
            "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
            "GRAPHRAG_API_BASE": "http://some/base",
        },
        clear=True,
    )
    # 使用 mock.patch 装饰器将环境变量中的 GRAPHRAG_API_KEY、GRAPHRAG_LLM_TYPE 和 GRAPHRAG_API_BASE 设置为指定值，并清除之前的设置
    def test_throws_if_azure_is_used_without_llm_deployment_name_envvar(self):
        # 预期引发 AzureDeploymentNameMissingError 异常
        with pytest.raises(AzureDeploymentNameMissingError):
            create_graphrag_config()
    
    @mock.patch.dict(os.environ, {"GRAPHRAG_API_KEY": "test"}, clear=True)
    # 使用 mock.patch 装饰器将环境变量中的 GRAPHRAG_API_KEY 设置为 "test"，并清除之前的设置
    def test_throws_if_azure_is_used_without_llm_deployment_name_obj(self):
        # 预期引发 AzureDeploymentNameMissingError 异常
        # 创建包含指定 LLG 参数的 GraphRag 配置对象，其中指定了 Azure LLM 的类型和 API 基础路径
        with pytest.raises(AzureDeploymentNameMissingError):
            create_graphrag_config(
                GraphRagConfigInput(
                    llm=LLMParametersInput(
                        type="azure_openai_chat", api_base="http://some/base"
                    )
                )
            )
    
    @mock.patch.dict(
        os.environ,
        {
            "GRAPHRAG_API_KEY": "test",
            "GRAPHRAG_EMBEDDING_TYPE": "azure_openai_embedding",
            "GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME": "x",
        },
        clear=True,
    )
    # 使用 mock.patch 装饰器将环境变量中的 GRAPHRAG_API_KEY、GRAPHRAG_EMBEDDING_TYPE 和 GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME 设置为指定值，并清除之前的设置
    def test_throws_if_azure_is_used_without_embedding_api_base_envvar(self):
        # 预期引发 AzureApiBaseMissingError 异常
        with pytest.raises(AzureApiBaseMissingError):
            create_graphrag_config()
    
    @mock.patch.dict(os.environ, {"GRAPHRAG_API_KEY": "test"}, clear=True)
    # 使用 mock.patch 装饰器将环境变量中的 GRAPHRAG_API_KEY 设置为 "test"，并清除之前的设置
    def test_throws_if_azure_is_used_without_embedding_api_base_obj(self):
        # 预期引发 AzureApiBaseMissingError 异常
        # 创建包含指定 Embeddings 配置参数的 GraphRag 配置对象，其中指定了 Azure LLM 的类型和部署名称
        with pytest.raises(AzureApiBaseMissingError):
            create_graphrag_config(
                GraphRagConfigInput(
                    embeddings=TextEmbeddingConfigInput(
                        llm=LLMParametersInput(
                            type="azure_openai_embedding",
                            deployment_name="x",
                        )
                    ),
                )
            )
    @mock.patch.dict(
        os.environ,
        {
            "GRAPHRAG_API_KEY": "test",
            "GRAPHRAG_API_BASE": "http://some/base",
            "GRAPHRAG_LLM_DEPLOYMENT_NAME": "x",
            "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
            "GRAPHRAG_EMBEDDING_TYPE": "azure_openai_embedding",
        },
        clear=True,
    )
    def test_throws_if_azure_is_used_without_embedding_deployment_name_envvar(self):
        # 使用 mock.patch.dict 设置环境变量，模拟对应的键值对
        with pytest.raises(AzureDeploymentNameMissingError):
            # 期望抛出 AzureDeploymentNameMissingError 异常
            create_graphrag_config()
    
    @mock.patch.dict(os.environ, {"GRAPHRAG_API_KEY": "test"}, clear=True)
    def test_throws_if_azure_is_used_without_embedding_deployment_name_obj(self):
        # 使用 mock.patch.dict 设置环境变量，模拟对应的键值对
        with pytest.raises(AzureDeploymentNameMissingError):
            # 期望抛出 AzureDeploymentNameMissingError 异常
            create_graphrag_config(
                GraphRagConfigInput(
                    llm=LLMParametersInput(
                        type="azure_openai_chat",
                        api_base="http://some/base",
                        deployment_name="model-deployment-name-x",
                    ),
                    embeddings=TextEmbeddingConfigInput(
                        llm=LLMParametersInput(
                            type="azure_openai_embedding",
                        )
                    ),
                )
            )
    
    @mock.patch.dict(os.environ, {"GRAPHRAG_API_KEY": "test"}, clear=True)
    def test_minimim_azure_config_object(self):
        # 使用 mock.patch.dict 设置环境变量，模拟对应的键值对
        config = create_graphrag_config(
            GraphRagConfigInput(
                llm=LLMParametersInput(
                    type="azure_openai_chat",
                    api_base="http://some/base",
                    deployment_name="model-deployment-name-x",
                ),
                embeddings=TextEmbeddingConfigInput(
                    llm=LLMParametersInput(
                        type="azure_openai_embedding",
                        deployment_name="model-deployment-name",
                    )
                ),
            )
        )
        assert config is not None  # 断言 config 不为 None
    
    @mock.patch.dict(
        os.environ,
        {
            "GRAPHRAG_API_KEY": "test",
            "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
            "GRAPHRAG_LLM_DEPLOYMENT_NAME": "x",
        },
        clear=True,
    )
    def test_throws_if_azure_is_used_without_api_base(self):
        # 使用 mock.patch.dict 设置环境变量，模拟对应的键值对
        with pytest.raises(AzureApiBaseMissingError):
            # 期望抛出 AzureApiBaseMissingError 异常
            create_graphrag_config()
    
    @mock.patch.dict(
        os.environ,
        {
            "GRAPHRAG_API_KEY": "test",
            "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
            "GRAPHRAG_LLM_API_BASE": "http://some/base",
        },
        clear=True,
    )
    def test_throws_if_azure_is_used_without_llm_deployment_name(self):
        # 使用 mock.patch.dict 设置环境变量，模拟对应的键值对
        with pytest.raises(AzureDeploymentNameMissingError):
            # 期望抛出 AzureDeploymentNameMissingError 异常
            create_graphrag_config()
    @mock.patch.dict(
        os.environ,
        {
            "GRAPHRAG_API_KEY": "test",
            "GRAPHRAG_LLM_TYPE": "azure_openai_chat",
            "GRAPHRAG_API_BASE": "http://some/base",
            "GRAPHRAG_LLM_DEPLOYMENT_NAME": "model-deployment-name-x",
            "GRAPHRAG_EMBEDDING_TYPE": "azure_openai_embedding",
        },
        clear=True,
    )
    # 使用 mock.patch.dict() 模拟设置环境变量字典，清空现有环境变量
    def test_throws_if_azure_is_used_without_embedding_deployment_name(self):
        # 使用 pytest.raises 检测是否会抛出 AzureDeploymentNameMissingError 异常
        with pytest.raises(AzureDeploymentNameMissingError):
            create_graphrag_config()

    @mock.patch.dict(
        os.environ,
        {"GRAPHRAG_API_KEY": "test", "GRAPHRAG_INPUT_FILE_TYPE": "csv"},
        clear=True,
    )
    # 使用 mock.patch.dict() 模拟设置环境变量字典，清空现有环境变量
    def test_csv_input_returns_correct_config(self):
        # 创建配置对象，传入 create_graphrag_config() 的结果，并断言 root_dir 正确
        config = create_pipeline_config(create_graphrag_config(root_dir="/some/root"))
        assert config.root_dir == "/some/root"
        # 确保输入是 CSV 输入
        assert isinstance(config.input, PipelineCSVInputConfig)
        assert (config.input.file_pattern or "") == ".*\\.csv$"  # type: ignore

    @mock.patch.dict(
        os.environ,
        {"GRAPHRAG_API_KEY": "test", "GRAPHRAG_INPUT_FILE_TYPE": "text"},
        clear=True,
    )
    # 使用 mock.patch.dict() 模拟设置环境变量字典，清空现有环境变量
    def test_text_input_returns_correct_config(self):
        # 创建配置对象，传入 create_graphrag_config() 的结果，并断言输入类型是文本
        config = create_pipeline_config(create_graphrag_config(root_dir="."))
        assert isinstance(config.input, PipelineTextInputConfig)
        assert config.input is not None
        assert (config.input.file_pattern or "") == ".*\\.txt$"  # type: ignore

    # 测试函数：确保所有环境变量的设置准确
    def test_all_env_vars_is_accurate(self):
        # 定义文档路径
        env_var_docs_path = Path("docsite/posts/config/env_vars.md")
        query_docs_path = Path("docsite/posts/query/3-cli.md")

        # 读取文档内容为字符串
        env_var_docs = env_var_docs_path.read_text(encoding="utf-8")
        query_docs = query_docs_path.read_text(encoding="utf-8")

        # 定义函数，从文本中找到环境变量名的集合
        def find_envvar_names(text) -> set[str]:
            pattern = r"`(GRAPHRAG_[^`]+)`"
            found = re.findall(pattern, text)
            found = {f for f in found if not f.endswith("_")}
            return {*found}

        # 查找环境变量名并合并
        graphrag_strings = find_envvar_names(env_var_docs) | find_envvar_names(
            query_docs
        )

        # 找出缺失的环境变量
        missing = {s for s in graphrag_strings if s not in ALL_ENV_VARS} - {
            # 移除已被基础 LLM 连接配置包含的环境变量
            "GRAPHRAG_LLM_API_KEY",
            "GRAPHRAG_LLM_API_BASE",
            "GRAPHRAG_LLM_API_VERSION",
            "GRAPHRAG_LLM_API_ORGANIZATION",
            "GRAPHRAG_LLM_API_PROXY",
            "GRAPHRAG_EMBEDDING_API_KEY",
            "GRAPHRAG_EMBEDDING_API_BASE",
            "GRAPHRAG_EMBEDDING_API_VERSION",
            "GRAPHRAG_EMBEDDING_API_ORGANIZATION",
            "GRAPHRAG_EMBEDDING_API_PROXY",
        }

        # 如果有缺失的环境变量，则抛出 ValueError 异常
        if missing:
            msg = f"{len(missing)} missing env vars: {missing}"
            print(msg)
            raise ValueError(msg)
    @mock.patch.dict(
        os.environ,
        {"GRAPHRAG_API_KEY": "test"},
        clear=True,
    )
    @mock.patch.dict(
        os.environ,
        ALL_ENV_VARS,
        clear=True,
    )
    @mock.patch.dict(os.environ, {"API_KEY_X": "test"}, clear=True)
    @mock.patch.dict(
        os.environ,
        {"GRAPHRAG_API_KEY": "test"},
        clear=True,
    )



    # 使用 `mock.patch.dict` 装饰器设置环境变量 `GRAPHRAG_API_KEY` 为 "test"，并清除所有之前设置的环境变量
    # 此装饰器用于测试 `create_graphrag_config` 函数在环境变量设置下的行为
    def test_prompt_file_reading(self):
        # 创建 `create_graphrag_config` 的配置对象 `config`，指定各个实体的提示文件路径
        config = create_graphrag_config({
            "entity_extraction": {"prompt": "tests/unit/config/prompt-a.txt"},
            "claim_extraction": {"prompt": "tests/unit/config/prompt-b.txt"},
            "community_reports": {"prompt": "tests/unit/config/prompt-c.txt"},
            "summarize_descriptions": {"prompt": "tests/unit/config/prompt-d.txt"},
        })
        
        # 对实体提取配置进行测试
        strategy = config.entity_extraction.resolved_strategy(".", "abc123")
        # 断言实体提取策略的提取提示为 "Hello, World! A"
        assert strategy["extraction_prompt"] == "Hello, World! A"
        # 断言编码名称为 "abc123"
        assert strategy["encoding_name"] == "abc123"

        # 对声明提取配置进行测试
        strategy = config.claim_extraction.resolved_strategy(".")
        # 断言声明提取策略的提取提示为 "Hello, World! B"
        assert strategy["extraction_prompt"] == "Hello, World! B"

        # 对社区报告配置进行测试
        strategy = config.community_reports.resolved_strategy(".")
        # 断言社区报告策略的提取提示为 "Hello, World! C"
        assert strategy["extraction_prompt"] == "Hello, World! C"

        # 对描述总结配置进行测试
        strategy = config.summarize_descriptions.resolved_strategy(".")
        # 断言描述总结策略的总结提示为 "Hello, World! D"
        assert strategy["summarize_prompt"] == "Hello, World! D"
@mock.patch.dict(
    os.environ,
    {
        "PIPELINE_LLM_API_KEY": "test",
        "PIPELINE_LLM_API_BASE": "http://test",
        "PIPELINE_LLM_API_VERSION": "v1",
        "PIPELINE_LLM_MODEL": "test-llm",
        "PIPELINE_LLM_DEPLOYMENT_NAME": "test",
    },
    clear=True,
)
# 使用 mock.patch.dict 来模拟修改 os.environ 的环境变量，以便在测试期间使用自定义的配置
def test_yaml_load_e2e():
    # 解析 YAML 字符串，将其转换为 Python 字典格式
    config_dict = yaml.safe_load(
        """
input:
  file_type: text

llm:
  type: azure_openai_chat
  api_key: ${PIPELINE_LLM_API_KEY}
  api_base: ${PIPELINE_LLM_API_BASE}
  api_version: ${PIPELINE_LLM_API_VERSION}
  model: ${PIPELINE_LLM_MODEL}
  deployment_name: ${PIPELINE_LLM_DEPLOYMENT_NAME}
  model_supports_json: True
  tokens_per_minute: 80000
  requests_per_minute: 900
  thread_count: 50
  concurrent_requests: 25
"""
    )
    # 将解析后的配置字典赋值给变量 model
    model = config_dict
    # 使用解析后的配置字典创建默认的图拉格配置参数
    parameters = create_graphrag_config(model, ".")

    # 断言检查默认参数中的值是否与预期的环境变量设置一致
    assert parameters.llm.api_key == "test"
    assert parameters.llm.model == "test-llm"
    assert parameters.llm.api_base == "http://test"
    assert parameters.llm.api_version == "v1"
    assert parameters.llm.deployment_name == "test"

    # 使用默认参数生成管道配置
    pipeline_config = create_pipeline_config(parameters, True)

    # 将生成的配置转换为 JSON 字符串
    config_str = pipeline_config.model_dump_json()
    # 断言检查是否成功将模板中的变量替换为实际的值
    assert "${PIPELINE_LLM_API_KEY}" not in config_str
    assert "${PIPELINE_LLM_API_BASE}" not in config_str
    assert "${PIPELINE_LLM_API_VERSION}" not in config_str
    assert "${PIPELINE_LLM_MODEL}" not in config_str
    assert "${PIPELINE_LLM_DEPLOYMENT_NAME}" not in config_str
```