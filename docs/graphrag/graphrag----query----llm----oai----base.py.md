# `.\graphrag\graphrag\query\llm\oai\base.py`

```py
# 版权声明和许可信息
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLM 和 Embedding 模型的基础类。"""

# 导入必要的库和模块
from abc import ABC, abstractmethod
from collections.abc import Callable

# 导入 OpenAI 相关模块
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

# 导入本地模块
from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.progress import ConsoleStatusReporter, StatusReporter

# 定义 BaseOpenAILLM 抽象基类
class BaseOpenAILLM(ABC):
    """Base OpenAI LLM implementation."""

    # 异步客户端和同步客户端属性定义
    _async_client: AsyncOpenAI | AsyncAzureOpenAI
    _sync_client: OpenAI | AzureOpenAI

    def __init__(self):
        # 调用 _create_openai_client 方法来初始化客户端
        self._create_openai_client()

    @abstractmethod
    def _create_openai_client(self):
        """Create a new synchronous and asynchronous OpenAI client instance."""
    
    def set_clients(
        self,
        sync_client: OpenAI | AzureOpenAI,
        async_client: AsyncOpenAI | AsyncAzureOpenAI,
    ):
        """
        Set the synchronous and asynchronous clients used for making API requests.

        Args:
            sync_client (OpenAI | AzureOpenAI): The sync client object.
            async_client (AsyncOpenAI | AsyncAzureOpenAI): The async client object.
        """
        # 设置同步和异步客户端对象
        self._sync_client = sync_client
        self._async_client = async_client

    @property
    def async_client(self) -> AsyncOpenAI | AsyncAzureOpenAI | None:
        """
        Get the asynchronous client used for making API requests.

        Returns
        -------
            AsyncOpenAI | AsyncAzureOpenAI: The async client object.
        """
        # 返回异步客户端对象
        return self._async_client

    @property
    def sync_client(self) -> OpenAI | AzureOpenAI | None:
        """
        Get the synchronous client used for making API requests.

        Returns
        -------
            OpenAI | AzureOpenAI: The sync client object.
        """
        # 返回同步客户端对象
        return self._sync_client

    @async_client.setter
    def async_client(self, client: AsyncOpenAI | AsyncAzureOpenAI):
        """
        Set the asynchronous client used for making API requests.

        Args:
            client (AsyncOpenAI | AsyncAzureOpenAI): The async client object.
        """
        # 设置异步客户端对象
        self._async_client = client

    @sync_client.setter
    def sync_client(self, client: OpenAI | AzureOpenAI):
        """
        Set the synchronous client used for making API requests.

        Args:
            client (OpenAI | AzureOpenAI): The sync client object.
        """
        # 设置同步客户端对象
        self._sync_client = client


# 定义 OpenAILLMImpl 类，继承自 BaseOpenAILLM 类
class OpenAILLMImpl(BaseOpenAILLM):
    """Orchestration OpenAI LLM Implementation."""

    # 状态报告器属性，默认为 ConsoleStatusReporter 对象
    _reporter: StatusReporter = ConsoleStatusReporter()
    def __init__(
        self,
        api_key: str | None = None,
        azure_ad_token_provider: Callable | None = None,
        deployment_name: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: OpenaiApiType = OpenaiApiType.OpenAI,
        organization: str | None = None,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        reporter: StatusReporter | None = None,
    ):
        # 初始化函数，用于实例化 OpenAI 客户端对象
        self.api_key = api_key  # 设置 API 密钥
        self.azure_ad_token_provider = azure_ad_token_provider  # Azure AD 访问令牌提供器
        self.deployment_name = deployment_name  # 部署名称
        self.api_base = api_base  # API 基础地址
        self.api_version = api_version  # API 版本
        self.api_type = api_type  # API 类型，默认为 OpenAI
        self.organization = organization  # 组织名称
        self.max_retries = max_retries  # 最大重试次数，默认为 10
        self.request_timeout = request_timeout  # 请求超时时间，默认为 180 秒
        self.reporter = reporter or ConsoleStatusReporter()  # 状态报告器，如果未提供则使用默认的控制台状态报告器

        try:
            # 创建 OpenAI 同步和异步客户端
            super().__init__()  # 调用父类的初始化方法，通常用于设置父类的一些属性或调用其初始化逻辑
        except Exception as e:
            self._reporter.error(
                message="Failed to create OpenAI client",  # 错误消息：无法创建 OpenAI 客户端
                details={self.__class__.__name__: str(e)},  # 错误详细信息，包含异常类名和具体异常信息
            )
            raise  # 抛出异常，向上层调用者传递异常信息
    def _create_openai_client(self):
        """Create a new OpenAI client instance."""
        # 检查所选择的 API 类型是否为 Azure OpenAI
        if self.api_type == OpenaiApiType.AzureOpenAI:
            # 如果是 Azure OpenAI，确保 api_base 参数不为空
            if self.api_base is None:
                msg = "api_base is required for Azure OpenAI"
                raise ValueError(msg)

            # 创建同步 Azure OpenAI 客户端实例
            sync_client = AzureOpenAI(
                api_key=self.api_key,
                azure_ad_token_provider=self.azure_ad_token_provider,
                organization=self.organization,
                # Azure 特定参数
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                azure_deployment=self.deployment_name,
                # 重试配置
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            # 创建异步 Azure OpenAI 客户端实例
            async_client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_ad_token_provider=self.azure_ad_token_provider,
                organization=self.organization,
                # Azure 特定参数
                api_version=self.api_version,
                azure_endpoint=self.api_base,
                azure_deployment=self.deployment_name,
                # 重试配置
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
            # 设置同步和异步客户端实例
            self.set_clients(sync_client=sync_client, async_client=async_client)

        else:
            # 如果不是 Azure OpenAI，创建 OpenAI 客户端实例
            sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                organization=self.organization,
                # 重试配置
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            # 创建异步 OpenAI 客户端实例
            async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                organization=self.organization,
                # 重试配置
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
            # 设置同步和异步客户端实例
            self.set_clients(sync_client=sync_client, async_client=async_client)
class OpenAITextEmbeddingImpl(BaseTextEmbedding):
    """Orchestration OpenAI Text Embedding Implementation."""

    _reporter: StatusReporter | None = None
    # _reporter 是一个类成员变量，用于存储 StatusReporter 类型的对象或者 None

    def _create_openai_client(self, api_type: OpenaiApiType):
        """Create a new synchronous and asynchronous OpenAI client instance."""
        # 创建一个新的同步和异步的 OpenAI 客户端实例的方法
```