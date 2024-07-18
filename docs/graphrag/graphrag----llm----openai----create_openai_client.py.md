# `.\graphrag\graphrag\llm\openai\create_openai_client.py`

```py
# 创建 OpenAI 客户端实例的函数，根据配置和是否为 Azure 客户端选择不同的实例化方式
@cache
def create_openai_client(
    configuration: OpenAIConfiguration, azure: bool
) -> OpenAIClientTypes:
    """Create a new OpenAI client instance."""
    # 如果是 Azure 客户端
    if azure:
        # 获取 API base 地址
        api_base = configuration.api_base
        # 如果 API base 地址为 None，则抛出数值错误异常
        if api_base is None:
            raise ValueError(API_BASE_REQUIRED_FOR_AZURE)

        # 记录信息：创建 Azure OpenAI 客户端，包括 API base 地址和部署名称
        log.info(
            "Creating Azure OpenAI client api_base=%s, deployment_name=%s",
            api_base,
            configuration.deployment_name,
        )

        # 设置认知服务终结点，默认为 Azure Cognitive Services 终结点
        if configuration.cognitive_services_endpoint is None:
            cognitive_services_endpoint = "https://cognitiveservices.azure.com/.default"
        else:
            cognitive_services_endpoint = configuration.cognitive_services_endpoint

        # 返回 AsyncAzureOpenAI 实例
        return AsyncAzureOpenAI(
            api_key=configuration.api_key if configuration.api_key else None,
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(), cognitive_services_endpoint
            )
            if not configuration.api_key
            else None,
            organization=configuration.organization,
            # Azure 特定配置
            api_version=configuration.api_version,
            azure_endpoint=api_base,
            azure_deployment=configuration.deployment_name,
            # 超时/重试配置 - 使用 Tenacity 进行重试，这里禁用重试
            timeout=configuration.request_timeout or 180.0,
            max_retries=0,
        )

    # 如果不是 Azure 客户端，记录信息：创建 OpenAI 客户端，包括基础 URL
    log.info("Creating OpenAI client base_url=%s", configuration.api_base)
    
    # 返回 AsyncOpenAI 实例
    return AsyncOpenAI(
        api_key=configuration.api_key,
        base_url=configuration.api_base,
        organization=configuration.organization,
        # 超时/重试配置 - 使用 Tenacity 进行重试，这里禁用重试
        timeout=configuration.request_timeout or 180.0,
        max_retries=0,
    )
```