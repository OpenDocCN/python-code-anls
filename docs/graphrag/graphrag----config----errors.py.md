# `.\graphrag\graphrag\config\errors.py`

```py
# 定义一个自定义异常类 ApiKeyMissingError，继承自 ValueError
class ApiKeyMissingError(ValueError):
    """LLM Key missing error."""

    # 初始化方法，带有一个布尔型参数 embedding，默认为 False
    def __init__(self, embedding: bool = False) -> None:
        """Init method definition."""
        # 根据 embedding 的布尔值确定 API 的类型
        api_type = "Embedding" if embedding else "Completion"
        # 根据 embedding 的布尔值确定 API 的键名
        api_key = "GRAPHRAG_EMBEDDING_API_KEY" if embedding else "GRAPHRAG_LLM_API_KEY"
        # 构造异常消息
        msg = f"API Key is required for {api_type} API. Please set either the OPENAI_API_KEY, GRAPHRAG_API_KEY or {api_key} environment variable."
        # 调用父类的初始化方法，传入异常消息
        super().__init__(msg)


# 定义一个自定义异常类 AzureApiBaseMissingError，继承自 ValueError
class AzureApiBaseMissingError(ValueError):
    """Azure API Base missing error."""

    # 初始化方法，带有一个布尔型参数 embedding，默认为 False
    def __init__(self, embedding: bool = False) -> None:
        """Init method definition."""
        # 根据 embedding 的布尔值确定 API 的类型
        api_type = "Embedding" if embedding else "Completion"
        # 根据 embedding 的布尔值确定 API 的基础地址的键名
        api_base = "GRAPHRAG_EMBEDDING_API_BASE" if embedding else "GRAPHRAG_API_BASE"
        # 构造异常消息
        msg = f"API Base is required for {api_type} API. Please set either the OPENAI_API_BASE, GRAPHRAG_API_BASE or {api_base} environment variable."
        # 调用父类的初始化方法，传入异常消息
        super().__init__(msg)


# 定义一个自定义异常类 AzureDeploymentNameMissingError，继承自 ValueError
class AzureDeploymentNameMissingError(ValueError):
    """Azure Deployment Name missing error."""

    # 初始化方法，带有一个布尔型参数 embedding，默认为 False
    def __init__(self, embedding: bool = False) -> None:
        """Init method definition."""
        # 根据 embedding 的布尔值确定 API 的类型
        api_type = "Embedding" if embedding else "Completion"
        # 根据 embedding 的布尔值确定 API 的部署名称的键名
        api_base = (
            "GRAPHRAG_EMBEDDING_DEPLOYMENT_NAME"
            if embedding
            else "GRAPHRAG_LLM_DEPLOYMENT_NAME"
        )
        # 构造异常消息
        msg = f"Deployment Name is required for {api_type} API. Please set either the OPENAI_DEPLOYMENT_NAME, GRAPHRAG_LLM_DEPLOYMENT_NAME or {api_base} environment variable."
        # 调用父类的初始化方法，传入异常消息
        super().__init__(msg)
```