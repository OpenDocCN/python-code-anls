# `.\AutoGPT\autogpts\autogpt\autogpt\core\ability\builtins\query_language_model.py`

```py
# 导入 logging 模块
import logging
# 导入 ClassVar 类型提示
from typing import ClassVar

# 导入 Ability、AbilityConfiguration 类
from autogpt.core.ability.base import Ability, AbilityConfiguration
# 导入 AbilityResult 类
from autogpt.core.ability.schema import AbilityResult
# 导入 LanguageModelConfiguration 类
from autogpt.core.planning.simple import LanguageModelConfiguration
# 导入 PluginLocation、PluginStorageFormat 类
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
# 导入 ModelProviderName、OpenAIModelName 类
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    ModelProviderName,
    OpenAIModelName,
)
# 导入 JSONSchema 类
from autogpt.core.utils.json_schema import JSONSchema

# 定义 QueryLanguageModel 类，继承自 Ability 类
class QueryLanguageModel(Ability):
    # 默认配置
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.QueryLanguageModel",
        ),
        language_model_required=LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT3,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.9,
        ),
    )

    # 初始化方法
    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
        language_model_provider: ChatModelProvider,
    ):
        # 设置日志记录器
        self._logger = logger
        # 设置配置
        self._configuration = configuration
        # 设置语言模型提供者
        self._language_model_provider = language_model_provider

    # 描述信息
    description: ClassVar[str] = (
        "Query a language model."
        " A query should be a question and any relevant context."
    )

    # 参数定义
    parameters: ClassVar[dict[str, JSONSchema]] = {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description=(
                "A query for a language model. "
                "A query should contain a question and any relevant context."
            ),
        )
    }
    # 异步函数，接受一个字符串类型的查询，返回一个 AbilityResult 对象
    async def __call__(self, query: str) -> AbilityResult:
        # 调用语言模型提供者的方法，创建聊天完成的模型响应
        model_response = await self._language_model_provider.create_chat_completion(
            model_prompt=[ChatMessage.user(query)],  # 使用用户输入的查询作为模型的输入
            functions=[],  # 不包含任何函数
            model_name=self._configuration.language_model_required.model_name,  # 使用指定的语言模型名称
        )
        # 返回一个 AbilityResult 对象，包含能力名称、参数、成功标志和模型响应的内容
        return AbilityResult(
            ability_name=self.name(),  # 设置能力名称
            ability_args={"query": query},  # 设置能力参数
            success=True,  # 设置成功标志为 True
            message=model_response.response.content or "",  # 设置消息为模型响应的内容，如果为空则设置为空字符串
        )
```