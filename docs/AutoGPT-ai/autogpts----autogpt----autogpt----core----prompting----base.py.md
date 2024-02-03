# `.\AutoGPT\autogpts\autogpt\autogpt\core\prompting\base.py`

```py
# 导入 abc 模块，用于定义抽象基类
import abc

# 从 autogpt.core.configuration 模块中导入 SystemConfiguration 类
from autogpt.core.configuration import SystemConfiguration
# 从 autogpt.core.resource.model_providers 模块中导入 AssistantChatMessage 类
from autogpt.core.resource.model_providers import AssistantChatMessage

# 从当前目录下的 schema 模块中导入 ChatPrompt 和 LanguageModelClassification 类
from .schema import ChatPrompt, LanguageModelClassification

# 定义一个抽象基类 PromptStrategy，继承自 abc.ABC
class PromptStrategy(abc.ABC):
    # 定义类属性 default_configuration，类型为 SystemConfiguration
    default_configuration: SystemConfiguration

    # 定义抽象属性 model_classification，返回类型为 LanguageModelClassification
    @property
    @abc.abstractmethod
    def model_classification(self) -> LanguageModelClassification:
        ...

    # 定义抽象方法 build_prompt，接受任意位置参数和关键字参数，返回类型为 ChatPrompt
    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs) -> ChatPrompt:
        ...

    # 定义抽象方法 parse_response_content，接受参数 response_content，类型为 AssistantChatMessage
    @abc.abstractmethod
    def parse_response_content(self, response_content: AssistantChatMessage):
        ...
```