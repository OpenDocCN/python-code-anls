# `.\agent\GenAINewsAgent\server\llms\base.py`

```
from abc import ABC, abstractmethod
from typing import List, Dict, Union

# 定义一个抽象基类 BaseLLM，继承自 ABC（Abstract Base Class）
class BaseLLM(ABC):

    # 初始化方法，接受一个可选的 api_key 参数和任意其他关键字参数
    def __init__(self, api_key: Union[str, None] = None, **kwargs):
        self.api_key = api_key  # 将传入的 api_key 参数赋给实例变量 self.api_key
        self.client = None       # 初始化实例变量 self.client 为 None
        self.extra_args = kwargs  # 将传入的其他关键字参数保存在实例变量 self.extra_args 中

    # 抽象方法 __call__，需要在子类中实现，异步方法（async），接受 model 字符串和消息列表 messages，以及其他关键字参数
    @abstractmethod
    async def __call__(self, model: str, messages: List[Dict], **kwargs):
        pass  # 占位符，子类必须实现该方法，根据 model 和 messages 参数执行相应操作
```