# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\BaseLLM.py`

```py
# 引入抽象基类（ABC）模块，用于定义抽象基类
from abc import ABC, abstractmethod

# 定义一个抽象基类 BaseLLM，继承自 ABC 抽象基类
class BaseLLM(ABC):

    # 初始化方法，无具体实现
    def __init__(self):
        pass
    
    # 抽象方法：初始化消息，具体实现由子类提供
    @abstractmethod
    def initialize_message(self):
        pass

    # 抽象方法：处理 AI 消息，具体实现由子类提供
    @abstractmethod    
    def ai_message(self, payload):
        pass

    # 抽象方法：处理系统消息，具体实现由子类提供
    @abstractmethod
    def system_message(self, payload):
        pass

    # 抽象方法：处理用户消息，具体实现由子类提供
    @abstractmethod
    def user_message(self, payload):
        pass

    # 抽象方法：获取响应消息，具体实现由子类提供
    @abstractmethod
    def get_response(self):
        pass

    # 抽象方法：打印提示信息，具体实现由子类提供
    @abstractmethod
    def print_prompt(self):
        pass
```