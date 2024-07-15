# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\LangChainGPT.py`

```py
# 导入所需模块和类
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from .BaseLLM import BaseLLM

# 导入标准库模块
import os
from dotenv import load_dotenv

# 定义 LangChainGPT 类，继承自 BaseLLM 类
class LangChainGPT(BaseLLM):

    # 初始化方法，接受一个可选的 model 参数，默认为 "gpt-3.5-turbo"
    def __init__(self, model="gpt-3.5-turbo"):
        # 调用父类的初始化方法
        super(LangChainGPT, self).__init__()
        # 将 model 参数赋给实例变量 self.model
        self.model = model
        
        # 检查环境变量中是否有 OPENAI_API_BASE
        if "OPENAI_API_BASE" in os.environ:
            # 载入环境变量
            load_dotenv()
            # 从环境变量中获取 OPENAI_API_BASE 和 OPENAI_API_KEY
            api_base = os.environ["OPENAI_API_BASE"]
            api_key = os.environ["OPENAI_API_KEY"]
            # 使用指定的 model 和 api_base 创建 ChatOpenAI 实例
            self.chat = ChatOpenAI(model=self.model, openai_api_base=api_base)
        else:
            # 如果环境变量中没有 OPENAI_API_BASE，则只使用指定的 model 创建 ChatOpenAI 实例
            self.chat = ChatOpenAI(model=self.model)
        
        # 初始化消息列表为空列表
        self.messages = []

    # 初始化消息列表方法，将 self.messages 清空
    def initialize_message(self):
        self.messages = []

    # 添加 AI 消息到消息列表方法，将 payload 作为内容创建 AIMessage 实例并加入 self.messages
    def ai_message(self, payload):
        self.messages.append(AIMessage(content=payload))

    # 添加系统消息到消息列表方法，将 payload 作为内容创建 SystemMessage 实例并加入 self.messages
    def system_message(self, payload):
        self.messages.append(SystemMessage(content=payload))

    # 添加用户消息到消息列表方法，将 payload 作为内容创建 HumanMessage 实例并加入 self.messages
    def user_message(self, payload):
        self.messages.append(HumanMessage(content=payload))

    # 获取聊天模型的响应方法，调用 chat 实例的方法并返回响应的内容
    def get_response(self):
        response = self.chat(self.messages)
        return response.content

    # 打印消息提示方法，遍历 self.messages 列表并打印每个消息
    def print_prompt(self):
        for message in self.messages:
            print(message)
```