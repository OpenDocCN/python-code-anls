# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\PrintLLM.py`

```py
# 从BaseLLM导入基础的语言模型类
from .BaseLLM import BaseLLM

# 定义一个名为PrintLLM的类，继承自BaseLLM类
class PrintLLM(BaseLLM):

    # 初始化方法，初始化一个空的消息列表
    def __init__(self):
        self.messages = []
        # 向消息列表添加调试用的初始消息
        self.messages.append("Noticing: This is a print LLM for debug.")
        self.messages.append("But you can also copy the prompt into GPT or Claude to debugging")

    # 初始化消息方法，重置消息列表为空并添加初始调试消息
    def initialize_message(self):
        self.messages = []
        self.messages.append("Noticing: This is a print LLM for debug.")
        self.messages.append("But you can also copy the prompt into GPT or Claude to debugging")

    # AI消息方法，接收一个有效负载，并将带有AI前缀的消息添加到消息列表中
    def ai_message(self, payload):
        self.messages.append("AI: \n" + payload)

    # 系统消息方法，接收一个有效负载，并将带有System前缀的消息添加到消息列表中
    def system_message(self, payload):
        self.messages.append("System: \n" + payload)

    # 用户消息方法，接收一个有效负载，并将带有User前缀的消息添加到消息列表中
    def user_message(self, payload):
        self.messages.append("User: \n" + payload)

    # 获取响应方法，遍历消息列表并打印每条消息，然后接收用户输入作为响应并返回
    def get_response(self):
        for message in self.messages:
            print(message)
        response = input("Please input your response: ")
        return response
    
    # 打印提示方法，遍历消息列表并打印每条消息
    def print_prompt(self):
        for message in self.messages:
            print(message)
```