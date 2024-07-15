# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\ErnieGPT.py`

```py
# ErnieGPT.py
# 从pyexpat模块中导入model
from pyexpat import model
# 导入erniebot模块
import erniebot 
# 导入操作系统环境变量模块
import os
# 导入copy模块
import copy

# 设置erniebot的API类型为环境变量中的APIType
erniebot.api_type = os.environ["APIType"]
# 设置erniebot的访问令牌为环境变量中的ErnieAccess
erniebot.access_token = os.environ["ErnieAccess"]

# 从.BaseLLM模块中导入BaseLLM类
from .BaseLLM import BaseLLM

# 定义ErnieGPT类，继承自BaseLLM类
class ErnieGPT(BaseLLM):

    def __init__(self,model="ernie-bot", ernie_trick = True ):
        # 调用父类BaseLLM的初始化方法
        super(ErnieGPT,self).__init__()
        # 设置模型名称
        self.model = model
        # 如果模型名称不在指定的列表中，抛出异常
        if model not in ["ernie-bot", "ernie-bot-turbo", "ernie-vilg-v2", "ernie-text-embedding", "ernie-bot-8k", "ernie-bot-4"]:
            raise Exception("Unknown Ernie model")
        # 初始化消息列表
        self.messages = []

        # 设置是否启用Ernie的技巧
        self.ernie_trick = ernie_trick
        
    # 初始化消息列表
    def initialize_message(self):
        self.messages = []

    # 添加AI消息到消息列表
    def ai_message(self, payload):
        # 如果消息列表为空，添加用户消息提示
        if len(self.messages) == 0:
            self.user_message("请根据我的要求进行角色扮演:")
        # 如果消息列表长度为奇数，添加助手角色消息
        elif len(self.messages) % 2 == 1:
            self.messages.append({"role":"assistant","content":payload})
        # 如果消息列表长度为偶数，添加用户角色消息，并与上一条消息连接
        elif len(self.messages)% 2 == 0:
            self.messages[-1]["content"] += "\n"+ payload

    # 添加系统消息到消息列表
    def system_message(self, payload):
        self.messages.append({"role":"user","content":payload}) 

    # 添加用户消息到消息列表
    def user_message(self, payload):
        # 如果消息列表长度为偶数，添加用户角色消息
        if len(self.messages) % 2 == 0:
            self.messages.append({"role":"user","content":payload})
        # 如果消息列表长度为奇数，将消息添加到上一条用户角色消息后
        elif len(self.messages)% 2 == 1:
            self.messages[-1]["content"] += "\n"+ payload

    # 获取Ernie的回复
    def get_response(self):
        # 深度复制消息列表
        chat_messages = copy.deepcopy(self.messages)
        # 将最后一条消息内容按换行符分割成行
        lines = chat_messages[-1]["content"].split('\n')

        # 如果启用Ernie的技巧，将提示语句插入倒数第二行
        if self.ernie_trick:
            lines.insert(-1, '请请模仿上述经典桥段进行回复\n')
        
        # 将修改后的内容连接为字符串，作为最后一条消息的内容
        chat_messages[-1]["content"] = '\n'.join(lines)

        # 使用erniebot的ChatCompletion接口生成回复
        response = erniebot.ChatCompletion.create(model=self.model, messages=chat_messages)
        # 返回生成的回复结果
        return response["result"]
    
    # 打印消息提示
    def print_prompt(self):
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")
```