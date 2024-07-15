# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\SparkGPT.py`

```py
# SparkGPT.py
# 导入SparkApi模块
from . import SparkApi
# 导入os模块，用于获取环境变量信息
import os

# 从环境变量中获取应用程序ID、API密钥和API密钥
appid = os.environ['APPID']
api_secret = os.environ['APISecret'] 
api_key = os.environ['APIKey']

# 导入BaseLLM类
from .BaseLLM import BaseLLM

# SparkGPT类继承自BaseLLM类
class SparkGPT(BaseLLM):

    # 初始化方法，设置模型类型和消息列表
    def __init__(self, model="Spark3.0"):
        super(SparkGPT,self).__init__()
        self.model_type = model
        self.messages = []
        # 根据模型类型设置不同的域和Spark API地址
        if self.model_type == "Spark2.0":
            self.domain = "generalv2"    # v2.0版本
            self.Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
        elif self.model_type == "Spark1.5":
            self.domain = "general"   # v1.5版本
            self.Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
        elif self.model_type == "Spark3.0":
            self.domain = "generalv3"   # v3.0版本
            self.Spark_url = "ws://spark-api.xf-yun.com/v3.1/chat"  # v3.0环境的地址
        else:
            # 如果模型类型未知，则抛出异常
            raise Exception("Unknown Spark model")
    
    # 初始化消息列表为空列表
    def initialize_message(self):
        self.messages = []

    # 处理AI的消息，根据消息列表中的角色决定如何处理payload
    def ai_message(self, payload):
        if len(self.messages) == 0:
            self.user_message("请根据我的要求进行角色扮演:")
        elif len(self.messages) % 2 == 1:
            self.messages.append({"role":"assistant","content":payload})
        elif len(self.messages)% 2 == 0:
            self.messages[-1]["content"] += "\n"+ payload

    # 添加系统消息到消息列表中
    def system_message(self, payload):
        self.messages.append({"role":"user","content":payload}) 

    # 添加用户消息到消息列表中
    def user_message(self, payload):
        if len(self.messages) % 2 == 0:
            self.messages.append({"role":"user","content":payload})
        elif len(self.messages)% 2 == 1:
            self.messages[-1]["content"] += "\n"+ payload

    # 获取响应，根据模型类型调用SparkApi的main方法获取答案
    def get_response(self):
        SparkApi.answer = ""
        # 根据模型类型设置不同的域和Spark API地址
        if self.model_type == "Spark2.0":
            self.domain = "generalv2"    # v2.0版本
            self.Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat"  # v2.0环境的地址
        elif self.model_type == "Spark1.5":
            self.domain = "general"   # v1.5版本
            self.Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat"  # v1.5环境的地址
        elif self.model_type == "Spark3.0":
            self.domain = "generalv3"   # v3.0版本
            self.Spark_url = "ws://spark-api.xf-yun.com/v3.1/chat"  # v3.0环境的地址
        else:
            # 如果模型类型未知，则抛出异常
            raise Exception("Unknown Spark model")
        # 调用SparkApi的main方法，并传递必要的参数
        SparkApi.main(appid, api_key, api_secret, self.Spark_url, self.domain, self.messages)
        return SparkApi.answer
    
    # 打印消息列表中的角色和内容
    def print_prompt(self):
        for message in self.messages:
            print(f"{message['role']}: {message['content']}")
```