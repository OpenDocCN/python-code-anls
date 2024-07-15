# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\ChatGLM2GPT.py`

```py
import torch
from .BaseLLM import BaseLLM  # 导入自定义的BaseLLM类
from transformers import AutoTokenizer, AutoModel  # 导入transformers库中的AutoTokenizer和AutoModel类
from peft import PeftModel  # 导入peft库中的PeftModel类

tokenizer_GLM = None  # 初始化全局变量tokenizer_GLM为None
model_GLM = None  # 初始化全局变量model_GLM为None

def initialize_GLM2LORA():
    global model_GLM, tokenizer_GLM  # 声明引用全局变量model_GLM和tokenizer_GLM

    if model_GLM is None:  # 如果model_GLM为None，则加载模型
        model_GLM = AutoModel.from_pretrained(
            "THUDM/chatglm2-6b",  # 使用预训练模型"THUDM/chatglm2-6b"
            torch_dtype=torch.float16,  # 设置模型的torch数据类型为float16
            device_map="auto",  # 自动选择设备
            trust_remote_code=True  # 信任远程代码
        )
        model_GLM = PeftModel.from_pretrained(
            model_GLM,  # 使用从AutoModel加载的模型作为PeftModel的参数
            "silk-road/Chat-Haruhi-Fusion_B"  # 加载预训练的PeftModel
        )

    if tokenizer_GLM is None:  # 如果tokenizer_GLM为None，则加载tokenizer
        tokenizer_GLM = AutoTokenizer.from_pretrained(
            "THUDM/chatglm2-6b",  # 使用预训练模型"THUDM/chatglm2-6b"
            use_fast=True,  # 使用快速tokenizer
            trust_remote_code=True  # 信任远程代码
        )

    return model_GLM, tokenizer_GLM  # 返回加载的模型和tokenizer实例

def GLM_tokenizer(text):
    return len(tokenizer_GLM.encode(text))  # 使用tokenizer_GLM对文本进行编码，并返回编码后的长度

class ChatGLM2GPT(BaseLLM):
    def __init__(self, model="haruhi-fusion"):
        super(ChatGLM2GPT, self).__init__()  # 调用父类BaseLLM的构造函数

        if model == "glm2-6b":  # 如果model为"glm2-6b"，加载ChatGLM2GPT的tokenizer和model
            self.tokenizer = AutoTokenizer.from_pretrained(
                "THUDM/chatglm2-6b",  # 使用预训练模型"THUDM/chatglm2-6b"
                use_fast=True,  # 使用快速tokenizer
                trust_remote_code=True  # 信任远程代码
            )
            self.model = AutoModel.from_pretrained(
                "THUDM/chatglm2-6b",  # 使用预训练模型"THUDM/chatglm2-6b"
                torch_dtype=torch.float16,  # 设置模型的torch数据类型为float16
                device_map="auto",  # 自动选择设备
                trust_remote_code=True  # 信任远程代码
            )
        elif model == "haruhi-fusion":  # 如果model为"haruhi-fusion"，调用initialize_GLM2LORA加载模型和tokenizer
            self.model, self.tokenizer = initialize_GLM2LORA()
        else:
            raise Exception("Unknown GLM model")  # 如果model不在已知的模型列表中，抛出异常"Unknown GLM model"

        self.messages = ""  # 初始化消息字符串为空

    def initialize_message(self):
        self.messages = ""  # 将消息字符串初始化为空字符串

    def ai_message(self, payload):
        self.messages = self.messages + "\n " + payload  # 向消息字符串添加AI生成的消息

    def system_message(self, payload):
        self.messages = self.messages + "\n " + payload  # 向消息字符串添加系统生成的消息

    def user_message(self, payload):
        self.messages = self.messages + "\n " + payload  # 向消息字符串添加用户输入的消息

    def get_response(self):
        with torch.no_grad():  # 禁用梯度计算
            response, history = self.model.chat(self.tokenizer, self.messages, history=[])  # 使用模型生成响应消息
            # print(response)
        return response  # 返回生成的响应消息

    def print_prompt(self):
        print(type(self.messages))  # 打印消息字符串的类型
        print(self.messages)  # 打印消息字符串本身
```