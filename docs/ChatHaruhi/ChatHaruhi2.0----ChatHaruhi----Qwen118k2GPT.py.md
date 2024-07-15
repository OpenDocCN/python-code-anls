# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\Qwen118k2GPT.py`

```py
import torch 
from .BaseLLM import BaseLLM  # 导入自定义的BaseLLM类
from transformers import AutoTokenizer, AutoModel  # 导入AutoTokenizer和AutoModel类
from peft import PeftModel  # 导入PeftModel类
from transformers import AutoModelForCausalLM, AutoTokenizer  # 导入AutoModelForCausalLM和AutoTokenizer类
from transformers.generation import GenerationConfig  # 导入GenerationConfig类

tokenizer_qwen = None  # 初始化Qwen模型的tokenizer为None
model_qwen = None  # 初始化Qwen模型为None

def initialize_Qwen2LORA(model):
    global model_qwen, tokenizer_qwen

    if model_qwen is None:
        # 从预训练模型加载Qwen模型，并设置设备映射为自动选择
        model_qwen = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            trust_remote_code=True
        )
        model_qwen = model_qwen.eval()  # 设置模型为评估模式

    if tokenizer_qwen is None:
        # 从预训练模型加载Qwen模型的tokenizer
        tokenizer_qwen = AutoTokenizer.from_pretrained(
            model, 
            trust_remote_code=True
        )

    return model_qwen, tokenizer_qwen

def Qwen_tokenizer(text):
    return len(tokenizer_qwen.encode(text))  # 返回使用Qwen模型的tokenizer编码文本的长度

class Qwen118k2GPT(BaseLLM):
    def __init__(self, model):
        super(Qwen118k2GPT, self).__init__()  # 调用父类BaseLLM的构造函数
        global model_qwen, tokenizer_qwen
        if model == "Qwen/Qwen-1_8B-Chat":
            # 如果模型是Qwen/Qwen-1_8B-Chat，则初始化Qwen模型和tokenizer
            tokenizer_qwen = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-1_8B-Chat", 
                trust_remote_code=True
            )
            model_qwen = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen-1_8B-Chat", 
                device_map="auto", 
                trust_remote_code=True
            ).eval()
            self.model = model_qwen  # 设置模型为Qwen模型
            self.tokenizer = tokenizer_qwen  # 设置tokenizer为Qwen模型的tokenizer
        elif "silk-road/" in model:
            # 如果模型路径包含"silk-road/"，则调用initialize_Qwen2LORA函数初始化Qwen模型和tokenizer
            self.model, self.tokenizer = initialize_Qwen2LORA(model)
        else:
            raise Exception("Unknown Qwen model")  # 抛出异常，未知的Qwen模型

        self.messages = ""  # 初始化消息为空字符串

    def initialize_message(self):
        self.messages = ""  # 将消息重置为空字符串

    def ai_message(self, payload):
        self.messages = "AI: " +  self.messages + "\n " + payload  # 将AI消息添加到消息中

    def system_message(self, payload):
        self.messages = "SYSTEM PROMPT: " + self.messages + "\n " + payload  # 将系统消息添加到消息中

    def user_message(self, payload):
        self.messages = "User: " + self.messages + "\n " + payload  # 将用户消息添加到消息中

    def get_response(self):
        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, self.messages, history=[])  # 使用模型生成响应
        return response  # 返回生成的响应
        
    def print_prompt(self):
        print(type(self.messages))  # 打印消息的类型
        print(self.messages)  # 打印消息内容
```