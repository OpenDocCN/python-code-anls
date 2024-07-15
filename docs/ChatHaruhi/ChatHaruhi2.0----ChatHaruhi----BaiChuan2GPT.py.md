# `.\Chat-Haruhi-Suzumiya\ChatHaruhi2.0\ChatHaruhi\BaiChuan2GPT.py`

```py
# 导入PyTorch库
import torch
# 导入基础长文本生成模型类BaseLLM
from .BaseLLM import BaseLLM
# 导入自动模型和分词器类
from transformers import AutoModelForCausalLM, AutoTokenizer
# 导入生成配置类
from transformers.generation.utils import GenerationConfig
# 导入PeftModel类
from peft import PeftModel

# 初始化白川2LORA模型的全局变量
tokenizer_BaiChuan = None
model_BaiChuan = None

# 初始化白川2LORA模型
def initialize_BaiChuan2LORA():
    global model_BaiChuan, tokenizer_BaiChuan
    
    # 如果模型尚未初始化，则从预训练模型加载白川2-13B-Chat模型
    if model_BaiChuan is None:
        model_BaiChuan = AutoModelForCausalLM.from_pretrained(
            "baichuan-inc/Baichuan2-13B-Chat",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # 使用PeftModel对模型进行后处理
        model_BaiChuan = PeftModel.from_pretrained(
            model_BaiChuan,
            "silk-road/Chat-Haruhi-Fusion_Baichuan2_13B"
        )
        # 加载模型的生成配置
        model_BaiChuan.generation_config = GenerationConfig.from_pretrained(
            "baichuan-inc/Baichuan2-13B-Chat"
        )
    
    # 如果分词器尚未初始化，则从预训练模型加载白川2-13B-Chat分词器
    if tokenizer_BaiChuan is None:
        tokenizer_BaiChuan =  AutoTokenizer.from_pretrained(
            "baichuan-inc/Baichuan2-13B-Chat", 
            use_fast=True, 
            trust_remote_code=True
        )
    
    # 返回初始化后的模型和分词器
    return model_BaiChuan, tokenizer_BaiChuan

# 白川2GPT模型的分词器类
class BaiChuan2GPT(BaseLLM):
    def __init__(self, model = "haruhi-fusion-baichuan"):
        super(BaiChuan2GPT, self).__init__()
        
        # 根据传入的模型参数选择初始化模型
        if model == "baichuan2-13b":
            # 使用预训练模型和分词器初始化模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                "baichuan-inc/Baichuan2-13B-Chat", 
                use_fast=True, 
                trust_remote_code=True
            ),
            self.model = AutoModelForCausalLM.from_pretrained(
                "baichuan-inc/Baichuan2-13B-Chat",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            # 加载模型的生成配置
            self.model.generation_config = GenerationConfig.from_pretrained(
                "baichuan-inc/Baichuan2-13B-Chat"
            )
        elif model == "haruhi-fusion-baichuan":
            # 根据初始化函数选择使用Haruhi融合模型
            self.model, self.tokenizer = initialize_BaiChuan2LORA()
        else:
            # 抛出异常，说明不支持的白川模型类型
            raise Exception("Unknown BaiChuan Model! Currently supported: [BaiChuan2-13B, haruhi-fusion-baichuan]")
        
        # 初始化消息列表
        self.messages = []

    # 初始化消息列表
    def initialize_message(self):
        self.messages = []

    # 添加AI角色的消息到消息列表
    def ai_message(self, payload):
        self.messages.append({"role": "assistant", "content": payload})

    # 添加系统角色的消息到消息列表
    def system_message(self, payload):
        self.messages.append({"role": "system", "content": payload})

    # 添加用户角色的消息到消息列表
    def user_message(self, payload):
        self.messages.append({"role": "user", "content": payload})

    # 获取生成回复
    def get_response(self):
        with torch.no_grad():
            # 调用模型进行对话生成
            response = self.model.chat(self.tokenizer, self.messages)
        return response
        
    # 打印提示信息，输出消息列表类型和内容
    def print_prompt(self):
        print(type(self.messages))
        print(self.messages)
```