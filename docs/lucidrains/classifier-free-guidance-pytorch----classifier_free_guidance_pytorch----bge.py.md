# `.\lucidrains\classifier-free-guidance-pytorch\classifier_free_guidance_pytorch\bge.py`

```
# 导入所需的模块和函数
from typing import List
from beartype import beartype

import torch
import transformers 
from transformers import AutoTokenizer, AutoModel, AutoConfig
transformers.logging.set_verbosity_error()

# 创建 BGEAdapter 类
class BGEAdapter():
    def __init__(
        self,
        name
    ):
        # 设置模型名称
        name = 'BAAI/bge-base-en-v1.5'
        # 根据模型名称加载对应的 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(name)
        # 根据模型名称加载对应的 model
        model = AutoModel.from_pretrained(name)
        # 根据模型名称加载对应的配置
        self.Config = AutoConfig.from_pretrained(name)
        
        # 如果有可用的 CUDA 设备，则将模型移动到 CUDA 上
        if torch.cuda.is_available():
            model = model.to("cuda")  
            
        # 设置对象的名称、模型和 tokenizer
        self.name =  name
        self.model = model
        self.tokenizer = tokenizer

    # 定义 dim_latent 属性，返回隐藏层的大小
    @property
    def dim_latent(self):
        return self.Config.hidden_size

    # 定义 max_text_len 属性，返回文本的最大长度
    @property
    def max_text_len(self):
        return 512

    # 定义 embed_text 方法，用于文本嵌入
    @torch.no_grad()
    @beartype
    def embed_text(
        self,
        texts: List[str],
        return_text_encodings = False,
        output_device = None
    ):
        # 使用 tokenizer 对文本进行编码
        encoded_input  = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to("cuda")
 
        # 将模型设置为评估模式
        self.model.eval()
         
        # 使用模型对编码后的输入进行推理
        with torch.no_grad():
            model_output = self.model(**encoded_input)  
            
        # 如果不需要返回文本编码，则返回规范化后的 CLS 嵌入
        if not return_text_encodings: 
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings  # 返回规范化后的 CLS 嵌入

        # 如果需要返回文本编码，则返回最后一个隐藏状态，并根据输出设备进行转换
        return model_output.last_hidden_state.to(output_device)
```