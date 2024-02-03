# `stable-diffusion-webui\modules\sd_hijack_xlmr.py`

```py
# 导入 torch 库
import torch

# 从 modules 模块中导入 sd_hijack_clip 和 devices
from modules import sd_hijack_clip, devices

# 定义一个继承自 FrozenCLIPEmbedderWithCustomWords 的类
class FrozenXLMREmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords):
    # 初始化方法
    def __init__(self, wrapped, hijack):
        # 调用父类的初始化方法
        super().__init__(wrapped, hijack)

        # 设置起始、结束和填充 token 的 id
        self.id_start = wrapped.config.bos_token_id
        self.id_end = wrapped.config.eos_token_id
        self.id_pad = wrapped.config.pad_token_id

        # 获取逗号 token 的 id，如果不存在则为 None
        self.comma_token = self.tokenizer.get_vocab().get(',', None)  # alt diffusion doesn't have </w> bits for comma

    # 使用 transformers 对 tokens 进行编码
    def encode_with_transformers(self, tokens):
        # 生成 attention mask，标记不是填充 token 的位置
        attention_mask = (tokens != self.id_pad).to(device=tokens.device, dtype=torch.int64)
        # 使用 wrapped 模型对 tokens 进行编码
        features = self.wrapped(input_ids=tokens, attention_mask=attention_mask)
        # 获取 projection_state 特征
        z = features['projection_state']

        return z

    # 对初始文本进行编码初始化
    def encode_embedding_init_text(self, init_text, nvpt):
        # 获取嵌入层
        embedding_layer = self.wrapped.roberta.embeddings
        # 使用 tokenizer 对初始文本进行编码
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        # 获取嵌入后的文本
        embedded = embedding_layer.token_embedding.wrapped(ids.to(devices.device)).squeeze(0)

        return embedded
```