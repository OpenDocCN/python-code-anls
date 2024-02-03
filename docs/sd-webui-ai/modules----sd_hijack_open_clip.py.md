# `stable-diffusion-webui\modules\sd_hijack_open_clip.py`

```py
# 导入 open_clip.tokenizer 模块
import open_clip.tokenizer
# 导入 torch 模块
import torch

# 从 modules 模块中导入 sd_hijack_clip 和 devices
from modules import sd_hijack_clip, devices
# 从 modules.shared 模块中导入 opts
from modules.shared import opts

# 获取 open_clip.tokenizer 模块中的 _tokenizer 对象
tokenizer = open_clip.tokenizer._tokenizer

# 定义一个类 FrozenOpenCLIPEmbedderWithCustomWords，继承自 sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase 类
class FrozenOpenCLIPEmbedderWithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase):
    # 初始化方法
    def __init__(self, wrapped, hijack):
        # 调用父类的初始化方法
        super().__init__(wrapped, hijack)

        # 获取逗号标记的编码
        self.comma_token = [v for k, v in tokenizer.encoder.items() if k == ',</w>'][0]
        # 获取起始标记的编码
        self.id_start = tokenizer.encoder["<start_of_text>"]
        # 获取结束标记的编码
        self.id_end = tokenizer.encoder["<end_of_text>"]
        # 定义填充标记的编码为 0
        self.id_pad = 0

    # 对文本进行分词处理
    def tokenize(self, texts):
        # 断言不使用旧的强调实现，因为 Open Clip 不支持
        assert not opts.use_old_emphasis_implementation, 'Old emphasis implementation not supported for Open Clip'

        # 对文本进行编码处理
        tokenized = [tokenizer.encode(text) for text in texts]

        return tokenized

    # 使用 transformers 对 tokens 进行编码
    def encode_with_transformers(self, tokens):
        # 根据 opts.CLIP_stop_at_last_layers 设置 self.wrapped.layer_idx
        z = self.wrapped.encode_with_transformer(tokens)

        return z

    # 对初始文本进行编码处理
    def encode_embedding_init_text(self, init_text, nvpt):
        # 对初始文本进行编码
        ids = tokenizer.encode(init_text)
        # 将编码转换为张量
        ids = torch.asarray([ids], device=devices.device, dtype=torch.int)
        # 获取嵌入向量
        embedded = self.wrapped.model.token_embedding.wrapped(ids).squeeze(0)

        return embedded

# 定义一个类 FrozenOpenCLIPEmbedder2WithCustomWords，继承自 sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase 类
class FrozenOpenCLIPEmbedder2WithCustomWords(sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase):
    # 初始化方法
    def __init__(self, wrapped, hijack):
        # 调用父类的初始化方法
        super().__init__(wrapped, hijack)

        # 获取逗号标记的编码
        self.comma_token = [v for k, v in tokenizer.encoder.items() if k == ',</w>'][0]
        # 获取起始标记的编码
        self.id_start = tokenizer.encoder["<start_of_text>"]
        # 获取结束标记的编码
        self.id_end = tokenizer.encoder["<end_of_text>"]
        # 定义填充标记的编码为 0
        self.id_pad = 0

    # 对文本进行分词处理
    def tokenize(self, texts):
        # 断言不使用旧的强调实现，因为 Open Clip 不支持
        assert not opts.use_old_emphasis_implementation, 'Old emphasis implementation not supported for Open Clip'

        # 对文本进行编码处理
        tokenized = [tokenizer.encode(text) for text in texts]

        return tokenized
    # 使用transformers对tokens进行编码
    def encode_with_transformers(self, tokens):
        # 调用wrapped对象的encode_with_transformer方法对tokens进行编码
        d = self.wrapped.encode_with_transformer(tokens)
        # 获取wrapped对象的指定层的输出
        z = d[self.wrapped.layer]

        # 获取d中的"pooled"字段
        pooled = d.get("pooled")
        # 如果pooled不为空，则将其赋值给z的pooled属性
        if pooled is not None:
            z.pooled = pooled

        # 返回z
        return z

    # 对初始文本进行编码初始化
    def encode_embedding_init_text(self, init_text, nvpt):
        # 使用tokenizer对init_text进行编码
        ids = tokenizer.encode(init_text)
        # 将ids转换为torch张量，指定设备和数据类型
        ids = torch.asarray([ids], device=devices.device, dtype=torch.int)
        # 使用wrapped对象的model的token_embedding对ids进行嵌入
        embedded = self.wrapped.model.token_embedding.wrapped(ids.to(self.wrapped.model.token_embedding.wrapped.weight.device)).squeeze(0)

        # 返回embedded
        return embedded
```