# `.\lucidrains\perfusion-pytorch\perfusion_pytorch\open_clip.py`

```py
# 导入必要的库
from beartype import beartype
from beartype.typing import List, Optional

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

import open_clip

# 定义一个函数，用于检查值是否存在
def exists(val):
    return val is not None

# 定义一个函数，用于对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1)

# 定义一个类，用于适配 OpenCLIP 模型
class OpenClipAdapter(nn.Module):
    @beartype
    def __init__(
        self,
        name = 'ViT-B/32',
        pretrained = 'laion400m_e32',
        tokenizer_name = 'ViT-B-32-quickgelu',
        eos_id = 49407
    ):
        super().__init__()

        # 创建 OpenCLIP 模型、预处理函数和 tokenizer
        clip, _, preprocess = open_clip.create_model_and_transforms(name, pretrained = pretrained)
        tokenizer = open_clip.get_tokenizer(tokenizer_name)

        self.clip = clip
        self.tokenizer = tokenizer
        self.eos_id = eos_id

        # 用于获取最终文本表示的钩子

        text_attention_final = self.find_layer('ln_final')
        self._dim_latent = text_attention_final.weight.shape[0]
        self.text_handle = text_attention_final.register_forward_hook(self._text_hook)

        # 标准化函数

        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    @property
    def device(self):
        return next(self.parameters()).device

    # 查找指定层
    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    # 清除钩子
    def clear(self):
        if self.cleared:
            return

        self.text_handle()

    # 文本钩子函数
    def _text_hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return self._dim_latent

    @property
    def max_text_len(self):
        return self.clip.positional_embedding.shape[0]

    @beartype
    def embed_texts(
        self,
        texts: List[str]
    ):
        # 对文本进行编码
        ids = self.tokenizer(texts)
        ids = ids.to(self.device)
        ids = ids[..., :self.max_text_len]

        is_eos_id = (ids == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        text_mask = text_mask & (ids != 0)
        assert not self.cleared

        # 编码文本并进行掩码
        text_embed = self.clip.encode_text(ids)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        return text_encodings.float(), text_mask
```