# `.\flux\src\flux\modules\conditioner.py`

```py
# 从 PyTorch 和 Transformers 库导入必要的模块
from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)

# 定义一个用于获取文本嵌入的类 HFEmbedder，继承自 nn.Module
class HFEmbedder(nn.Module):
    # 初始化方法
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        # 调用父类的初始化方法
        super().__init__()
        # 判断是否使用 CLIP 模型，根据版本名进行判断
        self.is_clip = version.startswith("openai")
        # 设置最大长度
        self.max_length = max_length
        # 根据是否使用 CLIP 模型选择输出的键
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        # 如果使用 CLIP 模型
        if self.is_clip:
            # 从预训练模型加载 tokenizer
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            # 从预训练模型加载 HF 模块
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
        else:
            # 如果使用 T5 模型
            # 从预训练模型加载 tokenizer
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            # 从预训练模型加载 HF 模块
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)

        # 将模型设置为评估模式，并且不计算梯度
        self.hf_module = self.hf_module.eval().requires_grad_(False)

    # 前向传播方法，处理输入文本并返回嵌入
    def forward(self, text: list[str]) -> Tensor:
        # 使用 tokenizer 对文本进行编码
        batch_encoding = self.tokenizer(
            text,
            truncation=True,  # 对超长文本进行截断
            max_length=self.max_length,  # 设置最大长度
            return_length=False,  # 不返回文本长度
            return_overflowing_tokens=False,  # 不返回溢出的标记
            padding="max_length",  # 填充到最大长度
            return_tensors="pt",  # 返回 PyTorch 张量
        )

        # 使用 HF 模块进行前向传播计算
        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),  # 将输入 ID 移动到模型所在设备
            attention_mask=None,  # 不使用注意力掩码
            output_hidden_states=False,  # 不返回隐藏状态
        )
        # 返回指定键对应的输出
        return outputs[self.output_key]
```