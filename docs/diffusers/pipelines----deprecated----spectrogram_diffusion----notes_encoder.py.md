# `.\diffusers\pipelines\deprecated\spectrogram_diffusion\notes_encoder.py`

```py
# 版权所有 2022 The Music Spectrogram Diffusion Authors.
# 版权所有 2024 The HuggingFace Team。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件是以“原样”基础分发的，
# 不提供任何形式的担保或条件，无论是明示还是暗示。
# 有关许可证的特定语言，见于许可证的许可和限制。

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 transformers 库导入模块工具类
from transformers.modeling_utils import ModuleUtilsMixin
# 从 transformers 库导入 T5 模型相关类
from transformers.models.t5.modeling_t5 import T5Block, T5Config, T5LayerNorm

# 从父级目录导入配置混合类和注册配置的装饰器
from ....configuration_utils import ConfigMixin, register_to_config
# 从父级目录导入模型混合类
from ....models import ModelMixin

# 定义 SpectrogramNotesEncoder 类，继承多个混合类
class SpectrogramNotesEncoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    # 注册构造函数到配置
    @register_to_config
    def __init__(
        self,
        max_length: int,      # 输入序列的最大长度
        vocab_size: int,      # 词汇表的大小
        d_model: int,         # 模型的维度
        dropout_rate: float,  # dropout 比率
        num_layers: int,      # 编码器的层数
        num_heads: int,       # 多头注意力机制的头数
        d_kv: int,            # 键和值的维度
        d_ff: int,            # 前馈网络的维度
        feed_forward_proj: str, # 前馈网络的投影类型
        is_decoder: bool = False, # 是否为解码器
    ):
        # 调用父类构造函数
        super().__init__()

        # 创建词嵌入层
        self.token_embedder = nn.Embedding(vocab_size, d_model)

        # 创建位置编码层
        self.position_encoding = nn.Embedding(max_length, d_model)
        # 设置位置编码的权重不需要梯度
        self.position_encoding.weight.requires_grad = False

        # 创建 dropout 层以在前向传播前进行正则化
        self.dropout_pre = nn.Dropout(p=dropout_rate)

        # 配置 T5 模型的参数
        t5config = T5Config(
            vocab_size=vocab_size,    # 词汇表大小
            d_model=d_model,           # 模型维度
            num_heads=num_heads,       # 注意力头数
            d_kv=d_kv,                 # 键值维度
            d_ff=d_ff,                 # 前馈网络维度
            dropout_rate=dropout_rate, # dropout 比率
            feed_forward_proj=feed_forward_proj, # 前馈网络类型
            is_decoder=is_decoder,     # 是否为解码器
            is_encoder_decoder=False,   # 是否为编码器-解码器架构
        )

        # 创建模块列表以存储编码器层
        self.encoders = nn.ModuleList()
        # 循环添加每个编码器层
        for lyr_num in range(num_layers):
            lyr = T5Block(t5config) # 创建 T5Block 实例
            self.encoders.append(lyr) # 将编码器层添加到列表

        # 创建层归一化层
        self.layer_norm = T5LayerNorm(d_model)
        # 创建 dropout 层以在输出后进行正则化
        self.dropout_post = nn.Dropout(p=dropout_rate)

    # 定义前向传播函数
    def forward(self, encoder_input_tokens, encoder_inputs_mask):
        # 通过词嵌入层获取嵌入表示
        x = self.token_embedder(encoder_input_tokens)

        # 获取输入序列的长度
        seq_length = encoder_input_tokens.shape[1]
        # 创建输入位置的张量
        inputs_positions = torch.arange(seq_length, device=encoder_input_tokens.device)
        # 将位置编码添加到嵌入表示
        x += self.position_encoding(inputs_positions)

        # 应用前向 dropout
        x = self.dropout_pre(x)

        # 反转注意力掩码
        input_shape = encoder_input_tokens.size()
        # 获取扩展的注意力掩码
        extended_attention_mask = self.get_extended_attention_mask(encoder_inputs_mask, input_shape)

        # 遍历每个编码器层，执行前向传播
        for lyr in self.encoders:
            x = lyr(x, extended_attention_mask)[0]
        # 应用层归一化
        x = self.layer_norm(x)

        # 返回经过后向 dropout 的输出和输入掩码
        return self.dropout_post(x), encoder_inputs_mask
```