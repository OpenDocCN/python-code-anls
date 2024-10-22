# `.\diffusers\pipelines\deprecated\spectrogram_diffusion\continuous_encoder.py`

```py
# 版权信息，声明此文件的版权所有者及相关法律条款
# Copyright 2022 The Music Spectrogram Diffusion Authors.
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 进行授权
# 仅在遵守许可证的情况下使用此文件
# 可以在以下网址获取许可证
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则软件按“原样”分发
# 不提供任何形式的担保或条件
# 请参见许可证以获取特定语言的权限和限制

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的神经网络模块
import torch.nn as nn
# 从 Transformers 库中导入模块工具混合类
from transformers.modeling_utils import ModuleUtilsMixin
# 从 Transformers 的 T5 模型导入相关组件
from transformers.models.t5.modeling_t5 import (
    T5Block,          # T5 模型中的块
    T5Config,         # T5 模型的配置类
    T5LayerNorm,      # T5 模型中的层归一化
)

# 从自定义配置工具导入混合类和注册到配置的装饰器
from ....configuration_utils import ConfigMixin, register_to_config
# 从自定义模型导入模型混合类
from ....models import ModelMixin


# 定义 SpectrogramContEncoder 类，继承自多个混合类
class SpectrogramContEncoder(ModelMixin, ConfigMixin, ModuleUtilsMixin):
    # 注册初始化方法到配置
    @register_to_config
    def __init__(
        self,
        input_dims: int,          # 输入维度
        targets_context_length: int,  # 目标上下文长度
        d_model: int,             # 模型维度
        dropout_rate: float,      # dropout 比率
        num_layers: int,          # 层数
        num_heads: int,           # 注意力头数
        d_kv: int,                # 键值维度
        d_ff: int,                # 前馈层维度
        feed_forward_proj: str,   # 前馈投影类型
        is_decoder: bool = False, # 是否为解码器
    ):
        # 调用父类初始化方法
        super().__init__()

        # 定义输入投影层，将输入维度映射到模型维度
        self.input_proj = nn.Linear(input_dims, d_model, bias=False)

        # 定义位置编码层，使用嵌入来表示位置
        self.position_encoding = nn.Embedding(targets_context_length, d_model)
        # 禁止位置编码权重更新
        self.position_encoding.weight.requires_grad = False

        # 定义前置 dropout 层
        self.dropout_pre = nn.Dropout(p=dropout_rate)

        # 创建 T5 模型的配置
        t5config = T5Config(
            d_model=d_model,            # 模型维度
            num_heads=num_heads,        # 注意力头数
            d_kv=d_kv,                  # 键值维度
            d_ff=d_ff,                  # 前馈层维度
            feed_forward_proj=feed_forward_proj,  # 前馈投影类型
            dropout_rate=dropout_rate,  # dropout 比率
            is_decoder=is_decoder,      # 是否为解码器
            is_encoder_decoder=False,   # 是否为编码器解码器
        )
        # 创建编码器层的模块列表
        self.encoders = nn.ModuleList()
        # 根据层数循环创建 T5 块并添加到编码器列表中
        for lyr_num in range(num_layers):
            lyr = T5Block(t5config)  # 创建 T5 块
            self.encoders.append(lyr)  # 添加到编码器列表

        # 定义层归一化层
        self.layer_norm = T5LayerNorm(d_model)
        # 定义后置 dropout 层
        self.dropout_post = nn.Dropout(p=dropout_rate)
    # 定义前向传播函数，接收编码器输入和编码器输入的掩码
    def forward(self, encoder_inputs, encoder_inputs_mask):
        # 通过输入投影将编码器输入转换为模型内部表示
        x = self.input_proj(encoder_inputs)
    
        # 计算终端相对位置编码
        max_positions = encoder_inputs.shape[1]  # 获取输入序列的最大长度
        input_positions = torch.arange(max_positions, device=encoder_inputs.device)  # 生成位置索引的张量
    
        # 计算输入序列的长度
        seq_lens = encoder_inputs_mask.sum(-1)  # 对掩码进行求和，得到每个序列的有效长度
        # 将位置索引张量根据序列长度进行滚动
        input_positions = torch.roll(input_positions.unsqueeze(0), tuple(seq_lens.tolist()), dims=0)
        # 将位置编码添加到输入特征中
        x += self.position_encoding(input_positions)
    
        # 对输入特征应用 dropout 以防止过拟合
        x = self.dropout_pre(x)
    
        # 反转注意力掩码
        input_shape = encoder_inputs.size()  # 获取输入张量的形状
        # 扩展注意力掩码以适应输入形状
        extended_attention_mask = self.get_extended_attention_mask(encoder_inputs_mask, input_shape)
    
        # 依次通过每个编码层
        for lyr in self.encoders:
            x = lyr(x, extended_attention_mask)[0]  # 通过编码器层处理输入并获取输出
        # 对输出应用层归一化
        x = self.layer_norm(x)
    
        # 返回处理后的输出和编码器输入掩码
        return self.dropout_post(x), encoder_inputs_mask
```