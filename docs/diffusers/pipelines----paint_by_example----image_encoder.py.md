# `.\diffusers\pipelines\paint_by_example\image_encoder.py`

```py
# 版权声明，说明代码的版权所有者及使用许可
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 该文件根据 Apache License, Version 2.0（"许可证"）授权； 
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件在“按现状”基础上分发，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参见许可证以了解有关权限和限制的具体语言。
import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块
from transformers import CLIPPreTrainedModel, CLIPVisionModel  # 导入 CLIP 相关模型

from ...models.attention import BasicTransformerBlock  # 导入基本变换器块
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁用 Pylint 对无效名称的警告


class PaintByExampleImageEncoder(CLIPPreTrainedModel):  # 定义图像编码器类，继承自预训练的 CLIP 模型
    def __init__(self, config, proj_size=None):  # 初始化方法，接收配置和可选的投影大小
        super().__init__(config)  # 调用父类初始化方法
        self.proj_size = proj_size or getattr(config, "projection_dim", 768)  # 设置投影大小，默认为 768

        self.model = CLIPVisionModel(config)  # 创建 CLIP 视觉模型实例
        self.mapper = PaintByExampleMapper(config)  # 创建映射器实例
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)  # 创建层归一化实例
        self.proj_out = nn.Linear(config.hidden_size, self.proj_size)  # 创建线性变换，用于输出投影

        # 用于缩放的无条件向量
        self.uncond_vector = nn.Parameter(torch.randn((1, 1, self.proj_size)))  # 初始化无条件向量为随机值

    def forward(self, pixel_values, return_uncond_vector=False):  # 前向传播方法，接收像素值和是否返回无条件向量的标志
        clip_output = self.model(pixel_values=pixel_values)  # 通过模型处理像素值
        latent_states = clip_output.pooler_output  # 获取池化输出作为潜在状态
        latent_states = self.mapper(latent_states[:, None])  # 使用映射器处理潜在状态
        latent_states = self.final_layer_norm(latent_states)  # 对潜在状态进行层归一化
        latent_states = self.proj_out(latent_states)  # 通过线性层进行投影
        if return_uncond_vector:  # 如果需要返回无条件向量
            return latent_states, self.uncond_vector  # 返回潜在状态和无条件向量

        return latent_states  # 否则仅返回潜在状态


class PaintByExampleMapper(nn.Module):  # 定义映射器类，继承自 PyTorch 的 nn.Module
    def __init__(self, config):  # 初始化方法，接收配置
        super().__init__()  # 调用父类初始化方法
        num_layers = (config.num_hidden_layers + 1) // 5  # 计算层数，确保至少为 1
        hid_size = config.hidden_size  # 获取隐藏层大小
        num_heads = 1  # 设置注意力头的数量为 1
        self.blocks = nn.ModuleList(  # 创建模块列表，包含多个变换器块
            [
                BasicTransformerBlock(hid_size, num_heads, hid_size, activation_fn="gelu", attention_bias=True)  # 添加变换器块
                for _ in range(num_layers)  # 根据层数创建多个块
            ]
        )

    def forward(self, hidden_states):  # 前向传播方法，接收隐藏状态
        for block in self.blocks:  # 遍历所有变换器块
            hidden_states = block(hidden_states)  # 依次通过每个块处理隐藏状态

        return hidden_states  # 返回最终的隐藏状态
```