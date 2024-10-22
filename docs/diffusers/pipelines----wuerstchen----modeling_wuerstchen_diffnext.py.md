# `.\diffusers\pipelines\wuerstchen\modeling_wuerstchen_diffnext.py`

```py
# 版权信息，标明该代码的版权所有者及许可证
# Copyright (c) 2023 Dominic Rampas MIT License
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 在 Apache 许可证 2.0（"许可证"）下获得许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件按"原样"提供，
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# 请参见许可证以获取特定于许可证的权限和限制。

# 导入数学库
import math

# 导入 NumPy 库以进行数组处理
import numpy as np
# 导入 PyTorch 库及其神经网络模块
import torch
import torch.nn as nn

# 从配置工具模块导入 ConfigMixin 和注册配置的方法
from ...configuration_utils import ConfigMixin, register_to_config
# 从模型工具模块导入 ModelMixin
from ...models.modeling_utils import ModelMixin
# 从本地模块导入模型组件
from .modeling_wuerstchen_common import AttnBlock, GlobalResponseNorm, TimestepBlock, WuerstchenLayerNorm

# 定义 WuerstchenDiffNeXt 类，继承自 ModelMixin 和 ConfigMixin
class WuerstchenDiffNeXt(ModelMixin, ConfigMixin):
    # 注册初始化方法到配置
    @register_to_config
    def __init__(
        self,
        c_in=4,  # 输入通道数，默认为 4
        c_out=4,  # 输出通道数，默认为 4
        c_r=64,  # 嵌入维度，默认为 64
        patch_size=2,  # 补丁大小，默认为 2
        c_cond=1024,  # 条件通道数，默认为 1024
        c_hidden=[320, 640, 1280, 1280],  # 隐藏层通道数配置
        nhead=[-1, 10, 20, 20],  # 注意力头数配置
        blocks=[4, 4, 14, 4],  # 各级块数配置
        level_config=["CT", "CTA", "CTA", "CTA"],  # 各级配置
        inject_effnet=[False, True, True, True],  # 是否注入 EfficientNet
        effnet_embd=16,  # EfficientNet 嵌入维度
        clip_embd=1024,  # CLIP 嵌入维度
        kernel_size=3,  # 卷积核大小
        dropout=0.1,  # dropout 比率
    ):
        # 初始化权重的方法
        def _init_weights(self, m):
            # 对卷积层和线性层进行通用初始化
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)  # 使用 Xavier 均匀分布初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏置初始化为 0

            # 对 EfficientNet 映射器进行初始化
            for mapper in self.effnet_mappers:
                if mapper is not None:
                    nn.init.normal_(mapper.weight, std=0.02)  # 条件初始化为正态分布
            nn.init.normal_(self.clip_mapper.weight, std=0.02)  # CLIP 映射器初始化
            nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # 输入嵌入初始化
            nn.init.constant_(self.clf[1].weight, 0)  # 输出分类器初始化为 0

            # 初始化块中的权重
            for level_block in self.down_blocks + self.up_blocks:
                for block in level_block:
                    if isinstance(block, ResBlockStageB):
                        block.channelwise[-1].weight.data *= np.sqrt(1 / sum(self.config.blocks))  # 权重缩放
                    elif isinstance(block, TimestepBlock):
                        nn.init.constant_(block.mapper.weight, 0)  # 将时间步映射器的权重初始化为 0

    # 生成位置嵌入的方法
    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions  # 将位置 r 乘以最大位置
        half_dim = self.c_r // 2  # 计算半维度
        emb = math.log(max_positions) / (half_dim - 1)  # 计算嵌入尺度
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()  # 生成嵌入
        emb = r[:, None] * emb[None, :]  # 扩展 r 的维度并进行乘法
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # 计算正弦和余弦嵌入并拼接
        if self.c_r % 2 == 1:  # 如果 c_r 为奇数，则进行零填充
            emb = nn.functional.pad(emb, (0, 1), mode="constant")  # 用常数进行填充
        return emb.to(dtype=r.dtype)  # 返回与 r 数据类型相同的嵌入
    # 生成 CLIP 嵌入
        def gen_c_embeddings(self, clip):
            # 将输入 clip 通过映射转换
            clip = self.clip_mapper(clip)
            # 对 clip 进行序列归一化处理
            clip = self.seq_norm(clip)
            # 返回处理后的 clip
            return clip
    
        # 下采样编码过程
        def _down_encode(self, x, r_embed, effnet, clip=None):
            # 初始化层级输出列表
            level_outputs = []
            # 遍历每一个下采样块
            for i, down_block in enumerate(self.down_blocks):
                effnet_c = None  # 初始化有效网络通道为 None
                # 遍历每个下采样块中的组件
                for block in down_block:
                    # 如果是残差块阶段 B
                    if isinstance(block, ResBlockStageB):
                        # 检查有效网络通道是否为 None
                        if effnet_c is None and self.effnet_mappers[i] is not None:
                            dtype = effnet.dtype  # 获取 effnet 的数据类型
                            # 进行双线性插值并创建有效网络通道
                            effnet_c = self.effnet_mappers[i](
                                nn.functional.interpolate(
                                    effnet.float(), size=x.shape[-2:], mode="bicubic", antialias=True, align_corners=True
                                ).to(dtype)
                            )
                        # 设置跳跃连接为有效网络通道
                        skip = effnet_c if self.effnet_mappers[i] is not None else None
                        # 通过当前块处理输入 x 和跳跃连接
                        x = block(x, skip)
                    # 如果是注意力块
                    elif isinstance(block, AttnBlock):
                        # 通过当前块处理输入 x 和 clip
                        x = block(x, clip)
                    # 如果是时间步块
                    elif isinstance(block, TimestepBlock):
                        # 通过当前块处理输入 x 和 r_embed
                        x = block(x, r_embed)
                    else:
                        # 通过当前块处理输入 x
                        x = block(x)
                # 将当前层输出插入到层级输出列表的开头
                level_outputs.insert(0, x)
            # 返回所有层级输出
            return level_outputs
    
        # 上采样解码过程
        def _up_decode(self, level_outputs, r_embed, effnet, clip=None):
            # 使用层级输出的第一个元素初始化 x
            x = level_outputs[0]
            # 遍历每一个上采样块
            for i, up_block in enumerate(self.up_blocks):
                effnet_c = None  # 初始化有效网络通道为 None
                # 遍历每个上采样块中的组件
                for j, block in enumerate(up_block):
                    # 如果是残差块阶段 B
                    if isinstance(block, ResBlockStageB):
                        # 检查有效网络通道是否为 None
                        if effnet_c is None and self.effnet_mappers[len(self.down_blocks) + i] is not None:
                            dtype = effnet.dtype  # 获取 effnet 的数据类型
                            # 进行双线性插值并创建有效网络通道
                            effnet_c = self.effnet_mappers[len(self.down_blocks) + i](
                                nn.functional.interpolate(
                                    effnet.float(), size=x.shape[-2:], mode="bicubic", antialias=True, align_corners=True
                                ).to(dtype)
                            )
                        # 设置跳跃连接为当前层级输出的第 i 个元素
                        skip = level_outputs[i] if j == 0 and i > 0 else None
                        # 如果有效网络通道不为 None
                        if effnet_c is not None:
                            # 如果跳跃连接不为 None，将其与有效网络通道拼接
                            if skip is not None:
                                skip = torch.cat([skip, effnet_c], dim=1)
                            else:
                                # 否则直接设置为有效网络通道
                                skip = effnet_c
                        # 通过当前块处理输入 x 和跳跃连接
                        x = block(x, skip)
                    # 如果是注意力块
                    elif isinstance(block, AttnBlock):
                        # 通过当前块处理输入 x 和 clip
                        x = block(x, clip)
                    # 如果是时间步块
                    elif isinstance(block, TimestepBlock):
                        # 通过当前块处理输入 x 和 r_embed
                        x = block(x, r_embed)
                    else:
                        # 通过当前块处理输入 x
                        x = block(x)
            # 返回最终处理后的 x
            return x
    # 定义前向传播函数，接受多个输入参数
        def forward(self, x, r, effnet, clip=None, x_cat=None, eps=1e-3, return_noise=True):
            # 如果 x_cat 不为 None，将 x 和 x_cat 沿着维度 1 拼接
            if x_cat is not None:
                x = torch.cat([x, x_cat], dim=1)
            # 处理条件嵌入
            r_embed = self.gen_r_embedding(r)
            # 如果 clip 不为 None，生成条件嵌入
            if clip is not None:
                clip = self.gen_c_embeddings(clip)
    
            # 模型块
            x_in = x  # 保存输入 x 以备后用
            x = self.embedding(x)  # 将输入 x 转换为嵌入表示
            # 下采样编码
            level_outputs = self._down_encode(x, r_embed, effnet, clip)
            # 上采样解码
            x = self._up_decode(level_outputs, r_embed, effnet, clip)
            # 将输出分成两个部分 a 和 b
            a, b = self.clf(x).chunk(2, dim=1)
            # 对 b 进行 sigmoid 激活，并进行缩放
            b = b.sigmoid() * (1 - eps * 2) + eps
            # 如果返回噪声，计算并返回
            if return_noise:
                return (x_in - a) / b
            else:
                return a, b  # 否则返回 a 和 b
# 定义一个残差块阶段 B，继承自 nn.Module
class ResBlockStageB(nn.Module):
    # 初始化函数，设置输入通道、跳跃连接通道、卷积核大小和丢弃率
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        # 调用父类的初始化方法
        super().__init__()
        # 创建深度卷积层，使用指定的卷积核大小和填充
        self.depthwise = nn.Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        # 创建层归一化层，设置元素可学习性为 False 和小的 epsilon 值
        self.norm = WuerstchenLayerNorm(c, elementwise_affine=False, eps=1e-6)
        # 创建一个顺序容器，包含线性层、GELU 激活、全局响应归一化、丢弃层和另一线性层
        self.channelwise = nn.Sequential(
            nn.Linear(c + c_skip, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            nn.Linear(c * 4, c),
        )

    # 定义前向传播函数
    def forward(self, x, x_skip=None):
        # 保存输入以进行残差连接
        x_res = x
        # 先进行深度卷积和层归一化
        x = self.norm(self.depthwise(x))
        # 如果有跳跃连接，则将其与当前输出连接
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        # 变换输入维度并通过通道层，最后恢复维度
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # 返回残差输出
        return x + x_res
```