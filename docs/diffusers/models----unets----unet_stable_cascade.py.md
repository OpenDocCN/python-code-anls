# `.\diffusers\models\unets\unet_stable_cascade.py`

```py
# 版权声明，指明该文件的版权归 HuggingFace 团队所有
# 
# 根据 Apache 2.0 许可协议进行许可；
# 除非符合许可，否则您不能使用此文件。
# 您可以在以下网址获得许可的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，
# 根据该许可分发的软件是按“原样”基础提供的，
# 不提供任何明示或暗示的保证或条件。
# 请参阅许可以获取特定语言的权限和
# 限制条款。

# 导入数学模块，用于数学计算
import math
# 从数据类模块导入数据类装饰器，用于简化类的定义
from dataclasses import dataclass
# 导入可选类型、元组和联合类型的类型注解
from typing import Optional, Tuple, Union

# 导入 NumPy 库，用于数组和矩阵操作
import numpy as np
# 导入 PyTorch 库及其子模块，用于构建和训练神经网络
import torch
import torch.nn as nn

# 从配置工具导入配置混合类和注册配置的函数
from ...configuration_utils import ConfigMixin, register_to_config
# 从加载器模块导入原始模型混合类
from ...loaders import FromOriginalModelMixin
# 从实用工具导入基础输出类
from ...utils import BaseOutput
# 从注意力处理器模块导入注意力类
from ..attention_processor import Attention
# 从建模工具模块导入模型混合类
from ..modeling_utils import ModelMixin

# 定义一个层归一化类，继承自 nn.LayerNorm
# 从 diffusers.pipelines.wuerstchen.modeling_wuerstchen_common 中复制，并重命名为 SDCascadeLayerNorm
class SDCascadeLayerNorm(nn.LayerNorm):
    # 初始化方法，接受可变参数并调用父类构造函数
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 重新排列 x 的维度，将其形状变为 (batch_size, height, width, channels)
        x = x.permute(0, 2, 3, 1)
        # 调用父类的前向传播方法进行层归一化
        x = super().forward(x)
        # 再次排列 x 的维度，返回到 (batch_size, channels, height, width) 形状
        return x.permute(0, 3, 1, 2)

# 定义时间步块类，继承自 nn.Module
class SDCascadeTimestepBlock(nn.Module):
    # 初始化方法，接受参数 c, c_timestep 和条件列表 conds
    def __init__(self, c, c_timestep, conds=[]):
        super().__init__()

        # 创建一个线性映射层，将时间步的输入转换为两倍的通道数
        self.mapper = nn.Linear(c_timestep, c * 2)
        # 保存条件列表
        self.conds = conds
        # 为每个条件创建一个线性映射层
        for cname in conds:
            setattr(self, f"mapper_{cname}", nn.Linear(c_timestep, c * 2))

    # 前向传播方法，接受输入 x 和时间步 t
    def forward(self, x, t):
        # 将时间步 t 拆分为多个部分
        t = t.chunk(len(self.conds) + 1, dim=1)
        # 使用 mapper 对第一个时间步进行线性映射，并拆分为 a 和 b
        a, b = self.mapper(t[0])[:, :, None, None].chunk(2, dim=1)
        # 遍历条件列表
        for i, c in enumerate(self.conds):
            # 获取条件的映射结果，并拆分为 ac 和 bc
            ac, bc = getattr(self, f"mapper_{c}")(t[i + 1])[:, :, None, None].chunk(2, dim=1)
            # 将映射结果加到 a 和 b 上
            a, b = a + ac, b + bc
        # 返回经过变换后的 x
        return x * (1 + a) + b

# 定义残差块类，继承自 nn.Module
class SDCascadeResBlock(nn.Module):
    # 初始化方法，接受多个参数定义残差块的结构
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        super().__init__()
        # 创建深度可分离卷积层
        self.depthwise = nn.Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        # 创建自定义的层归一化
        self.norm = SDCascadeLayerNorm(c, elementwise_affine=False, eps=1e-6)
        # 创建一个包含多个层的顺序模块
        self.channelwise = nn.Sequential(
            nn.Linear(c + c_skip, c * 4),  # 线性变换
            nn.GELU(),                      # 激活函数
            GlobalResponseNorm(c * 4),      # 全局响应归一化
            nn.Dropout(dropout),            # Dropout 层
            nn.Linear(c * 4, c),            # 输出层
        )

    # 前向传播方法，接受输入 x 和可选的跳跃连接 x_skip
    def forward(self, x, x_skip=None):
        # 保存输入 x 的副本，用于残差连接
        x_res = x
        # 经过深度卷积和归一化
        x = self.norm(self.depthwise(x))
        # 如果提供了跳跃连接，则将其与 x 拼接
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        # 对 x 进行通道变换并返回到原始形状
        x = self.channelwise(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # 返回残差连接的结果
        return x + x_res

# 定义全局响应归一化类，继承自 nn.Module
# 从 Facebook Research 的 ConvNeXt-V2 项目中获取代码
class GlobalResponseNorm(nn.Module):
    # 初始化方法，接收一个维度参数
        def __init__(self, dim):
            # 调用父类的初始化方法
            super().__init__()
            # 创建一个可学习的参数 gamma，形状为 (1, 1, 1, dim)，初始化为零
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            # 创建一个可学习的参数 beta，形状为 (1, 1, 1, dim)，初始化为零
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    # 前向传播方法，定义如何计算输出
        def forward(self, x):
            # 计算输入 x 的 L2 范数，维度为 (1, 2)，保持维度不变
            agg_norm = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            # 将范数归一化，通过均值进行标准化，防止除以零
            stand_div_norm = agg_norm / (agg_norm.mean(dim=-1, keepdim=True) + 1e-6)
            # 返回标准化后的 x 乘以 gamma，加上 beta 和原始 x，形成最终输出
            return self.gamma * (x * stand_div_norm) + self.beta + x
# 定义一个名为 SDCascadeAttnBlock 的类，继承自 nn.Module
class SDCascadeAttnBlock(nn.Module):
    # 初始化函数，接收多个参数设置
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        # 调用父类的初始化函数
        super().__init__()

        # 设置自注意力标志
        self.self_attn = self_attn
        # 创建归一化层，使用 SDCascadeLayerNorm
        self.norm = SDCascadeLayerNorm(c, elementwise_affine=False, eps=1e-6)
        # 创建注意力机制实例
        self.attention = Attention(query_dim=c, heads=nhead, dim_head=c // nhead, dropout=dropout, bias=True)
        # 创建键值映射层，由 SiLU 激活和线性层组成
        self.kv_mapper = nn.Sequential(nn.SiLU(), nn.Linear(c_cond, c))

    # 前向传播函数，接收输入和键值对
    def forward(self, x, kv):
        # 使用键值映射层处理 kv
        kv = self.kv_mapper(kv)
        # 对输入 x 进行归一化处理
        norm_x = self.norm(x)
        # 如果启用自注意力机制
        if self.self_attn:
            # 获取输入的批大小和通道数
            batch_size, channel, _, _ = x.shape
            # 将归一化后的输入和 kv 连接
            kv = torch.cat([norm_x.view(batch_size, channel, -1).transpose(1, 2), kv], dim=1)
        # 将注意力输出与原输入相加
        x = x + self.attention(norm_x, encoder_hidden_states=kv)
        # 返回处理后的输入
        return x


# 定义一个名为 UpDownBlock2d 的类，继承自 nn.Module
class UpDownBlock2d(nn.Module):
    # 初始化函数，接收输入和输出通道数、模式和启用标志
    def __init__(self, in_channels, out_channels, mode, enabled=True):
        # 调用父类的初始化函数
        super().__init__()
        # 如果模式不支持，抛出异常
        if mode not in ["up", "down"]:
            raise ValueError(f"{mode} not supported")
        # 根据模式创建上采样或下采样的插值层
        interpolation = (
            nn.Upsample(scale_factor=2 if mode == "up" else 0.5, mode="bilinear", align_corners=True)
            if enabled
            else nn.Identity()
        )
        # 创建卷积映射层
        mapping = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 根据模式将插值层和卷积层组合成模块列表
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == "up" else [mapping, interpolation])

    # 前向传播函数，接收输入 x
    def forward(self, x):
        # 遍历块并依次处理输入
        for block in self.blocks:
            x = block(x)
        # 返回处理后的输入
        return x


# 定义一个数据类 StableCascadeUNetOutput，继承自 BaseOutput
@dataclass
class StableCascadeUNetOutput(BaseOutput):
    # 初始化输出样本，默认值为 None
    sample: torch.Tensor = None


# 定义一个名为 StableCascadeUNet 的类，继承多个混入类
class StableCascadeUNet(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    # 设置支持梯度检查点标志
    _supports_gradient_checkpointing = True

    # 注册配置装饰器
    @register_to_config
    # 初始化方法，用于设置模型的参数
        def __init__(
            # 输入通道数，默认为16
            in_channels: int = 16,
            # 输出通道数，默认为16
            out_channels: int = 16,
            # 时间步比率嵌入维度，默认为64
            timestep_ratio_embedding_dim: int = 64,
            # 每个补丁的大小，默认为1
            patch_size: int = 1,
            # 条件维度，默认为2048
            conditioning_dim: int = 2048,
            # 每个块的输出通道数，默认为(2048, 2048)
            block_out_channels: Tuple[int] = (2048, 2048),
            # 每层的注意力头数，默认为(32, 32)
            num_attention_heads: Tuple[int] = (32, 32),
            # 每个块的下采样层数，默认为(8, 24)
            down_num_layers_per_block: Tuple[int] = (8, 24),
            # 每个块的上采样层数，默认为(24, 8)
            up_num_layers_per_block: Tuple[int] = (24, 8),
            # 下采样块的重复映射器，默认为(1, 1)
            down_blocks_repeat_mappers: Optional[Tuple[int]] = (
                1,
                1,
            ),
            # 上采样块的重复映射器，默认为(1, 1)
            up_blocks_repeat_mappers: Optional[Tuple[int]] = (1, 1),
            # 每层的块类型，默认为两个层的不同块类型
            block_types_per_layer: Tuple[Tuple[str]] = (
                ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
                ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
            ),
            # 文本输入通道数，可选
            clip_text_in_channels: Optional[int] = None,
            # 文本池化输入通道数，默认为1280
            clip_text_pooled_in_channels=1280,
            # 图像输入通道数，可选
            clip_image_in_channels: Optional[int] = None,
            # 序列长度，默认为4
            clip_seq=4,
            # EfficientNet输入通道数，可选
            effnet_in_channels: Optional[int] = None,
            # 像素映射器输入通道数，可选
            pixel_mapper_in_channels: Optional[int] = None,
            # 卷积核大小，默认为3
            kernel_size=3,
            # dropout率，默认为(0.1, 0.1)
            dropout: Union[float, Tuple[float]] = (0.1, 0.1),
            # 自注意力标志，默认为True
            self_attn: Union[bool, Tuple[bool]] = True,
            # 时间步条件类型，默认为("sca", "crp")
            timestep_conditioning_type: Tuple[str] = ("sca", "crp"),
            # 切换级别，可选
            switch_level: Optional[Tuple[bool]] = None,
        # 设置梯度检查点的方法，默认为False
        def _set_gradient_checkpointing(self, value=False):
            # 存储梯度检查点的布尔值
            self.gradient_checkpointing = value
    
        # 初始化权重的方法
        def _init_weights(self, m):
            # 如果是卷积层或线性层
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # 使用Xavier均匀分布初始化权重
                torch.nn.init.xavier_uniform_(m.weight)
                # 如果有偏置，则将偏置初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
            # 对文本池化映射器的权重进行正态分布初始化，标准差为0.02
            nn.init.normal_(self.clip_txt_pooled_mapper.weight, std=0.02)
            # 如果有文本映射器，则对其权重进行正态分布初始化
            nn.init.normal_(self.clip_txt_mapper.weight, std=0.02) if hasattr(self, "clip_txt_mapper") else None
            # 如果有图像映射器，则对其权重进行正态分布初始化
            nn.init.normal_(self.clip_img_mapper.weight, std=0.02) if hasattr(self, "clip_img_mapper") else None
    
            # 如果有EfficientNet映射器，则对其权重进行初始化
            if hasattr(self, "effnet_mapper"):
                nn.init.normal_(self.effnet_mapper[0].weight, std=0.02)  # 条件层
                nn.init.normal_(self.effnet_mapper[2].weight, std=0.02)  # 条件层
    
            # 如果有像素映射器，则对其权重进行初始化
            if hasattr(self, "pixels_mapper"):
                nn.init.normal_(self.pixels_mapper[0].weight, std=0.02)  # 条件层
                nn.init.normal_(self.pixels_mapper[2].weight, std=0.02)  # 条件层
    
            # 对嵌入层的权重进行Xavier均匀分布初始化
            torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # 输入层
            # 将分类器的权重初始化为0
            nn.init.constant_(self.clf[1].weight, 0)  # 输出层
    
            # 初始化块的权重
            for level_block in self.down_blocks + self.up_blocks:
                # 遍历每个块
                for block in level_block:
                    # 如果是SDCascadeResBlock类型
                    if isinstance(block, SDCascadeResBlock):
                        # 对最后一层的权重进行调整
                        block.channelwise[-1].weight.data *= np.sqrt(1 / sum(self.config.blocks[0]))
                    # 如果是SDCascadeTimestepBlock类型
                    elif isinstance(block, SDCascadeTimestepBlock):
                        # 将映射器的权重初始化为0
                        nn.init.constant_(block.mapper.weight, 0)
    # 定义获取时间步比率嵌入的方法，输入时间步比率和最大位置数
        def get_timestep_ratio_embedding(self, timestep_ratio, max_positions=10000):
            # 计算时间步比率与最大位置数的乘积
            r = timestep_ratio * max_positions
            # 计算嵌入维度的一半
            half_dim = self.config.timestep_ratio_embedding_dim // 2
    
            # 根据最大位置数和一半维度计算嵌入的基础值
            emb = math.log(max_positions) / (half_dim - 1)
            # 生成从0到half_dim的张量，乘以负的基础值并取指数
            emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
            # 将时间步比率和嵌入结合，扩展维度
            emb = r[:, None] * emb[None, :]
            # 将正弦和余弦值拼接在一起
            emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    
            # 如果嵌入维度为奇数，则进行零填充
            if self.config.timestep_ratio_embedding_dim % 2 == 1:  # zero pad
                emb = nn.functional.pad(emb, (0, 1), mode="constant")
    
            # 将嵌入转换为与r相同的数据类型并返回
            return emb.to(dtype=r.dtype)
    
        # 定义获取CLIP嵌入的方法，输入文本和图像的池化结果
        def get_clip_embeddings(self, clip_txt_pooled, clip_txt=None, clip_img=None):
            # 如果文本池的形状为二维，增加一个维度
            if len(clip_txt_pooled.shape) == 2:
                clip_txt_pool = clip_txt_pooled.unsqueeze(1)
            # 将文本池通过映射器转换并调整维度
            clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(
                clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.config.clip_seq, -1
            )
            # 如果提供了文本和图像，进行相应的映射和拼接
            if clip_txt is not None and clip_img is not None:
                clip_txt = self.clip_txt_mapper(clip_txt)
                # 如果图像的形状为二维，增加一个维度
                if len(clip_img.shape) == 2:
                    clip_img = clip_img.unsqueeze(1)
                # 将图像通过映射器转换并调整维度
                clip_img = self.clip_img_mapper(clip_img).view(
                    clip_img.size(0), clip_img.size(1) * self.config.clip_seq, -1
                )
                # 将文本、文本池和图像拼接在一起
                clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
            else:
                # 如果没有图像，只返回文本池
                clip = clip_txt_pool
            # 对最终的CLIP嵌入进行归一化并返回
            return self.clip_norm(clip)
    # 定义一个私有方法 _down_encode，接受输入 x，r_embed 和 clip
        def _down_encode(self, x, r_embed, clip):
            # 初始化一个空列表，用于存储每个层的输出
            level_outputs = []
            # 将 down_blocks、down_downscalers 和 down_repeat_mappers 组合成一个可迭代的元组
            block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
    
            # 如果处于训练模式并启用梯度检查点
            if self.training and self.gradient_checkpointing:
    
                # 定义一个用于创建自定义前向传播的方法
                def create_custom_forward(module):
                    # 定义一个自定义前向传播，接受任意输入
                    def custom_forward(*inputs):
                        return module(*inputs)
    
                    return custom_forward
    
                # 遍历 block_group 中的每一组 down_block、downscaler 和 repmap
                for down_block, downscaler, repmap in block_group:
                    # 使用 downscaler 对输入 x 进行下采样
                    x = downscaler(x)
                    # 遍历 repmap 的长度加一
                    for i in range(len(repmap) + 1):
                        # 遍历 down_block 中的每个块
                        for block in down_block:
                            # 如果块是 SDCascadeResBlock 类型
                            if isinstance(block, SDCascadeResBlock):
                                # 使用梯度检查点进行前向传播
                                x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, use_reentrant=False)
                            # 如果块是 SDCascadeAttnBlock 类型
                            elif isinstance(block, SDCascadeAttnBlock):
                                # 使用梯度检查点进行前向传播，传入 clip
                                x = torch.utils.checkpoint.checkpoint(
                                    create_custom_forward(block), x, clip, use_reentrant=False
                                )
                            # 如果块是 SDCascadeTimestepBlock 类型
                            elif isinstance(block, SDCascadeTimestepBlock):
                                # 使用梯度检查点进行前向传播，传入 r_embed
                                x = torch.utils.checkpoint.checkpoint(
                                    create_custom_forward(block), x, r_embed, use_reentrant=False
                                )
                            # 其他块类型
                            else:
                                # 使用梯度检查点进行前向传播
                                x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), use_reentrant=False)
                        # 如果 i 小于 repmap 的长度
                        if i < len(repmap):
                            # 使用当前的 repmap 对 x 进行处理
                            x = repmap[i](x)
                    # 将当前层的输出插入到 level_outputs 的开头
                    level_outputs.insert(0, x)
            # 如果不是训练模式或未启用梯度检查点
            else:
                # 遍历 block_group 中的每一组 down_block、downscaler 和 repmap
                for down_block, downscaler, repmap in block_group:
                    # 使用 downscaler 对输入 x 进行下采样
                    x = downscaler(x)
                    # 遍历 repmap 的长度加一
                    for i in range(len(repmap) + 1):
                        # 遍历 down_block 中的每个块
                        for block in down_block:
                            # 如果块是 SDCascadeResBlock 类型
                            if isinstance(block, SDCascadeResBlock):
                                # 直接对 x 进行前向传播
                                x = block(x)
                            # 如果块是 SDCascadeAttnBlock 类型
                            elif isinstance(block, SDCascadeAttnBlock):
                                # 直接对 x 进行前向传播，传入 clip
                                x = block(x, clip)
                            # 如果块是 SDCascadeTimestepBlock 类型
                            elif isinstance(block, SDCascadeTimestepBlock):
                                # 直接对 x 进行前向传播，传入 r_embed
                                x = block(x, r_embed)
                            # 其他块类型
                            else:
                                # 直接对 x 进行前向传播
                                x = block(x)
                        # 如果 i 小于 repmap 的长度
                        if i < len(repmap):
                            # 使用当前的 repmap 对 x 进行处理
                            x = repmap[i](x)
                    # 将当前层的输出插入到 level_outputs 的开头
                    level_outputs.insert(0, x)
            # 返回所有层的输出
            return level_outputs
    
        # 定义前向传播方法，接受多个参数
        def forward(
            self,
            sample,
            timestep_ratio,
            clip_text_pooled,
            clip_text=None,
            clip_img=None,
            effnet=None,
            pixels=None,
            sca=None,
            crp=None,
            return_dict=True,
    ):
        # 如果 pixels 参数为 None，则初始化为一个全零的张量，尺寸为 (3, 8, 8)
        if pixels is None:
            pixels = sample.new_zeros(sample.size(0), 3, 8, 8)

        # 处理时间步比率嵌入
        timestep_ratio_embed = self.get_timestep_ratio_embedding(timestep_ratio)
        # 遍历配置中的时间步条件类型
        for c in self.config.timestep_conditioning_type:
            # 如果条件类型是 "sca"，则使用 sca 作为条件
            if c == "sca":
                cond = sca
            # 如果条件类型是 "crp"，则使用 crp 作为条件
            elif c == "crp":
                cond = crp
            # 否则条件为 None
            else:
                cond = None
            # 如果 cond 为 None，则使用与 timestep_ratio 同形状的零张量
            t_cond = cond or torch.zeros_like(timestep_ratio)
            # 将时间步比率嵌入与条件嵌入进行拼接
            timestep_ratio_embed = torch.cat([timestep_ratio_embed, self.get_timestep_ratio_embedding(t_cond)], dim=1)
        # 获取 CLIP 嵌入
        clip = self.get_clip_embeddings(clip_txt_pooled=clip_text_pooled, clip_txt=clip_text, clip_img=clip_img)

        # 模型块
        # 对样本进行嵌入
        x = self.embedding(sample)
        # 如果存在 effnet_mapper 且 effnet 不为 None，则进行映射
        if hasattr(self, "effnet_mapper") and effnet is not None:
            x = x + self.effnet_mapper(
                # 对 effnet 进行上采样，调整到与 x 相同的空间尺寸
                nn.functional.interpolate(effnet, size=x.shape[-2:], mode="bilinear", align_corners=True)
            )
        # 如果存在 pixels_mapper，则进行映射
        if hasattr(self, "pixels_mapper"):
            x = x + nn.functional.interpolate(
                # 对 pixels 进行映射并上采样，调整到与 x 相同的空间尺寸
                self.pixels_mapper(pixels), size=x.shape[-2:], mode="bilinear", align_corners=True
            )
        # 通过下采样编码器处理 x 和其他嵌入
        level_outputs = self._down_encode(x, timestep_ratio_embed, clip)
        # 通过上采样解码器处理 level_outputs
        x = self._up_decode(level_outputs, timestep_ratio_embed, clip)
        # 使用分类器生成最终样本
        sample = self.clf(x)

        # 如果不需要返回字典格式的结果，则返回单个样本元组
        if not return_dict:
            return (sample,)
        # 返回 StableCascadeUNetOutput 对象，包含样本
        return StableCascadeUNetOutput(sample=sample)
```