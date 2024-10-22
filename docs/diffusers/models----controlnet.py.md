# `.\diffusers\models\controlnet.py`

```py
# 版权声明，标明版权所有者及其权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 该文件只能在符合许可证的情况下使用
# you may not use this file except in compliance with the License.
# 可以在以下地址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按“原样”分发，不提供任何形式的担保或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定语言所规定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入torch库，用于张量操作
from torch import nn  # 从torch库导入神经网络模块
from torch.nn import functional as F  # 从torch.nn导入功能性操作模块

# 导入配置相关的混合类和注册功能
from ..configuration_utils import ConfigMixin, register_to_config
# 导入原始模型的加载混合类
from ..loaders.single_file_model import FromOriginalModelMixin
# 导入基础输出类和日志记录功能
from ..utils import BaseOutput, logging
# 导入注意力处理器相关的组件
from .attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,  # 导入增加的KV注意力处理器
    CROSS_ATTENTION_PROCESSORS,  # 导入交叉注意力处理器
    AttentionProcessor,  # 导入注意力处理器基类
    AttnAddedKVProcessor,  # 导入增加KV的注意力处理器
    AttnProcessor,  # 导入基本的注意力处理器
)
# 导入嵌入相关的组件
from .embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
# 导入模型相关的混合类
from .modeling_utils import ModelMixin
# 导入UNet的二维块相关组件
from .unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,  # 导入二维交叉注意力下采样块
    DownBlock2D,  # 导入二维下采样块
    UNetMidBlock2D,  # 导入UNet的中间块
    UNetMidBlock2DCrossAttn,  # 导入具有交叉注意力的UNet中间块
    get_down_block,  # 导入获取下采样块的函数
)
# 导入UNet的条件模型
from .unets.unet_2d_condition import UNet2DConditionModel

# 获取logger实例，用于记录日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass  # 使用dataclass装饰器定义一个数据类
class ControlNetOutput(BaseOutput):
    """
    ControlNetModel的输出。

    参数：
        down_block_res_samples (`tuple[torch.Tensor]`):
            不同分辨率下的下采样激活元组，每个张量形状为`(batch_size, channel * resolution, height // resolution, width // resolution)`。
            输出可用于对原始UNet的下采样激活进行条件化。
        mid_down_block_re_sample (`torch.Tensor`):
            中间块（最低采样分辨率）的激活。每个张量形状为
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`。
            输出可用于对原始UNet的中间块激活进行条件化。
    """

    down_block_res_samples: Tuple[torch.Tensor]  # 下采样块的激活张量元组
    mid_block_res_sample: torch.Tensor  # 中间块的激活张量


class ControlNetConditioningEmbedding(nn.Module):
    """
    引用 https://arxiv.org/abs/2302.05543: “Stable Diffusion使用类似于VQ-GAN的预处理方法
    将整个512 × 512图像数据集转换为较小的64 × 64“潜在图像”，以实现稳定训练。
    这要求ControlNets将基于图像的条件转换为64 × 64特征空间，以匹配卷积大小。
    我们使用一个包含四个卷积层的小型网络E(·)，卷积核为4 × 4，步幅为2 × 2。
    # 文档字符串，描述这个模块的功能，提到使用 ReLU 激活函数，通道数为 16, 32, 64, 128，采用高斯权重初始化，并与整个模型共同训练，以将图像空间条件编码为特征图
    """

    # 初始化函数，用于定义该类的基本属性
    def __init__(
        # 条件嵌入通道数
        conditioning_embedding_channels: int,
        # 条件通道数，默认为 3（即 RGB 图像）
        conditioning_channels: int = 3,
        # 输出通道的元组，定义卷积层的通道数
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 定义输入卷积层，接收条件通道并输出第一个块的通道数
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        # 创建一个空的模块列表，用于存储后续的卷积块
        self.blocks = nn.ModuleList([])

        # 遍历 block_out_channels 列表，构建多个卷积块
        for i in range(len(block_out_channels) - 1):
            # 当前块的输入通道数
            channel_in = block_out_channels[i]
            # 下一个块的输出通道数
            channel_out = block_out_channels[i + 1]
            # 添加一个卷积层，保持输入通道数
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            # 添加另一个卷积层，改变输出通道数，同时步幅为 2，进行下采样
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        # 定义输出卷积层，将最后一个块的通道数映射到条件嵌入通道数，并使用零初始化
        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    # 前向传播函数，定义输入如何通过网络传递
    def forward(self, conditioning):
        # 通过输入卷积层处理条件输入，得到嵌入
        embedding = self.conv_in(conditioning)
        # 应用 SiLU 激活函数
        embedding = F.silu(embedding)

        # 遍历所有定义的卷积块，逐层处理嵌入
        for block in self.blocks:
            # 通过当前卷积块处理嵌入
            embedding = block(embedding)
            # 再次应用 SiLU 激活函数
            embedding = F.silu(embedding)

        # 通过输出卷积层处理嵌入
        embedding = self.conv_out(embedding)

        # 返回最终的嵌入结果
        return embedding
# 定义一个 ControlNet 模型类，继承自 ModelMixin, ConfigMixin, 和 FromOriginalModelMixin
class ControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    A ControlNet model.
    """  # 文档字符串，描述该类是一个 ControlNet 模型

    _supports_gradient_checkpointing = True  # 设置支持梯度检查点的标志为真

    @register_to_config  # 注册到配置中
    def __init__(  # 初始化方法，构造 ControlNetModel 的实例
        self,  # 指向实例本身的引用
        in_channels: int = 4,  # 输入通道数，默认为 4
        conditioning_channels: int = 3,  # 条件通道数，默认为 3
        flip_sin_to_cos: bool = True,  # 是否将正弦转换为余弦，默认为真
        freq_shift: int = 0,  # 频率偏移量，默认为 0
        down_block_types: Tuple[str, ...] = (  # 下采样块类型的元组
            "CrossAttnDownBlock2D",  # 第一个下采样块类型
            "CrossAttnDownBlock2D",  # 第二个下采样块类型
            "CrossAttnDownBlock2D",  # 第三个下采样块类型
            "DownBlock2D",  # 第四个下采样块类型
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",  # 中间块类型，默认为 UNet 中间块
        only_cross_attention: Union[bool, Tuple[bool]] = False,  # 是否仅使用交叉注意力，默认为假
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),  # 每个块的输出通道数
        layers_per_block: int = 2,  # 每个块的层数，默认为 2
        downsample_padding: int = 1,  # 下采样的填充，默认为 1
        mid_block_scale_factor: float = 1,  # 中间块的缩放因子，默认为 1
        act_fn: str = "silu",  # 激活函数类型，默认为 "silu"
        norm_num_groups: Optional[int] = 32,  # 规范化的组数，默认为 32
        norm_eps: float = 1e-5,  # 规范化的 epsilon 值，默认为 1e-5
        cross_attention_dim: int = 1280,  # 交叉注意力的维度，默认为 1280
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,  # 每个块的变换层数，默认为 1
        encoder_hid_dim: Optional[int] = None,  # 编码器隐藏维度，可选
        encoder_hid_dim_type: Optional[str] = None,  # 编码器隐藏维度类型，可选
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,  # 注意力头的维度，默认为 8
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,  # 注意力头数量，可选
        use_linear_projection: bool = False,  # 是否使用线性投影，默认为假
        class_embed_type: Optional[str] = None,  # 类嵌入类型，可选
        addition_embed_type: Optional[str] = None,  # 附加嵌入类型，可选
        addition_time_embed_dim: Optional[int] = None,  # 附加时间嵌入维度，可选
        num_class_embeds: Optional[int] = None,  # 类嵌入数量，可选
        upcast_attention: bool = False,  # 是否上调注意力，默认为假
        resnet_time_scale_shift: str = "default",  # ResNet 时间缩放偏移，默认为 "default"
        projection_class_embeddings_input_dim: Optional[int] = None,  # 投影类嵌入输入维度，可选
        controlnet_conditioning_channel_order: str = "rgb",  # ControlNet 条件通道顺序，默认为 "rgb"
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),  # 条件嵌入输出通道数
        global_pool_conditions: bool = False,  # 是否使用全局池化条件，默认为假
        addition_embed_type_num_heads: int = 64,  # 附加嵌入类型的头数量，默认为 64
    @classmethod
    def from_unet(  # 从 UNet 创建 ControlNetModel 的类方法
        cls,  # 指向类本身的引用
        unet: UNet2DConditionModel,  # 传入的 UNet2DConditionModel 实例
        controlnet_conditioning_channel_order: str = "rgb",  # ControlNet 条件通道顺序，默认为 "rgb"
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),  # 条件嵌入输出通道数
        load_weights_from_unet: bool = True,  # 是否从 UNet 加载权重，默认为真
        conditioning_channels: int = 3,  # 条件通道数，默认为 3
    @property  # 定义一个属性装饰器
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors  # 注释，说明该属性是从另一个模块复制过来的
    # 定义一个返回模型中所有注意力处理器的字典的方法
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r""" 
        返回：
            `dict` 的注意力处理器：包含模型中所有注意力处理器的字典，按其权重名称索引。
        """
        # 初始化一个空字典用于存储处理器
        processors = {}
    
        # 定义一个递归函数来添加注意力处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 如果模块具有获取处理器的方法，将其添加到处理器字典
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
    
            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用以处理子模块
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
            # 返回更新后的处理器字典
            return processors
    
        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数添加处理器
            fn_recursive_add_processors(name, module, processors)
    
        # 返回包含所有处理器的字典
        return processors
    
    # 从 UNet2DConditionModel 的 set_attn_processor 方法复制而来
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。
    
        参数：
            processor (`dict` of `AttentionProcessor` 或仅 `AttentionProcessor`):
                实例化的处理器类或处理器类的字典，将被设置为**所有** `Attention` 层的处理器。
                
                如果 `processor` 是一个字典，则键需要定义相应交叉注意力处理器的路径。当设置可训练的注意力处理器时，强烈推荐这样做。
    
        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())
    
        # 检查传入的处理器字典长度是否与注意力层数量匹配
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器字典，但处理器数量 {len(processor)} 与注意力层数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
            )
    
        # 定义一个递归函数来设置注意力处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块具有设置处理器的方法，进行处理器设置
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用以处理子模块
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数设置处理器
            fn_recursive_attn_processor(name, module, processor)
    
    # 从 UNet2DConditionModel 的 set_default_attn_processor 方法复制而来
    # 设置默认的注意力处理器，禁用自定义注意力处理器
    def set_default_attn_processor(self):
        # 文档字符串，说明该函数的作用
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        # 检查所有注意力处理器是否为已添加的 KV 注意力处理器
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 创建添加 KV 的注意力处理器
            processor = AttnAddedKVProcessor()
        # 检查所有注意力处理器是否为交叉注意力处理器
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 创建常规的注意力处理器
            processor = AttnProcessor()
        else:
            # 引发错误，说明注意力处理器类型不支持
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )
    
        # 设置选择的注意力处理器
        self.set_attn_processor(processor)
    
    # 从 diffusers.models.unets.unet_2d_condition 中复制的函数
    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        # 如果模块是特定类型，则设置其梯度检查点标志
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
            module.gradient_checkpointing = value
    
    # 前向传播方法，处理输入样本和相关参数
    def forward(
        self,
        sample: torch.Tensor,  # 输入样本张量
        timestep: Union[torch.Tensor, float, int],  # 当前时间步
        encoder_hidden_states: torch.Tensor,  # 编码器的隐藏状态
        controlnet_cond: torch.Tensor,  # ControlNet 的条件输入
        conditioning_scale: float = 1.0,  # 条件缩放因子
        class_labels: Optional[torch.Tensor] = None,  # 可选的类别标签
        timestep_cond: Optional[torch.Tensor] = None,  # 可选的时间步条件
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,  # 可选的额外条件参数
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数
        guess_mode: bool = False,  # 是否启用猜测模式
        return_dict: bool = True,  # 是否以字典形式返回结果
# 将给定的 PyTorch 模块的所有参数初始化为零
def zero_module(module):
    # 遍历模块的所有参数
    for p in module.parameters():
        # 将当前参数 p 的值初始化为零
        nn.init.zeros_(p)
    # 返回已修改的模块
    return module
```