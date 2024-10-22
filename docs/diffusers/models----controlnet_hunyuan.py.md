# `.\diffusers\models\controlnet_hunyuan.py`

```py
# 版权信息，声明代码的版权所有者和年份
# Copyright 2024 HunyuanDiT Authors, Qixun Wang and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版许可使用本文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 本文件的使用需遵循许可证的规定
# you may not use this file except in compliance with the License.
# 可通过以下网址获取许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“现状”提供的
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证中关于权限和限制的具体条款
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Dict, Optional, Union  # 导入用于类型提示的类型

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入配置混合和注册功能
from ..utils import logging  # 从工具模块导入日志功能
from .attention_processor import AttentionProcessor  # 导入注意力处理器
from .controlnet import BaseOutput, Tuple, zero_module  # 导入控制网络的基础输出和相关类型
from .embeddings import (  # 从嵌入模块导入多个类
    HunyuanCombinedTimestepTextSizeStyleEmbedding,
    PatchEmbed,
    PixArtAlphaTextProjection,
)
from .modeling_utils import ModelMixin  # 导入模型混合类
from .transformers.hunyuan_transformer_2d import HunyuanDiTBlock  # 导入 Hunyuan 二维变换器块

logger = logging.get_logger(__name__)  # 初始化日志记录器，使用当前模块名作为标识

@dataclass  # 将 HunyuanControlNetOutput 类标记为数据类
class HunyuanControlNetOutput(BaseOutput):  # 定义 HunyuanControlNetOutput 类，继承自 BaseOutput
    controlnet_block_samples: Tuple[torch.Tensor]  # 定义一个包含控制网络块样本的元组属性

class HunyuanDiT2DControlNetModel(ModelMixin, ConfigMixin):  # 定义 HunyuanDiT2DControlNetModel 类，继承自多个混合类
    @register_to_config  # 将该方法注册到配置系统
    def __init__(  # 定义构造函数
        self,
        conditioning_channels: int = 3,  # 初始化条件通道数，默认为 3
        num_attention_heads: int = 16,  # 初始化注意力头数，默认为 16
        attention_head_dim: int = 88,  # 初始化注意力头维度，默认为 88
        in_channels: Optional[int] = None,  # 可选的输入通道数，默认为 None
        patch_size: Optional[int] = None,  # 可选的补丁大小，默认为 None
        activation_fn: str = "gelu-approximate",  # 初始化激活函数，默认为“gelu-approximate”
        sample_size=32,  # 初始化样本大小，默认为 32
        hidden_size=1152,  # 初始化隐藏层大小，默认为 1152
        transformer_num_layers: int = 40,  # 初始化变换器层数，默认为 40
        mlp_ratio: float = 4.0,  # 初始化 MLP 比率，默认为 4.0
        cross_attention_dim: int = 1024,  # 初始化交叉注意力维度，默认为 1024
        cross_attention_dim_t5: int = 2048,  # 初始化 T5 交叉注意力维度，默认为 2048
        pooled_projection_dim: int = 1024,  # 初始化池化投影维度，默认为 1024
        text_len: int = 77,  # 初始化文本长度，默认为 77
        text_len_t5: int = 256,  # 初始化 T5 文本长度，默认为 256
        use_style_cond_and_image_meta_size: bool = True,  # 初始化样式条件和图像元大小的使用标志，默认为 True
    # 初始化父类
        ):
            super().__init__()
            # 设置注意力头数量
            self.num_heads = num_attention_heads
            # 计算内部维度
            self.inner_dim = num_attention_heads * attention_head_dim
    
            # 创建文本嵌入投影层
            self.text_embedder = PixArtAlphaTextProjection(
                in_features=cross_attention_dim_t5,  # 输入特征维度
                hidden_size=cross_attention_dim_t5 * 4,  # 隐藏层大小
                out_features=cross_attention_dim,  # 输出特征维度
                act_fn="silu_fp32",  # 激活函数
            )
    
            # 创建文本嵌入的可学习参数
            self.text_embedding_padding = nn.Parameter(
                torch.randn(text_len + text_len_t5, cross_attention_dim, dtype=torch.float32)  # 初始化随机张量
            )
    
            # 创建位置嵌入层
            self.pos_embed = PatchEmbed(
                height=sample_size,  # 高度
                width=sample_size,  # 宽度
                in_channels=in_channels,  # 输入通道数
                embed_dim=hidden_size,  # 嵌入维度
                patch_size=patch_size,  # 每个块的大小
                pos_embed_type=None,  # 位置嵌入类型
            )
    
            # 创建时间额外嵌入层
            self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
                hidden_size,  # 隐藏层大小
                pooled_projection_dim=pooled_projection_dim,  # 池化投影维度
                seq_len=text_len_t5,  # 序列长度
                cross_attention_dim=cross_attention_dim_t5,  # 跨注意力维度
                use_style_cond_and_image_meta_size=use_style_cond_and_image_meta_size,  # 是否使用样式条件和图像元大小
            )
    
            # 初始化控制网络模块列表
            self.controlnet_blocks = nn.ModuleList([])
    
            # 初始化 HunyuanDiT 模块列表
            self.blocks = nn.ModuleList(
                [
                    HunyuanDiTBlock(
                        dim=self.inner_dim,  # 模块维度
                        num_attention_heads=self.config.num_attention_heads,  # 注意力头数量
                        activation_fn=activation_fn,  # 激活函数
                        ff_inner_dim=int(self.inner_dim * mlp_ratio),  # 前馈层内部维度
                        cross_attention_dim=cross_attention_dim,  # 跨注意力维度
                        qk_norm=True,  # 是否使用 QK 归一化
                        skip=False,  # 是否跳过，首个模型的前半部分总为 False
                    )
                    for layer in range(transformer_num_layers // 2 - 1)  # 根据层数生成 HunyuanDiTBlock
                ]
            )
            # 初始化输入层
            self.input_block = zero_module(nn.Linear(hidden_size, hidden_size))  
            # 根据模块数量创建控制网络块
            for _ in range(len(self.blocks)):
                controlnet_block = nn.Linear(hidden_size, hidden_size)  # 初始化控制网络块
                controlnet_block = zero_module(controlnet_block)  # 零初始化
                self.controlnet_blocks.append(controlnet_block)  # 添加到控制网络模块列表
    
        @property  # 声明为属性
    # 定义一个返回注意力处理器的函数，返回值为字典类型，字典包含模型中所有的注意力处理器
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回值:
            `dict` 的注意力处理器: 一个字典，包含模型中所有注意力处理器，按其权重名称索引。
        """
        # 初始化一个空字典，用于存储处理器
        processors = {}

        # 定义递归函数，将注意力处理器添加到字典中
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否有 "get_processor" 方法
            if hasattr(module, "get_processor"):
                # 如果有，调用该方法并将处理器存入字典
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用函数，将子模块的处理器添加到字典中
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            # 返回更新后的处理器字典
            return processors

        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 递归调用函数，添加所有子模块的处理器
            fn_recursive_add_processors(name, module, processors)

        # 返回包含所有处理器的字典
        return processors

    # 定义一个设置注意力处理器的函数
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。

        参数:
            processor (`dict` of `AttentionProcessor` 或 `AttentionProcessor`):
                实例化的处理器类或处理器类字典，将作为所有 `Attention` 层的处理器。如果 `processor` 是字典，键需要定义
                对应的交叉注意力处理器的路径。强烈建议在设置可训练的注意力处理器时使用。
        """
        # 获取当前注意力处理器字典的键数量
        count = len(self.attn_processors.keys())

        # 检查传入的处理器字典数量是否与当前注意力层数量匹配
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器字典，但处理器的数量 {len(processor)} 与注意力层的数量: {count} 不匹配。请确保传入 {count} 个处理器类。"
            )

        # 定义递归函数，为模块设置注意力处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查模块是否有 "set_processor" 方法
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，直接设置处理器
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中获取对应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用函数，为子模块设置处理器
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 递归调用函数，设置所有子模块的处理器
            fn_recursive_attn_processor(name, module, processor)

    # 定义一个类方法，从 transformer 创建对象
    @classmethod
    def from_transformer(
        cls, transformer, conditioning_channels=3, transformer_num_layers=None, load_weights_from_transformer=True
    # 开始方法定义
        ):
            # 获取变换器的配置
            config = transformer.config
            # 获取激活函数
            activation_fn = config.activation_fn
            # 获取注意力头的维度
            attention_head_dim = config.attention_head_dim
            # 获取交叉注意力的维度
            cross_attention_dim = config.cross_attention_dim
            # 获取 T5 模型的交叉注意力维度
            cross_attention_dim_t5 = config.cross_attention_dim_t5
            # 获取隐藏层的大小
            hidden_size = config.hidden_size
            # 获取输入通道的数量
            in_channels = config.in_channels
            # 获取多层感知器的比率
            mlp_ratio = config.mlp_ratio
            # 获取注意力头的数量
            num_attention_heads = config.num_attention_heads
            # 获取补丁的大小
            patch_size = config.patch_size
            # 获取样本的大小
            sample_size = config.sample_size
            # 获取文本的长度
            text_len = config.text_len
            # 获取 T5 模型的文本长度
            text_len_t5 = config.text_len_t5
    
            # 设置条件通道
            conditioning_channels = conditioning_channels
            # 设置变换器层数，如果未提供，则使用配置中的默认值
            transformer_num_layers = transformer_num_layers or config.transformer_num_layers
    
            # 实例化 ControlNet 对象，传入多个参数
            controlnet = cls(
                conditioning_channels=conditioning_channels,
                transformer_num_layers=transformer_num_layers,
                activation_fn=activation_fn,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                cross_attention_dim_t5=cross_attention_dim_t5,
                hidden_size=hidden_size,
                in_channels=in_channels,
                mlp_ratio=mlp_ratio,
                num_attention_heads=num_attention_heads,
                patch_size=patch_size,
                sample_size=sample_size,
                text_len=text_len,
                text_len_t5=text_len_t5,
            )
            # 如果需要从变换器加载权重
            if load_weights_from_transformer:
                # 加载状态字典，忽略缺失的键
                key = controlnet.load_state_dict(transformer.state_dict(), strict=False)
                # 记录警告，显示缺失的键
                logger.warning(f"controlnet load from Hunyuan-DiT. missing_keys: {key[0]}")
            # 返回创建的 ControlNet 对象
            return controlnet
    
        # 定义前向传播方法
        def forward(
            # 隐藏状态输入
            hidden_states,
            # 时间步长
            timestep,
            # 控制网条件张量
            controlnet_cond: torch.Tensor,
            # 条件缩放因子，默认值为 1.0
            conditioning_scale: float = 1.0,
            # 编码器的隐藏状态，可选
            encoder_hidden_states=None,
            # 文本嵌入的掩码，可选
            text_embedding_mask=None,
            # T5 编码器的隐藏状态，可选
            encoder_hidden_states_t5=None,
            # T5 文本嵌入的掩码，可选
            text_embedding_mask_t5=None,
            # 图像元数据的大小，可选
            image_meta_size=None,
            # 风格参数，可选
            style=None,
            # 图像旋转嵌入，可选
            image_rotary_emb=None,
            # 是否返回字典格式的输出，默认值为 True
            return_dict=True,
# HunyuanDiT2DMultiControlNetModel 类，用于封装多个 HunyuanDiT2DControlNetModel 实例
class HunyuanDiT2DMultiControlNetModel(ModelMixin):
    r"""
    `HunyuanDiT2DMultiControlNetModel` 是用于 Multi-HunyuanDiT2DControlNetModel 的封装类

    该模块为多个 `HunyuanDiT2DControlNetModel` 实例提供封装。`forward()` API 设计上与 `HunyuanDiT2DControlNetModel` 兼容。

    参数:
        controlnets (`List[HunyuanDiT2DControlNetModel]`):
            在去噪过程中为 unet 提供额外的条件。必须将多个 `HunyuanDiT2DControlNetModel` 作为列表设置。
    """

    # 初始化方法，接收控制网络列表并调用父类构造函数
    def __init__(self, controlnets):
        super().__init__()  # 调用父类构造函数以初始化基类
        self.nets = nn.ModuleList(controlnets)  # 将控制网络列表封装为一个可训练的模块列表

    # 前向传播方法，处理输入并生成输出
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态
        timestep,  # 当前时间步
        controlnet_cond: torch.Tensor,  # 控制网络的条件张量
        conditioning_scale: float = 1.0,  # 条件缩放因子，默认值为 1.0
        encoder_hidden_states=None,  # 可选的编码器隐藏状态
        text_embedding_mask=None,  # 文本嵌入的掩码
        encoder_hidden_states_t5=None,  # 可选的 T5 编码器隐藏状态
        text_embedding_mask_t5=None,  # T5 文本嵌入的掩码
        image_meta_size=None,  # 图像元数据大小
        style=None,  # 样式信息
        image_rotary_emb=None,  # 图像旋转嵌入
        return_dict=True,  # 是否以字典形式返回结果，默认为 True
    ):
        """
        [`HunyuanDiT2DControlNetModel`] 的前向传播方法。

        参数：
        hidden_states (`torch.Tensor`，形状为 `(batch size, dim, height, width)`):
            输入张量。
        timestep ( `torch.LongTensor`，*可选*):
            用于指示去噪步骤。
        controlnet_cond ( `torch.Tensor` ):
            ControlNet 的条件输入。
        conditioning_scale ( `float` ):
            指示条件的比例。
        encoder_hidden_states ( `torch.Tensor`，形状为 `(batch size, sequence len, embed dims)`，*可选*):
            交叉注意力层的条件嵌入。这是 `BertModel` 的输出。
        text_embedding_mask: torch.Tensor
            形状为 `(batch, key_tokens)` 的注意力掩码，应用于 `encoder_hidden_states`。这是 `BertModel` 的输出。
        encoder_hidden_states_t5 ( `torch.Tensor`，形状为 `(batch size, sequence len, embed dims)`，*可选*):
            交叉注意力层的条件嵌入。这是 T5 文本编码器的输出。
        text_embedding_mask_t5: torch.Tensor
            形状为 `(batch, key_tokens)` 的注意力掩码，应用于 `encoder_hidden_states`。这是 T5 文本编码器的输出。
        image_meta_size (torch.Tensor):
            条件嵌入，指示图像大小
        style: torch.Tensor:
            条件嵌入，指示样式
        image_rotary_emb (`torch.Tensor`):
            在注意力计算中应用于查询和键张量的图像旋转嵌入。
        return_dict: bool
            是否返回字典。
        """
        # 遍历 controlnet_cond、conditioning_scale 和自有网络的组合
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            # 调用 controlnet 处理输入，返回区块样本
            block_samples = controlnet(
                hidden_states=hidden_states,  # 输入隐藏状态
                timestep=timestep,  # 输入时间步
                controlnet_cond=image,  # 输入图像条件
                conditioning_scale=scale,  # 输入条件比例
                encoder_hidden_states=encoder_hidden_states,  # 输入 BERT 编码的隐藏状态
                text_embedding_mask=text_embedding_mask,  # 输入 BERT 的注意力掩码
                encoder_hidden_states_t5=encoder_hidden_states_t5,  # 输入 T5 编码的隐藏状态
                text_embedding_mask_t5=text_embedding_mask_t5,  # 输入 T5 的注意力掩码
                image_meta_size=image_meta_size,  # 输入图像元数据大小
                style=style,  # 输入样式条件
                image_rotary_emb=image_rotary_emb,  # 输入图像旋转嵌入
                return_dict=return_dict,  # 指示是否返回字典
            )

            # 合并样本
            if i == 0:  # 如果是第一个样本
                control_block_samples = block_samples  # 初始化样本
            else:  # 如果不是第一个样本
                # 合并现有样本和新样本
                control_block_samples = [
                    control_block_sample + block_sample  # 对应位置样本相加
                    for control_block_sample, block_sample in zip(control_block_samples[0], block_samples[0])
                ]
                control_block_samples = (control_block_samples,)  # 转换为元组

        # 返回合并后的样本
        return control_block_samples
```