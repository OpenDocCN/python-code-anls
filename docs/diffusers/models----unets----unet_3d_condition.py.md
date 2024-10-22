# `.\diffusers\models\unets\unet_3d_condition.py`

```
# 版权声明，声明此代码的版权信息和所有权
# Copyright 2024 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
# 版权声明，声明此代码的版权信息和所有权
# Copyright 2024 The ModelScope Team.
#
# 许可声明，声明本代码使用的 Apache 许可证 2.0 版本
# Licensed under the Apache License, Version 2.0 (the "License");
# 使用此文件前需遵守许可证规定
# you may not use this file except in compliance with the License.
# 可在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 免责声明，说明软件在许可下按 "原样" 提供，不附加任何明示或暗示的保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 许可证中规定的权限和限制说明
# See the License for the specific language governing permissions and
# limitations under the License.

# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入所需的类型提示
from typing import Any, Dict, List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 PyTorch 神经网络模块
import torch.nn as nn
# 导入 PyTorch 的检查点工具
import torch.utils.checkpoint

# 导入配置相关的工具类和函数
from ...configuration_utils import ConfigMixin, register_to_config
# 导入 UNet2D 条件加载器混合类
from ...loaders import UNet2DConditionLoadersMixin
# 导入基本输出类和日志工具
from ...utils import BaseOutput, logging
# 导入激活函数获取工具
from ..activations import get_activation
# 导入各种注意力处理器相关组件
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,  # 导入添加键值对注意力处理器
    CROSS_ATTENTION_PROCESSORS,      # 导入交叉注意力处理器
    Attention,                       # 导入注意力类
    AttentionProcessor,              # 导入注意力处理器基类
    AttnAddedKVProcessor,            # 导入添加键值对的注意力处理器
    AttnProcessor,                   # 导入普通注意力处理器
    FusedAttnProcessor2_0,           # 导入融合注意力处理器
)
# 导入时间步嵌入和时间步类
from ..embeddings import TimestepEmbedding, Timesteps
# 导入模型混合类
from ..modeling_utils import ModelMixin
# 导入时间变换器模型
from ..transformers.transformer_temporal import TransformerTemporalModel
# 导入 3D UNet 相关的块
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,          # 导入交叉注意力下采样块
    CrossAttnUpBlock3D,            # 导入交叉注意力上采样块
    DownBlock3D,                   # 导入下采样块
    UNetMidBlock3DCrossAttn,      # 导入 UNet 中间交叉注意力块
    UpBlock3D,                     # 导入上采样块
    get_down_block,                # 导入获取下采样块的函数
    get_up_block,                  # 导入获取上采样块的函数
)

# 创建日志记录器，使用当前模块的名称
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义 UNet3DConditionOutput 数据类，继承自 BaseOutput
@dataclass
class UNet3DConditionOutput(BaseOutput):
    """
    [`UNet3DConditionModel`] 的输出类。

    参数：
        sample (`torch.Tensor` 的形状为 `(batch_size, num_channels, num_frames, height, width)`):
            基于 `encoder_hidden_states` 输入的隐藏状态输出。模型最后一层的输出。
    """

    sample: torch.Tensor  # 定义样本输出，类型为 PyTorch 张量

# 定义 UNet3DConditionModel 类，继承自多个混合类
class UNet3DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    条件 3D UNet 模型，接受噪声样本、条件状态和时间步，并返回形状为样本的输出。

    此模型继承自 [`ModelMixin`]。有关其通用方法的文档，请参阅超类文档（如下载或保存）。
    # 参数说明部分
    Parameters:
        # 输入/输出样本的高度和宽度，类型可以为整数或元组，默认为 None
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        # 输入样本的通道数，默认为 4
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        # 输出的通道数，默认为 4
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        # 使用的下采样块类型的元组，默认为指定的四种块
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D")`):
            The tuple of downsample blocks to use.
        # 使用的上采样块类型的元组，默认为指定的四种块
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D")`):
            The tuple of upsample blocks to use.
        # 每个块的输出通道数的元组，默认为 (320, 640, 1280, 1280)
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        # 每个块的层数，默认为 2
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        # 下采样卷积使用的填充，默认为 1
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        # 中间块使用的缩放因子，默认为 1.0
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        # 使用的激活函数，默认为 "silu"
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        # 用于归一化的组数，默认为 32；如果为 None，则跳过归一化和激活层
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, normalization and activation layers is skipped in post-processing.
        # 归一化使用的 epsilon 值，默认为 1e-5
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        # 交叉注意力特征的维度，默认为 1024
        cross_attention_dim (`int`, *optional*, defaults to 1024): The dimension of the cross attention features.
        # 注意力头的维度，默认为 64
        attention_head_dim (`int`, *optional*, defaults to 64): The dimension of the attention heads.
        # 注意力头的数量，类型为整数，默认为 None
        num_attention_heads (`int`, *optional*): The number of attention heads.
        # 时间条件投影层的维度，默认为 None
        time_cond_proj_dim (`int`, *optional*, defaults to `None`):
            The dimension of `cond_proj` layer in the timestep embedding.
    """

    # 是否支持梯度检查点，默认为 False
    _supports_gradient_checkpointing = False

    # 将此类注册到配置中
    @register_to_config
    # 初始化方法，用于创建类的实例
        def __init__(
            # 样本大小，默认为 None
            self,
            sample_size: Optional[int] = None,
            # 输入通道数量，默认为 4
            in_channels: int = 4,
            # 输出通道数量，默认为 4
            out_channels: int = 4,
            # 下采样块类型的元组，定义模型的下采样结构
            down_block_types: Tuple[str, ...] = (
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            # 上采样块类型的元组，定义模型的上采样结构
            up_block_types: Tuple[str, ...] = (
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
            ),
            # 每个块的输出通道数量，定义模型每个层的通道设置
            block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
            # 每个块的层数，默认为 2
            layers_per_block: int = 2,
            # 下采样时的填充大小，默认为 1
            downsample_padding: int = 1,
            # 中间块的缩放因子，默认为 1
            mid_block_scale_factor: float = 1,
            # 激活函数类型，默认为 "silu"
            act_fn: str = "silu",
            # 归一化组的数量，默认为 32
            norm_num_groups: Optional[int] = 32,
            # 归一化的 epsilon 值，默认为 1e-5
            norm_eps: float = 1e-5,
            # 跨注意力维度，默认为 1024
            cross_attention_dim: int = 1024,
            # 注意力头的维度，可以是单一整数或整数元组，默认为 64
            attention_head_dim: Union[int, Tuple[int]] = 64,
            # 注意力头的数量，可选参数
            num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
            # 时间条件投影维度，可选参数
            time_cond_proj_dim: Optional[int] = None,
        @property
        # 从 UNet2DConditionModel 复制的属性，获取注意力处理器
        # 返回所有注意力处理器的字典，以权重名称为索引
        def attn_processors(self) -> Dict[str, AttentionProcessor]:
            r"""
            Returns:
                `dict` of attention processors: A dictionary containing all attention processors used in the model with
                indexed by its weight name.
            """
            # 初始化处理器字典
            processors = {}
    
            # 递归添加处理器的函数
            def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
                # 如果模块有获取处理器的方法，添加到处理器字典中
                if hasattr(module, "get_processor"):
                    processors[f"{name}.processor"] = module.get_processor()
    
                # 遍历子模块，递归调用该函数
                for sub_name, child in module.named_children():
                    fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
                # 返回处理器字典
                return processors
    
            # 遍历当前类的子模块，调用递归添加处理器的函数
            for name, module in self.named_children():
                fn_recursive_add_processors(name, module, processors)
    
            # 返回所有处理器
            return processors
    
        # 从 UNet2DConditionModel 复制的设置注意力切片的方法
        # 从 UNet2DConditionModel 复制的设置注意力处理器的方法
    # 定义一个方法用于设置注意力处理器
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。
    
        参数：
            processor（`dict` of `AttentionProcessor` 或仅 `AttentionProcessor`）：
                实例化的处理器类或一个处理器类的字典，将作为所有 `Attention` 层的处理器。
    
                如果 `processor` 是一个字典，键需要定义相应的交叉注意力处理器的路径。
                在设置可训练的注意力处理器时，强烈推荐这样做。
    
        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())
    
        # 如果传入的处理器是字典，且数量不等于注意力层数量，抛出错误
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了一个处理器字典，但处理器的数量 {len(processor)} 与"
                f" 注意力层的数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
            )
    
        # 定义一个递归函数来设置每个模块的处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 如果模块有设置处理器的方法
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，直接设置处理器
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中获取相应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历子模块并递归调用处理器设置
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历当前对象的所有子模块，并调用递归设置函数
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)
    # 定义一个方法来启用前馈层的分块处理
        def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
            """
            设置注意力处理器以使用 [前馈分块](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers)。
    
            参数：
                chunk_size (`int`, *可选*):
                    前馈层的分块大小。如果未指定，将对维度为`dim`的每个张量单独运行前馈层。
                dim (`int`, *可选*, 默认为`0`):
                    应对哪个维度进行前馈计算的分块。可以选择 dim=0（批次）或 dim=1（序列长度）。
            """
            # 确保 dim 参数为 0 或 1
            if dim not in [0, 1]:
                raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")
    
            # 默认的分块大小为 1
            chunk_size = chunk_size or 1
    
            # 定义一个递归函数来设置每个模块的分块前馈处理
            def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
                # 如果模块具有设置分块前馈的属性，则设置它
                if hasattr(module, "set_chunk_feed_forward"):
                    module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
    
                # 遍历子模块，递归调用函数
                for child in module.children():
                    fn_recursive_feed_forward(child, chunk_size, dim)
    
            # 遍历当前实例的子模块，应用递归函数
            for module in self.children():
                fn_recursive_feed_forward(module, chunk_size, dim)
    
        # 定义一个方法来禁用前馈层的分块处理
        def disable_forward_chunking(self):
            # 定义一个递归函数来禁用分块前馈处理
            def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
                # 如果模块具有设置分块前馈的属性，则设置为 None
                if hasattr(module, "set_chunk_feed_forward"):
                    module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
    
                # 遍历子模块，递归调用函数
                for child in module.children():
                    fn_recursive_feed_forward(child, chunk_size, dim)
    
            # 遍历当前实例的子模块，应用递归函数，禁用分块
            for module in self.children():
                fn_recursive_feed_forward(module, None, 0)
    
        # 从 diffusers.models.unets.unet_2d_condition 中复制的方法，设置默认注意力处理器
        def set_default_attn_processor(self):
            """
            禁用自定义注意力处理器并设置默认注意力实现。
            """
            # 检查所有注意力处理器是否为添加的 KV 处理器
            if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
                processor = AttnAddedKVProcessor()  # 设置为添加的 KV 处理器
            # 检查所有注意力处理器是否为交叉注意力处理器
            elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
                processor = AttnProcessor()  # 设置为普通注意力处理器
            else:
                # 抛出异常，若注意力处理器类型不符合预期
                raise ValueError(
                    f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
                )
    
            # 设置选定的注意力处理器
            self.set_attn_processor(processor)
    
        # 定义一个私有方法来设置模块的梯度检查点
        def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
            # 检查模块是否属于特定类型
            if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
                module.gradient_checkpointing = value  # 设置梯度检查点值
    
        # 从 diffusers.models.unets.unet_2d_condition 中复制的方法，启用自由度
    # 启用 FreeU 机制，参数为两个缩放因子和两个增强因子的值
    def enable_freeu(self, s1, s2, b1, b2):
        r"""从 https://arxiv.org/abs/2309.11497 启用 FreeU 机制。

        缩放因子的后缀表示它们应用的阶段块。

        请参考 [官方仓库](https://github.com/ChenyangSi/FreeU) 以获取在不同管道（如 Stable Diffusion v1、v2 和 Stable Diffusion XL）中已知效果良好的值组合。

        Args:
            s1 (`float`):
                第1阶段的缩放因子，用于减弱跳跃特征的贡献，以减轻增强去噪过程中的“过平滑效应”。
            s2 (`float`):
                第2阶段的缩放因子，用于减弱跳跃特征的贡献，以减轻增强去噪过程中的“过平滑效应”。
            b1 (`float`): 第1阶段的缩放因子，用于增强骨干特征的贡献。
            b2 (`float`): 第2阶段的缩放因子，用于增强骨干特征的贡献。
        """
        # 遍历上采样块，给每个块设置缩放因子和增强因子
        for i, upsample_block in enumerate(self.up_blocks):
            # 设置第1阶段的缩放因子
            setattr(upsample_block, "s1", s1)
            # 设置第2阶段的缩放因子
            setattr(upsample_block, "s2", s2)
            # 设置第1阶段的增强因子
            setattr(upsample_block, "b1", b1)
            # 设置第2阶段的增强因子
            setattr(upsample_block, "b2", b2)

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.disable_freeu 复制
    # 禁用 FreeU 机制
    def disable_freeu(self):
        """禁用 FreeU 机制。"""
        # 定义 FreeU 机制的关键属性
        freeu_keys = {"s1", "s2", "b1", "b2"}
        # 遍历上采样块
        for i, upsample_block in enumerate(self.up_blocks):
            # 遍历 FreeU 关键属性
            for k in freeu_keys:
                # 如果上采样块有该属性，或者该属性值不为 None
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    # 将属性值设置为 None，禁用 FreeU
                    setattr(upsample_block, k, None)

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections 复制
    # 启用融合的 QKV 投影
    def fuse_qkv_projections(self):
        """
        启用融合的 QKV 投影。对于自注意力模块，所有投影矩阵（即查询、键、值）都被融合。对于交叉注意力模块，键和值投影矩阵被融合。

        <Tip warning={true}>

        此 API 是 🧪 实验性的。

        </Tip>
        """
        # 保存原始的注意力处理器
        self.original_attn_processors = None

        # 遍历注意力处理器
        for _, attn_processor in self.attn_processors.items():
            # 如果注意力处理器的类名中包含“Added”
            if "Added" in str(attn_processor.__class__.__name__):
                # 抛出错误，表示不支持具有附加 KV 投影的模型
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        # 保存当前的注意力处理器
        self.original_attn_processors = self.attn_processors

        # 遍历所有模块
        for module in self.modules():
            # 如果模块是 Attention 类型
            if isinstance(module, Attention):
                # 融合投影
                module.fuse_projections(fuse=True)

        # 设置注意力处理器为融合的注意力处理器
        self.set_attn_processor(FusedAttnProcessor2_0())

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections 复制
    # 定义一个方法，用于禁用已启用的融合 QKV 投影
    def unfuse_qkv_projections(self):
        """禁用已启用的融合 QKV 投影。
    
        <Tip warning={true}>
    
        该 API 是 🧪 实验性的。
    
        </Tip>
    
        """
        # 如果存在原始的注意力处理器，则设置当前的注意力处理器为原始处理器
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)
    
    # 定义前向传播方法，接受多个参数进行计算
    def forward(
        self,
        sample: torch.Tensor,  # 输入样本，张量格式
        timestep: Union[torch.Tensor, float, int],  # 当前时间步，可以是张量、浮点数或整数
        encoder_hidden_states: torch.Tensor,  # 编码器的隐藏状态，张量格式
        class_labels: Optional[torch.Tensor] = None,  # 类别标签，默认为 None
        timestep_cond: Optional[torch.Tensor] = None,  # 时间步条件，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为 None
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 跨注意力的关键字参数，默认为 None
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,  # 降级块的附加残差，默认为 None
        mid_block_additional_residual: Optional[torch.Tensor] = None,  # 中间块的附加残差，默认为 None
        return_dict: bool = True,  # 是否返回字典格式的结果，默认为 True
```