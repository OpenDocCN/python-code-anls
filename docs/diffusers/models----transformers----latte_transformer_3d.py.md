# `.\diffusers\models\transformers\latte_transformer_3d.py`

```py
# 版权声明，表明该代码的版权属于 Latte 团队和 HuggingFace 团队
# Copyright 2024 the Latte Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版（"许可证"）进行授权；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则根据许可证分发的软件是按 "原样" 基础进行的，
# 不提供任何形式的担保或条件，无论是明示的还是暗示的。
# 请参见许可证以了解有关特定语言的权限和限制。
from typing import Optional  # 从 typing 模块导入 Optional 类型，用于指示可选参数

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块

# 从配置工具导入 ConfigMixin 和注册配置的功能
from ...configuration_utils import ConfigMixin, register_to_config
# 导入与图像嵌入相关的类和函数
from ...models.embeddings import PixArtAlphaTextProjection, get_1d_sincos_pos_embed_from_grid
# 导入基础变换器块的定义
from ..attention import BasicTransformerBlock
# 导入图像块嵌入的定义
from ..embeddings import PatchEmbed
# 导入 Transformer 2D 模型输出的定义
from ..modeling_outputs import Transformer2DModelOutput
# 导入模型混合功能的定义
from ..modeling_utils import ModelMixin
# 导入自适应层归一化的定义
from ..normalization import AdaLayerNormSingle


# 定义一个 3D Transformer 模型类，继承自 ModelMixin 和 ConfigMixin
class LatteTransformer3DModel(ModelMixin, ConfigMixin):
    # 设置支持梯度检查点的标志为 True
    _supports_gradient_checkpointing = True

    """
    一个用于视频类数据的 3D Transformer 模型，相关论文链接：
    https://arxiv.org/abs/2401.03048，官方代码地址：
    https://github.com/Vchitect/Latte
    """
    # 参数说明
    Parameters:
        # 使用的多头注意力头数，默认为16
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        # 每个头的通道数，默认为88
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        # 输入通道数
        in_channels (`int`, *optional*):
            The number of channels in the input.
        # 输出通道数
        out_channels (`int`, *optional*):
            The number of channels in the output.
        # Transformer块的层数，默认为1
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        # dropout概率，默认为0.0
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        # 用于cross attention的编码器隐藏状态维度
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        # 配置TransformerBlocks的注意力是否包含偏置参数
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
        # 潜在图像的宽度（如果输入为离散类型，需指定）
        sample_size (`int`, *optional*): 
            The width of the latent images (specify if the input is **discrete**).
            # 在训练期间固定，用于学习位置嵌入数量。
        patch_size (`int`, *optional*): 
            # 在补丁嵌入层中使用的补丁大小。
            The size of the patches to use in the patch embedding layer.
        # 前馈中的激活函数，默认为"geglu"
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        # 训练期间使用的扩散步骤数
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states. During inference, you can denoise for up to but not more steps than
            `num_embeds_ada_norm`.
        # 使用的归一化类型，选项为"layer_norm"或"ada_layer_norm"
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of normalization to use. Options are `"layer_norm"` or `"ada_layer_norm"`.
        # 是否在归一化层中使用逐元素仿射，默认为True
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        # 归一化层中使用的epsilon值，默认为1e-5
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use in normalization layers.
        # 标注嵌入的通道数
        caption_channels (`int`, *optional*):
            The number of channels in the caption embeddings.
        # 视频类数据中的帧数
        video_length (`int`, *optional*):
            The number of frames in the video-like data.
    """ 
    # 注册配置的装饰器
    @register_to_config
    # 初始化方法，用于设置模型的参数
        def __init__(
            # 注意力头的数量，默认为16
            num_attention_heads: int = 16,
            # 每个注意力头的维度，默认为88
            attention_head_dim: int = 88,
            # 输入通道数，默认为None
            in_channels: Optional[int] = None,
            # 输出通道数，默认为None
            out_channels: Optional[int] = None,
            # 网络层数，默认为1
            num_layers: int = 1,
            # dropout比率，默认为0.0
            dropout: float = 0.0,
            # 跨注意力的维度，默认为None
            cross_attention_dim: Optional[int] = None,
            # 是否使用注意力偏置，默认为False
            attention_bias: bool = False,
            # 样本大小，默认为64
            sample_size: int = 64,
            # 每个patch的大小，默认为None
            patch_size: Optional[int] = None,
            # 激活函数，默认为"geglu"
            activation_fn: str = "geglu",
            # 自适应归一化的嵌入数量，默认为None
            num_embeds_ada_norm: Optional[int] = None,
            # 归一化类型，默认为"layer_norm"
            norm_type: str = "layer_norm",
            # 归一化是否进行逐元素仿射，默认为True
            norm_elementwise_affine: bool = True,
            # 归一化的epsilon值，默认为1e-5
            norm_eps: float = 1e-5,
            # caption的通道数，默认为None
            caption_channels: int = None,
            # 视频长度，默认为16
            video_length: int = 16,
        # 设置梯度检查点的函数，接收一个模块和一个布尔值
        def _set_gradient_checkpointing(self, module, value=False):
            # 将梯度检查点设置为给定的布尔值
            self.gradient_checkpointing = value
    
        # 前向传播方法，定义模型的前向计算
        def forward(
            # 输入的隐藏状态，类型为torch.Tensor
            hidden_states: torch.Tensor,
            # 可选的时间步长，类型为torch.LongTensor
            timestep: Optional[torch.LongTensor] = None,
            # 可选的编码器隐藏状态，类型为torch.Tensor
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的编码器注意力掩码，类型为torch.Tensor
            encoder_attention_mask: Optional[torch.Tensor] = None,
            # 是否启用时间注意力，默认为True
            enable_temporal_attentions: bool = True,
            # 是否返回字典形式的输出，默认为True
            return_dict: bool = True,
```