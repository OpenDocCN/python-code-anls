# `.\diffusers\models\unets\unet_motion_model.py`

```
# 版权声明，表明该文件的所有权及相关使用条款
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 (“许可证”) 授权;
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则根据许可证分发的软件
# 是在“按原样”基础上分发的，不提供任何形式的保证或条件，
# 无论是明示还是暗示。
# 有关许可证所管辖的权限和限制，请参见许可证。
#
# 导入所需的库和模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Any, Dict, Optional, Tuple, Union  # 导入类型提示相关的类型

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
import torch.utils.checkpoint  # 导入 PyTorch 的检查点功能

# 导入自定义的配置和加载工具
from ...configuration_utils import ConfigMixin, FrozenDict, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin, UNet2DConditionLoadersMixin
from ...utils import BaseOutput, deprecate, is_torch_version, logging  # 导入常用的工具函数
from ...utils.torch_utils import apply_freeu  # 导入应用 FreeU 的工具函数
from ..attention import BasicTransformerBlock  # 导入基础变换器模块
from ..attention_processor import (  # 导入注意力处理器相关的类
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from ..embeddings import TimestepEmbedding, Timesteps  # 导入时间步嵌入相关的类
from ..modeling_utils import ModelMixin  # 导入模型混合工具
from ..resnet import Downsample2D, ResnetBlock2D, Upsample2D  # 导入 ResNet 相关的模块
from ..transformers.dual_transformer_2d import DualTransformer2DModel  # 导入双重变换器模型
from ..transformers.transformer_2d import Transformer2DModel  # 导入 2D 变换器模型
from .unet_2d_blocks import UNetMidBlock2DCrossAttn  # 导入 U-Net 中间块
from .unet_2d_condition import UNet2DConditionModel  # 导入条件 U-Net 模型

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，便于调试和日志输出

@dataclass
class UNetMotionOutput(BaseOutput):  # 定义 UNetMotionOutput 数据类，继承自 BaseOutput
    """
    [`UNetMotionOutput`] 的输出。

    参数：
        sample (`torch.Tensor` 的形状为 `(batch_size, num_channels, num_frames, height, width)`):
            基于 `encoder_hidden_states` 输入的隐藏状态输出。模型最后一层的输出。
    """

    sample: torch.Tensor  # 定义 sample 属性，类型为 torch.Tensor


class AnimateDiffTransformer3D(nn.Module):  # 定义 AnimateDiffTransformer3D 类，继承自 nn.Module
    """
    一个用于视频类数据的变换器模型。
    # 参数说明部分，描述初始化函数中每个参数的用途
    Parameters:
        # 多头注意力机制中头的数量，默认为16
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        # 每个头中的通道数，默认为88
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        # 输入和输出的通道数，如果输入是**连续**，则需要指定
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        # Transformer块的层数，默认为1
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        # dropout概率，默认为0.0
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        # 使用的`encoder_hidden_states`维度数
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        # 配置`TransformerBlock`的注意力是否包含偏置参数
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        # 潜在图像的宽度，如果输入是**离散**，则需要指定
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            # 该值在训练期间固定，用于学习位置嵌入的数量
            This is fixed during training since it is used to learn a number of position embeddings.
        # 前馈中的激活函数，默认为"geglu"
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward. See `diffusers.models.activations.get_activation` for supported
            activation functions.
        # 配置`TransformerBlock`是否使用可学习的逐元素仿射参数进行归一化
        norm_elementwise_affine (`bool`, *optional*):
            Configure if the `TransformerBlock` should use learnable elementwise affine parameters for normalization.
        # 配置每个`TransformerBlock`是否包含两个自注意力层
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
        # 应用到序列输入的位置信息嵌入的类型
        positional_embeddings: (`str`, *optional*):
            The type of positional embeddings to apply to the sequence input before passing use.
        # 应用位置嵌入的最大序列长度
        num_positional_embeddings: (`int`, *optional*):
            The maximum length of the sequence over which to apply positional embeddings.
    """

    # 初始化方法定义
    def __init__(
        # 多头注意力机制中头的数量，默认为16
        self,
        num_attention_heads: int = 16,
        # 每个头中的通道数，默认为88
        attention_head_dim: int = 88,
        # 输入通道数，可选
        in_channels: Optional[int] = None,
        # 输出通道数，可选
        out_channels: Optional[int] = None,
        # Transformer块的层数，默认为1
        num_layers: int = 1,
        # dropout概率，默认为0.0
        dropout: float = 0.0,
        # 归一化分组数，默认为32
        norm_num_groups: int = 32,
        # 使用的`encoder_hidden_states`维度数，可选
        cross_attention_dim: Optional[int] = None,
        # 注意力是否包含偏置参数，默认为False
        attention_bias: bool = False,
        # 潜在图像的宽度，可选
        sample_size: Optional[int] = None,
        # 前馈中的激活函数，默认为"geglu"
        activation_fn: str = "geglu",
        # 归一化是否使用可学习的逐元素仿射参数，默认为True
        norm_elementwise_affine: bool = True,
        # 每个`TransformerBlock`是否包含两个自注意力层，默认为True
        double_self_attention: bool = True,
        # 位置信息嵌入的类型，可选
        positional_embeddings: Optional[str] = None,
        # 应用位置嵌入的最大序列长度，可选
        num_positional_embeddings: Optional[int] = None,
    ):
        # 调用父类的构造函数以初始化父类的属性
        super().__init__()
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置每个注意力头的维度
        self.attention_head_dim = attention_head_dim
        # 计算内部维度，等于注意力头数量与每个注意力头维度的乘积
        inner_dim = num_attention_heads * attention_head_dim

        # 设置输入通道数
        self.in_channels = in_channels

        # 定义归一化层，使用组归一化，允许可学习的偏移
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        # 定义输入线性变换层，将输入通道映射到内部维度
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. 定义变换器块
        self.transformer_blocks = nn.ModuleList(
            [
                # 创建指定数量的基本变换器块
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=num_positional_embeddings,
                )
                # 遍历创建 num_layers 个基本变换器块
                for _ in range(num_layers)
            ]
        )

        # 定义输出线性变换层，将内部维度映射回输入通道数
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        # 定义前向传播方法的输入参数
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        encoder_hidden_states: Optional[torch.LongTensor] = None,  # 编码器的隐藏状态，默认为 None
        timestep: Optional[torch.LongTensor] = None,  # 时间步，默认为 None
        class_labels: Optional[torch.LongTensor] = None,  # 类标签，默认为 None
        num_frames: int = 1,  # 帧数，默认值为 1
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 跨注意力参数，默认为 None
    # 该方法用于 [`AnimateDiffTransformer3D`] 的前向传播
    
        ) -> torch.Tensor:
            """
            方法参数说明：
                hidden_states (`torch.LongTensor`): 输入的隐状态，形状为 `(batch size, num latent pixels)` 或 `(batch size, channel, height, width)` 
                encoder_hidden_states ( `torch.LongTensor`, *可选*): 
                    交叉注意力层的条件嵌入。如果未提供，交叉注意力将默认使用自注意力。
                timestep ( `torch.LongTensor`, *可选*): 
                    用于指示去噪步骤的时间戳。
                class_labels ( `torch.LongTensor`, *可选*): 
                    用于指示类别标签的条件嵌入。
                num_frames (`int`, *可选*, 默认为 1): 
                    每个批次处理的帧数，用于重新形状隐状态。
                cross_attention_kwargs (`dict`, *可选*): 
                    可选的关键字字典，传递给 `AttentionProcessor`。
            返回值：
                torch.Tensor: 
                    输出张量。
            """
            # 1. 输入
            # 获取输入隐状态的形状信息
            batch_frames, channel, height, width = hidden_states.shape
            # 计算批次大小
            batch_size = batch_frames // num_frames
    
            # 将隐状态保留用于残差连接
            residual = hidden_states
    
            # 调整隐状态的形状以适应批次和帧数
            hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
            # 调整维度顺序以便后续处理
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
    
            # 对隐状态进行规范化
            hidden_states = self.norm(hidden_states)
            # 再次调整维度顺序并重塑为适当的形状
            hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)
    
            # 输入层投影
            hidden_states = self.proj_in(hidden_states)
    
            # 2. 处理块
            # 遍历每个变换块以处理隐状态
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,  # 当前的隐状态
                    encoder_hidden_states=encoder_hidden_states,  # 可选的编码器隐状态
                    timestep=timestep,  # 可选的时间戳
                    cross_attention_kwargs=cross_attention_kwargs,  # 可选的交叉注意力参数
                    class_labels=class_labels,  # 可选的类标签
                )
    
            # 3. 输出
            # 输出层投影
            hidden_states = self.proj_out(hidden_states)
            # 调整输出张量的形状
            hidden_states = (
                hidden_states[None, None, :]  # 添加维度
                .reshape(batch_size, height, width, num_frames, channel)  # 重塑为适当形状
                .permute(0, 3, 4, 1, 2)  # 调整维度顺序
                .contiguous()  # 确保内存连续性
            )
            # 最终调整输出的形状
            hidden_states = hidden_states.reshape(batch_frames, channel, height, width)
    
            # 将残差添加到输出中以形成最终输出
            output = hidden_states + residual
            # 返回最终的输出张量
            return output
# 定义一个名为 DownBlockMotion 的类，继承自 nn.Module
class DownBlockMotion(nn.Module):
    # 初始化方法，定义多个参数，包括输入输出通道、dropout 率等
    def __init__(
        self,
        in_channels: int,  # 输入通道数量
        out_channels: int,  # 输出通道数量
        temb_channels: int,  # 时间嵌入通道数量
        dropout: float = 0.0,  # dropout 率，默认为 0
        num_layers: int = 1,  # 网络层数，默认为 1
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 参数
        resnet_time_scale_shift: str = "default",  # ResNet 时间尺度偏移
        resnet_act_fn: str = "swish",  # ResNet 激活函数，默认为 swish
        resnet_groups: int = 32,  # ResNet 组数，默认为 32
        resnet_pre_norm: bool = True,  # ResNet 是否使用预归一化
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_downsample: bool = True,  # 是否添加下采样层
        downsample_padding: int = 1,  # 下采样时的填充
        temporal_num_attention_heads: Union[int, Tuple[int]] = 1,  # 时间注意力头数
        temporal_cross_attention_dim: Optional[int] = None,  # 时间交叉注意力维度
        temporal_max_seq_length: int = 32,  # 最大序列长度
        temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,  # 每个块的变换器层数
        temporal_double_self_attention: bool = True,  # 是否双重自注意力
    ):
    # 前向传播方法，接收隐藏状态和时间嵌入等参数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        num_frames: int = 1,  # 帧数，默认为 1
        *args,  # 接受任意位置参数
        **kwargs,  # 接受任意关键字参数
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:  # 返回隐藏状态和输出状态的张量或元组
        # 检查位置参数或关键字参数中的 scale 是否被传递
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 定义弃用信息，提示用户 scale 参数将被忽略
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用弃用函数，发出警告
            deprecate("scale", "1.0.0", deprecation_message)

        # 初始化输出状态为一个空元组
        output_states = ()

        # 将 ResNet 和运动模块进行配对
        blocks = zip(self.resnets, self.motion_modules)
        # 遍历每对 ResNet 和运动模块
        for resnet, motion_module in blocks:
            # 如果处于训练模式且启用了梯度检查点
            if self.training and self.gradient_checkpointing:
                # 定义一个自定义前向传播函数
                def create_custom_forward(module):
                    def custom_forward(*inputs):  # 自定义前向函数，接受任意输入
                        return module(*inputs)  # 返回模块的输出

                    return custom_forward  # 返回自定义前向函数

                # 如果 PyTorch 版本大于等于 1.11.0
                if is_torch_version(">=", "1.11.0"):
                    # 使用检查点机制来节省内存
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),  # 创建的自定义前向函数
                        hidden_states,  # 输入的隐藏状态
                        temb,  # 输入的时间嵌入
                        use_reentrant=False,  # 不使用重入
                    )
                else:
                    # 在较早版本中也使用检查点机制
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )

            else:
                # 如果不是训练模式，直接通过 ResNet 处理隐藏状态
                hidden_states = resnet(hidden_states, temb)

            # 使用运动模块处理当前的隐藏状态
            hidden_states = motion_module(hidden_states, num_frames=num_frames)

            # 将当前隐藏状态添加到输出状态中
            output_states = output_states + (hidden_states,)

        # 如果下采样器不为空
        if self.downsamplers is not None:
            # 遍历每个下采样器
            for downsampler in self.downsamplers:
                # 通过下采样器处理隐藏状态
                hidden_states = downsampler(hidden_states)

            # 将下采样后的隐藏状态添加到输出状态中
            output_states = output_states + (hidden_states,)

        # 返回最终的隐藏状态和输出状态
        return hidden_states, output_states
    # 初始化方法，用于设置网络的参数
        def __init__(
            # 输入通道数量
            self,
            in_channels: int,
            # 输出通道数量
            out_channels: int,
            # 时间嵌入通道数量
            temb_channels: int,
            # dropout 概率，默认为 0.0
            dropout: float = 0.0,
            # 网络层数，默认为 1
            num_layers: int = 1,
            # 每个块中的变换器层数，默认为 1
            transformer_layers_per_block: Union[int, Tuple[int]] = 1,
            # ResNet 中的 epsilon 值，默认为 1e-6
            resnet_eps: float = 1e-6,
            # ResNet 时间尺度偏移，默认为 "default"
            resnet_time_scale_shift: str = "default",
            # ResNet 激活函数，默认为 "swish"
            resnet_act_fn: str = "swish",
            # ResNet 中的组数，默认为 32
            resnet_groups: int = 32,
            # 是否在 ResNet 中使用预归一化，默认为 True
            resnet_pre_norm: bool = True,
            # 注意力头的数量，默认为 1
            num_attention_heads: int = 1,
            # 交叉注意力维度，默认为 1280
            cross_attention_dim: int = 1280,
            # 输出缩放因子，默认为 1.0
            output_scale_factor: float = 1.0,
            # 下采样填充，默认为 1
            downsample_padding: int = 1,
            # 是否添加下采样层，默认为 True
            add_downsample: bool = True,
            # 是否使用双交叉注意力，默认为 False
            dual_cross_attention: bool = False,
            # 是否使用线性投影，默认为 False
            use_linear_projection: bool = False,
            # 是否仅使用交叉注意力，默认为 False
            only_cross_attention: bool = False,
            # 是否提升注意力计算精度，默认为 False
            upcast_attention: bool = False,
            # 注意力类型，默认为 "default"
            attention_type: str = "default",
            # 时间交叉注意力维度，可选参数
            temporal_cross_attention_dim: Optional[int] = None,
            # 时间注意力头数量，默认为 8
            temporal_num_attention_heads: int = 8,
            # 时间序列的最大长度，默认为 32
            temporal_max_seq_length: int = 32,
            # 时间变换器块中的层数，默认为 1
            temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,
            # 是否使用双重自注意力，默认为 True
            temporal_double_self_attention: bool = True,
        # 前向传播方法，定义如何通过模型传递数据
        def forward(
            # 隐藏状态张量，输入到模型中的主要数据
            self,
            hidden_states: torch.Tensor,
            # 可选的时间嵌入张量
            temb: Optional[torch.Tensor] = None,
            # 可选的编码器隐藏状态
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的注意力掩码
            attention_mask: Optional[torch.Tensor] = None,
            # 每次处理的帧数，默认为 1
            num_frames: int = 1,
            # 可选的编码器注意力掩码
            encoder_attention_mask: Optional[torch.Tensor] = None,
            # 可选的交叉注意力参数
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的额外残差连接
            additional_residuals: Optional[torch.Tensor] = None,
    ):
        # 检查 cross_attention_kwargs 是否不为空
        if cross_attention_kwargs is not None:
            # 检查 scale 参数是否存在，若存在则发出警告
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # 初始化输出状态为空元组
        output_states = ()

        # 将自残差网络、注意力模块和运动模块组合成一个列表
        blocks = list(zip(self.resnets, self.attentions, self.motion_modules))
        # 遍历组合后的模块及其索引
        for i, (resnet, attn, motion_module) in enumerate(blocks):
            # 如果处于训练状态且启用了梯度检查点
            if self.training and self.gradient_checkpointing:

                # 定义自定义前向传播函数
                def create_custom_forward(module, return_dict=None):
                    # 定义实际的前向传播逻辑
                    def custom_forward(*inputs):
                        # 根据 return_dict 的值选择返回方式
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                # 定义检查点参数字典，根据 PyTorch 版本设置
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                # 使用检查点机制计算隐藏状态
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                # 通过注意力模块处理隐藏状态
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                # 在非训练模式下直接通过残差网络处理隐藏状态
                hidden_states = resnet(hidden_states, temb)

                # 通过注意力模块处理隐藏状态
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            # 通过运动模块处理隐藏状态
            hidden_states = motion_module(
                hidden_states,
                num_frames=num_frames,
            )

            # 如果是最后一对模块且有额外残差，则将其应用到隐藏状态
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            # 将当前隐藏状态添加到输出状态中
            output_states = output_states + (hidden_states,)

        # 如果存在下采样模块，则依次应用它们
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            # 将下采样后的隐藏状态添加到输出状态中
            output_states = output_states + (hidden_states,)

        # 返回最终的隐藏状态和输出状态
        return hidden_states, output_states
# 定义一个继承自 nn.Module 的类，用于交叉注意力上采样块
class CrossAttnUpBlockMotion(nn.Module):
    # 初始化方法，设置各层的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        prev_output_channel: int,  # 前一层输出的通道数
        temb_channels: int,  # 时间嵌入通道数
        resolution_idx: Optional[int] = None,  # 分辨率索引，默认为 None
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # 层数
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,  # 每个块的变换器层数
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 值
        resnet_time_scale_shift: str = "default",  # ResNet 时间缩放偏移
        resnet_act_fn: str = "swish",  # ResNet 激活函数
        resnet_groups: int = 32,  # ResNet 组数
        resnet_pre_norm: bool = True,  # 是否在前面进行归一化
        num_attention_heads: int = 1,  # 注意力头的数量
        cross_attention_dim: int = 1280,  # 交叉注意力的维度
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_upsample: bool = True,  # 是否添加上采样
        dual_cross_attention: bool = False,  # 是否使用双重交叉注意力
        use_linear_projection: bool = False,  # 是否使用线性投影
        only_cross_attention: bool = False,  # 是否仅使用交叉注意力
        upcast_attention: bool = False,  # 是否上浮注意力
        attention_type: str = "default",  # 注意力类型
        temporal_cross_attention_dim: Optional[int] = None,  # 时间交叉注意力维度，默认为 None
        temporal_num_attention_heads: int = 8,  # 时间注意力头数量
        temporal_max_seq_length: int = 32,  # 时间序列的最大长度
        temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,  # 时间块的变换器层数
    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],  # 之前隐藏状态的元组
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 交叉注意力的可选参数
        upsample_size: Optional[int] = None,  # 可选的上采样大小
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 可选的编码器注意力掩码
        num_frames: int = 1,  # 帧数，默认为 1
# 定义一个继承自 nn.Module 的类，用于上采样块
class UpBlockMotion(nn.Module):
    # 初始化方法，设置各层的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        prev_output_channel: int,  # 前一层输出的通道数
        out_channels: int,  # 输出通道数
        temb_channels: int,  # 时间嵌入通道数
        resolution_idx: Optional[int] = None,  # 分辨率索引，默认为 None
        dropout: float = 0.0,  # dropout 概率
        num_layers: int = 1,  # 层数
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 值
        resnet_time_scale_shift: str = "default",  # ResNet 时间缩放偏移
        resnet_act_fn: str = "swish",  # ResNet 激活函数
        resnet_groups: int = 32,  # ResNet 组数
        resnet_pre_norm: bool = True,  # 是否在前面进行归一化
        output_scale_factor: float = 1.0,  # 输出缩放因子
        add_upsample: bool = True,  # 是否添加上采样
        temporal_cross_attention_dim: Optional[int] = None,  # 时间交叉注意力维度，默认为 None
        temporal_num_attention_heads: int = 8,  # 时间注意力头数量
        temporal_max_seq_length: int = 32,  # 时间序列的最大长度
        temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,  # 时间块的变换器层数
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化空列表，用于存放 ResNet 模块
        resnets = []
        # 初始化空列表，用于存放运动模块
        motion_modules = []

        # 支持每个时间块的变换层数量为变量
        if isinstance(temporal_transformer_layers_per_block, int):
            # 将单个整数转换为与层数相同的元组
            temporal_transformer_layers_per_block = (temporal_transformer_layers_per_block,) * num_layers
        elif len(temporal_transformer_layers_per_block) != num_layers:
            # 检查传入的层数是否与预期一致
            raise ValueError(
                f"temporal_transformer_layers_per_block must be an integer or a list of integers of length {num_layers}"
            )

        # 遍历每层，构建 ResNet 和运动模块
        for i in range(num_layers):
            # 设定跳过连接的通道数
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            # 设定当前层的输入通道数
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # 添加 ResNetBlock2D 模块到 resnets 列表
            resnets.append(
                ResnetBlock2D(
                    # 输入通道数为当前层的输入和跳过连接的通道数之和
                    in_channels=resnet_in_channels + res_skip_channels,
                    # 输出通道数设定
                    out_channels=out_channels,
                    # 时间嵌入通道数
                    temb_channels=temb_channels,
                    # 小常数以避免除零
                    eps=resnet_eps,
                    # 组归一化的组数
                    groups=resnet_groups,
                    # Dropout 率
                    dropout=dropout,
                    # 时间嵌入的归一化方式
                    time_embedding_norm=resnet_time_scale_shift,
                    # 激活函数设定
                    non_linearity=resnet_act_fn,
                    # 输出尺度因子
                    output_scale_factor=output_scale_factor,
                    # 是否使用预归一化
                    pre_norm=resnet_pre_norm,
                )
            )

            # 添加 AnimateDiffTransformer3D 模块到 motion_modules 列表
            motion_modules.append(
                AnimateDiffTransformer3D(
                    # 注意力头的数量
                    num_attention_heads=temporal_num_attention_heads,
                    # 输入通道数
                    in_channels=out_channels,
                    # 当前层的变换层数量
                    num_layers=temporal_transformer_layers_per_block[i],
                    # 组归一化的组数
                    norm_num_groups=resnet_groups,
                    # 跨注意力维度
                    cross_attention_dim=temporal_cross_attention_dim,
                    # 是否使用注意力偏置
                    attention_bias=False,
                    # 激活函数类型
                    activation_fn="geglu",
                    # 位置信息嵌入类型
                    positional_embeddings="sinusoidal",
                    # 位置信息嵌入数量
                    num_positional_embeddings=temporal_max_seq_length,
                    # 每个注意力头的维度
                    attention_head_dim=out_channels // temporal_num_attention_heads,
                )
            )

        # 将 ResNet 模块列表转换为 nn.ModuleList
        self.resnets = nn.ModuleList(resnets)
        # 将运动模块列表转换为 nn.ModuleList
        self.motion_modules = nn.ModuleList(motion_modules)

        # 如果需要上采样，则初始化上采样模块
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            # 否则，设定为 None
            self.upsamplers = None

        # 设定梯度检查点标志为 False
        self.gradient_checkpointing = False
        # 保存分辨率索引
        self.resolution_idx = resolution_idx

    def forward(
        # 前向传播方法的参数定义
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        # 可选的时间嵌入
        temb: Optional[torch.Tensor] = None,
        # 上采样大小
        upsample_size=None,
        # 帧数，默认为 1
        num_frames: int = 1,
        # 额外的参数
        *args,
        **kwargs,
    # 函数返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 检查传入参数是否存在或 "scale" 参数是否为非 None
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 定义弃用提示信息
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用 deprecate 函数记录弃用警告
            deprecate("scale", "1.0.0", deprecation_message)
    
        # 检查 FreeU 是否启用，确保相关属性均不为 None
        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )
    
        # 将自定义模块打包成元组，方便遍历
        blocks = zip(self.resnets, self.motion_modules)
    
        # 遍历每一对 resnet 和 motion_module
        for resnet, motion_module in blocks:
            # 从隐藏状态元组中弹出最后一个隐藏状态
            res_hidden_states = res_hidden_states_tuple[-1]
            # 更新隐藏状态元组，移除最后一个元素
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
    
            # 如果启用 FreeU，则仅对前两个阶段进行操作
            if is_freeu_enabled:
                # 应用 FreeU 函数获取新的隐藏状态
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )
    
            # 将当前隐藏状态和残差隐藏状态在维度 1 上拼接
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
    
            # 如果在训练模式并且启用了梯度检查点
            if self.training and self.gradient_checkpointing:
                # 定义创建自定义前向传播函数
                def create_custom_forward(module):
                    # 定义自定义前向传播的实现
                    def custom_forward(*inputs):
                        return module(*inputs)
    
                    return custom_forward
    
                # 如果 torch 版本大于等于 1.11.0
                if is_torch_version(">=", "1.11.0"):
                    # 使用检查点机制保存内存
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        use_reentrant=False,
                    )
                else:
                    # 否则使用旧版检查点机制
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                # 否则直接通过 resnet 计算隐藏状态
                hidden_states = resnet(hidden_states, temb)
    
            # 通过 motion_module 处理隐藏状态，传入帧数
            hidden_states = motion_module(hidden_states, num_frames=num_frames)
    
        # 如果存在上采样器，则对每个上采样器进行处理
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                # 通过上采样器处理隐藏状态，传入上采样大小
                hidden_states = upsampler(hidden_states, upsample_size)
    
        # 返回最终处理后的隐藏状态
        return hidden_states
# 定义 UNetMidBlockCrossAttnMotion 类，继承自 nn.Module
class UNetMidBlockCrossAttnMotion(nn.Module):
    # 初始化方法，定义类的参数
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        temb_channels: int,  # 时间嵌入通道数
        dropout: float = 0.0,  # Dropout 率
        num_layers: int = 1,  # 层数
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,  # 每个块的变换层数
        resnet_eps: float = 1e-6,  # ResNet 的 epsilon 值
        resnet_time_scale_shift: str = "default",  # ResNet 时间尺度偏移
        resnet_act_fn: str = "swish",  # ResNet 激活函数类型
        resnet_groups: int = 32,  # ResNet 组数
        resnet_pre_norm: bool = True,  # 是否进行前置归一化
        num_attention_heads: int = 1,  # 注意力头数量
        output_scale_factor: float = 1.0,  # 输出缩放因子
        cross_attention_dim: int = 1280,  # 交叉注意力维度
        dual_cross_attention: bool = False,  # 是否使用双重交叉注意力
        use_linear_projection: bool = False,  # 是否使用线性投影
        upcast_attention: bool = False,  # 是否上升注意力精度
        attention_type: str = "default",  # 注意力类型
        temporal_num_attention_heads: int = 1,  # 时间注意力头数量
        temporal_cross_attention_dim: Optional[int] = None,  # 时间交叉注意力维度
        temporal_max_seq_length: int = 32,  # 时间序列最大长度
        temporal_transformer_layers_per_block: Union[int, Tuple[int]] = 1,  # 时间块的变换层数
    # 前向传播方法，定义输入和输出
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态的输入张量
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 可选的编码器注意力掩码
        num_frames: int = 1,  # 帧数
    # 该函数的返回类型为 torch.Tensor
        ) -> torch.Tensor:
            # 检查交叉注意力参数是否不为 None
            if cross_attention_kwargs is not None:
                # 如果参数中包含 "scale"，发出警告，说明该参数已弃用
                if cross_attention_kwargs.get("scale", None) is not None:
                    logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
    
            # 通过第一个残差网络处理隐藏状态
            hidden_states = self.resnets[0](hidden_states, temb)
    
            # 将注意力层、残差网络和运动模块打包在一起
            blocks = zip(self.attentions, self.resnets[1:], self.motion_modules)
            # 遍历每个注意力层、残差网络和运动模块
            for attn, resnet, motion_module in blocks:
                # 如果在训练模式下并且启用了梯度检查点
                if self.training and self.gradient_checkpointing:
    
                    # 创建自定义前向函数
                    def create_custom_forward(module, return_dict=None):
                        # 定义自定义前向函数，接受任意输入
                        def custom_forward(*inputs):
                            # 如果返回字典不为 None，使用返回字典调用模块
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                # 否则直接调用模块
                                return module(*inputs)
    
                        return custom_forward
    
                    # 根据 PyTorch 版本设置检查点参数
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    # 调用注意力模块并获取输出的第一个元素
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    # 使用梯度检查点对运动模块进行前向传播
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    # 使用梯度检查点对残差网络进行前向传播
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                else:
                    # 在非训练模式下直接调用注意力模块
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    # 调用运动模块，传入隐藏状态和帧数
                    hidden_states = motion_module(
                        hidden_states,
                        num_frames=num_frames,
                    )
                    # 调用残差网络处理隐藏状态
                    hidden_states = resnet(hidden_states, temb)
    
            # 返回处理后的隐藏状态
            return hidden_states
# 定义一个继承自 nn.Module 的运动模块类
class MotionModules(nn.Module):
    # 初始化方法，接收多个参数配置运动模块
    def __init__(
        self,
        in_channels: int,  # 输入通道数
        layers_per_block: int = 2,  # 每个模块块的层数，默认是 2
        transformer_layers_per_block: Union[int, Tuple[int]] = 8,  # 每个块中的变换层数
        num_attention_heads: Union[int, Tuple[int]] = 8,  # 注意力头的数量
        attention_bias: bool = False,  # 是否使用注意力偏差
        cross_attention_dim: Optional[int] = None,  # 交叉注意力维度
        activation_fn: str = "geglu",  # 激活函数，默认使用 "geglu"
        norm_num_groups: int = 32,  # 归一化组的数量
        max_seq_length: int = 32,  # 最大序列长度
    ):
        # 调用父类初始化方法
        super().__init__()
        # 初始化运动模块列表
        self.motion_modules = nn.ModuleList([])

        # 如果变换层数是整数，重复为每个模块块填充
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = (transformer_layers_per_block,) * layers_per_block
        # 检查变换层数与块层数是否匹配
        elif len(transformer_layers_per_block) != layers_per_block:
            raise ValueError(
                f"The number of transformer layers per block must match the number of layers per block, "
                f"got {layers_per_block} and {len(transformer_layers_per_block)}"
            )

        # 遍历每个模块块
        for i in range(layers_per_block):
            # 向运动模块列表添加 AnimateDiffTransformer3D 实例
            self.motion_modules.append(
                AnimateDiffTransformer3D(
                    in_channels=in_channels,  # 输入通道数
                    num_layers=transformer_layers_per_block[i],  # 当前块的变换层数
                    norm_num_groups=norm_num_groups,  # 归一化组的数量
                    cross_attention_dim=cross_attention_dim,  # 交叉注意力维度
                    activation_fn=activation_fn,  # 激活函数
                    attention_bias=attention_bias,  # 注意力偏差
                    num_attention_heads=num_attention_heads,  # 注意力头数量
                    attention_head_dim=in_channels // num_attention_heads,  # 每个注意力头的维度
                    positional_embeddings="sinusoidal",  # 使用正弦波的位置嵌入
                    num_positional_embeddings=max_seq_length,  # 位置嵌入的数量
                )
            )


# 定义一个运动适配器类，结合多个混合类
class MotionAdapter(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    # 初始化方法，配置多个运动适配器参数
    def __init__(
        self,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),  # 块输出通道
        motion_layers_per_block: Union[int, Tuple[int]] = 2,  # 每个运动块的层数
        motion_transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple[int]]] = 1,  # 每个运动块中的变换层数
        motion_mid_block_layers_per_block: int = 1,  # 中间块的层数
        motion_transformer_layers_per_mid_block: Union[int, Tuple[int]] = 1,  # 中间块中的变换层数
        motion_num_attention_heads: Union[int, Tuple[int]] = 8,  # 中间块的注意力头数量
        motion_norm_num_groups: int = 32,  # 中间块的归一化组数量
        motion_max_seq_length: int = 32,  # 中间块的最大序列长度
        use_motion_mid_block: bool = True,  # 是否使用中间块
        conv_in_channels: Optional[int] = None,  # 输入通道数
    ):
        pass  # 前向传播方法，尚未实现


# 定义一个修改后的条件 2D UNet 模型
class UNetMotionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    r"""
    一个修改后的条件 2D UNet 模型，接收嘈杂样本、条件状态和时间步，返回形状输出。

    该模型继承自 [`ModelMixin`]。查看超类文档以获取所有模型的通用方法实现（如下载或保存）。
    """

    # 支持梯度检查点
    _supports_gradient_checkpointing = True

    @register_to_config
    # 初始化方法，用于创建类的实例
    def __init__(
        # 可选参数，样本大小，默认为 None
        self,
        sample_size: Optional[int] = None,
        # 输入通道数，默认为 4
        in_channels: int = 4,
        # 输出通道数，默认为 4
        out_channels: int = 4,
        # 下采样块的类型元组
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlockMotion",  # 第一个下采样块类型
            "CrossAttnDownBlockMotion",  # 第二个下采样块类型
            "CrossAttnDownBlockMotion",  # 第三个下采样块类型
            "DownBlockMotion",            # 第四个下采样块类型
        ),
        # 上采样块的类型元组
        up_block_types: Tuple[str, ...] = (
            "UpBlockMotion",              # 第一个上采样块类型
            "CrossAttnUpBlockMotion",    # 第二个上采样块类型
            "CrossAttnUpBlockMotion",    # 第三个上采样块类型
            "CrossAttnUpBlockMotion",    # 第四个上采样块类型
        ),
        # 块的输出通道数元组
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        # 每个块的层数，默认为 2
        layers_per_block: Union[int, Tuple[int]] = 2,
        # 下采样填充，默认为 1
        downsample_padding: int = 1,
        # 中间块的缩放因子，默认为 1
        mid_block_scale_factor: float = 1,
        # 激活函数类型，默认为 "silu"
        act_fn: str = "silu",
        # 归一化的组数，默认为 32
        norm_num_groups: int = 32,
        # 归一化的 epsilon 值，默认为 1e-5
        norm_eps: float = 1e-5,
        # 交叉注意力的维度，默认为 1280
        cross_attention_dim: int = 1280,
        # 每个块的变换器层数，默认为 1
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        # 可选参数，反向变换器层数，默认为 None
        reverse_transformer_layers_per_block: Optional[Union[int, Tuple[int], Tuple[Tuple]]] = None,
        # 时间变换器的层数，默认为 1
        temporal_transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        # 可选参数，反向时间变换器层数，默认为 None
        reverse_temporal_transformer_layers_per_block: Optional[Union[int, Tuple[int], Tuple[Tuple]]] = None,
        # 每个中间块的变换器层数，默认为 None
        transformer_layers_per_mid_block: Optional[Union[int, Tuple[int]]] = None,
        # 每个中间块的时间变换器层数，默认为 1
        temporal_transformer_layers_per_mid_block: Optional[Union[int, Tuple[int]]] = 1,
        # 是否使用线性投影，默认为 False
        use_linear_projection: bool = False,
        # 注意力头的数量，默认为 8
        num_attention_heads: Union[int, Tuple[int, ...]] = 8,
        # 动作最大序列长度，默认为 32
        motion_max_seq_length: int = 32,
        # 动作注意力头的数量，默认为 8
        motion_num_attention_heads: Union[int, Tuple[int, ...]] = 8,
        # 可选参数，反向动作注意力头的数量，默认为 None
        reverse_motion_num_attention_heads: Optional[Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]]] = None,
        # 是否使用动作中间块，默认为 True
        use_motion_mid_block: bool = True,
        # 中间块的层数，默认为 1
        mid_block_layers: int = 1,
        # 编码器隐藏层维度，默认为 None
        encoder_hid_dim: Optional[int] = None,
        # 编码器隐藏层类型，默认为 None
        encoder_hid_dim_type: Optional[str] = None,
        # 可选参数，附加嵌入类型，默认为 None
        addition_embed_type: Optional[str] = None,
        # 可选参数，附加时间嵌入维度，默认为 None
        addition_time_embed_dim: Optional[int] = None,
        # 可选参数，投影类别嵌入的输入维度，默认为 None
        projection_class_embeddings_input_dim: Optional[int] = None,
        # 可选参数，时间条件投影维度，默认为 None
        time_cond_proj_dim: Optional[int] = None,
    # 类方法，用于从 UNet2DConditionModel 创建对象
    @classmethod
    def from_unet2d(
        cls,
        # UNet2DConditionModel 对象
        unet: UNet2DConditionModel,
        # 可选的运动适配器，默认为 None
        motion_adapter: Optional[MotionAdapter] = None,
        # 是否加载权重，默认为 True
        load_weights: bool = True,
    # 冻结 UNet2DConditionModel 的权重，只保留运动模块可训练，便于微调
    def freeze_unet2d_params(self) -> None:
        """Freeze the weights of just the UNet2DConditionModel, and leave the motion modules
        unfrozen for fine tuning.
        """
        # 冻结所有参数
        for param in self.parameters():
            # 将参数的 requires_grad 属性设置为 False，禁止梯度更新
            param.requires_grad = False

        # 解冻运动模块
        for down_block in self.down_blocks:
            # 获取当前下采样块的运动模块
            motion_modules = down_block.motion_modules
            for param in motion_modules.parameters():
                # 将运动模块参数的 requires_grad 属性设置为 True，允许梯度更新
                param.requires_grad = True

        for up_block in self.up_blocks:
            # 获取当前上采样块的运动模块
            motion_modules = up_block.motion_modules
            for param in motion_modules.parameters():
                # 将运动模块参数的 requires_grad 属性设置为 True，允许梯度更新
                param.requires_grad = True

        # 检查中间块是否具有运动模块
        if hasattr(self.mid_block, "motion_modules"):
            # 获取中间块的运动模块
            motion_modules = self.mid_block.motion_modules
            for param in motion_modules.parameters():
                # 将运动模块参数的 requires_grad 属性设置为 True，允许梯度更新
                param.requires_grad = True

    # 加载运动模块的状态字典
    def load_motion_modules(self, motion_adapter: Optional[MotionAdapter]) -> None:
        # 遍历运动适配器的下采样块
        for i, down_block in enumerate(motion_adapter.down_blocks):
            # 加载下采样块的运动模块状态字典
            self.down_blocks[i].motion_modules.load_state_dict(down_block.motion_modules.state_dict())
        # 遍历运动适配器的上采样块
        for i, up_block in enumerate(motion_adapter.up_blocks):
            # 加载上采样块的运动模块状态字典
            self.up_blocks[i].motion_modules.load_state_dict(up_block.motion_modules.state_dict())

        # 支持没有中间块的旧运动模块
        if hasattr(self.mid_block, "motion_modules"):
            # 加载中间块的运动模块状态字典
            self.mid_block.motion_modules.load_state_dict(motion_adapter.mid_block.motion_modules.state_dict())

    # 保存运动模块的状态
    def save_motion_modules(
        self,
        save_directory: str,
        is_main_process: bool = True,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> None:
        # 获取当前模型的状态字典
        state_dict = self.state_dict()

        # 提取所有运动模块的状态
        motion_state_dict = {}
        for k, v in state_dict.items():
            # 筛选出包含 "motion_modules" 的键值对
            if "motion_modules" in k:
                motion_state_dict[k] = v

        # 创建运动适配器实例
        adapter = MotionAdapter(
            block_out_channels=self.config["block_out_channels"],
            motion_layers_per_block=self.config["layers_per_block"],
            motion_norm_num_groups=self.config["norm_num_groups"],
            motion_num_attention_heads=self.config["motion_num_attention_heads"],
            motion_max_seq_length=self.config["motion_max_seq_length"],
            use_motion_mid_block=self.config["use_motion_mid_block"],
        )
        # 加载运动状态字典
        adapter.load_state_dict(motion_state_dict)
        # 保存适配器的预训练状态
        adapter.save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            safe_serialization=safe_serialization,
            variant=variant,
            push_to_hub=push_to_hub,
            **kwargs,
        )

    @property
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors 复制的属性
    # 定义一个方法，返回模型中所有注意力处理器的字典
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回值:
            `dict` 类型的注意力处理器: 包含模型中所有注意力处理器的字典，
            按照其权重名称索引。
        """
        # 初始化一个空字典，用于存储注意力处理器
        processors = {}

        # 定义一个递归函数，用于添加注意力处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否具有 'get_processor' 方法
            if hasattr(module, "get_processor"):
                # 将处理器添加到字典中，键为处理器名称
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用，继续添加子模块的处理器
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            # 返回处理器字典
            return processors

        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数添加所有处理器
            fn_recursive_add_processors(name, module, processors)

        # 返回最终的处理器字典
        return processors

    # 从 diffusers.models.unets.unet_2d_condition 中复制的方法，用于设置注意力处理器
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的注意力处理器。

        参数:
            processor (`dict` of `AttentionProcessor` 或仅 `AttentionProcessor`):
                实例化的处理器类或处理器类的字典，将被设置为**所有** `Attention` 层的处理器。

                如果 `processor` 是字典，键需要定义相应的交叉注意力处理器的路径。
                当设置可训练的注意力处理器时，强烈推荐这样做。

        """
        # 获取当前注意力处理器字典的键数量
        count = len(self.attn_processors.keys())

        # 检查传入的处理器字典长度是否与注意力层数量匹配
        if isinstance(processor, dict) and len(processor) != count:
            # 如果不匹配，抛出错误
            raise ValueError(
                f"传入了处理器字典，但处理器数量 {len(processor)} 与"
                f" 注意力层数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
            )

        # 定义一个递归函数，用于设置注意力处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查模块是否具有 'set_processor' 方法
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，直接设置处理器
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中弹出对应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用，继续设置子模块的处理器
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数设置所有处理器
            fn_recursive_attn_processor(name, module, processor)
    # 定义一个方法以启用前向分块处理
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        设置注意力处理器以使用[前馈分块](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers)。

        参数：
            chunk_size (`int`, *可选*):
                前馈层的块大小。如果未指定，将单独对维度为`dim`的每个张量运行前馈层。
            dim (`int`, *可选*, 默认为`0`):
                前馈计算应分块的维度。选择dim=0（批次）或dim=1（序列长度）。
        """
        # 检查dim参数是否在有效范围内（0或1）
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # 默认块大小为1
        chunk_size = chunk_size or 1

        # 定义递归前馈函数以设置模块的分块前馈处理
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            # 如果模块有set_chunk_feed_forward属性，设置块大小和维度
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            # 遍历模块的子模块
            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        # 遍历当前对象的子模块，应用递归前馈函数
        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # 定义一个方法以禁用前向分块处理
    def disable_forward_chunking(self) -> None:
        # 定义递归前馈函数以设置模块的分块前馈处理为None
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            # 如果模块有set_chunk_feed_forward属性，设置块大小和维度为None
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            # 遍历模块的子模块
            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        # 遍历当前对象的子模块，应用递归前馈函数
        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    # 从diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor复制的方法
    def set_default_attn_processor(self) -> None:
        """
        禁用自定义注意力处理器并设置默认的注意力实现。
        """
        # 如果所有注意力处理器都是ADDED_KV_ATTENTION_PROCESSORS类型
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 设置处理器为AttnAddedKVProcessor
            processor = AttnAddedKVProcessor()
        # 如果所有注意力处理器都是CROSS_ATTENTION_PROCESSORS类型
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            # 设置处理器为AttnProcessor
            processor = AttnProcessor()
        else:
            # 抛出错误，表示不能在不匹配的注意力处理器类型下调用该方法
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        # 设置当前对象的注意力处理器
        self.set_attn_processor(processor)

    # 定义一个方法以设置模块的梯度检查点
    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        # 检查模块是否为特定类型，如果是则设置其梯度检查点属性
        if isinstance(module, (CrossAttnDownBlockMotion, DownBlockMotion, CrossAttnUpBlockMotion, UpBlockMotion)):
            module.gradient_checkpointing = value

    # 从diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.enable_freeu复制的方法
    # 启用 FreeU 机制，接受四个浮点型缩放因子作为参数
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float) -> None:
        # 文档字符串，描述该方法的作用及参数含义
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        # 遍历上采样块，并为每个块设置缩放因子
        for i, upsample_block in enumerate(self.up_blocks):
            # 为上采样块设置阶段1的缩放因子
            setattr(upsample_block, "s1", s1)
            # 为上采样块设置阶段2的缩放因子
            setattr(upsample_block, "s2", s2)
            # 为上采样块设置阶段1的主干特征缩放因子
            setattr(upsample_block, "b1", b1)
            # 为上采样块设置阶段2的主干特征缩放因子
            setattr(upsample_block, "b2", b2)

    # 禁用 FreeU 机制
    def disable_freeu(self) -> None:
        # 文档字符串，描述该方法的作用
        """Disables the FreeU mechanism."""
        # 定义 FreeU 相关的键名集合
        freeu_keys = {"s1", "s2", "b1", "b2"}
        # 遍历上采样块
        for i, upsample_block in enumerate(self.up_blocks):
            # 遍历 FreeU 键名
            for k in freeu_keys:
                # 检查上采样块是否具有该属性或该属性是否不为 None
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    # 将上采样块的该属性设置为 None
                    setattr(upsample_block, k, None)

    # 启用融合的 QKV 投影
    def fuse_qkv_projections(self):
        # 文档字符串，描述该方法的作用
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        # 初始化原始注意力处理器为 None
        self.original_attn_processors = None

        # 遍历注意力处理器
        for _, attn_processor in self.attn_processors.items():
            # 检查注意力处理器类名中是否包含 "Added"
            if "Added" in str(attn_processor.__class__.__name__):
                # 抛出异常，说明不支持该操作
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        # 保存原始的注意力处理器
        self.original_attn_processors = self.attn_processors

        # 遍历所有模块
        for module in self.modules():
            # 检查模块是否为 Attention 类型
            if isinstance(module, Attention):
                # 融合投影
                module.fuse_projections(fuse=True)

        # 设置融合后的注意力处理器
        self.set_attn_processor(FusedAttnProcessor2_0())

    # 解融合 QKV 投影的方法（省略具体实现）
    # 定义一个禁用融合 QKV 投影的方法
    def unfuse_qkv_projections(self):
        """如果启用了，禁用融合 QKV 投影。
    
        <Tip warning={true}>
        
        此 API 是 🧪 实验性。
        
        </Tip>
    
        """
        # 检查原始注意力处理器是否不为 None
        if self.original_attn_processors is not None:
            # 设置当前注意力处理器为原始的注意力处理器
            self.set_attn_processor(self.original_attn_processors)
    
    # 定义前向传播方法，接收多个参数
    def forward(
        self,
        # 输入样本张量
        sample: torch.Tensor,
        # 时间步，可以是张量、浮点数或整数
        timestep: Union[torch.Tensor, float, int],
        # 编码器隐藏状态张量
        encoder_hidden_states: torch.Tensor,
        # 可选的时间步条件张量
        timestep_cond: Optional[torch.Tensor] = None,
        # 可选的注意力掩码张量
        attention_mask: Optional[torch.Tensor] = None,
        # 可选的交叉注意力参数字典
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # 可选的附加条件参数字典
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        # 可选的下块附加残差元组
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        # 可选的中间块附加残差张量
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        # 是否返回字典格式的结果，默认为 True
        return_dict: bool = True,
```