# `.\diffusers\models\transformers\transformer_temporal.py`

```py
# 版权声明，2024年HuggingFace团队所有，保留所有权利。
# 
# 根据Apache许可证第2.0版（"许可证"）授权；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下地址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，
# 否则根据许可证分发的软件是在"按原样"的基础上提供的，
# 不提供任何形式的明示或暗示的保证或条件。
# 有关许可证具体条款的信息，
# 请参阅许可证中的权限和限制。
# 导入dataclass装饰器，用于简化类的定义
from dataclasses import dataclass
# 导入Any、Dict和Optional类型，用于类型注解
from typing import Any, Dict, Optional

# 导入PyTorch库
import torch
# 从torch库中导入神经网络模块
from torch import nn

# 从配置工具中导入ConfigMixin类和注册配置函数
from ...configuration_utils import ConfigMixin, register_to_config
# 从工具模块中导入BaseOutput类
from ...utils import BaseOutput
# 从注意力模块中导入基本变换器块和时间基本变换器块
from ..attention import BasicTransformerBlock, TemporalBasicTransformerBlock
# 从嵌入模块中导入时间步嵌入和时间步类
from ..embeddings import TimestepEmbedding, Timesteps
# 从模型工具中导入ModelMixin类
from ..modeling_utils import ModelMixin
# 从ResNet模块中导入AlphaBlender类
from ..resnet import AlphaBlender

# 定义TransformerTemporalModelOutput类，继承自BaseOutput
@dataclass
class TransformerTemporalModelOutput(BaseOutput):
    """
    [`TransformerTemporalModel`]的输出。

    参数：
        sample (`torch.Tensor`形状为`(batch_size x num_frames, num_channels, height, width)`):
            基于`encoder_hidden_states`输入条件的隐藏状态输出。
    """

    # 定义sample属性，类型为torch.Tensor
    sample: torch.Tensor

# 定义TransformerTemporalModel类，继承自ModelMixin和ConfigMixin
class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """
    适用于视频类数据的变换器模型。
    # 参数说明
    Parameters:
        # 多头注意力的头数，默认为 16
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        # 每个头中的通道数，默认为 88
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        # 输入和输出的通道数（如果输入为 **连续**，则需要指定）
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        # Transformer 块的层数，默认为 1
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        # 使用的 dropout 概率，默认为 0.0
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        # 使用的 `encoder_hidden_states` 维度数
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        # 配置 `TransformerBlock` 的注意力是否应包含偏置参数
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        # 潜在图像的宽度（如果输入为 **离散**，则需要指定）
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            # 在训练期间固定使用，以学习位置嵌入的数量
            This is fixed during training since it is used to learn a number of position embeddings.
        # 前馈中使用的激活函数，默认为 "geglu"
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward. See `diffusers.models.activations.get_activation` for supported
            activation functions.
        # 配置 `TransformerBlock` 是否应使用可学习的逐元素仿射参数进行归一化
        norm_elementwise_affine (`bool`, *optional*):
            Configure if the `TransformerBlock` should use learnable elementwise affine parameters for normalization.
        # 配置每个 `TransformerBlock` 是否应包含两个自注意力层
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
        # 应用到序列输入之前的位置信息嵌入类型
        positional_embeddings: (`str`, *optional*):
            The type of positional embeddings to apply to the sequence input before passing use.
        # 应用位置嵌入的最大序列长度
        num_positional_embeddings: (`int`, *optional*):
            The maximum length of the sequence over which to apply positional embeddings.
    """

    # 注册到配置
    @register_to_config
    def __init__(
        # 初始化函数的参数
        self,
        # 多头注意力的头数，默认为 16
        num_attention_heads: int = 16,
        # 每个头中的通道数，默认为 88
        attention_head_dim: int = 88,
        # 输入通道数（可选）
        in_channels: Optional[int] = None,
        # 输出通道数（可选）
        out_channels: Optional[int] = None,
        # Transformer 块的层数，默认为 1
        num_layers: int = 1,
        # dropout 概率，默认为 0.0
        dropout: float = 0.0,
        # 归一化组数，默认为 32
        norm_num_groups: int = 32,
        # 使用的 `encoder_hidden_states` 维度数（可选）
        cross_attention_dim: Optional[int] = None,
        # 注意力是否包含偏置参数，默认为 False
        attention_bias: bool = False,
        # 潜在图像的宽度（可选）
        sample_size: Optional[int] = None,
        # 激活函数，默认为 "geglu"
        activation_fn: str = "geglu",
        # 是否使用可学习的逐元素仿射参数，默认为 True
        norm_elementwise_affine: bool = True,
        # 是否包含两个自注意力层，默认为 True
        double_self_attention: bool = True,
        # 位置信息嵌入类型（可选）
        positional_embeddings: Optional[str] = None,
        # 最大序列长度（可选）
        num_positional_embeddings: Optional[int] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置每个注意力头的维度
        self.attention_head_dim = attention_head_dim
        # 计算内部维度，即注意力头数量与每个注意力头维度的乘积
        inner_dim = num_attention_heads * attention_head_dim

        # 设置输入通道数
        self.in_channels = in_channels

        # 定义组归一化层，指定组数量、通道数、稳定数和是否启用仿射变换
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        # 定义输入线性变换层，映射输入通道到内部维度
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. 定义变换器块
        self.transformer_blocks = nn.ModuleList(
            [
                # 创建基本变换器块，并传入相应参数
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
                # 重复创建多个变换器块，数量由 num_layers 决定
                for d in range(num_layers)
            ]
        )

        # 定义输出线性变换层，将内部维度映射回输入通道数
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def forward(
        # 定义前向传播方法的输入参数
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.LongTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: torch.LongTensor = None,
        num_frames: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
# 定义一个用于视频类数据的 Transformer 模型
class TransformerSpatioTemporalModel(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    # 初始化函数，设置模型参数
    def __init__(
        self,
        num_attention_heads: int = 16,  # 多头注意力机制的头数，默认为16
        attention_head_dim: int = 88,  # 每个头的通道数，默认为88
        in_channels: int = 320,  # 输入和输出的通道数，默认为320
        out_channels: Optional[int] = None,  # 输出的通道数，如果输入是连续的则需要指定
        num_layers: int = 1,  # Transformer 块的层数，默认为1
        cross_attention_dim: Optional[int] = None,  # 编码器隐藏状态的维度
    ):
        super().__init__()  # 调用父类的初始化方法
        self.num_attention_heads = num_attention_heads  # 保存多头注意力的头数
        self.attention_head_dim = attention_head_dim  # 保存每个头的通道数

        inner_dim = num_attention_heads * attention_head_dim  # 计算内部维度
        self.inner_dim = inner_dim  # 保存内部维度

        # 2. 定义输入层
        self.in_channels = in_channels  # 保存输入通道数
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)  # 定义分组归一化层
        self.proj_in = nn.Linear(in_channels, inner_dim)  # 定义输入的线性变换

        # 3. 定义 Transformer 块
        self.transformer_blocks = nn.ModuleList(  # 创建 Transformer 块的模块列表
            [
                BasicTransformerBlock(  # 实例化基本的 Transformer 块
                    inner_dim,  # 传入内部维度
                    num_attention_heads,  # 传入多头注意力的头数
                    attention_head_dim,  # 传入每个头的通道数
                    cross_attention_dim=cross_attention_dim,  # 传入交叉注意力维度
                )
                for d in range(num_layers)  # 根据层数创建多个块
            ]
        )

        time_mix_inner_dim = inner_dim  # 定义时间混合内部维度
        self.temporal_transformer_blocks = nn.ModuleList(  # 创建时间 Transformer 块的模块列表
            [
                TemporalBasicTransformerBlock(  # 实例化时间基本 Transformer 块
                    inner_dim,  # 传入内部维度
                    time_mix_inner_dim,  # 传入时间混合内部维度
                    num_attention_heads,  # 传入多头注意力的头数
                    attention_head_dim,  # 传入每个头的通道数
                    cross_attention_dim=cross_attention_dim,  # 传入交叉注意力维度
                )
                for _ in range(num_layers)  # 根据层数创建多个块
            ]
        )

        time_embed_dim = in_channels * 4  # 定义时间嵌入的维度
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)  # 创建时间步嵌入
        self.time_proj = Timesteps(in_channels, True, 0)  # 定义时间投影
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy="learned_with_images")  # 定义时间混合器

        # 4. 定义输出层
        self.out_channels = in_channels if out_channels is None else out_channels  # 确定输出通道数
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)  # 定义输出的线性变换

        self.gradient_checkpointing = False  # 是否启用梯度检查点，默认为False
    # 定义一个名为 forward 的方法
        def forward(
            # 输入参数 hidden_states，类型为 torch.Tensor
            self,
            hidden_states: torch.Tensor,
            # 可选输入参数 encoder_hidden_states，类型为 torch.Tensor，默认为 None
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选输入参数 image_only_indicator，类型为 torch.Tensor，默认为 None
            image_only_indicator: Optional[torch.Tensor] = None,
            # 可选输入参数 return_dict，类型为 bool，默认为 True
            return_dict: bool = True,
```