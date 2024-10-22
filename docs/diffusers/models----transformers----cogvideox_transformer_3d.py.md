# `.\diffusers\models\transformers\cogvideox_transformer_3d.py`

```
# 版权声明，说明代码的版权所有者和使用许可
# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# 所有权利保留。
#
# 根据 Apache 许可证，第 2.0 版（"许可证"）进行授权；
# 除非遵循许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，
# 根据许可证分发的软件是按“原样”提供的，不附带任何明示或暗示的担保或条件。
# 有关许可证下特定权限和限制的信息，请参阅许可证。

# 从 typing 模块导入所需的类型
from typing import Any, Dict, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 PyTorch 导入神经网络模块
from torch import nn

# 导入其他模块中的工具和类
from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import is_torch_version, logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention import Attention, FeedForward
from ..attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from ..embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNorm, CogVideoXLayerNormZero

# 创建日志记录器，以便在模块中记录信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 使用装饰器，允许在图计算中可能的功能
@maybe_allow_in_graph
# 定义一个名为 CogVideoXBlock 的类，继承自 nn.Module
class CogVideoXBlock(nn.Module):
    r"""
    在 [CogVideoX](https://github.com/THUDM/CogVideo) 模型中使用的 Transformer 块。
    # 定义函数参数的文档字符串，描述各个参数的用途
    Parameters:
        dim (`int`):  # 输入和输出的通道数
            The number of channels in the input and output.
        num_attention_heads (`int`):  # 多头注意力中使用的头数
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):  # 每个头的通道数
            The number of channels in each head.
        time_embed_dim (`int`):  # 时间步嵌入的通道数
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):  # 使用的丢弃概率
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):  # 前馈网络中使用的激活函数
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):  # 是否在注意力投影层使用偏置
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):  # 是否在注意力中查询和键的投影后使用归一化
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):  # 是否使用可学习的逐元素仿射参数进行归一化
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):  # 归一化层的 epsilon 值
            Epsilon value for normalization layers.
        final_dropout (`bool`, defaults to `False`):  # 是否在最后的前馈层后应用最终的丢弃
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):  # 前馈层的自定义隐藏维度
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):  # 是否在前馈层中使用偏置
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):  # 是否在注意力输出投影层中使用偏置
            Whether or not to use bias in Attention output projection layer.
    """  # 结束文档字符串

    def __init__(  # 定义构造函数
        self,  # 实例自身
        dim: int,  # 输入和输出通道数
        num_attention_heads: int,  # 多头注意力中头数
        attention_head_dim: int,  # 每个头的通道数
        time_embed_dim: int,  # 时间步嵌入通道数
        dropout: float = 0.0,  # 默认丢弃概率
        activation_fn: str = "gelu-approximate",  # 默认激活函数
        attention_bias: bool = False,  # 默认不使用注意力偏置
        qk_norm: bool = True,  # 默认使用查询和键的归一化
        norm_elementwise_affine: bool = True,  # 默认使用逐元素仿射参数
        norm_eps: float = 1e-5,  # 默认归一化的 epsilon 值
        final_dropout: bool = True,  # 默认使用最终丢弃
        ff_inner_dim: Optional[int] = None,  # 前馈层的可选隐藏维度
        ff_bias: bool = True,  # 默认使用前馈层的偏置
        attention_out_bias: bool = True,  # 默认使用注意力输出层的偏置
    ):
        # 调用父类初始化方法
        super().__init__()

        # 1. Self Attention
        # 创建归一化层，处理时间嵌入维度和特征维度
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        # 创建自注意力机制，配置查询维度和头数等参数
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        # 创建另一个归一化层，用于后续的前馈网络
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        # 创建前馈网络，配置隐藏层维度及其他超参数
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # 获取编码器隐藏状态的序列长度
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        # 对输入的隐藏状态和编码器状态进行归一化和调制
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        # 执行自注意力机制，计算新的隐藏状态
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # 更新隐藏状态和编码器隐藏状态，结合注意力输出
        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        # 再次进行归一化和调制
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        # 将归一化后的隐藏状态和编码器状态连接，输入前馈网络
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        # 更新隐藏状态和编码器状态，结合前馈网络输出
        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        # 返回更新后的隐藏状态和编码器状态
        return hidden_states, encoder_hidden_states
# 定义一个用于视频数据的 Transformer 模型，继承自 ModelMixin 和 ConfigMixin
class CogVideoXTransformer3DModel(ModelMixin, ConfigMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).
    """

    # 设置支持梯度检查点
    _supports_gradient_checkpointing = True

    # 注册到配置中的初始化方法，定义多个超参数
    @register_to_config
    def __init__(
        # 注意力头的数量，默认值为 30
        num_attention_heads: int = 30,
        # 每个注意力头的维度，默认值为 64
        attention_head_dim: int = 64,
        # 输入通道的数量，默认值为 16
        in_channels: int = 16,
        # 输出通道的数量，可选，默认值为 16
        out_channels: Optional[int] = 16,
        # 是否翻转正弦到余弦，默认值为 True
        flip_sin_to_cos: bool = True,
        # 频率偏移量，默认值为 0
        freq_shift: int = 0,
        # 时间嵌入维度，默认值为 512
        time_embed_dim: int = 512,
        # 文本嵌入维度，默认值为 4096
        text_embed_dim: int = 4096,
        # 层的数量，默认值为 30
        num_layers: int = 30,
        # dropout 概率，默认值为 0.0
        dropout: float = 0.0,
        # 是否使用注意力偏置，默认值为 True
        attention_bias: bool = True,
        # 采样宽度，默认值为 90
        sample_width: int = 90,
        # 采样高度，默认值为 60
        sample_height: int = 60,
        # 采样帧数，默认值为 49
        sample_frames: int = 49,
        # 补丁大小，默认值为 2
        patch_size: int = 2,
        # 时间压缩比例，默认值为 4
        temporal_compression_ratio: int = 4,
        # 最大文本序列长度，默认值为 226
        max_text_seq_length: int = 226,
        # 激活函数类型，默认值为 "gelu-approximate"
        activation_fn: str = "gelu-approximate",
        # 时间步激活函数类型，默认值为 "silu"
        timestep_activation_fn: str = "silu",
        # 是否使用元素逐个仿射的归一化，默认值为 True
        norm_elementwise_affine: bool = True,
        # 归一化的 epsilon 值，默认值为 1e-5
        norm_eps: float = 1e-5,
        # 空间插值缩放因子，默认值为 1.875
        spatial_interpolation_scale: float = 1.875,
        # 时间插值缩放因子，默认值为 1.0
        temporal_interpolation_scale: float = 1.0,
        # 是否使用旋转位置嵌入，默认值为 False
        use_rotary_positional_embeddings: bool = False,
        # 是否使用学习的位置嵌入，默认值为 False
        use_learned_positional_embeddings: bool = False,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 计算内部维度，等于注意力头数与每个头的维度乘积
        inner_dim = num_attention_heads * attention_head_dim

        # 检查位置嵌入的使用情况，如果不支持则抛出错误
        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. 创建补丁嵌入层
        self.patch_embed = CogVideoXPatchEmbed(
            # 设置补丁大小
            patch_size=patch_size,
            # 输入通道数
            in_channels=in_channels,
            # 嵌入维度
            embed_dim=inner_dim,
            # 文本嵌入维度
            text_embed_dim=text_embed_dim,
            # 是否使用偏置
            bias=True,
            # 样本宽度
            sample_width=sample_width,
            # 样本高度
            sample_height=sample_height,
            # 样本帧数
            sample_frames=sample_frames,
            # 时间压缩比
            temporal_compression_ratio=temporal_compression_ratio,
            # 最大文本序列长度
            max_text_seq_length=max_text_seq_length,
            # 空间插值缩放
            spatial_interpolation_scale=spatial_interpolation_scale,
            # 时间插值缩放
            temporal_interpolation_scale=temporal_interpolation_scale,
            # 使用位置嵌入
            use_positional_embeddings=not use_rotary_positional_embeddings,
            # 使用学习的位置嵌入
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        # 创建嵌入丢弃层
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. 创建时间嵌入
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        # 创建时间步嵌入层
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        # 3. 定义时空变换器块
        self.transformer_blocks = nn.ModuleList(
            [
                # 创建多个变换器块
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                # 根据层数重复创建变换器块
                for _ in range(num_layers)
            ]
        )
        # 创建最终的层归一化
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. 输出块的定义
        self.norm_out = AdaLayerNorm(
            # 嵌入维度
            embedding_dim=time_embed_dim,
            # 输出维度
            output_dim=2 * inner_dim,
            # 是否使用元素级别的归一化
            norm_elementwise_affine=norm_elementwise_affine,
            # 归一化的epsilon值
            norm_eps=norm_eps,
            # 块的维度
            chunk_dim=1,
        )
        # 创建输出的线性层
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 设置梯度检查点的方法
    def _set_gradient_checkpointing(self, module, value=False):
        # 更新梯度检查点标志
        self.gradient_checkpointing = value

    @property
    # 从 diffusers.models.unets.unet_2d_condition 中复制的属性
    # 定义一个方法，返回注意力处理器的字典
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回值:
            `dict` 的注意力处理器: 一个字典，包含模型中所有使用的注意力处理器，以权重名称索引。
        """
        # 初始化一个空字典用于存储处理器
        processors = {}

        # 定义一个递归函数，用于添加注意力处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否具有获取处理器的方法
            if hasattr(module, "get_processor"):
                # 将处理器添加到字典中，键为处理器的名称
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用，处理子模块
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            # 返回处理器字典
            return processors

        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数，添加处理器
            fn_recursive_add_processors(name, module, processors)

        # 返回收集到的处理器字典
        return processors

    # 从 UNet2DConditionModel 中复制的方法，用于设置注意力处理器
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        设置用于计算注意力的处理器。

        参数:
            processor (`dict` 的 `AttentionProcessor` 或仅 `AttentionProcessor`):
                实例化的处理器类或处理器类的字典，将作为所有 `Attention` 层的处理器设置。

                如果 `processor` 是一个字典，键需要定义相应交叉注意力处理器的路径。
                在设置可训练的注意力处理器时，强烈建议这样做。

        """
        # 计算当前注意力处理器的数量
        count = len(self.attn_processors.keys())

        # 如果传入的处理器是字典，且数量与当前处理器不匹配，则抛出异常
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"传入了处理器字典，但处理器数量 {len(processor)} 与注意力层数量 {count} 不匹配。请确保传入 {count} 个处理器类。"
            )

        # 定义一个递归函数，用于设置注意力处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查模块是否具有设置处理器的方法
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，则直接设置
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中弹出对应的处理器并设置
                    module.set_processor(processor.pop(f"{name}.processor"))

            # 遍历模块的所有子模块
            for sub_name, child in module.named_children():
                # 递归调用，处理子模块
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 遍历当前对象的所有子模块
        for name, module in self.named_children():
            # 调用递归函数，设置处理器
            fn_recursive_attn_processor(name, module, processor)

    # 从 UNet2DConditionModel 中复制的方法，涉及融合 QKV 投影
    # 定义融合 QKV 投影的方法
        def fuse_qkv_projections(self):
            """
            启用融合的 QKV 投影。对于自注意力模块，所有投影矩阵（即查询、键、值）都被融合。
            对于交叉注意力模块，键和值投影矩阵被融合。
    
            <Tip warning={true}>
    
            此 API 是 🧪 实验性的。
    
            </Tip>
            """
            # 初始化原始注意力处理器为 None
            self.original_attn_processors = None
    
            # 遍历所有注意力处理器
            for _, attn_processor in self.attn_processors.items():
                # 如果注意力处理器的类名包含 "Added"
                if "Added" in str(attn_processor.__class__.__name__):
                    # 抛出异常，表示不支持有额外 KV 投影的模型
                    raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")
    
            # 保存原始注意力处理器
            self.original_attn_processors = self.attn_processors
    
            # 遍历所有模块
            for module in self.modules():
                # 如果模块是 Attention 类型
                if isinstance(module, Attention):
                    # 融合投影
                    module.fuse_projections(fuse=True)
    
            # 设置注意力处理器为融合的处理器
            self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())
    
        # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections 拷贝而来
        def unfuse_qkv_projections(self):
            """禁用融合的 QKV 投影（如果已启用）。
    
            <Tip warning={true}>
    
            此 API 是 🧪 实验性的。
    
            </Tip>
    
            """
            # 如果原始注意力处理器不为 None
            if self.original_attn_processors is not None:
                # 设置注意力处理器为原始处理器
                self.set_attn_processor(self.original_attn_processors)
    
        # 定义前向传播方法
        def forward(
            # 隐藏状态输入的张量
            hidden_states: torch.Tensor,
            # 编码器隐藏状态的张量
            encoder_hidden_states: torch.Tensor,
            # 时间步的整数或浮点数
            timestep: Union[int, float, torch.LongTensor],
            # 可选的时间步条件张量
            timestep_cond: Optional[torch.Tensor] = None,
            # 可选的图像旋转嵌入
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            # 返回字典的布尔值，默认为 True
            return_dict: bool = True,
```