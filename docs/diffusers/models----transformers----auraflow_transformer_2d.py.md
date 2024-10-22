# `.\diffusers\models\transformers\auraflow_transformer_2d.py`

```
# 版权声明，指明该文件的作者和许可证信息
# Copyright 2024 AuraFlow Authors, The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证，版本 2.0（“许可证”）进行授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意另有规定，按照许可证分发的软件
# 是以“原样”基础分发，不提供任何形式的担保或条件，
# 明示或暗示。
# 请参阅许可证以获取有关权限和
# 限制的具体语言。

# 从 typing 模块导入 Any、Dict 和 Union 类型
from typing import Any, Dict, Union

# 导入 PyTorch 及其神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从配置和工具模块导入所需类和函数
from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import is_torch_version, logging
from ...utils.torch_utils import maybe_allow_in_graph
from ..attention_processor import (
    Attention,
    AttentionProcessor,
    AuraFlowAttnProcessor2_0,
    FusedAuraFlowAttnProcessor2_0,
)
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_outputs import Transformer2DModelOutput
from ..modeling_utils import ModelMixin
from ..normalization import AdaLayerNormZero, FP32LayerNorm

# 创建一个日志记录器，便于记录信息和错误
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个函数，用于找到 n 的下一个可被 k 整除的数
def find_multiple(n: int, k: int) -> int:
    # 如果 n 可以被 k 整除，直接返回 n
    if n % k == 0:
        return n
    # 否则返回下一个可被 k 整除的数
    return n + k - (n % k)

# 定义 AuraFlowPatchEmbed 类，表示一个嵌入模块
# 不使用卷积来进行投影，同时使用学习到的位置嵌入
class AuraFlowPatchEmbed(nn.Module):
    # 初始化函数，设置嵌入模块的参数
    def __init__(
        self,
        height=224,  # 输入图像高度
        width=224,   # 输入图像宽度
        patch_size=16,  # 每个补丁的大小
        in_channels=3,   # 输入通道数（例如，RGB图像）
        embed_dim=768,   # 嵌入维度
        pos_embed_max_size=None,  # 最大位置嵌入大小
    ):
        super().__init__()

        # 计算补丁数量
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.pos_embed_max_size = pos_embed_max_size

        # 定义线性层，将补丁投影到嵌入空间
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        # 定义位置嵌入参数，随机初始化
        self.pos_embed = nn.Parameter(torch.randn(1, pos_embed_max_size, embed_dim) * 0.1)

        # 保存补丁大小和图像的补丁高度和宽度
        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        # 保存基础大小
        self.base_size = height // patch_size
    # 根据输入的高度和宽度选择基于维度的嵌入索引
    def pe_selection_index_based_on_dim(self, h, w):
        # 计算基于补丁大小的高度和宽度
        h_p, w_p = h // self.patch_size, w // self.patch_size
        # 生成原始位置嵌入的索引
        original_pe_indexes = torch.arange(self.pos_embed.shape[1])
        # 计算最大高度和宽度
        h_max, w_max = int(self.pos_embed_max_size**0.5), int(self.pos_embed_max_size**0.5)
        # 将索引视图调整为二维网格
        original_pe_indexes = original_pe_indexes.view(h_max, w_max)
        # 计算起始行和结束行
        starth = h_max // 2 - h_p // 2
        endh = starth + h_p
        # 计算起始列和结束列
        startw = w_max // 2 - w_p // 2
        endw = startw + w_p
        # 选择指定范围的原始位置嵌入索引
        original_pe_indexes = original_pe_indexes[starth:endh, startw:endw]
        # 返回展平的索引
        return original_pe_indexes.flatten()
    
    # 前向传播函数
    def forward(self, latent):
        # 获取输入的批量大小、通道数、高度和宽度
        batch_size, num_channels, height, width = latent.size()
        # 调整潜在张量的形状以适应补丁结构
        latent = latent.view(
            batch_size,
            num_channels,
            height // self.patch_size,
            self.patch_size,
            width // self.patch_size,
            self.patch_size,
        )
        # 重新排列维度并展平
        latent = latent.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        # 应用投影层
        latent = self.proj(latent)
        # 获取嵌入索引
        pe_index = self.pe_selection_index_based_on_dim(height, width)
        # 返回潜在张量与位置嵌入的和
        return latent + self.pos_embed[:, pe_index]
# 取自原始的 Aura 流推理代码。
# 我们的前馈网络只使用 GELU，而 Aura 使用 SiLU。
class AuraFlowFeedForward(nn.Module):
    # 初始化方法，接收输入维度和隐藏层维度（如果未提供则设为 4 倍输入维度）
    def __init__(self, dim, hidden_dim=None) -> None:
        # 调用父类构造函数
        super().__init__()
        # 如果没有提供隐藏层维度，则计算为输入维度的 4 倍
        if hidden_dim is None:
            hidden_dim = 4 * dim

        # 计算最终隐藏层维度，取隐藏层维度的 2/3
        final_hidden_dim = int(2 * hidden_dim / 3)
        # 将最终隐藏层维度调整为 256 的倍数
        final_hidden_dim = find_multiple(final_hidden_dim, 256)

        # 创建第一个线性层，不使用偏置
        self.linear_1 = nn.Linear(dim, final_hidden_dim, bias=False)
        # 创建第二个线性层，不使用偏置
        self.linear_2 = nn.Linear(dim, final_hidden_dim, bias=False)
        # 创建输出投影层，不使用偏置
        self.out_projection = nn.Linear(final_hidden_dim, dim, bias=False)

    # 前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 经过第一个线性层并使用 SiLU 激活函数，然后与第二个线性层的输出相乘
        x = F.silu(self.linear_1(x)) * self.linear_2(x)
        # 经过输出投影层
        x = self.out_projection(x)
        # 返回处理后的张量
        return x


class AuraFlowPreFinalBlock(nn.Module):
    # 初始化方法，接收嵌入维度和条件嵌入维度
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        # 调用父类构造函数
        super().__init__()

        # 定义 SiLU 激活函数
        self.silu = nn.SiLU()
        # 创建线性层，输出维度为嵌入维度的两倍，不使用偏置
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=False)

    # 前向传播方法
    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # 对条件嵌入应用 SiLU 激活并转换为与 x 相同的数据类型，然后通过线性层
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        # 将嵌入分成两个部分：缩放和偏移
        scale, shift = torch.chunk(emb, 2, dim=1)
        # 更新 x，使用缩放和偏移进行调整
        x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        # 返回调整后的张量
        return x


@maybe_allow_in_graph
class AuraFlowSingleTransformerBlock(nn.Module):
    """类似于 `AuraFlowJointTransformerBlock`，但只使用一个 DiT 而不是 MMDiT。"""

    # 初始化方法，接收输入维度、注意力头数量和每个头的维度
    def __init__(self, dim, num_attention_heads, attention_head_dim):
        # 调用父类构造函数
        super().__init__()

        # 创建层归一化对象，设置维度和不使用偏置，归一化类型为 "fp32_layer_norm"
        self.norm1 = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")

        # 创建注意力处理器
        processor = AuraFlowAttnProcessor2_0()
        # 创建注意力机制对象，设置参数
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="fp32_layer_norm",
            out_dim=dim,
            bias=False,
            out_bias=False,
            processor=processor,
        )

        # 创建第二层归一化对象，设置维度和不使用偏置
        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        # 创建前馈网络对象，隐藏层维度为输入维度的 4 倍
        self.ff = AuraFlowFeedForward(dim, dim * 4)

    # 前向传播方法，接收隐藏状态和条件嵌入
    def forward(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor):
        # 保存输入的残差
        residual = hidden_states

        # 进行归一化和投影
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # 经过注意力机制处理
        attn_output = self.attn(hidden_states=norm_hidden_states)

        # 将注意力输出与残差相结合，并进行第二次归一化
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        # 更新 hidden_states，使用缩放和偏移
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        # 经过前馈网络处理
        ff_output = self.ff(hidden_states)
        # 更新 hidden_states，使用门控机制
        hidden_states = gate_mlp.unsqueeze(1) * ff_output
        # 将残差与更新后的 hidden_states 相加
        hidden_states = residual + hidden_states

        # 返回最终的隐藏状态
        return hidden_states


@maybe_allow_in_graph
# 定义 AuraFlow 的 Transformer 块类，继承自 nn.Module
class AuraFlowJointTransformerBlock(nn.Module):
    r"""
    Transformer block for Aura Flow. Similar to SD3 MMDiT. Differences (non-exhaustive):

        * QK Norm in the attention blocks
        * No bias in the attention blocks
        * Most LayerNorms are in FP32

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        is_last (`bool`): Boolean to determine if this is the last block in the model.
    """

    # 初始化方法，接受输入维度、注意力头数和每个头的维度
    def __init__(self, dim, num_attention_heads, attention_head_dim):
        # 调用父类构造函数
        super().__init__()

        # 创建第一个层归一化对象，不使用偏置，采用 FP32 类型
        self.norm1 = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")
        # 创建上下文的层归一化对象，同样不使用偏置
        self.norm1_context = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")

        # 实例化注意力处理器
        processor = AuraFlowAttnProcessor2_0()
        # 创建注意力机制对象，配置查询维度、头数等参数
        self.attn = Attention(
            query_dim=dim,                       # 查询向量的维度
            cross_attention_dim=None,            # 交叉注意力的维度，未使用
            added_kv_proj_dim=dim,               # 添加的键值投影维度
            added_proj_bias=False,                # 不使用添加的偏置
            dim_head=attention_head_dim,         # 每个头的维度
            heads=num_attention_heads,            # 注意力头的数量
            qk_norm="fp32_layer_norm",           # QK 的归一化类型
            out_dim=dim,                         # 输出维度
            bias=False,                           # 不使用偏置
            out_bias=False,                       # 不使用输出偏置
            processor=processor,                  # 传入的处理器
            context_pre_only=False,               # 不仅仅使用上下文
        )

        # 创建第二个层归一化对象，不使用元素级的仿射变换和偏置
        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        # 创建前馈神经网络对象，输出维度是输入维度的四倍
        self.ff = AuraFlowFeedForward(dim, dim * 4)
        # 创建上下文的第二个层归一化对象
        self.norm2_context = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        # 创建上下文的前馈神经网络对象
        self.ff_context = AuraFlowFeedForward(dim, dim * 4)

    # 定义前向传播方法，接受隐藏状态、编码器的隐藏状态和时间嵌入
    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
    ):
        # 初始化残差为当前的隐藏状态
        residual = hidden_states
        # 初始化残差上下文为编码器的隐藏状态
        residual_context = encoder_hidden_states

        # 归一化和投影操作
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        # 对编码器隐藏状态进行归一化和投影
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 注意力机制计算
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
        )

        # 处理注意力输出以更新 `hidden_states`
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        # 对隐藏状态进行缩放和偏移
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        # 使用前馈网络处理隐藏状态
        hidden_states = gate_mlp.unsqueeze(1) * self.ff(hidden_states)
        # 将更新后的隐藏状态与残差相加
        hidden_states = residual + hidden_states

        # 处理注意力输出以更新 `encoder_hidden_states`
        encoder_hidden_states = self.norm2_context(residual_context + c_gate_msa.unsqueeze(1) * context_attn_output)
        # 对编码器隐藏状态进行缩放和偏移
        encoder_hidden_states = encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        # 使用前馈网络处理编码器隐藏状态
        encoder_hidden_states = c_gate_mlp.unsqueeze(1) * self.ff_context(encoder_hidden_states)
        # 将更新后的编码器隐藏状态与残差上下文相加
        encoder_hidden_states = residual_context + encoder_hidden_states

        # 返回编码器隐藏状态和更新后的隐藏状态
        return encoder_hidden_states, hidden_states
# 定义一个2D Transformer模型类，继承自ModelMixin和ConfigMixin
class AuraFlowTransformer2DModel(ModelMixin, ConfigMixin):
    r"""
    介绍AuraFlow中提出的2D Transformer模型（https://blog.fal.ai/auraflow/）。

    参数：
        sample_size (`int`): 潜在图像的宽度。由于用于学习位置嵌入，因此在训练过程中是固定的。
        patch_size (`int`): 将输入数据转换为小块的大小。
        in_channels (`int`, *optional*, defaults to 16): 输入通道的数量。
        num_mmdit_layers (`int`, *optional*, defaults to 4): 要使用的MMDiT Transformer块的层数。
        num_single_dit_layers (`int`, *optional*, defaults to 4):
            要使用的Transformer块的层数。这些块使用连接的图像和文本表示。
        attention_head_dim (`int`, *optional*, defaults to 64): 每个头的通道数。
        num_attention_heads (`int`, *optional*, defaults to 18): 用于多头注意力的头数。
        joint_attention_dim (`int`, *optional*): 要使用的`encoder_hidden_states`维度数量。
        caption_projection_dim (`int`): 投影`encoder_hidden_states`时使用的维度数量。
        out_channels (`int`, defaults to 16): 输出通道的数量。
        pos_embed_max_size (`int`, defaults to 4096): 从图像潜在值中嵌入的最大位置数量。
    """

    # 支持梯度检查点
    _supports_gradient_checkpointing = True

    # 将该方法注册到配置中
    @register_to_config
    def __init__(
        # 潜在图像的宽度，默认为64
        sample_size: int = 64,
        # 输入数据的小块大小，默认为2
        patch_size: int = 2,
        # 输入通道的数量，默认为4
        in_channels: int = 4,
        # MMDiT Transformer块的层数，默认为4
        num_mmdit_layers: int = 4,
        # 单一Transformer块的层数，默认为32
        num_single_dit_layers: int = 32,
        # 每个头的通道数，默认为256
        attention_head_dim: int = 256,
        # 多头注意力的头数，默认为12
        num_attention_heads: int = 12,
        # `encoder_hidden_states`的维度数量，默认为2048
        joint_attention_dim: int = 2048,
        # 投影时使用的维度数量，默认为3072
        caption_projection_dim: int = 3072,
        # 输出通道的数量，默认为4
        out_channels: int = 4,
        # 从图像潜在值中嵌入的最大位置数量，默认为1024
        pos_embed_max_size: int = 1024,
    ):
        # 初始化父类
        super().__init__()
        # 设置默认输出通道为输入通道数
        default_out_channels = in_channels
        # 如果提供了输出通道数，则使用该值，否则使用默认值
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        # 计算内部维度为注意力头数与每个注意力头维度的乘积
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        # 创建位置嵌入对象，使用配置中的样本大小和补丁大小
        self.pos_embed = AuraFlowPatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        # 创建线性层用于上下文嵌入，不使用偏置
        self.context_embedder = nn.Linear(
            self.config.joint_attention_dim, self.config.caption_projection_dim, bias=False
        )
        # 创建时间步嵌入对象，配置频道数和频率下采样
        self.time_step_embed = Timesteps(num_channels=256, downscale_freq_shift=0, scale=1000, flip_sin_to_cos=True)
        # 创建时间步投影层，输入频道数为256，嵌入维度为内部维度
        self.time_step_proj = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)

        # 创建联合变换器模块列表，根据配置中的层数
        self.joint_transformer_blocks = nn.ModuleList(
            [
                AuraFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_mmdit_layers)
            ]
        )
        # 创建单一变换器模块列表，根据配置中的层数
        self.single_transformer_blocks = nn.ModuleList(
            [
                AuraFlowSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for _ in range(self.config.num_single_dit_layers)
            ]
        )

        # 创建最终块的归一化层，维度为内部维度
        self.norm_out = AuraFlowPreFinalBlock(self.inner_dim, self.inner_dim)
        # 创建线性投影层，将内部维度映射到补丁大小平方与输出通道数的乘积，不使用偏置
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        # https://arxiv.org/abs/2309.16588
        # 防止注意力图中的伪影
        self.register_tokens = nn.Parameter(torch.randn(1, 8, self.inner_dim) * 0.02)

        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    @property
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors 复制的属性
    # 定义一个返回注意力处理器的函数，返回类型为字典
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # 初始化一个空字典用于存储处理器
        processors = {}
    
        # 定义递归函数用于添加处理器
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否具有获取处理器的方法
            if hasattr(module, "get_processor"):
                # 将处理器添加到字典中，键为处理器名称
                processors[f"{name}.processor"] = module.get_processor()
    
            # 遍历模块的子模块
            for sub_name, child in module.named_children():
                # 递归调用以添加子模块的处理器
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
    
            # 返回处理器字典
            return processors
    
        # 遍历当前对象的子模块
        for name, module in self.named_children():
            # 调用递归函数以添加所有子模块的处理器
            fn_recursive_add_processors(name, module, processors)
    
        # 返回包含所有处理器的字典
        return processors
    
    # 定义设置注意力处理器的函数
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.
    
        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.
    
                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
    
        """
        # 获取当前注意力处理器的数量
        count = len(self.attn_processors.keys())
    
        # 检查传入的处理器字典长度是否与注意力层数量一致
        if isinstance(processor, dict) and len(processor) != count:
            # 如果不一致，抛出错误
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )
    
        # 定义递归函数用于设置处理器
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            # 检查模块是否具有设置处理器的方法
            if hasattr(module, "set_processor"):
                # 如果处理器不是字典，直接设置处理器
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    # 从字典中获取并设置对应的处理器
                    module.set_processor(processor.pop(f"{name}.processor"))
    
            # 遍历子模块
            for sub_name, child in module.named_children():
                # 递归调用以设置子模块的处理器
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
        # 遍历当前对象的子模块
        for name, module in self.named_children():
            # 调用递归函数以设置所有子模块的处理器
            fn_recursive_attn_processor(name, module, processor)
    
    # 该函数用于融合注意力层中的 QKV 投影
    # 定义一个方法以启用融合的 QKV 投影
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        # 初始化原始的注意力处理器为 None
        self.original_attn_processors = None

        # 遍历所有的注意力处理器
        for _, attn_processor in self.attn_processors.items():
            # 如果注意力处理器类名中包含 "Added"，则抛出错误
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        # 保存当前的注意力处理器以便后续恢复
        self.original_attn_processors = self.attn_processors

        # 遍历所有模块以查找注意力模块
        for module in self.modules():
            # 检查模块是否为 Attention 类型
            if isinstance(module, Attention):
                # 对注意力模块进行融合投影处理
                module.fuse_projections(fuse=True)

        # 设置新的注意力处理器为融合的处理器
        self.set_attn_processor(FusedAuraFlowAttnProcessor2_0())

    # 从 UNet2DConditionModel 类复制的方法，用于取消融合的 QKV 投影
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        # 如果原始的注意力处理器不为 None，则恢复为原始处理器
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    # 定义一个方法以设置模块的梯度检查点
    def _set_gradient_checkpointing(self, module, value=False):
        # 检查模块是否具有梯度检查点属性
        if hasattr(module, "gradient_checkpointing"):
            # 将梯度检查点属性设置为给定值
            module.gradient_checkpointing = value

    # 定义前向传播方法
    def forward(
        # 接收隐藏状态的浮点张量
        hidden_states: torch.FloatTensor,
        # 可选的编码器隐藏状态的浮点张量
        encoder_hidden_states: torch.FloatTensor = None,
        # 可选的时间步长的长整型张量
        timestep: torch.LongTensor = None,
        # 是否返回字典格式的标志，默认为 True
        return_dict: bool = True,
```