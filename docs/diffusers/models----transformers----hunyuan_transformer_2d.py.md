# `.\diffusers\models\transformers\hunyuan_transformer_2d.py`

```
# 版权所有 2024 HunyuanDiT 作者，Qixun Wang 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（"许可证"）进行许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，否则根据许可证分发的软件均按 "原样" 基础提供，
# 不提供任何种类的保证或条件，无论是明示或暗示的。
# 有关许可证的具体条款和条件，请参阅许可证。
from typing import Dict, Optional, Union  # 导入字典、可选和联合类型定义

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块

from ...configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入混合类和注册功能
from ...utils import logging  # 从工具包导入日志记录功能
from ...utils.torch_utils import maybe_allow_in_graph  # 导入可能允许图形内操作的功能
from ..attention import FeedForward  # 从注意力模块导入前馈网络
from ..attention_processor import Attention, AttentionProcessor, FusedHunyuanAttnProcessor2_0, HunyuanAttnProcessor2_0  # 导入注意力处理器
from ..embeddings import (  # 导入嵌入模块
    HunyuanCombinedTimestepTextSizeStyleEmbedding,  # 组合时间步、文本、大小和样式的嵌入
    PatchEmbed,  # 图像补丁嵌入
    PixArtAlphaTextProjection,  # 像素艺术文本投影
)
from ..modeling_outputs import Transformer2DModelOutput  # 导入 2D 变换器模型输出类型
from ..modeling_utils import ModelMixin  # 导入模型混合类
from ..normalization import AdaLayerNormContinuous, FP32LayerNorm  # 导入自适应层归一化和 FP32 层归一化

logger = logging.get_logger(__name__)  # 创建当前模块的日志记录器，禁用 pylint 警告

class AdaLayerNormShift(nn.Module):  # 定义自适应层归一化偏移类，继承自 nn.Module
    r"""  # 类文档字符串，描述类的功能
    Norm layer modified to incorporate timestep embeddings.  # 归一化层，修改以包含时间步嵌入

    Parameters:  # 参数说明
        embedding_dim (`int`): The size of each embedding vector.  # 嵌入向量的大小
        num_embeddings (`int`): The size of the embeddings dictionary.  # 嵌入字典的大小
    """

    def __init__(self, embedding_dim: int, elementwise_affine=True, eps=1e-6):  # 初始化方法
        super().__init__()  # 调用父类初始化方法
        self.silu = nn.SiLU()  # 定义 SiLU 激活函数
        self.linear = nn.Linear(embedding_dim, embedding_dim)  # 定义线性层，输入输出维度均为嵌入维度
        self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=elementwise_affine, eps=eps)  # 定义层归一化

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:  # 定义前向传播方法
        shift = self.linear(self.silu(emb.to(torch.float32)).to(emb.dtype))  # 计算偏移量
        x = self.norm(x) + shift.unsqueeze(dim=1)  # 对输入进行归一化并加上偏移
        return x  # 返回处理后的张量


@maybe_allow_in_graph  # 装饰器，可能允许在计算图中使用
class HunyuanDiTBlock(nn.Module):  # 定义 Hunyuan-DiT 模型中的变换器块类
    r"""  # 类文档字符串，描述类的功能
    Transformer block used in Hunyuan-DiT model (https://github.com/Tencent/HunyuanDiT). Allow skip connection and  # Hunyuan-DiT 模型中的变换器块，允许跳过连接和
    QKNorm  # QKNorm 功能
    # 参数说明部分，定义各参数的类型和作用
        Parameters:
            dim (`int`):  # 输入和输出的通道数
                The number of channels in the input and output.
            num_attention_heads (`int`):  # 多头注意力机制中使用的头数
                The number of heads to use for multi-head attention.
            cross_attention_dim (`int`, *optional*):  # 跨注意力的编码器隐藏状态向量的大小
                The size of the encoder_hidden_states vector for cross attention.
            dropout (`float`, *optional*, defaults to 0.0):  # 用于正则化的丢弃概率
                The dropout probability to use.
            activation_fn (`str`, *optional*, defaults to `"geglu"`):  # 前馈网络中使用的激活函数
                Activation function to be used in feed-forward.
            norm_elementwise_affine (`bool`, *optional*, defaults to `True`):  # 是否使用可学习的元素逐个仿射参数进行归一化
                Whether to use learnable elementwise affine parameters for normalization.
            norm_eps (`float`, *optional*, defaults to 1e-6):  # 加到归一化层分母的小常数，以防止除以零
                A small constant added to the denominator in normalization layers to prevent division by zero.
            final_dropout (`bool`, *optional*, defaults to False):  # 在最后的前馈层后是否应用最终丢弃
                Whether to apply a final dropout after the last feed-forward layer.
            ff_inner_dim (`int`, *optional*):  # 前馈块中隐藏层的大小，默认为 None
                The size of the hidden layer in the feed-forward block. Defaults to `None`.
            ff_bias (`bool`, *optional*, defaults to `True`):  # 前馈块中是否使用偏置
                Whether to use bias in the feed-forward block.
            skip (`bool`, *optional*, defaults to `False`):  # 是否使用跳过连接，默认为下块和中块的 False
                Whether to use skip connection. Defaults to `False` for down-blocks and mid-blocks.
            qk_norm (`bool`, *optional*, defaults to `True`):  # 在 QK 计算中是否使用归一化，默认为 True
                Whether to use normalization in QK calculation. Defaults to `True`.
        """
    
        # 构造函数的定义，初始化各参数
        def __init__(
            self,
            dim: int,  # 输入和输出的通道数
            num_attention_heads: int,  # 多头注意力机制中使用的头数
            cross_attention_dim: int = 1024,  # 默认的跨注意力维度
            dropout=0.0,  # 默认的丢弃概率
            activation_fn: str = "geglu",  # 默认的激活函数
            norm_elementwise_affine: bool = True,  # 默认使用可学习的仿射参数
            norm_eps: float = 1e-6,  # 默认的归一化小常数
            final_dropout: bool = False,  # 默认不应用最终丢弃
            ff_inner_dim: Optional[int] = None,  # 默认的前馈块隐藏层大小
            ff_bias: bool = True,  # 默认使用偏置
            skip: bool = False,  # 默认不使用跳过连接
            qk_norm: bool = True,  # 默认在 QK 计算中使用归一化
    ):
        # 调用父类构造函数
        super().__init__()

        # 定义三个块，每个块都有自己的归一化层。
        # 注意：新版本发布时，检查 norm2 和 norm3
        # 1. 自注意力机制
        self.norm1 = AdaLayerNormShift(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        # 创建自注意力机制的实例
        self.attn1 = Attention(
            query_dim=dim,  # 查询向量的维度
            cross_attention_dim=None,  # 交叉注意力的维度，未使用
            dim_head=dim // num_attention_heads,  # 每个头的维度
            heads=num_attention_heads,  # 注意力头的数量
            qk_norm="layer_norm" if qk_norm else None,  # 查询和键的归一化方法
            eps=1e-6,  # 数值稳定性常数
            bias=True,  # 是否使用偏置
            processor=HunyuanAttnProcessor2_0(),  # 注意力处理器的实例
        )

        # 2. 交叉注意力机制
        self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        # 创建交叉注意力机制的实例
        self.attn2 = Attention(
            query_dim=dim,  # 查询向量的维度
            cross_attention_dim=cross_attention_dim,  # 交叉注意力的维度
            dim_head=dim // num_attention_heads,  # 每个头的维度
            heads=num_attention_heads,  # 注意力头的数量
            qk_norm="layer_norm" if qk_norm else None,  # 查询和键的归一化方法
            eps=1e-6,  # 数值稳定性常数
            bias=True,  # 是否使用偏置
            processor=HunyuanAttnProcessor2_0(),  # 注意力处理器的实例
        )
        # 3. 前馈网络
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        # 创建前馈网络的实例
        self.ff = FeedForward(
            dim,  # 输入维度
            dropout=dropout,  # dropout 比例
            activation_fn=activation_fn,  # 激活函数
            final_dropout=final_dropout,  # 最终 dropout 比例
            inner_dim=ff_inner_dim,  # 内部维度，通常是 dim 的倍数
            bias=ff_bias,  # 是否使用偏置
        )

        # 4. 跳跃连接
        if skip:  # 如果启用跳跃连接
            self.skip_norm = FP32LayerNorm(2 * dim, norm_eps, elementwise_affine=True)  # 创建归一化层
            self.skip_linear = nn.Linear(2 * dim, dim)  # 创建线性层
        else:  # 如果不启用跳跃连接
            self.skip_linear = None  # 设置为 None

        # 将块大小默认为 None
        self._chunk_size = None  # 初始化块大小
        self._chunk_dim = 0  # 初始化块维度

    # 从 diffusers.models.attention.BasicTransformerBlock 复制的设置块前馈方法
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # 设置块前馈
        self._chunk_size = chunk_size  # 设置块大小
        self._chunk_dim = dim  # 设置块维度

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态
        temb: Optional[torch.Tensor] = None,  # 额外的嵌入
        image_rotary_emb=None,  # 图像旋转嵌入
        skip=None,  # 跳跃连接标志
    ) -> torch.Tensor:
        # 注意：以下代码块中的计算总是在归一化之后进行。
        # 0. 长跳跃连接
        # 如果 skip_linear 不为 None，执行跳跃连接
        if self.skip_linear is not None:
            # 将当前的隐藏状态与跳跃连接的输出在最后一维上拼接
            cat = torch.cat([hidden_states, skip], dim=-1)
            # 对拼接后的结果进行归一化处理
            cat = self.skip_norm(cat)
            # 通过线性层处理归一化后的结果，更新隐藏状态
            hidden_states = self.skip_linear(cat)

        # 1. 自注意力
        # 对当前隐藏状态进行归一化，准备进行自注意力计算
        norm_hidden_states = self.norm1(hidden_states, temb)  ### checked: self.norm1 is correct
        # 计算自注意力的输出
        attn_output = self.attn1(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        # 将自注意力的输出加到隐藏状态上，形成新的隐藏状态
        hidden_states = hidden_states + attn_output

        # 2. 交叉注意力
        # 将交叉注意力的输出加到当前的隐藏状态上
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),  # 先进行归一化
            encoder_hidden_states=encoder_hidden_states,  # 使用编码器的隐藏状态
            image_rotary_emb=image_rotary_emb,  # 传递旋转嵌入
        )

        # 前馈网络层 ### TODO: 在状态字典中切换 norm2 和 norm3
        # 对当前的隐藏状态进行归一化处理，准备进入前馈网络
        mlp_inputs = self.norm3(hidden_states)
        # 通过前馈网络处理归一化后的输入，更新隐藏状态
        hidden_states = hidden_states + self.ff(mlp_inputs)

        # 返回最终的隐藏状态
        return hidden_states
# 定义 HunyuanDiT2DModel 类，继承自 ModelMixin 和 ConfigMixin
class HunyuanDiT2DModel(ModelMixin, ConfigMixin):
    """
    HunYuanDiT: 基于 Transformer 的扩散模型。

    继承 ModelMixin 和 ConfigMixin 以与 diffusers 的采样器 StableDiffusionPipeline 兼容。

    参数:
        num_attention_heads (`int`, *可选*, 默认为 16):
            多头注意力的头数。
        attention_head_dim (`int`, *可选*, 默认为 88):
            每个头的通道数。
        in_channels (`int`, *可选*):
            输入和输出的通道数（如果输入为 **连续**，需指定）。
        patch_size (`int`, *可选*):
            输入的补丁大小。
        activation_fn (`str`, *可选*, 默认为 `"geglu"`):
            前馈网络中使用的激活函数。
        sample_size (`int`, *可选*):
            潜在图像的宽度。训练期间固定使用，以学习位置嵌入的数量。
        dropout (`float`, *可选*, 默认为 0.0):
            使用的 dropout 概率。
        cross_attention_dim (`int`, *可选*):
            clip 文本嵌入中的维度数量。
        hidden_size (`int`, *可选*):
            条件嵌入层中隐藏层的大小。
        num_layers (`int`, *可选*, 默认为 1):
            使用的 Transformer 块的层数。
        mlp_ratio (`float`, *可选*, 默认为 4.0):
            隐藏层大小与输入大小的比率。
        learn_sigma (`bool`, *可选*, 默认为 `True`):
             是否预测方差。
        cross_attention_dim_t5 (`int`, *可选*):
            t5 文本嵌入中的维度数量。
        pooled_projection_dim (`int`, *可选*):
            池化投影的大小。
        text_len (`int`, *可选*):
            clip 文本嵌入的长度。
        text_len_t5 (`int`, *可选*):
            T5 文本嵌入的长度。
        use_style_cond_and_image_meta_size (`bool`,  *可选*):
            是否使用风格条件和图像元数据大小。版本 <=1.1 为 True，版本 >= 1.2 为 False
    """

    # 注册到配置中
    @register_to_config
    def __init__(
        # 多头注意力的头数，默认为 16
        self,
        num_attention_heads: int = 16,
        # 每个头的通道数，默认为 88
        attention_head_dim: int = 88,
        # 输入和输出的通道数，默认为 None
        in_channels: Optional[int] = None,
        # 输入的补丁大小，默认为 None
        patch_size: Optional[int] = None,
        # 激活函数，默认为 "gelu-approximate"
        activation_fn: str = "gelu-approximate",
        # 潜在图像的宽度，默认为 32
        sample_size=32,
        # 条件嵌入层中隐藏层的大小，默认为 1152
        hidden_size=1152,
        # 使用的 Transformer 块的层数，默认为 28
        num_layers: int = 28,
        # 隐藏层大小与输入大小的比率，默认为 4.0
        mlp_ratio: float = 4.0,
        # 是否预测方差，默认为 True
        learn_sigma: bool = True,
        # clip 文本嵌入中的维度数量，默认为 1024
        cross_attention_dim: int = 1024,
        # 正则化类型，默认为 "layer_norm"
        norm_type: str = "layer_norm",
        # t5 文本嵌入中的维度数量，默认为 2048
        cross_attention_dim_t5: int = 2048,
        # 池化投影的大小，默认为 1024
        pooled_projection_dim: int = 1024,
        # clip 文本嵌入的长度，默认为 77
        text_len: int = 77,
        # T5 文本嵌入的长度，默认为 256
        text_len_t5: int = 256,
        # 是否使用风格条件和图像元数据大小，默认为 True
        use_style_cond_and_image_meta_size: bool = True,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 根据是否学习 sigma 决定输出通道数
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        # 设置注意力头的数量
        self.num_heads = num_attention_heads
        # 计算内部维度，等于注意力头数量乘以每个头的维度
        self.inner_dim = num_attention_heads * attention_head_dim

        # 初始化文本嵌入器，用于将输入特征投影到更高维空间
        self.text_embedder = PixArtAlphaTextProjection(
            # 输入特征维度
            in_features=cross_attention_dim_t5,
            # 隐藏层大小为输入特征的四倍
            hidden_size=cross_attention_dim_t5 * 4,
            # 输出特征维度
            out_features=cross_attention_dim,
            # 激活函数设置为"siluf_fp32"
            act_fn="silu_fp32",
        )

        # 初始化文本嵌入的填充参数，使用随机正态分布初始化
        self.text_embedding_padding = nn.Parameter(
            torch.randn(text_len + text_len_t5, cross_attention_dim, dtype=torch.float32)
        )

        # 初始化位置嵌入，构建图像的补丁嵌入
        self.pos_embed = PatchEmbed(
            # 补丁的高度
            height=sample_size,
            # 补丁的宽度
            width=sample_size,
            # 输入通道数
            in_channels=in_channels,
            # 嵌入维度
            embed_dim=hidden_size,
            # 补丁大小
            patch_size=patch_size,
            # 位置嵌入类型设置为 None
            pos_embed_type=None,
        )

        # 初始化时间和风格嵌入，结合时间步和文本大小
        self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
            # 隐藏层大小
            hidden_size,
            # 池化投影维度
            pooled_projection_dim=pooled_projection_dim,
            # 输入序列长度
            seq_len=text_len_t5,
            # 交叉注意力维度
            cross_attention_dim=cross_attention_dim_t5,
            # 是否使用风格条件和图像元数据大小
            use_style_cond_and_image_meta_size=use_style_cond_and_image_meta_size,
        )

        # 初始化 HunyuanDiT 块列表
        self.blocks = nn.ModuleList(
            [
                # 为每一层创建 HunyuanDiTBlock
                HunyuanDiTBlock(
                    # 内部维度
                    dim=self.inner_dim,
                    # 注意力头数量
                    num_attention_heads=self.config.num_attention_heads,
                    # 激活函数
                    activation_fn=activation_fn,
                    # 前馈网络内部维度
                    ff_inner_dim=int(self.inner_dim * mlp_ratio),
                    # 交叉注意力维度
                    cross_attention_dim=cross_attention_dim,
                    # 查询-键归一化开启
                    qk_norm=True,  # 详情见 http://arxiv.org/abs/2302.05442
                    # 如果当前层数大于层数的一半，则跳过
                    skip=layer > num_layers // 2,
                )
                # 遍历层数
                for layer in range(num_layers)
            ]
        )

        # 初始化输出的自适应层归一化
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        # 初始化输出的线性层，将内部维度映射到输出通道数
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel 中复制的代码，用于融合 QKV 投影，更新为 FusedHunyuanAttnProcessor2_0
    def fuse_qkv_projections(self):
        """ 
        启用融合的 QKV 投影。对于自注意力模块，所有投影矩阵（即查询、键、值）都被融合。 
        对于交叉注意力模块，键和值投影矩阵被融合。

        <Tip warning={true}>
        
        该 API 是 🧪 实验性的。

        </Tip>
        """
        # 初始化原始注意力处理器为 None
        self.original_attn_processors = None

        # 遍历所有注意力处理器
        for _, attn_processor in self.attn_processors.items():
            # 检查注意力处理器类名中是否包含 "Added"
            if "Added" in str(attn_processor.__class__.__name__):
                # 如果包含，则抛出错误，表示不支持融合 QKV 投影
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        # 保存当前的注意力处理器以备后用
        self.original_attn_processors = self.attn_processors

        # 遍历当前模块中的所有子模块
        for module in self.modules():
            # 检查模块是否为 Attention 类型
            if isinstance(module, Attention):
                # 对于 Attention 模块，启用投影融合
                module.fuse_projections(fuse=True)

        # 设置融合的注意力处理器
        self.set_attn_processor(FusedHunyuanAttnProcessor2_0())

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections 复制
    def unfuse_qkv_projections(self):
        """ 
        如果已启用，则禁用融合的 QKV 投影。

        <Tip warning={true}>
        
        该 API 是 🧪 实验性的。

        </Tip>

        """
        # 检查是否有原始注意力处理器
        if self.original_attn_processors is not None:
            # 恢复为原始注意力处理器
            self.set_attn_processor(self.original_attn_processors)

    @property
    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors 复制
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        返回:
            `dict` 类型的注意力处理器：一个字典，包含模型中使用的所有注意力处理器，按权重名称索引。
        """
        # 初始化处理器字典
        processors = {}

        # 定义递归添加处理器的函数
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            # 检查模块是否具有 get_processor 方法
            if hasattr(module, "get_processor"):
                # 将处理器添加到字典中
                processors[f"{name}.processor"] = module.get_processor()

            # 遍历子模块
            for sub_name, child in module.named_children():
                # 递归调用以添加子模块的处理器
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        # 遍历当前模块的所有子模块
        for name, module in self.named_children():
            # 调用递归函数添加处理器
            fn_recursive_add_processors(name, module, processors)

        # 返回处理器字典
        return processors

    # 从 diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor 复制
    # 定义设置注意力处理器的方法，接收一个注意力处理器或处理器字典
        def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
            r"""
            设置用于计算注意力的注意力处理器。
    
            参数：
                processor（`dict` of `AttentionProcessor` 或 `AttentionProcessor`）:
                    已实例化的处理器类或将作为处理器设置的处理器类字典
                    用于**所有** `Attention` 层。
    
                    如果 `processor` 是字典，则键需要定义相应的交叉注意力处理器路径。
                    当设置可训练的注意力处理器时，这强烈建议。
    
            """
            # 计算当前注意力处理器的数量
            count = len(self.attn_processors.keys())
    
            # 检查传入的处理器是否为字典，并验证其长度与注意力层数量是否一致
            if isinstance(processor, dict) and len(processor) != count:
                raise ValueError(
                    # 抛出错误，提示处理器数量与注意力层数量不匹配
                    f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                    f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
                )
    
            # 定义递归设置注意力处理器的内部函数
            def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
                # 检查模块是否具有设置处理器的方法
                if hasattr(module, "set_processor"):
                    # 如果处理器不是字典，直接设置处理器
                    if not isinstance(processor, dict):
                        module.set_processor(processor)
                    else:
                        # 从字典中取出处理器并设置
                        module.set_processor(processor.pop(f"{name}.processor"))
    
                # 遍历模块的子模块，递归调用设置处理器的方法
                for sub_name, child in module.named_children():
                    fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)
    
            # 遍历当前对象的所有子模块，调用递归设置处理器的方法
            for name, module in self.named_children():
                fn_recursive_attn_processor(name, module, processor)
    
        # 定义设置默认注意力处理器的方法
        def set_default_attn_processor(self):
            """
            禁用自定义注意力处理器，并设置默认的注意力实现。
            """
            # 调用设置注意力处理器的方法，使用默认的 HunyuanAttnProcessor2_0
            self.set_attn_processor(HunyuanAttnProcessor2_0())
    
        # 定义前向传播的方法，接收多个输入参数
        def forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states=None,
            text_embedding_mask=None,
            encoder_hidden_states_t5=None,
            text_embedding_mask_t5=None,
            image_meta_size=None,
            style=None,
            image_rotary_emb=None,
            controlnet_block_samples=None,
            return_dict=True,
        # 从 diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking 复制的代码
    # 定义一个方法以启用前馈层的分块处理，参数为分块大小和维度
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
            """
            设置注意力处理器使用 [前馈分块处理](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers)。
    
            参数:
                chunk_size (`int`, *可选*):
                    前馈层的分块大小。如果未指定，将对每个维度为 `dim` 的张量单独运行前馈层。
                dim (`int`, *可选*, 默认为 `0`):
                    前馈计算应分块的维度。选择 dim=0（批处理）或 dim=1（序列长度）。
            """
            # 检查维度是否为 0 或 1
            if dim not in [0, 1]:
                # 抛出值错误，提示维度设置不当
                raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")
    
            # 默认分块大小为 1
            chunk_size = chunk_size or 1
    
            # 定义递归函数以设置前馈层的分块处理
            def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
                # 如果模块具有设置分块前馈的属性，则进行设置
                if hasattr(module, "set_chunk_feed_forward"):
                    module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
    
                # 遍历模块的所有子模块并递归调用
                for child in module.children():
                    fn_recursive_feed_forward(child, chunk_size, dim)
    
            # 遍历当前对象的所有子模块并应用递归函数
            for module in self.children():
                fn_recursive_feed_forward(module, chunk_size, dim)
    
        # 从 diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking 复制
        def disable_forward_chunking(self):
            # 定义递归函数以禁用前馈层的分块处理
            def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
                # 如果模块具有设置分块前馈的属性，则进行设置
                if hasattr(module, "set_chunk_feed_forward"):
                    module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
    
                # 遍历模块的所有子模块并递归调用
                for child in module.children():
                    fn_recursive_feed_forward(child, chunk_size, dim)
    
            # 遍历当前对象的所有子模块并应用递归函数，禁用分块处理
            for module in self.children():
                fn_recursive_feed_forward(module, None, 0)
```