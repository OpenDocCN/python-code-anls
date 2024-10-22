# `.\diffusers\models\attention.py`

```py
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache 许可证，第 2.0 版（“许可证”）许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件
# 按“原样”分发，不附有任何明示或暗示的担保或条件。
# 请参见许可证以获取有关权限和
# 限制的具体语言。
from typing import Any, Dict, List, Optional, Tuple  # 导入所需的类型注解

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数式 API
from torch import nn  # 从 PyTorch 导入神经网络模块

from ..utils import deprecate, logging  # 从上级目录导入工具函数和日志记录模块
from ..utils.torch_utils import maybe_allow_in_graph  # 导入可能允许图形计算的工具
from .activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU  # 导入不同激活函数
from .attention_processor import Attention, JointAttnProcessor2_0  # 导入注意力处理器
from .embeddings import SinusoidalPositionalEmbedding  # 导入正弦位置嵌入
from .normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm  # 导入各种归一化方法


logger = logging.get_logger(__name__)  # 创建一个模块级别的日志记录器


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # 定义一个函数，按块处理前馈神经网络，以节省内存
    # 检查隐藏状态的维度是否能够被块大小整除
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )  # 如果不能整除，抛出一个值错误，提示块大小设置不正确

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size  # 计算可以生成的块数量
    # 按块处理隐藏状态并将结果拼接成一个张量
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],  # 将每个块传入前馈模块进行处理
        dim=chunk_dim,  # 在指定维度上拼接结果
    )
    return ff_output  # 返回拼接后的输出


@maybe_allow_in_graph  # 可能允许在图形计算中使用此类
class GatedSelfAttentionDense(nn.Module):  # 定义一个门控自注意力密集层类，继承自 nn.Module
    r"""  # 类文档字符串，描述类的功能和参数

    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.  # 查询的通道数
        context_dim (`int`): The number of channels in the context.  # 上下文的通道数
        n_heads (`int`): The number of heads to use for attention.  # 注意力头的数量
        d_head (`int`): The number of channels in each head.  # 每个头的通道数
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()  # 调用父类构造函数

        # 因为需要拼接视觉特征和对象特征，所以需要一个线性投影
        self.linear = nn.Linear(context_dim, query_dim)  # 创建一个线性层，用于上下文到查询的映射

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)  # 初始化注意力层
        self.ff = FeedForward(query_dim, activation_fn="geglu")  # 初始化前馈层，激活函数为 GEGLU

        self.norm1 = nn.LayerNorm(query_dim)  # 初始化第一个层归一化
        self.norm2 = nn.LayerNorm(query_dim)  # 初始化第二个层归一化

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))  # 注册注意力参数 alpha_attn
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))  # 注册密集参数 alpha_dense

        self.enabled = True  # 设置 enabled 属性为 True
    # 前向传播函数，接收输入张量 x 和对象张量 objs，返回处理后的张量
        def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
            # 如果未启用该模块，直接返回输入张量 x
            if not self.enabled:
                return x
    
            # 获取输入张量的第二维大小，表示视觉特征的数量
            n_visual = x.shape[1]
            # 通过线性层处理对象张量
            objs = self.linear(objs)
    
            # 将输入张量和处理后的对象张量拼接，进行归一化后计算注意力，并调整输入张量
            x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
            # 使用另一个归一化层处理 x，并通过前馈网络调整 x
            x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))
    
            # 返回处理后的张量 x
            return x
# 装饰器，可能允许在计算图中使用此类
@maybe_allow_in_graph
# 定义一个联合变换器块类，继承自 nn.Module
class JointTransformerBlock(nn.Module):
    r"""
    根据 MMDiT 架构定义的变换器块，介绍于 Stable Diffusion 3.

    参考文献: https://arxiv.org/abs/2403.03206

    参数:
        dim (`int`): 输入和输出中的通道数量.
        num_attention_heads (`int`): 用于多头注意力的头数.
        attention_head_dim (`int`): 每个头中的通道数.
        context_pre_only (`bool`): 布尔值，决定是否添加与处理 `context` 条件相关的一些块.
    """

    # 初始化函数，定义参数
    def __init__(self, dim, num_attention_heads, attention_head_dim, context_pre_only=False):
        # 调用父类构造函数
        super().__init__()

        # 记录是否仅处理上下文
        self.context_pre_only = context_pre_only
        # 根据上下文类型设置归一化类型
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        # 创建一个适应性层归一化对象
        self.norm1 = AdaLayerNormZero(dim)

        # 根据上下文归一化类型创建相应的归一化对象
        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            # 如果归一化类型未知，抛出错误
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        # 检查是否有缩放点积注意力函数
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointAttnProcessor2_0()  # 使用相应的处理器
        else:
            # 如果不支持，抛出错误
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        # 初始化注意力模块
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
        )

        # 创建层归一化对象
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # 创建前馈网络对象
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # 如果不是仅处理上下文，则创建上下文的归一化和前馈网络
        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            # 如果仅处理上下文，则设置为 None
            self.norm2_context = None
            self.ff_context = None

        # 将块大小默认设置为 None
        self._chunk_size = None
        # 设置块维度为 0
        self._chunk_dim = 0

    # 从基本变换器块复制的方法，设置块的前馈网络
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # 设置块的前馈网络大小
        self._chunk_size = chunk_size
        # 设置块维度
        self._chunk_dim = dim
    # 定义前向传播函数，接受隐藏状态、编码器隐藏状态和时间嵌入作为输入
        def forward(
            self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor
        ):
            # 对隐藏状态进行归一化，并计算门控、多头自注意力和MLP的偏移和缩放值
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    
            # 判断是否仅使用上下文信息
            if self.context_pre_only:
                # 仅归一化编码器隐藏状态
                norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            else:
                # 对编码器隐藏状态进行归一化，并计算相应的门控和偏移值
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )
    
            # 进行注意力计算
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states
            )
    
            # 处理注意力输出以更新隐藏状态
            attn_output = gate_msa.unsqueeze(1) * attn_output  # 应用门控机制
            hidden_states = hidden_states + attn_output  # 更新隐藏状态
    
            # 对隐藏状态进行第二次归一化
            norm_hidden_states = self.norm2(hidden_states)
            # 结合缩放和偏移进行调整
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            # 如果设置了分块大小，则进行分块前馈处理以节省内存
            if self._chunk_size is not None:
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                # 否则直接执行前馈操作
                ff_output = self.ff(norm_hidden_states)
            # 应用门控机制更新前馈输出
            ff_output = gate_mlp.unsqueeze(1) * ff_output
    
            # 更新隐藏状态
            hidden_states = hidden_states + ff_output
    
            # 处理编码器隐藏状态的注意力输出
            if self.context_pre_only:
                # 如果仅使用上下文，则编码器隐藏状态设为 None
                encoder_hidden_states = None
            else:
                # 应用门控机制更新上下文注意力输出
                context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                encoder_hidden_states = encoder_hidden_states + context_attn_output  # 更新编码器隐藏状态
    
                # 对编码器隐藏状态进行第二次归一化
                norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                # 结合缩放和偏移进行调整
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                # 如果设置了分块大小，则进行分块前馈处理以节省内存
                if self._chunk_size is not None:
                    context_ff_output = _chunked_feed_forward(
                        self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                    )
                else:
                    # 否则直接执行前馈操作
                    context_ff_output = self.ff_context(norm_encoder_hidden_states)
                # 应用门控机制更新编码器隐藏状态
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    
            # 返回更新后的编码器隐藏状态和隐藏状态
            return encoder_hidden_states, hidden_states
# 装饰器，可能允许在计算图中使用此类
@maybe_allow_in_graph
# 定义一个基本的 Transformer 块，继承自 nn.Module
class BasicTransformerBlock(nn.Module):
    r"""
    一个基本的 Transformer 块。

    参数:
        dim (`int`): 输入和输出中的通道数。
        num_attention_heads (`int`): 用于多头注意力的头数。
        attention_head_dim (`int`): 每个头的通道数。
        dropout (`float`, *可选*, 默认为 0.0): 使用的丢弃概率。
        cross_attention_dim (`int`, *可选*): 用于交叉注意力的 encoder_hidden_states 向量的大小。
        activation_fn (`str`, *可选*, 默认为 `"geglu"`): 在前馈中使用的激活函数。
        num_embeds_ada_norm (:
            obj: `int`, *可选*): 在训练期间使用的扩散步骤数量。参见 `Transformer2DModel`。
        attention_bias (:
            obj: `bool`, *可选*, 默认为 `False`): 配置注意力是否应该包含偏置参数。
        only_cross_attention (`bool`, *可选*):
            是否仅使用交叉注意力层。在这种情况下使用两个交叉注意力层。
        double_self_attention (`bool`, *可选*):
            是否使用两个自注意力层。在这种情况下不使用交叉注意力层。
        upcast_attention (`bool`, *可选*):
            是否将注意力计算上溯到 float32。这对于混合精度训练很有用。
        norm_elementwise_affine (`bool`, *可选*, 默认为 `True`):
            是否为归一化使用可学习的逐元素仿射参数。
        norm_type (`str`, *可选*, 默认为 `"layer_norm"`):
            要使用的归一化层。可以是 `"layer_norm"`、`"ada_norm"` 或 `"ada_norm_zero"`。
        final_dropout (`bool`, *可选*, 默认为 False):
            是否在最后的前馈层之后应用最终的丢弃。
        attention_type (`str`, *可选*, 默认为 `"default"`):
            要使用的注意力类型。可以是 `"default"`、`"gated"` 或 `"gated-text-image"`。
        positional_embeddings (`str`, *可选*, 默认为 `None`):
            要应用的位置嵌入的类型。
        num_positional_embeddings (`int`, *可选*, 默认为 `None`):
            要应用的最大位置嵌入数量。
    """
    # 初始化方法，设置模型参数
        def __init__(
            self,
            dim: int,  # 模型维度
            num_attention_heads: int,  # 注意力头的数量
            attention_head_dim: int,  # 每个注意力头的维度
            dropout=0.0,  # dropout 概率
            cross_attention_dim: Optional[int] = None,  # 交叉注意力维度，默认为 None
            activation_fn: str = "geglu",  # 激活函数类型，默认为 'geglu'
            num_embeds_ada_norm: Optional[int] = None,  # 自适应规范化的嵌入数量
            attention_bias: bool = False,  # 是否使用注意力偏置
            only_cross_attention: bool = False,  # 是否仅使用交叉注意力
            double_self_attention: bool = False,  # 是否双重自注意力
            upcast_attention: bool = False,  # 是否提升注意力精度
            norm_elementwise_affine: bool = True,  # 是否进行逐元素仿射规范化
            norm_type: str = "layer_norm",  # 规范化类型，支持多种类型
            norm_eps: float = 1e-5,  # 规范化的 epsilon 值
            final_dropout: bool = False,  # 最终层的 dropout 开关
            attention_type: str = "default",  # 注意力机制类型，默认为 'default'
            positional_embeddings: Optional[str] = None,  # 位置嵌入的类型
            num_positional_embeddings: Optional[int] = None,  # 位置嵌入的数量
            ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,  # 自适应规范化连续条件嵌入维度
            ada_norm_bias: Optional[int] = None,  # 自适应规范化偏置
            ff_inner_dim: Optional[int] = None,  # 前馈网络的内部维度
            ff_bias: bool = True,  # 前馈网络是否使用偏置
            attention_out_bias: bool = True,  # 输出注意力是否使用偏置
        def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):  # 设置分块前馈网络的方法
            # 设置分块前馈网络的大小和维度
            self._chunk_size = chunk_size  # 存储分块大小
            self._chunk_dim = dim  # 存储维度
    
        def forward(  # 前向传播方法
            self,
            hidden_states: torch.Tensor,  # 输入的隐藏状态
            attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为 None
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态
            encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码
            timestep: Optional[torch.LongTensor] = None,  # 时间步长
            cross_attention_kwargs: Dict[str, Any] = None,  # 交叉注意力的参数
            class_labels: Optional[torch.LongTensor] = None,  # 类别标签
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,  # 添加的条件参数
# 定义一个前馈层的类，继承自 nn.Module
class LuminaFeedForward(nn.Module):
    r"""
    一个前馈层。

    参数：
        hidden_size (`int`):
            模型隐藏层的维度。该参数决定了模型隐藏表示的宽度。
        intermediate_size (`int`): 前馈层的中间维度。
        multiple_of (`int`, *optional*): 确保隐藏维度是该值的倍数。
        ffn_dim_multiplier (float, *optional*): 自定义的隐藏维度乘数。默认为 None。
    """

    # 初始化方法，接受多个参数
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 将 inner_dim 调整为原来的 2/3
        inner_dim = int(2 * inner_dim / 3)
        # 如果提供了 ffn_dim_multiplier，则调整 inner_dim
        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        # 将 inner_dim 调整为 multiple_of 的倍数
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        # 创建第一个线性层，从 dim 到 inner_dim，不使用偏置
        self.linear_1 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        # 创建第二个线性层，从 inner_dim 到 dim，不使用偏置
        self.linear_2 = nn.Linear(
            inner_dim,
            dim,
            bias=False,
        )
        # 创建第三个线性层，从 dim 到 inner_dim，不使用偏置
        self.linear_3 = nn.Linear(
            dim,
            inner_dim,
            bias=False,
        )
        # 初始化 SiLU 激活函数
        self.silu = FP32SiLU()

    # 前向传播方法，定义模型的前向计算逻辑
    def forward(self, x):
        # 依次通过线性层和激活函数计算输出
        return self.linear_2(self.silu(self.linear_1(x)) * self.linear_3(x))


# 用于图中可能允许的装饰器定义一个基本的变换器块类
@maybe_allow_in_graph
class TemporalBasicTransformerBlock(nn.Module):
    r"""
    针对视频数据的基本变换器块。

    参数：
        dim (`int`): 输入和输出中的通道数。
        time_mix_inner_dim (`int`): 用于时间注意力的通道数。
        num_attention_heads (`int`): 多头注意力使用的头数。
        attention_head_dim (`int`): 每个头中的通道数。
        cross_attention_dim (`int`, *optional*): 用于交叉注意力的 encoder_hidden_states 向量大小。
    """

    # 初始化方法，接受多个参数
    def __init__(
        self,
        dim: int,
        time_mix_inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: Optional[int] = None,
    ):
        # 初始化父类
        super().__init__()
        # 判断是否为时间混合内部维度，设置标志
        self.is_res = dim == time_mix_inner_dim

        # 创建输入层归一化层
        self.norm_in = nn.LayerNorm(dim)

        # 定义三个模块，每个模块都有自己的归一化层
        # 1. 自注意力模块
        # 创建前馈神经网络，输入维度和输出维度设置为时间混合内部维度，激活函数为 GEGLU
        self.ff_in = FeedForward(
            dim,
            dim_out=time_mix_inner_dim,
            activation_fn="geglu",
        )

        # 创建第一个归一化层
        self.norm1 = nn.LayerNorm(time_mix_inner_dim)
        # 创建自注意力层，设置查询维度、头数和头维度
        self.attn1 = Attention(
            query_dim=time_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        # 2. 交叉注意力模块
        # 检查交叉注意力维度是否为 None
        if cross_attention_dim is not None:
            # 当前仅在自注意力中使用 AdaLayerNormZero
            # 第二个交叉注意力模块返回的调制块数量没有意义
            self.norm2 = nn.LayerNorm(time_mix_inner_dim)
            # 创建交叉注意力层，设置查询维度和交叉注意力维度
            self.attn2 = Attention(
                query_dim=time_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )  # 如果 encoder_hidden_states 为 None，则为自注意力
        else:
            # 如果没有交叉注意力，归一化层和注意力层设置为 None
            self.norm2 = None
            self.attn2 = None

        # 3. 前馈神经网络模块
        # 创建第二个归一化层
        self.norm3 = nn.LayerNorm(time_mix_inner_dim)
        # 创建前馈神经网络，输入维度为时间混合内部维度，激活函数为 GEGLU
        self.ff = FeedForward(time_mix_inner_dim, activation_fn="geglu")

        # 让块大小默认为 None
        self._chunk_size = None
        # 让块维度默认为 None
        self._chunk_dim = None

    # 设置块前馈的方法，接受可选的块大小和其他参数
    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # 设置块前馈的块大小
        self._chunk_size = chunk_size
        # 块维度硬编码为 1，以获得更好的速度与内存平衡
        self._chunk_dim = 1

    # 前向传播方法，接受隐藏状态和帧数以及可选的编码器隐藏状态
    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 注意归一化始终在后续计算之前应用
        # 0. 自注意力
        # 获取批次大小，通常是隐藏状态的第一个维度
        batch_size = hidden_states.shape[0]

        # 获取批次帧数、序列长度和通道数
        batch_frames, seq_length, channels = hidden_states.shape
        # 根据帧数计算新的批次大小
        batch_size = batch_frames // num_frames

        # 调整隐藏状态形状以适应新批次大小和帧数
        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        # 改变维度顺序以便后续操作
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        # 重新调整形状为(batch_size * seq_length, num_frames, channels)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

        # 保存残差以便后续使用
        residual = hidden_states
        # 对隐藏状态应用输入归一化
        hidden_states = self.norm_in(hidden_states)

        # 如果存在分块大小，则使用分块前馈函数
        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff_in, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            # 否则，直接应用前馈函数
            hidden_states = self.ff_in(hidden_states)

        # 如果使用残差连接，则将残差添加回隐藏状态
        if self.is_res:
            hidden_states = hidden_states + residual

        # 对隐藏状态进行归一化
        norm_hidden_states = self.norm1(hidden_states)
        # 计算自注意力输出
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        # 将自注意力输出与隐藏状态相加
        hidden_states = attn_output + hidden_states

        # 3. 交叉注意力
        # 如果存在第二个注意力层，则计算交叉注意力
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            # 将交叉注意力输出与隐藏状态相加
            hidden_states = attn_output + hidden_states

        # 4. 前馈
        # 对隐藏状态进行归一化
        norm_hidden_states = self.norm3(hidden_states)

        # 如果存在分块大小，则使用分块前馈函数
        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            # 否则，直接应用前馈函数
            ff_output = self.ff(norm_hidden_states)

        # 如果使用残差连接，则将前馈输出与隐藏状态相加
        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            # 否则，仅使用前馈输出
            hidden_states = ff_output

        # 调整隐藏状态形状以适应新批次大小和帧数
        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        # 改变维度顺序以便后续操作
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        # 重新调整形状为(batch_size * num_frames, seq_length, channels)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个 SkipFFTransformerBlock 类，继承自 nn.Module
class SkipFFTransformerBlock(nn.Module):
    # 初始化方法，接收多个参数来设置层的属性
    def __init__(
        self,
        dim: int,  # 输入的特征维度
        num_attention_heads: int,  # 注意力头的数量
        attention_head_dim: int,  # 每个注意力头的维度
        kv_input_dim: int,  # 键值对输入的维度
        kv_input_dim_proj_use_bias: bool,  # 是否在 KV 映射中使用偏置
        dropout=0.0,  # dropout 比率
        cross_attention_dim: Optional[int] = None,  # 交叉注意力的维度，可选
        attention_bias: bool = False,  # 是否使用注意力偏置
        attention_out_bias: bool = True,  # 是否使用输出的注意力偏置
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果 KV 输入维度与特征维度不一致，则定义 KV 映射层
        if kv_input_dim != dim:
            self.kv_mapper = nn.Linear(kv_input_dim, dim, kv_input_dim_proj_use_bias)
        else:
            self.kv_mapper = None  # 否则不使用 KV 映射

        # 定义第一个归一化层
        self.norm1 = RMSNorm(dim, 1e-06)

        # 定义第一个注意力层
        self.attn1 = Attention(
            query_dim=dim,  # 查询的维度
            heads=num_attention_heads,  # 注意力头数量
            dim_head=attention_head_dim,  # 每个头的维度
            dropout=dropout,  # dropout 比率
            bias=attention_bias,  # 是否使用注意力偏置
            cross_attention_dim=cross_attention_dim,  # 交叉注意力的维度
            out_bias=attention_out_bias,  # 输出是否使用偏置
        )

        # 定义第二个归一化层
        self.norm2 = RMSNorm(dim, 1e-06)

        # 定义第二个注意力层
        self.attn2 = Attention(
            query_dim=dim,  # 查询的维度
            cross_attention_dim=cross_attention_dim,  # 交叉注意力的维度
            heads=num_attention_heads,  # 注意力头数量
            dim_head=attention_head_dim,  # 每个头的维度
            dropout=dropout,  # dropout 比率
            bias=attention_bias,  # 是否使用注意力偏置
            out_bias=attention_out_bias,  # 输出是否使用偏置
        )

    # 前向传播方法
    def forward(self, hidden_states, encoder_hidden_states, cross_attention_kwargs):
        # 复制交叉注意力的参数，如果没有则初始化为空字典
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        # 如果存在 KV 映射层，则对编码器的隐藏状态进行映射
        if self.kv_mapper is not None:
            encoder_hidden_states = self.kv_mapper(F.silu(encoder_hidden_states))

        # 对输入的隐藏状态进行归一化
        norm_hidden_states = self.norm1(hidden_states)

        # 计算第一个注意力层的输出
        attn_output = self.attn1(
            norm_hidden_states,  # 归一化后的隐藏状态
            encoder_hidden_states=encoder_hidden_states,  # 编码器的隐藏状态
            **cross_attention_kwargs,  # 其他交叉注意力的参数
        )

        # 更新隐藏状态
        hidden_states = attn_output + hidden_states

        # 对更新后的隐藏状态进行第二次归一化
        norm_hidden_states = self.norm2(hidden_states)

        # 计算第二个注意力层的输出
        attn_output = self.attn2(
            norm_hidden_states,  # 归一化后的隐藏状态
            encoder_hidden_states=encoder_hidden_states,  # 编码器的隐藏状态
            **cross_attention_kwargs,  # 其他交叉注意力的参数
        )

        # 更新隐藏状态
        hidden_states = attn_output + hidden_states

        # 返回最终的隐藏状态
        return hidden_states


# 定义一个 FreeNoiseTransformerBlock 类，继承自 nn.Module
@maybe_allow_in_graph
class FreeNoiseTransformerBlock(nn.Module):
    r"""
    A FreeNoise Transformer block.  # FreeNoise Transformer 块的文档字符串

    """
    # 初始化方法，设置模型的各种参数
        def __init__(
            # 模型的维度
            self,
            dim: int,
            # 注意力头的数量
            num_attention_heads: int,
            # 每个注意力头的维度
            attention_head_dim: int,
            # dropout 概率，默认为 0.0
            dropout: float = 0.0,
            # 交叉注意力的维度，默认为 None
            cross_attention_dim: Optional[int] = None,
            # 激活函数的名称，默认为 "geglu"
            activation_fn: str = "geglu",
            # 自适应归一化的嵌入数量，默认为 None
            num_embeds_ada_norm: Optional[int] = None,
            # 是否使用注意力偏差，默认为 False
            attention_bias: bool = False,
            # 是否仅使用交叉注意力，默认为 False
            only_cross_attention: bool = False,
            # 是否使用双重自注意力，默认为 False
            double_self_attention: bool = False,
            # 是否上溯注意力，默认为 False
            upcast_attention: bool = False,
            # 归一化是否使用逐元素仿射，默认为 True
            norm_elementwise_affine: bool = True,
            # 归一化的类型，默认为 "layer_norm"
            norm_type: str = "layer_norm",
            # 归一化的 epsilon 值，默认为 1e-5
            norm_eps: float = 1e-5,
            # 最终是否使用 dropout，默认为 False
            final_dropout: bool = False,
            # 位置嵌入的类型，默认为 None
            positional_embeddings: Optional[str] = None,
            # 位置嵌入的数量，默认为 None
            num_positional_embeddings: Optional[int] = None,
            # 前馈网络内部维度，默认为 None
            ff_inner_dim: Optional[int] = None,
            # 前馈网络是否使用偏差，默认为 True
            ff_bias: bool = True,
            # 注意力输出是否使用偏差，默认为 True
            attention_out_bias: bool = True,
            # 上下文长度，默认为 16
            context_length: int = 16,
            # 上下文步幅，默认为 4
            context_stride: int = 4,
            # 权重方案，默认为 "pyramid"
            weighting_scheme: str = "pyramid",
        # 获取帧索引的方法，返回一对帧索引的列表
        def _get_frame_indices(self, num_frames: int) -> List[Tuple[int, int]]:
            # 初始化帧索引列表
            frame_indices = []
            # 遍历所有帧，步幅为上下文步幅
            for i in range(0, num_frames - self.context_length + 1, self.context_stride):
                # 当前窗口的起始帧
                window_start = i
                # 当前窗口的结束帧，确保不超过总帧数
                window_end = min(num_frames, i + self.context_length)
                # 将窗口索引添加到列表
                frame_indices.append((window_start, window_end))
            # 返回帧索引列表
            return frame_indices
    
        # 获取帧权重的方法，返回权重列表
        def _get_frame_weights(self, num_frames: int, weighting_scheme: str = "pyramid") -> List[float]:
            # 如果权重方案为 "pyramid"
            if weighting_scheme == "pyramid":
                # 判断帧数是否为偶数
                if num_frames % 2 == 0:
                    # 生成偶数帧的权重列表
                    weights = list(range(1, num_frames // 2 + 1))
                    # 反转并连接权重列表
                    weights = weights + weights[::-1]
                else:
                    # 生成奇数帧的权重列表
                    weights = list(range(1, num_frames // 2 + 1))
                    # 添加中间权重并反转连接
                    weights = weights + [num_frames // 2 + 1] + weights[::-1]
            else:
                # 抛出不支持的权重方案错误
                raise ValueError(f"Unsupported value for weighting_scheme={weighting_scheme}")
    
            # 返回权重列表
            return weights
    
        # 设置自由噪声属性的方法，无返回值
        def set_free_noise_properties(
            self, context_length: int, context_stride: int, weighting_scheme: str = "pyramid"
        ) -> None:
            # 设置上下文长度
            self.context_length = context_length
            # 设置上下文步幅
            self.context_stride = context_stride
            # 设置权重方案
            self.weighting_scheme = weighting_scheme
    
        # 设置块前馈的方法，无返回值
        def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0) -> None:
            # 设置块大小和维度
            self._chunk_size = chunk_size
            self._chunk_dim = dim
    
        # 前向传播方法，处理输入隐藏状态
        def forward(
            self,
            # 输入的隐藏状态张量
            hidden_states: torch.Tensor,
            # 可选的注意力掩码
            attention_mask: Optional[torch.Tensor] = None,
            # 可选的编码器隐藏状态
            encoder_hidden_states: Optional[torch.Tensor] = None,
            # 可选的编码器注意力掩码
            encoder_attention_mask: Optional[torch.Tensor] = None,
            # 交叉注意力的额外参数
            cross_attention_kwargs: Dict[str, Any] = None,
            # 可变参数
            *args,
            # 关键字参数
            **kwargs,
# 定义一个前馈层类，继承自 nn.Module
class FeedForward(nn.Module):
    r"""
    前馈层。

    参数:
        dim (`int`): 输入的通道数。
        dim_out (`int`, *可选*): 输出的通道数。如果未给定，默认为 `dim`。
        mult (`int`, *可选*, 默认为 4): 用于隐藏维度的乘数。
        dropout (`float`, *可选*, 默认为 0.0): 使用的 dropout 概率。
        activation_fn (`str`, *可选*, 默认为 `"geglu"`): 前馈中使用的激活函数。
        final_dropout (`bool` *可选*, 默认为 False): 是否应用最终的 dropout。
        bias (`bool`, 默认为 True): 是否在线性层中使用偏置。
    """

    # 初始化方法
    def __init__(
        self,
        dim: int,  # 输入通道数
        dim_out: Optional[int] = None,  # 输出通道数（可选）
        mult: int = 4,  # 隐藏维度乘数
        dropout: float = 0.0,  # dropout 概率
        activation_fn: str = "geglu",  # 激活函数类型
        final_dropout: bool = False,  # 是否应用最终 dropout
        inner_dim=None,  # 隐藏层维度（可选）
        bias: bool = True,  # 是否使用偏置
    ):
        # 调用父类构造函数
        super().__init__()
        # 如果未指定 inner_dim，则计算为 dim 乘以 mult
        if inner_dim is None:
            inner_dim = int(dim * mult)
        # 如果未指定 dim_out，则设置为 dim
        dim_out = dim_out if dim_out is not None else dim

        # 根据选择的激活函数创建对应的激活层
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)  # GELU 激活函数
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)  # 近似 GELU
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)  # GEGLU 激活函数
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)  # 近似 GEGLU
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)  # SwiGLU 激活函数

        # 初始化一个模块列表，用于存储层
        self.net = nn.ModuleList([])
        # 添加激活函数层到网络
        self.net.append(act_fn)
        # 添加 dropout 层到网络
        self.net.append(nn.Dropout(dropout))
        # 添加线性层到网络，输入为 inner_dim，输出为 dim_out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # 如果 final_dropout 为真，则添加最终的 dropout 层
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 检查是否有额外的参数，或是否传递了过期的 scale 参数
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)  # 发出关于 scale 参数的弃用警告
        # 遍历网络中的每一层，依次对 hidden_states 进行处理
        for module in self.net:
            hidden_states = module(hidden_states)  # 将当前层应用于输入
        # 返回处理后的隐藏状态
        return hidden_states
```