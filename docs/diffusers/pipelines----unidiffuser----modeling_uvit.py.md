# `.\diffusers\pipelines\unidiffuser\modeling_uvit.py`

```py
# 导入数学库
import math
# 从 typing 模块导入可选和联合类型
from typing import Optional, Union

# 导入 PyTorch 库
import torch
# 从 torch 模块导入神经网络相关类
from torch import nn

# 导入配置混合器和注册配置的工具
from ...configuration_utils import ConfigMixin, register_to_config
# 导入模型混合器
from ...models import ModelMixin
# 从注意力模型导入前馈网络
from ...models.attention import FeedForward
# 从注意力处理器导入注意力机制
from ...models.attention_processor import Attention
# 从嵌入模型导入时间步嵌入、时间步和获取二维正弦余弦位置嵌入的函数
from ...models.embeddings import TimestepEmbedding, Timesteps, get_2d_sincos_pos_embed
# 从建模输出导入 Transformer2DModelOutput
from ...models.modeling_outputs import Transformer2DModelOutput
# 从规范化模型导入自适应层归一化
from ...models.normalization import AdaLayerNorm
# 导入日志工具
from ...utils import logging

# 创建一个名为 __name__ 的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义一个不带梯度的截断正态分布初始化函数
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # 从 PyTorch 官方库复制的函数，直到在几个正式版本中被纳入 - RW
    # 基于 https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf 的方法
    def norm_cdf(x):
        # 计算标准正态累积分布函数
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    # 检查均值是否在区间外，若是则发出警告
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        logger.warning(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect."
        )

    # 在不计算梯度的上下文中执行以下操作
    with torch.no_grad():
        # 通过使用截断均匀分布生成值，然后使用正态分布的逆CDF转换
        # 获取上下累积分布函数值
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # 在区间 [l, u] 内均匀填充张量，然后转换为 [2l-1, 2u-1]。
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # 使用逆CDF变换获取截断的标准正态分布
        tensor.erfinv_()

        # 转换为指定的均值和标准差
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # 限制张量值确保在适当范围内
        tensor.clamp_(min=a, max=b)
        # 返回处理后的张量
        return tensor

# 定义截断正态分布初始化的公共接口函数
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # 指定参数类型
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""用从截断正态分布中抽取的值填充输入张量。值实际上是从正态分布 :math:`\mathcal{N}(\text{mean},
    \text{std}^2)` 中抽取的，超出 :math:`[a, b]` 的值会重新抽取，直到它们在范围内。用于生成随机值的方法在 :math:`a \leq \text{mean} \leq b` 时效果最佳。

    参数：
        tensor: n 维的 `torch.Tensor`
        mean: 正态分布的均值
        std: 正态分布的标准差
        a: 最小截断值
        b: 最大截断值
    示例：
        >>> w = torch.empty(3, 5) >>> nn.init.trunc_normal_(w)
    """
    # 调用内部函数生成截断正态分布值
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# 定义一个图像到补丁嵌入的模块类
class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    # 初始化类的构造函数
        def __init__(
            # 图像的高度，默认为224
            height=224,
            # 图像的宽度，默认为224
            width=224,
            # 每个patch的大小，默认为16
            patch_size=16,
            # 输入通道数，默认为3（RGB图像）
            in_channels=3,
            # 嵌入维度，默认为768
            embed_dim=768,
            # 是否使用层归一化，默认为False
            layer_norm=False,
            # 是否将输入展平，默认为True
            flatten=True,
            # 卷积是否使用偏置，默认为True
            bias=True,
            # 是否使用位置嵌入，默认为True
            use_pos_embed=True,
        ):
            # 调用父类的构造函数
            super().__init__()
    
            # 计算总patch的数量
            num_patches = (height // patch_size) * (width // patch_size)
            # 存储是否展平的标志
            self.flatten = flatten
            # 存储是否使用层归一化的标志
            self.layer_norm = layer_norm
    
            # 创建卷积层用于特征提取
            self.proj = nn.Conv2d(
                # 输入通道数
                in_channels, 
                # 输出嵌入维度
                embed_dim, 
                # 卷积核的大小
                kernel_size=(patch_size, patch_size), 
                # 步幅等于patch_size
                stride=patch_size, 
                # 是否使用偏置
                bias=bias
            )
            # 如果使用层归一化，则初始化层归一化对象
            if layer_norm:
                self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
            # 否则将归一化对象设置为None
            else:
                self.norm = None
    
            # 存储是否使用位置嵌入的标志
            self.use_pos_embed = use_pos_embed
            # 如果使用位置嵌入，生成并注册位置嵌入
            if self.use_pos_embed:
                pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches**0.5))
                # 将位置嵌入注册为模型的缓冲区
                self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
    
        # 定义前向传播方法
        def forward(self, latent):
            # 通过卷积层处理输入数据
            latent = self.proj(latent)
            # 如果需要展平，执行展平和转置操作
            if self.flatten:
                latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # 如果使用层归一化，则应用层归一化
            if self.layer_norm:
                latent = self.norm(latent)
            # 如果使用位置嵌入，则返回加上位置嵌入的结果
            if self.use_pos_embed:
                return latent + self.pos_embed
            # 否则返回处理后的结果
            else:
                return latent
# 定义一个名为 SkipBlock 的类，继承自 nn.Module
class SkipBlock(nn.Module):
    # 初始化方法，接收一个整数参数 dim
    def __init__(self, dim: int):
        # 调用父类的初始化方法
        super().__init__()

        # 定义一个线性变换层，输入维度为 2 * dim，输出维度为 dim
        self.skip_linear = nn.Linear(2 * dim, dim)

        # 使用 torch.nn.LayerNorm 进行层归一化，处理维度为 dim 的张量
        self.norm = nn.LayerNorm(dim)

    # 前向传播方法，接收输入 x 和跳跃连接 skip
    def forward(self, x, skip):
        # 将 x 和 skip 沿最后一个维度连接，并通过线性变换层处理
        x = self.skip_linear(torch.cat([x, skip], dim=-1))
        # 对处理后的张量进行层归一化
        x = self.norm(x)

        # 返回处理后的结果
        return x


# 定义一个名为 UTransformerBlock 的类，继承自 nn.Module
# 这是对 BasicTransformerBlock 的修改，支持 pre-LayerNorm 和 post-LayerNorm 配置
class UTransformerBlock(nn.Module):
    r"""
    对 BasicTransformerBlock 的修改，支持 pre-LayerNorm 和 post-LayerNorm 配置。

    参数：
        dim (`int`): 输入和输出的通道数。
        num_attention_heads (`int`): 用于多头注意力的头数。
        attention_head_dim (`int`): 每个头的通道数。
        dropout (`float`, *可选*, 默认值为 0.0): 使用的 dropout 概率。
        cross_attention_dim (`int`, *可选*): 用于交叉注意力的 encoder_hidden_states 向量的大小。
        activation_fn (`str`, *可选*, 默认值为 `"geglu"`):
            在前馈网络中使用的激活函数。
        num_embeds_ada_norm (:obj: `int`, *可选*):
            训练期间使用的扩散步骤数。参见 `Transformer2DModel`。
        attention_bias (:obj: `bool`, *可选*, 默认值为 `False`):
            配置注意力是否包含偏置参数。
        only_cross_attention (`bool`, *可选*):
            是否仅使用交叉注意力层。在这种情况下使用两个交叉注意力层。
        double_self_attention (`bool`, *可选*):
            是否使用两个自注意力层。在这种情况下不使用交叉注意力层。
        upcast_attention (`bool`, *可选*):
            在执行注意力计算时，是否将查询和键的类型提升为 float32。
        norm_elementwise_affine (`bool`, *可选*):
            在层归一化期间是否使用可学习的逐元素仿射参数。
        norm_type (`str`, 默认值为 `"layer_norm"`):
            使用的层归一化实现类型。
        pre_layer_norm (`bool`, *可选*):
            是否在注意力和前馈操作之前执行层归一化（"pre-LayerNorm"），
            而不是之后（"post-LayerNorm"）。注意 `BasicTransformerBlock` 使用 pre-LayerNorm，例如
            `pre_layer_norm = True`。
        final_dropout (`bool`, *可选*):
            是否在前馈网络后使用最终的 Dropout 层。
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
            # dropout 概率，默认值为 0.0
            dropout=0.0,
            # 交叉注意力的维度，可选
            cross_attention_dim: Optional[int] = None,
            # 激活函数的类型，默认为 "geglu"
            activation_fn: str = "geglu",
            # 可选的自适应归一化的嵌入数量
            num_embeds_ada_norm: Optional[int] = None,
            # 是否使用注意力偏置
            attention_bias: bool = False,
            # 是否仅使用交叉注意力
            only_cross_attention: bool = False,
            # 是否双重自注意力
            double_self_attention: bool = False,
            # 是否提升注意力精度
            upcast_attention: bool = False,
            # 归一化时是否使用元素级仿射变换
            norm_elementwise_affine: bool = True,
            # 归一化的类型，默认为 "layer_norm"
            norm_type: str = "layer_norm",
            # 是否使用预层归一化
            pre_layer_norm: bool = True,
            # 是否在最终阶段使用 dropout
            final_dropout: bool = False,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 仅使用交叉注意力的标志
        self.only_cross_attention = only_cross_attention

        # 确定是否使用 AdaLayerNorm，依据 num_embeds_ada_norm 和 norm_type
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        # 预先进行层归一化的标志
        self.pre_layer_norm = pre_layer_norm

        # 如果 norm_type 是 "ada_norm" 或 "ada_norm_zero"，且未定义 num_embeds_ada_norm，抛出错误
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # 1. 自注意力层
        self.attn1 = Attention(
            # 查询向量维度
            query_dim=dim,
            # 注意力头的数量
            heads=num_attention_heads,
            # 每个注意力头的维度
            dim_head=attention_head_dim,
            # Dropout 比例
            dropout=dropout,
            # 是否使用偏置
            bias=attention_bias,
            # 交叉注意力的维度（仅在只使用交叉注意力时设定）
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            # 是否上溯注意力计算
            upcast_attention=upcast_attention,
        )

        # 2. 交叉注意力层
        if cross_attention_dim is not None or double_self_attention:
            self.attn2 = Attention(
                # 查询向量维度
                query_dim=dim,
                # 交叉注意力的维度（在双自注意力的情况下设为 None）
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                # 注意力头的数量
                heads=num_attention_heads,
                # 每个注意力头的维度
                dim_head=attention_head_dim,
                # Dropout 比例
                dropout=dropout,
                # 是否使用偏置
                bias=attention_bias,
                # 是否上溯注意力计算
                upcast_attention=upcast_attention,
            )  # 如果 encoder_hidden_states 为 None 则视为自注意力
        else:
            # 若不需要交叉注意力，则将其设置为 None
            self.attn2 = None

        # 根据是否使用 AdaLayerNorm 来选择层归一化的实现
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            # 使用标准的层归一化
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        # 如果有交叉注意力维度或使用双自注意力
        if cross_attention_dim is not None or double_self_attention:
            # 目前只在自注意力中使用 AdaLayerNormZero，因为只有一个注意力块
            # 如果在第二个交叉注意力块返回的调制块数目将没有意义
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
        else:
            # 如果没有交叉注意力，则将其设置为 None
            self.norm2 = None

        # 3. 前馈层
        # 对前馈层的输出进行标准层归一化
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        # 初始化前馈神经网络
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

    def forward(
        # 输入的隐藏状态
        hidden_states,
        # 注意力掩码
        attention_mask=None,
        # 编码器的隐藏状态
        encoder_hidden_states=None,
        # 编码器的注意力掩码
        encoder_attention_mask=None,
        # 时间步
        timestep=None,
        # 交叉注意力的额外参数
        cross_attention_kwargs=None,
        # 类别标签
        class_labels=None,
    ):
        # 预处理层归一化
        if self.pre_layer_norm:
            # 如果使用自适应层归一化，则传递时间步
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            else:
                # 否则直接进行层归一化
                norm_hidden_states = self.norm1(hidden_states)
        else:
            # 如果不使用预处理层归一化，直接使用输入的隐藏状态
            norm_hidden_states = hidden_states

        # 1. 自注意力机制
        # 如果没有提供交叉注意力的参数，则使用空字典
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # 进行自注意力计算，可能会传入编码器的隐藏状态和注意力掩码
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        # 后处理层归一化
        if not self.pre_layer_norm:
            # 如果不使用预处理层归一化，进行归一化处理
            if self.use_ada_layer_norm:
                attn_output = self.norm1(attn_output, timestep)
            else:
                attn_output = self.norm1(attn_output)

        # 将自注意力的输出与输入的隐藏状态相加
        hidden_states = attn_output + hidden_states

        if self.attn2 is not None:
            # 预处理层归一化
            if self.pre_layer_norm:
                # 如果使用自适应层归一化，则传递时间步
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
            else:
                # 否则直接使用输入的隐藏状态
                norm_hidden_states = hidden_states
            # TODO (Birch-San): 这里应该正确准备编码器注意力掩码
            # 准备注意力掩码

            # 2. 交叉注意力机制
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )

            # 后处理层归一化
            if not self.pre_layer_norm:
                attn_output = self.norm2(attn_output, timestep) if self.use_ada_layer_norm else self.norm2(attn_output)

            # 将交叉注意力的输出与输入的隐藏状态相加
            hidden_states = attn_output + hidden_states

        # 3. 前馈神经网络
        # 预处理层归一化
        if self.pre_layer_norm:
            norm_hidden_states = self.norm3(hidden_states)
        else:
            # 否则直接使用输入的隐藏状态
            norm_hidden_states = hidden_states

        # 进行前馈神经网络计算
        ff_output = self.ff(norm_hidden_states)

        # 后处理层归一化
        if not self.pre_layer_norm:
            ff_output = self.norm3(ff_output)

        # 将前馈神经网络的输出与输入的隐藏状态相加
        hidden_states = ff_output + hidden_states

        # 返回最终的隐藏状态
        return hidden_states
# 类似于 UTransformerBlock，但在块的残差路径上使用 LayerNorm
# 从 diffusers.models.attention.BasicTransformerBlock 修改而来
class UniDiffuserBlock(nn.Module):
    r"""
    对 BasicTransformerBlock 的修改，支持 pre-LayerNorm 和 post-LayerNorm 配置，并将
    LayerNorm 应用在块的残差路径上。这与 [original UniDiffuser
    implementation](https://github.com/thu-ml/unidiffuser/blob/main/libs/uvit_multi_post_ln_v1.py#L104) 中的 transformer 块相匹配。

    参数：
        dim (`int`): 输入和输出的通道数。
        num_attention_heads (`int`): 用于多头注意力的头数。
        attention_head_dim (`int`): 每个头的通道数。
        dropout (`float`, *可选*, 默认为 0.0): 使用的丢弃概率。
        cross_attention_dim (`int`, *可选*): 跨注意力的 encoder_hidden_states 向量的大小。
        activation_fn (`str`, *可选*, 默认为 `"geglu"`):
            在前馈网络中使用的激活函数。
        num_embeds_ada_norm (:obj: `int`, *可选*):
            训练期间使用的扩散步骤数量。见 `Transformer2DModel`。
        attention_bias (:obj: `bool`, *可选*, 默认为 `False`):
            配置注意力是否包含偏置参数。
        only_cross_attention (`bool`, *可选*):
            是否仅使用跨注意力层。在这种情况下，使用两个跨注意力层。
        double_self_attention (`bool`, *可选*):
            是否使用两个自注意力层。在这种情况下，不使用跨注意力层。
        upcast_attention (`bool`, *可选*):
            在执行注意力计算时，是否将查询和键上升到 float() 类型。
        norm_elementwise_affine (`bool`, *可选*):
            在层归一化期间，是否使用可学习的逐元素仿射参数。
        norm_type (`str`, 默认为 `"layer_norm"`):
            使用的层归一化实现。
        pre_layer_norm (`bool`, *可选*):
            是否在注意力和前馈操作之前执行层归一化（“pre-LayerNorm”），
            而不是之后（“post-LayerNorm”）。原始 UniDiffuser 实现是 post-LayerNorm
            (`pre_layer_norm = False`)。
        final_dropout (`bool`, *可选*):
            在前馈网络之后是否使用最终的 Dropout 层。
    """
    # 初始化方法，用于设置模型的基本参数
        def __init__(
            # 模型的维度
            self,
            dim: int,
            # 注意力头的数量
            num_attention_heads: int,
            # 每个注意力头的维度
            attention_head_dim: int,
            # dropout 概率，默认值为 0.0
            dropout=0.0,
            # 可选的交叉注意力维度
            cross_attention_dim: Optional[int] = None,
            # 激活函数的类型，默认使用 "geglu"
            activation_fn: str = "geglu",
            # 可选的自适应归一化的嵌入数量
            num_embeds_ada_norm: Optional[int] = None,
            # 是否使用注意力偏置，默认值为 False
            attention_bias: bool = False,
            # 是否仅使用交叉注意力，默认值为 False
            only_cross_attention: bool = False,
            # 是否使用双重自注意力，默认值为 False
            double_self_attention: bool = False,
            # 是否上溯注意力，默认值为 False
            upcast_attention: bool = False,
            # 归一化时是否使用元素-wise 仿射变换，默认值为 True
            norm_elementwise_affine: bool = True,
            # 归一化的类型，默认使用 "layer_norm"
            norm_type: str = "layer_norm",
            # 是否在前面使用层归一化，默认值为 False
            pre_layer_norm: bool = False,
            # 最终是否使用 dropout，默认值为 True
            final_dropout: bool = True,
    ):
        # 初始化父类
        super().__init__()
        # 设置是否仅使用交叉注意力
        self.only_cross_attention = only_cross_attention

        # 判断是否使用自适应层归一化
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        # 设置预层归一化
        self.pre_layer_norm = pre_layer_norm

        # 检查归一化类型和自适应嵌入数量的有效性
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                # 抛出异常信息，提示未定义自适应嵌入数量
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # 1. 自注意力层
        self.attn1 = Attention(
            # 设置查询维度、头数、头维度和其他参数
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            # 根据是否仅使用交叉注意力选择交叉注意力维度
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. 交叉注意力层
        if cross_attention_dim is not None or double_self_attention:
            self.attn2 = Attention(
                # 设置查询和交叉注意力的维度及其他参数
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # 如果 encoder_hidden_states 为 None，则为自注意力
        else:
            # 如果没有交叉注意力维度，设置为 None
            self.attn2 = None

        # 如果使用自适应层归一化，初始化相应的归一化层
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            # 否则使用标准层归一化
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

        # 如果有交叉注意力维度或双自注意力
        if cross_attention_dim is not None or double_self_attention:
            # 目前仅在自注意力中使用 AdaLayerNormZero
            self.norm2 = (
                # 根据是否使用自适应层归一化选择归一化层
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
        else:
            # 如果没有交叉注意力，则设置为 None
            self.norm2 = None

        # 3. 前馈层
        # 初始化第三层的标准层归一化
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        # 初始化前馈网络
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

    def forward(
        # 定义前向传播函数的输入参数
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        # 按照 diffusers transformer block 实现，将 LayerNorm 放在残差连接上
        # 预 LayerNorm
        if self.pre_layer_norm:
            # 如果使用自适应 LayerNorm，应用它
            if self.use_ada_layer_norm:
                hidden_states = self.norm1(hidden_states, timestep)
            else:
                # 否则，直接应用 LayerNorm
                hidden_states = self.norm1(hidden_states)

        # 1. 自注意力
        # 如果 cross_attention_kwargs 为 None，则初始化为空字典
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        # 执行自注意力操作，获取输出
        attn_output = self.attn1(
            hidden_states,
            # 根据条件选择 encoder_hidden_states
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            # 应用注意力掩码
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        # 将自注意力输出与输入隐藏状态相加
        hidden_states = attn_output + hidden_states

        # 按照 diffusers transformer block 实现，将 LayerNorm 放在残差连接上
        # 后 LayerNorm
        if not self.pre_layer_norm:
            # 如果使用自适应 LayerNorm，应用它
            if self.use_ada_layer_norm:
                hidden_states = self.norm1(hidden_states, timestep)
            else:
                # 否则，直接应用 LayerNorm
                hidden_states = self.norm1(hidden_states)

        # 如果 attn2 存在
        if self.attn2 is not None:
            # 预 LayerNorm
            if self.pre_layer_norm:
                # 根据条件应用 norm2
                hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
            # TODO (Birch-San): 这里应该正确准备 encoder_attention 掩码
            # 在这里准备注意力掩码

            # 2. 跨注意力
            # 执行跨注意力操作，获取输出
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                # 应用 encoder 的注意力掩码
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )

            # 将跨注意力输出与输入隐藏状态相加
            hidden_states = attn_output + hidden_states

            # 后 LayerNorm
            if not self.pre_layer_norm:
                hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

        # 3. 前馈网络
        # 预 LayerNorm
        if self.pre_layer_norm:
            # 应用 norm3
            hidden_states = self.norm3(hidden_states)

        # 通过前馈网络获取输出
        ff_output = self.ff(hidden_states)

        # 将前馈网络输出与输入隐藏状态相加
        hidden_states = ff_output + hidden_states

        # 后 LayerNorm
        if not self.pre_layer_norm:
            # 应用 norm3
            hidden_states = self.norm3(hidden_states)

        # 返回最终的隐藏状态
        return hidden_states
# 从 diffusers.models.transformer_2d.Transformer2DModel 修改而来
# 修改变换块结构，使其类似 U-Net，遵循 U-ViT 的设计
# 目前仅支持补丁样式输入和 torch.nn.LayerNorm
# 相关链接: https://github.com/baofff/U-ViT
class UTransformer2DModel(ModelMixin, ConfigMixin):
    """
    基于 [U-ViT](https://github.com/baofff/U-ViT) 架构的变换器模型，适用于图像数据。
    与 [`Transformer2DModel`] 相比，此模型在变换块之间具有跳跃连接，以 "U" 形状连接，
    类似于 U-Net。仅支持连续（实际嵌入）输入，这些输入通过 [`PatchEmbed`] 层嵌入，
    然后重塑为 (b, t, d) 的形状。
    """

    @register_to_config
    # 初始化 UTransformer2DModel 的构造函数
    def __init__(
        # 注意力头的数量，默认为 16
        num_attention_heads: int = 16,
        # 每个注意力头的维度，默认为 88
        attention_head_dim: int = 88,
        # 输入通道数，默认为 None
        in_channels: Optional[int] = None,
        # 输出通道数，默认为 None
        out_channels: Optional[int] = None,
        # 变换层的数量，默认为 1
        num_layers: int = 1,
        # dropout 的比例，默认为 0.0
        dropout: float = 0.0,
        # 规范化时的组数量，默认为 32
        norm_num_groups: int = 32,
        # 跨注意力维度，默认为 None
        cross_attention_dim: Optional[int] = None,
        # 是否使用注意力偏置，默认为 False
        attention_bias: bool = False,
        # 样本大小，默认为 None
        sample_size: Optional[int] = None,
        # 向量嵌入的数量，默认为 None
        num_vector_embeds: Optional[int] = None,
        # 补丁大小，默认为 2
        patch_size: Optional[int] = 2,
        # 激活函数，默认为 "geglu"
        activation_fn: str = "geglu",
        # 自适应规范化的嵌入数量，默认为 None
        num_embeds_ada_norm: Optional[int] = None,
        # 是否使用线性投影，默认为 False
        use_linear_projection: bool = False,
        # 是否仅使用跨注意力，默认为 False
        only_cross_attention: bool = False,
        # 是否上溯注意力，默认为 False
        upcast_attention: bool = False,
        # 规范化类型，默认为 "layer_norm"
        norm_type: str = "layer_norm",
        # 块类型，默认为 "unidiffuser"
        block_type: str = "unidiffuser",
        # 是否使用预层规范化，默认为 False
        pre_layer_norm: bool = False,
        # 规范化时是否使用元素级的仿射，默认为 True
        norm_elementwise_affine: bool = True,
        # 是否使用补丁位置嵌入，默认为 False
        use_patch_pos_embed=False,
        # 前馈层的最终 dropout，默认为 False
        ff_final_dropout: bool = False,
    # 前向传播函数定义
    def forward(
        # 隐藏状态输入
        hidden_states,
        # 编码器隐藏状态，默认为 None
        encoder_hidden_states=None,
        # 时间步，默认为 None
        timestep=None,
        # 类别标签，默认为 None
        class_labels=None,
        # 跨注意力相关参数，默认为 None
        cross_attention_kwargs=None,
        # 是否返回字典格式的输出，默认为 True
        return_dict: bool = True,
        # 隐藏状态是否为嵌入，默认为 False
        hidden_states_is_embedding: bool = False,
        # 是否进行反补丁操作，默认为 True
        unpatchify: bool = True,
class UniDiffuserModel(ModelMixin, ConfigMixin):
    """
    图像-文本 [UniDiffuser](https://arxiv.org/pdf/2303.06555.pdf) 模型的变换器模型。
    这是 [`UTransformer2DModel`] 的修改版本，具有用于 VAE 嵌入潜图像、CLIP 嵌入图像和 CLIP 嵌入提示的输入和输出头（详见论文）。
    """

    @register_to_config
    # 初始化方法，用于设置类的属性和参数
        def __init__(
            # 文本的维度，默认为768
            text_dim: int = 768,
            # CLIP图像的维度，默认为512
            clip_img_dim: int = 512,
            # 文本标记的数量，默认为77
            num_text_tokens: int = 77,
            # 注意力头的数量，默认为16
            num_attention_heads: int = 16,
            # 每个注意力头的维度，默认为88
            attention_head_dim: int = 88,
            # 输入通道的数量，可选
            in_channels: Optional[int] = None,
            # 输出通道的数量，可选
            out_channels: Optional[int] = None,
            # 网络层的数量，默认为1
            num_layers: int = 1,
            # dropout比率，默认为0.0
            dropout: float = 0.0,
            # 规范化的组数量，默认为32
            norm_num_groups: int = 32,
            # 跨注意力的维度，可选
            cross_attention_dim: Optional[int] = None,
            # 注意力偏差，默认为False
            attention_bias: bool = False,
            # 采样大小，可选
            sample_size: Optional[int] = None,
            # 向量嵌入的数量，可选
            num_vector_embeds: Optional[int] = None,
            # 图像块的大小，可选
            patch_size: Optional[int] = None,
            # 激活函数，默认为"geglu"
            activation_fn: str = "geglu",
            # 自适应规范化嵌入的数量，可选
            num_embeds_ada_norm: Optional[int] = None,
            # 是否使用线性投影，默认为False
            use_linear_projection: bool = False,
            # 仅使用跨注意力，默认为False
            only_cross_attention: bool = False,
            # 是否上调注意力，默认为False
            upcast_attention: bool = False,
            # 规范化类型，默认为"layer_norm"
            norm_type: str = "layer_norm",
            # 块类型，默认为"unidiffuser"
            block_type: str = "unidiffuser",
            # 是否使用预层规范化，默认为False
            pre_layer_norm: bool = False,
            # 是否使用时间步嵌入，默认为False
            use_timestep_embedding=False,
            # 规范化的元素逐项仿射，默认为True
            norm_elementwise_affine: bool = True,
            # 是否使用块位置嵌入，默认为False
            use_patch_pos_embed=False,
            # 前馈层的最终dropout，默认为True
            ff_final_dropout: bool = True,
            # 是否使用数据类型嵌入，默认为False
            use_data_type_embedding: bool = False,
        # 装饰器，表示该方法在Torch JIT编译时会被忽略
        @torch.jit.ignore
        def no_weight_decay(self):
            # 返回不需要权重衰减的参数
            return {"pos_embed"}
    
        # 前向传播方法，定义输入和计算流程
        def forward(
            # 潜在图像嵌入的张量
            latent_image_embeds: torch.Tensor,
            # 图像嵌入的张量
            image_embeds: torch.Tensor,
            # 提示嵌入的张量
            prompt_embeds: torch.Tensor,
            # 时间步图像的张量或数值
            timestep_img: Union[torch.Tensor, float, int],
            # 时间步文本的张量或数值
            timestep_text: Union[torch.Tensor, float, int],
            # 数据类型，可选，默认为1
            data_type: Optional[Union[torch.Tensor, float, int]] = 1,
            # 编码器隐藏状态，默认为None
            encoder_hidden_states=None,
            # 跨注意力的额外参数，默认为None
            cross_attention_kwargs=None,
```