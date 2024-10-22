# `.\diffusers\models\transformers\transformer_flux.py`

```py
# 版权声明，标明版权归属及相关许可信息
# Copyright 2024 Black Forest Labs, The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版的规定，使用该文件的条款
# Licensed under the Apache License, Version 2.0 (the "License");
# 你不得使用该文件，除非遵守许可证
# You may not use this file except in compliance with the License.
# 许可证的副本可以在以下网址获得
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，按许可证分发的软件在 "按现状" 的基础上提供，
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 参见许可证了解有关权限和限制的具体条款
# See the License for the specific language governing permissions and
# limitations under the License.


# 导入类型注解以支持类型提示
from typing import Any, Dict, List, Optional, Union

# 导入 PyTorch 及其神经网络模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从配置和加载模块导入所需的类和混合
from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
# 从注意力模块导入所需的类
from ...models.attention import FeedForward
from ...models.attention_processor import Attention, FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0
# 从模型工具导入基础模型类
from ...models.modeling_utils import ModelMixin
# 从归一化模块导入自适应层归一化类
from ...models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
# 从工具模块导入各种实用功能
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
# 从嵌入模块导入所需的类
from ..embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings
# 从输出模块导入 Transformer 模型输出类
from ..modeling_outputs import Transformer2DModelOutput


# 获取日志记录器以供本模块使用
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# YiYi 待办事项: 重构与 rope 相关的函数/类
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    # 确保输入维度为偶数
    assert dim % 2 == 0, "The dimension must be even."

    # 计算缩放因子，范围从 0 到 dim，步长为 2
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    # 计算 omega 值
    omega = 1.0 / (theta**scale)

    # 获取批次大小和序列长度
    batch_size, seq_length = pos.shape
    # 使用爱因斯坦求和约定计算输出
    out = torch.einsum("...n,d->...nd", pos, omega)
    # 计算余弦和正弦值
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    # 堆叠余弦和正弦结果
    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    # 重塑输出形状
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    # 返回输出的浮点数形式
    return out.float()


# YiYi 待办事项: 重构与 rope 相关的函数/类
class EmbedND(nn.Module):
    # 初始化方法，接收维度、theta 和轴维度
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()  # 调用父类初始化
        self.dim = dim  # 存储维度
        self.theta = theta  # 存储 theta 参数
        self.axes_dim = axes_dim  # 存储轴维度

    # 前向传播方法，接受输入 ID
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]  # 获取轴的数量
        # 计算嵌入并沿着维度 -3 连接结果
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        # 返回添加维度的嵌入
        return emb.unsqueeze(1)


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206
    # 参数说明文档
    Parameters:
        dim (`int`): 输入和输出的通道数量
        num_attention_heads (`int`): 多头注意力机制中使用的头数量
        attention_head_dim (`int`): 每个头的通道数量
        context_pre_only (`bool`): 布尔值，确定是否添加与处理 `context` 条件相关的一些模块
    """

    # 初始化函数，设置模型参数
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        # 调用父类构造函数
        super().__init__()
        # 计算 MLP 隐藏层的维度
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # 创建自适应层归一化实例
        self.norm = AdaLayerNormZeroSingle(dim)
        # 创建线性变换层，将输入维度映射到 MLP 隐藏维度
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        # 使用 GELU 激活函数
        self.act_mlp = nn.GELU(approximate="tanh")
        # 创建线性变换层，将 MLP 输出和输入维度合并并映射回输入维度
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        # 创建注意力处理器实例
        processor = FluxSingleAttnProcessor2_0()
        # 初始化注意力机制，配置相关参数
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    # 前向传播函数，定义输入的处理方式
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        # 保存输入的残差用于后续相加
        residual = hidden_states
        # 进行层归一化，并得到归一化后的隐藏状态和门控值
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        # 对归一化后的隐藏状态进行线性变换和激活
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        # 计算注意力输出
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # 将注意力输出和 MLP 隐藏状态在最后一维拼接
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        # 扩展门控值维度以便后续运算
        gate = gate.unsqueeze(1)
        # 使用门控机制对输出进行加权，并将其与残差相加
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        # 如果数据类型为 float16，则对输出进行裁剪，避免溢出
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        # 返回最终的隐藏状态
        return hidden_states
# 装饰器，可能允许该类在计算图中使用
@maybe_allow_in_graph
# 定义 FluxTransformerBlock 类，继承自 nn.Module
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    # 初始化方法，接受多个参数以配置 Transformer 块
    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        # 调用父类构造函数
        super().__init__()

        # 初始化第一个自适应层归一化
        self.norm1 = AdaLayerNormZero(dim)

        # 初始化上下文的自适应层归一化
        self.norm1_context = AdaLayerNormZero(dim)

        # 检查 PyTorch 是否支持 scaled_dot_product_attention
        if hasattr(F, "scaled_dot_product_attention"):
            # 创建 Attention 处理器
            processor = FluxAttnProcessor2_0()
        else:
            # 如果不支持，抛出异常
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        # 初始化注意力层
        self.attn = Attention(
            query_dim=dim,  # 查询维度
            cross_attention_dim=None,  # 交叉注意力维度
            added_kv_proj_dim=dim,  # 额外键值投影维度
            dim_head=attention_head_dim,  # 每个头的维度
            heads=num_attention_heads,  # 注意力头的数量
            out_dim=dim,  # 输出维度
            context_pre_only=False,  # 上下文预处理标志
            bias=True,  # 是否使用偏置
            processor=processor,  # 注意力处理器
            qk_norm=qk_norm,  # 查询键的归一化方式
            eps=eps,  # 稳定性常数
        )

        # 初始化第二个层归一化
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # 初始化前馈网络
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # 初始化上下文的第二个层归一化
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # 初始化上下文的前馈网络
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # 让块大小默认为 None
        self._chunk_size = None
        # 设定块维度为 0
        self._chunk_dim = 0

    # 前向传播方法，定义输入及其处理
    def forward(
        self,
        hidden_states: torch.FloatTensor,  # 输入的隐藏状态
        encoder_hidden_states: torch.FloatTensor,  # 编码器的隐藏状态
        temb: torch.FloatTensor,  # 额外的嵌入信息
        image_rotary_emb=None,  # 可选的图像旋转嵌入
    ):
        # 对隐藏状态进行归一化处理，并计算门控相关的多头自注意力值
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # 对编码器的隐藏状态进行归一化处理，并计算门控相关的多头自注意力值
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 注意力机制计算
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # 处理注意力输出以更新 `hidden_states`
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        # 对更新后的隐藏状态进行第二次归一化处理
        norm_hidden_states = self.norm2(hidden_states)
        # 结合门控机制调整归一化后的隐藏状态
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        # 前馈网络处理
        ff_output = self.ff(norm_hidden_states)
        # 结合门控机制调整前馈网络输出
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        # 更新 `hidden_states`
        hidden_states = hidden_states + ff_output

        # 处理编码器隐藏状态的注意力输出
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # 对编码器隐藏状态进行第二次归一化处理
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        # 结合门控机制调整归一化后的编码器隐藏状态
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 对编码器的前馈网络处理
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        # 更新编码器隐藏状态
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        # 对半精度数据进行范围裁剪，避免溢出
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        # 返回更新后的编码器和隐藏状态
        return encoder_hidden_states, hidden_states
# 定义 FluxTransformer2DModel 类，继承自多个混合类以获取其功能
class FluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Flux 中引入的 Transformer 模型。

    参考文献: https://blackforestlabs.ai/announcing-black-forest-labs/

    参数:
        patch_size (`int`): 将输入数据转换为小块的块大小。
        in_channels (`int`, *可选*, 默认为 16): 输入的通道数量。
        num_layers (`int`, *可选*, 默认为 18): 使用的 MMDiT 块的层数。
        num_single_layers (`int`, *可选*, 默认为 18): 使用的单 DiT 块的层数。
        attention_head_dim (`int`, *可选*, 默认为 64): 每个头的通道数。
        num_attention_heads (`int`, *可选*, 默认为 18): 用于多头注意力的头数。
        joint_attention_dim (`int`, *可选*): 用于 `encoder_hidden_states` 维度的数量。
        pooled_projection_dim (`int`): 投影 `pooled_projections` 时使用的维度数量。
        guidance_embeds (`bool`, 默认为 False): 是否使用引导嵌入。
    """

    # 支持梯度检查点，减少内存使用
    _supports_gradient_checkpointing = True

    # 注册到配置中，初始化模型参数
    @register_to_config
    def __init__(
        # 定义块的大小，默认为 1
        self,
        patch_size: int = 1,
        # 定义输入通道的数量，默认为 64
        in_channels: int = 64,
        # 定义 MMDiT 块的层数，默认为 19
        num_layers: int = 19,
        # 定义单 DiT 块的层数，默认为 38
        num_single_layers: int = 38,
        # 定义每个注意力头的通道数，默认为 128
        attention_head_dim: int = 128,
        # 定义多头注意力的头数，默认为 24
        num_attention_heads: int = 24,
        # 定义用于 `encoder_hidden_states` 的维度，默认为 4096
        joint_attention_dim: int = 4096,
        # 定义投影的维度，默认为 768
        pooled_projection_dim: int = 768,
        # 定义是否使用引导嵌入，默认为 False
        guidance_embeds: bool = False,
        # 定义 ROPE 的轴维度，默认值为 [16, 56, 56]
        axes_dims_rope: List[int] = [16, 56, 56],
    ):
        # 调用父类构造函数初始化
        super().__init__()
        # 设置输出通道数为输入通道数
        self.out_channels = in_channels
        # 计算内层维度为注意力头数量乘以每个头的维度
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        # 创建位置嵌入对象，用于维度和轴的设置
        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=axes_dims_rope)
        # 根据是否使用引导嵌入选择合并时间步引导文本投影嵌入类
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        # 创建时间文本嵌入对象，使用前面选择的类
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        # 创建线性层用于上下文嵌入
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        # 创建线性层用于输入嵌入
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        # 创建多个变换器块的模块列表
        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        # 创建多个单一变换器块的模块列表
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        # 创建自适应层归一化层作为输出层
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        # 创建线性投影层，将内层维度映射到输出通道的形状
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    # 定义设置梯度检查点的函数
    def _set_gradient_checkpointing(self, module, value=False):
        # 检查模块是否具有梯度检查点属性
        if hasattr(module, "gradient_checkpointing"):
            # 设置模块的梯度检查点属性
            module.gradient_checkpointing = value

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
```