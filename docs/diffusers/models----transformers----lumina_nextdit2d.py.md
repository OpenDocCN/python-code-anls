# `.\diffusers\models\transformers\lumina_nextdit2d.py`

```py
# 版权声明，表明此代码的版权归 2024 Alpha-VLLM 作者及 HuggingFace 团队所有
# 
# 根据 Apache 2.0 许可证（"许可证"）进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有规定，软件在许可证下分发是按"原样"基础，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参阅许可证以获取有关权限和
# 限制的具体信息。

from typing import Any, Dict, Optional  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

# 从配置和工具模块导入必要的类和函数
from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import logging
from ..attention import LuminaFeedForward  # 导入自定义的前馈网络
from ..attention_processor import Attention, LuminaAttnProcessor2_0  # 导入注意力处理器
from ..embeddings import (
    LuminaCombinedTimestepCaptionEmbedding,  # 导入组合时间步长的嵌入
    LuminaPatchEmbed,  # 导入补丁嵌入
)
from ..modeling_outputs import Transformer2DModelOutput  # 导入模型输出类
from ..modeling_utils import ModelMixin  # 导入模型混合类
from ..normalization import LuminaLayerNormContinuous, LuminaRMSNormZero, RMSNorm  # 导入不同的归一化方法

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，禁用 pylint 的名称警告


class LuminaNextDiTBlock(nn.Module):  # 定义一个名为 LuminaNextDiTBlock 的类，继承自 nn.Module
    """
    LuminaNextDiTBlock 用于 LuminaNextDiT2DModel。

    参数：
        dim (`int`): 输入特征的嵌入维度。
        num_attention_heads (`int`): 注意力头的数量。
        num_kv_heads (`int`):
            键和值特征中的注意力头数量（如果使用 GQA），
            或设置为 None 以与查询相同。
        multiple_of (`int`): 前馈网络层的倍数。
        ffn_dim_multiplier (`float`): 前馈网络层维度的乘数因子。
        norm_eps (`float`): 归一化层的 epsilon 值。
        qk_norm (`bool`): 查询和键的归一化。
        cross_attention_dim (`int`): 输入文本提示的跨注意力嵌入维度。
        norm_elementwise_affine (`bool`, *可选*, 默认为 True)，
    """

    def __init__(  # 初始化方法
        self,
        dim: int,  # 输入特征的维度
        num_attention_heads: int,  # 注意力头的数量
        num_kv_heads: int,  # 键和值特征的头数量
        multiple_of: int,  # 前馈网络层的倍数
        ffn_dim_multiplier: float,  # 前馈网络维度的乘数
        norm_eps: float,  # 归一化的 epsilon 值
        qk_norm: bool,  # 是否对查询和键进行归一化
        cross_attention_dim: int,  # 跨注意力嵌入的维度
        norm_elementwise_affine: bool = True,  # 是否使用逐元素仿射归一化，默认值为 True
    ) -> None:  # 定义方法的返回类型为 None，表示不返回任何值
        super().__init__()  # 调用父类的构造函数，初始化父类的属性
        self.head_dim = dim // num_attention_heads  # 计算每个注意力头的维度

        self.gate = nn.Parameter(torch.zeros([num_attention_heads]))  # 创建一个可学习的参数，初始化为零，大小为注意力头的数量

        # Self-attention  # 定义自注意力机制
        self.attn1 = Attention(  # 创建第一个注意力层
            query_dim=dim,  # 查询的维度
            cross_attention_dim=None,  # 交叉注意力的维度，此处为 None 表示不使用
            dim_head=dim // num_attention_heads,  # 每个头的维度
            qk_norm="layer_norm_across_heads" if qk_norm else None,  # 如果 qk_norm 为真，使用跨头层归一化
            heads=num_attention_heads,  # 注意力头的数量
            kv_heads=num_kv_heads,  # 键值对的头数量
            eps=1e-5,  # 数值稳定性参数
            bias=False,  # 不使用偏置项
            out_bias=False,  # 输出层不使用偏置项
            processor=LuminaAttnProcessor2_0(),  # 使用指定的注意力处理器
        )
        self.attn1.to_out = nn.Identity()  # 输出层使用恒等映射

        # Cross-attention  # 定义交叉注意力机制
        self.attn2 = Attention(  # 创建第二个注意力层
            query_dim=dim,  # 查询的维度
            cross_attention_dim=cross_attention_dim,  # 交叉注意力的维度
            dim_head=dim // num_attention_heads,  # 每个头的维度
            qk_norm="layer_norm_across_heads" if qk_norm else None,  # 如果 qk_norm 为真，使用跨头层归一化
            heads=num_attention_heads,  # 注意力头的数量
            kv_heads=num_kv_heads,  # 键值对的头数量
            eps=1e-5,  # 数值稳定性参数
            bias=False,  # 不使用偏置项
            out_bias=False,  # 输出层不使用偏置项
            processor=LuminaAttnProcessor2_0(),  # 使用指定的注意力处理器
        )

        self.feed_forward = LuminaFeedForward(  # 创建前馈神经网络层
            dim=dim,  # 输入维度
            inner_dim=4 * dim,  # 内部维度，通常为输入维度的四倍
            multiple_of=multiple_of,  # 确保内部维度是某个数字的倍数
            ffn_dim_multiplier=ffn_dim_multiplier,  # 前馈网络维度的乘数
        )

        self.norm1 = LuminaRMSNormZero(  # 创建第一个 RMS 归一化层
            embedding_dim=dim,  # 归一化的嵌入维度
            norm_eps=norm_eps,  # 归一化的 epsilon 参数
            norm_elementwise_affine=norm_elementwise_affine,  # 是否使用元素级仿射变换
        )
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)  # 创建前馈网络的 RMS 归一化层

        self.norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)  # 创建第二个 RMS 归一化层
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)  # 创建前馈网络的第二个 RMS 归一化层

        self.norm1_context = RMSNorm(cross_attention_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)  # 创建上下文的 RMS 归一化层

    def forward(  # 定义前向传播方法
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: torch.Tensor,  # 注意力掩码张量
        image_rotary_emb: torch.Tensor,  # 图像旋转嵌入张量
        encoder_hidden_states: torch.Tensor,  # 编码器的隐藏状态张量
        encoder_mask: torch.Tensor,  # 编码器的掩码张量
        temb: torch.Tensor,  # 位置编码或时间编码张量
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # 可选的交叉注意力参数字典
    ):
        """
        执行 LuminaNextDiTBlock 的前向传递。

        参数:
            hidden_states (`torch.Tensor`): LuminaNextDiTBlock 的输入隐藏状态。
            attention_mask (`torch.Tensor): 对应隐藏状态的注意力掩码。
            image_rotary_emb (`torch.Tensor`): 预计算的余弦和正弦频率。
            encoder_hidden_states: (`torch.Tensor`): 通过 Gemma 编码器处理的文本提示的隐藏状态。
            encoder_mask (`torch.Tensor`): 文本提示的隐藏状态注意力掩码。
            temb (`torch.Tensor`): 带有文本提示嵌入的时间步嵌入。
            cross_attention_kwargs (`Dict[str, Any]`): 交叉注意力的参数。
        """
        # 保存输入的隐藏状态，以便后续使用
        residual = hidden_states

        # 自注意力
        # 对隐藏状态进行归一化，并计算门控机制的输出
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        # 计算自注意力的输出
        self_attn_output = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=image_rotary_emb,
            **cross_attention_kwargs,
        )

        # 交叉注意力
        # 对编码器的隐藏状态进行归一化
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        # 计算交叉注意力的输出
        cross_attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=encoder_mask,
            query_rotary_emb=image_rotary_emb,
            key_rotary_emb=None,
            **cross_attention_kwargs,
        )
        # 将交叉注意力的输出进行缩放
        cross_attn_output = cross_attn_output * self.gate.tanh().view(1, 1, -1, 1)
        # 将自注意力和交叉注意力的输出混合
        mixed_attn_output = self_attn_output + cross_attn_output
        # 将混合输出展平，以便后续处理
        mixed_attn_output = mixed_attn_output.flatten(-2)
        # 线性投影
        # 通过线性层处理混合输出，得到新的隐藏状态
        hidden_states = self.attn2.to_out[0](mixed_attn_output)

        # 更新隐藏状态，加入残差连接和门控机制
        hidden_states = residual + gate_msa.unsqueeze(1).tanh() * self.norm2(hidden_states)

        # 通过前馈网络计算输出
        mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))

        # 更新隐藏状态，加入前馈网络输出和门控机制
        hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)

        # 返回最终的隐藏状态
        return hidden_states
# 定义一个名为 LuminaNextDiT2DModel 的类，继承自 ModelMixin 和 ConfigMixin
class LuminaNextDiT2DModel(ModelMixin, ConfigMixin):
    """
    LuminaNextDiT: 使用 Transformer 主干的扩散模型。

    继承 ModelMixin 和 ConfigMixin 以兼容 diffusers 的 StableDiffusionPipeline 采样器。

    参数：
        sample_size (`int`): 潜在图像的宽度。此值在训练期间固定，因为
            它用于学习位置嵌入的数量。
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            图像中每个补丁的大小。此参数定义输入到模型中的补丁的分辨率。
        in_channels (`int`, *optional*, defaults to 4):
            模型的输入通道数量。通常，这与输入图像的通道数量匹配。
        hidden_size (`int`, *optional*, defaults to 4096):
            模型隐藏层的维度。此参数决定了模型隐藏表示的宽度。
        num_layers (`int`, *optional*, default to 32):
            模型中的层数。此值定义了神经网络的深度。
        num_attention_heads (`int`, *optional*, defaults to 32):
            每个注意力层中的注意力头数量。此参数指定使用多少个独立的注意力机制。
        num_kv_heads (`int`, *optional*, defaults to 8):
            注意力机制中的键值头数量，如果与注意力头数量不同。如果为 None，则默认值为 num_attention_heads。
        multiple_of (`int`, *optional*, defaults to 256):
            隐藏大小应该是一个倍数的因子。这可以帮助优化某些硬件
            配置。
        ffn_dim_multiplier (`float`, *optional*):
            前馈网络维度的乘数。如果为 None，则使用基于
            模型配置的默认值。
        norm_eps (`float`, *optional*, defaults to 1e-5):
            添加到归一化层的分母中的一个小值，用于数值稳定性。
        learn_sigma (`bool`, *optional*, defaults to True):
            模型是否应该学习 sigma 参数，该参数可能与预测中的不确定性或方差相关。
        qk_norm (`bool`, *optional*, defaults to True):
            指示注意力机制中的查询和键是否应该被归一化。
        cross_attention_dim (`int`, *optional*, defaults to 2048):
            文本嵌入的维度。此参数定义了用于模型的文本表示的大小。
        scaling_factor (`float`, *optional*, defaults to 1.0):
            应用于模型某些参数或层的缩放因子。此参数可用于调整模型操作的整体规模。
    """
    # 注册到配置
    @register_to_config
    def __init__(
        # 样本大小，默认值为128
        self,
        sample_size: int = 128,
        # 补丁大小，默认为2，表示图像切割块的大小
        patch_size: Optional[int] = 2,
        # 输入通道数，默认为4，表示输入数据的特征通道
        in_channels: Optional[int] = 4,
        # 隐藏层大小，默认为2304
        hidden_size: Optional[int] = 2304,
        # 网络层数，默认为32
        num_layers: Optional[int] = 32,
        # 注意力头数量，默认为32
        num_attention_heads: Optional[int] = 32,
        # KV头的数量，默认为None
        num_kv_heads: Optional[int] = None,
        # 数量的倍数，默认为256
        multiple_of: Optional[int] = 256,
        # FFN维度乘数，默认为None
        ffn_dim_multiplier: Optional[float] = None,
        # 归一化的epsilon值，默认为1e-5
        norm_eps: Optional[float] = 1e-5,
        # 是否学习方差，默认为True
        learn_sigma: Optional[bool] = True,
        # 是否进行QK归一化，默认为True
        qk_norm: Optional[bool] = True,
        # 交叉注意力维度，默认为2048
        cross_attention_dim: Optional[int] = 2048,
        # 缩放因子，默认为1.0
        scaling_factor: Optional[float] = 1.0,
    ) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 设置样本大小属性
        self.sample_size = sample_size
        # 设置补丁大小属性
        self.patch_size = patch_size
        # 设置输入通道数属性
        self.in_channels = in_channels
        # 根据是否学习方差设置输出通道数
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        # 设置隐藏层大小属性
        self.hidden_size = hidden_size
        # 设置注意力头数量属性
        self.num_attention_heads = num_attention_heads
        # 计算并设置每个注意力头的维度
        self.head_dim = hidden_size // num_attention_heads
        # 设置缩放因子属性
        self.scaling_factor = scaling_factor

        # 创建补丁嵌入层，并初始化其参数
        self.patch_embedder = LuminaPatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=hidden_size, bias=True
        )

        # 创建一个可学习的填充标记，初始化为空张量
        self.pad_token = nn.Parameter(torch.empty(hidden_size))

        # 创建时间和标题的组合嵌入层
        self.time_caption_embed = LuminaCombinedTimestepCaptionEmbedding(
            hidden_size=min(hidden_size, 1024), cross_attention_dim=cross_attention_dim
        )

        # 创建包含多个层的模块列表
        self.layers = nn.ModuleList(
            [
                # 在模块列表中添加多个下一代块
                LuminaNextDiTBlock(
                    hidden_size,
                    num_attention_heads,
                    num_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    cross_attention_dim,
                )
                for _ in range(num_layers)  # 根据层数循环
            ]
        )
        # 创建层归一化输出层
        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )
        # 注释掉的最终层的初始化（若需要可取消注释）
        # self.final_layer = LuminaFinalLayer(hidden_size, patch_size, self.out_channels)

        # 确保隐藏层大小与注意力头数量的关系，保证为4的倍数
        assert (hidden_size // num_attention_heads) % 4 == 0, "2d rope needs head dim to be divisible by 4"

    # 前向传播函数定义
    def forward(
        # 隐藏状态的输入张量
        self,
        hidden_states: torch.Tensor,
        # 时间步的输入张量
        timestep: torch.Tensor,
        # 编码器的隐藏状态张量
        encoder_hidden_states: torch.Tensor,
        # 编码器的掩码张量
        encoder_mask: torch.Tensor,
        # 图像的旋转嵌入张量
        image_rotary_emb: torch.Tensor,
        # 交叉注意力的其他参数，默认为None
        cross_attention_kwargs: Dict[str, Any] = None,
        # 是否返回字典形式的结果，默认为True
        return_dict=True,
    # LuminaNextDiT 的前向传播函数
    ) -> torch.Tensor:
            """
            前向传播的 LuminaNextDiT 模型。
    
            参数:
                hidden_states (torch.Tensor): 输入张量，形状为 (N, C, H, W)。
                timestep (torch.Tensor): 扩散时间步的张量，形状为 (N,).
                encoder_hidden_states (torch.Tensor): 描述特征的张量，形状为 (N, D)。
                encoder_mask (torch.Tensor): 描述特征掩码的张量，形状为 (N, L)。
            """
            # 通过补丁嵌入器处理隐藏状态，获取掩码、图像大小和图像旋转嵌入
            hidden_states, mask, img_size, image_rotary_emb = self.patch_embedder(hidden_states, image_rotary_emb)
            # 将图像旋转嵌入转移到与隐藏状态相同的设备上
            image_rotary_emb = image_rotary_emb.to(hidden_states.device)
    
            # 生成时间嵌入，结合时间步和编码器隐藏状态
            temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)
    
            # 将编码器掩码转换为布尔值
            encoder_mask = encoder_mask.bool()
            # 对每一层进行遍历，更新隐藏状态
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    mask,
                    image_rotary_emb,
                    encoder_hidden_states,
                    encoder_mask,
                    temb=temb,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
    
            # 对隐藏状态进行归一化处理
            hidden_states = self.norm_out(hidden_states, temb)
    
            # 反补丁操作
            height_tokens = width_tokens = self.patch_size  # 获取补丁大小
            height, width = img_size[0]  # 从图像大小中提取高度和宽度
            batch_size = hidden_states.size(0)  # 获取批次大小
            sequence_length = (height // height_tokens) * (width // width_tokens)  # 计算序列长度
            # 调整隐藏状态的形状，以适应输出要求
            hidden_states = hidden_states[:, :sequence_length].view(
                batch_size, height // height_tokens, width // width_tokens, height_tokens, width_tokens, self.out_channels
            )
            # 调整维度以获得最终输出
            output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)
    
            # 如果不需要返回字典，则返回输出元组
            if not return_dict:
                return (output,)
    
            # 返回 Transformer2DModelOutput 的结果
            return Transformer2DModelOutput(sample=output)
```