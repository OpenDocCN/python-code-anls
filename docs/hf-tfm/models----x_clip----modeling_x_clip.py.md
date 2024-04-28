# `.\transformers\models\x_clip\modeling_x_clip.py`

```
# 设置文件编码为utf-8
# 版权声明
# 2022年版权所属于Microsoft Research和The HuggingFace Team，保留所有权利。
# 根据Apache许可证2.0版（“许可证”）获得许可。您只能在遵守许可证的情况下使用此文件。
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则在“原样”基础上分发软件，不提供任何形式的明示或暗示保证或条件。请参阅许可证，了解具体语言下的权限和限制。

""" PyTorch X-CLIP model.""" 
# 导入所需的库和模块
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 定义全局日志记录器
logger = logging.get_logger(__name__)

# 预训练模型的名单
_CHECKPOINT_FOR_DOC = "microsoft/xclip-base-patch32"

XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/xclip-base-patch32",
    # 查看所有 X-CLIP 模型 https://huggingface.co/models?filter=x-clip
]


# 对比损失函数，改编自 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# 从transformers.models.clip.modeling_clip.clip_loss中复制，将clip->x_clip
def x_clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

@dataclass
class XCLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for video-text similarity. (用于视频文本相似性的对比损失，如果`return_loss`为`True`时返回)
        logits_per_video (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`):
            The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text similarity scores. (视频嵌入和文本嵌入之间的缩放点积分数。这代表了视频文本相似性得分)
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`):
            The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video similarity scores. (文本嵌入和视频嵌入之间的缩放点积分数。这代表了文本视频相似性得分)
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`XCLIPTextModel`]. (通过将投影层应用于[`XCLIPTextModel`]的池化输出获得的文本嵌入)
        video_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The video embeddings obtained by applying the projection layer to the pooled output of
            [`XCLIPVisionModel`]. (通过将投影层应用于[`XCLIPVisionModel`]的池化输出获得的视频嵌入)
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPTextModel`]. ( [`XCLIPTextModel`]的输出)
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`XCLIPVisionModel`]. ( [`XCLIPVisionModel`]的输出)
        mit_output (`BaseModelOutputWithPooling`):
            The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short). ( `XCLIPMultiframeIntegrationTransformer` (MIT的缩写)的输出)
    """

    loss: Optional[torch.FloatTensor] = None  # 初始化损失为None
    logits_per_video: torch.FloatTensor = None  # 初始化视频对应文本的对比分数为None
    logits_per_text: torch.FloatTensor = None  # 初始化文本对应视频的对比分数为None
    text_embeds: torch.FloatTensor = None  # 初始化文本嵌入为None
    video_embeds: torch.FloatTensor = None  # 初始化视频嵌入为None
    text_model_output: BaseModelOutputWithPooling = None  # 初始化文本模型输出为None
    vision_model_output: BaseModelOutputWithPooling = None  # 初始化视觉模型输出为None
    mit_output: BaseModelOutputWithPooling = None  # 初始化MIT模型输出为None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["text_model_output", "vision_model_output", "mit_output"]  # 如果 键 不在["text_model_output", "vision_model_output", "mit_output"]中
            else getattr(self, k).to_tuple()  # 则返回相应值的元组形式
            for k in self.keys()  # 对于self的键进行遍历
        )
# 从transformers.models.clip.modeling_clip.CLIPVisionEmbeddings复制而来，将CLIP更改为XCLIP
class XCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: XCLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 嵌入维度
        self.image_size = config.image_size  # 图像尺寸
        self.patch_size = config.patch_size  # 补丁尺寸

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))  # 类别嵌入向量

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,  # 输入通道数
            out_channels=self.embed_dim,  # 输出通道数，与嵌入维度相同
            kernel_size=self.patch_size,  # 卷积核大小，即补丁大小
            stride=self.patch_size,  # 步长，与补丁大小相同
            bias=False,  # 不使用偏置
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 补丁数量
        self.num_positions = self.num_patches + 1  # 位置编码的数量，补丁数量加1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # 位置嵌入
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)  # 位置编码张量，用于记录位置信息

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # 批次大小
        target_dtype = self.patch_embedding.weight.dtype  # 目标数据类型
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # 补丁嵌入，使用卷积提取图像特征
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 将补丁嵌入展平并转置

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # 类别嵌入向量扩展为与批次大小相同的维度
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 将类别嵌入向量和补丁嵌入向量拼接在一起
        embeddings = embeddings + self.position_embedding(self.position_ids)  # 加上位置嵌入
        return embeddings


# 从transformers.models.clip.modeling_clip.CLIPTextEmbeddings复制而来，将CLIP更改为XCLIP
class XCLIPTextEmbeddings(nn.Module):
    def __init__(self, config: XCLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size  # 嵌入维度

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)  # 词嵌入
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)  # 位置嵌入

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )  # 位置编码张量，用于记录位置信息

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]  # 序列长度

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]  # 如果位置编码未提供，则使用默认的位置编码

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)  # 如果输入的嵌入向量未提供，则使用词嵌入

        position_embeddings = self.position_embedding(position_ids)  # 获取位置嵌入
        embeddings = inputs_embeds + position_embeddings  # 将词嵌入和位置嵌入相加得到最终的嵌入表示

        return embeddings


# 从transformers.models.clip.modeling_clip.CLIPAttention复制而来，将CLIP更改为XCLIP
class XCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 如果 embed_dim 不能被 num_heads 整除，则引发错误
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scale = self.head_dim**-0.5
        # 用于 dropout 的概率
        self.dropout = config.attention_dropout

        # 线性层，用于投影 key
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # 线性层，用于投影 value
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # 线性层，用于投影 query
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        # 输出投影层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量形状调整为 [bsz, num_heads, seq_len, head_dim]，并交换维度
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
):
    """Perform multi-headed attention computation."""
    # Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->XCLIP
class XCLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 第一个全连接层
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 第二个全连接层
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播，先通过第一个全连接层，再通过激活函数，最后通过第二个全连接层
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->XCLIP
class XCLIPEncoderLayer(nn.Module):
    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # 自注意力层
        self.self_attn = XCLIPAttention(config)
        # 第一个 LayerNorm 层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 多层感知机
        self.mlp = XCLIPMLP(config)
        # 第二个 LayerNorm 层
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
):
    """Perform one forward pass of the encoder layer."""
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存输入的hidden_states作为残差连接的一部分
        residual = hidden_states

        # 执行 layer normalization 操作
        hidden_states = self.layer_norm1(hidden_states)
        
        # 执行自注意力机制操作，返回处理后的 hidden_states 和注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        
        # 将输入的 hidden_states 与经过自注意力机制处理后的 hidden_states 相加，实现残差连接
        hidden_states = residual + hidden_states

        # 保存输入的hidden_states作为残差连接的一部分
        residual = hidden_states
        
        # 执行 layer normalization 操作
        hidden_states = self.layer_norm2(hidden_states)
        
        # 执行MLP操作
        hidden_states = self.mlp(hidden_states)
        
        # 将输入的 hidden_states 与经过MLP处理后的 hidden_states 相加，实现残差连接
        hidden_states = residual + hidden_states

        # 返回处理后的 hidden_states
        outputs = (hidden_states,)

        # 如果设置了返回 attentions tensors，则将 attentions tensors 放入返回结果中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义一个函数，用于实现 Stochastic Depth (随机深度) 功能的实现
# 参数包括输入张量、丢弃概率和是否训练中
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果丢弃概率等于0或者不在训练中，直接返回输入张量
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 根据输入张量的形状创建随机张量
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    # 计算最终输出张量
    output = input.div(keep_prob) * random_tensor
    return output


# 定义一个类，用于实现 XCLIP 论文中的 Drop Path 功能
class XCLIPDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用前面定义的 drop_path 函数
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# 定义一个类，用于实现 XCLIP 论文中的 Vision Encoder Layer
class XCLIPVisionEncoderLayer(nn.Module):
    """
    This corresponds to the `CrossFramelAttentionBlock` class in the original implementation.
    """

    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.num_frames = config.num_frames
        self.embed_dim = config.hidden_size

        # 初始化网络的不同部分，包括消息传递层、LayerNorm 层、注意力机制、Drop Path 层、MLP 层等
        self.message_fc = nn.Linear(self.embed_dim, self.embed_dim)
        self.message_ln = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.message_attn = XCLIPAttention(config)

        self.drop_path = XCLIPDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.self_attn = XCLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = XCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
````
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入层的输入，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): 大小为 `(batch, 1, tgt_len, src_len)` 的注意力遮盖，
                其中填充元素由非常大的负值表示。
                `(config.encoder_attention_heads,)`.
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于文本模型的因果遮盖。遮盖值选在 `[0, 1]` 之间:
                - 1表示**未遮盖**的标记,
                - 0表示**遮盖**的标记。
                [什么是注意力遮盖?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请查看返回的张量中的 `attentions`。
        """
        batch_time, seq_length, hidden_size = hidden_states.size()
        batch_size = batch_time // self.num_frames
        msg_token = self.message_fc(hidden_states[:, 0, :])
        msg_token = msg_token.view(batch_size, self.num_frames, hidden_size)

        msg_token = msg_token + self.drop_path(self.message_attn(self.message_ln(msg_token))[0])
        # 添加虚拟的序列维度
        msg_token = msg_token.view(-1, 1, hidden_size)

        hidden_states = torch.cat([hidden_states, msg_token], dim=1)

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        hidden_states = hidden_states[:, :seq_length, :]

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 创建一个名为 XCLIPPreTrainedModel 的类，它是 PreTrainedModel 的子类
class XCLIPPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化和一个简单的接口，用于下载和加载预训练模型。
    """

    # 定义 config_class 属性为 XCLIPConfig
    config_class = XCLIPConfig
    # 定义 base_model_prefix 属性为 "x_clip"
    base_model_prefix = "x_clip"
    # 定义 supports_gradient_checkpointing 属性为 True
    supports_gradient_checkpointing = True
    # 初始化权重函数，用于初始化模型中的权重参数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_factor
        # 如果模块是文本嵌入模块
        if isinstance(module, XCLIPTextEmbeddings):
            # 初始化 token 嵌入权重参数
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            # 初始化位置嵌入权重参数
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        # 如果模块是视觉嵌入模块
        elif isinstance(module, XCLIPVisionEmbeddings):
            # 重新获取初始化因子
            factor = self.config.initializer_factor
            # 初始化类别嵌入权重参数
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            # 初始化补丁嵌入权重参数
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            # 初始化位置嵌入权重参数
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # 如果模块是自注意力模块
        elif isinstance(module, XCLIPAttention):
            # 重新获取初始化因子
            factor = self.config.initializer_factor
            # 计算输入投影的标准差
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算输出投影的标准差
            out_proj_std = (module.embed_dim**-0.5) * factor
            # 初始化查询投影权重参数
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            # 初始化键投影权重参数
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            # 初始化值投影权重参数
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            # 初始化输出投影权重参数
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果模块是多层感知机模块
        elif isinstance(module, XCLIPMLP):
            # 重新获取初始化因子
            factor = self.config.initializer_factor
            # 计算输入投影的标准差
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算全连接层的标准差
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 初始化第一个全连接层权重参数
            nn.init.normal_(module.fc1.weight, std=fc_std)
            # 初始化第二个全连接层权重参数
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # 如果模块是 XCLIP 模型
        elif isinstance(module, XCLIPModel):
            # 重新获取初始化因子
            factor = self.config.initializer_factor
            # 初始化文本投影权重参数
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * factor,
            )
            # 初始化视觉投影权重参数
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * factor,
            )
            # 初始化提示视觉投影权重参数
            nn.init.normal_(module.prompts_visual_projection, mean=0.0, std=module.vision_embed_dim**-0.5 * factor)
        # 如果模块是多帧集成变换器模块
        elif isinstance(module, XCLIPMultiframeIntegrationTransformer):
            # 初始化位置嵌入参数
            nn.init.normal_(module.position_embedding, std=self.config.initializer_factor)

        # 如果模块是 LayerNorm 层
        if isinstance(module, nn.LayerNorm):
            # 初始化偏置参数为零
            module.bias.data.zero_()
            # 初始化缩放参数为单位矩阵
            module.weight.data.fill_(1.0)
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 初始化权重参数
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            # 如果存在偏置参数，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
# 定义 X_CLIP_START_DOCSTRING，包含模型的说明文档
X_CLIP_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`XCLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 X_CLIP_TEXT_INPUTS_DOCSTRING，包含文本输入的说明文档
X_CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义 X_CLIP_VISION_INPUTS_DOCSTRING，包含视觉输入的说明文档
X_CLIP_VISION_INPUTS_DOCSTRING = r"""
    # 参数列表:
    # pixel_values: torch.FloatTensor类型，形状为(batch_size, num_channels, height, width)，像素值。默认情况下将忽略填充。可以使用AutoImageProcessor获得像素值。有关详细信息，请参见CLIPImageProcessor.__call__。
    # output_attentions: bool类型，可选，默认为False。是否返回所有注意力层的注意力张量。有关详细信息，请参见返回的张量中的attentions。
    # output_hidden_states: bool类型，可选，默认为False。是否返回所有层的隐藏状态。有关详细信息，请参见返回的张量中的hidden_states。
    # return_dict: bool类型，可选，默认为False。是否返回~utils.ModelOutput而不是普通元组。
"""
X_CLIP_INPUTS_DOCSTRING是用来描述X-CLIP模型输入参数的文档字符串。

Args:
    input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        输入序列标记在词汇表中的索引。默认情况下，将忽略填充。
        
        可以使用`AutoTokenizer`获取索引。有关详细信息，请参见`PreTrainedTokenizer.encode`和`PreTrainedTokenizer.__call__`。
        
        [什么是输入 ID？](../glossary#input-ids)
    attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        避免在填充标记索引上执行注意力的掩码。在`[0, 1]`范围内选择掩码值：

        - 对于**未掩码**的标记为1，
        - 对于**掩码**的标记为0。
        
        [什么是注意力掩码？](../glossary#attention-mask)
    position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        输入序列标记在位置嵌入中的位置索引。在范围`[0, config.max_position_embeddings - 1]`中选择。

        [什么是位置 ID？](../glossary#position-ids)
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        像素值。默认情况下，将忽略填充。可以使用`AutoImageProcessor`获取像素值。有关详细信息，请参见`CLIPImageProcessor.__call__`。
    return_loss (`bool`, *optional*):
        是否返回对比损失。
    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量中的`attentions`。
    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量中的`hidden_states`。
    return_dict (`bool`, *optional*):
        是否返回一个[`~utils.ModelOutput`]而不是一个普通的元组。
"""


class XCLIPEncoder(nn.Module):
    """
    X-CLIP模型的Transformer编码器，由`config.num_hidden_layers`个自注意力层组成。每一层都是一个[`XCLIPEncoderLayer`]。

    Args:
        config: XCLIPConfig
    """

    def __init__(self, config: XCLIPConfig):
        super().__init__()
        self.config = config
        # 创建`config.num_hidden_layers`个`XCLIPEncoderLayer`层，并组成`nn.ModuleList`
        self.layers = nn.ModuleList([XCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点默认关闭
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义 XCLIPTextTransformer 类，继承自 nn.Module
    class XCLIPTextTransformer(nn.Module):
        # 初始化方法，接受一个配置参数 config: XCLIPTextConfig
        def __init__(self, config: XCLIPTextConfig):
            # 调用父类的初始化方法
            super().__init__()
            # 将配置参数保存在实例属性中
            self.config = config
            # 获取隐藏层维度
            embed_dim = config.hidden_size
            # 初始化文本嵌入层
            self.embeddings = XCLIPTextEmbeddings(config)
            # 初始化编码器层
            self.encoder = XCLIPEncoder(config)
            # 初始化最终的层归一化层
            self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 定义前向传播方法，参数和返回类型都有类型注解和文档字符串
        @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPTextConfig)
        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    # 定义返回类型为 Union[Tuple, BaseModelOutputWithPooling]
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 文档字符串（返回说明）
        r"""
        Returns:

        """
        # 确定输出注意力和隐藏状态的参数是否已设置，否则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未指定 input_ids，则抛出错误
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        # 获取 input_ids 的形状
        input_shape = input_ids.size()
        # 将 input_ids 调整视图为形状 (-1, input_shape[-1])
        input_ids = input_ids.view(-1, input_shape[-1])

        # 获取输入嵌入状态
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # X_CLIP 的文本模型使用因果掩码，这里进行准备
        # 链接到相应的 GitHub 代码行
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        # 如果提供了注意力掩码，则进行扩展
        if attention_mask is not None:
            # 将注意力掩码扩展为形状 [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # 将嵌入的输入数据传递给编码器
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 对最后隐藏状态进行归一化
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # 计算文本嵌入，取自每个序列的 eot 嵌入（eot_token 是每个序列中最高的编号）
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0]), input_ids.argmax(dim=-1)]

        # 如果不返回字典，则返回一个元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class XCLIPTextModel(XCLIPPreTrainedModel):
    config_class = XCLIPTextConfig

    def __init__(self, config: XCLIPTextConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 初始化文本模型
        self.text_model = XCLIPTextTransformer(config)
        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, XCLIPTextModel

        >>> model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        # 调用文本模型的前向传播
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class XCLIPVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`XCLIPVisionEncoderLayer`].

    Args:
        config: XCLIPConfig
    """

    def __init__(self, config: XCLIPConfig):
        # 调用父类构造函数初始化模型
        super().__init__()
        # 保存配置
        self.config = config
        # 创建多个视觉编码器层
        self.layers = nn.ModuleList([XCLIPVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志，默认为关闭
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):  # 缺少了右括号
    # 初始化方法，接收一个 XCLIPVisionConfig 类型的参数
    def __init__(self, config: XCLIPVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置参数保存到实例变量中
        self.config = config
        # 从配置参数中获取嵌入维度
        embed_dim = config.hidden_size

        # 创建 XCLIPVisionEmbeddings 对象
        self.embeddings = XCLIPVisionEmbeddings(config)
        # 创建预层归一化（LayerNorm）对象
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 创建 XCLIPVisionEncoder 对象
        self.encoder = XCLIPVisionEncoder(config)
        # 创建后层归一化（LayerNorm）对象
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 标记模型前向传播方法的输入信息和返回结果信息
    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPVisionConfig)
    # 模型前向传播方法
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Returns:
        """
        # 如果输出注意力未指定，则使用配置的输出注意力设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用配置的输出隐藏状态设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典未指定，则使用配置的使用返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将像素值传入嵌入层
        hidden_states = self.embeddings(pixel_values)
        # 对输出进行预层归一化
        hidden_states = self.pre_layernorm(hidden_states)

        # 将预处理后的隐藏状态传入编码器
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 获取汇聚输出
        pooled_output = last_hidden_state[:, 0, :]
        # 对汇聚输出进行后层归一化
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不返回字典，则返回元组类型的结果
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 否则返回 BaseModelOutputWithPooling 类型的结果
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class XCLIPVisionModel(XCLIPPreTrainedModel):
    # 设置配置类为 XCLIPVisionConfig
    config_class = XCLIPVisionConfig
    # 定义主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    def __init__(self, config: XCLIPVisionConfig):
        # 调用父类构造函数，传入配置
        super().__init__(config)
        # 创建 XCLIPVisionTransformer 对象并赋值给 vision_model
        self.vision_model = XCLIPVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回 vision_model 对象的嵌入层
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(X_CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=XCLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class XCLIPMultiframeIntegrationTransformer(nn.Module):
    """
    This corresponds to the `MultiframeIntegrationTransformer` class in the original implementation.
    """

    def __init__(self, config: XCLIPVisionConfig):
        # 调用父类构造函数
        super().__init__()
        # 创建可学习的 position embedding
        self.position_embedding = nn.Parameter(torch.empty(1, config.num_frames, config.hidden_size))
        # 创建 XCLIPEncoder 对象并赋值给 encoder
        self.encoder = XCLIPEncoder(config)

    def forward(
        self,
        hidden_states,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        residual = hidden_states
        # 添加位置编码到 hidden_states
        hidden_states = hidden_states + self.position_embedding
        # 调用 encoder 的 forward 方法
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]

        last_hidden_state = last_hidden_state.type(hidden_states.dtype) + residual
        # 计算平均值作为 pooled_output
        pooled_output = last_hidden_state.mean(dim=1, keepdim=False)

        if not return_dict:
            # 如果 return_dict 为 False，则返回元组
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            # 返回包含池化输出和各种隐藏状态、注意力的 BaseModelOutputWithPooling 对象
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class XCLIPCrossAttention(nn.Module):
    # 定义一个多头注意力类，对应 'Attention Is All You Need' 论文
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 读取配置中的注意力头数
        self.num_heads = config.prompt_num_attention_heads

        # 从配置中读取维度信息
        dim = config.projection_dim
        # 根据头数计算每个头的维度
        head_dim = dim // self.num_heads
        # 计算缩放因子
        self.scale = head_dim**-0.5

        # 三个线性投影层，分别用于计算查询、键、值
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # 注意力机制中的 dropout
        self.attn_drop = nn.Dropout(config.prompt_attention_dropout)
        # 线性投影层的 dropout
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(config.prompt_projection_dropout)

    # 将输入张量重塑成适合注意力机制的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数
    def forward(self, queries, keys, values):
        """Input shape: Batch x Time x Channel"""
        # 获取输入的形状信息
        batch_size, query_seq_len, hidden_size = queries.shape
        batch_size, key_seq_len, hidden_size = keys.shape
        
        # 计算查询、键、值的三个线性变换
        queries = (
            self.q_proj(queries)
            .reshape(batch_size, query_seq_len, self.num_heads, hidden_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(keys)
            .reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        values = (
            self.v_proj(values)
            .reshape(batch_size, key_seq_len, self.num_heads, hidden_size // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # 计算注意力分数
        attn = (queries @ keys.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 进行加权求和以得到输出
        x = (attn @ values).transpose(1, 2).reshape(batch_size, query_seq_len, hidden_size)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class PromptGeneratorLayer(nn.Module):
    # 定义 PromptGeneratorLayer 类，用于生成提示的模型层
    def __init__(self, config):
        super().__init__()

        embed_dim = config.projection_dim
        # 使用 config.projection_dim 初始化 embed_dim
        self.cross_attn = XCLIPCrossAttention(config)
        # 初始化跨媒体注意力层
        self.norm1 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        # 对文本输入进行 LayerNormalization
        self.norm3 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        # 对文本输入进行 LayerNormalization
        self.mlp = nn.Sequential(
            # 使用线性层进行特征变换
            nn.Linear(embed_dim, embed_dim * 4),
            # 使用激活函数进行非线性变换
            ACT2FN[config.prompt_hidden_act],
            # 添加丢弃层，防止过拟合
            nn.Dropout(config.prompt_attention_dropout),
            # 使用线性层进行特征变换
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, visual):
        # 对输入的文本进行跨媒体注意力计算
        x = x + self.cross_attn(self.norm1(x), visual, visual)
        # 对文本进行非线性变换
        x = x + self.mlp(self.norm3(x))
        return x


class XCLIPPromptGenerator(nn.Module):
    """This corresponds to the `VideoSpecificPrompt` class in the original implementation."""
    # 定义 XCLIPPromptGenerator 类，用于生成 XCLIP 的提示
    def __init__(self, config):
        super().__init__()
        embed_dim = config.projection_dim
        # 使用 config.projection_dim 初始化 embed_dim
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.vision_config.layer_norm_eps)
        # 对视觉输入进行 LayerNormalization
        self.decoder = nn.ModuleList([PromptGeneratorLayer(config) for _ in range(config.prompt_layers)])
        # 初始化多层提示生成器
        self.alpha = nn.Parameter(torch.ones(embed_dim) * config.prompt_alpha)
        # 初始化提示的权重参数

    def forward(self, text, visual):
        visual = self.layernorm(visual)
        # 对视觉输入进行 LayerNormalization
        for layer in self.decoder:
            # 遍历多层提示生成器进行提示生成
            text = layer(text, visual)

        return self.alpha * text
        # 返回提示的加权结果


@add_start_docstrings(X_CLIP_START_DOCSTRING)
class XCLIPModel(XCLIPPreTrainedModel):
    config_class = XCLIPConfig
```  
    # 初始化函数，接受一个 XCLIPConfig 类型的参数，调用父类（基类）的初始化函数
    def __init__(self, config: XCLIPConfig):
        super().__init__(config)

        # 如果 config.text_config 不是 XCLIPTextConfig 类型，则抛出数值错误
        if not isinstance(config.text_config, XCLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type XCLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 如果 config.vision_config 不是 XCLIPVisionConfig 类型，则抛出数值错误
        if not isinstance(config.vision_config, XCLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type XCLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 获取 text_config 和 vision_config
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度、文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建文本模型和视觉模型
        self.text_model = XCLIPTextTransformer(text_config)
        self.vision_model = XCLIPVisionTransformer(vision_config)

        # 创建视觉投影和文本投影层
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 创建视觉提示的 LayerNorm 和投影
        self.prompts_visual_layernorm = nn.LayerNorm(self.vision_embed_dim, eps=config.vision_config.layer_norm_eps)
        self.prompts_visual_projection = nn.Parameter(torch.randn(self.vision_embed_dim, self.projection_dim))

        # 复制视觉配置，设置 MIT Transformer 相关参数
        mit_config = copy(vision_config)
        mit_config.hidden_size = vision_config.mit_hidden_size
        mit_config.intermediate_size = vision_config.mit_intermediate_size
        mit_config.num_hidden_layers = vision_config.mit_num_hidden_layers
        mit_config.num_attention_heads = vision_config.mit_num_attention_heads
        self.mit = XCLIPMultiframeIntegrationTransformer(mit_config)

        # 创建提示生成器
        self.prompts_generator = XCLIPPromptGenerator(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加文档字符串到模型前向方法
    @add_start_docstrings_to_model_forward(X_CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 获取文本特征的方法
    def get_text_features(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`XCLIPTextModel`].
    
        Examples:
    
        ```python
        >>> from transformers import AutoTokenizer, AutoModel
    
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
        >>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
    
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use X_CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 使用文本模型获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 从文本输出中获取文本特征
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
    
        return text_embeds
    
    # 获取视频特征的方法  
    def get_video_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        pass
    
    # 前向传播方法
    @add_start_docstrings_to_model_forward(X_CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=XCLIPOutput, config_class=XCLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        pass
```