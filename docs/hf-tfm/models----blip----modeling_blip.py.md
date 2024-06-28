# `.\models\blip\modeling_blip.py`

```py
# 设置文件编码格式为 UTF-8

# 导入警告模块，用于处理警告信息
import warnings

# 导入 dataclass 模块，用于定义数据类
from dataclasses import dataclass

# 导入类型提示相关的模块
from typing import Any, Optional, Tuple, Union

# 导入 PyTorch 框架及其相关模块
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import normalize

# 导入自定义模块和函数
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 导入 BLIP 模型的配置类
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig

# 导入文本相关的 BLIP 模型类
from .modeling_blip_text import BlipTextLMHeadModel, BlipTextModel

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# BLIP 模型的检查点名称
_CHECKPOINT_FOR_DOC = "Salesforce/blip-vqa-base"

# BLIP 预训练模型的列表
BLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/blip-vqa-base",
    "Salesforce/blip-vqa-capfilt-large",
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-itm-base-coco",
    "Salesforce/blip-itm-large-coco",
    "Salesforce/blip-itm-base-flickr",
    "Salesforce/blip-itm-large-flickr",
    # 查看所有 BLIP 模型列表：https://huggingface.co/models?filter=blip
]


# 从 transformers.models.clip.modeling_clip.contrastive_loss 复制的对比损失函数
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    计算对比损失，使用交叉熵损失函数。
    Args:
        logits (torch.Tensor): 模型预测的 logits.

    Returns:
        torch.Tensor: 对比损失值.
    """
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# 从 transformers.models.clip.modeling_clip.clip_loss 复制并修改为 blip_loss
def blip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    计算 BLIP 损失，包括文本和图像的对比损失的平均值。
    Args:
        similarity (torch.Tensor): 模型预测的相似性张量.

    Returns:
        torch.Tensor: BLIP 损失值.
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class BlipForConditionalGenerationModelOutput(ModelOutput):
    """
    BLIP 生成条件模型的输出类，继承自 BaseModelOutput，并包含最后隐藏状态的图像嵌入池化结果。
    该类还添加了来自文本解码器的损失项。
    """
    pass
    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Languge modeling loss from the text decoder.
            文本解码器生成的语言建模损失（如果提供了标签）。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head of the text decoder model.
            文本解码器模型的语言建模头部的预测分数。
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*):
            The image embeddings obtained after applying the Vision Transformer model to the input image.
            应用视觉Transformer模型到输入图像后得到的图像嵌入。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the model.
            模型最后一层输出的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            每层模型输出的隐藏状态，以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重，经过注意力softmax后的权重，用于计算自注意力头中的加权平均值。

    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    @property
    def decoder_logits(self):
        """
        Deprecated property to access logits. Use `logits` attribute instead.
        获取logits的过时属性。请使用`logits`属性。
        """
        warnings.warn(
            "`decoder_logits` attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the `logits` attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.logits
# 数据类，用于表示BlipTextVision模型的输出结果，继承自ModelOutput基类
@dataclass
class BlipTextVisionModelOutput(ModelOutput):
    """
    从视觉模型输出基类改编而来，还包含了最后隐藏状态的图像嵌入。这个类还添加了文本解码器的损失项。

    Args:
        loss (`torch.FloatTensor`，形状为 `(1,)`，可选，当提供`labels`时返回):
            文本解码器的语言建模损失。
        image_embeds (`torch.FloatTensor`，形状为 `(batch_size, output_dim)`，可选，当模型初始化时使用 `with_projection=True` 返回):
            通过将池化输出应用于投影层获得的图像嵌入。
        last_hidden_state (`torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列输出。
        hidden_states (`tuple(torch.FloatTensor)`，可选，当传入 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            `torch.FloatTensor` 元组（如果模型有嵌入层，则返回一个用于每层输出的嵌入输出 + 每层输出的隐藏状态），
            形状为 `(batch_size, sequence_length, hidden_size)`。

            模型每层输出的隐藏状态，以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`，可选，当传入 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            `torch.FloatTensor` 元组（每层一个），
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None  # 损失项，默认为None
    image_embeds: Optional[torch.FloatTensor] = None  # 图像嵌入，默认为None
    last_hidden_state: torch.FloatTensor = None  # 最后一层隐藏状态的输出，默认为None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 隐藏状态的元组，默认为None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 注意力权重的元组，默认为None


# 数据类，用于表示BlipImageTextMatching模型的输出结果，继承自ModelOutput基类
@dataclass
class BlipImageTextMatchingModelOutput(ModelOutput):
    """
    从视觉模型输出基类改编而来，还包含了最后隐藏状态的图像嵌入。这个类还添加了文本解码器的损失项以及图像文本相似度分数。
    """
    """
    Args:
        itm_score (`torch.FloatTensor`):
            The image-text similarity scores.
            图像与文本之间的相似性分数。
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
            文本解码器产生的语言建模损失，当提供了`labels`时返回。
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
            通过将投影层应用于池化输出得到的图像嵌入。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            模型最后一层输出的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            每层模型输出的隐藏状态组成的元组，如果模型有嵌入层则包括嵌入输出，形状为`(batch_size, sequence_length, hidden_size)`。
        vision_pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
            Last layer hidden-state of the vision of the vision-only branch of the model.
            模型视觉分支的最后一层隐藏状态。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重，经过注意力softmax后的加权平均值，用于自注意力头的计算。
        question_embeds (`torch.FloatTensor`):
            The question embeddings obtained by the text projection layer.
            通过文本投影层得到的问题嵌入。
    """

    itm_score: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    vision_pooler_output: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    question_embeds: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class BlipOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image: (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text: (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds: (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds: (`torch.FloatTensor` of shape `(batch_size, output_dim)`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output: (`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output: (`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None  # 初始化为可选的浮点数张量，用于存储图像-文本相似性的对比损失
    logits_per_image: torch.FloatTensor = None  # 存储图像嵌入与文本嵌入之间的点积得分，表示图像-文本的相似性分数
    logits_per_text: torch.FloatTensor = None  # 存储文本嵌入与图像嵌入之间的点积得分，表示文本-图像的相似性分数
    text_embeds: torch.FloatTensor = None  # 存储通过投影层应用到[`BlipTextModel`]池化输出得到的文本嵌入
    image_embeds: torch.FloatTensor = None  # 存储通过投影层应用到[`BlipVisionModel`]池化输出得到的图像嵌入
    text_model_output: BaseModelOutputWithPooling = None  # 存储[`BlipTextModel`]的输出，包含池化层的基本模型输出
    vision_model_output: BaseModelOutputWithPooling = None  # 存储[`BlipVisionModel`]的输出，包含池化层的基本模型输出

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class BlipVisionEmbeddings(nn.Module):
    """
    A module for handling vision embeddings in the Blip model.

    Args:
        config (BlipVisionConfig): Configuration object for the BlipVisionEmbeddings module.
    """

    def __init__(self, config: BlipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入维度为配置中的隐藏尺寸
        self.image_size = config.image_size  # 设置图像大小为配置中的图像尺寸
        self.patch_size = config.patch_size  # 设置补丁大小为配置中的补丁尺寸

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))  # 初始化类别嵌入参数

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )  # 创建卷积层，用于从图像中提取补丁特征嵌入

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 计算图像中的补丁数量
        self.num_positions = self.num_patches + 1  # 计算位置嵌入的数量，包括额外的类别嵌入

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))  # 初始化位置嵌入参数
    # 定义前向传播方法，接收像素数值作为输入，并返回张量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取输入张量的批量大小
        batch_size = pixel_values.shape[0]
        # 获取目标数据类型，与补丁嵌入权重的数据类型相同
        target_dtype = self.patch_embedding.weight.dtype
        # 使用补丁嵌入层处理输入像素值，将像素值转换为指定数据类型，并形成补丁嵌入
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        # 将补丁嵌入展平，并调换维度以适应后续操作
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 扩展类别嵌入以匹配批次大小，并转换为目标数据类型
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        # 将类别嵌入与补丁嵌入连接起来形成最终嵌入
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 将位置嵌入加到嵌入张量中（位置嵌入可能与输入批次大小不完全匹配）
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        # 返回最终的嵌入张量作为前向传播的输出
        return embeddings
# 从 transformers.models.clip.modeling_clip.CLIPTextEmbeddings 复制而来，将 CLIP 替换为 Blip
class BlipTextEmbeddings(nn.Module):
    def __init__(self, config: BlipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        # 初始化 token_embedding，用于词嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 初始化 position_embedding，用于位置编码
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 创建 position_ids 缓冲区，用于位置编码，持久化为非连续内存块
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 获取序列长度
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        # 如果未提供 position_ids，则使用预先初始化的 position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供 inputs_embeds，则通过 token_embedding 获取嵌入
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 获取位置编码
        position_embeddings = self.position_embedding(position_ids)
        # 将输入嵌入和位置编码相加作为最终的嵌入表示
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class BlipAttention(nn.Module):
    """来自 'Attention Is All You Need' 论文的多头注意力机制"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 必须被 num_heads 整除
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 缩放因子为 head_dim 的负半数
        self.scale = self.head_dim**-0.5
        # dropout 层
        self.dropout = nn.Dropout(config.attention_dropout)

        # 线性层 qkv，用于查询、键、值的线性变换
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)

        # 输出投影层，用于最终的线性映射
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将输入张量重塑为多头注意力矩阵的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        #
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # 获取隐藏状态张量的维度信息
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 使用 self.qkv 对隐藏状态进行变换，生成混合的查询、键、值张量
        mixed_qkv = (
            self.qkv(hidden_states)
            .reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # 计算注意力分数，使用 query 和 key 的点积
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        # 缩放注意力分数
        attention_scores = attention_scores * self.scale

        # 将注意力分数归一化为概率分布
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力分数应用 dropout
        attention_probs = self.dropout(attention_probs)

        # 如果有头部掩码，则应用到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算加权后的值张量，生成上下文层
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        # 重新调整上下文层的形状以匹配 self.projection 的输入要求
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # 使用 self.projection 将上下文层映射到输出空间
        output = self.projection(context_layer)

        # 根据需要决定是否输出注意力分数
        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs
# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Blip
class BlipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]  # 从配置中获取激活函数
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 第一个全连接层
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 第二个全连接层

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 输入经过第一个全连接层
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states = self.fc2(hidden_states)  # 经过第二个全连接层
        return hidden_states


class BlipEncoderLayer(nn.Module):
    def __init__(self, config: BlipConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = BlipAttention(config)  # 自注意力机制
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第一个层标准化层
        self.mlp = BlipMLP(config)  # MLP网络
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 第二个层标准化层

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
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
        residual = hidden_states  # 残差连接

        hidden_states = self.layer_norm1(hidden_states)  # 应用第一个层标准化
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )  # 自注意力机制计算
        hidden_states = hidden_states + residual  # 添加残差连接
        residual = hidden_states  # 更新残差连接

        hidden_states = self.layer_norm2(hidden_states)  # 应用第二个层标准化
        hidden_states = self.mlp(hidden_states)  # 经过MLP网络

        hidden_states = hidden_states + residual  # 再次添加残差连接

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，添加到输出中

        return outputs


class BlipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BlipConfig  # 模型配置类
    base_model_prefix = "blip"  # 基础模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    # 初始化模型中特定模块的权重和偏置
    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_range
        
        # 如果模块是卷积层、嵌入层或线性层，则使用正态分布初始化权重，并将偏置置零
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        # 如果模块是 BlipVisionEmbeddings 类型，则根据视觉配置初始化位置嵌入和类别嵌入
        if isinstance(module, BlipVisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(
                module.position_embedding,
                mean=0.0,
                std=factor,
            )

            nn.init.trunc_normal_(
                module.class_embedding,
                mean=0.0,
                std=factor,
            )

        # 如果模块是 LayerNorm 类型，则将偏置置零并将权重填充为 1.0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 对于线性层，如果存在偏置，则将偏置置零
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
# BLIP_START_DOCSTRING 是一个包含模型描述信息的原始文本字符串，用于指示此模型继承自 PreTrainedModel，
# 并提供了有关模型类通用方法的信息。详细内容可以在 PreTrainedModel 类的文档中找到，
# 包括下载、保存、调整输入嵌入大小、修剪头等功能。
BLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BlipConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# BLIP_TEXT_INPUTS_DOCSTRING 是关于模型文本输入参数的描述信息，包括 input_ids、attention_mask、position_ids 等参数的说明。
# 每个参数的数据类型和形状都有详细描述，以及如何获取输入 IDs 和如何使用注意力掩码等细节。
BLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoProcessor`]. See [`BlipProcessor.__call__`] for details.

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

# BLIP_VISION_INPUTS_DOCSTRING 是一个空字符串，用于定义模型视觉输入的文档字符串，目前未提供任何信息。
BLIP_VISION_INPUTS_DOCSTRING = r"""
    
    # 参数 pixel_values 是一个 torch.FloatTensor，表示像素值，其形状为 (batch_size, num_channels, height, width)
    # 默认情况下会忽略填充值。可以使用 BlipImageProcessor 获取像素值。
    # 参见 BlipImageProcessor.__call__ 获取更多详情。

    # 是否返回所有注意力层的注意力张量。在返回的张量中，详见 attentions。
    # 默认为 False。

    # 是否返回所有层的隐藏状态。在返回的张量中，详见 hidden_states。
    # 默认为 False。

    # 是否返回一个 utils.ModelOutput 对象，而不是普通的元组。
    # 默认为 False。
"""

BLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoProcessor`]. See [`BlipProcessor.__call__`] for details.

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
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`BlipImageProcessor`]. See [`BlipImageProcessor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class BlipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`BlipEncoderLayer`].

    Args:
        config (`BlipConfig`):
            The corresponding vision configuration for the `BlipEncoder`.
    """

    def __init__(self, config: BlipConfig):
        super().__init__()
        self.config = config
        # 创建一个包含多个 BlipEncoderLayer 实例的列表，列表长度为 config.num_hidden_layers
        self.layers = nn.ModuleList([BlipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点，默认为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # Determine whether to use the provided `output_attentions` value or fallback to the model's default
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # Determine whether to use the provided `output_hidden_states` value or fallback to the model's default
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Determine whether to use the provided `return_dict` value or fallback to the model's default
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Initialize empty tuples based on output configuration to store encoder states and attentions
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Start with the embedded inputs as the initial hidden states
        hidden_states = inputs_embeds

        # Iterate through each encoder layer in the model
        for idx, encoder_layer in enumerate(self.layers):
            # If configured to return hidden states, append current hidden states to encoder states
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            # Perform gradient checkpointing if enabled during training
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # Otherwise, directly pass inputs to the encoder layer
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # Update hidden states with the output from the encoder layer
            hidden_states = layer_outputs[0]

            # If configured to return attentions, append current layer's attentions to all_attentions
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # If configured to return hidden states, append final hidden states to encoder states
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # If return_dict is False, return a tuple of relevant outputs; otherwise, return a ModelOutput object
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
class BlipVisionModel(BlipPreTrainedModel):
    main_input_name = "pixel_values"  # 设置主要输入名称为"pixel_values"
    config_class = BlipVisionConfig  # 指定配置类为BlipVisionConfig

    def __init__(self, config: BlipVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = BlipVisionEmbeddings(config)  # 初始化图像嵌入模块
        self.encoder = BlipEncoder(config)  # 初始化编码器模块
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 初始化后层归一化模块

        self.post_init()  # 执行额外的初始化步骤

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=BlipVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        前向传播函数

        Returns:
            根据return_dict返回相应的输出对象
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")  # 如果未提供pixel_values则抛出数值错误

        hidden_states = self.embeddings(pixel_values)  # 将输入的pixel_values转换为嵌入向量

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 使用编码器处理嵌入向量，得到编码器的输出

        last_hidden_state = encoder_outputs[0]  # 获取编码器输出的最后隐藏状态
        last_hidden_state = self.post_layernorm(last_hidden_state)  # 对最后隐藏状态进行层归一化处理

        pooled_output = last_hidden_state[:, 0, :]  # 获取池化输出
        pooled_output = self.post_layernorm(pooled_output)  # 对池化输出进行层归一化处理

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]  # 如果不返回字典，则返回元组形式的输出

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )  # 返回包含池化输出和其他编码器输出的BaseModelOutputWithPooling对象

    def get_input_embeddings(self):
        return self.embeddings  # 返回嵌入模块的实例
    def __init__(self, config: BlipConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 检查配置对象中的文本配置是否为BlipTextConfig类型，如果不是则抛出数值错误异常
        if not isinstance(config.text_config, BlipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type BlipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查配置对象中的视觉配置是否为BlipVisionConfig类型，如果不是则抛出数值错误异常
        if not isinstance(config.vision_config, BlipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type BlipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 从配置对象中获取文本配置和视觉配置
        text_config = config.text_config
        vision_config = config.vision_config

        # 初始化模型的投影维度、文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本模型和视觉模型，分别使用文本配置和视觉配置
        self.text_model = BlipTextModel(text_config)
        self.vision_model = BlipVisionModel(vision_config)

        # 初始化视觉投影层和文本投影层，分别映射视觉和文本嵌入到投影维度空间，无偏置
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        # 初始化对数尺度参数，使用配置中的初始值
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 调用后初始化函数，用于权重初始化和最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`BlipTextModel`].

        Examples:

        ```
        >>> from transformers import AutoProcessor, BlipModel

        >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # 如果未指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用文本模型，传入输入的ids、注意力掩码、位置ids和是否返回字典的标志
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        # 获取文本模型的汇总输出（pooled output）
        pooled_output = text_outputs[1]

        # 将汇总输出投影到文本投影层，得到文本特征
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`BlipVisionModel`].

        Examples:

        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipModel

        >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```

        Initialize `return_dict` to `self.config.use_return_dict` if `return_dict` is not provided.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取视觉模型的输出，可以选择是否返回字典格式的输出
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=return_dict)

        # 从视觉模型的输出中获取池化后的特征向量
        pooled_output = vision_outputs[1]  # pooled_output
        # 将池化后的特征向量应用于视觉投影层，得到最终的图像特征表示
        image_features = self.visual_projection(pooled_output)

        # 返回图像特征表示
        return image_features

    @add_start_docstrings_to_model_forward(BLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipOutput, config_class=BlipConfig)
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
        """
        BLIP模型的前向传播方法。

        Args:
            input_ids (Optional[torch.LongTensor], optional): 输入的token IDs. Defaults to None.
            pixel_values (Optional[torch.FloatTensor], optional): 输入的像素值. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): 注意力遮罩. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): 位置 IDs. Defaults to None.
            return_loss (Optional[bool], optional): 是否返回损失值. Defaults to None.
            output_attentions (Optional[bool], optional): 是否返回注意力权重. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否返回隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否以字典格式返回输出. Defaults to None.

        Returns:
            BLIP模型的输出，类型为`BlipOutput`，根据`return_dict`参数决定返回方式.

        """
@add_start_docstrings(
    """
    BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass
    `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,
    the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption
    from the text input. If no text input is provided, the decoder will start with the [BOS] token only.
    """,
    BLIP_START_DOCSTRING,
)
class BlipForConditionalGeneration(BlipPreTrainedModel):
    # 定义配置类为 BlipConfig
    config_class = BlipConfig
    # 定义权重共享的键列表
    _tied_weights_keys = ["text_decoder.cls.predictions.decoder.bias"]
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    def __init__(self, config: BlipConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 使用 BlipVisionModel 初始化视觉模型
        self.vision_model = BlipVisionModel(config.vision_config)

        # 使用 BlipTextLMHeadModel 初始化文本解码器
        self.text_decoder = BlipTextLMHeadModel(config.text_config)

        # 设置解码器的起始输入为 BOS 标记的 ID
        self.decoder_input_ids = config.text_config.bos_token_id
        # 设置解码器的填充标记的 ID
        self.decoder_pad_token_id = config.text_config.pad_token_id

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回视觉模型的嵌入模块
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipForConditionalGenerationModelOutput, config_class=BlipVisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        # forward 方法的参数说明文档添加 BLIP_VISION_INPUTS_DOCSTRING
        # 替换返回文档字符串的输出类型和配置类为 BlipVisionConfig
        ):
    ) -> Union[Tuple, BlipForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A picture of"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")

        >>> outputs = model(**inputs)
        ```"""

        # 如果 return_dict 参数未指定，则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 output_attentions 参数未指定，则使用模型配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数未指定，则使用模型配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 使用视觉模型处理像素值，根据参数返回不同的结果
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取视觉输出的第一个元素，即图像嵌入
        image_embeds = vision_outputs[0]

        # 使用文本解码器处理输入的信息，生成输出结果
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        # 如果 return_dict 为 False，则返回多个输出元组
        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        # 如果 return_dict 为 True，则返回 BlipForConditionalGenerationModelOutput 对象
        return BlipForConditionalGenerationModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            pixel_values (*torch.FloatTensor* of shape *(batch_size, num_channels, image_height, image_width)*:
                Input image to be processed
            input_ids (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:


        Examples:
        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForConditionalGeneration

        >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        two cats sleeping on a couch
        ```
        """

        # 获取批处理大小
        batch_size = pixel_values.shape[0]
        
        # 使用视觉模型处理输入图像，获取视觉输出
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        # 从视觉输出中提取图像嵌入
        image_embeds = vision_outputs[0]

        # 创建图像注意力掩码，用于避免在填充标记索引上执行注意力
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        # 如果输入的input_ids是列表，则转换为torch.LongTensor
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        # 如果input_ids为None，则创建包含开始和结束标记的输入序列
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        # 设置输入序列的开始标记为配置中的开始标记
        input_ids[:, 0] = self.config.text_config.bos_token_id

        # 调整注意力掩码，移除最后一个标记以对齐输入序列
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        # 使用文本解码器生成文本输出
        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        # 返回生成的输出
        return outputs
@add_start_docstrings(
    """
    BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text
    decoder. The vision encoder will encode the input image, the text encoder will encode the input question together
    with the encoding of the image, and the text decoder will output the answer to the question.
    """,
    BLIP_START_DOCSTRING,
)
class BlipForQuestionAnswering(BlipPreTrainedModel):
    config_class = BlipConfig
    _tied_weights_keys = ["text_decoder.cls.predictions.decoder.bias"]

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        # Initialize the vision encoder model using the provided vision configuration
        self.vision_model = BlipVisionModel(config.vision_config)

        # Initialize the text encoder model using the provided text configuration,
        # with pooling layer excluded
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # Initialize the text decoder model using the provided text configuration
        self.text_decoder = BlipTextLMHeadModel(config.text_config)

        # Store special token IDs for decoder inputs
        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id

        # Initialize weights and perform any necessary post-initialization steps
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # Return the patch embedding module from the vision encoder's embeddings
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the BLIP model for question answering.

        Args:
            input_ids (:obj:`torch.LongTensor`):
                Indices of input sequence tokens in the vocabulary.
            pixel_values (:obj:`torch.FloatTensor`):
                Pixel values of images (shape batch_size x channels x height x width).
            decoder_input_ids (:obj:`torch.LongTensor`, optional):
                Optional input for decoder. If provided, computes the loss and returns the logits.
            decoder_attention_mask (:obj:`torch.LongTensor`, optional):
                Optional attention mask for the decoder input.
            attention_mask (:obj:`torch.LongTensor`, optional):
                Optional attention mask for the input.
            output_attentions (:obj:`bool`, optional):
                Whether to return attentions weights.
            output_hidden_states (:obj:`bool`, optional):
                Whether to return hidden states.
            labels (:obj:`torch.LongTensor`, optional):
                Labels for computing the cross-entropy loss.
            return_dict (:obj:`bool`, optional):
                Whether to return a dictionary.

        Returns:
            :class:`~transformers.BlipTextVisionModelOutput`: A subclass of :class:`~transformers.ModelOutput`.
        """
        # Implementation of the forward pass is provided by the decorated function

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ):
        """
        Generate output sequences for the given inputs.

        Args:
            input_ids (:obj:`torch.LongTensor`):
                Indices of input sequence tokens in the vocabulary.
            pixel_values (:obj:`torch.FloatTensor`):
                Pixel values of images (shape batch_size x channels x height x width).
            attention_mask (:obj:`torch.LongTensor`, optional):
                Optional attention mask for the input.
            **generate_kwargs:
                Additional keyword arguments for generation (e.g., max_length, num_beams).

        Returns:
            :obj:`torch.LongTensor`: Generated sequences.
        """
        # Implementation of the generation process is provided by the decorated function
    ) -> torch.LongTensor:
        r"""
        重写 *generate* 函数以便将模型用作条件生成器

        Parameters:
            input_ids (*torch.LongTensor* of shape *(batch_size, sequence_length)*):
                用作生成提示的序列。
            pixel_values (*torch.FloatTensor* of shape *(batch_size, num_channels, image_height, image_width)*:
                要处理的输入图像。
            attention_mask (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                遮罩，避免在填充令牌索引上执行注意力。遮罩值选在 `[0, 1]` 中。`1` 表示未被掩盖的令牌，`0` 表示被掩盖的令牌。
            **generate_kwargs:
                传递给解码器 *generate* 函数的额外参数

        Examples:
        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForQuestionAnswering

        >>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are in the picture?"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]  # 提取视觉模型的输出中的图像嵌入表示

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)  # 创建图像的注意力遮罩

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)  # 如果输入的是列表，将其转换为 torch.LongTensor

        question_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )  # 使用文本编码器处理输入的文本和图像嵌入

        question_embeds = question_outputs[0]  # 提取文本编码器的输出中的问题嵌入表示

        question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long).to(question_embeds.device)  # 创建问题的注意力遮罩

        bos_ids = torch.full(
            (question_embeds.size(0), 1), fill_value=self.decoder_start_token_id, device=question_embeds.device
        )  # 创建包含起始标记的张量

        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
            **generate_kwargs,
        )  # 使用文本解码器生成输出序列

        return outputs  # 返回生成的输出序列
# 定义 BLIP 图像文本检索模型，包含视觉和文本投影器以及顶部的分类头部。用于图像文本检索任务，给定图像和文本，模型返回文本与图像相关性的概率。
@add_start_docstrings(
    """
    BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of
    image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to
    the image.
    """,
    BLIP_START_DOCSTRING,
)
class BlipForImageTextRetrieval(BlipPreTrainedModel):
    # 使用 BlipConfig 类型的配置
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        # 调用父类构造函数，传入配置
        super().__init__(config)

        # 初始化视觉模型，使用 BlipVisionModel 和视觉配置
        self.vision_model = BlipVisionModel(config.vision_config)

        # 初始化文本编码器，使用 BlipTextModel 和文本配置，不添加池化层
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # 视觉投影层，线性变换视觉隐藏状态的维度到图像文本隐藏大小
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)

        # 文本投影层，线性变换文本隐藏状态的维度到图像文本隐藏大小
        self.text_proj = nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)

        # 图像文本匹配头部，线性层输出大小为 2，用于二分类任务
        self.itm_head = nn.Linear(config.text_config.hidden_size, 2)

        # 解码器的填充标记 ID，根据配置的填充标记 ID 初始化
        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )

        # 解码器的起始标记 ID，根据配置的起始标记 ID 初始化
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入，返回视觉模型的 patch 嵌入层
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipTextVisionModelOutput, config_class=BlipVisionConfig)
    # 重写 forward 方法，使用 BLIP_VISION_INPUTS_DOCSTRING 和 BlipTextVisionModelOutput 来替换返回值的文档字符串
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        use_itm_head: Optional[bool] = True,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数：input_ids 是文本输入的 token ID，pixel_values 是视觉输入的像素值
        # use_itm_head 控制是否使用图像文本匹配头部，attention_mask 控制注意力机制的掩码
        # output_attentions 和 output_hidden_states 控制是否输出注意力权重和隐藏状态
        # return_dict 控制是否返回字典形式的输出
        #
        # 输出类型为 BlipTextVisionModelOutput，配置类为 BlipVisionConfig
    ) -> Union[Tuple, BlipTextVisionModelOutput]:
        r"""
        Returns:

        Examples:

        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForImageTextRetrieval

        >>> model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "an image of a cat"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```
        """
        # 如果 return_dict 参数不为 None，则使用该值；否则使用 self.config.use_return_dict 的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 output_attentions 参数不为 None，则使用该值；否则使用 self.config.output_attentions 的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数不为 None，则使用该值；否则使用 self.config.output_hidden_states 的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 使用 vision_model 处理图像数据，获取视觉模型的输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取图像的嵌入表示
        image_embeds = vision_outputs[0]
        # 创建与图像嵌入相同大小的注意力掩码
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        # 如果 use_itm_head 为真，则使用 text_encoder 处理输入问题文本，并应用 itm_head 进行匹配分数计算
        if use_itm_head:
            # 使用 text_encoder 处理文本数据，将图像嵌入作为 encoder_hidden_states 提供给文本编码器
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=return_dict,
            )
            # 如果 return_dict 为 False，则使用第一个元素作为输出；否则使用 last_hidden_state
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            # 使用 itm_head 计算问题嵌入的匹配分数
            output = self.itm_head(question_embeds[:, 0, :])
        else:
            # 使用 text_encoder 处理文本数据，获取问题文本的嵌入表示
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            # 如果 return_dict 为 False，则使用第一个元素作为输出；否则使用 last_hidden_state
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            # 规范化图像嵌入，并通过 vision_proj 将其投影到与问题文本嵌入相同的空间中
            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            # 规范化问题文本嵌入，并通过 text_proj 进行同样的投影
            text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

            # 计算图像嵌入与问题文本嵌入之间的相似度分数
            output = image_feat @ text_feat.t()

        # 如果 return_dict 为 False，则返回多个元组，确保输出中没有 None 值
        if not return_dict:
            outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
            return tuple(output for output in outputs if output is not None)

        # 如果 return_dict 为 True，则返回 BlipImageTextMatchingModelOutput 对象，包含 ITM 计算的结果和相关信息
        return BlipImageTextMatchingModelOutput(
            itm_score=output,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            question_embeds=question_embeds,
        )
```