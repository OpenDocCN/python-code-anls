# `.\transformers\models\clip\modeling_clip.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权使用
# 除非符合许可证要求或书面同意，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
""" PyTorch CLIP model."""

# 导入所需模块和库
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
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"

# 预训练模型存档列表
CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/clip-vit-base-patch32",
    # 查看所有 CLIP 模型 https://huggingface.co/models?filter=clip
]

# 对比损失函数，改编自 https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# CLIP 损失函数
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

# CLIP 视觉模型输出的基类，包含最后隐藏状态的池化图像嵌入
@dataclass
class CLIPVisionModelOutput(ModelOutput):
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
            获取通过将投影层应用于pooler_output获得的图像嵌入。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            模型最后一层输出的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
            每一层输出的模型隐藏状态的元组，以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            经过注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.modeling_outputs import ModelOutput

# 基于 ModelOutput 的文本模型输出的基类，同时包含最后隐藏状态的汇集
@dataclass
class CLIPTextModelOutput(ModelOutput):
    # 文本嵌入，当模型用 with_projection=True 初始化时可选返回
    text_embeds: Optional[torch.FloatTensor] = None
    # 模型最后一层的隐藏状态序列
    last_hidden_state: torch.FloatTensor = None
    # 隐藏状态的元组，当 output_hidden_states=True 传递时或 config.output_hidden_states=True 时返回
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力的元组，当 output_attentions=True 传递时或 config.output_attentions=True 时返回
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """

    # 定义类的属性，用于存储不同类型的数据
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    # 定义类的方法，将属性转换为元组
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果属性不是"text_model_output"或"vision_model_output"，则直接返回属性值；否则调用属性的to_tuple()方法
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 定义 CLIP 视觉嵌入模块的类
class CLIPVisionEmbeddings(nn.Module):
    # 初始化函数，接受 CLIP 视觉配置对象作为参数
    def __init__(self, config: CLIPVisionConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模块的配置属性
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 创建类别嵌入参数，用于表示图像的整体特征
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 创建图像块嵌入层，将图像分块并转换为嵌入表示
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算图像块数量和位置嵌入维度
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        # 创建位置嵌入层
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 将位置 ID 缓存，以便在序列化时被导出
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # 前向传播函数，接受像素值作为输入，返回嵌入表示
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        # 确定目标数据类型为与权重相同的数据类型
        target_dtype = self.patch_embedding.weight.dtype
        # 对输入的像素值进行块嵌入
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 创建类别嵌入张量，并将其与块嵌入张量连接起来
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 添加位置嵌入张量到嵌入张量中
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # 返回嵌入张量
        return embeddings


# 定义 CLIP 文本嵌入模块的类
class CLIPTextEmbeddings(nn.Module):
    # 初始化函数，接受 CLIP 文本配置对象作为参数
    def __init__(self, config: CLIPTextConfig):
        # 调用父类的初始化函数
        super().__init__()
        embed_dim = config.hidden_size

        # 创建词嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 创建位置嵌入层
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 创建位置 ID 缓存，以便在序列化时被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # 前向传播函数，接受输入文本 ID 序列或嵌入张量，返回嵌入表示
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 获取序列长度
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        # 如果位置 ID 为空，则使用预定义的位置 ID
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果输入的嵌入张量为空，则使用词嵌入层将文本 ID 序列转换为嵌入表示
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 获取位置嵌入张量
        position_embeddings = self.position_embedding(position_ids)
        # 将词嵌入张量和位置嵌入张量相加得到最终的嵌入表示
        embeddings = inputs_embeds + position_embeddings

        # 返回嵌入表示
        return embeddings


# 定义 CLIP 注意力模块的类，基于 "Attention Is All You Need" 论文中的多头注意力机制
class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置参数
        self.config = config
        # 设置嵌入维度
        self.embed_dim = config.hidden_size
        # 设置注意力头的数量
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查是否能够整除
        if self.head_dim * self.num_heads != self.embed_dim:
            # 如果不能整除，抛出数值错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 计算缩放因子
        self.scale = self.head_dim**-0.5
        # 设置注意力的丢弃率
        self.dropout = config.attention_dropout

        # 初始化线性变换层
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 将张量重塑为指定形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
class CLIPMLP(nn.Module):
    def __init__(self, config):
        # 初始化 CLIPMLP 类
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        # 初始化 CLIPEncoderLayer 类
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
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
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    # 初始化模型的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        # 如果是文本嵌入层，初始化 token_embedding 和 position_embedding 的权重
        if isinstance(module, CLIPTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        # 如果是视觉嵌入层，初始化 class_embedding、patch_embedding 和 position_embedding 的权重
        elif isinstance(module, CLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # 如果是注意力层，初始化注意力相关的投影矩阵权重
        elif isinstance(module, CLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果是多层感知机（MLP），初始化全连接层的权重
        elif isinstance(module, CLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # 如果是整个 CLIP 模型，初始化文本和视觉投影层的权重
        elif isinstance(module, CLIPModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        # 如果是具有投影层的视觉模型，初始化视觉投影层的权重
        elif isinstance(module, CLIPVisionModelWithProjection):
            nn.init.normal_(
                module.visual_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        # 如果是具有投影层的文本模型，初始化文本投影层的权重
        elif isinstance(module, CLIPTextModelWithProjection):
            nn.init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )

        # 如果是 LayerNorm 层，初始化偏置和缩放参数
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是带有偏置的线性层，初始化偏置为零
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
# CLIP_START_DOCSTRING 是模型文档字符串的起始部分，介绍了该模型继承自 PreTrainedModel，提供了通用方法，
# 同时也是 PyTorch 的 torch.nn.Module 子类，可以按照常规 PyTorch Module 使用，并参考 PyTorch 文档。
# 参数 config 是 CLIPConfig 类的实例，包含模型的所有参数。初始化时只加载配置文件，不加载模型权重。
# 若要加载模型权重，可以使用 PreTrainedModel.from_pretrained 方法。
CLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# CLIP_TEXT_INPUTS_DOCSTRING 是处理文本输入的函数的文档字符串，详细说明了各个参数的含义和用法。
CLIP_TEXT_INPUTS_DOCSTRING = r"""
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

# CLIP_VISION_INPUTS_DOCSTRING 是处理视觉输入的函数的文档字符串，当前为空。
CLIP_VISION_INPUTS_DOCSTRING = r"""
"""   
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下将忽略填充。可以使用[`AutoImageProcessor`]获取像素值。有关详细信息，请参见[`CLIPImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请查看返回的张量下的`attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请查看返回的张量下的`hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通元组。
"""

CLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下将忽略填充。

            可以使用 [`AutoTokenizer`] 获取索引。详情请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            避免对填充标记索引执行注意力操作的掩码。掩码值选择在 `[0, 1]` 范围内：

            - 对于**未掩码**的标记，值为 1，
            - 对于**掩码**的标记，值为 0。

            [什么是注意力掩码?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。

            [什么是位置 ID?](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下将忽略填充。像素值可以使用 [`AutoImageProcessor`] 获取。详情请参阅 [`CLIPImageProcessor.__call__`]。

        return_loss (`bool`, *optional*):
            是否返回对比损失。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。返回的张量中的 `attentions` 有更多细节。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。返回的张量中的 `hidden_states` 有更多细节。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""


class CLIPEncoder(nn.Module):
    """
    Transformer 编码器，由 `config.num_hidden_layers` 个自注意力层组成。每一层是一个 [`CLIPEncoderLayer`]。

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        # 创建由多个 CLIPEncoderLayer 组成的层列表
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class CLIPTextTransformer(nn.Module):
    # 初始化函数，接受一个 CLIPTextConfig 类型的参数
    def __init__(self, config: CLIPTextConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的配置信息保存到 self.config 中
        self.config = config
        # 从配置信息中获取隐藏层大小作为嵌入维度
        embed_dim = config.hidden_size
        # 创建 CLIPTextEmbeddings 对象并保存到 self.embeddings 中
        self.embeddings = CLIPTextEmbeddings(config)
        # 创建 CLIPEncoder 对象并保存到 self.encoder 中
        self.encoder = CLIPEncoder(config)
        # 创建一个 LayerNorm 层，用于最终输出的归一化
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 用于计算 `pooled_output` 的结束标记 token ID
        self.eos_token_id = config.eos_token_id

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """The text model from CLIP without any head or projection on top.""",  # 添加起始文档字符串，说明这是一个不带头或顶部投影的 CLIP 文本模型
    CLIP_START_DOCSTRING,  # 添加起始文档字符串，引用了 CLIP_START_DOCSTRING
)
class CLIPTextModel(CLIPPreTrainedModel):  # 定义 CLIPTextModel 类，继承自 CLIPPreTrainedModel
    config_class = CLIPTextConfig  # 设置配置类为 CLIPTextConfig

    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]  # 定义不需要分割的模块列表

    def __init__(self, config: CLIPTextConfig):  # 初始化函数，传入配置参数 config
        super().__init__(config)  # 调用父类构造函数
        self.text_model = CLIPTextTransformer(config)  # 创建 CLIPTextTransformer 对象，并赋值给 self.text_model
        # Initialize weights and apply final processing
        self.post_init()  # 初始化权重并应用最终处理

    def get_input_embeddings(self) -> nn.Module:  # 定义获取输入嵌入的方法
        return self.text_model.embeddings.token_embedding  # 返回文本模型的嵌入层的 token_embedding

    def set_input_embeddings(self, value):  # 定义设置输入嵌入的方法
        self.text_model.embeddings.token_embedding = value  # 设置文本模型的嵌入层的 token_embedding

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)  # 添加起始文档字符串到模型前向传播函数，引用了 CLIP_TEXT_INPUTS_DOCSTRING
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)  # 替换返回文档字符串的注释，输出类型为BaseModelOutputWithPooling，配置类为CLIPTextConfig
    def forward(  # 定义前向传播函数
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:  # 前向传播函数的类型注解
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""  # 前向传播函数的返回文档字符串，包含示例
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果 return_dict 不为 None，则使用 return_dict，否则使用配置中的 use_return_dict

        return self.text_model(  # 调用文本模型的前向传播
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPVisionTransformer(nn.Module):  # 定义 CLIPVisionTransformer 类，继承自 nn.Module
    def __init__(self, config: CLIPVisionConfig):  # 初始化函数，传入配置参数 config
        super().__init__()  # 调用父类构造函数
        self.config = config  # 将配置参数 config 赋值给 self.config
        embed_dim = config.hidden_size  # 获取隐藏大小配置参数

        self.embeddings = CLIPVisionEmbeddings(config)  # 创建 CLIPVisionEmbeddings 对象，并赋值给 self.embeddings
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 创建 LayerNorm 层，并赋值给 self.pre_layrnorm
        self.encoder = CLIPEncoder(config)  # 创建 CLIPEncoder 对象，并赋值给 self.encoder
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 创建 LayerNorm 层，并赋值给 self.post_layernorm

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)  # 添加起始文档字符串到模型前向传播函数，引用了 CLIP_VISION_INPUTS_DOCSTRING
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)  # 替换返回文档字符串的注释，输出类型为BaseModelOutputWithPooling，配置类为CLIPVisionConfig
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回模型前向传播的结果。

        Args:
            pixel_values: 输入像素值的张量，可选参数，默认为 None。
            output_attentions: 是否输出注意力权重，可选参数，默认为 None。
            output_hidden_states: 是否输出隐藏状态，可选参数，默认为 None。
            return_dict: 是否返回字典格式的结果，可选参数，默认为 None。

        Returns:
            模型的输出结果。

        """
        # 如果未提供像素值，则抛出 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将输入像素值经过嵌入层和预层归一化处理
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        # 使用编码器处理嵌入后的隐藏状态
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出中的最后一层隐藏状态和池化输出
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不返回字典格式的结果，则将结果以元组形式返回
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回字典格式的结果
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用 add_start_docstrings 装饰器为 CLIPVisionModel 类添加文档字符串，描述其作为 CLIP 的视觉模型，没有额外的头部或顶部投影
# 通过 CLIP_START_DOCSTRING 引入 CLIP 模型的基本文档字符串
class CLIPVisionModel(CLIPPreTrainedModel):
    # 指定配置类为 CLIPVisionConfig
    config_class = CLIPVisionConfig
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 不需要拆分的模块列表
    _no_split_modules = ["CLIPEncoderLayer"]

    # 初始化方法
    def __init__(self, config: CLIPVisionConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 CLIP 视觉变换器模型
        self.vision_model = CLIPVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 前向传播方法
    # 使用 add_start_docstrings_to_model_forward 装饰器添加前向传播的文档字符串
    # 使用 replace_return_docstrings 装饰器替换返回值的文档字符串
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 如果 return_dict 为 None，则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 CLIP 视觉模型的前向传播
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 使用 add_start_docstrings 装饰器为 CLIPModel 类添加文档字符串，引入 CLIP 的基本文档字符串
class CLIPModel(CLIPPreTrainedModel):
    # 指定配置类为 CLIPConfig
    config_class = CLIPConfig
    # 初始化函数，接受一个 CLIPConfig 类型的参数
    def __init__(self, config: CLIPConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 检查 config.text_config 是否为 CLIPTextConfig 类型，如果不是则抛出数值错误
        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.vision_config 是否为 CLIPVisionConfig 类型，如果不是则抛出数值错误
        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 获取 text_config 和 vision_config
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度、文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本模型和视觉模型
        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        # 初始化视觉投影和文本投影
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加文档字符串到模型前向函数
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # 使用 CLIP 模型的配置字段替代视觉和文本组件的字段（如果指定了）
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用文本模型，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从文本输出中获取池化输出
        pooled_output = text_outputs[1]
        # 将池化输出投影到文本特征空间
        text_features = self.text_projection(pooled_output)

        # 返回文本特征
        return text_features

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass the pixel values and other specified parameters to the vision model
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the pooled output from the vision model
        pooled_output = vision_outputs[1]  # pooled_output
        # Apply the visual projection layer to the pooled output to get image features
        image_features = self.visual_projection(pooled_output)

        # Return the image features
        return image_features

    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
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
# 为 CLIP 文本模型添加一个投影层（在汇总输出之上的线性层）
@add_start_docstrings(
    """
    CLIP Text Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLIP_START_DOCSTRING,
)
class CLIPTextModelWithProjection(CLIPPreTrainedModel):
    # 使用 CLIPTextConfig 类型的配置
    config_class = CLIPTextConfig

    # 不需要拆分的模块列表
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)

        # 初始化文本模型
        self.text_model = CLIPTextTransformer(config)

        # 初始化文本投影层
        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    # 前向传播函数
    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPTextModelOutput, config_class=CLIPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPTextModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModelWithProjection

        >>> model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取文本模型的输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取汇总输出
        pooled_output = text_outputs[1]

        # 对汇总输出进行投影
        text_embeds = self.text_projection(pooled_output)

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


# 为 CLIP 视觉模型添加一个投影层（在汇总输出之上的线性层）
@add_start_docstrings(
    """
    CLIP Vision Model with a projection layer on top (a linear layer on top of the pooled output).
    # 多行字符串的结束标记，用于多行注释的结束
    """,
    # 定义了一个标识符，用于指示文档字符串的开始
    CLIP_START_DOCSTRING,
class CLIPVisionModelWithProjection(CLIPPreTrainedModel):
    # 指定配置类
    config_class = CLIPVisionConfig
    # 主输入名称
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        # 调用父类初始化方法
        super().__init__(config)

        # 创建 CLIP 视觉模型
        self.vision_model = CLIPVisionTransformer(config)

        # 创建视觉投影层，将隐藏状态映射到投影维度
        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回视觉模型中的补丁嵌入层
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPVisionModelOutput, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPVisionModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModelWithProjection

        >>> model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> image_embeds = outputs.image_embeds
        ```"""
        # 如果未提供 return_dict，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取视觉模型的输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取池化后的输出
        pooled_output = vision_outputs[1]

        # 将池化后的输出进行视觉投影
        image_embeds = self.visual_projection(pooled_output)

        # 如果不要求返回字典，则返回元组
        if not return_dict:
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        # 否则，返回 CLIP 视觉模型输出对象
        return CLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )
```