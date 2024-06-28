# `.\models\clip\modeling_clip.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，2021年由OpenAI团队和HuggingFace团队版权所有
#
# 根据Apache许可证2.0版（"许可证"）授权；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，不提供任何明示或暗示的担保或条件。
# 请查阅许可证了解具体法律权限和限制。
""" PyTorch CLIP模型。"""

# 导入必要的模块和库
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入内部模块和函数
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 文档字符串中的常规说明
_CONFIG_FOR_DOC = "CLIPConfig"
_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"

# 图像分类相关文档字符串
_IMAGE_CLASS_CHECKPOINT = "openai/clip-vit-base-patch32"
_IMAGE_CLASS_EXPECTED_OUTPUT = "LABEL_0"

# 预训练的CLIP模型存档列表
CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/clip-vit-base-patch32",
    # 更多CLIP模型详见 https://huggingface.co/models?filter=clip
]

# 对比损失函数，改编自
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # 使用交叉熵损失计算对比损失
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    # 计算对比损失的同时考虑文本和图像
    caption_loss = contrastive_loss(similarity)  # 计算文本损失
    image_loss = contrastive_loss(similarity.t())  # 计算图像损失
    return (caption_loss + image_loss) / 2.0

@dataclass
class CLIPVisionModelOutput(ModelOutput):
    """
    CLIP视觉模型输出的基类，同时包含最后隐藏状态的池化图像嵌入。
    """
    """
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 可选参数，表示模型初始化时如果使用了投影层，则返回此投影层应用于池化输出后得到的图像嵌入向量
    image_embeds: Optional[torch.FloatTensor] = None
    # 必需参数，表示模型最后一层的输出隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`
    last_hidden_state: torch.FloatTensor = None
    # 可选参数，当设置 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，
    # 是一个元组，包含模型每一层的隐藏状态的输出，如果模型有嵌入层则还包括初始嵌入的输出
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选参数，当设置 `output_attentions=True` 或 `config.output_attentions=True` 时返回，
    # 是一个元组，包含每一层的注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 dataclass 装饰器声明一个数据类，表示 CLIP 模型的文本输出结果，继承自 ModelOutput。
@dataclass
class CLIPTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 可选字段：文本嵌入，类型为 torch.FloatTensor，形状为 (batch_size, output_dim)
    text_embeds: Optional[torch.FloatTensor] = None
    # 必需字段：最后一个隐藏层的隐藏状态，类型为 torch.FloatTensor，形状为 (batch_size, sequence_length, hidden_size)
    last_hidden_state: torch.FloatTensor = None
    # 可选字段：各层的隐藏状态元组，每个元素是 torch.FloatTensor，形状为 (batch_size, sequence_length, hidden_size)
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选字段：注意力权重元组，每个元素是 torch.FloatTensor，形状为 (batch_size, num_heads, sequence_length, sequence_length)
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# 使用 dataclass 装饰器声明一个数据类，表示 CLIP 模型的输出结果，继承自 ModelOutput。
@dataclass
class CLIPOutput(ModelOutput):
    """
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

    # Optional: Loss tensor representing contrastive loss for image-text similarity
    loss: Optional[torch.FloatTensor] = None
    # Optional: Scores indicating image-text similarity (image_batch_size x text_batch_size)
    logits_per_image: torch.FloatTensor = None
    # Optional: Scores indicating text-image similarity (text_batch_size x image_batch_size)
    logits_per_text: torch.FloatTensor = None
    # Optional: Text embeddings derived from CLIPTextModel's pooled output
    text_embeds: torch.FloatTensor = None
    # Optional: Image embeddings derived from CLIPVisionModel's pooled output
    image_embeds: torch.FloatTensor = None
    # Optional: Output object from CLIPTextModel with pooling
    text_model_output: BaseModelOutputWithPooling = None
    # Optional: Output object from CLIPVisionModel with pooling
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        # Convert all attributes except 'text_model_output' and 'vision_model_output' to a tuple
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
class CLIPVisionEmbeddings(nn.Module):
    # CLIP 视觉嵌入模块，继承自 nn.Module 类
    def __init__(self, config: CLIPVisionConfig):
        # 初始化函数，接受 CLIPVisionConfig 类型的配置参数
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 类别嵌入向量，作为可学习参数
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 图像块嵌入层，使用 Conv2d 实现，将图像分割为块并转换为嵌入表示
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算图像中的块数和位置嵌入维度
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        # 注册位置索引张量，用于嵌入位置编码
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 前向传播函数，接收像素值张量并返回嵌入表示的张量
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # 对输入像素值进行图像块嵌入
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 类别嵌入张量扩展到每个样本
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # 将类别嵌入和图像块嵌入连接成一个张量
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 加上位置嵌入
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPTextEmbeddings(nn.Module):
    # CLIP 文本嵌入模块，继承自 nn.Module 类
    def __init__(self, config: CLIPTextConfig):
        # 初始化函数，接受 CLIPTextConfig 类型的配置参数
        super().__init__()
        embed_dim = config.hidden_size

        # 词汇表嵌入层和位置嵌入层，使用 Embedding 实现
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 注册位置索引张量，用于嵌入位置编码
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 前向传播函数，接收输入的词汇 IDs 或嵌入表示，返回文本嵌入表示的张量
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 获取位置嵌入
        position_embeddings = self.position_embedding(position_ids)
        # 计算最终的文本嵌入张量
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # CLIP 注意力模块，继承自 nn.Module 类
    # 初始化函数，用于初始化一个注意力机制模型实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置参数保存在实例中
        self.config = config
        # 从配置中获取隐藏层大小作为嵌入维度
        self.embed_dim = config.hidden_size
        # 从配置中获取注意力头的数量
        self.num_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 检查 embed_dim 是否能被 num_heads 整除，否则抛出异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 计算缩放因子，用于注意力分数的缩放
        self.scale = self.head_dim**-0.5
        # 从配置中获取注意力机制的 dropout 率
        self.dropout = config.attention_dropout

        # 初始化线性变换层，用于查询、键、值和输出的投影
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 辅助函数，用于调整张量形状以适应多头注意力的计算
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重新形状为 [bsz, seq_len, num_heads, head_dim]
        reshaped_tensor = tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
        # 交换维度，变成 [bsz, num_heads, seq_len, head_dim]
        transposed_tensor = reshaped_tensor.transpose(1, 2).contiguous()
        return transposed_tensor

    # 前向传播函数，实现注意力机制的计算过程
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 保存配置信息到实例变量中
        self.activation_fn = ACT2FN[config.hidden_act]  # 根据配置选择激活函数
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 创建线性层 fc1
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 创建线性层 fc2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 输入通过线性层 fc1
        hidden_states = self.activation_fn(hidden_states)  # 应用激活函数
        hidden_states = self.fc2(hidden_states)  # 再次通过线性层 fc2
        return hidden_states  # 返回处理后的隐藏状态


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # 保存隐藏尺寸到实例变量
        self.self_attn = CLIPAttention(config)  # 创建自注意力机制
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 创建层归一化层1
        self.mlp = CLIPMLP(config)  # 创建多层感知机 MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 创建层归一化层2

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
        residual = hidden_states  # 保存输入隐藏状态作为残差连接的起点

        hidden_states = self.layer_norm1(hidden_states)  # 应用层归一化层1
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )  # 使用自注意力机制处理隐藏状态

        hidden_states = residual + hidden_states  # 添加残差连接

        residual = hidden_states  # 更新残差连接起点为当前隐藏状态

        hidden_states = self.layer_norm2(hidden_states)  # 应用层归一化层2
        hidden_states = self.mlp(hidden_states)  # 输入通过多层感知机 MLP

        hidden_states = residual + hidden_states  # 添加残差连接

        outputs = (hidden_states,)  # 将输出打包为元组

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，添加到输出元组中

        return outputs  # 返回输出元组
        

class CLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPConfig  # 指定配置类
    base_model_prefix = "clip"  # 模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    # 初始化模型权重的函数，根据不同的模块类型设置不同的初始化策略
    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_factor
        
        # 如果模块是 CLIPTextEmbeddings 类型
        if isinstance(module, CLIPTextEmbeddings):
            # 初始化 token_embedding 和 position_embedding 的权重
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        
        # 如果模块是 CLIPVisionEmbeddings 类型
        elif isinstance(module, CLIPVisionEmbeddings):
            # 初始化 class_embedding, patch_embedding 和 position_embedding 的权重
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        
        # 如果模块是 CLIPAttention 类型
        elif isinstance(module, CLIPAttention):
            # 初始化注意力机制中的投影权重
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        
        # 如果模块是 CLIPMLP 类型
        elif isinstance(module, CLIPMLP):
            # 初始化多层感知机中的全连接层权重
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        
        # 如果模块是 CLIPModel 类型
        elif isinstance(module, CLIPModel):
            # 初始化 CLIPModel 中的文本和视觉投影权重
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
        
        # 如果模块是 CLIPVisionModelWithProjection 类型
        elif isinstance(module, CLIPVisionModelWithProjection):
            # 初始化视觉模型中的投影权重
            nn.init.normal_(
                module.visual_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        
        # 如果模块是 CLIPTextModelWithProjection 类型
        elif isinstance(module, CLIPTextModelWithProjection):
            # 初始化文本模型中的投影权重
            nn.init.normal_(
                module.text_projection.weight,
                std=self.config.hidden_size**-0.5 * self.config.initializer_factor,
            )
        
        # 如果模块是 nn.LayerNorm 类型
        if isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 的偏置和权重
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # 如果模块是 nn.Linear 类型并且有偏置项
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将线性层的偏置项初始化为零
            module.bias.data.zero_()
# CLIP_START_DOCSTRING 是一个包含模型介绍和配置参数说明的原始字符串文档
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

# CLIP_TEXT_INPUTS_DOCSTRING 是一个包含关于文本输入参数的原始字符串文档
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

# CLIP_VISION_INPUTS_DOCSTRING 是一个空的字符串文档，用于表示视觉输入的参数说明
CLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        # `pixel_values` 是一个 torch.FloatTensor，表示图像像素值，形状为 `(batch_size, num_channels, height, width)`
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.

        # `output_attentions` 是一个布尔值，可选参数，默认为 False
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        # `output_hidden_states` 是一个布尔值，可选参数，默认为 False
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        # `return_dict` 是一个布尔值，可选参数，默认为 False
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
    CLIP_INPUTS_DOCSTRING = r"""
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
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
                [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
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
    
    
    class CLIPEncoder(nn.Module):
        """
        Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
        [`CLIPEncoderLayer`].
    
        Args:
            config: CLIPConfig
        """
    
        def __init__(self, config: CLIPConfig):
            super().__init__()
            self.config = config
            # Initialize `num_hidden_layers` instances of CLIPEncoderLayer
            self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
            self.gradient_checkpointing = False
    
        def forward(
            self,
            inputs_embeds,
            attention_mask: Optional[torch.Tensor] = None,
            causal_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            # Forward pass through each layer of the encoder
            # `inputs_embeds` are the embedded input tokens
            # `attention_mask` masks padding tokens from attention calculation
            # `causal_attention_mask` masks future tokens for autoregressive tasks
            # `output_attentions` controls whether to output attentions tensors
            # `output_hidden_states` controls whether to output hidden states of layers
            # `return_dict` controls whether to return a ModelOutput or a tuple
            pass  # Placeholder for actual implementation
    
    class CLIPTextTransformer(nn.Module):
    # 初始化方法，接受一个配置对象 config: CLIPTextConfig
    def __init__(self, config: CLIPTextConfig):
        # 调用父类初始化方法
        super().__init__()
        # 将传入的配置对象保存到实例变量 self.config 中
        self.config = config
        # 从配置对象中获取隐藏层的维度作为嵌入的维度
        embed_dim = config.hidden_size
        # 创建 CLIPTextEmbeddings 对象并保存到实例变量 self.embeddings 中
        self.embeddings = CLIPTextEmbeddings(config)
        # 创建 CLIPEncoder 对象并保存到实例变量 self.encoder 中
        self.encoder = CLIPEncoder(config)
        # 创建一个 LayerNorm 层，并设定输入维度为 embed_dim，epsilon 值为 config.layer_norm_eps
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 为了计算 `pooled_output`，保存 EOS token 的 ID 到实例变量 self.eos_token_id 中
        self.eos_token_id = config.eos_token_id

    # 前向传播方法，使用装饰器将其文档字符串添加到模型的前向传播方法中
    # 使用 CLIP_TEXT_INPUTS_DOCSTRING 描述输入参数
    # 使用 replace_return_docstrings 装饰器，指定输出类型为 BaseModelOutputWithPooling，并使用 CLIPTextConfig 类描述配置
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器为类添加文档字符串，描述这是一个不带头或顶部投影的 CLIP 文本模型
@add_start_docstrings(
    """The text model from CLIP without any head or projection on top.""",
    CLIP_START_DOCSTRING,
)
class CLIPTextModel(CLIPPreTrainedModel):
    # 设置配置类为 CLIPTextConfig
    config_class = CLIPTextConfig

    # 定义不需要分割的模块列表
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        # 使用给定的配置初始化 CLIPTextTransformer 模型
        self.text_model = CLIPTextTransformer(config)
        # 调用初始化函数，初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入的方法，返回文本模型中的 token 嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    # 设置输入嵌入的方法，设置文本模型中的 token 嵌入为给定的值
    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    # 重写 forward 方法，使用装饰器为其添加文档字符串，描述输入参数和返回值的类型
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
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Returns:

        Examples:

        ```
        >>> from transformers import AutoTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        # 如果 return_dict 为 None，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 text_model 的 forward 方法，传递参数并返回结果
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # 初始化视觉嵌入、前层归一化、编码器和后层归一化
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 使用装饰器为 forward 方法添加文档字符串，描述输入参数和返回值的类型
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    # 定义一个方法 `forward`，用于执行模型的前向传播操作
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        执行前向传播操作，并返回模型输出的相关结果。

        Returns:
            根据 `return_dict` 参数的值返回不同的结果组合。
        """

        # 如果 `output_attentions` 参数为 None，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 `output_hidden_states` 参数为 None，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 `return_dict` 参数为 None，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 `pixel_values` 为 None，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值 `pixel_values` 输入到嵌入层 `embeddings` 中得到隐藏状态 `hidden_states`
        hidden_states = self.embeddings(pixel_values)
        # 在嵌入层输出的隐藏状态上应用预层归一化 `pre_layrnorm`
        hidden_states = self.pre_layrnorm(hidden_states)

        # 将处理后的隐藏状态 `hidden_states` 输入到编码器 `encoder` 中
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后一层隐藏状态 `last_hidden_state`
        last_hidden_state = encoder_outputs[0]
        # 从最后隐藏状态中提取池化输出 `pooled_output`
        pooled_output = last_hidden_state[:, 0, :]
        # 在池化输出上应用后层归一化 `post_layernorm`
        pooled_output = self.post_layernorm(pooled_output)

        # 如果 `return_dict` 为 False，则返回包含多个元组的结果
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果 `return_dict` 为 True，则返回一个包含多个属性的 `BaseModelOutputWithPooling` 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """The vision model from CLIP without any head or projection on top.""",
    CLIP_START_DOCSTRING,
)
# 定义 CLIPVisionModel 类，继承自 CLIPPreTrainedModel
class CLIPVisionModel(CLIPPreTrainedModel):
    # 使用 CLIPVisionConfig 作为配置类
    config_class = CLIPVisionConfig
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 不需要拆分的模块列表
    _no_split_modules = ["CLIPEncoderLayer"]

    # 初始化函数，接受一个 CLIPVisionConfig 类型的参数 config
    def __init__(self, config: CLIPVisionConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建 CLIPVisionTransformer 对象，并赋值给 self.vision_model
        self.vision_model = CLIPVisionTransformer(config)
        # 调用自定义的后初始化函数
        self.post_init()

    # 返回模型的输入嵌入层
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 前向传播函数，接受多个可选参数并返回 Union[Tuple, BaseModelOutputWithPooling] 类型的值
    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Returns:
        
        Examples:
        
        ```
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.vision_model 的前向传播函数，并返回结果
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# 定义 CLIPModel 类，继承自 CLIPPreTrainedModel，带有 CLIP_START_DOCSTRING 的说明文档
@add_start_docstrings(CLIP_START_DOCSTRING)
class CLIPModel(CLIPPreTrainedModel):
    # 使用 CLIPConfig 作为配置类
    config_class = CLIPConfig
    # 不需要拆分的模块列表
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]
    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        # 检查配置是否符合预期类型，否则引发值错误异常
        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查配置是否符合预期类型，否则引发值错误异常
        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 将文本和视觉配置提取到局部变量中
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度和文本嵌入维度，从配置中提取
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本模型和视觉模型
        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        # 创建用于视觉和文本投影的线性层，无偏置
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        # 创建并初始化logit_scale作为模型参数
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

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

        ```
        >>> from transformers import AutoTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        # 检查是否提供了输出注意力信息，如果没有则使用模型的配置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否提供了输出隐藏状态信息，如果没有则使用模型的配置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否提供了返回字典的信息，如果没有则使用模型的配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用文本模型的前向传播，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从文本输出中获取池化后的输出（通常是第二个元素）
        pooled_output = text_outputs[1]
        # 将池化后的输出应用于文本投影层，得到文本特征
        text_features = self.text_projection(pooled_output)

        # 返回文本特征作为函数的输出
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

        ```
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
        # 设置返回类型为 torch.FloatTensor，代表图像特征向量的形状为 (batch_size, output_dim)
        # 这些特征向量是通过将池化输出应用到 CLIPVisionModel 的投影层上获得的
        # 返回图像特征向量
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定，则使用 CLIP 模型配置中的 output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定，则使用 CLIP 模型配置中的 output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果未指定，则使用 CLIP 模型配置中的 use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 使用 CLIP 模型的视觉部分进行处理，传入像素值、注意力输出、隐藏状态输出和返回字典选项

        pooled_output = vision_outputs[1]  # 从视觉输出中获取池化后的输出
        image_features = self.visual_projection(pooled_output)
        # 将池化输出应用于视觉投影层，生成图像特征向量

        return image_features
"""
CLIP Vision Model with a projection layer on top (a linear layer on top of the pooled output).
"""
@add_start_docstrings(
    """
    CLIP Text Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLIP_START_DOCSTRING,
)
class CLIPTextModelWithProjection(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)

        # Initialize the text model component using CLIPTextTransformer
        self.text_model = CLIPTextTransformer(config)

        # Linear projection layer to transform hidden_size to projection_dim
        self.text_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # Return the token embeddings from CLIPTextTransformer
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        # Set new token embeddings for CLIPTextTransformer
        self.text_model.embeddings.token_embedding = value

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

        ```
        >>> from transformers import AutoTokenizer, CLIPTextModelWithProjection

        >>> model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input through the text model to get text_outputs
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract pooled_output from text_outputs
        pooled_output = text_outputs[1]

        # Project pooled_output using text_projection linear layer
        text_embeds = self.text_projection(pooled_output)

        if not return_dict:
            # If return_dict is False, return tuple of outputs
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        # If return_dict is True, return CLIPTextModelOutput with specified attributes
        return CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )
    """
    将字符串 CLIP_START_DOCSTRING 插入到三引号字符串中
    CLIP_START_DOCSTRING 通常是一个文档字符串的起始标记
    """
    CLIP_START_DOCSTRING,
# 定义一个继承自 CLIPPreTrainedModel 的类，用于视觉模型和投影
class CLIPVisionModelWithProjection(CLIPPreTrainedModel):
    # 设置配置类为 CLIPVisionConfig
    config_class = CLIPVisionConfig
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法，接受一个 CLIPVisionConfig 类型的配置对象
    def __init__(self, config: CLIPVisionConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 CLIPVisionTransformer 类的实例，作为视觉模型
        self.vision_model = CLIPVisionTransformer(config)

        # 创建一个线性层，用于视觉投影，输入维度为 config.hidden_size，输出维度为 config.projection_dim，无偏置
        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # 执行后续的初始化权重和处理步骤
        self.post_init()

    # 获取输入嵌入的方法，返回视觉模型中的 patch_embedding 模块
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 前向传播方法，接受像素值 pixel_values 等多个可选参数，返回 Union[Tuple, CLIPVisionModelOutput] 类型
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

        ```
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
        ```
        """
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict 参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型的前向传播方法，获取视觉输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取池化后的输出，命名为 pooled_output
        pooled_output = vision_outputs[1]  # pooled_output

        # 对 pooled_output 进行视觉投影，得到图像嵌入 image_embeds
        image_embeds = self.visual_projection(pooled_output)

        # 如果 return_dict 为 False，则返回元组形式的输出
        if not return_dict:
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        # 如果 return_dict 为 True，则返回 CLIPVisionModelOutput 类型的结构化输出
        return CLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )


# 添加关于图像分类的描述性注释，继承自 CLIPPreTrainedModel 的类
@add_start_docstrings(
    """
    CLIP vision encoder with an image classification head on top (a linear layer on top of the pooled final hidden states of
    the patch tokens) e.g. for ImageNet.
    """,
    CLIP_START_DOCSTRING,
)
class CLIPForImageClassification(CLIPPreTrainedModel):
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 初始化方法，接受一个 CLIPConfig 类型的配置参数
    def __init__(self, config: CLIPConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置实例变量 num_labels，用于指定分类任务的类别数
        self.num_labels = config.num_labels
        
        # 根据配置中的视觉模型配置信息创建视觉模型，使用 CLIPVisionTransformer 类
        self.vision_model = CLIPVisionTransformer(config.vision_config)

        # 分类器头部部分，根据 num_labels 的值决定使用全连接层还是恒等映射
        self.classifier = (
            nn.Linear(config.vision_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 执行后续的初始化步骤和最终处理
        self.post_init()

    # 前向传播方法，接受像素值、标签以及其他配置参数，返回模型输出结果
    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数描述：
        # pixel_values: 图像的像素值张量，可选
        # labels: 标签张量，可选
        # output_attentions: 是否输出注意力权重张量，可选
        # output_hidden_states: 是否输出隐藏状态张量，可选
        # return_dict: 是否返回字典类型的结果，可选
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否输出注意力权重，默认与模型配置一致
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态，默认与模型配置一致
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否使用返回字典，默认与模型配置一致
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入数据传递给视觉模型，获取输出
        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出，通常是模型输出的第一个元素
        sequence_output = outputs[0]

        # 对补丁令牌进行平均池化
        sequence_output = torch.mean(sequence_output[:, 1:, :], dim=1)
        
        # 应用分类器，生成分类器的 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行处理
            labels = labels.to(logits.device)
            # 根据问题类型设置模型配置
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不要求返回字典形式的输出，则返回元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 ImageClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```