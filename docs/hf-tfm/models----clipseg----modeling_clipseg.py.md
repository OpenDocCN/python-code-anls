# `.\models\clipseg\modeling_clipseg.py`

```
# 设置文件编码为UTF-8

# 版权声明及许可证信息，该文件受 Apache 许可证版本 2.0 保护
# 除非符合许可证的规定，否则不得使用本文件
# 可以通过以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0

# 导入必要的库和模块
""" PyTorch CLIPSeg 模型."""

# 复制对象
import copy
# 数学计算函数库
import math
# 数据类装饰器
from dataclasses import dataclass
# 任意类型
from typing import Any, Optional, Tuple, Union

# 导入PyTorch库
import torch
# PyTorch模块
import torch.utils.checkpoint
# 导入神经网络模块
from torch import nn

# 导入自定义的模块
# 用于处理激活函数
from ...activations import ACT2FN
# 模型的注意力掩码工具函数
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
# 模型输出类
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
# 预训练模型基类
from ...modeling_utils import PreTrainedModel
# 工具函数
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 模型检查点，用于文档
_CHECKPOINT_FOR_DOC = "CIDAS/clipseg-rd64-refined"

# 预训练模型存档列表
CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "CIDAS/clipseg-rd64-refined",
    # 可在 https://huggingface.co/models?filter=clipseg 查看所有 CLIPSeg 模型
]

# 对比损失函数，源自 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # 计算交叉熵损失
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# 复制自 transformers.models.clip.modeling_clip.clip_loss，用 CLIPSeg 替换 CLIP
def clipseg_loss(similarity: torch.Tensor) -> torch.Tensor:
    # 计算标题损失
    caption_loss = contrastive_loss(similarity)
    # 计算图像损失
    image_loss = contrastive_loss(similarity.t())
    # 返回损失的平均值
    return (caption_loss + image_loss) / 2.0

@dataclass
# 复制自 transformers.models.clip.modeling_clip.CLIPOutput，用 CLIPSeg 替换 CLIP
class CLIPSegOutput(ModelOutput):
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
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegVisionModel`].
    """

    # Optional attribute: Contrastive loss for image-text similarity.
    loss: Optional[torch.FloatTensor] = None
    # Optional attribute: Scores of image-text similarity.
    logits_per_image: torch.FloatTensor = None
    # Optional attribute: Scores of text-image similarity.
    logits_per_text: torch.FloatTensor = None
    # Optional attribute: Text embeddings after projection from CLIPSegTextModel output.
    text_embeds: torch.FloatTensor = None
    # Optional attribute: Image embeddings after projection from CLIPSegVisionModel output.
    image_embeds: torch.FloatTensor = None
    # Optional attribute: Output of CLIPSegTextModel including pooling.
    text_model_output: BaseModelOutputWithPooling = None
    # Optional attribute: Output of CLIPSegVisionModel including pooling.
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        # Convert attributes to a tuple, handling special cases for complex objects.
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
@dataclass
class CLIPSegDecoderOutput(ModelOutput):
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            分类得分，用于每个像素的分类。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            包含多个元素的元组，每个元素是 `torch.FloatTensor` 类型，表示每个层的隐藏状态输出，如果模型有嵌入层则还包含嵌入层的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            包含多个元素的元组，每个元素是 `torch.FloatTensor` 类型，表示每个层的注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力权重经过 softmax 后的值，用于计算自注意力头中的加权平均值。
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class CLIPSegImageSegmentationOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, 当 `return_loss` 为 `True` 时返回):
            图像与文本相似性的对比损失。
        ...
        vision_model_output (`BaseModelOutputWithPooling`):
            [`CLIPSegVisionModel`] 的输出。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    conditional_embeddings: torch.FloatTensor = None
    pooled_output: torch.FloatTensor = None
    vision_model_output: BaseModelOutputWithPooling = None
    decoder_output: CLIPSegDecoderOutput = None

    def to_tuple(self) -> Tuple[Any]:
        """
        将对象转换为元组形式，包含所有属性值。特殊处理 `vision_model_output` 和 `decoder_output` 属性，
        将它们转换为元组形式。
        """
        return tuple(
            self[k] if k not in ["vision_model_output", "decoder_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CLIPSegVisionEmbeddings(nn.Module):
    """
    从 `transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.__init__` 复制而来，将 `CLIP` 替换为 `CLIPSeg`。
    """
    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # 设置嵌入向量的维度为隐藏大小
        self.image_size = config.image_size  # 图像尺寸从配置中获取
        self.patch_size = config.patch_size  # 补丁大小从配置中获取

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))  # 类别嵌入，使用随机张量初始化

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,  # 输入通道数从配置中获取
            out_channels=self.embed_dim,  # 输出通道数设置为嵌入向量维度
            kernel_size=self.patch_size,  # 卷积核大小设置为补丁大小
            stride=self.patch_size,  # 卷积步长设置为补丁大小，实现非重叠补丁提取
            bias=False,  # 不使用偏置项
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2  # 计算图像中补丁的数量
        self.num_positions = self.num_patches + 1  # 位置嵌入的数量，比补丁数量多一个用于类别嵌入
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)  # 位置嵌入，根据数量和维度创建
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)  # 注册位置 ID 的缓冲区

    def interpolate_position_embeddings(self, new_size):
        if len(new_size) != 2:
            raise ValueError("new_size should consist of 2 values")  # 抛出异常，如果 new_size 长度不为 2

        num_patches_one_direction = int(self.num_patches**0.5)  # 在一个方向上的补丁数的平方根
        # 在二维中插值位置嵌入
        a = self.position_embedding.weight[1:].T.view(
            1, self.config.hidden_size, num_patches_one_direction, num_patches_one_direction
        )
        b = (
            nn.functional.interpolate(a, new_size, mode="bicubic", align_corners=False)  # 使用双三次插值方法插值
            .squeeze(0)
            .view(self.config.hidden_size, new_size[0] * new_size[1])  # 调整形状以适应新大小
            .T
        )
        result = torch.cat([self.position_embedding.weight[:1], b])  # 将插值结果与第一个位置嵌入拼接

        return result

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]  # 批量大小
        patch_embeds = self.patch_embedding(pixel_values)  # 提取补丁嵌入
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # 展平和转置以匹配形状

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)  # 扩展类别嵌入以匹配批量大小
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)  # 拼接类别嵌入和补丁嵌入

        if embeddings.shape[1] != self.num_positions:
            new_shape = int(math.sqrt(embeddings.shape[1] - 1))  # 计算新的形状大小
            embeddings = embeddings + self.interpolate_position_embeddings((new_shape, new_shape))  # 插值位置嵌入并加到嵌入向量上
            embeddings = embeddings.to(embeddings.dtype)  # 将嵌入向量转换为指定的数据类型
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)  # 添加位置嵌入到嵌入向量中

        return embeddings
# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->CLIPSeg
class CLIPSegTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPSegTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        # 定义词嵌入层，用于将输入的token转换成对应的向量表示
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 定义位置嵌入层，用于表示输入token的位置信息
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 创建并注册一个持久化的buffer，用于存储位置ID，以确保在序列化时被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 获取序列的长度，如果没有提供input_ids，则使用inputs_embeds的长度
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        # 如果未提供位置ID，则使用预先注册的位置ID
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供嵌入向量，根据input_ids生成token嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 根据位置ID生成位置嵌入向量
        position_embeddings = self.position_embedding(position_ids)
        # 将token嵌入向量和位置嵌入向量相加得到最终的嵌入表示
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPAttention with CLIP->CLIPSeg
class CLIPSegAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 缩放因子，用于缩放点积注意力
        self.scale = self.head_dim**-0.5
        # 注意力层的dropout比率
        self.dropout = config.attention_dropout

        # 线性变换函数，用于计算Q、K、V和输出
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将输入的tensor进行reshape操作，以便进行多头注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 略
    # 定义初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置对象保存在实例变量中
        self.config = config
        # 根据配置对象中的隐藏激活函数名称从预定义的字典 ACT2FN 中获取对应的激活函数，并保存在实例变量中
        self.activation_fn = ACT2FN[config.hidden_act]
        # 创建一个全连接层，输入大小为配置对象中的隐藏大小，输出大小为配置对象中的中间大小，并保存在实例变量中
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建另一个全连接层，输入大小为配置对象中的中间大小，输出大小为配置对象中的隐藏大小，并保存在实例变量中
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    # 定义前向传播方法，接受一个 torch.Tensor 类型的隐藏状态输入，并返回一个 torch.Tensor 类型的输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态传入第一个全连接层进行线性变换，并更新隐藏状态变量
        hidden_states = self.fc1(hidden_states)
        # 将经过线性变换后的隐藏状态通过预先定义的激活函数进行非线性变换，并更新隐藏状态变量
        hidden_states = self.activation_fn(hidden_states)
        # 将经过激活函数变换后的隐藏状态再次传入第二个全连接层进行线性变换，并更新隐藏状态变量
        hidden_states = self.fc2(hidden_states)
        # 返回经过两个全连接层和激活函数处理后的最终隐藏状态作为输出
        return hidden_states
# 从transformers.models.clip.modeling_clip.CLIPEncoderLayer复制而来，修改为CLIPSeg
class CLIPSegEncoderLayer(nn.Module):
    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # 从配置中获取隐藏层大小
        self.self_attn = CLIPSegAttention(config)  # 初始化自注意力机制
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 初始化第一个层归一化
        self.mlp = CLIPSegMLP(config)  # 初始化多层感知机
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)  # 初始化第二个层归一化

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): 注意力遮罩，形状为 `(batch, 1, tgt_len, src_len)`，其中填充元素用非常大的负值表示
            causal_attention_mask (`torch.FloatTensor`): 因果注意力遮罩，形状为 `(batch, 1, tgt_len, src_len)`，用于生成因果关系
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions`。
        """
        residual = hidden_states  # 保存残差连接

        hidden_states = self.layer_norm1(hidden_states)  # 第一个层归一化
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )  # 执行自注意力机制

        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 保存残差连接
        hidden_states = self.layer_norm2(hidden_states)  # 第二个层归一化
        hidden_states = self.mlp(hidden_states)  # 执行多层感知机
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则添加到输出中

        return outputs
    # 初始化模型的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # 从配置中获取初始化因子

        # 如果 module 是 CLIPSegTextEmbeddings 类型的实例
        if isinstance(module, CLIPSegTextEmbeddings):
            # 对 token_embedding 和 position_embedding 的权重进行正态分布初始化
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)

        # 如果 module 是 CLIPSegVisionEmbeddings 类型的实例
        elif isinstance(module, CLIPSegVisionEmbeddings):
            # 对 class_embedding、patch_embedding 和 position_embedding 的权重进行正态分布初始化
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)

        # 如果 module 是 CLIPSegAttention 类型的实例
        elif isinstance(module, CLIPSegAttention):
            # 根据模型参数和初始化因子计算各个权重的标准差并进行正态分布初始化
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)

        # 如果 module 是 CLIPSegMLP 类型的实例
        elif isinstance(module, CLIPSegMLP):
            # 根据模型参数和初始化因子计算各个权重的标准差并进行正态分布初始化
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)

        # 如果 module 是 CLIPSegModel 类型的实例
        elif isinstance(module, CLIPSegModel):
            # 对 text_projection 和 visual_projection 的权重进行正态分布初始化
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )

        # 如果 module 是 nn.LayerNorm 类型的实例
        if isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 层的偏置项初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 如果 module 是 nn.Linear 类型的实例且具有偏置项
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将 Linear 层的偏置项初始化为零
            module.bias.data.zero_()
# 定义一个多行字符串，包含关于 CLIPSeg 模型的文档说明，描述了它是一个 PyTorch 的 nn.Module 子类，如何使用以及参数说明
CLIPSEG_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CLIPSegConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义文本输入的文档说明，解释了输入参数包括 input_ids、attention_mask、position_ids、output_attentions、output_hidden_states 和 return_dict
CLIPSEG_TEXT_INPUTS_DOCSTRING = r"""
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

# 定义视觉输入的文档说明，该段落暂未提供具体内容，留空
CLIPSEG_VISION_INPUTS_DOCSTRING = r"""
    # 定义函数签名和参数说明
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下将忽略填充。可以使用 [`AutoImageProcessor`] 获取像素值。有关详细信息，请参见 [`CLIPImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

CLIPSEG_INPUTS_DOCSTRING = r"""
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


# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->CLIPSeg
class CLIPSegEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPSegEncoderLayer`].

    Args:
        config: CLIPSegConfig
    """

    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        # 初始化函数，接收一个 CLIPSegConfig 对象作为参数，用于配置当前编码器
        self.config = config
        # 创建一个包含多个 CLIPSegEncoderLayer 的层列表，数量由 config.num_hidden_layers 决定
        self.layers = nn.ModuleList([CLIPSegEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标记，默认为 False，表示不启用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义 CLIPSegTextTransformer 类，继承自 nn.Module
class CLIPSegTextTransformer(nn.Module):
    # 从 transformers.models.clip.modeling_clip.CLIPTextTransformer.__init__ 复制而来，将 CLIP 替换为 CLIPSeg
    def __init__(self, config: CLIPSegTextConfig):
        super().__init__()
        # 将传入的配置对象保存到实例变量中
        self.config = config
        # 从配置中获取隐藏层大小作为嵌入维度
        embed_dim = config.hidden_size
        # 初始化嵌入层
        self.embeddings = CLIPSegTextEmbeddings(config)
        # 初始化编码器
        self.encoder = CLIPSegEncoder(config)
        # 初始化最终的 LayerNorm 层，用于归一化输出
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 用于计算 `pooled_output` 的属性
        self.eos_token_id = config.eos_token_id

    # 将 CLIPSEG_TEXT_INPUTS_DOCSTRING 添加到模型前向方法的文档字符串中
    # 使用 replace_return_docstrings 将返回文档字符串替换为 BaseModelOutputWithPooling 的输出类型和 CLIPSegTextConfig 配置类
    # 从 transformers.models.clip.modeling_clip.CLIPTextTransformer.forward 复制而来，将 clip 替换为 clipseg，CLIP 替换为 CLIPSeg
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):



# 定义 CLIPSegTextModel 类，继承自 CLIPSegPreTrainedModel
class CLIPSegTextModel(CLIPSegPreTrainedModel):
    # 指定配置类为 CLIPSegTextConfig
    config_class = CLIPSegTextConfig

    # 指定不分割的模块名称列表
    _no_split_modules = ["CLIPSegTextEmbeddings", "CLIPSegEncoderLayer"]

    # 从 CLIPSegPreTrainedModel.__init__ 继承，初始化函数
    def __init__(self, config: CLIPSegTextConfig):
        super().__init__(config)
        # 初始化文本模型，使用 CLIPSegTextTransformer
        self.text_model = CLIPSegTextTransformer(config)
        # 调用 post_init 方法完成权重初始化和最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    # 将 CLIPSEG_TEXT_INPUTS_DOCSTRING 添加到模型前向方法的文档字符串中
    # 使用 replace_return_docstrings 将返回文档字符串替换为 BaseModelOutputWithPooling 的输出类型和 CLIPSegTextConfig 配置类
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPSegTextModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegTextModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        调用 self 对象的 text_model 方法，传入各种参数来进行文本模型的推理和处理
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
# 定义一个名为 CLIPSegVisionTransformer 的类，继承自 nn.Module
class CLIPSegVisionTransformer(nn.Module):
    # 从 transformers.models.clip.modeling_clip.CLIPVisionTransformer.__init__ 复制并修改为 CLIPSeg->CLIPSegVision
    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__()
        # 将传入的配置参数保存到实例变量中
        self.config = config
        # 从配置中获取嵌入维度
        embed_dim = config.hidden_size

        # 初始化嵌入层对象，用于处理视觉输入数据
        self.embeddings = CLIPSegVisionEmbeddings(config)
        # 初始化第一个 LayerNorm 层，用于归一化嵌入层的输出
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 初始化编码器对象，处理嵌入层的输出
        self.encoder = CLIPSegEncoder(config)
        # 初始化第二个 LayerNorm 层，用于归一化编码器的输出
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 从 transformers.models.clip.modeling_clip.CLIPVisionTransformer.forward 复制的文档字符串
    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # 如果未提供像素值，则抛出值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值通过嵌入层处理，得到隐藏状态
        hidden_states = self.embeddings(pixel_values)
        # 对隐藏状态进行预 LayerNorm 处理
        hidden_states = self.pre_layrnorm(hidden_states)

        # 将预处理后的隐藏状态传入编码器，得到编码器的输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 提取池化输出，即最后一个隐藏状态的第一个位置的向量
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行后 LayerNorm 处理
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不使用返回字典，则返回一个元组，包含最后隐藏状态、池化输出和其他编码器输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果使用返回字典，则返回一个 BaseModelOutputWithPooling 对象，包含最后隐藏状态、池化输出、隐藏状态和注意力
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# 定义一个名为 CLIPSegVisionModel 的类，继承自 CLIPSegPreTrainedModel
class CLIPSegVisionModel(CLIPSegPreTrainedModel):
    # 指定配置类为 CLIPSegVisionConfig
    config_class = CLIPSegVisionConfig
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法，接受配置对象作为参数
    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__(config)
        # 初始化 CLIPSegVisionTransformer 模型，用于处理视觉任务
        self.vision_model = CLIPSegVisionTransformer(config)
        # 执行初始化权重和应用最终处理的方法
        self.post_init()

    # 返回嵌入层对象的方法
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding
    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        此方法定义了模型的前向传播逻辑，用于推理过程。

        Args:
            pixel_values (Optional[torch.FloatTensor], optional): 输入图像的像素值张量。默认为None。
            output_attentions (Optional[bool], optional): 是否输出注意力权重。默认为None。
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态。默认为None。
            return_dict (Optional[bool], optional): 是否返回字典形式的输出。默认为None。

        Returns:
            Union[Tuple, BaseModelOutputWithPooling]: 根据return_dict决定返回类型，可能是元组或BaseModelOutputWithPooling对象。

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegVisionModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegVisionModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```
        """
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
# 使用装饰器为类添加文档字符串，以提供类的基本信息
@add_start_docstrings(CLIPSEG_START_DOCSTRING)
class CLIPSegModel(CLIPSegPreTrainedModel):
    # 指定配置类为CLIPSegConfig
    config_class = CLIPSegConfig

    def __init__(self, config: CLIPSegConfig):
        # 调用父类构造函数初始化模型
        super().__init__(config)

        # 检查文本配置是否为CLIPSegTextConfig类型，否则引发值错误异常
        if not isinstance(config.text_config, CLIPSegTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPSegTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查视觉配置是否为CLIPSegVisionConfig类型，否则引发值错误异常
        if not isinstance(config.vision_config, CLIPSegVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPSegVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 获取文本和视觉配置
        text_config = config.text_config
        vision_config = config.vision_config

        # 初始化模型的维度信息
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本和视觉模型
        self.text_model = CLIPSegTextTransformer(text_config)
        self.vision_model = CLIPSegVisionTransformer(vision_config)

        # 创建视觉投影层和文本投影层
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        
        # 初始化对数尺度参数
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 执行初始化权重和最终处理
        self.post_init()

    # 使用装饰器为方法添加文档字符串，以提供方法的输入参数和功能描述
    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
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
            applying the projection layer to the pooled output of [`CLIPSegTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPSegModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
        # 如果输出注意力未指定，则使用 self.config.output_attentions；否则使用指定的输出注意力
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用 self.config.output_hidden_states；否则使用指定的输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典未指定，则使用 self.config.use_return_dict；否则使用指定的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用文本模型进行前向传播，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从文本输出中取出汇总输出
        pooled_output = text_outputs[1]
        # 将汇总输出应用于文本投影层，生成文本特征
        text_features = self.text_projection(pooled_output)

        # 返回生成的文本特征
        return text_features

    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
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
            applying the projection layer to the pooled output of [`CLIPSegVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPSegModel

        >>> processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use CLIPSEG model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass the input arguments to the vision model and retrieve its outputs
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the pooled output from the vision model's outputs
        pooled_output = vision_outputs[1]  # pooled_output
        # Apply visual projection layer to the pooled output to obtain image features
        image_features = self.visual_projection(pooled_output)

        # Return the computed image features
        return image_features
# 定义一个 CLIPSeg 解码器层，与 CLIPSegEncoderLayer 相同，不同之处在于在 self-attention/MLP 之后应用归一化，而不是之前。
class CLIPSegDecoderLayer(nn.Module):
    """
    CLIPSeg decoder layer, which is identical to `CLIPSegEncoderLayer`, except that normalization is applied after
    self-attention/MLP, rather than before.
    """

    # 从 transformers.models.clip.modeling_clip.CLIPEncoderLayer.__init__ 复制而来，仅将 CLIP 改为 CLIPSeg
    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        # 设定嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 初始化 self-attention 层
        self.self_attn = CLIPSegAttention(config)
        # 第一层归一化
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 初始化 MLP 层
        self.mlp = CLIPSegMLP(config)
        # 第二层归一化
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
            causal_attention_mask (`torch.FloatTensor`): mask applied to causal attention
                `(batch, 1, tgt_len, src_len)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        # 保存残差连接
        residual = hidden_states

        # 应用 self-attention 模块，得到新的 hidden_states 和 attention weights
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        # 添加残差连接并应用第一层归一化
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        # 再次保存残差连接
        residual = hidden_states

        # 应用 MLP 层
        hidden_states = self.mlp(hidden_states)

        # 添加残差连接并应用第二层归一化
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        # 准备输出
        outputs = (hidden_states,)

        # 如果需要输出 attention weights，则将它们加入输出
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    # 初始化方法，接收一个配置对象作为参数
    def __init__(self, config: CLIPSegConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置条件层编号
        self.conditional_layer = config.conditional_layer

        # FILM（Feature-wise Linear Modulation）网络中的乘法线性层，将投影维度映射到减少维度
        self.film_mul = nn.Linear(config.projection_dim, config.reduce_dim)
        # FILM网络中的加法线性层，同样映射投影维度到减少维度
        self.film_add = nn.Linear(config.projection_dim, config.reduce_dim)

        # 如果配置指定使用复杂的转置卷积
        if config.use_complex_transposed_convolution:
            # 计算转置卷积核的大小
            transposed_kernels = (config.vision_config.patch_size // 4, config.vision_config.patch_size // 4)

            # 创建转置卷积层的序列模型
            self.transposed_convolution = nn.Sequential(
                # 普通卷积层，用于降低特征维度
                nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                # 第一个转置卷积层，使用指定的核大小和步幅
                nn.ConvTranspose2d(
                    config.reduce_dim,
                    config.reduce_dim // 2,
                    kernel_size=transposed_kernels[0],
                    stride=transposed_kernels[0],
                ),
                nn.ReLU(),
                # 第二个转置卷积层，使用不同的核大小和步幅
                nn.ConvTranspose2d(
                    config.reduce_dim // 2, 1, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]
                ),
            )
        else:
            # 创建简单的转置卷积层，使用指定的核大小和步幅
            self.transposed_convolution = nn.ConvTranspose2d(
                config.reduce_dim, 1, config.vision_config.patch_size, stride=config.vision_config.patch_size
            )

        # 提取层的深度，即要减少特征维度的层数
        depth = len(config.extract_layers)
        # 创建多个线性层，用于将视觉特征的隐藏大小映射到减少维度
        self.reduces = nn.ModuleList(
            [nn.Linear(config.vision_config.hidden_size, config.reduce_dim) for _ in range(depth)]
        )

        # 复制视觉配置，用于解码器的配置
        decoder_config = copy.deepcopy(config.vision_config)
        # 设置解码器的隐藏大小为减少维度后的大小
        decoder_config.hidden_size = config.reduce_dim
        # 设置解码器的注意力头数和中间层大小
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        # 设置解码器的激活函数为ReLU
        decoder_config.hidden_act = "relu"
        # 创建多个CLIPSeg解码层，与提取层数量相同
        self.layers = nn.ModuleList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers))])

    # 前向传播方法，接收隐藏状态、条件嵌入、输出注意力和隐藏状态标志等参数
    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        ):
        all_hidden_states = () if output_hidden_states else None  # 初始化存储所有隐藏状态的元组，如果不输出隐藏状态则为None
        all_attentions = () if output_attentions else None  # 初始化存储所有注意力权重的元组，如果不输出注意力权重则为None

        activations = hidden_states[::-1]  # 将隐藏状态列表倒序排列

        output = None  # 初始化输出变量为None
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
            if output is not None:
                output = reduce(activation) + output  # 如果输出不为None，应用reduce函数并累加到output中
            else:
                output = reduce(activation)  # 如果输出为None，直接应用reduce函数

            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(
                    conditional_embeddings
                )  # 如果当前层是条件层，则应用条件嵌入乘法和加法操作到output上

                output = output.permute(1, 0, 2)  # 调整output的维度顺序

            layer_outputs = layer(
                output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions
            )  # 应用当前层的前向传播函数，传入output和相应的注意力掩码参数

            output = layer_outputs[0]  # 更新output为当前层的输出结果

            if output_hidden_states:
                all_hidden_states += (output,)  # 如果需要输出隐藏状态，将当前层输出的隐藏状态添加到all_hidden_states中

            if output_attentions:
                all_attentions += (layer_outputs[1],)  # 如果需要输出注意力权重，将当前层输出的注意力权重添加到all_attentions中

        output = output[:, 1:, :].permute(0, 2, 1)  # 移除CLS标记并重塑维度为[batch_size, reduce_dim, seq_len]

        size = int(math.sqrt(output.shape[2]))  # 计算输出的第三维度的平方根作为size

        batch_size = conditional_embeddings.shape[0]  # 获取条件嵌入的批量大小
        output = output.view(batch_size, output.shape[1], size, size)  # 调整output的维度形状

        logits = self.transposed_convolution(output).squeeze(1)  # 应用转置卷积层到output上并压缩第一维度

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_attentions] if v is not None)  # 如果不返回字典形式的结果，返回包含非空元素的元组

        return CLIPSegDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )  # 否则，返回CLIPSegDecoderOutput对象，包含logits、hidden_states和attentions字段
@add_start_docstrings(
    """
    CLIPSeg model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.
    """,
    CLIPSEG_START_DOCSTRING,
)
class CLIPSegForImageSegmentation(CLIPSegPreTrainedModel):
    config_class = CLIPSegConfig

    def __init__(self, config: CLIPSegConfig):
        super().__init__(config)

        self.config = config

        # Initialize CLIPSegModel with provided configuration
        self.clip = CLIPSegModel(config)
        
        # Store the list of layers to extract features from
        self.extract_layers = config.extract_layers

        # Initialize CLIPSegDecoder with provided configuration
        self.decoder = CLIPSegDecoder(config)

        # Initialize model weights and apply final processing
        self.post_init()

    def get_conditional_embeddings(
        self,
        batch_size: int = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        conditional_pixel_values: Optional[torch.Tensor] = None,
    ):
        if input_ids is not None:
            # Compute conditional embeddings from text inputs
            if len(input_ids) != batch_size:
                raise ValueError("Make sure to pass as many prompt texts as there are query images")
            with torch.no_grad():
                # Retrieve text features from CLIP model
                conditional_embeddings = self.clip.get_text_features(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids
                )
        elif conditional_pixel_values is not None:
            # Compute conditional embeddings from image inputs
            if len(conditional_pixel_values) != batch_size:
                raise ValueError("Make sure to pass as many prompt images as there are query images")
            with torch.no_grad():
                # Retrieve image features from CLIP model
                conditional_embeddings = self.clip.get_image_features(conditional_pixel_values)
        else:
            raise ValueError(
                "Invalid conditional, should be either provided as `input_ids` or `conditional_pixel_values`"
            )

        return conditional_embeddings

    @add_start_docstrings_to_model_forward(CLIPSEG_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPSegImageSegmentationOutput, config_class=CLIPSegTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        conditional_pixel_values: Optional[torch.FloatTensor] = None,
        conditional_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Method signature conforms to CLIPSEG_INPUTS_DOCSTRING specifications
        # and replaces return type description with CLIPSegImageSegmentationOutput
        pass  # Placeholder for actual implementation, not provided here
```