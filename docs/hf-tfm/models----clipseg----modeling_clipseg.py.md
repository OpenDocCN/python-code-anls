# `.\transformers\models\clipseg\modeling_clipseg.py`

```
# 设置文件编码为 UTF-8
# 版权声明及许可信息
import copy  # 导入 copy 模块
import math  # 导入 math 模块
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Any, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import nn  # 从 PyTorch 导入 nn 模块

from ...activations import ACT2FN  # 导入激活函数相关的模块
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask  # 导入自定义的注意力掩码相关函数
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling  # 导入模型输出相关的模块
from ...modeling_utils import PreTrainedModel  # 导入预训练模型相关的模块
from ...utils import (  # 从工具函数模块导入相关函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_clipseg import CLIPSegConfig, CLIPSegTextConfig, CLIPSegVisionConfig  # 从当前目录下的配置文件导入配置类

logger = logging.get_logger(__name__)  # 获取 logger 对象


_CHECKPOINT_FOR_DOC = "CIDAS/clipseg-rd64-refined"  # 用于文档的检查点路径

CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型的存档列表
    "CIDAS/clipseg-rd64-refined",
    # 查看所有 CLIPSeg 模型 https://huggingface.co/models?filter=clipseg
]


# 对比损失函数，改编自 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# 从 transformers.models.clip.modeling_clip.clip_loss 中复制的 clipseg_loss 函数
def clipseg_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)  # 计算标题损失
    image_loss = contrastive_loss(similarity.t())  # 计算图像损失
    return (caption_loss + image_loss) / 2.0  # 返回标题损失和图像损失的平均值


@dataclass  # 使用 dataclass 装饰器定义数据类
# 从 transformers.models.clip.modeling_clip.CLIPOutput 复制的 CLIPSegOutput 类，替换 CLIP 为 CLIPSeg
class CLIPSegOutput(ModelOutput):
    """
```  
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.  # 对图像-文本相似度的对比损失
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.  # 图像嵌入和文本嵌入之间的缩放点积分数。表示图像-文本相似度得分。
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.  # 文本嵌入和图像嵌入之间的缩放点积分数。表示文本-图像相似度得分。
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegTextModel`].  # 通过将投影层应用于 [`CLIPSegTextModel`] 的汇总输出而获得的文本嵌入。
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPSegVisionModel`].  # 通过将投影层应用于 [`CLIPSegVisionModel`] 的汇总输出而获得的图像嵌入。
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegTextModel`].  # [`CLIPSegTextModel`] 的输出。
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegVisionModel`].  # [`CLIPSegVisionModel`] 的输出。
    """

    loss: Optional[torch.FloatTensor] = None  # 对比损失，默认为空
    logits_per_image: torch.FloatTensor = None  # 图像嵌入和文本嵌入之间的缩放点积分数，默认为空
    logits_per_text: torch.FloatTensor = None  # 文本嵌入和图像嵌入之间的缩放点积分数，默认为空
    text_embeds: torch.FloatTensor = None  # 文本嵌入，默认为空
    image_embeds: torch.FloatTensor = None  # 图像嵌入，默认为空
    text_model_output: BaseModelOutputWithPooling = None  # 文本模型的输出，默认为空
    vision_model_output: BaseModelOutputWithPooling = None  # 图像模型的输出，默认为空

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
@dataclass
class CLIPSegDecoderOutput(ModelOutput):
    """
    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            Classification scores for each pixel.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None  # 分类器对每个像素的分类分数
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 隐藏状态，包含每个层的输出，如果设置了`output_hidden_states=True`或`config.output_hidden_states=True`时返回
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重，用于计算自注意力头中的加权平均值，如果设置了`output_attentions=True`或`config.output_attentions=True`时返回


@dataclass
class CLIPSegImageSegmentationOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        ...
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`CLIPSegVisionModel`].
    """

    loss: Optional[torch.FloatTensor] = None  # 图像-文本相似性的对比损失，当`return_loss`为`True`时返回
    logits: torch.FloatTensor = None  # 分类器的输出
    conditional_embeddings: torch.FloatTensor = None  # 条件嵌入
    pooled_output: torch.FloatTensor = None  # 池化后的输出
    vision_model_output: BaseModelOutputWithPooling = None  # [`CLIPSegVisionModel`]的输出
    decoder_output: CLIPSegDecoderOutput = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["vision_model_output", "decoder_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CLIPSegVisionEmbeddings(nn.Module):
    # 从transformers.models.clip.modeling_clip.CLIPVisionEmbeddings.__init__复制而来的
    # 初始化方法，接受一个 CLIPSegVisionConfig 类型的参数 config
    def __init__(self, config: CLIPSegVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置参数保存到实例的 config 属性中
        self.config = config
        # 设置嵌入维度为隐藏层大小
        self.embed_dim = config.hidden_size
        # 设置图像大小为配置参数中的图像大小
        self.image_size = config.image_size
        # 设置补丁大小为配置参数中的补丁大小
        self.patch_size = config.patch_size

        # 初始化类别嵌入，为嵌入维度大小的随机张量
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 初始化补丁嵌入，使用二维卷积层进行处理，输入通道数为图像通道数，输出通道数为嵌入维度，卷积核大小为补丁大小，步长为补丁大小，不带偏置
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算补丁数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 计算位置嵌入的数量，为补丁数量加一
        self.num_positions = self.num_patches + 1
        # 初始化位置嵌入，使用 Embedding 层，输入为位置数量，输出为嵌入维度
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 注册缓冲区 position_ids，保存位置嵌入的索引，持久性为非持久性
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # 插值位置嵌入的方法，接受一个新的大小参数 new_size
    def interpolate_position_embeddings(self, new_size):
        # 如果新大小参数不是长度为 2 的列表，则抛出 ValueError 异常
        if len(new_size) != 2:
            raise ValueError("new_size should consist of 2 values")

        # 计算每个方向的补丁数量
        num_patches_one_direction = int(self.num_patches**0.5)
        # 对位置嵌入进行二维插值
        a = self.position_embedding.weight[1:].T.view(
            1, self.config.hidden_size, num_patches_one_direction, num_patches_one_direction
        )
        b = (
            nn.functional.interpolate(a, new_size, mode="bicubic", align_corners=False)
            .squeeze(0)
            .view(self.config.hidden_size, new_size[0] * new_size[1])
            .T
        )
        # 将插值结果与原始位置嵌入拼接在一起
        result = torch.cat([self.position_embedding.weight[:1], b])

        return result

    # 前向传播方法，接受像素值作为输入，返回嵌入张量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取批次大小
        batch_size = pixel_values.shape[0]
        # 对输入像素值进行补丁嵌入，返回嵌入张量，形状为 [*， 宽度, 网格, 网格]
        patch_embeds = self.patch_embedding(pixel_values)
        # 将补丁嵌入张量扁平化，并进行维度转置
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 扩展类别嵌入，使其与补丁嵌入形状匹配
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # 将类别嵌入和补丁嵌入拼接在一起，得到总的嵌入张量
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # 如果总的嵌入张量形状中的位置数量与预期不同，则进行插值处理
        if embeddings.shape[1] != self.num_positions:
            # 计算新的形状，取总的嵌入张量形状中位置数量的平方根作为新的大小
            new_shape = int(math.sqrt(embeddings.shape[1] - 1))
            # 对嵌入张量进行位置插值
            embeddings = embeddings + self.interpolate_position_embeddings((new_shape, new_shape))
            # 将插值结果转换为与嵌入张量相同的数据类型
            embeddings = embeddings.to(embeddings.dtype)
        else:
            # 若位置数量与预期相同，则直接使用位置嵌入
            embeddings = embeddings + self.position_embedding(self.position_ids)

        # 返回嵌入张量
        return embeddings
# 从transformers.models.clip.modeling_clip.CLIPTextEmbeddings复制而来，用于CLIPSeg的文本嵌入
class CLIPSegTextEmbeddings(nn.Module):
    def __init__(self, config: CLIPSegTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        # 创建一个token嵌入层，将输入的token映射到embed_dim维度的向量
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 创建一个位置嵌入层，将位置编码映射到embed_dim维度的向量
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 为position_ids创建一个持久化缓冲区，用于存储位置编码的张量
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

        # 如果未提供位置编码，则使用预设的位置编码
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供输入嵌入，根据输入的token id获取对应的token嵌入
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 获取位置编码的嵌入
        position_embeddings = self.position_embedding(position_ids)
        # 将token嵌入和位置编码的嵌入相加，得到最终的嵌入向量
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# 从transformers.models.clip.modeling_clip.CLIPAttention复制而来，用于CLIPSeg的注意力机制
class CLIPSegAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim必须可以被num_heads整除（得到`embed_dim`:{self.embed_dim}和`num_heads`:{self.num_heads}）。"
            )
        # 缩放因子，用于调整注意力分数
        self.scale = self.head_dim**-0.5
        # dropout概率
        self.dropout = config.attention_dropout

        # 对键、值、查询进行线性投影
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # 用于调整张量形状的函数，将张量转换为期望的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类（nn.Module）的初始化方法
        super().__init__()
        # 将传入的配置信息保存在实例中
        self.config = config
        # 根据配置信息中的激活函数类型选择相应的激活函数，并保存在实例中
        self.activation_fn = ACT2FN[config.hidden_act]
        # 创建一个全连接层，输入维度为配置中的隐藏层大小，输出维度为配置中的中间层大小
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建一个全连接层，输入维度为配置中的中间层大小，输出维度为配置中的隐藏层大小
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    # 前向传播函数，用于定义模型的前向计算逻辑
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用第一个全连接层进行计算，得到中间层的输出
        hidden_states = self.fc1(hidden_states)
        # 将中间层的输出通过激活函数进行非线性变换
        hidden_states = self.activation_fn(hidden_states)
        # 使用第二个全连接层进行计算，得到最终的隐藏状态输出
        hidden_states = self.fc2(hidden_states)
        # 返回隐藏状态输出
        return hidden_states
# 从 CLIPSeg 模型中复制了 CLIPEncoderLayer 类，用于 CLIPSegEncoderLayer
class CLIPSegEncoderLayer(nn.Module):
    def __init__(self, config: CLIPSegConfig):
        super().__init__()
        # 设置嵌入维度为隐藏尺寸
        self.embed_dim = config.hidden_size
        # 初始化自注意力机制
        self.self_attn = CLIPSegAttention(config)
        # 初始化第一个 LayerNorm 层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 初始化 MLP
        self.mlp = CLIPSegMLP(config)
        # 初始化第二个 LayerNorm 层
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
            hidden_states (`torch.FloatTensor`): 输入到该层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): 注意力掩码，大小为
                `(batch, 1, tgt_len, src_len)`，其中填充元素由非常大的负值指示。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。更多细节见返回的张量中的 `attentions`。
        """
        # 保存残差连接
        residual = hidden_states

        # 第一个 LayerNorm 层
        hidden_states = self.layer_norm1(hidden_states)
        # 自注意力机制
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 残差连接
        hidden_states = residual + hidden_states

        # 保存残差连接
        residual = hidden_states
        # 第二个 LayerNorm 层
        hidden_states = self.layer_norm2(hidden_states)
        # MLP
        hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        # 如果要输出注意力权重
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPSegPreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和预训练模型下载加载的抽象类。
    """

    # CLIPSegConfig 类
    config_class = CLIPSegConfig
    # 基础模型前缀
    base_model_prefix = "clip"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_factor
        # 如果模块是 CLIPSegTextEmbeddings 类型
        if isinstance(module, CLIPSegTextEmbeddings):
            # 初始化 token_embedding 权重
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            # 初始化 position_embedding 权重
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        # 如果模块是 CLIPSegVisionEmbeddings 类型
        elif isinstance(module, CLIPSegVisionEmbeddings):
            factor = self.config.initializer_factor
            # 初始化 class_embedding 权重
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            # 初始化 patch_embedding 权重
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            # 初始化 position_embedding 权重
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # 如果模块是 CLIPSegAttention 类型
        elif isinstance(module, CLIPSegAttention):
            factor = self.config.initializer_factor
            # 初始化输入投影权重标准差
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 初始化输出投影权重标准差
            out_proj_std = (module.embed_dim**-0.5) * factor
            # 初始化 q_proj 权重
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            # 初始化 k_proj 权重
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            # 初始化 v_proj 权重
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            # 初始化 out_proj 权重
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果模块是 CLIPSegMLP 类型
        elif isinstance(module, CLIPSegMLP):
            factor = self.config.initializer_factor
            # 初始化输入投影权重标准差
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 初始化全连接层权重标准差
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 初始化 fc1 权重
            nn.init.normal_(module.fc1.weight, std=fc_std)
            # 初始化 fc2 权重
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # 如果模块是 CLIPSegModel 类型
        elif isinstance(module, CLIPSegModel):
            # 初始化文本投影权重
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            # 初始化视觉投影权重
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )

        # 如果模块是 nn.LayerNorm 类型
        if isinstance(module, nn.LayerNorm):
            # 将偏置项置零
            module.bias.data.zero_()
            # 将权重项填充为 1.0
            module.weight.data.fill_(1.0)
        # 如果模块是 nn.Linear 类型并且具有偏置项
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置项置零
            module.bias.data.zero_()
CLIPSEG_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CLIPSegConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


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


CLIPSEG_VISION_INPUTS_DOCSTRING = r"""
    This part is currently empty. It might be intended for documentation related to vision inputs, but there's no content provided yet.
"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下，如果提供了填充，则会被忽略。可以使用 [`AutoImageProcessor`] 获取像素值。有关详细信息，请参见 [`CLIPImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
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
        output_attentions (`bool`, *optional`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional`):
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
        # 初始化函数，创建一个CLIPSegEncoder对象
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建多个CLIPSegEncoderLayer层，数量为config.num_hidden_layers
        self.layers = nn.ModuleList([CLIPSegEncoderLayer(config) for _ in range(config.num_hidden_layers)])
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
class CLIPSegTextTransformer(nn.Module):
    # 定义一个名为CLIPSegTextTransformer的类，继承自nn.Module
    # 从transformers.models.clip.modeling_clip.CLIPTextTransformer.__init__中复制代码，将CLIP替换为CLIPSeg
    def __init__(self, config: CLIPSegTextConfig):
        # 初始化方法，接受一个CLIPSegTextConfig类型的参数config
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        # 创建一个CLIPSegTextEmbeddings对象并赋值给self.embeddings
        self.embeddings = CLIPSegTextEmbeddings(config)
        # 创建一个CLIPSegEncoder对象并赋值给self.encoder
        self.encoder = CLIPSegEncoder(config)
        # 创建一个具有指定维度和eps的LayerNorm对象并赋值给self.final_layer_norm
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 用于计算`pooled_output`
        self.eos_token_id = config.eos_token_id

    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegTextConfig)
    # 从transformers.models.clip.modeling_clip.CLIPTextTransformer.forward中复制代码，将clip->clipseg, CLIP->CLIPSeg
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

class CLIPSegTextModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegTextConfig

    _no_split_modules = ["CLIPSegTextEmbeddings", "CLIPSegEncoderLayer"]

    def __init__(self, config: CLIPSegTextConfig):
        # 初始化方法，接受一个CLIPSegTextConfig类型的参数config
        super().__init__(config)
        # 创建一个CLIPSegTextTransformer对象并赋值给self.text_model
        self.text_model = CLIPSegTextTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回self.text_model.embeddings.token_embedding
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        # 将value赋值给self.text_model.embeddings.token_embedding
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegTextConfig)
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
        >>> from transformers import AutoTokenizer, CLIPSegTextModel

        >>> tokenizer = AutoTokenizer.from_pretrained("CIDAS/clipseg-rd64-refined")
        >>> model = CLIPSegTextModel.from_pretrained("CIDAS/clipseg-rd64-refined")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        # 调用文本模型，传入参数并返回结果
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
class CLIPSegVisionTransformer(nn.Module):
    # 从transformers.models.clip.modeling_clip.CLIPVisionTransformer.__init__复制而来，将CLIP改为CLIPSeg
    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # 初始化嵌入层
        self.embeddings = CLIPSegVisionEmbeddings(config)
        # 初始化LayerNorm层
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 初始化编码器
        self.encoder = CLIPSegEncoder(config)
        # 初始化LayerNorm层
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegVisionConfig)
    # 从transformers.models.clip.modeling_clip.CLIPVisionTransformer.forward复制而来
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        返回：

        """
        # 如果没有传入像素值，引发错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 获取嵌入层的输出
        hidden_states = self.embeddings(pixel_values)
        # 对嵌入层的输出进行LayerNorm
        hidden_states = self.pre_layrnorm(hidden_states)

        # 将嵌入层的输出传入编码器
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的输出，包括最后一层隐藏状态和汇总输出
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        # 对汇总输出进行LayerNorm
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不要求返回字典，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果要求返回字典，则返回BaseModelOutputWithPooling对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPSegVisionModel(CLIPSegPreTrainedModel):
    config_class = CLIPSegVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPSegVisionConfig):
        super().__init__(config)
        # 初始化视觉模型
        self.vision_model = CLIPSegVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回嵌入层
        return self.vision_model.embeddings.patch_embedding
    # 将模型输入的文档字符串添加到模型前向传播方法中，以便自动生成 API 文档
    @add_start_docstrings_to_model_forward(CLIPSEG_VISION_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串为指定的输出类型和配置类
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPSegVisionConfig)
    # 定义模型的前向传播方法，接受像素值、是否输出注意力、是否输出隐藏状态和是否返回字典等参数，返回包含池化输出的元组或 BaseModelOutputWithPooling 类型对象
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

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
        ```"""
        # 调用视觉模型的前向传播方法，传递像素值、是否输出注意力、是否输出隐藏状态和是否返回字典等参数
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
# 根据给定的 CLIPSeg 的文档字符串添加注释
@add_start_docstrings(CLIPSEG_START_DOCSTRING)
# 定义 CLIPSegModel 类，继承自 CLIPSegPreTrainedModel 类
class CLIPSegModel(CLIPSegPreTrainedModel):
    # 指定配置类为 CLIPSegConfig
    config_class = CLIPSegConfig

    # 初始化方法，接收一个 CLIPSegConfig 对象作为参数
    def __init__(self, config: CLIPSegConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 检查 config.text_config 是否为 CLIPSegTextConfig 类型
        if not isinstance(config.text_config, CLIPSegTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPSegTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.vision_config 是否为 CLIPSegVisionConfig 类型
        if not isinstance(config.vision_config, CLIPSegVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPSegVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 从 config 中获取 text_config 和 vision_config
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度为 config.projection_dim
        self.projection_dim = config.projection_dim
        # 设置文本嵌入维度为 text_config.hidden_size
        self.text_embed_dim = text_config.hidden_size
        # 设置视觉嵌入维度为 vision_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建 CLIPSegTextTransformer 对象，并赋值给 self.text_model
        self.text_model = CLIPSegTextTransformer(text_config)
        # 创建 CLIPSegVisionTransformer 对象，并赋值给 self.vision_model
        self.vision_model = CLIPSegVisionTransformer(vision_config)

        # 创建线性层，用于视觉嵌入的投影，将其维度从 self.vision_embed_dim 转换为 self.projection_dim
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        # 创建线性层，用于文本嵌入的投影，将其维度从 self.text_embed_dim 转换为 self.projection_dim
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        # 创建一个可学习的参数，用于缩放对数输出
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

    # 根据给定的文档字符串添加注释
    @add_start_docstrings_to_model_forward(CLIPSEG_TEXT_INPUTS_DOCSTRING)
    # 获取文本特征的方法
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get text outputs from the text model
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the pooled output from text outputs and apply text projection
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        # Return the text features
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
        # 使用 CLIPSEG 模型的配置替代视觉和文本组件的一些字段（如果指定）
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将视觉输出传递给视觉模型
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取汇总输出
        pooled_output = vision_outputs[1]  # pooled_output
        # 应用视觉投影层得到图像特征
        image_features = self.visual_projection(pooled_output)

        # 返回图像特征
        return image_features

    @add_start_docstrings_to_model_forward(CLIPSEG_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPSegOutput, config_class=CLIPSegConfig)
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
class CLIPSegDecoderLayer(nn.Module):
    """
    CLIPSeg decoder layer, which is identical to `CLIPSegEncoderLayer`, except that normalization is applied after
    self-attention/MLP, rather than before.
    """

    # Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer.__init__ with CLIP->CLIPSeg
    def __init__(self, config: CLIPSegConfig):
        # 初始化 CLIPSegDecoderLayer 类
        super().__init__()
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 创建自注意力机制层
        self.self_attn = CLIPSegAttention(config)
        # 创建第一个 LayerNorm 层
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 创建多层感知机层
        self.mlp = CLIPSegMLP(config)
        # 创建第二个 LayerNorm 层
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
        # 保存残差连接的输入
        residual = hidden_states

        # 进行自注意力计算
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )

        # 添加残差连接，并进行 LayerNorm
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        # 保存残差连接的输入
        residual = hidden_states
        # 经过多层感知机层
        hidden_states = self.mlp(hidden_states)
        # 添加残差连接，并进行 LayerNorm
        hidden_states = residual + hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        # 输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPSegDecoder(CLIPSegPreTrainedModel):
    # 初始化函数，接受一个 CLIPSegConfig 对象作为参数
    def __init__(self, config: CLIPSegConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 从配置中获取条件层的信息
        self.conditional_layer = config.conditional_layer

        # 使用线性层进行 FiLM 操作的乘法部分
        self.film_mul = nn.Linear(config.projection_dim, config.reduce_dim)
        # 使用线性层进行 FiLM 操作的加法部分
        self.film_add = nn.Linear(config.projection_dim, config.reduce_dim)

        # 如果配置指定使用复杂的转置卷积
        if config.use_complex_transposed_convolution:
            # 计算转置卷积核的大小
            transposed_kernels = (config.vision_config.patch_size // 4, config.vision_config.patch_size // 4)

            # 定义转置卷积网络
            self.transposed_convolution = nn.Sequential(
                # 3x3 卷积层
                nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
                # ReLU 激活函数
                nn.ReLU(),
                # 第一个转置卷积层
                nn.ConvTranspose2d(
                    config.reduce_dim,
                    config.reduce_dim // 2,
                    kernel_size=transposed_kernels[0],
                    stride=transposed_kernels[0],
                ),
                # ReLU 激活函数
                nn.ReLU(),
                # 第二个转置卷积层
                nn.ConvTranspose2d(
                    config.reduce_dim // 2, 1, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]
                ),
            )
        else:
            # 如果不使用复杂的转置卷积，则直接定义一个转置卷积层
            self.transposed_convolution = nn.ConvTranspose2d(
                config.reduce_dim, 1, config.vision_config.patch_size, stride=config.vision_config.patch_size
            )

        # 获取需要进行降维的层数
        depth = len(config.extract_layers)
        # 定义多个线性层作为特征图降维
        self.reduces = nn.ModuleList(
            [nn.Linear(config.vision_config.hidden_size, config.reduce_dim) for _ in range(depth)]
        )

        # 复制视觉编码器的配置并修改以用于解码器
        decoder_config = copy.deepcopy(config.vision_config)
        decoder_config.hidden_size = config.reduce_dim
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.hidden_act = "relu"
        # 定义多个 CLIPSegDecoderLayer 作为解码器层
        self.layers = nn.ModuleList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers))])

    # 前向传播函数
    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        # 初始化所有隐藏状态的变量，如果不输出隐藏状态则置为 None
        all_hidden_states = () if output_hidden_states else None
        # 初始化所有注意力权重的变量，如果不输出注意力权重则置为 None
        all_attentions = () if output_attentions else None

        # 将隐藏状态列表反转，以便从后往前遍历
        activations = hidden_states[::-1]

        # 初始化输出变量
        output = None
        # 遍历隐藏状态、层和降维函数的组合
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
            # 如果输出不为空，则将当前层的激活结果与输出相加
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)

            # 如果当前层是条件层，则进行条件归一化操作
            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(
                    conditional_embeddings
                )
                output = output.permute(1, 0, 2)

            # 调用当前层的前向传播函数
            layer_outputs = layer(
                output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions
            )

            # 更新输出为当前层的输出
            output = layer_outputs[0]

            # 如果需要输出隐藏状态，则将当前层的隐藏状态加入到列表中
            if output_hidden_states:
                all_hidden_states += (output,)

            # 如果需要输出注意力权重，则将当前层的注意力权重加入到列表中
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 移除特殊标记（CLS）并重新整形输出
        output = output[:, 1:, :].permute(0, 2, 1)  # remove cls token and reshape to [batch_size, reduce_dim, seq_len]

        # 计算新输出的尺寸
        size = int(math.sqrt(output.shape[2]))

        # 获取条件嵌入的批量大小
        batch_size = conditional_embeddings.shape[0]
        # 重新调整输出的形状以匹配转置卷积层的输入
        output = output.view(batch_size, output.shape[1], size, size)

        # 将输出传递给转置卷积层并压缩最后一个维度
        logits = self.transposed_convolution(output).squeeze()

        # 如果不返回字典，则返回元组，排除空值
        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_attentions] if v is not None)

        # 返回 CLIPSegDecoderOutput 类的实例，包含 logits、隐藏状态和注意力权重
        return CLIPSegDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
@add_start_docstrings(
    """
    CLIPSeg model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.
    """,
    CLIPSEG_START_DOCSTRING,
)
# 定义 CLIPSegForImageSegmentation 类，继承自 CLIPSegPreTrainedModel 类
class CLIPSegForImageSegmentation(CLIPSegPreTrainedModel):
    # 设置配置类为 CLIPSegConfig
    config_class = CLIPSegConfig

    # 初始化方法
    def __init__(self, config: CLIPSegConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 将参数配置保存到实例属性中
        self.config = config

        # 创建 CLIPSegModel 实例
        self.clip = CLIPSegModel(config)
        # 获取要提取特征的层列表
        self.extract_layers = config.extract_layers

        # 创建 CLIPSegDecoder 实例
        self.decoder = CLIPSegDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取条件嵌入向量的方法
    def get_conditional_embeddings(
        self,
        batch_size: int = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        conditional_pixel_values: Optional[torch.Tensor] = None,
    ):
        # 如果传入了 input_ids
        if input_ids is not None:
            # 从文本计算条件嵌入向量
            if len(input_ids) != batch_size:
                raise ValueError("Make sure to pass as many prompt texts as there are query images")
            # 使用 CLIP 模型获取文本特征
            with torch.no_grad():
                conditional_embeddings = self.clip.get_text_features(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids
                )
        # 如果传入了 conditional_pixel_values
        elif conditional_pixel_values is not None:
            # 从图像计算条件嵌入向量
            if len(conditional_pixel_values) != batch_size:
                raise ValueError("Make sure to pass as many prompt images as there are query images")
            # 使用 CLIP 模型获取图像特征
            with torch.no_grad():
                conditional_embeddings = self.clip.get_image_features(conditional_pixel_values)
        else:
            # 抛出异常，条件不合法
            raise ValueError(
                "Invalid conditional, should be either provided as `input_ids` or `conditional_pixel_values`"
            )

        return conditional_embeddings

    # 前向传播方法
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
```