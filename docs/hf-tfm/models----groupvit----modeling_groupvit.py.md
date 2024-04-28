# `.\models\groupvit\modeling_groupvit.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用此文件
# 可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件。请查看许可证以获取特定语言的权限和限制
""" PyTorch GroupViT 模型。"""

# 导入必要的库
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

# 导入自定义的模块
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
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "nvidia/groupvit-gcc-yfcc"

# 预训练模型的存档列表
GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/groupvit-gcc-yfcc",
    # 查看所有 GroupViT 模型 https://huggingface.co/models?filter=groupvit
]

# 对比损失函数，改编自 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# 从 transformers.models.clip.modeling_clip.clip_loss 复制的函数，将 clip->groupvit
def groupvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

# 硬 softmax 函数
def hard_softmax(logits: torch.Tensor, dim: int):
    y_soft = logits.softmax(dim)
    # 直通
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

# Gumbel softmax 函数
def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # 更稳定的实现 https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
        torch.tensor(1.0, device=logits.device, dtype=logits.dtype),
    )
    gumbels = gumbel_dist.sample(logits.shape)
    # 计算 Gumbel 分布的采样值，用于近似离散分布
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    # 对 Gumbel 分布的采样值进行 softmax 操作，得到软化的概率分布
    y_soft = gumbels.softmax(dim)

    if hard:
        # 使用直通法（Straight through）进行硬化操作
        # 找到概率最大的位置作为硬化后的输出
        index = y_soft.max(dim, keepdim=True)[1]
        # 创建一个与 logits 相同形状的全零张量，将硬化后的位置设为 1.0
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        # 计算直通法的输出
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # 使用重参数化技巧（Reparametrization trick）
        ret = y_soft
    # 返回最终结果
    return ret
# 调整注意力图的大小，使其适应指定的高度和宽度
def resize_attention_map(attentions, height, width, align_corners=False):
    """
    Args:
        attentions (`torch.Tensor`): attention map of shape [batch_size, groups, feat_height*feat_width]
        height (`int`): height of the output attention map
        width (`int`): width of the output attention map
        align_corners (`bool`, *optional*): the `align_corner` argument for `nn.functional.interpolate`.

    Returns:
        `torch.Tensor`: resized attention map of shape [batch_size, groups, height, width]
    """

    # 计算缩放比例
    scale = (height * width // attentions.shape[2]) ** 0.5
    if height > width:
        feat_width = int(np.round(width / scale))
        feat_height = attentions.shape[2] // feat_width
    else:
        feat_height = int(np.round(height / scale))
        feat_width = attentions.shape[2] // feat_height

    batch_size = attentions.shape[0]
    groups = attentions.shape[1]  # number of group token
    # 重塑注意力图的形状
    attentions = attentions.reshape(batch_size, groups, feat_height, feat_width)
    # 使用双线性插值调整注意力图的大小
    attentions = nn.functional.interpolate(
        attentions, size=(height, width), mode="bilinear", align_corners=align_corners
    )
    return attentions


# 从注意力图中获取分组信息
def get_grouping_from_attentions(attentions, hw_shape):
    """
    Args:
        attentions (`tuple(torch.FloatTensor)`: tuple of attention maps returned by `GroupViTVisionTransformer`
        hw_shape (`tuple(int)`): height and width of the output attention map
    Returns:
        `torch.Tensor`: the attention map of shape [batch_size, groups, height, width]
    """

    attn_maps = []
    with torch.no_grad():
        prev_attn_masks = None
        for attn_masks in attentions:
            # 调整注意力掩码的形状
            attn_masks = attn_masks.permute(0, 2, 1).contiguous()
            if prev_attn_masks is None:
                prev_attn_masks = attn_masks
            else:
                prev_attn_masks = prev_attn_masks @ attn_masks
            # 调整注意力图的大小
            cur_attn_map = resize_attention_map(prev_attn_masks.permute(0, 2, 1).contiguous(), *hw_shape)
            attn_maps.append(cur_attn_map)

    # 最终的分组注意力图
    final_grouping = attn_maps[-1]

    return final_grouping


# 分组ViT交叉注意力层
class GroupViTCrossAttentionLayer(nn.Module):
    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()
        self.attn = GroupViTAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GroupViTMLP(config)
        self.norm_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    # 定义一个前向传播函数，接受查询(query)和键(key)作为输入
    def forward(self, query, key):
        # 将查询作为输入
        x = query
        # 将查询与注意力函数的输出相加
        x = x + self.attn(query, encoder_hidden_states=key)[0]
        # 将查询与多层感知机(MLP)函数的输出相加
        x = x + self.mlp(self.norm2(x))
        # 对结果进行后处理
        x = self.norm_post(x)
        # 返回处理后的结果
        return x
class GroupViTAssignAttention(nn.Module):
    # 定义 GroupViTAssignAttention 类
    def __init__(self, config: GroupViTVisionConfig):
        # 初始化函数，接受 GroupViTVisionConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化函数

        # 缩放因子为 hidden_size 的倒数
        self.scale = config.hidden_size**-0.5

        # query 投影层
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # key 投影层
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # value 投影层
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 最终投影层
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 分配的 epsilon 值
        self.assign_eps = config.assign_eps

    def get_attn(self, attn, gumbel=True, hard=True):
        # 获取注意力分布
        if gumbel and self.training:
            # 如果使用 Gumbel Softmax，并且处于训练模式
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
        else:
            if hard:
                # 如果使用 Hard Softmax
                attn = hard_softmax(attn, dim=-2)
            else:
                # 使用普通 Softmax
                attn = nn.functional.softmax(attn, dim=-2)

        return attn

    def forward(self, query, key):
        # 前向传播函数
        value = key
        # [batch_size, query_length, channels]
        query = self.q_proj(query)

        # [batch_size, key_length, channels]
        key = self.k_proj(key)

        # [batch_size, key_length, channels]
        value = self.v_proj(value)

        # [batch_size, query_length, key_length]
        raw_attn = (query @ key.transpose(-2, -1)) * self.scale

        # 获取注意力分布
        attn = self.get_attn(raw_attn)
        soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)

        # 对注意力分布进行归一化
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)

        # 计算输出
        out = attn @ value

        out = self.proj(out)

        return out, soft_attn


class GroupViTTokenAssign(nn.Module):
    # 定义 GroupViTTokenAssign 类
    def __init__(self, config: GroupViTVisionConfig, num_group_token, num_output_group):
        # 初始化函数，接受 GroupViTVisionConfig 类型的参数 config，以及 num_group_token 和 num_output_group
        super().__init__()
        # 调用父类的初始化函数
        self.num_output_group = num_output_group
        # 输出组���数量

        # 对 group_tokens 进行归一化
        self.norm_tokens = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 分配 MLP 比率
        assign_mlp_ratio = (
            config.assign_mlp_ratio
            if isinstance(config.assign_mlp_ratio, collections.abc.Iterable)
            else (config.assign_mlp_ratio, config.assign_mlp_ratio)
        )
        tokens_dim, channels_dim = [int(x * config.hidden_size) for x in assign_mlp_ratio]
        # 中间 MLP 层
        self.mlp_inter = GroupViTMixerMLP(config, num_group_token, tokens_dim, num_output_group)
        # 对 tokens 后进行归一化
        self.norm_post_tokens = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 对 x 进行归一化
        self.norm_x = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 前分配注意力层
        self.pre_assign_attn = GroupViTCrossAttentionLayer(config)

        # 分配注意力层
        self.assign = GroupViTAssignAttention(config)
        # 对新 x 进行归一化
        self.norm_new_x = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 通道 MLP 层
        self.mlp_channels = GroupViTMLP(config, config.hidden_size, channels_dim, config.hidden_size)
    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [batch_size, num_group_tokens, channels]

        Returns:
            projected_group_tokens (torch.Tensor): [batch_size, num_output_groups, channels]
        """
        # 使用 MLP 对组标记进行投影，将形状从 [B, num_group_tokens, C] 转换为 [B, num_output_groups, C]
        projected_group_tokens = self.mlp_inter(group_tokens)
        # 对投影后的组标记进行后续的归一化处理
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, image_tokens, group_tokens):
        """
        Args:
            image_tokens (`torch.Tensor`): image tokens, of shape [batch_size, input_length, channels]
            group_tokens (`torch.Tensor`): group tokens, [batch_size, num_group_tokens, channels]
        """

        # 对组标记进行归一化处理
        group_tokens = self.norm_tokens(group_tokens)
        # 对图像标记进行归一化处理
        image_tokens = self.norm_x(image_tokens)
        # 对组标记进行投影
        projected_group_tokens = self.project_group_token(group_tokens)
        # 在预分配的注意力机制中使用投影后的组标记和图像标记
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, image_tokens)
        # 分配注意力并更新图像标记
        new_image_tokens, attention = self.assign(projected_group_tokens, image_tokens)
        # 将新的图像标记与投影后的组标记相加
        new_image_tokens += projected_group_tokens

        # 对新的图像标记进行通道 MLP 处理
        new_image_tokens = new_image_tokens + self.mlp_channels(self.norm_new_x(new_image_tokens))

        return new_image_tokens, attention
# 定义一个数据类，用于存储 GroupViT 模型的输出结果
@dataclass
class GroupViTModelOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        segmentation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`):
            Classification scores for each pixel.

            <Tip warning={true}>

            The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is
            to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the
            original image size as post-processing. You should always check your logits shape and resize as needed.

            </Tip>

        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of
            [`GroupViTTextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`GroupViTVisionModel`].
        text_model_output (`BaseModelOutputWithPooling`):
            The output of the [`GroupViTTextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`GroupViTVisionModel`].
    """

    # 定义类的属性
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    segmentation_logits: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    # 将类转换为元组的方法
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            # 如果属性不是 "text_model_output" 或 "vision_model_output"，则直接返���属性值；否则调用属性的 to_tuple 方法
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# 定义一个模块，用于将图像转换为补丁嵌入
class GroupViTPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    # 初始化方法
    def __init__(
        self,
        image_size: int = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
    # 初始化函数，继承父类的初始化方法
    ):
        # 如果 image_size 是可迭代对象，则保持不变，否则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        # 如果 patch_size 是可迭代对象，则保持不变，否则转换为元组
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 设置图像大小、补丁大小和补丁数量
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 创建卷积层，用于将输入图像转换为嵌入维度
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 获取输入张量的形状信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果不需要插值位置编码
        if not interpolate_pos_encoding:
            # 如果输入图像大小与模型要求的图像大小不匹配，则引发 ValueError 异常
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        # 将输入图像通过卷积层投影并展平，然后转置维度
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # 返回处理后的张量
        return x
class GroupViTVisionEmbeddings(nn.Module):
    # 定义 GroupViTVisionEmbeddings 类，继承自 nn.Module
    def __init__(self, config: GroupViTVisionConfig):
        # 初始化方法，接受一个 GroupViTVisionConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化方法

        self.patch_embeddings = GroupViTPatchEmbeddings(
            # 创建 patch_embeddings 属性，使用 GroupViTPatchEmbeddings 类初始化
            image_size=config.image_size,
            # 设置图像大小为 config 中的 image_size
            patch_size=config.patch_size,
            # 设置 patch 大小为 config 中的 patch_size
            num_channels=config.num_channels,
            # 设置通道数为 config 中的 num_channels
            embed_dim=config.hidden_size,
            # 设置嵌入维度为 config 中的 hidden_size
        )
        num_patches = self.patch_embeddings.num_patches
        # 获取 patch_embeddings 中的 num_patches 属性
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        # 创建 position_embeddings 属性，使用 torch.zeros 初始化，形状为 (1, num_patches, hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 创建 dropout 属性，使用 nn.Dropout 初始化，概率为 config 中的 dropout
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 layernorm 属性，使用 nn.LayerNorm 初始化，隐藏层大小为 config 中的 hidden_size，eps 为 config 中的 layer_norm_eps
        self.config = config
        # 创建 config 属性，保存传入的 config 参数

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # 定义 interpolate_pos_encoding 方法，接受 embeddings、height、width 三个参数，返回 torch.Tensor 类型的数据
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        # 方法的说明文档

        npatch = embeddings.shape[1]
        # 获取 embeddings 的第二维大小，即 patch 的数量
        if npatch == self.position_embeddings.shape[1] and height == width:
            # 如果 patch 数量与 position_embeddings 的第二维大小相等，并且高度等于宽度
            return self.position_embeddings
            # 返回 position_embeddings
        patch_pos_embed = self.position_embeddings
        # 将 position_embeddings 赋值给 patch_pos_embed
        num_original_pos_embed = patch_pos_embed.shape[1]
        # 获取原始位置编码的数量
        dim = embeddings.shape[-1]
        # 获取 embeddings 的最后一维大小
        feat_height = height // self.config.patch_size
        # 计算特征高度
        feat_width = width // self.config.patch_size
        # 计算特征宽度
        feat_height, feat_width = feat_height + 0.1, feat_width + 0.1
        # 对特征高度和宽度进行微小调整，避免浮点误差
        original_height = original_width = math.sqrt(num_original_pos_embed)
        # 计算原始高度和宽度
        reshaped_patch_pos_embed = patch_pos_embed.reshape(1, int(original_height), int(original_width), dim).permute(
            0, 3, 1, 2
        )
        # 重新整形位置编码
        scale_factor = (feat_height / original_height, feat_width / original_width)
        # 计算缩放因子
        patch_pos_embed = nn.functional.interpolate(
            reshaped_patch_pos_embed,
            scale_factor=scale_factor,
            mode="bicubic",
            align_corners=False,
        )
        # 使用插值方法对位置编码进行插值
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # 调整位置编码的维度
        return patch_pos_embed
        # 返回调整后的位置编码
    # 前向传播函数，接收像素值张量和是否插值位置编码作为参数，返回处理后的张量
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 获取输入像素值张量的形状信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 将像素值张量转换为补丁嵌入
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 对嵌入进行 LayerNorm 处理
        embeddings = self.layernorm(embeddings)

        # 获取处理后嵌入的形状信息
        batch_size, seq_len, _ = embeddings.size()

        # 如果需要插值位置编码，则对嵌入进行插值位置编码处理
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 否则直接添加位置编码
            embeddings = embeddings + self.position_embeddings

        # 对嵌入进行 Dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入张量
        return embeddings
# 定义一个名为 GroupViTTextEmbeddings 的类，继承自 nn.Module 类，用于处理文本嵌入
class GroupViTTextEmbeddings(nn.Module):
    # 初始化函数，接受一个 GroupViTTextConfig 类型的参数 config
    def __init__(self, config: GroupViTTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        # 创建一个词嵌入层，将词汇表大小和嵌入维度作为参数
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 创建一个位置嵌入层，将最大位置嵌入长度和嵌入维度作为参数
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 创建一个名为 position_ids 的缓冲区，存储位置 id，用于序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # 前向传播函数，接受输入的文本 id、位置 id 和嵌入向量，返回嵌入结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 获取序列长度
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        # 如果位置 id 为空，则使用预先定义的位置 id
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果输入嵌入向量为空，则使用词嵌入层获取嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 获取位置嵌入向量
        position_embeddings = self.position_embedding(position_ids)
        # 将词嵌入向量和位置嵌入向量相加得到最终嵌入结果
        embeddings = inputs_embeds + position_embeddings

        return embeddings


# 定义一个名为 GroupViTStage 的类，对应 GroupViT 实现中的 GroupingLayer 类
class GroupViTStage(nn.Module):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        depth: int,
        num_prev_group_token: int,
        num_group_token: int,
        num_output_group: int,
    ):
        super().__init__()
        self.depth = depth
        self.num_group_token = num_group_token
        # 如果存在分组 token，则创建一个参数化的分组 token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, config.hidden_size))
        else:
            self.group_token = None
        # 创建一个由 GroupViTEncoderLayer 组成的���列表
        self.layers = nn.ModuleList([GroupViTEncoderLayer(config) for _ in range(depth)])

        # 如果存在分组 token，则创建一个 GroupViTTokenAssign 实例用于下采样
        if num_group_token > 0:
            self.downsample = GroupViTTokenAssign(
                config=config,
                num_group_token=num_group_token,
                num_output_group=num_output_group,
            )
        else:
            self.downsample = None

        # 如果存在前一个分组 token 和当前分组 token，则创建一个分组投影器
        if num_prev_group_token > 0 and num_group_token > 0:
            self.group_projector = nn.Sequential(
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                GroupViTMixerMLP(config, num_prev_group_token, config.hidden_size // 2, num_group_token),
            )
        else:
            self.group_projector = None

    # 返回是否存在分组 token 的属性
    @property
    def with_group_token(self):
        return self.group_token is not None

    # 将输入 x 拆分为两部分，一部分是前面的 token，一部分是分组 token
    def split_x(self, x):
        if self.with_group_token:
            return x[:, : -self.num_group_token], x[:, -self.num_group_token :]
        else:
            return x, None
    # 将输入张量 x 与 group_token 进行拼接，如果 group_token 为 None，则直接返回 x
    def concat_x(self, x: torch.Tensor, group_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    # 前向传播函数，接收隐藏状态 hidden_states 和前一组 token prev_group_token 作为输入
    # 可选择是否输出注意力矩阵
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_group_token: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the grouping tensors of Grouping block.
        """
        # 如果模型包含 group_token
        if self.with_group_token:
            # 将 group_token 扩展到与 hidden_states 相同的形状
            group_token = self.group_token.expand(hidden_states.size(0), -1, -1)
            # 如果存在 group_projector，则对 group_token 进行投影
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None

        # 初始化 x 为 hidden_states
        x = hidden_states

        # 将 x 与 group_token 进行拼接
        cat_x = self.concat_x(x, group_token)
        # 遍历每个层进行前向传播
        for layer in self.layers:
            # 调用每个层的前向传播函数
            layer_out = layer(cat_x, attention_mask=None, causal_attention_mask=None)
            cat_x = layer_out[0]

        # 将 cat_x 拆分为 x 和 group_token
        x, group_token = self.split_x(cat_x)

        # 初始化 attention 为 None
        attention = None
        # 如果存在 downsample 模块，则对 x 和 group_token 进行下采样
        if self.downsample is not None:
            x, attention = self.downsample(x, group_token)

        # 将 x 和 group_token 作为输出
        outputs = (x, group_token)
        # 如果需要输出注意力矩阵，则将 attention 加入输出
        if output_attentions:
            outputs = outputs + (attention,)

        return outputs
class GroupViTMLP(nn.Module):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        output_size: Optional[int] = None,
    ):
        # 初始化函数，定义 GroupViTMLP 类
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        # 设置隐藏层大小，默认为配置中的隐藏层大小
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        # 设置中间层大小，默认为配置中的中间层大小
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        # 设置输出层大小，默认为隐藏层大小
        output_size = output_size if output_size is not None else hidden_size
        # 创建全连接层 fc1 和 fc2
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，对隐藏状态进行全连接层操作
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class GroupViTMixerMLP(GroupViTMLP):
    def forward(self, x):
        # GroupViTMixerMLP 类的前向传播函数，调用父类的前向传播函数并进行转置操作
        x = super().forward(x.transpose(1, 2))
        return x.transpose(1, 2)


class GroupViTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        # 初始化函数，定义 GroupViTAttention 类
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
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # 创建线性变换层 k_proj, v_proj, q_proj, out_proj
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重塑张量形状，用于多头注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->GroupViT
class GroupViTEncoderLayer(nn.Module):
    # 定义 GroupViTEncoderLayer 类
    # 初始化函数，接受一个 GroupViTConfig 类型的参数
    def __init__(self, config: GroupViTConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size
        # 创建 GroupViTAttention 对象
        self.self_attn = GroupViTAttention(config)
        # 创建 LayerNorm 层，用于归一化隐藏状态
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 创建 GroupViTMLP 对象
        self.mlp = GroupViTMLP(config)
        # 创建第二个 LayerNorm 层，用于归一化隐藏状态
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # 前向传播函数
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
        # 保存残差连接
        residual = hidden_states

        # 对隐藏状态进行 LayerNorm 归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 使用自注意力机制处理隐藏状态
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 保存残差连接
        residual = hidden_states
        # 对隐藏状态进行第二次 LayerNorm 归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 使用 MLP 处理隐藏状态
        hidden_states = self.mlp(hidden_states)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 将隐藏状态作为输出
        outputs = (hidden_states,)

        # 如果需要输出注意��权重，则将注意力权重添加到输出中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
class GroupViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 GroupViTConfig
    config_class = GroupViTConfig
    # 设置基础模型前缀为 "groupvit"
    base_model_prefix = "groupvit"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""

        # 初始化范围为配置中的初始化范围
        init_range = self.config.initializer_range
        # 如果是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=init_range)
            # 如果有偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零
            module.bias.data.zero_()
            # 初始化权重为 1.0
            module.weight.data.fill_(1.0)

        # 获取初始化因子
        factor = self.config.initializer_factor
        # 如果是 GroupViTTextEmbeddings
        if isinstance(module, GroupViTTextEmbeddings):
            # 初始化 token_embedding 和 position_embedding 的权重
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        # 如果是 GroupViTAttention
        elif isinstance(module, GroupViTAttention):
            # 计算初始化标准差
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            # 初始化权重
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果是 GroupViTMLP
        elif isinstance(module, GroupViTMLP):
            # 计算初始化标准差
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 初始化权重
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)


GROUPVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`GroupViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GROUPVIT_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，将忽略填充。
            # 可以使用 [`CLIPTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]` 之间：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。

        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。
        
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
定义了 GROUPVIT_VISION_INPUTS_DOCSTRING 字符串，用于描述 GroupViT 模型的输入参数
"""
GROUPVIT_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

"""
定义了 GROUPVIT_INPUTS_DOCSTRING 字符串，用于描述 GroupViT 模型的输入参数
"""
GROUPVIT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`CLIPTokenizer`]. See [`PreTrainedTokenizer.encode`] and
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
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`CLIPImageProcessor.__call__`] for details.
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


"""
定义了 GroupViTVisionEncoder 类，作为 GroupViT 模型的视觉编码器
"""
class GroupViTVisionEncoder(nn.Module):
    # 初始化函数，接受一个 GroupViTVisionConfig 类型的参数，并调用父类的初始化函数
    def __init__(self, config: GroupViTVisionConfig) -> None:
        super().__init__()
        # 将传入的配置参数保存到对象中
        self.config = config
        # 创建一个 nn.ModuleList 对象，其中包含多个 GroupViTStage 对象
        self.stages = nn.ModuleList(
            [
                GroupViTStage(
                    config=config,
                    depth=config.depths[i],
                    num_group_token=config.num_group_tokens[i],
                    num_output_group=config.num_output_groups[i],
                    num_prev_group_token=config.num_output_groups[i - 1] if i > 0 else 0,
                )
                for i in range(len(config.depths))
            ]
        )
        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False

    # 前向传播函数，接受输入的隐藏状态和一些可选参数，返回模型输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        # 根据传入的参数或配置参数确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据传入的参数或配置参数确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据传入的参数或配置参数确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 初始化存储所有隐藏状态和注意力权重的变量
        all_hidden_states = () if output_hidden_states else None
        all_groupings = () if output_attentions else None

        # 初始化组 token 为 None
        group_tokens = None

        # 遍历所有阶段，并进行前向传播
        for i, stage in enumerate(self.stages):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前阶段的前向传播函数，获取输出
            layer_outputs = stage(hidden_states, group_tokens, output_attentions)

            # 更新隐藏状态和组 token
            hidden_states = layer_outputs[0]
            group_tokens = layer_outputs[1]

            # 如果需要输出注意力权重，并且当前阶段有注意力权重输出，则将其添加到 all_groupings 中
            if output_attentions and layer_outputs[2] is not None:
                all_groupings = all_groupings + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则返回隐藏状态、所有隐藏状态和所有注意力权重
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_groupings] if v is not None)
        # 否则返回一个 BaseModelOutput 对象，包含最终隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_groupings
        )
class GroupViTTextEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self-attention layers. Each layer is a
    [`GroupViTEncoderLayer`].

    Args:
        config: GroupViTTextConfig
    """

    def __init__(self, config: GroupViTTextConfig):
        # 初始化函数，创建 GroupViTTextEncoder 类的实例
        super().__init__()
        self.config = config
        # 创建包含多个 GroupViTEncoderLayer 实例的 ModuleList
        self.layers = nn.ModuleList([GroupViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# Copied from transformers.models.clip.modeling_clip.CLIPTextTransformer with CLIPText->GroupViTText, CLIPEncoder->GroupViTTextEncoder, CLIP_TEXT->GROUPVIT_TEXT
class GroupViTTextTransformer(nn.Module):
    def __init__(self, config: GroupViTTextConfig):
        # 初始化函数，创建 GroupViTTextTransformer 类的实例
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        # 创建 GroupViTTextEmbeddings 实例
        self.embeddings = GroupViTTextEmbeddings(config)
        # 创建 GroupViTTextEncoder 实例
        self.encoder = GroupViTTextEncoder(config)
        # 创建 LayerNorm 实例
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=GroupViTTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class GroupViTTextModel(GroupViTPreTrainedModel):
    config_class = GroupViTTextConfig

    def __init__(self, config: GroupViTTextConfig):
        # 初始化函数，创建 GroupViTTextModel 类的实例
        super().__init__(config)
        self.text_model = GroupViTTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 获取输入 embeddings
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        # 设置输入 embeddings
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=GroupViTTextConfig)
    # 定义一个前向传播函数，接受输入的参数和返回值类型
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        position_ids: Optional[torch.Tensor] = None,  # 位置 ID
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
    ) -> Union[Tuple, BaseModelOutputWithPooling]:  # 返回值类型为元组或带池化的基础模型输出

        # 返回函数的说明文档
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, GroupViTTextModel

        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = GroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        # 调用文本模型的前向传播函数，传入参数并返回结果
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
class GroupViTVisionTransformer(nn.Module):
    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = GroupViTVisionEmbeddings(config)
        self.encoder = GroupViTVisionEncoder(config)
        self.layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=GroupViTVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]

        # normalize the last hidden state
        last_hidden_state = self.layernorm(last_hidden_state)
        pooled_output = last_hidden_state.mean(dim=1)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class GroupViTVisionModel(GroupViTPreTrainedModel):
    config_class = GroupViTVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: GroupViTVisionConfig):
        super().__init__(config)
        self.vision_model = GroupViTVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> GroupViTPatchEmbeddings:
        return self.vision_model.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=GroupViTVisionConfig)
    # 定义一个方法用于前向传播，接受输入参数包括像素值、是否输出注意力、是否输出隐藏状态、是否返回字典
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
        >>> from transformers import AutoProcessor, GroupViTVisionModel

        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = GroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        # 调用视觉模型进行前向传播，传入像素值、是否输出注意力、是否输出隐藏状态、是否返回字典
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
# 添加起始文档字符串到 GroupViTModel 类
@add_start_docstrings(GROUPVIT_START_DOCSTRING)
class GroupViTModel(GroupViTPreTrainedModel):
    # 设置配置类为 GroupViTConfig
    config_class = GroupViTConfig

    # 初始化函数，接受 GroupViTConfig 类型的参数
    def __init__(self, config: GroupViTConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 检查 config.text_config 是否为 GroupViTTextConfig 类型，如果不是则抛出异常
        if not isinstance(config.text_config, GroupViTTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type GroupViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.vision_config 是否为 GroupViTVisionConfig 类型，如果不是则抛出异常
        if not isinstance(config.vision_config, GroupViTVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type GroupViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 获取 text_config 和 vision_config
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置各种维度参数
        self.projection_dim = config.projection_dim
        self.projection_intermediate_dim = config.projection_intermediate_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建文本模型和视觉模型
        self.text_model = GroupViTTextTransformer(text_config)
        self.vision_model = GroupViTVisionTransformer(vision_config)

        # 创建视觉投影层
        self.visual_projection = nn.Sequential(
            nn.Linear(self.vision_embed_dim, self.projection_intermediate_dim, bias=True),
            nn.BatchNorm1d(self.projection_intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True),
        )
        # 创建文本投影层
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.projection_intermediate_dim, bias=True),
            nn.BatchNorm1d(self.projection_intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True),
        )
        # 创建 logit_scale 参数
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化权重并���用最终处理
        self.post_init()

    # 添加起始文档字符串到 get_text_features 方法
    @add_start_docstrings_to_model_forward(GROUPVIT_TEXT_INPUTS_DOCSTRING)
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
            applying the projection layer to the pooled output of [`GroupViTTextModel`].

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, GroupViTModel

        >>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use GROUPVIT model's config for some fields (if specified) instead of those of vision & text components.
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

        # Get the pooled output from text outputs and apply text projection to get text features
        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
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
            applying the projection layer to the pooled output of [`GroupViTVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, GroupViTModel

        >>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # Use GROUPVIT model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass the pixel values and other optional arguments to the vision model
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the pooled output from vision model
        pooled_output = vision_outputs[1]  # pooled_output
        # Apply visual projection to the pooled output to get image features
        image_features = self.visual_projection(pooled_output)

        # Return the image features
        return image_features

    @add_start_docstrings_to_model_forward(GROUPVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GroupViTModelOutput, config_class=GroupViTConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_segmentation: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```