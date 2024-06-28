# `.\models\groupvit\modeling_groupvit.py`

```
# 定义了编码格式为 UTF-8

import collections.abc  # 导入 collections.abc 模块，用于处理集合和容器数据类型
import math  # 导入数学模块，用于数学运算
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器，用于简化类的定义
from typing import Any, Optional, Tuple, Union  # 导入类型提示模块，用于声明函数参数和返回值类型

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 深度学习库
import torch.utils.checkpoint  # 导入 PyTorch 检查点模块，用于内存优化
from torch import nn  # 从 PyTorch 导入神经网络模块

from ...activations import ACT2FN  # 导入激活函数模块中的 ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask  # 导入自定义的注意力掩码函数
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling  # 导入模型输出相关类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型类
from ...utils import (  # 从工具模块导入多个实用函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig  # 导入 GroupViT 相关的配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "nvidia/groupvit-gcc-yfcc"  # 设置模型检查点的文档字符串

GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 定义预训练模型的存档列表
    "nvidia/groupvit-gcc-yfcc",
    # 可以在 https://huggingface.co/models?filter=groupvit 查看所有 GroupViT 模型
]


# 对比损失函数，从 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html 改编而来
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    计算对比损失函数。

    Args:
        logits (torch.Tensor): 模型输出的对比分数张量

    Returns:
        torch.Tensor: 对比损失值张量
    """
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# 从 transformers.models.clip.modeling_clip.clip_loss 复制并修改为 groupvit
def groupvit_loss(similarity: torch.Tensor) -> torch.Tensor:
    """
    计算 GroupViT 模型的损失函数。

    Args:
        similarity (torch.Tensor): 相似性分数张量

    Returns:
        torch.Tensor: GroupViT 损失值张量
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


def hard_softmax(logits: torch.Tensor, dim: int):
    """
    实现硬件 softmax 函数。

    Args:
        logits (torch.Tensor): 模型输出的 logits 张量
        dim (int): softmax 操作的维度

    Returns:
        torch.Tensor: 硬件 softmax 后的张量
    """
    y_soft = logits.softmax(dim)
    # 直通机制。
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    """
    实现 Gumbel softmax 函数。

    Args:
        logits (torch.Tensor): 模型输出的 logits 张量
        tau (float): Gumbel 分布的温度参数，默认为 1
        hard (bool): 是否使用硬件 softmax，即直通机制，默认为 False
        dim (int): softmax 操作的维度，默认为 -1

    Returns:
        torch.Tensor: Gumbel softmax 后的张量
    """
    # 更稳定的方式 https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0.0, device=logits.device, dtype=logits.dtype),
        torch.tensor(1.0, device=logits.device, dtype=logits.dtype),
    )
    gumbels = gumbel_dist.sample(logits.shape)
    gumbels = (logits + gumbels) / tau  # 计算 Gumbel 分布的样本，公式为 (logits + gumbels) / tau

    y_soft = gumbels.softmax(dim)  # 对 Gumbel 分布样本进行 softmax 操作，得到软化后的分布 y_soft

    if hard:
        # 使用直通法（Straight through）进行硬化操作
        index = y_soft.max(dim, keepdim=True)[1]  # 找到每行中最大值的索引，用于硬化操作
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        # 利用 scatter_ 方法在指定维度上将索引处置为 1.0，从而得到硬化后的独热编码 y_hard
        ret = y_hard - y_soft.detach() + y_soft  # 使用直通法修正硬化结果，得到最终输出 ret
    else:
        # 使用重参数化技巧（Reparametrization trick）
        ret = y_soft  # 直接返回软化后的分布 y_soft
    return ret
# 定义一个函数，用于调整注意力图的大小
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
        # 根据高度缩放比例计算特征图的宽度和高度
        feat_width = int(np.round(width / scale))
        feat_height = attentions.shape[2] // feat_width
    else:
        # 根据宽度缩放比例计算特征图的高度和宽度
        feat_height = int(np.round(height / scale))
        feat_width = attentions.shape[2] // feat_height

    batch_size = attentions.shape[0]
    groups = attentions.shape[1]  # 表示组token的数量
    # 将原始形状的注意力图重塑为 [batch_size, groups, feat_height, feat_width]
    attentions = attentions.reshape(batch_size, groups, feat_height, feat_width)
    # 使用双线性插值方法调整注意力图的大小到指定的 [height, width]
    attentions = nn.functional.interpolate(
        attentions, size=(height, width), mode="bilinear", align_corners=align_corners
    )
    return attentions


# 定义一个函数，从注意力图中获取分组信息
def get_grouping_from_attentions(attentions, hw_shape):
    """
    Args:
        attentions (`tuple(torch.FloatTensor)`): tuple of attention maps returned by `GroupViTVisionTransformer`
        hw_shape (`tuple(int)`): height and width of the output attention map
    Returns:
        `torch.Tensor`: the attention map of shape [batch_size, groups, height, width]
    """

    attn_maps = []
    with torch.no_grad():
        prev_attn_masks = None
        # 遍历每个注意力图
        for attn_masks in attentions:
            # 将注意力掩码重排列为 [batch_size, num_groups, height x width]
            attn_masks = attn_masks.permute(0, 2, 1).contiguous()
            if prev_attn_masks is None:
                prev_attn_masks = attn_masks
            else:
                # 将前一个注意力掩码与当前注意力掩码相乘
                prev_attn_masks = prev_attn_masks @ attn_masks
            # 调用 resize_attention_map 函数调整当前的注意力图大小
            cur_attn_map = resize_attention_map(prev_attn_masks.permute(0, 2, 1).contiguous(), *hw_shape)
            attn_maps.append(cur_attn_map)

    # 返回最终的分组注意力图，形状为 [batch_size, num_groups, height, width]
    final_grouping = attn_maps[-1]

    return final_grouping


# 定义一个类，实现 GroupViT 模型的跨注意力层
class GroupViTCrossAttentionLayer(nn.Module):
    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()
        self.attn = GroupViTAttention(config)  # 初始化 GroupViTAttention
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # LayerNorm 层
        self.mlp = GroupViTMLP(config)  # 初始化 GroupViTMLP
        self.norm_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # LayerNorm 层
    # 定义神经网络模型的前向传播函数，接受查询(query)和键(key)作为输入
    def forward(self, query, key):
        # 将输入的查询(query)赋值给变量x
        x = query
        # 将查询(query)和键(key)传递给自注意力机制(attn)，并将返回的结果与x相加
        x = x + self.attn(query, encoder_hidden_states=key)[0]
        # 将x传递给多层感知机(mlp)和层归一化(norm2)，然后将得到的结果与x相加
        x = x + self.mlp(self.norm2(x))
        # 将处理后的x传递给后层归一化(norm_post)
        x = self.norm_post(x)
        # 返回处理后的结果x作为前向传播的输出
        return x
class GroupViTAssignAttention(nn.Module):
    # 定义一个 GroupViTAssignAttention 类，继承自 nn.Module
    def __init__(self, config: GroupViTVisionConfig):
        # 初始化方法，接受一个配置参数 config
        super().__init__()
        # 初始化 scale 参数为 hidden_size 的倒数平方
        self.scale = config.hidden_size**-0.5

        # 定义线性层 q_proj，将输入特征大小映射到 hidden_size
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义线性层 k_proj，将输入特征大小映射到 hidden_size
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义线性层 v_proj，将输入特征大小映射到 hidden_size
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义线性层 proj，将输入特征大小映射到 hidden_size
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        # 设置 assign_eps 参数
        self.assign_eps = config.assign_eps

    def get_attn(self, attn, gumbel=True, hard=True):
        # 根据 gumbel 和 hard 参数获取注意力分布
        if gumbel and self.training:
            # 如果 gumbel 为 True 且处于训练模式，使用 gumbel_softmax 进行注意力分配
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
        else:
            # 否则，根据 hard 参数选择 softmax 或者 hard_softmax
            if hard:
                attn = hard_softmax(attn, dim=-2)
            else:
                attn = nn.functional.softmax(attn, dim=-2)

        return attn

    def forward(self, query, key):
        # 前向传播函数，接受 query 和 key 作为输入
        value = key
        # 将 query 映射到 hidden_size 维度
        query = self.q_proj(query)

        # 将 key 映射到 hidden_size 维度
        key = self.k_proj(key)

        # 将 value 映射到 hidden_size 维度
        value = self.v_proj(value)

        # 计算原始的注意力分数，query 和 key 的点积，乘以缩放因子 scale
        raw_attn = (query @ key.transpose(-2, -1)) * self.scale

        # 获取注意力分布，调用 get_attn 方法
        attn = self.get_attn(raw_attn)
        # 获取软化的注意力分布，关闭 gumbel 并使用 softmax
        soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)

        # 归一化注意力分布
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)

        # 计算最终输出，注意力分布乘以 value
        out = attn @ value

        # 将输出再次映射到 hidden_size 维度
        out = self.proj(out)

        return out, soft_attn


class GroupViTTokenAssign(nn.Module):
    # 定义一个 GroupViTTokenAssign 类，继承自 nn.Module
    def __init__(self, config: GroupViTVisionConfig, num_group_token, num_output_group):
        # 初始化方法，接受配置参数 config、群组 token 数量 num_group_token 和输出群组数 num_output_group
        super().__init__()
        # 设置输出的群组数量
        self.num_output_group = num_output_group
        # 对群组 token 进行层归一化
        self.norm_tokens = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 处理 assign_mlp_ratio 参数，确保其为可迭代对象，如果不是则复制为元组
        assign_mlp_ratio = (
            config.assign_mlp_ratio
            if isinstance(config.assign_mlp_ratio, collections.abc.Iterable)
            else (config.assign_mlp_ratio, config.assign_mlp_ratio)
        )
        # 计算 token 和 channels 维度的大小，基于 assign_mlp_ratio 和 hidden_size
        tokens_dim, channels_dim = [int(x * config.hidden_size) for x in assign_mlp_ratio]
        
        # 创建 GroupViTMixerMLP 层，处理群组 token 和输出群组数
        self.mlp_inter = GroupViTMixerMLP(config, num_group_token, tokens_dim, num_output_group)
        
        # 对 tokens 后的归一化处理
        self.norm_post_tokens = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 对 x 进行归一化处理
        self.norm_x = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 创建 GroupViTCrossAttentionLayer 层作为预分配注意力的层
        self.pre_assign_attn = GroupViTCrossAttentionLayer(config)
        
        # 创建 GroupViTAssignAttention 层，处理注意力分配
        self.assign = GroupViTAssignAttention(config)
        
        # 对新的 x 进行归一化处理
        self.norm_new_x = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 创建 GroupViTMLP 层，处理 hidden_size 和 channels 维度
        self.mlp_channels = GroupViTMLP(config, config.hidden_size, channels_dim, config.hidden_size)
    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [batch_size, num_group_tokens, channels]

        Returns:
            projected_group_tokens (torch.Tensor): [batch_size, num_output_groups, channels]
        """
        # 使用 self.mlp_inter 对 group_tokens 进行线性变换和非线性变换
        projected_group_tokens = self.mlp_inter(group_tokens)
        # 对变换后的 group tokens 进行后续的归一化处理
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, image_tokens, group_tokens):
        """
        Args:
            image_tokens (`torch.Tensor`): image tokens, of shape [batch_size, input_length, channels]
            group_tokens (`torch.Tensor`): group tokens, [batch_size, num_group_tokens, channels]
        """

        # 对输入的 group tokens 进行归一化处理
        group_tokens = self.norm_tokens(group_tokens)
        # 对输入的 image tokens 进行归一化处理
        image_tokens = self.norm_x(image_tokens)
        
        # 调用 project_group_token 方法进行处理，得到投影后的 group tokens
        projected_group_tokens = self.project_group_token(group_tokens)
        
        # 使用 self.pre_assign_attn 对投影后的 group tokens 和 image tokens 进行预分配注意力
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, image_tokens)
        
        # 使用 self.assign 方法将投影后的 group tokens 分配到 image tokens 上，并返回新的 image tokens 和注意力分布
        new_image_tokens, attention = self.assign(projected_group_tokens, image_tokens)
        
        # 将投影后的 group tokens 添加到新的 image tokens 上
        new_image_tokens += projected_group_tokens
        
        # 对新的 image tokens 进行通道维度的 MLP 处理
        new_image_tokens = new_image_tokens + self.mlp_channels(self.norm_new_x(new_image_tokens))

        return new_image_tokens, attention
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

    loss: Optional[torch.FloatTensor] = None  # 损失，用于图像文本相似性对比的对比损失
    logits_per_image: torch.FloatTensor = None  # 图像嵌入和文本嵌入之间的缩放点积分数，表示图像-文本的相似性分数
    logits_per_text: torch.FloatTensor = None  # 文本嵌入和图像嵌入之间的缩放点积分数，表示文本-图像的相似性分数
    segmentation_logits: torch.FloatTensor = None  # 每个像素的分类分数，形状为 (batch_size, config.num_labels, logits_height, logits_width)

    text_embeds: torch.FloatTensor = None  # 应用投影层到文本模型输出汇总输出后得到的文本嵌入
    image_embeds: torch.FloatTensor = None  # 应用投影层到视觉模型输出汇总输出后得到的图像嵌入
    text_model_output: BaseModelOutputWithPooling = None  # [`GroupViTTextModel`] 的输出，带有池化的基础模型输出
    vision_model_output: BaseModelOutputWithPooling = None  # [`GroupViTVisionModel`] 的输出，带有池化的基础模型输出

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class GroupViTPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        image_size: int = 224,  # 图像大小，默认为 224
        patch_size: Union[int, Tuple[int, int]] = 16,  # 补丁大小，默认为 16
        num_channels: int = 3,  # 图像通道数，默认为 3
        embed_dim: int = 768,  # 嵌入维度，默认为 768
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 确保 image_size 是一个可迭代对象，如果不是则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        # 确保 patch_size 是一个可迭代对象，如果不是则转换为元组
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像可以划分成的补丁数目
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 设置对象的属性：图像大小、补丁大小、补丁数目
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 创建一个卷积层用于投影，将输入图像划分为补丁，并映射到嵌入维度
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 获取输入张量的批量大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果不需要插值位置编码
        if not interpolate_pos_encoding:
            # 检查输入图像的尺寸是否与模型期望的图像尺寸相匹配
            if height != self.image_size[0] or width != self.image_size[1]:
                # 抛出值错误异常，指示输入图像尺寸不匹配模型期望的尺寸
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        # 对输入图像进行投影，然后将结果展平并转置
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # 返回处理后的张量
        return x
class GroupViTVisionEmbeddings(nn.Module):
    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()

        # 初始化图像块嵌入层，使用 GroupViTPatchEmbeddings 类
        self.patch_embeddings = GroupViTPatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        # 计算图像块的数量
        num_patches = self.patch_embeddings.num_patches
        # 初始化位置嵌入为可学习参数，维度为 (1, num_patches, hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.dropout)
        # 初始化 LayerNorm 层，对隐藏状态进行归一化
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 保存配置信息
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 获取当前图像块的数量
        npatch = embeddings.shape[1]
        # 如果当前图像块数量与位置嵌入的数量相等，并且图像的高度和宽度相等，则直接返回位置嵌入
        if npatch == self.position_embeddings.shape[1] and height == width:
            return self.position_embeddings

        # 否则，进行位置嵌入的插值操作，以适应更高分辨率的图像
        patch_pos_embed = self.position_embeddings
        num_original_pos_embed = patch_pos_embed.shape[1]
        dim = embeddings.shape[-1]
        feat_height = height // self.config.patch_size
        feat_width = width // self.config.patch_size

        # 添加一个小的数值以避免插值时的浮点误差
        feat_height, feat_width = feat_height + 0.1, feat_width + 0.1
        original_height = original_width = math.sqrt(num_original_pos_embed)

        # 将位置嵌入重塑为 (1, original_height, original_width, dim)，并调整维度顺序
        reshaped_patch_pos_embed = patch_pos_embed.reshape(1, int(original_height), int(original_width), dim).permute(
            0, 3, 1, 2
        )

        # 计算缩放因子
        scale_factor = (feat_height / original_height, feat_width / original_width)

        # 使用双三次插值方法进行位置嵌入的插值
        patch_pos_embed = nn.functional.interpolate(
            reshaped_patch_pos_embed,
            scale_factor=scale_factor,
            mode="bicubic",
            align_corners=False,
        )

        # 调整维度顺序并展平为 (1, -1, dim) 的形式
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed
    # 定义前向传播方法，接受像素值张量和是否插值位置编码的标志，返回处理后的张量
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 获取输入张量的批量大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        
        # 使用 patch_embeddings 方法将像素值张量转换为嵌入向量
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 对嵌入向量进行 LayerNorm 处理
        embeddings = self.layernorm(embeddings)

        # 获取处理后嵌入向量的批量大小和序列长度
        batch_size, seq_len, _ = embeddings.size()

        # 如果设置了插值位置编码标志，对每个 token 添加插值位置编码
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 否则，直接将预先计算的位置编码加到嵌入向量上
            embeddings = embeddings + self.position_embeddings

        # 对处理后的嵌入向量进行 Dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入向量作为输出
        return embeddings
# 从transformers.models.clip.modeling_clip.CLIPTextEmbeddings复制过来，改名为GroupViTTextEmbeddings，用于处理文本嵌入
class GroupViTTextEmbeddings(nn.Module):
    def __init__(self, config: GroupViTTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        # 初始化token_embedding，使用Embedding层，形状为(vocab_size, embed_dim)
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 初始化position_embedding，使用Embedding层，形状为(max_position_embeddings, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 创建position_ids张量，形状为(1, max_position_embeddings)，并且在序列化时被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # 前向传播函数，接受input_ids、position_ids和inputs_embeds等参数，返回嵌入张量
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 如果input_ids不为None，计算序列长度seq_length
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        # 如果position_ids为None，则使用预先创建的self.position_ids的前seq_length部分
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果inputs_embeds为None，则使用token_embedding层对input_ids进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 计算位置嵌入，使用position_embedding对position_ids进行嵌入
        position_embeddings = self.position_embedding(position_ids)
        # 计算最终的嵌入张量，将token嵌入和位置嵌入相加
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class GroupViTStage(nn.Module):
    """这对应于GroupViT实现中的`GroupingLayer`类。"""

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

        # 如果num_group_token大于0，则创建形状为(1, num_group_token, hidden_size)的可学习参数group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, config.hidden_size))
        else:
            self.group_token = None

        # 创建包含depth个GroupViTEncoderLayer层的模块列表
        self.layers = nn.ModuleList([GroupViTEncoderLayer(config) for _ in range(depth)])

        # 如果num_group_token大于0，则创建GroupViTTokenAssign层作为downsample
        if num_group_token > 0:
            self.downsample = GroupViTTokenAssign(
                config=config,
                num_group_token=num_group_token,
                num_output_group=num_output_group,
            )
        else:
            self.downsample = None

        # 如果num_prev_group_token和num_group_token都大于0，则创建group_projector作为组投影器
        if num_prev_group_token > 0 and num_group_token > 0:
            self.group_projector = nn.Sequential(
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                GroupViTMixerMLP(config, num_prev_group_token, config.hidden_size // 2, num_group_token),
            )
        else:
            self.group_projector = None

    # 返回是否存在group_token的布尔属性
    @property
    def with_group_token(self):
        return self.group_token is not None

    # 将输入张量x拆分为两部分，如果存在group_token则将最后num_group_token部分作为分组token返回
    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None
    def concat_x(self, x: torch.Tensor, group_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 如果 group_token 为 None，则直接返回输入张量 x
        if group_token is None:
            return x
        # 否则，将输入张量 x 与 group_token 沿着 dim=1 进行拼接
        return torch.cat([x, group_token], dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_group_token: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入到层的张量，形状为 `(batch, seq_len, embed_dim)`
            prev_group_token (`torch.FloatTensor`, *optional*): 上一个组令牌的张量，形状为 `(batch, 1, embed_dim)`
            output_attentions (`bool`, *optional*):
                是否返回 Grouping block 的注意力张量。
        """
        # 如果模型配置中包含 group_token
        if self.with_group_token:
            # 扩展 group_token 以匹配隐藏状态张量的批处理大小
            group_token = self.group_token.expand(hidden_states.size(0), -1, -1)
            # 如果存在组投影器，则对 group_token 进行投影
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            # 否则，将 group_token 设置为 None
            group_token = None

        # 初始化 x 为隐藏状态张量
        x = hidden_states

        # 将 x 和 group_token 拼接起来
        cat_x = self.concat_x(x, group_token)
        
        # 遍历所有层，并将拼接后的张量传入每一层
        for layer in self.layers:
            # 调用每一层的 forward 方法，并传入适当的注意力掩码
            layer_out = layer(cat_x, attention_mask=None, causal_attention_mask=None)
            # 更新 cat_x 为当前层的输出
            cat_x = layer_out[0]

        # 分离拼接后的张量 x 和 group_token
        x, group_token = self.split_x(cat_x)

        # 如果存在下采样操作，则对 x 和 group_token 进行下采样
        attention = None
        if self.downsample is not None:
            x, attention = self.downsample(x, group_token)

        # 输出结果包括 x 和 group_token
        outputs = (x, group_token)
        # 如果需要输出注意力张量，则将注意力张量添加到输出中
        if output_attentions:
            outputs = outputs + (attention,)

        # 返回最终的输出元组
        return outputs
class GroupViTMLP(nn.Module):
    def __init__(
        self,
        config: GroupViTVisionConfig,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        output_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        # 设置隐藏层大小，默认为配置中的隐藏层大小
        hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        # 设置中间层大小，默认为配置中的中间层大小
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        # 设置输出层大小，默认为隐藏层大小
        output_size = output_size if output_size is not None else hidden_size
        # 创建线性层，用于 MLP 的第一层和第二层
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，执行线性变换和激活函数
        hidden_states = self.fc1(hidden_states)  # 第一线性层
        hidden_states = self.activation_fn(hidden_states)  # 激活函数
        hidden_states = self.fc2(hidden_states)  # 第二线性层
        return hidden_states


class GroupViTMixerMLP(GroupViTMLP):
    def forward(self, x):
        # 继承 GroupViTMLP 的前向传播，对输入进行转置处理
        x = super().forward(x.transpose(1, 2))
        return x.transpose(1, 2)


class GroupViTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 检查是否可以均匀划分 embed_dim 为 num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 缩放因子，用于多头注意力机制
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        # 线性变换，用于 Q、K、V 和输出层的投影
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重塑输入张量，以便适应多头注意力的计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        # 执行多头注意力的前向传播
        # 使用 Q、K、V 线性映射
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑 Q、K、V 张量，以便适应多头注意力计算
        query_states = self._shape(query_states, -1, hidden_states.size(0))
        key_states = self._shape(key_states, -1, hidden_states.size(0))
        value_states = self._shape(value_states, -1, hidden_states.size(0))

        # 计算注意力分数
        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attn_scores = attn_scores * self.scale

        # 应用注意力掩码（如果有）
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # 计算 softmax 归一化得到注意力权重
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 应用 dropout
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # 将注意力权重乘以 V（值）张量
        attn_output = torch.matmul(attn_probs, value_states)

        # 将多头注意力的输出重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(hidden_states.size())

        # 执行最终的线性映射投影
        attn_output = self.out_proj(attn_output)

        return attn_output


class GroupViTEncoderLayer(nn.Module):
    def __init__(self, config: GroupViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = GroupViTAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = GroupViTMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)


# 初始化函数，接受一个配置对象 config，配置对象包含隐藏大小和层标准化的 epsilon 值
def __init__(self, config: GroupViTConfig):
    # 调用父类的初始化方法
    super().__init__()
    # 设置嵌入维度为配置中的隐藏大小
    self.embed_dim = config.hidden_size
    # 创建自注意力层对象，使用给定的配置
    self.self_attn = GroupViTAttention(config)
    # 创建第一个层标准化层，使用嵌入维度和配置中的 epsilon 值
    self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    # 创建 MLP（多层感知器）对象，使用给定的配置
    self.mlp = GroupViTMLP(config)
    # 创建第二个层标准化层，使用嵌入维度和配置中的 epsilon 值
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

        # 对输入的隐藏状态进行第一次层标准化
        hidden_states = self.layer_norm1(hidden_states)
        # 使用自注意力层处理标准化后的隐藏状态，同时传入注意力掩码等参数
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 加上残差连接
        hidden_states = residual + hidden_states

        residual = hidden_states
        # 对更新后的隐藏状态再次进行层标准化
        hidden_states = self.layer_norm2(hidden_states)
        # 使用 MLP 处理标准化后的隐藏状态
        hidden_states = self.mlp(hidden_states)
        # 再次加上残差连接
        hidden_states = residual + hidden_states

        # 设置输出为包含更新后的隐藏状态的元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出元组
        return outputs
# GroupViTPreTrainedModel 类，继承自 PreTrainedModel 类，用于处理权重初始化和预训练模型下载与加载的抽象类
class GroupViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类变量，指定使用 GroupViTConfig 类作为配置类
    config_class = GroupViTConfig
    # 基础模型前缀，指定为 "groupvit"
    base_model_prefix = "groupvit"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型权重的方法
    def _init_weights(self, module):
        """Initialize the weights"""

        # 获取初始化范围
        init_range = self.config.initializer_range
        # 如果模块是线性层或二维卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重，均值为 0.0，标准差为初始化范围
            module.weight.data.normal_(mean=0.0, std=init_range)
            # 如果存在偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)

        # 获取初始化因子
        factor = self.config.initializer_factor
        # 如果模块是 GroupViTTextEmbeddings 类的实例
        if isinstance(module, GroupViTTextEmbeddings):
            # 使用正态分布初始化 token_embedding 的权重，均值为 0.0，标准差为初始化因子乘以 0.02
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            # 使用正态分布初始化 position_embedding 的权重，均值为 0.0，标准差为初始化因子乘以 0.02
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        # 如果模块是 GroupViTAttention 类的实例
        elif isinstance(module, GroupViTAttention):
            # 计算输入投影的标准差
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算输出投影的标准差
            out_proj_std = (module.embed_dim**-0.5) * factor
            # 使用正态分布初始化 q_proj、k_proj、v_proj、out_proj 的权重，标准差分别为对应的标准差
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果模块是 GroupViTMLP 类的实例
        elif isinstance(module, GroupViTMLP):
            # 计算输入投影的标准差
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算全连接层的标准差
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 使用正态分布初始化 fc1、fc2 的权重，标准差分别为对应的标准差
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)


# GROUPVIT_START_DOCSTRING 文档字符串，用于描述 GroupViTPreTrainedModel 类的使用方法和参数说明
GROUPVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`GroupViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# GROUPVIT_TEXT_INPUTS_DOCSTRING 文档字符串，预留用于描述 GroupViT 模型的文本输入参数说明
GROUPVIT_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列的标记索引，用于词汇表中的标记。默认情况下会忽略填充部分。

            # 可以使用`CLIPTokenizer`获取索引。详见`PreTrainedTokenizer.encode`和`PreTrainedTokenizer.__call__`。

            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免对填充标记索引执行注意力操作的掩码。掩码值在 `[0, 1]` 范围内：

            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。

            # [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择在 `[0, config.max_position_embeddings - 1]` 范围内。

            # [什么是位置 ID？](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关详细信息，请查看返回的张量中的`attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关详细信息，请查看返回的张量中的`hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通元组。
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


class GroupViTVisionEncoder(nn.Module):
    """
    Vision encoder module for GroupViT.
    """

    def __init__(self):
        """
        Initialize the GroupViT vision encoder.
        """
        # 调用父类的初始化方法
        super().__init__()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> torch.FloatTensor:
        """
        Forward pass of the GroupViT vision encoder.

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

        Returns:
            torch.FloatTensor: Output tensor from the vision encoder.

        Notes:
            The detailed behavior of this method should be found in the GroupViT model's documentation.
        """
        # 实现 GroupViT 视觉编码器的前向传播
        pass
    # 初始化函数，接受一个 GroupViTVisionConfig 类型的参数 config
    # 调用父类的初始化方法
    def __init__(self, config: GroupViTVisionConfig) -> None:
        super().__init__()
        # 将传入的配置对象保存到实例变量中
        self.config = config
        # 创建一个 nn.ModuleList 来保存 GroupViTStage 的实例对象列表
        self.stages = nn.ModuleList(
            [
                # 使用列表推导式创建 GroupViTStage 实例对象的列表
                GroupViTStage(
                    config=config,  # 传入初始化时的 config 参数
                    depth=config.depths[i],  # 取 depths 列表中第 i 个元素作为 depth 参数
                    num_group_token=config.num_group_tokens[i],  # 取 num_group_tokens 列表中第 i 个元素作为 num_group_token 参数
                    num_output_group=config.num_output_groups[i],  # 取 num_output_groups 列表中第 i 个元素作为 num_output_group 参数
                    num_prev_group_token=config.num_output_groups[i - 1] if i > 0 else 0,  # 计算 num_prev_group_token 参数
                )
                # 遍历 config.depths 列表的索引范围
                for i in range(len(config.depths))
            ]
        )
        # 是否使用梯度检查点，默认为 False
        self.gradient_checkpointing = False

    # 前向传播函数，接受 torch.Tensor 类型的 hidden_states 参数以及几个可选的 bool 类型参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果 output_attentions 参数不为 None，则使用它；否则使用 self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数不为 None，则使用它；否则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 参数不为 None，则使用它；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果需要输出隐藏状态，则初始化 all_hidden_states 为空元组；否则设置为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力机制，则初始化 all_groupings 为空元组；否则设置为 None
        all_groupings = () if output_attentions else None

        # 初始化 group_tokens 为 None
        group_tokens = None

        # 遍历 self.stages 中的每个阶段
        for i, stage in enumerate(self.stages):
            # 如果需要输出隐藏状态，则将当前 hidden_states 加入 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前阶段的 forward 方法，计算该阶段的输出
            layer_outputs = stage(hidden_states, group_tokens, output_attentions)

            # 更新 hidden_states 和 group_tokens 为当前阶段的输出结果
            hidden_states = layer_outputs[0]
            group_tokens = layer_outputs[1]

            # 如果需要输出注意力机制，并且当前阶段的输出中包含 attentions，则将其加入 all_groupings 中
            if output_attentions and layer_outputs[2] is not None:
                all_groupings = all_groupings + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终的 hidden_states 加入 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则返回隐藏状态、隐藏状态列表和注意力机制列表的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_groupings] if v is not None)
        # 否则，返回一个 BaseModelOutput 对象，包含最终的隐藏状态、隐藏状态列表和注意力机制列表
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
        super().__init__()
        self.config = config
        # 创建包含多个 `GroupViTEncoderLayer` 的层列表，数量为 config.num_hidden_layers
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
    ):
        """
        Forward pass of the encoder.

        Args:
            inputs_embeds: Embedded input tokens.
            attention_mask: Mask to avoid attention on padding tokens.
            causal_attention_mask: Mask to apply causal masking in attention layers.
            output_attentions: Whether to output attentions weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary instead of a tuple.

        Returns:
            BaseModelOutputWithPooling: Output with pooled representation and optionally attentions and hidden states.
        """
        # Implementation details of the forward pass are in the actual GroupViTTextEncoderLayer implementation.
        pass


# Copied from transformers.models.clip.modeling_clip.CLIPTextTransformer with CLIPText->GroupViTText, CLIPEncoder->GroupViTTextEncoder, CLIP_TEXT->GROUPVIT_TEXT
class GroupViTTextTransformer(nn.Module):
    def __init__(self, config: GroupViTTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        # 初始化 GroupViTTextEmbeddings，用于输入的嵌入表示
        self.embeddings = GroupViTTextEmbeddings(config)
        # 初始化 GroupViTTextEncoder，用于编码输入序列
        self.encoder = GroupViTTextEncoder(config)
        # 对最终输出进行 LayerNorm 处理
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
    ):
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Mask to avoid attention on padding tokens.
            position_ids: IDs indicating the position of each token in the sequence.
            output_attentions: Whether to output attentions weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary instead of a tuple.

        Returns:
            BaseModelOutputWithPooling: Output with pooled representation and optionally attentions and hidden states.
        """
        # Implementation details of the forward pass are in the actual GroupViTTextModel implementation.
        pass


class GroupViTTextModel(GroupViTPreTrainedModel):
    config_class = GroupViTTextConfig

    def __init__(self, config: GroupViTTextConfig):
        super().__init__(config)
        # 初始化 GroupViTTextTransformer 作为文本模型的主体
        self.text_model = GroupViTTextTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

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
    ):
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs.
            attention_mask: Mask to avoid attention on padding tokens.
            position_ids: IDs indicating the position of each token in the sequence.
            output_attentions: Whether to output attentions weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary instead of a tuple.

        Returns:
            BaseModelOutputWithPooling: Output with pooled representation and optionally attentions and hidden states.
        """
        # Implementation details of the forward pass are in the actual GroupViTTextModel implementation.
        pass
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
        此方法定义了模型的前向传播过程，接受多个输入参数并返回模型输出或元组。

        Returns:
            模型的输出或元组，包含了模型的不同部分或汇总结果。

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer, GroupViTTextModel

        >>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
        >>> model = GroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```
        """
        调用文本模型的前向传播，传递各种输入参数以控制模型行为和输出。
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

        # 初始化嵌入层
        self.embeddings = GroupViTVisionEmbeddings(config)
        # 初始化编码器
        self.encoder = GroupViTVisionEncoder(config)
        # 初始化 LayerNorm 层
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
        模型的前向传播方法

        返回:
            BaseModelOutputWithPooling 或 Tuple

        """
        # 确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，引发数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 嵌入层
        hidden_states = self.embeddings(pixel_values)

        # 编码器的输出
        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏状态并进行归一化
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.layernorm(last_hidden_state)
        pooled_output = last_hidden_state.mean(dim=1)

        # 如果不使用返回字典，返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 使用 BaseModelOutputWithPooling 类封装输出
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
        # 初始化视觉模型
        self.vision_model = GroupViTVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> GroupViTPatchEmbeddings:
        # 返回嵌入层的补丁嵌入对象
        return self.vision_model.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(GROUPVIT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=GroupViTVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        此方法用于模型的前向推断，接受多个参数并返回一个包含输出的对象。

        Returns:
            返回一个包含模型输出的对象，通常包括最后一层隐藏状态和池化后的输出。

        Examples:
        以下是一些使用示例，展示了如何使用该方法进行图像特征提取和推断。

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
        >>> last_hidden_state = outputs.last_hidden_state  # 获取最后一层的隐藏状态
        >>> pooled_output = outputs.pooler_output  # 获取经过池化的输出（通常是CLS状态）
        ```
        """
        调用视觉模型的前向方法，传递参数并返回结果。
        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
# 使用装饰器为 GroupViTModel 类添加文档字符串，用于模型初始化和使用说明
@add_start_docstrings(GROUPVIT_START_DOCSTRING)
class GroupViTModel(GroupViTPreTrainedModel):
    # 设置 config_class 属性为 GroupViTConfig 类
    config_class = GroupViTConfig

    # 初始化方法，接受一个 GroupViTConfig 类型的参数 config
    def __init__(self, config: GroupViTConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 检查 config.text_config 是否为 GroupViTTextConfig 类型，若不是则抛出异常
        if not isinstance(config.text_config, GroupViTTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type GroupViTTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.vision_config 是否为 GroupViTVisionConfig 类型，若不是则抛出异常
        if not isinstance(config.vision_config, GroupViTVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type GroupViTVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 从 config 中获取 text_config 和 vision_config 对象
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置模型的投影维度和投影中间维度
        self.projection_dim = config.projection_dim
        self.projection_intermediate_dim = config.projection_intermediate_dim
        # 设置文本嵌入维度和视觉嵌入维度
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建 GroupViTTextTransformer 和 GroupViTVisionTransformer 模型
        self.text_model = GroupViTTextTransformer(text_config)
        self.vision_model = GroupViTVisionTransformer(vision_config)

        # 定义视觉特征的投影层
        self.visual_projection = nn.Sequential(
            nn.Linear(self.vision_embed_dim, self.projection_intermediate_dim, bias=True),
            nn.BatchNorm1d(self.projection_intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True),
        )
        # 定义文本特征的投影层
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.projection_intermediate_dim, bias=True),
            nn.BatchNorm1d(self.projection_intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_intermediate_dim, self.projection_dim, bias=True),
        )
        # 创建一个可学习的 logit_scale 参数
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 执行初始化权重和最终处理
        self.post_init()

    # 使用装饰器为 get_text_features 方法添加文档字符串，指定模型前向传播时输入的文本相关参数
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
        # Check if output_hidden_states is specified; otherwise, use the config's value.
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Determine if return_dict is explicitly set; if not, use the config's default setting.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve text model outputs with specified inputs and optional settings.
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the pooled output from the text model's outputs.
        pooled_output = text_outputs[1]
        # Project the pooled output to obtain text features.
        text_features = self.text_projection(pooled_output)

        # Return the computed text features.
        return text_features
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
        # 如果指定了输出注意力机制的配置，则使用该配置；否则使用 GROUPVIT 模型默认配置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果指定了输出隐藏状态的配置，则使用该配置；否则使用 GROUPVIT 模型默认配置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果指定了返回字典的配置，则使用该配置；否则使用 GROUPVIT 模型默认配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型（GroupViTVisionModel）来获取视觉特征表示
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从视觉输出中获取池化后的特征表示
        pooled_output = vision_outputs[1]  # pooled_output
        # 将池化后的特征表示投影到视觉投影层，得到最终的图像特征表示
        image_features = self.visual_projection(pooled_output)

        # 返回图像特征表示作为模型的输出
        return image_features
```