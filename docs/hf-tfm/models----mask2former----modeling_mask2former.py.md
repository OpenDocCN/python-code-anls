# `.\transformers\models\mask2former\modeling_mask2former.py`

```py
# 设置文件编码格式为 UTF-8
# 版权声明，所有权归 Meta Platforms, Inc. 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本 (以下简称“许可证”) 进行许可;
# 除非符合许可证的规定，否则您不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或以书面形式同意，否则不得根据许可证分发软件
# 根据许可证以“原样”分发，不提供任何担保或条件，无论是明示的还是暗示的
# 请参阅许可证了解具体规定和限制
""" PyTorch Mask2Former model."""

# 导入需要的库
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ... import AutoBackbone
from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    replace_return_docstrings,
    requires_backends,
)
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_mask2former import Mask2FormerConfig

# 如果已经安装了 scipy 库，则导入 linear_sum_assignment 方法
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 日志记录器
logger = logging.get_logger(__name__)

# 用于文档的常量
_CONFIG_FOR_DOC = "Mask2FormerConfig"
_CHECKPOINT_FOR_DOC = "facebook/mask2former-swin-small-coco-instance"
_IMAGE_PROCESSOR_FOR_DOC = "Mask2FormerImageProcessor"

# 预训练模型的列表
MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mask2former-swin-small-coco-instance",
    # 查看所有 mask2former 模型 https://huggingface.co/models?filter=mask2former
]

# Mask2Former 像素解码器的模型输出类
@dataclass
class Mask2FormerPixelDecoderOutput(ModelOutput):
    """
    Mask2Former's pixel decoder module output, practically a Multi-Scale Deformable Attention based decoder. It returns
    the mask features and the multiscale features.

    Args:
        multi_scale_features (`tuple(torch.FloatTensor)`):
            Tuple of multi-scale features of scales [1/8, 1/16, 1/32] and shape `(batch_size, num_channels, height,
            width)`from the Multi-Scale Deformable Attenntion based Pixel Decoder.
        mask_features (`torch.FloatTensor`):
            Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel Decoder
            Layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from pixel decoder. Returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`
    """

    multi_scale_features: Tuple[torch.FloatTensor] = None
    mask_features: torch.FloatTensor = None
    # 定义一个名为attentions的可选类型变量，其类型为包含torch.FloatTensor的元组，初始赋值为None
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import BaseModelOutputWithCrossAttentions, ModelOutput
import torch

@dataclass
class Mask2FormerMaskedAttentionDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the Transformer decoder. This class adds two attributes to
    BaseModelOutputWithCrossAttentions for mask predictions logits and a tuple of intermediate decoder activations,
    i.e. the output of each decoder layer, each of them gone through a layernorm.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs. Returned when `output_hidden_states=True`.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads. Returned when `output_attentions=True`.
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`):
            Tuple of mask predictions from all layers of the transformer decoder.
        intermediate_hidden_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    intermediate_hidden_states: Tuple[torch.FloatTensor] = None


@dataclass
class Mask2FormerPixelLevelModuleOutput(ModelOutput):
    """
    Mask2Former's pixel level module output. It returns the output of the encoder (optional) and all hidden states
    (multi-scale features) from the `decoder`. By default, the `encoder` is a Swin Backbone and the `decoder` is a
    Multi-Scale Deformable Attention based decoder.

    The `decoder_last_hidden_state` are the **per-pixel embeddings** while `decoder_hidden_states` refer to multi-scale
    feature maps produced using **multi-scaling strategy** defined in the paper.


    """
    # 定义函数参数，最后一个编码器阶段的最后隐藏状态，形状为(batch_size, num_channels, height, width)
    encoder_last_hidden_state: torch.FloatTensor = None
    # 定义函数参数，编码器在每个阶段输出的隐藏状态（也称为特征图）的元组，形状为(batch_size, num_channels, height, width)，如果output_hidden_states设置为True，则返回
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义函数参数，解码器最后一个像素解码器层的1/4比例特征
    decoder_last_hidden_state: torch.FloatTensor = None
    # 定义函数参数，解码器在每个阶段输出的隐藏状态（也称为特征图）的元组，形状为(batch_size, num_channels, height, width)
    decoder_hidden_states: Tuple[torch.FloatTensor] = None
# 定义一个名为 Mask2FormerModelOutput 的数据类，继承自 ModelOutput 类
class Mask2FormerModelOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            # 编码器模型最后一个隐藏状态（最终特征图），形状为 `(batch_size, num_channels, height, width)`，可选。
            Last hidden states (final feature map) of the last stage of the encoder model (backbone). Returned when
            `output_hidden_states=True` is passed.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            # 编码器模型的隐藏状态元组（嵌入层输出 + 每个阶段的输出），形状为 `(batch_size, num_channels, height, width)`，可选。
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            # 像素解码器模型最后一个隐藏状态（最终特征图），形状为 `(batch_size, num_channels, height, width)`，可选。
            Last hidden states (final feature map) of the last stage of the pixel decoder model.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, , *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            # 像素解码器模型的隐藏状态元组（嵌入层输出 + 每个阶段的输出），形状为 `(batch_size, num_channels, height, width)`，可选。
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_last_hidden_state (`tuple(torch.FloatTensor)`):
            # 变换器解码器的最后一个隐藏状态 (`batch_size, sequence_length, hidden_size`)
            Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            # 变换器解码器的隐藏状态元组（嵌入层输出 + 每个阶段的输出），形状为 `(batch_size, sequence_length, hidden_size)`，可���。
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_intermediate_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            # 变换器解码器的中间输出（每个解码器层的输出），经过 layernorm 处理。
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`)
            # 变换器解码器中每层的蒙版预测。
            Mask Predictions from each layer in the transformer decoder.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed):
            # 变换器解码器的自注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，可选。
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self attentions weights from transformer decoder.
    # 定义一个名为encoder_last_hidden_state的变量，并初始化为None，类型为torch.FloatTensor
    encoder_last_hidden_state: torch.FloatTensor = None
    # 定义一个名为pixel_decoder_last_hidden_state的变量，并初始化为None，类型为torch.FloatTensor
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    # 定义一个名为transformer_decoder_last_hidden_state的变量，并初始化为None，类型为torch.FloatTensor
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    # 定义一个名为encoder_hidden_states的变量，并初始化为None，类型为Optional[Tuple[torch.FloatTensor]]
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个名为pixel_decoder_hidden_states的变量，并初始化为None，类型为Optional[Tuple[torch.FloatTensor]]
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个名为transformer_decoder_hidden_states的变量，并初始化为None，类型为Optional[Tuple[torch.FloatTensor]]
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个名为transformer_decoder_intermediate_states的变量，并初始化为None, 类型为Tuple[torch.FloatTensor]
    transformer_decoder_intermediate_states: Tuple[torch.FloatTensor] = None
    # 定义一个名为masks_queries_logits的变量，并初始化为None，类型为Tuple[torch.FloatTensor]
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    # 定义一个名为attentions的变量，并初始化为None，类型为Optional[Tuple[torch.FloatTensor]]
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器创建 Mask2FormerForUniversalSegmentationOutput 类
@dataclass
class Mask2FormerForUniversalSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerForUniversalSegmentationOutput`].

    This output can be directly passed to [`~Mask2FormerImageProcessor.post_process_semantic_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_instance_segmentation`] or
    [`~Mask2FormerImageProcessor.post_process_panoptic_segmentation`] to compute final segmentation maps. Please, see
    [`~Mask2FormerImageProcessor] for details regarding usage.

    """

    # 损失值，可选的浮点数张量
    loss: Optional[torch.FloatTensor] = None
    # 类别查询的预测对数，用于模型输出的预测
    class_queries_logits: torch.FloatTensor = None
    # 掩码查询的预测对数，用于模型输出的预测
    masks_queries_logits: torch.FloatTensor = None
    # 辅助预测值列表
    auxiliary_logits: Optional[List[Dict[str, torch.FloatTensor]]] = None
    # 编码器最后隐藏状态
    encoder_last_hidden_state: torch.FloatTensor = None
    # 像素解码器最后隐藏状态
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    # 特征变换器最后隐藏状态
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    # 编码器隐藏状态的元组，可选
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 像素解码器隐藏状态的元组，可选
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 特征变换器隐藏状态，可选
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    # 注意力的元组，可选
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 改编自 https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False, **kwargs
) -> torch.Tensor:
    """
    A wrapper around `torch.nn.functional.grid_sample` to support 3D point_coordinates tensors.

    Args:
        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):
            A tensor that contains features map on a height * width grid
        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,:
        2)):
            A tensor that contains [0, 1] * [0, 1] normalized point coordinates
        add_dim (`bool`):
            boolean value to keep track of added dimension

    Returns:
        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid):
            A tensor that contains features for points in `point_coordinates`.
    """
    # 如果坐标点的维度是3，则表示有额外的维度，将 add_dim 设置为 True，并添加一个维度
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # 使用 nn.function.grid_sample 获取经过双线性插值的 `point_coordinates` 点的特征
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    # 如果有额外的维度，去掉这个维度
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features


# 从 transformers.models.maskformer.modeling_maskformer.dice_loss 中复制
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    计算 DICE 损失，与掩码的广义 IOU 类似：
    # 输入为张量inputs和标签labels，以及掩码的数量num_masks
    """
    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    In practice, since `labels` is a binary mask, (only 0s and 1s), dice can be computed as follow

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).
        num_masks (`int`):
            The number of masks present in the current batch, used for normalization.

    Returns:
        `torch.Tensor`: The computed loss.
    """
    # 计算输入数据的概率值，并将其扁平化为一维张量
    probs = inputs.sigmoid().flatten(1)
    # 计算分子：2 * (概率 * 标签)的和
    numerator = 2 * (probs * labels).sum(-1)
    # 计算分母：概率的和 + 标签的和
    denominator = probs.sum(-1) + labels.sum(-1)
    # 计算损失：1 - (分子 + 1) / (分母 + 1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 对每个样本的损失求和并除以样本数
    loss = loss.sum() / num_masks
    # 返回计算得到的损失
    return loss
# 定义函数sigmoid_cross_entropy_loss，计算输入和标签之间的sigmoid交叉熵损失
def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    # 创建一个基于输入和标签的BCEWithLogitsLoss损失函数
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    # 计算交叉熵损失
    cross_entropy_loss = criterion(inputs, labels)

    # 计算平均损失，并按照num_masks进行归一化
    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss


# 从transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss中复制函数
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    一种配对版本的Dice损失，使用时请参考`dice_loss`。

    Args:
        inputs (`torch.Tensor`):
            代表掩码的张量
        labels (`torch.Tensor`):
            与输入形状相同的张量。存储每个输入元素的二元分类标签(0代表负类，1代表正类)。

    Returns:
        `torch.Tensor`: 每对之间的计算损失。
    """
    # 对输入进行sigmoid处理，并平坦化为1维
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # 使用广播获取一个[num_queries, NUM_CLASSES]矩阵
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


# 定义函数pair_wise_sigmoid_cross_entropy_loss，计算输入和标签之间的交叉熵损失
def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    一种配对版本的交叉熵损失，使用时请参考`sigmoid_cross_entropy_loss`。

    Args:
        inputs (`torch.Tensor`):
            代表掩码的张量。
        labels (`torch.Tensor`):
            与输入形状相同的张量。存储每个输入元素的二元分类标签(0代表负类，1代表正类)。

    Returns:
        loss (`torch.Tensor`): 每对之间的计算损失。
    """

    height_and_width = inputs.shape[1]

    # 创建一个基于输入和标签的BCEWithLogitsLoss损失函数
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    loss_pos = torch.matmul(cross_entropy_loss_pos, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg, (1 - labels).T)
    loss = loss_pos + loss_neg
    loss = loss / height_and_width
    return loss


# 从https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py进行了调整
class Mask2FormerHungarianMatcher(nn.Module):
    """该类计算网络��标签和预测之间的赋值关系。
    # 由于效率原因，标签不包括没有对象的情况。因此，通常预测数量大于标签数量。在这种情况下，我们对最佳预测进行一对一匹配，而其他的不匹配（因此被视为非对象）。

    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544
    ):
        """创建匹配器

        参数:
            cost_class (`float`, *可选*, 默认为 1.0):
                匹配成本中分类错误的相对权重。
            cost_mask (`float`, *可选*, 默认为 1.0):
                这是二进制掩码的焦点损失在匹配成本中的相对权重。
            cost_dice (`float`, *可选*, 默认为 1.0):
                这是二值掩码的骰子损失在匹配成本中的相对权重。
            num_points (`int`, *可选*, 默认为 12544):
                用于在其上计算掩码损失的点数。对于所有预测和地面实况掩码均匀采样相同一组 K 点，以构建二分匹配的成本矩阵。
        """
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            # 如果所有成本为 0，则引发值错误
            raise ValueError("所有成本不能为 0")

        # 分配初始值
        self.num_points = num_points
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        class_labels: torch.Tensor,
# 导入需要的库
# 定义 Mask2FormerLoss 类，继承自 nn.Module
class Mask2FormerLoss(nn.Module):
    # 初始化函数，接受配置和权重字典作为参数
    def __init__(self, config: Mask2FormerConfig, weight_dict: Dict[str, float]):
        """
        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we
        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair
        of matched ground-truth / prediction (supervise class and mask)

        Args:
            config (`Mask2FormerConfig`):
                The configuration for Mask2Former model also containing loss calculation specific parameters.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
        """
        # 调用父类初始化函数
        super().__init__()
        # 检查并导入所需后端库
        requires_backends(self, ["scipy"])
        # 初始化属性
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

        # Weight to apply to the null class
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        # 将空类权重注册到 buffer
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio

        # 初始化 Matcher 对象
        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=1.0,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=self.num_points,
        )

    # 辅助函数，计算列表中各维度的最大值
    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # 从原始实现中修改而来，将传入的张量列表按批次填充到相同大小
    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # 获取批次中的最大尺寸
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        # 计算批次最终大小
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        # 创建零张量作为填充后的张量
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # 遍历张量，将其填充到最大尺寸
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks
```  
    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ) -> Dict[str, Tensor]:
        """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """
        # Store predicted logits
        pred_logits = class_queries_logits
        # Extract batch size and number of queries
        batch_size, num_queries, _ = pred_logits.shape
        # Define CrossEntropyLoss criterion with specified weight
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        # Get the indices for predicted permutation
        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)
        # Concatenate target classes for each query
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )  # shape of (batch_size, num_queries)
        # Initialize target classes tensor with a default value
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        # Fill target classes tensor with the target classes obtained from the indices
        target_classes[idx] = target_classes_o
        # Transpose predicted logits for compatibility with target classes tensor
        pred_logits_transposed = pred_logits.transpose(1, 2)
        # Compute cross entropy loss
        loss_ce = criterion(pred_logits_transposed, target_classes)
        # Store computed losses
        losses = {"loss_cross_entropy": loss_ce}
        # Return the computed losses
        return losses

    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: Tuple[np.array],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """计算与掩码相关的损失，使用 sigmoid_cross_entropy_loss 和 dice loss。

        Args:
            masks_queries_logits (`torch.Tensor`):
                形状为 `(batch_size, num_queries, height, width)` 的张量。
            mask_labels (`torch.Tensor`):
                形状为 `(labels, height, width)` 的掩码标签列表。
            indices (`Tuple[np.array])`:
                由匈牙利匹配器计算得到的索引。
            num_masks (`int)`:
                掩码的数量，用于归一化。

        Returns:
            losses (`Dict[str, Tensor]`): 包含两个键的 `torch.Tensor` 字典:
            - **loss_mask** -- 使用预测和真实掩码之间的 sigmoid 交叉熵损失计算的损失。
            - **loss_dice** -- 使用预测和真实掩码之间的 dice 损失计算的损失。
        """
        # 获取预测排列的索引
        src_idx = self._get_predictions_permutation_indices(indices)
        # 获取目标排列的索引
        tgt_idx = self._get_targets_permutation_indices(indices)
        # 形状为 (batch_size * num_queries, height, width) 的预测掩码
        pred_masks = masks_queries_logits[src_idx]
        # 形状为 (batch_size, num_queries, height, width) 的目标掩码
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        # 不需要对预测进行上采样，因为我们使用了归一化坐标
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        # 采样点坐标
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                lambda logits: self.calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

        # 计算损失
        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # 根据索引重新排列预测
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices
    # 获取目标排列的索引
    def _get_targets_permutation_indices(self, indices):
        # 按照给定的索引对标签进行排列
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    # 计算不确定性
    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        在 Mask2Former 论文中，不确定性被定义为 'logits' 中前景类别的预测与 0.0 之间的 L1 距离。

        Args:
            logits (`torch.Tensor`):
            一个形状为 (R, 1, ...) 的张量，对于特定类别或类别不可知，其中 R 是所有图像中预测的掩模总数，C 是：
            前景类别的数量。值为对数。

        Returns:
            scores (`torch.Tensor`): 一个形状为 (R, 1, ...) 的张量，其中包含不确定性分数，最不确定的位置具有最高的不确定性分数。
        """
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores

    # 使用不确定性采样点
    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_labels: List[torch.Tensor],
        auxiliary_predictions: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.Tensor:
        """
        This function defines the forward pass of the model. It takes input logits for mask and class queries, along with
        their corresponding labels, and optional auxiliary predictions. It samples points based on uncertainty and returns
        the coordinates of sampled points.

        Args:
            masks_queries_logits (`torch.Tensor`):
                Logit predictions for mask queries.
            class_queries_logits (`torch.Tensor`):
                Logit predictions for class queries.
            mask_labels (`List[torch.Tensor]`):
                List of mask labels.
            class_labels (`List[torch.Tensor]`):
                List of class labels.
            auxiliary_predictions (`Optional[Dict[str, torch.Tensor]]`):
                Optional dictionary of auxiliary predictions.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for sampled points.
        """

        num_boxes = masks_queries_logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        # Get sampled prediction value for the point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate the uncertainties based on the sampled prediction values of the points
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        # Select uncertain points based on uncertainties
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        # Add random points if needed
        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        return point_coordinates
    ) -> Dict[str, torch.Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, num_labels)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from
                the inner layers of the Mask2FormerMaskedAttentionDecoder.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], the dictionary contains additional
            losses for each auxiliary predictions.
        """

        # retrieve the matching between the outputs of the last layer and the labels
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        # compute the average number of target masks for normalization purposes
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        # get all the losses
        losses: Dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        # in case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs["masks_queries_logits"]
                class_queries_logits = aux_outputs["class_queries_logits"]
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f"{key}_{idx}": value for key, value in loss_dict.items()}
                losses.update(loss_dict)

        return losses
    # 定义一个方法，用于计算批次中目标掩模的平均数量，以便进行归一化处理
    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        # 计算批次中目标掩模的总数量
        num_masks = sum([len(classes) for classes in class_labels])
        # 将目标掩模数量转换为 PyTorch 张量，并指定数据类型和设备
        num_masks_pt = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        # 返回目标掩模数量的 PyTorch 张量
        return num_masks_pt
# 从transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention中复制的多尺度可变形注意力函数
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    # 获取输入value的形状信息
    batch_size, _, num_heads, hidden_dim = value.shape
    # 获取采样位置的形状信息
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # 将value按照空间形状分割成列表
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    # 计算采样网格
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # 将value_list展平并转置，得到value_l_
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # 将采样网格展平并转置，得到sampling_grid_l_
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # 使用双线性插值进行采样，得到sampling_value_l_
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # 调整注意力权重的形状
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    # 计算输出
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# 从transformers.models.maskformer.modeling_maskformer.MaskFormerSinePositionEmbedding复制的Mask2FormerSinePositionEmbedding类
class Mask2FormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        # 初始化函数，设置位置编码的参数
    # 初始化函数，继承父类的初始化方法
    def __init__(
        super().__init__()
        # 如果 scale 不为 None 且 normalize 为 False，则抛出数值错误
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 设置实例变量
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    # 前向传播函数
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # 如果 mask 为 None，则创建全零的 mask 张量
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        # 取反操作，得到非 mask 部分的张量
        not_mask = (~mask).to(x.dtype)
        # 沿着第一个维度进行累加操作，得到 y 坐标
        y_embed = not_mask.cumsum(1)
        # 沿着第二个维度进行累加操作，得到 x 坐标
        x_embed = not_mask.cumsum(2)
        # 如果需要归一化
        if self.normalize:
            eps = 1e-6
            # 对 y 坐标进行归一化
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            # 对 x 坐标进行归一化
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 创建维度张量
        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        # 计算温度值
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        # 计算 x 坐标的位置编码
        pos_x = x_embed[:, :, :, None] / dim_t
        # 计算 y 坐标的位置编码
        pos_y = y_embed[:, :, :, None] / dim_t
        # 对 x 坐标进行 sin 和 cos 处理，然后拼接
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 对 y 坐标进行 sin 和 cos 处理，然后拼接
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 拼接 x 和 y 坐标，然后进行维度变换
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # 返回位置编码结果
        return pos
# 定义一个名为 Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention 的类，继承自 nn.Module
class Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    # 初始化函数，接受 embed_dim（嵌入维度）、num_heads（注意力头数）、n_levels（层级数）、n_points（采样点数）作为参数
    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        super().__init__()
        # 如果 embed_dim 不能被 num_heads 整除，则抛出 ValueError 异常
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        # 计算每个注意力头的维度
        dim_per_head = embed_dim // num_heads
        # 检查 dim_per_head 是否是 2 的幂
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            # 如果不是 2 的幂，则发出警告
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        # 设置 im2col_step 为 128
        self.im2col_step = 128

        # 初始化类的属性
        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        # 初始化线性层，用于生成采样偏移量
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        # 初始化线性层，用于生成注意力权重
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        # 初始化线性层，用于投影值
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # 初始化线性层，用于最终输出投影
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    # 将位置嵌入添加到张量中
    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    # 前向传播函数，接受隐藏状态、注意力掩码、编码器隐藏状态��编码器注意力掩码、位置嵌入、参考点、空间形状、层级起始索引、是否输出注意力权重等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings: Optional[torch.Tensor] = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
        # 如果位置编码不为空，则将位置编码添加到隐藏状态中，然后再投影到查询和键
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # 获取隐藏状态的批量大小、查询数量和特征维度
        batch_size, num_queries, _ = hidden_states.shape
        # 获取编码器隐藏状态的批量大小、序列长度和特征维度
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        # 检查空间形状与编码器隐藏状态序列长度是否对齐
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        # 对编码器隐藏状态进行值投影
        value = self.value_proj(encoder_hidden_states)
        # 如果存在注意力遮罩，则将其应用到值中
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        # 获取采样偏移
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        # 获取注意力权重
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        # 对注意力权重进行 softmax 归一化
        attention_weights = nn.functional.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        
        # 如果参考点的最后一个维度为2
        if reference_points.shape[-1] == 2:
            # 计算偏移标准化器
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            # 计算采样位置
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        # 如果参考点的最后一个维度为4
        elif reference_points.shape[-1] == 4:
            # 计算采样位置
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        # 使用多尺度可变形注意力计算输出
        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        # 对输出进行投影
        output = self.output_proj(output)

        # 返回输出和注意力权重
        return output, attention_weights
# 定义一个Mask2FormerPixelDecoderEncoderLayer类，继承自nn.Module
class Mask2FormerPixelDecoderEncoderLayer(nn.Module):
    # 初始化函数，接受一个Mask2FormerConfig类型的config参数
    def __init__(self, config: Mask2FormerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置embed_dim为config中的feature_size
        self.embed_dim = config.feature_size
        # 创建一个Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention对象作为self_attn属性
        self.self_attn = Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            n_levels=3,
            n_points=4,
        )

        # 初始化self_attn_layer_norm为一个LayerNorm对象，参数为embed_dim
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置dropout为config中的dropout
        self.dropout = config.dropout
        # 设置activation_fn为relu函数
        self.activation_fn = nn.functional.relu
        # 设置activation_dropout为config中的dropout
        self.activation_dropout = config.dropout
        # 初始化fc1为一个Linear对象，输入维度为embed_dim，输出维度为config中的encoder_feedforward_dim
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        # 初始化fc2为一个Linear对象，输入维度为config中的encoder_feedforward_dim，输出维度为embed_dim
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        # 初始化final_layer_norm为一个LayerNorm对象，参数为embed_dim
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，接受多个参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                输入层的输入。
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                注意力掩码。
            position_embeddings (`torch.FloatTensor`, *optional*):
                位置嵌入，将添加到 `hidden_states` 中。
            reference_points (`torch.FloatTensor`, *optional*):
                参考点。
            spatial_shapes (`torch.LongTensor`, *optional*):
                主干特征图的空间形状。
            level_start_index (`torch.LongTensor`, *optional*):
                级别起始索引。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量中的 `attentions`。
        """
        residual = hidden_states

        # 在多尺度特征图上应用多尺度可变形注意力模块。
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights.transpose(1, 0),)

        return outputs
# 从transformers.models.detr.modeling_deformable_detr.DeformableDetrEncoder修改为Mask2FormerPixelDecoderEncoderOnly类
class Mask2FormerPixelDecoderEncoderOnly(nn.Module):
    """
    Transformer编码器，由config.encoder_layers个可变形注意力层组成。每一层都是一个Mask2FormerPixelDecoderEncoderLayer。编码器通过多个可变形注意力层更新了多尺度特征图。

    Args:
        config: Mask2FormerConfig
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()

        self.config = config
        self.dropout = config.dropout
        self.layers = nn.ModuleList(
            [Mask2FormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)]
        )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        获取每个特征图的参考点。在解码器中使用。

        Args:
            spatial_shapes (`torch.LongTensor`):
                每个特征图的空间形状，形状为`(num_feature_levels, 2)`。
            valid_ratios (`torch.FloatTensor`):
                每个特征图的有效比率，形状为`(batch_size, num_feature_levels, 2)`。
            device (`torch.device`):
                创建张量的设备。
        Returns:
            `torch.FloatTensor`，形状为`(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
# 从transformers.models.detr.modeling_deformable_detr.DeformableDetrModel修改为Mask2FormerPixelDecoder类
class Mask2FormerPixelDecoder(nn.Module):
    # 初始化函数，接受配置和特征通道作为参数
    def __init__(self, config: Mask2FormerConfig, feature_channels):
        # 调用父类的初始化函数
        super().__init__()

        # 保存配置信息
        self.config = config

        # 获取特征维度和掩码特征维度
        feature_dim = config.feature_size
        mask_dim = config.mask_feature_size
        num_pos_features = feature_dim // 2

        # 创建位置编码对象
        self.position_embedding = Mask2FormerSinePositionEmbedding(num_pos_feats=num_pos_features, normalize=True)
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels :]

        # 保存特征级别的步长和通道信息
        self.transformer_feature_strides = config.feature_strides[-self.num_feature_levels :]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, feature_dim))

        # 创建输入投影层
        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                )
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            self.input_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(transformer_in_channels[-1], feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                ]
            )

        # 创建编码器对象和掩码投影层
        self.encoder = Mask2FormerPixelDecoderEncoderOnly(config)
        self.mask_projection = nn.Conv2d(feature_dim, mask_dim, kernel_size=1, stride=1, padding=0)

        # 额外的 FPN 级别
        stride = min(self.transformer_feature_strides)
        self.common_stride = config.common_stride
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        # 创建侧边卷积和输出卷积层
        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, feature_dim),
            )

            output_conv = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(),
            )
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # 按照分辨率从低到高的顺序排列卷积层
        self.lateral_convolutions = lateral_convs[::-1]
        self.output_convolutions = output_convs[::-1]
    def get_valid_ratio(self, mask, dtype=torch.float32):
        """计算所有特征图的有效比例"""

        # 获取mask的形状信息
        _, height, width = mask.shape
        # 计算每一行有效像素点的数量
        valid_height = torch.sum(~mask[:, :, 0], 1)
        # 计算每一列有效像素点的数量
        valid_width = torch.sum(~mask[:, 0, :], 1)
        # 计算高度方向的有效比例
        valid_ratio_heigth = valid_height.to(dtype) / height
        # 计算宽度方向的有效比例
        valid_ratio_width = valid_width.to(dtype) / width
        # 将高度和宽度方向的有效比例合并成一个张量
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def forward(
        self,
        features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
class Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, config: Mask2FormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`Mask2FormerConfig`]):
                The configuration used to instantiate this model.
        """
        # 初始化函数，创建一个新的 Mask2FormerPixelLevelModule 类实例
        super().__init__()

        # 使用配置中的参数实例化一个自动选择的骨干网络
        self.encoder = AutoBackbone.from_config(config.backbone_config)
        # 使用配置和骨干网络的通道数实例化一个 Mask2FormerPixelDecoder 对象
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        # 将输入像素值通过骨干网络得到特征图
        backbone_features = self.encoder(pixel_values).feature_maps
        # 将特征图输入到解码器中，得到解码器的输出
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

        # 返回 Mask2FormerPixelLevelModuleOutput 对象，包含编码器和解码器的隐藏状态
        return Mask2FormerPixelLevelModuleOutput(
            encoder_last_hidden_state=backbone_features[-1],
            encoder_hidden_states=tuple(backbone_features) if output_hidden_states else None,
            decoder_last_hidden_state=decoder_output.mask_features,
            decoder_hidden_states=decoder_output.multi_scale_features,
        )


# Modified from transformers.models.detr.modeling_detr.DetrAttention with Detr->Mask2Former
class Mask2FormerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Here, we add position embeddings to the queries and
    keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        # 初始化函数，创建一个新的 Mask2FormerAttention 类实例
        super().__init__()
        # 初始化注意力模型的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        # 初始化线性变换层，用于将输入进行线性变换
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        # 重塑张量的形状，用于多头注意力计算
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        # 将位置嵌入加到输入张量中，用于位置编码
        return tensor if position_embeddings is None else tensor + position_embeddings
    # 定义一个前向传播函数，接受隐藏状态、注意力掩码、位置嵌入、键值状态、键值位置嵌入和是否输出注意力权重作为输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量，默认为None
        position_embeddings: Optional[torch.Tensor] = None,  # 可选的位置嵌入张量，默认为None
        key_value_states: Optional[torch.Tensor] = None,  # 可选的键值状态张量，默认为None
        key_value_position_embeddings: Optional[torch.Tensor] = None,  # 可选的键值位置嵌入张量，默认为None
        output_attentions: bool = False,  # 是否输出注意力权重，默认为False
class Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    """
    The Mask2FormerMaskedAttentionDecoderLayer is made up of self-attention, cross (masked) attention as well as FFN
    blocks. The cross attention block used as part of `Mask2FormerMaskedAttentionDecoderLayer` is actually a `masked
    attention` block that restricts the attention to localized features centered around predicted segments which leads
    to faster convergence and improved performance. The order of self and cross (i.e. masked) attention blocks have
    also been swapped in Mask2FormerMaskedAttentionDecoder compared to a standard DetrDecoder as an optimization
    improvement.

    Args:
        config (`Mask2FormerConfig`):
            The configuration used to initialize the Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim
        self.pre_norm = self.config.pre_norm
        self.self_attn = Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )

        self.dropout = self.config.dropout
        self.activation_fn = ACT2FN[self.config.activation_function]
        self.activation_dropout = self.config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.config.num_attention_heads, self.config.dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        # 定义 Masked(Cross)-Attention Block 中的变量
        cross_attn_weights = None
        self_attn_weights = None

        # 保存残差连接
        residual = hidden_states

        # 进行交叉注意力计算
        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        # 对 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 进行层归一化
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # 保存残差连接
        residual = hidden_states

        # 进行自注意力计算
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        # 对 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 进行层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存残差连接
        residual = hidden_states
        # 使用激活函数激活全连接层 fc1
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用全连接层 fc2
        hidden_states = self.fc2(hidden_states)
        # 对 hidden_states 进行 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 进行层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 输出结果为 hidden_states
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将 self_attn_weights 和 cross_attn_weights 添加到输出中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    # 前向传播函数，用于预训练
    def forward_pre(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        # 定义 Masked(Cross)-Attention Block
        cross_attn_weights = None
        self_attn_weights = None

        # 保存残差连接
        residual = hidden_states

        # 对隐藏状态进行 Layer Normalization
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # 进行 Cross-Attention 操作
        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        # 对输出结果进行 Dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 定义 Self Attention Block
        residual = hidden_states

        # 对隐藏状态进行 Layer Normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 进行 Self-Attention 操作
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        # 对输出结果进行 Dropout 操作
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # Fully Connected 操作
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states

        # 输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则添加到输出结果中
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                输入到层的形状为`(seq_len, batch, embed_dim)`的张量。
            attention_mask (`torch.FloatTensor`):
                形状为`(1, seq_len, tgt_len, src_len)`的注意力掩码。
            position_embeddings (`torch.FloatTensor`, *可选*):
                添加到掩码注意力层中键的位置嵌入。
            query_position_embeddings (`torch.FloatTensor`, *可选*):
                添加到自注意力层中查询和键的位置嵌入。
            encoder_hidden_states (`torch.FloatTensor`):
                形状为`(seq_len, batch, embed_dim)`的层的交叉注意力输入。
            encoder_attention_mask (`torch.FloatTensor`):
                大小为`(1, seq_len, tgt_len, src_len)`的编码器注意力掩码。
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量下的`attentions`。
        """

        if self.pre_norm:
            outputs = self.forward_pre(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
        else:
            outputs = self.forward_post(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

        return outputs
class Mask2FormerMaskedAttentionDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a
    [`Mask2FormerMaskedAttentionDecoderLayer`]. The decoder updates the query embeddings through multiple cross
    (masked) and self-attention layers. The decoder uses a new **masked attention** mechanism instead of the standard
    cross-attention, which extracts localized features by constraining cross-attention to within the foreground region
    of the predicted mask for each query, instead of attending to the full feature map.

    Args:
        config (`Mask2FormerConfig`):
            Configuration used to instantiate Mask2FormerMaskedAttentionDecoder.
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()

        self.config = config
        self.mask_feature_size = config.mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.dropout
        self.num_feature_levels = 3  # level embedding (3 scales)
        self.decoder_layers = config.decoder_layers - 1

        self.layers = nn.ModuleList(
            [Mask2FormerMaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)]
        )
        self.layernorm = nn.LayerNorm(config.hidden_dim)

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=config.hidden_dim,
            num_heads=config.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        multi_stage_positional_embeddings: torch.Tensor = None,
        pixel_embeddings: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        query_position_embeddings: torch.Tensor = None,
        feature_size_list: List = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 定义 Mask2FormerMaskedAttentionDecoder 类，用于 Transformer 解码器
        # 该解码器由 *config.decoder_layers* 层组成，每一层是一个 Mask2FormerMaskedAttentionDecoderLayer
        # 通过多个交叉（掩码）和自注意力层更新查询嵌入
        # 该解码器使用新的 **掩码注意力** 机制，而不是标准的交叉注意力，通过将交叉注意力限制在每个查询的预测掩码的前景区域内，而不是关注整个特征图，提取局部特征

        # 初始化 Mask2FormerMaskedAttentionDecoder 类
        def __init__(self, config: Mask2FormerConfig):
            super().__init__()

            self.config = config
            self.mask_feature_size = config.mask_feature_size
            self.dropout = config.dropout
            self.layerdrop = config.dropout
            self.num_feature_levels = 3  # level embedding (3 scales)
            self.decoder_layers = config.decoder_layers - 1

            # 创建包含多个 Mask2FormerMaskedAttentionDecoderLayer 的层列表
            self.layers = nn.ModuleList(
                [Mask2FormerMaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)]
            )
            self.layernorm = nn.LayerNorm(config.hidden_dim)

            # 创建 Mask2FormerMaskPredictor 对象
            self.mask_predictor = Mask2FormerMaskPredictor(
                hidden_size=config.hidden_dim,
                num_heads=config.num_attention_heads,
                mask_feature_size=self.mask_feature_size,
            )

            self.gradient_checkpointing = False

        def forward(
            self,
            inputs_embeds: torch.Tensor = None,
            multi_stage_positional_embeddings: torch.Tensor = None,
            pixel_embeddings: torch.Tensor = None,
            encoder_hidden_states: torch.Tensor = None,
            query_position_embeddings: torch.Tensor = None,
            feature_size_list: List = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            # 前向传播函数，接收输入参数并返回输出结果


# Copied from transformers.models.maskformer.modeling_maskformer.PredictionBlock with MaskFormer->Mask2Former
class Mask2FormerPredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # 继承 nn.Module 类，定义 Mask2FormerPredictionBlock 类
        # 初始化函数，接收输入维度、输出维度和激活函数作为参数

        # 创建包含线性层和激活函数的列表
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        # 前向传播函数，接收输入张量并返回输出张量
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerMLPPredictionHead(nn.Module):
    # 定义 Mask2FormerMLPPredictionHead 类
    # 初始化多层感知器（MLP）模型
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                输入维度。
            hidden_dim (`int`):
                隐藏层维度。
            output_dim (`int`):
                输出维度。
            num_layers (int, *optional*, defaults to 3):
                层数。
        """
        # 调用父类的初始化方法
        super().__init__()
        # 构建输入维度和隐藏层维度列表
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        # 构建隐藏层维度和输出维度列表
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        # 初始化神经网络层列表
        self.layers = []
        # 遍历输入维度和输出维度列表
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            # 根据层数确定激活函数
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            # 创建 Mask2FormerPredictionBlock 层
            layer = Mask2FormerPredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            # 为了向后兼容，需要注册每个层
            self.add_module(str(i), layer)

    # 前向传播函数
    def forward(self, input: Tensor) -> Tensor:
        # 初始化隐藏状态为输入
        hidden_state = input
        # 遍历神经网络层列表，进行前向传播
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        # 返回最终隐藏状态
        return hidden_state
class Mask2FormerMaskPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mask_feature_size: torch.Tensor):
        """
        This class is used to get the predicted mask for a given Mask2FormerMaskedAttentionDecoder layer. It also
        generates the binarized attention mask associated with the given predicted mask. The attention mask obtained
        using predicted mask of the (l-1)th decoder layer is fed to the cross(masked)-attention block of the next
        decoder layer as input.

        Args:
            hidden_size (`int`):
                The feature dimension of the Mask2FormerMaskedAttentionDecoder
            num_heads (`int`):
                The number of heads used in the Mask2FormerMaskedAttentionDecoder
            mask_feature_size (`torch.Tensor`):
                one of the output dimensions of the predicted masks for each query
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.mask_embedder = Mask2FormerMLPPredictionHead(self.hidden_size, self.hidden_size, mask_feature_size)

    def forward(self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: int = None):
        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))

        # Equivalent to einsum('bqc, bchw -> bqhw') but jit friendly
        batch_size, num_queries, num_channels = mask_embeddings.shape
        _, _, height, width = pixel_embeddings.shape
        outputs_mask = torch.zeros((batch_size, num_queries, height, width), device=mask_embeddings.device)
        for c in range(num_channels):
            outputs_mask += mask_embeddings[..., c][..., None, None] * pixel_embeddings[:, None, c]

        attention_mask = nn.functional.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()

        return outputs_mask, attention_mask


class Mask2FormerTransformerModule(nn.Module):
    """
    The Mask2Former's transformer module.
    """
    # 初始化函数，接受输入特征数和配置参数
    def __init__(self, in_features: int, config: Mask2FormerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 从配置参数中获取隐藏层维度
        hidden_dim = config.hidden_dim
        # 设置特征级别数量为3
        self.num_feature_levels = 3
        # 初始化位置编码器
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        # 初始化查询嵌入层
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        # 初始化查询特征嵌入层
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
        # 初始化输入投影列表
        self.input_projections = []

        # 循环创建特征级别数量的输入投影层
        for _ in range(self.num_feature_levels):
            # 如果输入特征数不等于隐藏层维度或者需要强制输入投影
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())

        # 初始化解码器
        self.decoder = Mask2FormerMaskedAttentionDecoder(config=config)
        # 初始化级别嵌入层
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    # 前向传播函数
    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Mask2FormerMaskedAttentionDecoderOutput:
        # 初始化多个特征级别的列表
        multi_stage_features = []
        # 初始化多个特征级别的位置嵌入列表
        multi_stage_positional_embeddings = []
        # 初始化尺寸列表
        size_list = []

        # 遍历特征级别数量
        for i in range(self.num_feature_levels):
            # 记录特征尺寸
            size_list.append(multi_scale_features[i].shape[-2:])
            # 计算多个特征级别的位置嵌入
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            # 计算多个特征级别的特征
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # 将位置嵌入展平为 (height*width, batch_size, num_channels)
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            # 将特征展平为 (height*width, batch_size, num_channels)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        # 获取 batch_size
        _, batch_size, _ = multi_stage_features[0].shape

        # 创建查询嵌入张量 [num_queries, batch_size, num_channels]
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # 创建查询特征张量 [num_queries, batch_size, num_channels]
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # 调用解码器进行解码
        decoder_output = self.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # 返回解码器输出
        return decoder_output
# 定义 MASK2FORMER_START_DOCSTRING，包含模型的基本信息和参数说明
MASK2FORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Mask2FormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 MASK2FORMER_INPUTS_DOCSTRING，包含模型的输入参数说明
MASK2FORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.preprocess`] for details.
        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).

            [What are attention masks?](../glossary#attention-mask)
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of Detr's decoder attention layers.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~Mask2FormerModelOutput`] instead of a plain tuple.
"""

# 定义 Mask2FormerPreTrainedModel 类，继承自 PreTrainedModel
class Mask2FormerPreTrainedModel(PreTrainedModel):
    config_class = Mask2FormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

# 添加模型输出的文档字符串和参数说明
@add_start_docstrings(
    "The bare Mask2Former Model outputting raw hidden-states without any specific head on top.",
    MASK2FORMER_START_DOCSTRING,
)
# 定义 Mask2FormerModel 类，继承自 Mask2FormerPreTrainedModel
class Mask2FormerModel(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    # 初始化方法，接受配置参数并初始化模型的���件
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        self.transformer_module = Mask2FormerTransformerModule(in_features=config.feature_size, config=config)

        self.post_init()

    # 前向传播方法，接受输入参数并返回模型输出
    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    # 定义一个字符串，描述了 Mask2Former 模型用于实例/语义/全景分割的情况
    "The Mask2Former Model with heads on top for instance/semantic/panoptic segmentation.",
    # 使用 MASK2FORMER_START_DOCSTRING 开始一个文档字符串
    MASK2FORMER_START_DOCSTRING,
# 定义一个类，继承自 Mask2FormerPreTrainedModel
class Mask2FormerForUniversalSegmentation(Mask2FormerPreTrainedModel):
    # 定义类属性 main_input_name
    main_input_name = "pixel_values"

    # 初始化方法，接受一个 Mask2FormerConfig 类型的参数 config
    def __init__(self, config: Mask2FormerConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 Mask2FormerModel 对象
        self.model = Mask2FormerModel(config)

        # 定义一个字典 weight_dict，存储损失函数的权重
        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        # 创建一个线性层，用于类别预测
        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)

        # 创建一个损失函数对象 Mask2FormerLoss
        self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
        # 调用后续初始化方法
        self.post_init()

    # 定义一个方法，计算损失函数的字典
    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        # 调用损失函数对象计算损失
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        # 根据权重字典对每个损失进行加权
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    # 定义一个方法，计算总损失
    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    # 定义一个方法，获取辅助预测的 logits
    def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
        auxiliary_logits: List[Dict(str, Tensor)] = []

        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append({"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes})

        return auxiliary_logits

    # 前向传播方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(MASK2FORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Mask2FormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: Optional[List[Tensor]] = None,
        class_labels: Optional[List[Tensor]] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_auxiliary_logits: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```