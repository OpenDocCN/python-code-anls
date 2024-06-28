# `.\models\mask2former\modeling_mask2former.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指出 Meta Platforms, Inc. 和 The HuggingFace Inc. 团队的版权信息
# 版权遵循 Apache 许可证 2.0 版本，详见 https://www.apache.org/licenses/LICENSE-2.0
# 除非符合许可证要求或书面同意，否则禁止使用本文件中的代码
# 本代码基于 "AS IS" 基础分发，无论是明示还是暗示的，均不提供任何形式的保证或条件
# 更多信息请参见许可证内容

""" PyTorch Mask2Former model. """

# 导入所需的库和模块
import math  # 导入数学库
import warnings  # 导入警告处理模块
from dataclasses import dataclass  # 导入 dataclass 用于数据类的装饰器
from typing import Dict, List, Optional, Tuple  # 导入类型提示相关的工具

import numpy as np  # 导入 NumPy 库
import torch  # 导入 PyTorch 深度学习库
from torch import Tensor, nn  # 导入 PyTorch 的张量和神经网络模块

# 导入 Hugging Face 库中的模块和函数
from ...activations import ACT2FN  # 导入激活函数映射
from ...file_utils import (  # 导入文件工具函数
    ModelOutput,  # 模型输出类
    add_start_docstrings,  # 添加文档字符串的装饰器
    add_start_docstrings_to_model_forward,  # 为模型前向方法添加文档字符串的装饰器
    is_scipy_available,  # 检查是否安装了 SciPy 库
    replace_return_docstrings,  # 替换返回文档字符串的装饰器
    requires_backends,  # 要求后端支持的装饰器
)
from ...modeling_outputs import (  # 导入模型输出相关的类
    BaseModelOutput,  # 基础模型输出类
    BaseModelOutputWithCrossAttentions,  # 带跨注意力机制的基础模型输出类
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型的工具函数
from ...pytorch_utils import is_torch_greater_or_equal_than_2_1  # 检查是否是 PyTorch 2.1 及以上版本的工具函数
from ...utils import is_accelerate_available, logging  # 导入加速库是否可用和日志工具
from ...utils.backbone_utils import load_backbone  # 导入加载骨干网络的工具函数
from .configuration_mask2former import Mask2FormerConfig  # 导入 Mask2Former 的配置类

# 如果安装了 SciPy 库，则导入线性求和分配功能
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 如果加速库可用，则导入部分状态和减少工具
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 模型文档相关信息
_CONFIG_FOR_DOC = "Mask2FormerConfig"  # 配置文档名称
_CHECKPOINT_FOR_DOC = "facebook/mask2former-swin-small-coco-instance"  # 预训练模型检查点信息
_IMAGE_PROCESSOR_FOR_DOC = "Mask2FormerImageProcessor"  # 图像处理器信息

# 预训练模型存档列表
MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mask2former-swin-small-coco-instance",  # Facebook 提供的 Mask2Former 预训练模型
    # 可以查看所有的 Mask2Former 模型列表 https://huggingface.co/models?filter=mask2former
]

@dataclass
class Mask2FormerPixelDecoderOutput(ModelOutput):
    """
    Mask2Former's pixel decoder module output, practically a Multi-Scale Deformable Attention based decoder. It returns
    the mask features and the multiscale features.
    """
    pass
    Args:
        multi_scale_features (`tuple(torch.FloatTensor)`):
            Tuple of multi-scale features of scales [1/8, 1/16, 1/32] and shape `(batch_size, num_channels, height,
            width)`from the Multi-Scale Deformable Attention based Pixel Decoder.
            多尺度特征的元组，包含比例为 [1/8, 1/16, 1/32] 的特征，形状为 `(batch_size, num_channels, height, width)`，
            来自基于多尺度可变注意力的像素解码器。
        mask_features (`torch.FloatTensor`):
            Tensor of shape `(batch_size, num_channels, height, width)`, 1/4 scale features from the last Pixel Decoder
            Layer.
            形状为 `(batch_size, num_channels, height, width)` 的张量，来自最后一个像素解码器层的1/4比例特征。
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from pixel decoder. Returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`
            可选的注意力权重元组，每个元素的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，
            表示像素解码器中的注意力权重。在设置 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
    """

    multi_scale_features: Tuple[torch.FloatTensor] = None
    mask_features: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class Mask2FormerMaskedAttentionDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Mask2FormerMaskedAttentionDecoderOutput 类用于表示 Transformer 解码器的输出。
    它在 BaseModelOutputWithCrossAttentions 的基础上添加了两个属性：mask 预测的 logits 和中间解码器激活的元组，
    即每个解码器层的输出，每个输出都经过 layernorm 处理。

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            一个元组，包含 `torch.FloatTensor` 类型的张量。第一个张量是从嵌入层输出的结果，其余每个张量对应每个层的输出，
            形状为 `(batch_size, sequence_length, hidden_size)`。当 `output_hidden_states=True` 时返回。
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            一个元组，包含 `torch.FloatTensor` 类型的张量，每个张量的形状为 `(batch_size, num_heads, sequence_length,
            sequence_length)`。表示经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均。
            当 `output_attentions=True` 时返回。
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`):
            一个元组，包含 Transformer 解码器所有层的 mask 预测 logits。
        intermediate_hidden_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            中间解码器激活的元组，即每个解码器层的输出，每个输出都经过 layernorm 处理。
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    intermediate_hidden_states: Tuple[torch.FloatTensor] = None


@dataclass
class Mask2FormerPixelLevelModuleOutput(ModelOutput):
    """
    Mask2FormerPixelLevelModuleOutput 类表示 Mask2Former 模型的像素级模块输出。
    它返回了编码器的输出（可选）以及 `decoder` 的所有隐藏状态（多尺度特征）。
    默认情况下，`encoder` 是 Swin 骨干网络，`decoder` 是基于多尺度可变形注意力的解码器。

    `decoder_last_hidden_state` 是每个像素的嵌入，而 `decoder_hidden_states` 指的是使用论文中定义的多尺度策略产生的多尺度特征图。

    Args:
        decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            解码器最后一层的每个像素的嵌入。
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            一个元组，包含 `torch.FloatTensor` 类型的张量。表示使用多尺度策略生成的多尺度特征图。
    """
    # 定义函数的参数列表，包括四个输入参数，均为torch.FloatTensor类型
    Args:
        encoder_last_hidden_state (`torch.FloatTensor`):
            编码器最后的隐藏状态，即最后阶段编码器的最终特征图，形状为`(batch_size, num_channels, height, width)`
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            编码器每个阶段输出的隐藏状态的元组。每个元素是形状为`(batch_size, num_channels, height, width)`的torch.FloatTensor。
            如果设置了output_hidden_states为True，则返回此参数。
        decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width))`:
            解码器最后一个Pixel解码层的1/4比例特征。
        decoder_hidden_states (`tuple(torch.FloatTensor)`):
            解码器每个阶段输出的隐藏状态的元组。每个元素是形状为`(batch_size, num_channels, height, width)`的torch.FloatTensor。
        """
    
    # 初始化函数内的变量，默认值为None
    encoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_last_hidden_state: torch.FloatTensor = None
    decoder_hidden_states: Tuple[torch.FloatTensor] = None
@dataclass
class Mask2FormerModelOutput(ModelOutput):
    """
    Class for outputs of [`Mask2FormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone). Returned when
            `output_hidden_states=True` is passed.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
            Last hidden states (final feature map) of the last stage of the pixel decoder model.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_last_hidden_state (`tuple(torch.FloatTensor)`):
            Final output of the transformer decoder `(batch_size, sequence_length, hidden_size)`.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage. Returned when `output_hidden_states=True` is passed.
        transformer_decoder_intermediate_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
            Intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a
            layernorm.
        masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`)
            Mask Predictions from each layer in the transformer decoder.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `tuple(torch.FloatTensor)` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Self attentions weights from transformer decoder.
    """

    # 定义一个数据类，用于存储 Mask2FormerModel 的输出结果，包括各个模型阶段的隐藏状态和注意力权重

    encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
        # encoder 最后一个隐藏状态（即最终特征图），形状为 `(batch_size, num_channels, height, width)`，可选参数
        Last hidden states (final feature map) of the last stage of the encoder model.

    encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        # encoder 隐藏状态的元组，每个元素为 `torch.FloatTensor`，形状为 `(batch_size, num_channels, height, width)`
        Tuple of `torch.FloatTensor` representing hidden states of the encoder model at each stage.

    pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`, *optional*):
        # pixel decoder 最后一个隐藏状态（即最终特征图），形状为 `(batch_size, num_channels, height, width)`，可选参数
        Last hidden states of the last stage of the pixel decoder model.

    pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        # pixel decoder 隐藏状态的元组，每个元素为 `torch.FloatTensor`，形状为 `(batch_size, num_channels, height, width)`
        Tuple of `torch.FloatTensor` representing hidden states of the pixel decoder model at each stage.

    transformer_decoder_last_hidden_state (`tuple(torch.FloatTensor)`):
        # transformer decoder 最终输出，形状为 `(batch_size, sequence_length, hidden_size)`
        Final output of the transformer decoder.

    transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
        # transformer decoder 隐藏状态的元组，每个元素为 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`
        Tuple of `torch.FloatTensor` representing hidden states of the transformer decoder at each stage.

    transformer_decoder_intermediate_states (`tuple(torch.FloatTensor)` of shape `(num_queries, 1, hidden_size)`):
        # transformer decoder 中间层激活的元组，每个元素为 `torch.FloatTensor`，形状为 `(num_queries, 1, hidden_size)`
        Intermediate decoder activations, each gone through a layernorm.

    masks_queries_logits (`tuple(torch.FloatTensor)` of shape `(batch_size, num_queries, height, width)`)
        # transformer decoder 中每层的 mask 预测，形状为 `(batch_size, num_queries, height, width)`
        Mask Predictions from each layer in the transformer decoder.

    attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed):
        # transformer decoder 自注意力权重的元组，每个元素为 `tuple(torch.FloatTensor)`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
        Self attentions weights from transformer decoder.
    """
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_intermediate_states: Tuple[torch.FloatTensor] = None
    masks_queries_logits: Tuple[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    """
@dataclass
class Mask2FormerForUniversalSegmentationOutput(ModelOutput):
    """
    [`Mask2FormerForUniversalSegmentationOutput`]的输出类。

    这个输出可以直接传递给[`~Mask2FormerImageProcessor.post_process_semantic_segmentation`]、
    [`~Mask2FormerImageProcessor.post_process_instance_segmentation`]或
    [`~Mask2FormerImageProcessor.post_process_panoptic_segmentation`]以计算最终的分割图。
    请参阅[`~Mask2FormerImageProcessor`]获取有关使用的详细信息。
    """

    loss: Optional[torch.FloatTensor] = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: Optional[List[Dict[str, torch.FloatTensor]]] = None
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# Adapted from https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False, **kwargs
) -> torch.Tensor:
    """
    一个对`torch.nn.functional.grid_sample`进行包装的函数，支持3D点坐标张量。

    Args:
        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):
            包含在高度*宽度网格上的特征映射的张量
        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,
        2)):
            包含[0, 1] * [0, 1]规范化点坐标的张量
        add_dim (`bool`):
            用于跟踪是否添加了维度

    Returns:
        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid)):
            包含`point_coordinates`中点的特征的张量。
    """
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # 使用nn.functional.grid_sample通过双线性插值获取`point_coordinates`中点的特征
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features


# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    计算DICE损失，类似于掩码的广义IOU，计算方式如下：
    """
    计算二进制分割任务中的 Dice Loss。

    Args:
        inputs (`torch.Tensor`):
            表示一个掩码的张量。
        labels (`torch.Tensor`):
            与输入张量具有相同形状的张量。存储每个元素的二进制分类标签
            （0表示负类，1表示正类）。
        num_masks (`int`):
            当前批次中存在的掩码数量，用于归一化。

    Returns:
        `torch.Tensor`: 计算得到的损失值。
    """
    # 计算概率，并将结果展平为二维数组
    probs = inputs.sigmoid().flatten(1)
    # 计算 Dice 损失的分子部分
    numerator = 2 * (probs * labels).sum(-1)
    # 计算 Dice 损失的分母部分
    denominator = probs.sum(-1) + labels.sum(-1)
    # 计算最终的 Dice 损失
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 将损失值对每个掩码进行求和并进行归一化
    loss = loss.sum() / num_masks
    return loss
# 定义一个函数，计算输入张量和标签之间的 sigmoid 交叉熵损失
def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            任意形状的浮点张量。
        labels (`torch.Tensor`):
            与输入张量形状相同的张量。存储每个输入元素的二元分类标签
            （0 表示负类，1 表示正类）。

    Returns:
        loss (`torch.Tensor`): 计算得到的损失张量。
    """
    # 使用 BCEWithLogitsLoss 函数定义损失计算方式，不进行汇总
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    # 计算交叉熵损失
    cross_entropy_loss = criterion(inputs, labels)

    # 计算平均损失，并按 num_masks 汇总
    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss


# 从 transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss 复制过来的代码
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    一对一版本的 Dice 损失，参见 `dice_loss` 的用法。

    Args:
        inputs (`torch.Tensor`):
            表示掩码的张量。
        labels (`torch.Tensor`):
            与输入张量形状相同的张量。存储每个输入元素的二元分类标签
            （0 表示负类，1 表示正类）。

    Returns:
        `torch.Tensor`: 每对之间计算得到的损失。
    """
    # 对输入张量应用 sigmoid 函数，并展平到第一维度
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * torch.matmul(inputs, labels.T)
    # 使用广播获取一个 [num_queries, NUM_CLASSES] 的矩阵
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


# 定义一个函数，计算输入张量和标签之间的一对一 sigmoid 交叉熵损失
def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    一对一版本的交叉熵损失，参见 `sigmoid_cross_entropy_loss` 的用法。

    Args:
        inputs (`torch.Tensor`):
            表示掩码的张量。
        labels (`torch.Tensor`):
            与输入张量形状相同的张量。存储每个输入元素的二元分类标签
            （0 表示负类，1 表示正类）。

    Returns:
        loss (`torch.Tensor`): 每对之间计算得到的损失。
    """

    # 获取输入张量的高度和宽度
    height_and_width = inputs.shape[1]

    # 使用 BCEWithLogitsLoss 函数定义损失计算方式，不进行汇总
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    # 分别计算正类和负类的交叉熵损失
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    # 计算正类和负类的损失
    loss_pos = torch.matmul(cross_entropy_loss_pos / height_and_width, labels.T)
    loss_neg = torch.matmul(cross_entropy_loss_neg / height_and_width, (1 - labels).T)
    # 组合正类和负类的损失
    loss = loss_pos + loss_neg
    return loss


# 从 https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py 调整而来
class Mask2FormerHungarianMatcher(nn.Module):
    """这个类计算标签和网络预测之间的分配。
    """
    For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
    predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544
    ):
        """Creates the matcher

        Params:
            cost_class (`float`, *optional*, defaults to 1.0):
                Relative weight of the classification error in the matching cost.
            cost_mask (`float`, *optional*,  defaults to 1.0):
                This is the relative weight of the focal loss of the binary mask in the matching cost.
            cost_dice (`float`, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost.
            num_points (`int`, *optional*, defaults to 12544):
                No. of points to sample on which the mask loss will be calculated. The same set of K points are
                uniformly sampled for all prediction and ground truth masks to construct the cost matrix for bipartite
                matching.
        """
        # 调用父类初始化方法
        super().__init__()
        # 如果分类、掩模和 Dice 损失权重均为零，则抛出异常
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")

        # 初始化属性
        self.num_points = num_points  # 设置采样点的数量
        self.cost_class = cost_class  # 设置分类错误的权重
        self.cost_mask = cost_mask    # 设置掩模损失的权重
        self.cost_dice = cost_dice    # 设置 Dice 损失的权重

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        class_queries_logits: torch.Tensor,
        mask_labels: torch.Tensor,
        class_labels: torch.Tensor,
# Adapted from https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py

# 定义 Mask2FormerLoss 类，继承自 nn.Module
class Mask2FormerLoss(nn.Module):
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
        super().__init__()
        requires_backends(self, ["scipy"])  # 确保需要的后端库被加载
        self.num_labels = config.num_labels  # 从配置中获取标签数量
        self.weight_dict = weight_dict  # 保存权重字典

        # Weight to apply to the null class
        self.eos_coef = config.no_object_weight  # 获取空对象的权重系数
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)  # 将权重缓存起来

        # pointwise mask loss parameters
        self.num_points = config.train_num_points  # 获取训练点数
        self.oversample_ratio = config.oversample_ratio  # 获取过采样比例
        self.importance_sample_ratio = config.importance_sample_ratio  # 获取重要性采样比例

        # 初始化匈牙利匹配器，用于计算损失
        self.matcher = Mask2FormerHungarianMatcher(
            cost_class=1.0,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=self.num_points,
        )

    # 从 sizes 列表中找到每个维度的最大值，并返回最大值列表
    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    # 将输入的张量列表进行填充，使它们的尺寸达到批次中最大的尺寸，并返回填充后的张量及其对应的填充掩码
    # 函数功能类似于原始实现中的 nested_tensor_from_tensor_list()
    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # 获取批次中的最大尺寸
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        # 计算最终的批次形状
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # 将张量填充到最大尺寸
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks
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
        # Assigning the predicted logits to a local variable
        pred_logits = class_queries_logits
        # Extracting dimensions from the predicted logits tensor
        batch_size, num_queries, _ = pred_logits.shape
        # Defining the cross entropy loss criterion with optional weights
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        # Obtaining permutation indices for predictions based on the Hungarian matcher
        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)
        # Concatenating target mask labels based on indices
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )  # shape of (batch_size, num_queries)
        # Creating a tensor filled with a specific value for target classes
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        # Assigning the concatenated target classes to their respective positions using indices
        target_classes[idx] = target_classes_o
        # Transposing the predicted logits tensor for cross entropy computation
        pred_logits_transposed = pred_logits.transpose(1, 2)
        # Calculating cross entropy loss between transposed logits and target classes
        loss_ce = criterion(pred_logits_transposed, target_classes)
        # Constructing dictionary containing the computed cross entropy loss
        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        indices: Tuple[np.array],
        num_masks: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth.
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth,
              masks.
        """
        # 获取预测排序后的索引
        src_idx = self._get_predictions_permutation_indices(indices)
        # 获取目标排序后的索引
        tgt_idx = self._get_targets_permutation_indices(indices)
        # shape (batch_size * num_queries, height, width)
        # 从预测的logits中选择对应src_idx的预测掩码
        pred_masks = masks_queries_logits[src_idx]
        # shape (batch_size, num_queries, height, width)
        # 将目标掩码进行填充以匹配批次中最大的图像，并在num_labels维度上堆叠
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        # 根据tgt_idx选择目标掩码
        target_masks = target_masks[tgt_idx]

        # 由于使用了归一化坐标，不需要对预测进行上采样
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

            # 使用采样点从目标掩码中获取点标签，不对齐角点
            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)

        # 使用采样点从预测掩码中获取点logits，不对齐角点
        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

        # 计算损失
        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        # 清理临时变量
        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        # 根据indices对预测进行排列
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices
    # 根据给定的索引重新排列标签
    def _get_targets_permutation_indices(self, indices):
        # 创建批次索引，使每个标签在批次中重复出现
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        # 创建目标索引，将所有目标标签连接成一个张量
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    # 计算不确定性分数
    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        在Mask2Former论文中，不确定性被估计为logits中前景类的预测与0.0之间的L1距离。

        Args:
            logits (`torch.Tensor`): 形状为(R, 1, ...)的张量，R为所有预测掩码的总数，C为前景类的数量，值为logits。

        Returns:
            scores (`torch.Tensor`): 形状为(R, 1, ...)的张量，包含不确定性分数，不确定位置的分数最高。
        """
        # 计算不确定性分数，使用-logits的绝对值
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores

    # 使用不确定性函数采样点
    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function samples points in [0, 1] * [0, 1] coordinate space based on uncertainty of logits predictions.

        Args:
            logits (`torch.Tensor`):
                Logit predictions for bounding boxes.
            uncertainty_function:
                Function to calculate uncertainties based on logit predictions.
            num_points (`int`):
                Number of points to sample.
            oversample_ratio (`int`):
                Oversampling ratio for point sampling.
            importance_sample_ratio (`float`):
                Ratio of points sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates of sampled points.
        """

        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random coordinates for points within each bounding box
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)

        # Sample logits values at the sampled coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)

        # Calculate uncertainties based on the sampled logits values
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        # Select uncertain points based on top uncertainties
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        # Add random points to complete the required number of points
        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )

        return point_coordinates
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
                Contains logits for predicted masks.
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, num_labels)`.
                Contains logits for predicted class labels.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
                Ground truth masks.
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
                Ground truth class labels.
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from
                the inner layers of the Mask2FormerMaskedAttentionDecoder.
                Dictionary of auxiliary predictions from intermediate layers.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted and ground truth masks.
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
    # 计算每个批次中目标掩码的平均数量，用于归一化目的
    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        # 计算每个样本中类标签列表的总长度，即目标掩码的总数
        num_masks = sum([len(classes) for classes in class_labels])
        # 将总数转换为张量，并指定数据类型和设备
        num_masks = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        # 初始化世界大小为 1
        world_size = 1
        # 检查是否可用加速功能
        if is_accelerate_available():
            # 检查是否存在部分状态的共享状态
            if PartialState._shared_state != {}:
                # 使用 reduce 函数对目标掩码数目进行归约操作
                num_masks = reduce(num_masks)
                # 获取部分状态的进程数
                world_size = PartialState().num_processes

        # 对目标掩码数目进行截断操作，确保不低于 1
        num_masks = torch.clamp(num_masks / world_size, min=1)
        # 返回归一化后的目标掩码数目张量
        return num_masks
# 从transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention复制而来
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    # 获取输入张量的形状信息
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # 将value按照空间形状切分成不同的部分
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    # 计算采样网格的位置
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # 将value_list中的每个部分展平，并且重塑成合适的形状
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # 调整采样网格的形状以匹配value_l_的大小，并使用双线性插值进行采样
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # 重新组织注意力权重的形状，以便与采样值匹配
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    # 计算最终的输出，进行加权求和并调整形状
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    # 转置输出张量并确保连续的内存布局
    return output.transpose(1, 2).contiguous()


# 从transformers.models.maskformer.modeling_maskformer.MaskFormerSinePositionEmbedding复制而来，并将类名更改为Mask2FormerSinePositionEmbedding
class Mask2FormerSinePositionEmbedding(nn.Module):
    """
    这是一个更标准的位置嵌入版本，与“Attention is all you need”论文中使用的非常相似，通用于处理图像。
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        ):
            super().__init__()
            # 调用父类的构造方法，初始化父类
            if scale is not None and normalize is False:
                # 如果 scale 不为 None 而且 normalize 为 False，则抛出数值错误异常
                raise ValueError("normalize should be True if scale is passed")
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            self.scale = 2 * math.pi if scale is None else scale

        def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
            if mask is None:
                # 如果 mask 参数为 None，则创建一个全零的 mask 张量，与 x 的尺寸一致
                mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
            # 计算非 mask 的张量，将 mask 取反后转换为 x 的数据类型
            not_mask = (~mask).to(x.dtype)
            # 沿着第1维和第2维度累积非 mask 值，得到 y 和 x 的位置编码
            y_embed = not_mask.cumsum(1)
            x_embed = not_mask.cumsum(2)
            if self.normalize:
                # 如果需要进行归一化
                eps = 1e-6
                # 对 y_embed 和 x_embed 进行归一化处理，乘以 self.scale，防止除零错误
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            # 创建维度张量 dim_t，其值为从0到 num_pos_feats-1 的整数序列
            dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).type_as(x)
            # 计算温度的幂次方，用于位置编码
            dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

            # 计算 x 和 y 的位置编码
            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            # 使用正弦和余弦函数对位置编码进行变换，并展平后拼接得到 pos
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            # 返回位置编码张量 pos
            return pos
# 基于 Deformable DETR 模型中的 DeformableDetrMultiscaleDeformableAttention 类进行修改
class Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
    """
    在 Deformable DETR 中提出的多尺度可变形注意力机制。
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        super().__init__()
        # 确保 embed_dim 可以被 num_heads 整除
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        # 计算每个头部的维度
        dim_per_head = embed_dim // num_heads
        # 检查 dim_per_head 是否为2的幂
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        # 每次 im2col 转换的步长
        self.im2col_step = 128

        # 初始化模型的参数
        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        # 定义用于偏移量的线性层
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        # 定义注意力权重的线性层
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        # 对值进行投影的线性层
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # 对输出进行投影的线性层
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        # 如果存在位置嵌入，将其加到张量中
        return tensor if position_embeddings is None else tensor + position_embeddings

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
    ):
        # 在投影到查询和键之前，向隐藏状态添加位置嵌入
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # 获取隐藏状态张量的形状信息
        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        
        # 检查空间形状与编码器隐藏状态序列长度是否对齐
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        # 对编码器隐藏状态进行值投影
        value = self.value_proj(encoder_hidden_states)

        # 如果存在注意力掩码，则反转注意力掩码
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], float(0))

        # 重塑值张量的形状以便多头注意力机制处理
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)

        # 计算采样偏移量
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )

        # 计算注意力权重
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )

        # 对注意力权重进行 softmax 归一化
        attention_weights = nn.functional.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )

        # 如果参考点张量的最后一个维度为2
        if reference_points.shape[-1] == 2:
            # 计算采样位置
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        # 如果参考点张量的最后一个维度为4
        elif reference_points.shape[-1] == 4:
            # 计算采样位置
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            # 抛出异常，参考点张量的最后一个维度必须是2或4
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")

        # 多尺度可变形注意力机制计算输出
        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)

        # 对输出进行最终的投影
        output = self.output_proj(output)

        # 返回输出和注意力权重
        return output, attention_weights
# 定义一个名为 Mask2FormerPixelDecoderEncoderLayer 的神经网络模块类，继承自 nn.Module
class Mask2FormerPixelDecoderEncoderLayer(nn.Module):
    # 初始化函数，接收一个 Mask2FormerConfig 类型的参数 config
    def __init__(self, config: Mask2FormerConfig):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()
        # 设置 embed_dim 属性为 config 中的 feature_size，表示特征的维度大小
        self.embed_dim = config.feature_size
        # 初始化 self_attn 属性为 Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention 类的实例
        # 参数包括 embed_dim（特征维度）、num_heads（注意力头数）、n_levels（多尺度层数）、n_points（变形注意力的采样点数）
        self.self_attn = Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            n_levels=3,
            n_points=4,
        )

        # 初始化 self_attn_layer_norm 属性为 LayerNorm 层，输入维度为 embed_dim
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置 dropout 属性为 config 中的 dropout 概率
        self.dropout = config.dropout
        # 设置 activation_fn 属性为 relu 激活函数
        self.activation_fn = nn.functional.relu
        # 设置 activation_dropout 属性为 config 中的 dropout 概率
        self.activation_dropout = config.dropout
        # 初始化 fc1 属性为 Linear 层，输入维度为 embed_dim，输出维度为 config 中的 encoder_feedforward_dim
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        # 初始化 fc2 属性为 Linear 层，输入维度为 config 中的 encoder_feedforward_dim，输出维度为 embed_dim
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        # 初始化 final_layer_norm 属性为 LayerNorm 层，输入维度为 embed_dim
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，定义网络的数据流向
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                输入到层的输入。
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                注意力遮罩。
            position_embeddings (`torch.FloatTensor`, *optional*):
                位置嵌入，将要添加到 `hidden_states` 中。
            reference_points (`torch.FloatTensor`, *optional*):
                参考点。
            spatial_shapes (`torch.LongTensor`, *optional*):
                主干特征图的空间形状。
            level_start_index (`torch.LongTensor`, *optional*):
                层级起始索引。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 以获取更多细节。
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
            # 如果在训练中
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                # 如果任何一个张量中有无穷大或者NaN值
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                # 将张量值限制在一个较小的值范围内，避免溢出
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            # 如果需要输出注意力权重
            outputs += (attn_weights.transpose(1, 0),)

        return outputs
# 修改自 transformers.models.detr.modeling_deformable_detr.DeformableDetrEncoder 的 DeformableDetrEncoder -> Mask2FormerPixelDecoderEncoderOnly
class Mask2FormerPixelDecoderEncoderOnly(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`Mask2FormerPixelDecoderEncoderLayer`]. The encoder updates the flattened multi-scale feature maps through
    multiple deformable attention layers.

    Args:
        config: Mask2FormerConfig
    """

    def __init__(self, config: Mask2FormerConfig):
        super().__init__()

        # 保存配置信息
        self.config = config
        # 定义 dropout 概率
        self.dropout = config.dropout
        # 创建多个 Mask2FormerPixelDecoderEncoderLayer 层，数量由 config.encoder_layers 决定
        self.layers = nn.ModuleList(
            [Mask2FormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)]
        )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of each feature map, has shape of `(num_feature_levels, 2)`.
            valid_ratios (`torch.FloatTensor`):
                Valid ratios of each feature map, has shape of `(batch_size, num_feature_levels, 2)`.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        # 初始化 reference_points_list 作为一个空列表
        reference_points_list = []
        # 遍历每个特征图的空间形状
        for lvl, (height, width) in enumerate(spatial_shapes):
            # 创建网格矩阵 ref_y, ref_x 作为参考点的 y 和 x 坐标
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
                indexing="ij",
            )
            # 将 ref_y, ref_x 转换成一维数组，然后按比例缩放
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            # 将 ref_x, ref_y 合并成 ref 的形式
            ref = torch.stack((ref_x, ref_y), -1)
            # 将当前级别的参考点 ref 添加到 reference_points_list 中
            reference_points_list.append(ref)

        # 拼接所有级别的 reference_points
        reference_points = torch.cat(reference_points_list, 1)
        # 根据 valid_ratios 调整 reference_points 的形状
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
# Modified from from transformers.models.detr.modeling_deformable_detr.DeformableDetrModel with DeformableDetrModel->Mask2FormerPixelDecoder
class Mask2FormerPixelDecoder(nn.Module):
    # 初始化函数，接受配置对象和特征通道数作为参数
    def __init__(self, config: Mask2FormerConfig, feature_channels):
        # 调用父类的初始化方法
        super().__init__()

        # 将配置对象保存在实例变量中
        self.config = config

        # 从配置对象中获取特征大小和掩码特征大小
        feature_dim = config.feature_size
        mask_dim = config.mask_feature_size
        # 计算每一层的正向特征数量
        num_pos_features = feature_dim // 2

        # 创建位置嵌入对象，使用正向特征数量和归一化参数
        self.position_embedding = Mask2FormerSinePositionEmbedding(num_pos_feats=num_pos_features, normalize=True)
        # 定义特征层级数量为3
        self.num_feature_levels = 3
        # 从输入特征通道中提取变压器输入通道数
        transformer_in_channels = feature_channels[-self.num_feature_levels :]

        # 获取变压器特征步幅和通道信息
        self.transformer_feature_strides = config.feature_strides[-self.num_feature_levels :]
        self.feature_channels = feature_channels
        # 创建层级嵌入参数，使用指定的特征层级和特征维度
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, feature_dim))

        # 创建输入投影层
        if self.num_feature_levels > 1:
            input_projections_list = []
            # 遍历反向变压器输入通道列表
            for in_channels in transformer_in_channels[::-1]:
                # 使用卷积层和分组规范化创建顺序模块
                input_projections_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                )
            # 使用模块列表创建输入投影层
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            # 若特征层级为1，创建单一模块列表
            self.input_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(transformer_in_channels[-1], feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                ]
            )

        # 创建解码器编码器对象
        self.encoder = Mask2FormerPixelDecoderEncoderOnly(config)
        # 创建掩码投影卷积层，使用特征维度和掩码特征维度
        self.mask_projection = nn.Conv2d(feature_dim, mask_dim, kernel_size=1, stride=1, padding=0)

        # 额外的特征金字塔网络层级
        stride = min(self.transformer_feature_strides)
        self.common_stride = config.common_stride
        # 计算FPN层级数目
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        # 遍历特征通道列表的前几个通道，创建侧向和输出卷积层
        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            # 创建侧向卷积层，使用1x1卷积和分组规范化
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, feature_dim),
            )

            # 创建输出卷积层，使用3x3卷积、分组规范化和ReLU激活函数
            output_conv = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(),
            )
            # 将侧向和输出卷积层作为模块添加到模型中
            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            # 将侧向和输出卷积层添加到列表中
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # 将侧向卷积层和输出卷积层反转顺序，以便从低分辨率到高分辨率排序
        self.lateral_convolutions = lateral_convs[::-1]
        self.output_convolutions = output_convs[::-1]
    # 计算输入掩码中每个特征图的有效比率
    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""

        # 获取掩码的形状信息，并解构为通道、高度、宽度
        _, height, width = mask.shape
        
        # 计算每个特征图在高度上的有效像素数量，即非零像素数
        valid_height = torch.sum(~mask[:, :, 0], 1)
        
        # 计算每个特征图在宽度上的有效像素数量，即非零像素数
        valid_width = torch.sum(~mask[:, 0, :], 1)
        
        # 将有效像素数转换为有效比率，使用给定的数据类型
        valid_ratio_heigth = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        
        # 将高度和宽度的有效比率合并成一个张量，形状为 [batch_size, 2]
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        
        # 返回有效比率张量
        return valid_ratio

    # 模型前向传播函数
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
        super().__init__()

        # 加载指定配置的骨干网络
        self.encoder = load_backbone(config)
        # 使用骨干网络的通道数初始化像素解码器
        self.decoder = Mask2FormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> Mask2FormerPixelLevelModuleOutput:
        # 通过骨干网络获取特征图
        backbone_features = self.encoder(pixel_values).feature_maps
        # 使用解码器处理特征图，生成输出
        decoder_output = self.decoder(backbone_features, output_hidden_states=output_hidden_states)

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
        super().__init__()
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

        # 初始化线性变换层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        # 重塑张量以便进行多头注意力计算
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        # 如果存在位置编码张量，则将其加到输入张量上
        return tensor if position_embeddings is None else tensor + position_embeddings
    # 定义一个方法 `forward`，用于执行模型的前向传播过程
    def forward(
        self,
        # 输入参数 `hidden_states`，表示模型的隐藏状态，是一个张量
        hidden_states: torch.Tensor,
        # 输入参数 `attention_mask`，表示注意力掩码，可以为空
        attention_mask: Optional[torch.Tensor] = None,
        # 输入参数 `position_embeddings`，表示位置嵌入，可以为空
        position_embeddings: Optional[torch.Tensor] = None,
        # 输入参数 `key_value_states`，表示键值状态，可以为空
        key_value_states: Optional[torch.Tensor] = None,
        # 输入参数 `key_value_position_embeddings`，表示键值位置嵌入，可以为空
        key_value_position_embeddings: Optional[torch.Tensor] = None,
        # 输入参数 `output_attentions`，表示是否输出注意力权重，默认为 False
        output_attentions: bool = False,
    """
    Mask2FormerMaskedAttentionDecoderLayer由self-attention、交叉（masked）attention和FFN块组成。
    在Mask2FormerMaskedAttentionDecoderLayer中使用的交叉attention实际上是一种限制注意力在预测段周围局部特征的masked attention块，
    这导致更快的收敛和更好的性能。相比标准的DetrDecoder，Mask2FormerMaskedAttentionDecoder中的self和cross（即masked）attention块的顺序被交换，
    这是一种优化改进。

    Args:
        config (`Mask2FormerConfig`):
            用于初始化Mask2FormerMaskedAttentionDecoder的配置。
    """
    
    def __init__(self, config: Mask2FormerConfig):
        super().__init__()
        self.config = config
        self.embed_dim = self.config.hidden_dim  # 设置嵌入维度为配置中的隐藏维度
        self.pre_norm = self.config.pre_norm  # 设置预规范化标志为配置中的预规范化标志
        
        # 初始化self-attention层，使用Mask2FormerAttention类
        self.self_attn = Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )
        
        self.dropout = self.config.dropout  # 设置dropout率为配置中的dropout率
        self.activation_fn = ACT2FN[self.config.activation_function]  # 根据配置选择激活函数
        self.activation_dropout = self.config.dropout  # 设置激活函数的dropout率为配置中的dropout率
        
        # 初始化self-attention层的LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 初始化交叉attention层，使用nn.MultiheadAttention类
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.config.num_attention_heads, self.config.dropout)
        
        # 初始化交叉attention层的LayerNorm
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 初始化前向传播网络的第一个线性层
        self.fc1 = nn.Linear(self.embed_dim, self.config.dim_feedforward)
        
        # 初始化前向传播网络的第二个线性层
        self.fc2 = nn.Linear(self.config.dim_feedforward, self.embed_dim)
        
        # 初始化最终输出的LayerNorm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        如果位置编码pos不为None，则将其添加到张量tensor中；否则返回原始张量tensor。

        Args:
            tensor (torch.Tensor): 输入张量
            pos (Optional[Tensor]): 位置编码张量，可选

        Returns:
            torch.Tensor: 处理后的张量
        """
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
    ):
        # Masked(Cross)-Attention Block
        cross_attn_weights = None  # 初始化跨注意力权重为 None
        self_attn_weights = None    # 初始化自注意力权重为 None

        residual = hidden_states    # 保存输入的隐藏状态作为残差连接的基准

        # 执行跨注意力机制
        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),  # 使用位置嵌入增强查询
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),  # 使用位置嵌入增强键
            value=encoder_hidden_states[level_index],  # 使用编码器隐藏状态作为值
            attn_mask=encoder_attention_mask,  # 编码器注意力掩码
            key_padding_mask=None,  # 键的填充掩码暂未指定
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 使用丢弃来处理隐藏状态
        hidden_states = residual + hidden_states  # 残差连接
        hidden_states = self.cross_attn_layer_norm(hidden_states)  # 使用层归一化处理隐藏状态

        # Self Attention Block
        residual = hidden_states  # 保存当前隐藏状态作为自注意力块的残差基准

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,  # 使用当前隐藏状态
            position_embeddings=query_position_embeddings,  # 查询位置嵌入
            attention_mask=None,  # 注意力掩码暂未指定
            output_attentions=True,  # 输出注意力权重
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 使用丢弃来处理隐藏状态
        hidden_states = residual + hidden_states  # 残差连接
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 使用层归一化处理隐藏状态

        # Fully Connected
        residual = hidden_states  # 保存当前隐藏状态作为全连接块的残差基准
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 使用激活函数处理第一个全连接层
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 使用激活函数的丢弃
        hidden_states = self.fc2(hidden_states)  # 第二个全连接层
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 第二个全连接层的丢弃
        hidden_states = residual + hidden_states  # 残差连接
        hidden_states = self.final_layer_norm(hidden_states)  # 使用层归一化处理隐藏状态

        outputs = (hidden_states,)  # 输出为处理后的隐藏状态

        if output_attentions:  # 如果需要输出注意力权重
            outputs += (self_attn_weights, cross_attn_weights)  # 将自注意力和跨注意力权重添加到输出中

        return outputs  # 返回输出结果

    def forward_pre(
        self,
        hidden_states: torch.Tensor,
        level_index: int = None,  # 编码器层索引
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码（可选）
        position_embeddings: Optional[torch.Tensor] = None,  # 位置嵌入（可选）
        query_position_embeddings: Optional[torch.Tensor] = None,  # 查询位置嵌入（可选）
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态（可选）
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力掩码（可选）
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重（默认为 False）
        # Masked(Cross)-Attention Block
        cross_attn_weights = None  # 初始化交叉注意力权重为None
        self_attn_weights = None   # 初始化自注意力权重为None

        residual = hidden_states   # 保存原始的隐藏状态作为残差连接的输入

        hidden_states = self.cross_attn_layer_norm(hidden_states)  # 使用层归一化处理隐藏状态

        # 执行交叉注意力计算
        hidden_states, cross_attn_weights = self.cross_attn(
            query=self.with_pos_embed(hidden_states, query_position_embeddings),
            key=self.with_pos_embed(encoder_hidden_states[level_index], position_embeddings[level_index]),
            value=encoder_hidden_states[level_index],
            attn_mask=encoder_attention_mask,
            key_padding_mask=None,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对隐藏状态应用Dropout
        hidden_states = residual + hidden_states  # 执行残差连接

        # Self Attention Block
        residual = hidden_states   # 保存当前隐藏状态作为自注意力的残差连接输入

        hidden_states = self.self_attn_layer_norm(hidden_states)  # 使用层归一化处理隐藏状态

        # 执行自注意力计算
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对隐藏状态应用Dropout
        hidden_states = residual + hidden_states  # 执行残差连接

        # Fully Connected
        residual = hidden_states   # 保存当前隐藏状态作为全连接的残差连接输入

        hidden_states = self.final_layer_norm(hidden_states)  # 使用层归一化处理隐藏状态
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 使用激活函数处理全连接层1
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 对全连接结果应用Dropout
        hidden_states = self.fc2(hidden_states)  # 执行全连接层2
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对全连接结果应用Dropout
        hidden_states = residual + hidden_states  # 执行残差连接

        outputs = (hidden_states,)  # 将隐藏状态作为输出的第一个元素

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)  # 如果需要输出注意力权重，则将自注意力和交叉注意力的权重添加到输出中

        return outputs  # 返回最终的输出结果

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
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                输入到层的张量，形状为 `(seq_len, batch, embed_dim)`。
            attention_mask (`torch.FloatTensor`):
                注意力遮罩张量，形状为 `(1, seq_len, tgt_len, src_len)`。
            position_embeddings (`torch.FloatTensor`, *可选*):
                添加到掩码注意力层中键的位置嵌入。
            query_position_embeddings (`torch.FloatTensor`, *可选*):
                添加到自注意力层中查询和键的位置嵌入。
            encoder_hidden_states (`torch.FloatTensor`):
                层的交叉注意力输入张量，形状为 `(seq_len, batch, embed_dim)`。
            encoder_attention_mask (`torch.FloatTensor`):
                编码器注意力遮罩张量，大小为 `(1, seq_len, tgt_len, src_len)`。
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 以获取更多细节。
        """

        # 如果使用预归一化
        if self.pre_norm:
            # 调用预归一化前向传播函数
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
            # 调用后归一化前向传播函数
            outputs = self.forward_post(
                hidden_states=hidden_states,
                level_index=level_index,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

        # 返回模型层的输出
        return outputs
# 定义一个基于 Transformer 的解码器类，包含多个层
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

        # 初始化解码器的配置和参数
        self.config = config
        self.mask_feature_size = config.mask_feature_size  # 掩码特征大小
        self.dropout = config.dropout  # 丢弃率
        self.layerdrop = config.dropout  # 层丢弃率
        self.num_feature_levels = 3  # 级别嵌入数（3个规模的嵌入）
        self.decoder_layers = config.decoder_layers - 1  # 解码层数

        # 创建解码器层列表，每层是一个 Mask2FormerMaskedAttentionDecoderLayer 实例
        self.layers = nn.ModuleList(
            [Mask2FormerMaskedAttentionDecoderLayer(self.config) for _ in range(self.decoder_layers)]
        )
        self.layernorm = nn.LayerNorm(config.hidden_dim)  # 归一化层

        # 创建掩码预测器，用于生成掩码预测
        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=config.hidden_dim,
            num_heads=config.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

        self.gradient_checkpointing = False  # 梯度检查点开关

    # 前向传播函数定义
    def forward(
        self,
        inputs_embeds: torch.Tensor = None,  # 输入嵌入
        multi_stage_positional_embeddings: torch.Tensor = None,  # 多阶段位置嵌入
        pixel_embeddings: torch.Tensor = None,  # 像素嵌入
        encoder_hidden_states: torch.Tensor = None,  # 编码器隐藏状态
        query_position_embeddings: torch.Tensor = None,  # 查询位置嵌入
        feature_size_list: List = None,  # 特征大小列表
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
# 从 transformers.models.maskformer.modeling_maskformer.PredictionBlock 复制，将 MaskFormer 改为 Mask2Former
class Mask2FormerPredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # 保持子模块索引，仿佛是顺序块的一部分
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    # 前向传播函数定义
    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        # 应用每一层线性变换和激活函数到输入张量上
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                The input dimensions.
            hidden_dim (`int`):
                The hidden dimensions.
            output_dim (`int`):
                The output dimensions.
            num_layers (int, *optional*, defaults to 3):
                The number of layers.
        """
        super().__init__()  # 调用父类的初始化方法

        # 定义每层的输入和输出维度
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []  # 初始化存储层的列表
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            # 根据层数选择激活函数，最后一层使用恒等映射作为激活函数
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            # 创建 Mask2FormerPredictionBlock 实例作为当前层
            layer = Mask2FormerPredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)  # 将当前层添加到层列表中

            # 为了向后兼容，特别是当类继承自 nn.Sequential 时
            # 在 nn.Sequential 的子类中，层的名称是它在序列中的索引
            # 在 nn.Module 的子类中，它们根据分配给它们的实例属性命名，例如 self.my_layer_name = Layer()
            # 由于不能给实例属性整数名称（例如 self.0 是不允许的），因此需要显式注册模块
            self.add_module(str(i), layer)  # 将当前层以索引 i 的字符串形式注册为模块

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input  # 初始化输入数据为隐藏状态

        # 逐层计算前向传播
        for layer in self.layers:
            hidden_state = layer(hidden_state)  # 应用当前层到隐藏状态

        return hidden_state  # 返回最终的隐藏状态作为输出
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

        # Initialize the mask_embedder using Mask2FormerMLPPredictionHead module
        self.mask_embedder = Mask2FormerMLPPredictionHead(self.hidden_size, self.hidden_size, mask_feature_size)

    def forward(self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: int = None):
        # Generate mask embeddings using the mask_embedder
        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))

        # Check if the model is in tracing mode or compiling mode for TorchScript
        is_tracing = (
            torch.jit.is_tracing()
            or isinstance(outputs, torch.fx.Proxy)
            or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
        )

        # Sum up over the channels using either a loop (if not using Torch 2.1 or higher) or einsum
        if is_tracing and not is_torch_greater_or_equal_than_2_1:
            # Loop through channels and accumulate outputs_mask
            batch_size, num_queries, num_channels = mask_embeddings.shape
            _, _, height, width = pixel_embeddings.shape
            outputs_mask = torch.zeros((batch_size, num_queries, height, width), device=mask_embeddings.device)
            for c in range(num_channels):
                outputs_mask += mask_embeddings[..., c][..., None, None] * pixel_embeddings[:, None, c]
        else:
            # Use einsum to perform tensor contraction
            outputs_mask = torch.einsum("bqc, bchw -> bqhw", mask_embeddings, pixel_embeddings)

        # Resize the outputs_mask to attention_mask_target_size using bilinear interpolation
        attention_mask = nn.functional.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        # Apply sigmoid activation and reshape for multi-head attention compatibility
        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # Binarize the attention_mask based on a threshold of 0.5 and detach it from the computation graph
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()

        # Return the generated outputs_mask and attention_mask
        return outputs_mask, attention_mask


class Mask2FormerTransformerModule(nn.Module):
    """
    The Mask2Former's transformer module.
    """
    def __init__(self, in_features: int, config: Mask2FormerConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        # 初始化位置编码器，使用 Mask2FormerSinePositionEmbedding 类
        self.position_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        # 初始化查询的嵌入层，使用 nn.Embedding 类
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        # 初始化查询的特征嵌入层，使用 nn.Embedding 类
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)
        # 输入投影层列表
        self.input_projections = []

        # 根据 num_feature_levels 创建输入投影层
        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                # 如果输入特征维度不等于隐藏维度或者配置要求强制投影，则添加卷积层
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                # 否则添加空的序列（空的 nn.Sequential()）
                self.input_projections.append(nn.Sequential())

        # 初始化解码器，使用 Mask2FormerMaskedAttentionDecoder 类
        self.decoder = Mask2FormerMaskedAttentionDecoder(config=config)
        # 等级嵌入层，使用 nn.Embedding 类
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Mask2FormerMaskedAttentionDecoderOutput:
        # 多尺度特征列表
        multi_stage_features = []
        # 多尺度位置嵌入列表
        multi_stage_positional_embeddings = []
        # 尺寸列表
        size_list = []

        # 遍历 num_feature_levels
        for i in range(self.num_feature_levels):
            # 记录每个特征的尺寸
            size_list.append(multi_scale_features[i].shape[-2:])
            # 获取多尺度位置嵌入并展平
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            # 获取多尺度特征并展平，加上等级嵌入
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # 转置操作，将维度重新排列为 (height*width, batch_size, num_channels)
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        # 获取 batch_size
        _, batch_size, _ = multi_stage_features[0].shape

        # 查询嵌入，扩展为 [num_queries, batch_size, hidden_dim]
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # 查询特征嵌入，扩展为 [num_queries, batch_size, hidden_dim]
        query_features = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # 调用解码器进行解码操作
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

        # 返回解码器的输出
        return decoder_output
# 定义一个长字符串，描述了该模型是一个 PyTorch 的 `torch.nn.Module` 的子类，用于普通的 PyTorch 模型使用，并引用了 PyTorch 文档以获取有关一般用法和行为的信息。
MASK2FORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Mask2FormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义了另一个长字符串，描述了模型的输入参数和可选参数的详细说明。
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

# 定义了一个模型类 `Mask2FormerModel`，继承自 `Mask2FormerPreTrainedModel`，表示 Mask2Former 模型的主体结构。
@add_start_docstrings(
    "The bare Mask2Former Model outputting raw hidden-states without any specific head on top.",
    MASK2FORMER_START_DOCSTRING,
)
class Mask2FormerModel(Mask2FormerPreTrainedModel):
    main_input_name = "pixel_values"

    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        # 初始化模型的像素级模块和 Transformer 模块，使用给定的配置参数
        self.pixel_level_module = Mask2FormerPixelLevelModule(config)
        self.transformer_module = Mask2FormerTransformerModule(in_features=config.feature_size, config=config)

        self.post_init()

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
    "The Mask2Former Model with heads on top for instance/semantic/panoptic segmentation.",
    MASK2FORMER_START_DOCSTRING,
    )
    class Mask2FormerForUniversalSegmentation(Mask2FormerPreTrainedModel):
        main_input_name = "pixel_values"

        def __init__(self, config: Mask2FormerConfig):
            super().__init__(config)
            # 使用给定配置初始化 Mask2FormerModel 模型
            self.model = Mask2FormerModel(config)

            # 初始化损失权重字典，包括交叉熵损失、Mask 损失和 Dice 损失的权重
            self.weight_dict: Dict[str, float] = {
                "loss_cross_entropy": config.class_weight,
                "loss_mask": config.mask_weight,
                "loss_dice": config.dice_weight,
            }

            # 创建一个线性层用于类别预测，输出维度为 config.num_labels + 1
            self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)

            # 初始化损失函数，使用 Mask2FormerLoss 类，传入配置和权重字典
            self.criterion = Mask2FormerLoss(config=config, weight_dict=self.weight_dict)
            # 调用后初始化方法
            self.post_init()

        def get_loss_dict(
            self,
            masks_queries_logits: Tensor,
            class_queries_logits: Tensor,
            mask_labels: Tensor,
            class_labels: Tensor,
            auxiliary_predictions: Dict[str, Tensor],
        ) -> Dict[str, Tensor]:
            # 计算损失字典，调用 self.criterion 对象的 __call__ 方法
            loss_dict: Dict[str, Tensor] = self.criterion(
                masks_queries_logits=masks_queries_logits,
                class_queries_logits=class_queries_logits,
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_predictions,
            )

            # 根据 self.weight_dict 中的权重对每个损失进行加权，包括辅助损失
            for key, weight in self.weight_dict.items():
                for loss_key, loss in loss_dict.items():
                    if key in loss_key:
                        loss *= weight

            return loss_dict

        def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
            # 计算总损失，将损失字典中的所有值相加
            return sum(loss_dict.values())

        def get_auxiliary_logits(self, classes: torch.Tensor, output_masks: torch.Tensor):
            # 获取辅助预测的 logits 列表
            auxiliary_logits: List[Dict(str, Tensor)] = []

            # 遍历输出的 masks 和 classes，排除最后一个元素（用于辅助任务）
            for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
                auxiliary_logits.append({"masks_queries_logits": aux_binary_masks, "class_queries_logits": aux_classes})

            return auxiliary_logits

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
        ):
            # 正向传播函数，接收多个参数，返回 Mask2FormerForUniversalSegmentationOutput 对象
```