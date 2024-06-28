# `.\models\oneformer\modeling_oneformer.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明，版权归 SHI Labs 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可 2.0 版本使用本文件，除非符合许可，否则不得使用此文件
# 您可以在以下网址获取许可的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 不提供任何明示或暗示的担保或条件
""" PyTorch OneFormer 模型 """
# 导入必要的库和模块
import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

# 导入内部模块和函数
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_accelerate_available,
    is_scipy_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig

# 如果加速库可用，导入部分状态和减少函数
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

# 获取模块内的日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点名称
_CONFIG_FOR_DOC = "OneFormerConfig"
_CHECKPOINT_FOR_DOC = "shi-labs/oneformer_ade20k_swin_tiny"

# 预训练模型的存档列表
ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shi-labs/oneformer_ade20k_swin_tiny",
    # 更多 OneFormer 模型请查看 https://huggingface.co/models?filter=oneformer
]

# 如果 SciPy 可用，导入线性求解模块
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


# 函数定义：克隆模块多次并返回模块列表
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# 从 transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention 复制的函数
# 多尺度可变形注意力机制
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # 根据空间形状将值分割成列表
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    # 采样位置的网格化表达
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # 对每个空间尺寸进行枚举，获取当前层的高度和宽度
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        # 将采样网格在当前层的维度上转置并展平
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        # 使用双线性插值对采样值进行网格采样
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    # 调整注意力权重的维度顺序并展平
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()
# Copied from transformers.models.maskformer.modeling_maskformer.dice_loss
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    r"""
    Compute the DICE loss, similar to generalized IOU for masks as follows:

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
    # 将输入经过 sigmoid 函数转换，并展平成一维
    probs = inputs.sigmoid().flatten(1)
    # 计算 DICE 损失的分子部分
    numerator = 2 * (probs * labels).sum(-1)
    # 计算 DICE 损失的分母部分
    denominator = probs.sum(-1) + labels.sum(-1)
    # 计算最终的 DICE 损失
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 对 batch 中每个样本的损失求平均，再进行归一化
    loss = loss.sum() / num_masks
    return loss


# Copied from transformers.models.mask2former.modeling_mask2former.sigmoid_cross_entropy_loss
def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    # 使用 BCEWithLogitsLoss 计算交叉熵损失，不进行损失函数的 reduction
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    cross_entropy_loss = criterion(inputs, labels)
    # 对每个样本的损失取平均，并进行归一化
    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss


# Copied from transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    A pair wise version of the dice loss, see `dice_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        `torch.Tensor`: The computed loss between each pairs.
    """
    # 将输入经过 sigmoid 函数转换，并展平成一维
    inputs = inputs.sigmoid().flatten(1)
    # 计算 pairwise DICE 损失的分子部分
    numerator = 2 * torch.matmul(inputs, labels.T)
    # 使用广播来获取 [num_queries, NUM_CLASSES] 的矩阵
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    # 计算 pairwise DICE 损失
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss
# Copied from transformers.models.mask2former.modeling_mask2former.pair_wise_sigmoid_cross_entropy_loss
def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""
    A pair wise version of the cross entropy loss, see `sigmoid_cross_entropy_loss` for usage.

    Args:
        inputs (`torch.Tensor`):
            A tensor representing a mask.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss between each pairs.
    """

    # 计算输入张量的高度和宽度
    height_and_width = inputs.shape[1]

    # 使用二元交叉熵损失函数，设置不进行汇总
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    # 计算正类的交叉熵损失
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    # 计算负类的交叉熵损失
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    # 计算正类的损失
    loss_pos = torch.matmul(cross_entropy_loss_pos / height_and_width, labels.T)
    # 计算负类的损失
    loss_neg = torch.matmul(cross_entropy_loss_neg / height_and_width, (1 - labels).T)
    # 总损失为正负类损失之和
    loss = loss_pos + loss_neg
    return loss


# Copied from transformers.models.mask2former.modeling_mask2former.sample_point
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
        height_grid, width_grid)):
            A tensor that contains features for points in `point_coordinates`.
    """
    if point_coordinates.dim() == 3:
        add_dim = True
        # 在第二维度上插入一个维度
        point_coordinates = point_coordinates.unsqueeze(2)

    # 使用双线性插值，通过点坐标在输入特征图中获取特征
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    if add_dim:
        # 如果添加了维度，则压缩第三个维度
        point_features = point_features.squeeze(3)

    return point_features


# Refactored from https://github.com/SHI-Labs/OneFormer/blob/33ebb56ed34f970a30ae103e786c0cb64c653d9a/oneformer/modeling/matcher.py#L93
class OneFormerHungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544
    ):
        """
        Initialize the OneFormer Hungarian Matcher.

        Args:
            cost_class (`float`):
                Cost associated with class differences.
            cost_mask (`float`):
                Cost associated with mask differences.
            cost_dice (`float`):
                Cost associated with Dice similarity differences.
            num_points (`int`):
                Number of points used in the matcher.
        """
        super().__init__()
        # 初始化匈牙利匹配器的各个成本参数和点数
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points
    ):
        """
        This class computes an assignment between the labels and the predictions of the network.

        For efficiency reasons, the labels don't include the no_object. Because of this, in general, there are more
        predictions than labels. In this case, we do a 1-to-1 matching of the best predictions, while the others are
        un-matched (and thus treated as non-objects).

        Params:
            cost_class (float, *optional*, defaults to 1.0):
                This is the relative weight of the classification error in the matching cost.
            cost_mask (float, *optional*,  defaults to 1.0):
                This is the relative weight of the sigmoid ce loss of the binary mask in the matching cost.
            cost_dice (float, *optional*, defaults to 1.0):
                This is the relative weight of the dice loss of the binary mask in the matching cost
            num_points (int, *optional*, defaults to 12544):
                Number of points to be sampled for dice and mask loss matching cost.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 检查参数，如果所有的损失权重都为0，则抛出异常
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")
        # 设置分类误差的匹配成本权重
        self.cost_class = cost_class
        # 设置二进制掩码的 Sigmoid 交叉熵损失的匹配成本权重
        self.cost_mask = cost_mask
        # 设置二进制掩码的 Dice 损失的匹配成本权重
        self.cost_dice = cost_dice
        # 设置用于 Dice 和掩码损失匹配成本的采样点数
        self.num_points = num_points

    @torch.no_grad()
# 定义一个名为 OneFormerLoss 的新的神经网络模块，继承自 nn.Module 类
class OneFormerLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,  # 类别数，用于分类损失计算
        matcher: OneFormerHungarianMatcher,  # 用于计算预测和标签之间匹配的模块
        weight_dict: Dict[str, float],  # 不同损失的权重字典
        eos_coef: float,  # 空类别的权重系数
        num_points: int,  # 用于Dice和mask损失计算的采样点数
        oversample_ratio: float,  # 点损失计算所需的过采样比率
        importance_sample_ratio: float,  # 点损失计算所需的重要性采样比率
        contrastive_temperature: float = None,  # 用于缩放对比损失的温度参数
    ):
        """
        This class computes the losses using the class predictions, mask predictions and the contrastive queries.

        Oneformer calculates the classification CE loss on the class predictions. Mask predictions are used for
        calculating the binary CE loss and dice loss. The contrastive queries are used for calculating the contrastive
        loss.

        Args:
            num_labels (`int`):
                The number of classes.
            matcher (`OneFormerHungarianMatcher`):
                A torch module that computes the assigments between the predictions and labels.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
            eos_coef (`float`):
                Weight to apply to the null class.
            num_points (`int`):
                Number of points to be sampled for dice and mask loss calculations.
            oversample_ratio (`float`):
                Required for pointwise loss calculation.
            importance_sample_ratio (`float`):
                Required for pointwise loss calculation.
            contrastive_temperature (`float`):
                Temperature for scaling the contrastive logits.
        """
        # 检查是否需要后端库 "scipy"
        requires_backends(self, ["scipy"])
        # 调用父类的初始化方法
        super().__init__()
        # 初始化模块内部变量
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        # 创建一个权重向量，用于控制空类别的权重
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        # 将权重向量注册为模块的缓冲区（buffer）
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        # 设置点损失（pointwise mask loss）的参数
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.contrastive_temperature = contrastive_temperature
        # 如果设置了对比损失的温度参数，则初始化一个对数缩放参数
        if self.contrastive_temperature is not None:
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / contrastive_temperature)))

    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        """
        Compute the maximum values along each axis of a 2D list.

        Args:
            the_list (`List[List[int]]`): 二维列表，用于计算每个轴向的最大值

        Returns:
            `List[int]`: 包含每个轴向最大值的列表
        """
        # 初始化最大值列表为第一个子列表
        maxes = the_list[0]
        # 遍历剩余的子列表，更新每个轴向的最大值
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes
    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # get the maximum size in the batch
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        # compute final batch shape including batch size and max image dimensions
        batch_shape = [batch_size] + max_size
        b, _, h, w = batch_shape
        # get metadata from the first tensor
        dtype = tensors[0].dtype
        device = tensors[0].device
        # create a tensor filled with zeros to hold padded images
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        # create a mask for padding regions initialized to True
        padding_masks = torch.ones((b, h, w), dtype=torch.bool, device=device)
        # pad each tensor in the batch to match the dimensions of the largest tensor
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
            # update padding mask to mark actual data regions as False
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_contrastive(self, contrastive_queries_logits: Tensor, text_queries: Tensor):
        """Compute the query-text contrastive loss.

        Args:
            contrastive_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, hidden_dim`
            text_queries (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, hidden_dim`
        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_contrastive** -- The query-text contrastive loss computed using task-guided queries
                                    and text queries derived from input text list.
        """

        # convert contrastive queries logits to float and normalize across hidden_dim
        image_queries = contrastive_queries_logits.float()
        image_queries = nn.functional.normalize(image_queries.flatten(1), dim=-1)
        # normalize text queries across hidden_dim
        text_queries = nn.functional.normalize(text_queries.flatten(1), dim=-1)

        # clamp and exponentiate logit scale value to prevent exploding gradients
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)

        # compute logits for text and image queries
        logits_per_text = torch.matmul(text_queries, image_queries.t()) * logit_scale
        logits_per_img = logits_per_text.t()

        # compute cross-entropy loss for image and text queries
        loss_img = nn.functional.cross_entropy(
            logits_per_img, torch.arange(len(logits_per_img), device=logits_per_text.device)
        )
        loss_text = nn.functional.cross_entropy(
            logits_per_text, torch.arange(len(logits_per_text), device=logits_per_text.device)
        )

        # total contrastive loss is the sum of losses for image and text queries
        loss_contrastive = loss_img + loss_text

        # pack losses into a dictionary for returning
        losses = {"loss_contrastive": loss_contrastive}
        return losses

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ):
        # This method is not fully provided in the snippet and thus cannot be annotated.
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
        pred_logits = class_queries_logits  # Assign class_queries_logits to pred_logits
        batch_size, num_queries, _ = pred_logits.shape  # Extract batch_size and num_queries from pred_logits shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)  # Define cross entropy loss criterion

        # Obtain indices for permutation using Hungarian matcher
        idx = self._get_predictions_permutation_indices(indices)

        # Concatenate target classes from class_labels using indices
        # shape = (batch_size, num_queries)
        target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])

        # Initialize target_classes tensor with num_classes and place on device
        # shape = (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_classes, dtype=torch.int64, device=pred_logits.device
        )

        # Assign values from target_classes_o to target_classes based on idx
        target_classes[idx] = target_classes_o

        # Transpose pred_logits to shape (batch_size, num_labels, num_queries)
        pred_logits_transposed = pred_logits.transpose(1, 2)

        # Compute cross entropy loss using transposed logits and target classes
        loss_ce = criterion(pred_logits_transposed, target_classes)

        # Prepare losses dictionary with the computed cross entropy loss
        losses = {"loss_cross_entropy": loss_ce}
        return losses
    ) -> Dict[str, Tensor]:
        """计算与掩码相关的损失，使用焦点和Dice损失。

        Args:
            masks_queries_logits (`torch.Tensor`):
                形状为 `batch_size, num_queries, height, width` 的张量
                掩码查询的预测logits
            mask_labels (`torch.Tensor`):
                形状为 `(labels, height, width)` 的掩码标签列表
            indices (`Tuple[np.array])`:
                由匈牙利匹配器计算得到的索引
            num_masks (`int)`:
                掩码的数量，用于归一化

        Returns:
            `Dict[str, Tensor]`: 包含两个键的 `torch.Tensor` 字典:
            - **loss_mask** -- 使用sigmoid交叉熵损失在预测掩码和真实掩码之间计算的损失
            - **loss_dice** -- 使用Dice损失在预测掩码和真实掩码之间计算的损失
        """
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # 形状为 (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # 形状为 (batch_size, num_queries, height, width)
        # 对所有目标进行填充并将其堆叠到 num_labels 维度
        # 将预测结果插值到目标大小，我们必须添加一个维度来使用 interpolate
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # 采样点的坐标
            point_coords = self.sample_points_using_uncertainty(
                pred_masks,
                self.calculate_uncertainty,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # 获取地面真实标签
            point_labels = sample_point(target_masks, point_coords, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coords, align_corners=False).squeeze(1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    # 从transformers.models.mask2former.modeling_mask2former.Mask2FormerLoss.calculate_uncertainty复制而来
    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算不确定性分数，根据 Mask2Former 论文，不确定性被定义为预测logit与0.0之间的L1距离，
        针对`logits`中前景类别`classes`的预测。

        Args:
            logits (`torch.Tensor`):
            形状为 (R, 1, ...) 的张量，用于特定类别或类别无关，其中 R 是所有图像中预测的总数目，
            C 是前景类别的数量。值为logits。

        Returns:
            scores (`torch.Tensor`): 形状为 (R, 1, ...) 的张量，包含不确定性分数，其中不确定性最高的位置具有最高的分数。
        """
        uncertainty_scores = -(torch.abs(logits))
        return uncertainty_scores

    # 从 transformers.models.mask2former.modeling_mask2former.Mask2FormerLoss.sample_points_using_uncertainty 复制而来
    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    def _get_predictions_permutation_indices(self, indices):
        """
        This method generates permutation indices for predictions based on the provided indices.

        Args:
            indices (list):
                List of tuples containing indices for predictions and targets.

        Returns:
            batch_indices (torch.Tensor):
                Indices indicating batch-wise association of predictions.
            predictions_indices (torch.Tensor):
                Indices indicating the order of predictions after permutation.
        """
        # Create batch indices for predictions
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        
        # Create indices for predictions based on the provided `indices` list
        predictions_indices = torch.cat([src for (src, _) in indices])
        
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        """
        This method generates permutation indices for targets based on the provided indices.

        Args:
            indices (list):
                List of tuples containing indices for predictions and targets.

        Returns:
            batch_indices (torch.Tensor):
                Indices indicating batch-wise association of targets.
            target_indices (torch.Tensor):
                Indices indicating the order of targets after permutation.
        """
        # Create batch indices for targets
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        
        # Create indices for targets based on the provided `indices` list
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        
        return batch_indices, target_indices
    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        contrastive_queries_logits: Tensor,
        mask_labels: List[Tensor],
        class_labels: List[Tensor],
        text_queries: Tensor,
        auxiliary_predictions: Optional[Dict[str, Tensor]] = None,
        calculate_contrastive_loss: bool = True,
    ):
        """
        Defines the forward pass of the model.

        Args:
            masks_queries_logits (Tensor): Logits for mask prediction queries.
            class_queries_logits (Tensor): Logits for class prediction queries.
            contrastive_queries_logits (Tensor): Logits for contrastive learning queries.
            mask_labels (List[Tensor]): List of mask labels.
            class_labels (List[Tensor]): List of class labels.
            text_queries (Tensor): Queries related to text inputs.
            auxiliary_predictions (Optional[Dict[str, Tensor]]): Optional auxiliary predictions.
            calculate_contrastive_loss (bool): Flag indicating whether to calculate contrastive loss.

        Returns:
            None
        """
        pass

    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.

        Args:
            class_labels (torch.Tensor): Tensor of class labels.
            device (torch.device): Device on which tensors reside.

        Returns:
            torch.Tensor: Average number of masks per batch, normalized.
        """
        # Calculate the total number of masks across the batch
        num_masks = sum([len(classes) for classes in class_labels])

        # Convert to a tensor and move to specified device
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=device)

        # Default world size
        world_size = 1

        # Adjust based on distributed settings if available
        if is_accelerate_available():  # Check if using NVIDIA's Accelerate
            if PartialState._shared_state != {}:
                num_masks = reduce(num_masks)  # Reduces across distributed processes
                world_size = PartialState().num_processes

        # Normalize the number of masks per batch, ensuring it's at least 1
        num_masks = torch.clamp(num_masks / world_size, min=1)

        return num_masks
# 定义一个数据类 OneFormerTransformerDecoderOutput，继承自BaseModelOutput，用于Transformer解码器的输出
@dataclass
class OneFormerTransformerDecoderOutput(BaseModelOutput):
    """
    Base class for outputs of the Transformer decoder. This class adds attributes for class predictions, mask
    predictions and contrastive logits to BaseModelOutputWithCrossAttentions.

    Args:
        object_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`):
            Queries representation for the region proposals.
        contrastive_logits (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_dim)`):
            Queries representation for the contrastive loss.
        prediction_masks (`torch.FloatTensor` of shape `(batch_size, num_queries, height, width)`):
            Mask predictions from last layer of the transformer decoder.
        prediction_class (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes+1)`):
            Class predictions from last layer of the transformer decoder.
        auxiliary_predictions (Tuple of Dict of `str, torch.FloatTensor`, *optional*):
            Tuple of class and mask predictions from each layer of the transformer decoder.
    """

    # Transformer解码器的输出属性
    object_queries: torch.FloatTensor = None  # 区域建议的查询表示
    contrastive_logits: Optional[torch.FloatTensor] = None  # 对比损失的查询表示
    prediction_masks: torch.FloatTensor = None  # Transformer解码器最后一层的预测掩码
    prediction_class: torch.FloatTensor = None  # Transformer解码器最后一层的类别预测
    auxiliary_predictions: Optional[Tuple[Dict[str, torch.FloatTensor]]] = None  # 各层次的类别和掩码预测的元组



# 定义一个数据类 OneFormerPixelDecoderOutput，继承自ModelOutput，表示OneFormer的像素解码器模块输出
@dataclass
# 从transformers.models.mask2former.modeling_mask2former.Mask2FormerPixelDecoderOutput复制到OneFormerPixelDecoderOutput
class OneFormerPixelDecoderOutput(ModelOutput):
    """
    OneFormer's pixel decoder module output, practically a Multi-Scale Deformable Attention based decoder. It returns
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

    # OneFormer的像素解码器模块输出属性
    multi_scale_features: Tuple[torch.FloatTensor] = None  # 多尺度特征的元组，包括1/8、1/16、1/32比例的特征
    mask_features: torch.FloatTensor = None  # 形状为(batch_size, num_channels, height, width)的掩码特征，来自像素解码器的最后一层
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 像素解码器每一层的注意力权重的元组，形状为(batch_size, num_heads, sequence_length, sequence_length)



# 定义一个数据类 OneFormerPixelLevelModuleOutput，继承自ModelOutput，表示OneFormer的像素级模块输出
@dataclass
class OneFormerPixelLevelModuleOutput(ModelOutput):
    """
    OneFormer's pixel level module output. It returns both the last and (optionally) the hidden states from the
    """

    # OneFormer的像素级模块输出属性
    # 此处省略了该类的详细描述和参数说明部分，需要根据实际情况添加
    # `encoder` 和 `decoder` 分别是编码器和解码器模型的特征表示。默认情况下，编码器是Swin/Dinat骨干网络，解码器是基于多尺度可变形注意力的解码器。
    
    Args:
        encoder_features (List of `(torch.FloatTensor)`):
            一个列表，包含了 `torch.FloatTensor` 类型的特征图，形状为 `(batch_size, num_channels, height, width)`。
            这些特征图是模型在每个阶段输出的隐藏状态（也称为特征图）。
    
        decoder_features (List of `(torch.FloatTensor)`):
            一个列表，包含了 `torch.FloatTensor` 类型的特征图，形状为 `(batch_size, num_channels, height, width)`。
            这些特征图是模型在每个阶段输出的隐藏状态（也称为特征图）。
    
        decoder_last_feature (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width))`:
            来自最后一个像素解码层的1/4尺度特征图。
# 使用 dataclass 装饰器定义 OneFormerModelOutput 类，用于保存一个模型的输出数据
@dataclass
class OneFormerModelOutput(ModelOutput):
    """
    Class for outputs of [`OneFormerModel`]. This class returns all the needed hidden states to compute the logits.

    """

    # 编码器的隐藏状态，类型为可选的元组，包含了 torch.FloatTensor 类型的数据
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 像素解码器的隐藏状态，类型为可选的元组，包含了 torch.FloatTensor 类型的数据
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 变换器解码器的隐藏状态，类型为可选的 torch.FloatTensor 类型
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    # 变换器解码器的对象查询，类型为 torch.FloatTensor
    transformer_decoder_object_queries: torch.FloatTensor = None
    # 变换器解码器的对比查询，类型为可选的 torch.FloatTensor
    transformer_decoder_contrastive_queries: Optional[torch.FloatTensor] = None
    # 变换器解码器的掩码预测，类型为 torch.FloatTensor
    transformer_decoder_mask_predictions: torch.FloatTensor = None
    # 变换器解码器的类别预测，类型为 torch.FloatTensor
    transformer_decoder_class_predictions: torch.FloatTensor = None
    # 变换器解码器的辅助预测，类型为可选的字典元组，包含了 torch.FloatTensor 类型的数据
    transformer_decoder_auxiliary_predictions: Optional[Tuple[Dict[str, torch.FloatTensor]]] = None
    # 文本查询，类型为可选的 torch.FloatTensor
    text_queries: Optional[torch.FloatTensor] = None
    # 任务令牌，类型为 torch.FloatTensor
    task_token: torch.FloatTensor = None
    # 注意力权重，类型为可选的元组，包含了 torch.FloatTensor 类型的数据
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器定义 OneFormerForUniversalSegmentationOutput 类，用于保存一个模型的输出数据
@dataclass
class OneFormerForUniversalSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`OneFormerForUniversalSegmentationOutput`].

    This output can be directly passed to [`~OneFormerImageProcessor.post_process_semantic_segmentation`] or
    [`~OneFormerImageProcessor.post_process_instance_segmentation`] or
    [`~OneFormerImageProcessor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~OneFormerImageProcessor] for details regarding usage.

    """

    # 损失值，类型为可选的 torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None
    # 类别查询的对数，类型为 torch.FloatTensor
    class_queries_logits: torch.FloatTensor = None
    # 掩码查询的对数，类型为 torch.FloatTensor
    masks_queries_logits: torch.FloatTensor = None
    # 辅助预测列表，每个元素是包含了 torch.FloatTensor 类型数据的字典
    auxiliary_predictions: List[Dict[str, torch.FloatTensor]] = None
    # 编码器的隐藏状态，类型为可选的元组，包含了 torch.FloatTensor 类型的数据
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 像素解码器的隐藏状态，类型为可选的列表，每个元素是 torch.FloatTensor 类型的数据
    pixel_decoder_hidden_states: Optional[List[torch.FloatTensor]] = None
    # 变换器解码器的隐藏状态，类型为可选的 torch.FloatTensor
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    # 变换器解码器的对象查询，类型为 torch.FloatTensor
    transformer_decoder_object_queries: torch.FloatTensor = None
    # 变换器解码器的对比查询，类型为可选的 torch.FloatTensor
    transformer_decoder_contrastive_queries: Optional[torch.FloatTensor] = None
    # 变换器解码器的掩码预测，类型为 torch.FloatTensor
    transformer_decoder_mask_predictions: torch.FloatTensor = None
    # 变换器解码器的类别预测，类型为 torch.FloatTensor
    transformer_decoder_class_predictions: torch.FloatTensor = None
    # 变换器解码器的辅助预测，类型为可选的列表，每个元素是包含了 torch.FloatTensor 类型数据的字典
    transformer_decoder_auxiliary_predictions: Optional[List[Dict[str, torch.FloatTensor]]] = None
    # 文本查询，类型为可选的 torch.FloatTensor
    text_queries: Optional[torch.FloatTensor] = None
    # 任务令牌，类型为 torch.FloatTensor
    task_token: torch.FloatTensor = None
    # 注意力权重，类型为可选的元组，每个元素是包含了 torch.FloatTensor 类型数据的元组
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


# 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrFrozenBatchNorm2d 修改而来，用于 OneFormerPixelDecoder 的冻结批量归一化操作
class OneFormerPixelDecoderFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """
    # 初始化函数，用于初始化一个 BatchNorm2d 对象
    def __init__(self, n):
        # 调用父类的初始化函数
        super().__init__()
        # 注册一个名为 "weight" 的缓冲区，初始化为全1的张量，形状为 (n,)
        self.register_buffer("weight", torch.ones(n))
        # 注册一个名为 "bias" 的缓冲区，初始化为全0的张量，形状为 (n,)
        self.register_buffer("bias", torch.zeros(n))
        # 注册一个名为 "running_mean" 的缓冲区，初始化为全0的张量，形状为 (n,)
        self.register_buffer("running_mean", torch.zeros(n))
        # 注册一个名为 "running_var" 的缓冲区，初始化为全1的张量，形状为 (n,)
        self.register_buffer("running_var", torch.ones(n))

    # 加载模型状态的函数，用于从给定的 state_dict 中加载模型的状态
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 构建跟踪批次数的键名
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # 如果 state_dict 中存在跟踪批次数的键名，则从 state_dict 中删除该键名
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类的加载函数，加载模型状态
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # 前向传播函数，用于执行 Batch Normalization 的前向计算
    def forward(self, x):
        # 将权重张量重塑为 (1, n, 1, 1) 的形状
        weight = self.weight.reshape(1, -1, 1, 1)
        # 将偏置张量重塑为 (1, n, 1, 1) 的形状
        bias = self.bias.reshape(1, -1, 1, 1)
        # 将 running_var 张量重塑为 (1, n, 1, 1) 的形状
        running_var = self.running_var.reshape(1, -1, 1, 1)
        # 将 running_mean 张量重塑为 (1, n, 1, 1) 的形状
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        # 设置一个很小的数 epsilon，用于稳定计算
        epsilon = 1e-5
        # 计算 scale 参数，用于归一化输入 x
        scale = weight * (running_var + epsilon).rsqrt()
        # 计算 bias 参数，用于调整归一化后的输出
        bias = bias - running_mean * scale
        # 执行 Batch Normalization 的前向计算，返回归一化后的结果
        return x * scale + bias
# 从 transformers.models.detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention 修改而来，用于 OneFormerPixelDecoderEncoder
class OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
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
        dim_per_head = embed_dim // num_heads
        # 检查 dim_per_head 是否为2的幂
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        # 图像块化的步长
        self.im2col_step = 128

        # 初始化模型参数
        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        # 线性层，用于生成采样偏移量
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        # 线性层，用于计算注意力权重
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        # 线性层，用于将输入数据映射到输出数据
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        # 线性层，用于最终输出映射
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    # 将位置编码加到输入张量中
    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
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
        # 正向传播函数，接收多个输入参数并计算输出结果
        # add position embeddings to the hidden states before projecting to queries and keys
        # 如果提供了位置嵌入，则将其添加到隐藏状态中，以便在投影到查询和键之前使用
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # 获取隐藏状态张量的形状信息：批量大小、查询数、特征维度
        batch_size, num_queries, _ = hidden_states.shape
        # 获取编码器隐藏状态张量的形状信息：批量大小、序列长度、特征维度
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        # 检查空间形状与编码器隐藏状态序列长度是否对齐
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        # 将编码器隐藏状态投影到值空间
        value = self.value_proj(encoder_hidden_states)
        # 如果提供了注意力掩码，则反转注意力掩码
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], float(0))  # 在注意力掩码为True的位置填充0
        # 重新调整值张量的形状以便后续多头注意力计算
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        # 计算采样偏移量，用于形成采样位置
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        # 计算注意力权重，用于多尺度可变形注意力的加权求和
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        # 对注意力权重进行softmax归一化，以确保权重和为1
        attention_weights = nn.functional.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # 如果参考点张量的最后一个维度为2
        if reference_points.shape[-1] == 2:
            # 根据空间形状和采样偏移量计算采样位置
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        # 如果参考点张量的最后一个维度为4
        elif reference_points.shape[-1] == 4:
            # 根据参考点、采样偏移量和缩放系数计算采样位置
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            # 抛出异常，说明参考点张量的最后一个维度不符合预期
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
        # 使用PyTorch实现的多尺度可变形注意力机制，计算输出
        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        # 对输出进行最终的投影操作
        output = self.output_proj(output)

        # 返回计算得到的输出和注意力权重
        return output, attention_weights
# 定义一个名为 OneFormerPixelDecoderEncoderLayer 的自定义神经网络层，继承自 nn.Module
class OneFormerPixelDecoderEncoderLayer(nn.Module):
    # 初始化函数，接受一个 OneFormerConfig 类型的参数 config
    def __init__(self, config: OneFormerConfig):
        super().__init__()
        # 设置 self.embed_dim 为 config 中的卷积维度 conv_dim
        self.embed_dim = config.conv_dim
        # 初始化 self.self_attn 为 OneFormerPixelDecoderEncoderMultiscaleDeformableAttention 类的实例
        # 参数包括 embed_dim（嵌入维度）、num_heads（注意力头数）、n_levels（多尺度级数）、n_points（采样点数）
        self.self_attn = OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            n_levels=3,
            n_points=4,
        )

        # 初始化 self.self_attn_layer_norm 为 LayerNorm 层，对 self.embed_dim 维度进行归一化
        # 使用 config 中的 layer_norm_eps 作为 epsilon 参数
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 设置 self.dropout 为 config 中的 dropout 率
        self.dropout = config.dropout
        # 设置 self.activation_fn 为 relu 激活函数
        self.activation_fn = nn.functional.relu
        # 设置 self.activation_dropout 为 config 中的 dropout 率
        self.activation_dropout = config.dropout
        # 初始化 self.fc1 为 Linear 层，输入维度为 self.embed_dim，输出维度为 config 中的 encoder_feedforward_dim
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        # 初始化 self.fc2 为 Linear 层，输入维度为 config 中的 encoder_feedforward_dim，输出维度为 self.embed_dim
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        # 初始化 self.final_layer_norm 为 LayerNorm 层，对 self.embed_dim 维度进行归一化
        # 使用 config 中的 layer_norm_eps 作为 epsilon 参数
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # 设置 self.is_training 为 config 中的 is_training 布尔值
        self.is_training = config.is_training

    # 前向传播函数定义，接受多个输入参数并返回结果
    def forward(
        self,
        hidden_states: torch.Tensor,           # 输入的隐藏状态张量
        attention_mask: torch.Tensor,          # 注意力掩码张量
        position_embeddings: torch.Tensor = None,     # 位置嵌入张量（可选）
        reference_points=None,                  # 参考点（可选）
        spatial_shapes=None,                    # 空间形状（可选）
        level_start_index=None,                 # 级别起始索引（可选）
        output_attentions: bool = False,        # 是否输出注意力（默认为 False）
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                输入层的输入数据。
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                注意力掩码。
            position_embeddings (`torch.FloatTensor`, *optional*):
                位置嵌入，用于加到 `hidden_states` 上。
            reference_points (`torch.FloatTensor`, *optional*):
                参考点。
            spatial_shapes (`torch.LongTensor`, *optional*):
                主干特征图的空间形状。
            level_start_index (`torch.LongTensor`, *optional*):
                等级开始索引。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关详细信息，请参阅返回的张量中的 `attentions`。
        """
        residual = hidden_states  # 保存原始的 hidden_states

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

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.is_training)  # 应用 dropout
        hidden_states = residual + hidden_states  # 残差连接
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 层归一化

        residual = hidden_states  # 保存残差连接后的结果
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 应用激活函数和第一个全连接层
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.is_training)  # 应用 dropout

        hidden_states = self.fc2(hidden_states)  # 第二个全连接层
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.is_training)  # 应用 dropout

        hidden_states = residual + hidden_states  # 残差连接
        hidden_states = self.final_layer_norm(hidden_states)  # 最终的层归一化

        if self.is_training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)  # 处理无穷大或 NaN 值

        outputs = (hidden_states,)  # 输出结果为 hidden_states 的元组形式

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则添加到输出结果中

        return outputs  # 返回输出结果
"""
Modified from from transformers.models.detr.modeling_deformable_detr.DeformableDetrEncoder with DeformableDetrEncoder->OneFormerPixelDecoderEncoderOnly
"""
# 定义一个名为 OneFormerPixelDecoderEncoderOnly 的类，继承自 nn.Module
class OneFormerPixelDecoderEncoderOnly(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* deformable attention layers. Each layer is a
    [`OneFormerPixelDecoderEncoderLayer`].

    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.

    Args:
        config: OneFormerConfig
    """

    def __init__(self, config: OneFormerConfig):
        super().__init__()

        self.config = config
        self.dropout = config.dropout
        # 创建一个由多个 OneFormerPixelDecoderEncoderLayer 实例组成的层列表
        self.layers = nn.ModuleList([OneFormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        # 遍历空间形状列表，为每个特征图获取参考点
        for lvl, (height, width) in enumerate(spatial_shapes):
            # 创建网格，生成高度和宽度上的参考点网格
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
            )
            # 根据有效比率调整参考点位置
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # 将参考点列表拼接为一个张量
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
    ):
        pass  # 这里定义了 forward 方法，但是没有具体的实现内容
    # 计算输入掩码的有效比率，返回每个特征图的有效比率
    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""

        # 获取掩码的形状，并提取高度和宽度信息
        _, height, width = mask.shape
        
        # 计算每个特征图在高度上的有效像素数目
        valid_height = torch.sum(~mask[:, :, 0], 1)
        
        # 计算每个特征图在宽度上的有效像素数目
        valid_width = torch.sum(~mask[:, 0, :], 1)
        
        # 将有效高度像素数转换为有效比率，并指定数据类型
        valid_ratio_heigth = valid_height.to(dtype) / height
        
        # 将有效宽度像素数转换为有效比率，并指定数据类型
        valid_ratio_width = valid_width.to(dtype) / width
        
        # 将宽度和高度的有效比率堆叠成一个张量，形状为 [batch_size, 2]
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        
        # 返回每个特征图的有效比率张量
        return valid_ratio

    # 模型的前向传播函数，处理特征输入并可选地返回多个附加输出
    def forward(
        self,
        features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
# Modified from from transformers.models.mask2former.modeling_mask2former.Mask2FormerPixelLevelModule with Mask2->One
class OneFormerPixelLevelModule(nn.Module):
    def __init__(self, config: OneFormerConfig):
        """
        Pixel Level Module proposed in [Masked-attention Mask Transformer for Universal Image
        Segmentation](https://arxiv.org/abs/2112.01527). It runs the input image through a backbone and a pixel
        decoder, generating multi-scale feature maps and pixel embeddings.

        Args:
            config ([`OneFormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()
        # 加载指定配置的背景模型
        self.encoder = load_backbone(config)
        # 使用给定配置和背景模型通道数实例化像素解码器
        self.decoder = OneFormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> OneFormerPixelLevelModuleOutput:
        # 提取输入像素值的特征图列表
        features: List[Tensor] = self.encoder(pixel_values).feature_maps
        # 使用解码器生成像素级特征，并可选择输出隐藏状态
        decoder_output: OneFormerPixelDecoderOutput = self.decoder(features, output_hidden_states=output_hidden_states)
        # 返回像素级模块的输出对象，包括编码器和解码器生成的特征
        return OneFormerPixelLevelModuleOutput(
            encoder_features=tuple(features),
            decoder_features=decoder_output.multi_scale_features,
            decoder_last_feature=decoder_output.mask_features,
        )


# Modified from transformers.models.detr.modeling_detr.DetrAttention with Detr->OneFormer
class OneFormerAttention(nn.Module):
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
        # 每个头部的维度
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        # 缩放因子，用于缩放点积注意力的输出
        self.scaling = self.head_dim**-0.5

        # 线性变换，用于查询、键、值的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        # 将张量重塑为适合多头注意力的形状
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        # 添加位置嵌入到输入张量中，如果位置嵌入不为None
        return tensor if position_embeddings is None else tensor + position_embeddings
    # 定义一个方法 `forward`，用于模型的前向传播
    # `self` 是指向当前实例化的对象的引用
    # `hidden_states` 参数用来接收输入的隐藏状态张量，通常是模型的输入
    # `attention_mask` 参数是一个可选的注意力掩码张量，用于指定哪些位置需要忽略注意力
    # `position_embeddings` 参数是一个可选的位置嵌入张量，用于处理输入序列的位置信息
    # `key_value_states` 参数是一个可选的键值状态张量，通常用于注意力机制中的键值对
    # `key_value_position_embeddings` 参数是一个可选的键值位置嵌入张量，用于处理键值状态的位置信息
    # `output_attentions` 参数是一个布尔值，用于指示是否输出注意力权重
class OneFormerTransformerDecoderSelfAttentionLayer(nn.Module):
    def __init__(
        self, embed_dim, num_heads, dropout=0.0, activation="relu", normalize_before=False, layer_norm_eps=1e-05
    ):
        super().__init__()
        # 初始化自注意力层，使用 OneFormerAttention 类
        self.self_attn = OneFormerAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, is_decoder=True)

        # Layer normalization 层，对输入进行归一化
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 激活函数，根据提供的激活函数字符串选择对应的函数
        self.activation = ACT2FN[activation]
        # 是否在自注意力之前进行归一化
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 如果位置编码不为 None，则将位置编码加到张量上
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        output,
        output_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 执行自注意力操作，返回处理后的输出和注意力权重
        output2, attention_weights = self.self_attn(
            hidden_states=output, position_embeddings=query_pos, attention_mask=output_mask, output_attentions=True
        )
        # 应用 dropout 到输出上
        output = output + self.dropout(output2)
        # 应用 Layer normalization 到输出上
        output = self.norm(output)

        return output, attention_weights

    def forward_pre(
        self,
        output,
        output_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 应用 Layer normalization 到输出上
        output2 = self.norm(output)
        # 执行自注意力操作，返回处理后的输出和注意力权重
        output2, attention_weights = self.self_attn(
            hidden_states=output2, position_embeddings=query_pos, attention_mask=output_mask, output_attentions=True
        )
        # 应用 dropout 到输出上
        output = output + self.dropout(output2)

        return output, attention_weights

    def forward(
        self,
        output,
        output_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 根据 normalize_before 属性选择 forward_pre 或 forward_post 方法执行
        if self.normalize_before:
            return self.forward_pre(output, output_mask, output_key_padding_mask, query_pos)
        return self.forward_post(output, output_mask, output_key_padding_mask, query_pos)


class OneFormerTransformerDecoderCrossAttentionLayer(nn.Module):
    def __init__(
        self, embed_dim, num_heads, dropout=0.0, activation="relu", normalize_before=False, layer_norm_eps=1e-05
    ):
        super().__init__()
        # 初始化跨注意力层，使用 nn.MultiheadAttention 类
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        # Layer normalization 层，对输入进行归一化
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 激活函数，根据提供的激活函数字符串选择对应的函数
        self.activation = ACT2FN[activation]
        # 是否在自注意力之前进行归一化
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 如果位置编码不为 None，则将位置编码加到张量上
        return tensor if pos is None else tensor + pos
    # 定义 Transformer 模型的前向传播函数，用于处理“后归一化”情况
    def forward_post(
        self,
        output,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 使用 multihead_attn 函数进行多头注意力计算
        output2, attention_weights = self.multihead_attn(
            query=self.with_pos_embed(output, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        # 使用 dropout 进行输出的正则化
        output = output + self.dropout(output2)
        # 对输出进行归一化
        output = self.norm(output)

        return output, attention_weights

    # 定义 Transformer 模型的前向传播函数，用于处理“先归一化”情况
    def forward_pre(
        self,
        output,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 对输出进行归一化
        output2 = self.norm(output)
        # 使用 multihead_attn 函数进行多头注意力计算
        output2, attention_weights = self.multihead_attn(
            query=self.with_pos_embed(output2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        # 使用 dropout 进行输出的正则化
        output = output + self.dropout(output2)

        return output, attention_weights

    # 定义 Transformer 模型的前向传播函数，根据 normalize_before 标志选择具体的前向传播方式
    def forward(
        self,
        output,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 如果 normalize_before 为真，则使用前归一化的前向传播方式
        if self.normalize_before:
            return self.forward_pre(output, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        # 否则使用后归一化的前向传播方式
        return self.forward_post(output, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
class OneFormerTransformerDecoderFFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        # 实现前馈模型
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 创建线性层，输入维度为d_model，输出维度为dim_feedforward
        self.dropout = nn.Dropout(dropout)  # 创建dropout层
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 创建线性层，输入维度为dim_feedforward，输出维度为d_model

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)  # 创建LayerNorm层，输入维度为d_model，eps为layer_norm_eps

        self.activation = ACT2FN[activation]  # 激活函数为ACT2FN中对应的激活函数
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos  # 如果pos为None，则返回tensor，否则返回tensor + pos

    def forward_post(self, output):
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))  # 前馈模型的具体实现
        output = output + self.dropout(output2)  # 加上dropout后的输出到原始输出
        output = self.norm(output)  # 对输出进行LayerNorm
        return output

    def forward_pre(self, output):
        output2 = self.norm(output)  # 对输出进行LayerNorm
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2)))  # 前馈模型的具体实现
        output = output + self.dropout(output2)  # 加上dropout后的输出到原始输出
        return output

    def forward(self, output):
        if self.normalize_before:
            return self.forward_pre(output)  # 如果normalize_before为True，则调用forward_pre方法
        return self.forward_post(output)  # 否则调用forward_post方法


class OneFormerMLPPredictionHead(nn.Module):
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
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)  # 输入维度列表
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]  # 输出维度列表

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(
                PredictionBlock(in_dim, out_dim, activation=nn.ReLU() if i < num_layers - 1 else nn.Identity())  # 创建PredictionBlock对象，并添加到layers列表中
            )

        self.layers = nn.Sequential(*layers)  # 将layers列表转换为Sequential层

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)  # 调用layers的forward方法，即对输入进行前向传播


# refactored from original implementation
class OneFormerTransformerDecoderLayer(nn.Module):
    # 初始化方法，接收一个配置参数 config: OneFormerConfig
    def __init__(self, config: OneFormerConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为配置参数中的隐藏维度
        self.embed_dim = config.hidden_dim
        # 设定特征级别数量为 3
        self.num_feature_levels = 3

        # 创建一个跨注意力层对象，使用配置中的参数进行初始化
        self.cross_attn = OneFormerTransformerDecoderCrossAttentionLayer(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=0.0,
            normalize_before=config.pre_norm,
            layer_norm_eps=config.layer_norm_eps,
        )

        # 创建一个自注意力层对象，使用配置中的参数进行初始化
        self.self_attn = OneFormerTransformerDecoderSelfAttentionLayer(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=0.0,
            normalize_before=config.pre_norm,
            layer_norm_eps=config.layer_norm_eps,
        )

        # 创建一个前馈神经网络层对象，使用配置中的参数进行初始化
        self.ffn = OneFormerTransformerDecoderFFNLayer(
            d_model=self.embed_dim,
            dim_feedforward=config.dim_feedforward,
            dropout=0.0,
            normalize_before=config.pre_norm,
            layer_norm_eps=config.layer_norm_eps,
        )

    # 前向传播方法定义，接收多个输入参数，包括索引、输出张量、多阶段特征、多阶段位置嵌入等
    def forward(
        self,
        index: int,
        output: torch.Tensor,
        multi_stage_features: List[torch.Tensor],
        multi_stage_positional_embeddings: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        query_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            index (`int`): Transformer 解码器中层的索引。
            output (`torch.FloatTensor`): 对象查询，形状为 `(N, batch, hidden_dim)`
            multi_stage_features (`List[torch.Tensor]`): 像素解码器中的多尺度特征。
            multi_stage_positional_embeddings (`List[torch.Tensor]`):
                多尺度特征的位置嵌入。
            attention_mask (`torch.FloatTensor`): 用于掩蔽的注意力掩码。
            query_embeddings (`torch.FloatTensor`, *optional*):
                被添加到自注意力层中的查询和键的位置嵌入。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 以获取更多详细信息。
        """

        level_index = index % self.num_feature_levels
        attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False

        # Masked Cross Attention
        # 执行掩蔽交叉注意力操作
        output, cross_attn_weights = self.cross_attn(
            output,
            multi_stage_features[level_index],
            memory_mask=attention_mask,
            memory_key_padding_mask=None,  # 这里不对填充区域应用掩蔽
            pos=multi_stage_positional_embeddings[level_index],
            query_pos=query_embeddings,
        )

        # Self Attention
        # 执行自注意力操作
        output, self_attn_weights = self.self_attn(
            output,
            output_mask=None,
            output_key_padding_mask=None,
            query_pos=query_embeddings,
        )

        # Fully Connected
        # 执行全连接层操作
        output = self.ffn(output)

        outputs = (output,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
# 定义一个基于多层自注意力和多头注意力的解码器层模块
class OneFormerTransformerDecoderQueryTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        # 定义自注意力层和多头注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 实现前向传播模型
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 定义三个 LayerNorm 层和对应的 dropout 层
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 激活函数选择
        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    # 辅助函数，用于将位置编码加到张量中
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos



# 定义一个基于多层解码器层的模块
class OneFormerTransformerDecoderQueryTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # 使用 _get_clones 函数创建多层解码器层
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    # 前向传播函数，接收多个输入参数并进行处理
    def forward(
        self,
        output,
        memory,
        output_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        intermediate = []

        # 对每一层进行迭代处理
        for layer in self.layers:
            output = layer(
                output,
                memory,
                output_mask=output_mask,
                memory_mask=memory_mask,
                output_key_padding_mask=output_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            # 如果设置了 return_intermediate，则将当前层的输出添加到 intermediate 列表中
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # 如果定义了 norm 层，则对最终输出进行归一化处理
        if self.norm is not None:
            output = self.norm(output)
            # 如果设置了 return_intermediate，则替换 intermediate 列表中的最后一个元素为归一化后的输出
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        # 如果设置了 return_intermediate，则返回 intermediate 列表的堆叠结果
        if self.return_intermediate:
            return torch.stack(intermediate)

        # 否则，返回未经扩展的输出张量
        return output.unsqueeze(0)
    def forward_post(
        self,
        output,
        memory,
        output_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 使用位置编码嵌入输出向量
        q = k = self.with_pos_embed(output, query_pos)
        # 执行自注意力机制，计算输出的第一部分
        output2 = self.self_attn(q, k, value=output, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]  # 取自注意力机制的输出结果
        # 应用 dropout，并将结果添加到原始输出上
        output = output + self.dropout1(output2)
        # 执行层归一化操作
        output = self.norm1(output)
        # 执行多头注意力机制，计算输出的第二部分
        output2 = self.multihead_attn(
            query=self.with_pos_embed(output, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        output2 = output2[0]  # 取多头注意力机制的输出结果
        # 应用 dropout，并将结果添加到原始输出上
        output = output + self.dropout2(output2)
        # 执行层归一化操作
        output = self.norm2(output)
        # 经过线性层和激活函数变换的结果
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        # 应用 dropout，并将结果添加到原始输出上
        output = output + self.dropout3(output2)
        # 执行层归一化操作
        output = self.norm3(output)
        return output

    def forward_pre(
        self,
        output,
        memory,
        output_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 执行层归一化操作
        output2 = self.norm1(output)
        # 使用位置编码嵌入输出向量
        q = k = self.with_pos_embed(output2, query_pos)
        # 执行自注意力机制，计算输出的第一部分
        output2 = self.self_attn(q, k, value=output2, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]  # 取自注意力机制的输出结果
        # 应用 dropout，并将结果添加到原始输出上
        output = output + self.dropout1(output2)
        # 执行层归一化操作
        output2 = self.norm2(output)
        # 执行多头注意力机制，计算输出的第二部分
        output2 = self.multihead_attn(
            query=self.with_pos_embed(output2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        output2 = output2[0]  # 取多头注意力机制的输出结果
        # 应用 dropout，并将结果添加到原始输出上
        output = output + self.dropout2(output2)
        # 执行层归一化操作
        output2 = self.norm3(output)
        # 经过线性层和激活函数变换的结果
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2))))
        # 应用 dropout，并将结果添加到原始输出上
        output = output + self.dropout3(output2)
        return output

    def forward(
        self,
        output,
        memory,
        output_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 统一的前向传播方法，根据模型的设定调用相应的前向传播方式
        if self.pre_norm:
            return self.forward_pre(
                output, memory, output_mask, memory_mask, output_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        else:
            return self.forward_post(
                output, memory, output_mask, memory_mask, output_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
    ):
        # 如果标志为 normalize_before，则调用前序处理函数 forward_pre
        if self.normalize_before:
            return self.forward_pre(
                output,
                memory,
                output_mask,
                memory_mask,
                output_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        # 否则调用后序处理函数 forward_post
        return self.forward_post(
            output,
            memory,
            output_mask,
            memory_mask,
            output_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )
class OneFormerTransformerDecoderQueryTransformer(nn.Module):
    """
    定义一个Transformer解码器模块，用于处理查询转换任务。

    Args:
        d_model (int): 模型的输入和输出维度，默认为512。
        nhead (int): 多头注意力机制中注意头的数量，默认为8。
        num_decoder_layers (int): 解码器堆叠的层数，默认为6。
        dim_feedforward (int): 每个位置的前馈神经网络的维度，默认为2048。
        dropout (float): Dropout概率，默认为0.1。
        activation (str): 激活函数类型，默认为"relu"。
        normalize_before (bool): 是否在每个子层之前进行LayerNorm，默认为False。
        return_intermediate_dec (bool): 是否返回每个解码层的中间结果，默认为False。
        layer_norm_eps (float): LayerNorm中的epsilon值，默认为1e-05。
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()

        # 创建一个Transformer解码器层对象
        decoder_layer = OneFormerTransformerDecoderQueryTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before, layer_norm_eps
        )
        # 创建一个LayerNorm层，用于解码器的输出
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # 创建一个完整的Transformer解码器
        self.decoder = OneFormerTransformerDecoderQueryTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, mask, query_embed, pos_embed, task_token=None):
        """
        定义了Transformer解码器的前向传播逻辑。

        Args:
            src (torch.Tensor): 输入的源数据，形状为(batch_size, seq_len, d_model)。
            mask (torch.Tensor): 掩码张量，形状为(batch_size, seq_len)，标记哪些位置需要屏蔽。
            query_embed (torch.Tensor): 查询嵌入向量，形状为(seq_len_query, d_model)，用于引导解码器。
            pos_embed (torch.Tensor): 位置嵌入向量，形状为(seq_len, d_model)，提供位置信息。
            task_token (torch.Tensor): 任务令牌，形状为(1, 1, d_model)，用于特定任务的解码。

        Returns:
            torch.Tensor: 解码器的输出，形状为(batch_size, d_model, seq_len_query)，经过转置以匹配预期输出形状。
        """

        batch_size = src.shape[0]
        # 将输入的src张量展平并重新排列维度
        src = src.flatten(2).permute(2, 0, 1)
        # 将位置嵌入向量展平并重新排列维度
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # 将查询嵌入向量增加一个维度，并重复以匹配批量大小
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
        
        # 如果存在掩码，则将其展平以匹配src的形状
        if mask is not None:
            mask = mask.flatten(1)

        # 如果未提供任务令牌，则创建一个全零的查询张量
        if task_token is None:
            queries = torch.zeros_like(query_embed)
        else:
            # 否则重复任务令牌以匹配查询嵌入的形状
            queries = task_token.repeat(query_embed.shape[0], 1, 1)

        # 调用Transformer解码器进行解码，并传递额外的位置和掩码信息
        queries = self.decoder(queries, src, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        # 将输出进行转置，以符合预期的输出形状
        return queries.transpose(1, 2)


class OneFormerTransformerDecoder(nn.Module):
    """
    Transformer解码器模块。
    """
    # 初始化函数，用于创建一个 OneFormerDecoder 对象
    def __init__(self, in_channels: int, config: OneFormerConfig):
        # 调用父类的初始化方法
        super().__init__()
        
        # 将配置信息保存在对象中
        self.config = config

        # 从配置中获取参数并保存到对象中
        self.dropout = config.dropout
        self.num_heads = config.num_attention_heads
        self.is_training = config.is_training
        self.use_task_norm = config.use_task_norm
        self.use_auxiliary_loss = config.use_auxiliary_loss

        # 创建查询变换器对象 OneFormerTransformerDecoderQueryTransformer
        self.query_transformer = OneFormerTransformerDecoderQueryTransformer(
            d_model=config.hidden_dim,
            dropout=config.dropout,
            nhead=config.num_attention_heads,
            dim_feedforward=config.dim_feedforward,
            num_decoder_layers=config.query_dec_layers,
            normalize_before=config.pre_norm,
            return_intermediate_dec=False,
            layer_norm_eps=config.layer_norm_eps,
        )

        # 创建解码器层归一化对象 nn.LayerNorm
        self.decoder_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        # 设定特征级别数量为3
        self.num_feature_levels = 3

        # 创建解码器层的模块列表，每个元素是 OneFormerTransformerDecoderLayer 对象
        self.layers = nn.ModuleList(
            [OneFormerTransformerDecoderLayer(config) for _ in range(config.decoder_layers - 1)]
        )

        # 创建查询输入的投影层 nn.Conv2d，将输入通道数转换为隐藏维度数
        self.query_input_projection = nn.Conv2d(in_channels, config.hidden_dim, kernel_size=1)

        # 创建类别嵌入层 nn.Linear，用于分类任务
        self.class_embed = nn.Linear(config.hidden_dim, config.num_labels + 1)

        # 创建掩码预测头部 OneFormerMLPPredictionHead，用于预测掩码任务
        self.mask_embed = OneFormerMLPPredictionHead(
            config.hidden_dim,
            config.hidden_dim,
            config.mask_dim,
            3,
        )

    # 前向传播函数，定义了网络的数据流向和处理逻辑
    def forward(
        self,
        task_token=None,
        multi_stage_features=None,
        multi_stage_positional_embeddings=None,
        mask_features=None,
        query_features=None,
        query_embeddings=None,
        query_embedder=None,
        size_list=None,
        output_attentions=None,
    ):
        ):
        # 如果使用任务规范化，则对任务标记进行规范化处理
        if self.use_task_norm:
            task_token = self.decoder_norm(task_token)

        # 使用查询转换器处理查询特征，生成对象查询
        object_queries = self.query_transformer(
            query_features,
            None,
            query_embedder.weight[:-1],  # 使用查询嵌入器的权重（排除最后一个）
            self.query_input_projection(mask_features),  # 对掩码特征进行查询输入投影
            task_token if self.use_task_norm else None,  # 如果使用任务规范化，则传入任务标记
        )

        # 对象查询重新排列维度
        object_queries = object_queries[0].permute(1, 0, 2)

        # 将对象查询与任务标记连接起来，生成输出
        queries = torch.cat([object_queries, task_token], dim=0)

        # 克隆查询作为输出
        output = queries.clone()

        # 初始化中间类别预测和中间掩码预测列表
        intermediate_class_predictions = []
        intermediate_mask_predictions = []

        # 在可学习的查询特征上执行预测头部操作
        outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(
            output, mask_features, attention_mask_target_size=size_list[0]
        )
        intermediate_class_predictions.append(outputs_class)
        intermediate_mask_predictions.append(outputs_mask)

        # 初始化注意力机制元组
        attentions = ()

        # 遍历所有层进行变换
        for index, layer in enumerate(self.layers):
            # 在当前层上进行变换操作，更新输出和注意力
            layer_outputs = layer(
                index=index,
                output=output,
                multi_stage_features=multi_stage_features,
                multi_stage_positional_embeddings=multi_stage_positional_embeddings,
                attention_mask=attention_mask,
                query_embeddings=query_embeddings,
                output_attentions=output_attentions,
            )

            # 更新输出和注意力元组
            output = layer_outputs[0]
            attentions += (layer_outputs[1:],)

            # 继续在当前输出上执行预测头部操作
            outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(
                output, mask_features, attention_mask_target_size=size_list[(index + 1) % self.num_feature_levels]
            )
            intermediate_class_predictions.append(outputs_class)
            intermediate_mask_predictions.append(outputs_mask)

        # 检查中间掩码预测的数量是否与层数相同
        if not len(intermediate_mask_predictions) == len(self.layers) + 1:
            raise ValueError(
                "Intermediate predictions in the transformer decoder must have the same number of elements as number"
                " of layers"
            )

        # 从最后一个层的输出重新排列对象查询
        object_queries = layer_outputs[0].permute(1, 0, 2)

        # 重新排列对比日志
        contrastive_logits = queries.permute(1, 0, 2)

        # 返回Transformer解码器的输出对象
        return OneFormerTransformerDecoderOutput(
            object_queries=object_queries,
            contrastive_logits=contrastive_logits,
            prediction_masks=intermediate_mask_predictions[-1],  # 最终的预测掩码
            prediction_class=intermediate_class_predictions[-1],  # 最终的类别预测
            auxiliary_predictions=self._get_aux_predictions(
                intermediate_class_predictions, intermediate_mask_predictions
            ) if self.use_auxiliary_loss else None,  # 如果使用辅助损失，则生成辅助预测
            attentions=attentions,  # 返回注意力机制元组
        )
    # 定义一个方法，用于处理前向预测头部的输出，生成类别预测、掩码预测和注意力掩码
    def forward_prediction_heads(self, output, mask_features, attention_mask_target_size):
        # 对decoder输出进行归一化处理
        decoder_output = self.decoder_norm(output)
        # 调换维度顺序，通常是从(batch, seq_len, ...)到(seq_len, batch, ...)的转置操作
        decoder_output = decoder_output.transpose(0, 1)
        # 生成类别预测，使用class_embed模块
        outputs_class = self.class_embed(decoder_output)
        # 使用mask_embed模块生成掩码预测
        mask_embed = self.mask_embed(decoder_output)
        # 使用torch.einsum执行张量乘积操作，生成掩码预测输出
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # 使用nn.functional.interpolate函数进行插值操作，调整outputs_mask的大小
        attention_mask = nn.functional.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        # 要求attention_mask使用bool类型
        # 如果传入的是BoolTensor，则True位置不允许进行注意力操作，False位置保持不变
        attention_mask = (
            attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5
        ).bool()
        # 将attention_mask从计算图中分离出来，不参与梯度计算
        attention_mask = attention_mask.detach()

        # 返回类别预测、掩码预测和注意力掩码
        return outputs_class, outputs_mask, attention_mask

    # 使用torch.jit.unused装饰器标记一个方法，表明该方法在torchscript中不被使用
    def _get_aux_predictions(self, outputs_class, outputs_seg_masks):
        # 这是一个解决方法，以使torchscript可以正常工作，因为torchscript不支持非同构值的字典，
        # 比如一个字典同时包含张量和列表的情况。
        # 创建一个列表aux_list，包含类别查询的logits和掩码查询的logits
        aux_list = [
            {"class_queries_logits": a, "masks_queries_logits": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        # 将列表转换为元组返回
        return tuple(aux_list)
class OneFormerTransformerModule(nn.Module):
    """
    The OneFormer's transformer module.
    """

    def __init__(self, in_features: int, config: OneFormerConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3  # 设置特征级别的数量为3
        self.position_embedder = OneFormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)  # 初始化位置编码器
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)  # 初始化查询嵌入层
        self.input_projections = []

        # 根据特征级别数量循环添加输入投影层或空序列
        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_proj:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))  # 添加卷积层
            else:
                self.input_projections.append(nn.Sequential())  # 添加空序列

        self.decoder = OneFormerTransformerDecoder(in_channels=in_features, config=config)  # 初始化解码器
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)  # 初始化特征级别嵌入层

    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        task_token: Tensor,
        output_attentions: bool = False,
    ) -> OneFormerTransformerDecoderOutput:
        if not len(multi_scale_features) == self.num_feature_levels:
            raise ValueError(
                f"Number of elements in multi_scale_features ({len(multi_scale_features)}) and num_feature_levels"
                f" ({self.num_feature_levels}) do not match!"
            )

        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])  # 获取特征的空间维度大小
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))  # 计算位置编码并展平
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )  # 应用输入投影和特征级别嵌入

            # 将 NxCxHxW 展平为 HWxNxC
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape  # 获取批量大小

        # QxNxC
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)  # 扩展查询嵌入的维度
        task_token = task_token.unsqueeze(0)  # 增加任务标记的维度

        query_features = self.position_embedder(mask_features, None)  # 计算掩码特征的位置编码

        return self.decoder(
            task_token=task_token,
            multi_stage_features=multi_stage_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            mask_features=mask_features,
            query_features=query_features,
            query_embeddings=query_embeddings,
            query_embedder=self.queries_embedder,
            size_list=size_list,
            output_attentions=output_attentions,
        )  # 调用解码器进行前向传播
# Copied from transformers.models.maskformer.modeling_maskformer.MaskFormerSinePositionEmbedding with Mask->One
class OneFormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        # 检查是否传入了 scale 参数但未设置 normalize=True，如果是，则抛出异常
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 初始化位置编码器的参数
        self.num_pos_feats = num_pos_feats  # 位置特征的数量
        self.temperature = temperature  # 温度参数
        self.normalize = normalize  # 是否进行归一化
        self.scale = 2 * math.pi if scale is None else scale  # 缩放参数，默认为 2π

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # 如果未提供 mask 参数，则创建一个全零张量作为 mask
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        # 根据 mask 创建反向 mask，用于计算位置编码
        not_mask = (~mask).to(x.dtype)
        # 在垂直和水平方向上计算累积的非 mask 数量，作为位置编码的一部分
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        # 如果设置了 normalize=True，则对位置编码进行归一化处理
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 创建维度参数用于计算位置编码
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).type_as(x)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        # 计算位置编码的 x 和 y 分量
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 使用正弦和余弦函数对位置编码进行转换，然后展平为二维张量
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 将 x 和 y 的位置编码连接，并转置张量维度
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# Copied from transformers.models.maskformer.modeling_maskformer.PredictionBlock
class PredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        # 创建包含线性层和激活函数的列表，以模拟 Sequential 块的子模块索引
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # 将每个层作为模块添加到当前模块中
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        # 遍历并逐层应用线性层和激活函数
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class OneFormerTextMapperAttention(nn.Module):
    # 这里应该添加你的注释
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 如果没有提供 qk_scale，则使用默认的缩放因子，用于缩放注意力计算中的权重
        self.scale = qk_scale or head_dim ** -0.5

        # 创建查询（q）、键（k）、值（v）的线性投影层，并考虑是否包含偏置项
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # 注意力机制中的 dropout 层，用于在训练过程中随机丢弃部分注意力权重
        self.attn_drop = nn.Dropout(attn_drop)
        # 最终输出的线性映射层，用于将多头注意力的结果映射回原始空间
        self.proj = nn.Linear(dim, dim)
        # 用于在最终输出时随机丢弃部分结果的 dropout 层
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        batch_size, q_sequence_length, num_channels = q.shape

        # 检查键（k）和值（v）张量的形状是否相同，如果不同则抛出异常
        if not k.shape == v.shape:
            raise ValueError(f"keys ({list(k.shape)}) and values ({list(v.shape)}) have different shapes!")

        batch_size, k_sequence_length, num_channels = k.shape

        # 使用线性投影层对查询（q）、键（k）、值（v）进行映射，并重新组织维度以便进行多头注意力计算
        q = self.q_proj(q).reshape(batch_size, q_sequence_length, self.num_heads, num_channels // self.num_heads)
        k = self.k_proj(k).reshape(batch_size, k_sequence_length, self.num_heads, num_channels // self.num_heads)
        v = self.v_proj(v).reshape(batch_size, k_sequence_length, self.num_heads, num_channels // self.num_heads)

        # 计算注意力分数，使用 einsum 进行批量矩阵乘法，并乘以缩放因子
        attn = torch.einsum("bnkc,bmkc->bknm", q, k) * self.scale

        # 对注意力分数进行 softmax 归一化，以获得注意力权重
        attn = attn.softmax(dim=-1)

        # 根据注意力权重加权得到最终的多头注意力结果，并重新组织维度
        output = torch.einsum("bknm,bmkc->bnkc", attn, v).reshape(batch_size, q_sequence_length, num_channels)

        # 将多头注意力的结果经过线性映射层，并应用 dropout 层
        output = self.proj(output)
        output = self.proj_drop(output)

        return output
class OneFormerTextTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        # 初始化自注意力机制，用于处理当前层的输入
        self.self_attn = OneFormerTextMapperAttention(d_model, nhead, proj_drop=dropout)
        # 初始化交叉注意力机制，用于处理当前层输入与记忆之间的关系
        self.cross_attn = OneFormerTextMapperAttention(d_model, nhead, proj_drop=dropout)

        # 初始化三个 LayerNorm 层，分别用于不同位置的归一化处理
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # 初始化 dropout 层，用于网络训练过程中的随机丢弃
        self.dropout = nn.Dropout(dropout)

        # 初始化 MLP 网络，用于映射和转换特征表示
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, hidden_state, mem):
        # Self-Attention 操作，包括归一化和残差连接
        q = k = v = self.norm1(hidden_state)
        hidden_state = hidden_state + self.self_attn(q, k, v)
        # Cross-Attention 操作，包括归一化和残差连接
        q = self.norm2(hidden_state)
        hidden_state = hidden_state + self.cross_attn(q, mem, mem)
        # MLP 网络操作，包括残差连接和最终的归一化处理
        hidden_state = hidden_state + self.dropout(self.mlp(self.norm3(hidden_state)))
        return hidden_state


class OneFormerTextContextDecoder(nn.Module):
    def __init__(
        self,
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=6,
        visual_dim=1024,
        dropout=0.1,
        layer_norm_eps=1e-05,
        **kwargs,
    ):
        super().__init__()

        # 初始化记忆映射层，将视觉特征映射到 transformer 宽度上
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim, eps=layer_norm_eps),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width, eps=layer_norm_eps),
        )

        # 初始化文本映射层，将文本特征映射到 transformer 宽度上
        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim, eps=layer_norm_eps),
            nn.Linear(visual_dim, transformer_width),
        )

        # 初始化多层 Transformer 解码器
        self.decoder = nn.ModuleList(
            [
                OneFormerTextTransformerDecoderLayer(transformer_width, transformer_heads, dropout, layer_norm_eps)
                for _ in range(transformer_layers)
            ]
        )

        # 初始化输出映射层，将 transformer 宽度映射回视觉特征维度
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width, eps=layer_norm_eps), nn.Linear(transformer_width, visual_dim)
        )

    def forward(self, text, visual):
        # 对视觉特征进行映射和归一化处理
        visual = self.memory_proj(visual)
        # 对文本特征进行映射和归一化处理
        hidden_state = self.text_proj(text)

        # 逐层处理解码器
        for layer in self.decoder:
            hidden_state = layer(hidden_state, visual)

        # 最终输出映射和归一化处理，将结果映射回视觉特征维度
        return self.out_proj(hidden_state)


class OneFormerTextMLP(nn.Module):
    def __init__(
        self,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        output_size: Optional[int] = None,
        ...
    ):
        super().__init__()
        # 未完整给出，但可以理解为用于处理文本的 MLP 网络的初始化
    ):
        super().__init__()  # 调用父类的初始化方法，初始化神经网络模型的基础结构
        self.activation_fn = ACT2FN["quick_gelu"]  # 设置激活函数为快速GELU函数，从预定义的ACT2FN字典中获取
        hidden_size = hidden_size  # 设置隐藏层大小
        intermediate_size = intermediate_size  # 设置中间层大小
        output_size = output_size  # 设置输出层大小
        self.fc1 = nn.Linear(hidden_size, intermediate_size)  # 创建第一个全连接层，输入大小为hidden_size，输出大小为intermediate_size
        self.fc2 = nn.Linear(intermediate_size, output_size)  # 创建第二个全连接层，输入大小为intermediate_size，输出大小为output_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)  # 将输入hidden_states传入第一个全连接层进行计算
        hidden_states = self.activation_fn(hidden_states)  # 将全连接层的输出应用激活函数
        hidden_states = self.fc2(hidden_states)  # 将经过激活函数的输出传入第二个全连接层进行计算
        return hidden_states  # 返回最终的输出结果
class OneFormerTextTransformerLayer(nn.Module):
    def __init__(self, width: int, heads: int, attn_mask: torch.Tensor, layer_norm_eps=1e-05):
        super().__init__()
        # 初始化自注意力机制模块
        self.self_attn = nn.MultiheadAttention(width, heads)
        # 初始化第一层归一化模块
        self.layer_norm1 = nn.LayerNorm(width, eps=layer_norm_eps)
        # 初始化多层感知机模块
        self.mlp = OneFormerTextMLP(width, width * 4, width)
        # 初始化第二层归一化模块
        self.layer_norm2 = nn.LayerNorm(width, eps=layer_norm_eps)
        # 存储注意力掩码张量
        self.attn_mask = attn_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        # 第一层归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 自注意力机制计算
        hidden_states = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )[0]
        # 残差连接
        hidden_states = residual + hidden_states

        residual = hidden_states
        # 第二层归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 多层感知机前向传播
        hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + hidden_states

        return hidden_states


class OneFormerTextTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        use_checkpoint=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        self.width = width
        self.num_layers = layers
        # 创建由多个 OneFormerTextTransformerLayer 组成的层序列
        self.layers = nn.Sequential(
            *[OneFormerTextTransformerLayer(width, heads, attn_mask, layer_norm_eps) for _ in range(layers)]
        )
        # 是否使用梯度检查点
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states: torch.Tensor):
        for layer in self.layers:
            if self.use_checkpoint:
                # 如果使用梯度检查点，则调用 _gradient_checkpointing_func 方法
                hidden_states = self._gradient_checkpointing_func(layer, hidden_states)
            else:
                # 否则直接调用层的 forward 方法
                hidden_states = layer(hidden_states)
        return hidden_states


class OneFormerTextEncoder(nn.Module):
    def __init__(
        self,
        context_length: int,
        width: int,
        layers: int,
        vocab_size,
        use_checkpoint=False,
        layer_norm_eps=1e-05,
    ):
        super().__init__()
        # 根据宽度计算注意力头数
        heads = width // 64
        self.context_length = context_length
        self.width = width
        # 初始化 OneFormerTextTransformer 模块
        self.transformer = OneFormerTextTransformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(),  # 构建注意力掩码
            use_checkpoint=use_checkpoint,
            layer_norm_eps=layer_norm_eps,
        )
        # 初始化位置嵌入参数
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        # 初始化最终的归一化模块
        self.ln_final = nn.LayerNorm(width, eps=layer_norm_eps)
        # 初始化 token 的嵌入层
        self.token_embedding = nn.Embedding(vocab_size, width)
    # 创建注意力掩码，延迟创建，使得视觉令牌之间可以完全关注
    mask = torch.empty(self.context_length, self.context_length)
    # 使用 PyTorch 的加法注意力掩码；填充为负无穷大
    mask.fill_(float("-inf"))
    # 将注意力掩码的下三角置零，保留上三角，实现因果注意力
    mask.triu_(1)  # zero out the lower diagonal
    # 返回构建好的注意力掩码
    return mask

    # 前向传播函数定义，接收文本数据作为输入
    hidden_state = self.token_embedding(text)
    # 加上位置嵌入向量
    hidden_state = hidden_state + self.positional_embedding
    # 将张量维度重新排列为 (sequence_length, batch_size, embedding_dim)
    hidden_state = hidden_state.permute(1, 0, 2)
    # 应用 Transformer 模型
    hidden_state = self.transformer(hidden_state)
    # 将张量维度重新排列为 (batch_size, sequence_length, embedding_dim)
    hidden_state = hidden_state.permute(1, 0, 2)
    # 应用最终的 layer normalization
    hidden_state = self.ln_final(hidden_state)
    # 从每个序列中选择最高概率的 token，作为输出隐藏状态
    hidden_state = hidden_state[torch.arange(hidden_state.shape[0]), text.argmax(dim=-1)]
    # 返回最终输出的隐藏状态
    return hidden_state
class OneFormerTextMapper(nn.Module):
    # OneFormerTextMapper 类定义
    def __init__(self, config: OneFormerConfig):
        super().__init__()
        # 初始化文本编码器，使用配置中的参数
        self.text_encoder = OneFormerTextEncoder(
            context_length=config.text_encoder_context_length,
            width=config.text_encoder_width,
            layers=config.text_encoder_num_layers,
            vocab_size=config.text_encoder_vocab_size,
            layer_norm_eps=config.layer_norm_eps,
        )

        # 初始化文本投影器，使用配置中的参数
        self.text_projector = OneFormerMLPPredictionHead(
            config.text_encoder_width,
            config.hidden_dim,
            config.hidden_dim,
            config.text_encoder_proj_layers,
        )
        
        # 如果配置中指定了上下文长度，则初始化上下文嵌入层；否则置为 None
        if config.text_encoder_n_ctx > 0:
            self.prompt_ctx = nn.Embedding(
                config.text_encoder_n_ctx,
                config.text_encoder_width,
            )
        else:
            self.prompt_ctx = None

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        # 编码输入文本并返回结果
        text_queries = self.encode_text(inputs)
        return text_queries

    def encode_text(self, text):
        # 检查输入文本的维度，确保为 2 或 3
        if text.ndim is None:
            raise ValueError("text must not be NoneType")
        if text.ndim not in [2, 3]:
            raise ValueError("Number of dimensions in text must be 2 or 3")
        
        squeeze_dim = False
        num_text = 1
        
        # 如果输入文本维度为 3，则重塑以进行批处理处理
        if text.ndim == 3:
            num_text = text.shape[1]
            batch_size, num_text, hidden_dim = text.shape
            text = text.reshape(batch_size * num_text, hidden_dim)
            squeeze_dim = True

        # 使用文本编码器对文本进行编码
        encoded_text = self.text_encoder(text)

        # 使用文本投影器对编码后的文本进行投影
        text_queries = self.text_projector(encoded_text)

        # 如果之前进行了维度压缩，则重新调整输出维度
        if squeeze_dim:
            _, hidden_dim = text_queries.shape
            text_queries = text_queries.reshape(batch_size, num_text, hidden_dim)
            # 如果存在上下文嵌入层，则将其与文本查询拼接
            if self.prompt_ctx is not None:
                text_queries_ctx = self.prompt_ctx.weight.unsqueeze(0).repeat(text_queries.shape[0], 1, 1)
                text_queries = torch.cat([text_queries, text_queries_ctx], dim=1)

        return text_queries


class OneFormerTaskModel(nn.Module):
    # OneFormerTaskModel 类定义
    def __init__(self, config: OneFormerConfig):
        super().__init__()
        # 初始化任务 MLP，使用配置中的参数
        self.task_mlp = OneFormerMLPPredictionHead(
            config.task_seq_len,
            config.hidden_dim,
            config.hidden_dim,
            2,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # 使用任务 MLP 处理输入并返回结果
        task_tokens = self.task_mlp(inputs)
        return task_tokens


ONEFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.
    Parameters:
        config ([`OneFormerConfig`]): Model configuration class with all the parameters of the model.
            初始化模型配置类，包含模型的所有参数。
            使用配置文件初始化时，不会加载与模型相关的权重，只加载配置信息。
            若要加载模型权重，请参阅 [`~PreTrainedModel.from_pretrained`] 方法。
"""
ONEFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`OneFormerProcessor`]. See
            [`OneFormerProcessor.__call__`] for details.
        task_inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Task inputs. Task inputs can be obtained using [`AutoImageProcessor`]. See [`OneFormerProcessor.__call__`]
            for details.
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
            Whether or not to return a [`~OneFormerModelOutput`] instead of a plain tuple.
"""


class OneFormerPreTrainedModel(PreTrainedModel):
    config_class = OneFormerConfig  # 指定配置类为 OneFormerConfig
    base_model_prefix = "model"  # 基础模型前缀设为 "model"
    main_input_name = "pixel_values"  # 主要输入名称设为 "pixel_values"


@add_start_docstrings(
    "The bare OneFormer Model outputting raw hidden-states without any specific head on top.",
    ONEFORMER_START_DOCSTRING,
)
class OneFormerModel(OneFormerPreTrainedModel):
    main_input_name = ["pixel_values", "task_inputs"]  # 主要输入名称包括 "pixel_values" 和 "task_inputs"

    def __init__(self, config: OneFormerConfig):
        super().__init__(config)
        self.pixel_level_module = OneFormerPixelLevelModule(config)  # 创建像素级模块
        self.transformer_module = OneFormerTransformerModule(in_features=config.conv_dim, config=config)  # 创建变换器模块
        self.task_encoder = OneFormerTaskModel(config)  # 创建任务编码器
        self.is_training = config.is_training  # 获取是否训练的标志

        if self.is_training:
            self.text_mapper = OneFormerTextMapper(config)  # 若在训练状态，则创建文本映射器
        else:
            self.text_mapper = None

        self.post_init()  # 完成初始化后的处理

    @add_start_docstrings_to_model_forward(ONEFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OneFormerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        task_inputs: Tensor,
        text_inputs: Optional[Tensor] = None,
        pixel_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    "OneFormer Model for instance, semantic and panoptic image segmentation.",
    ONEFORMER_START_DOCSTRING,
)
class OneFormerForUniversalSegmentation(OneFormerPreTrainedModel):
    main_input_name = ["pixel_values", "task_inputs"]

    # 初始化函数，接受一个 OneFormerConfig 对象作为参数
    def __init__(self, config: OneFormerConfig):
        # 调用父类构造函数，传入配置参数
        super().__init__(config)
        # 根据配置参数创建一个 OneFormerModel 对象
        self.model = OneFormerModel(config)

        # 根据配置参数创建一个 OneFormerHungarianMatcher 对象
        self.matcher = OneFormerHungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=config.train_num_points,
        )

        # 设置损失权重字典，用于加权不同类型的损失函数
        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
            "loss_contrastive": config.contrastive_weight,
        }

        # 根据配置参数创建一个 OneFormerLoss 对象作为损失函数
        self.criterion = OneFormerLoss(
            num_classes=config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
            num_points=config.train_num_points,
            oversample_ratio=config.oversample_ratio,
            importance_sample_ratio=config.importance_sample_ratio,
            contrastive_temperature=config.contrastive_temperature,
        )

        # 执行额外的初始化步骤
        self.post_init()

    # 计算损失函数字典的函数，返回损失函数字典
    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        contrastive_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
        text_queries: Tensor,
        auxiliary_predictions: Dict[str, Tensor],
        calculate_contrastive_loss: bool,
    ) -> Dict[str, Tensor]:
        # 调用损失函数计算器 criterion 计算损失
        loss_dict: Dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            contrastive_queries_logits=contrastive_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            text_queries=text_queries,
            auxiliary_predictions=auxiliary_predictions,
            calculate_contrastive_loss=calculate_contrastive_loss,
        )

        # 根据权重字典 weight_dict 对损失进行加权处理
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        # 返回加权后的损失函数字典
        return loss_dict

    # 计算总损失的函数，返回总损失值
    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        # 对损失字典中所有损失值进行求和
        return sum(loss_dict.values())

    # 添加模型输入的文档字符串和输出类型的文档字符串
    @add_start_docstrings_to_model_forward(ONEFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OneFormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `forward`，用于模型的前向传播
    # self 表示类的实例本身，这里是一个类方法
    # pixel_values: Tensor 是输入的像素数据张量
    # task_inputs: Tensor 是任务相关的输入张量
    # text_inputs: Optional[Tensor] 是可选的文本输入张量，默认为 None
    # mask_labels: Optional[List[Tensor]] 是可选的掩码标签列表，默认为 None
    # class_labels: Optional[List[Tensor]] 是可选的类别标签列表，默认为 None
    # pixel_mask: Optional[Tensor] 是可选的像素掩码张量，默认为 None
    # output_auxiliary_logits: Optional[bool] 是可选的是否输出辅助 logits，默认为 None
    # output_hidden_states: Optional[bool] 是可选的是否输出隐藏状态，默认为 None
    # output_attentions: Optional[bool] 是可选的是否输出注意力权重，默认为 None
    # return_dict: Optional[bool] 是可选的是否返回字典形式的输出，默认为 None
```