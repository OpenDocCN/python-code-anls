# `.\transformers\models\oneformer\modeling_oneformer.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 2022年由 SHI Labs 和 The HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）授权；
# 您不得使用本文件，除非符合许可证要求。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，否则根据许可证分发的软件均按“原样”分发，
# 不附带任何担保或条件，无论是明示的还是暗示的。
# 有关特定语言的详细信息，请参阅许可证，
# 限制在许可证下的许可。
""" PyTorch OneFormer 模型。"""
import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast

from ... import AutoBackbone
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_oneformer import OneFormerConfig

# 获取 logger
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "OneFormerConfig"
_CHECKPOINT_FOR_DOC = "shi-labs/oneformer_ade20k_swin_tiny"

# 预训练模型列表
ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shi-labs/oneformer_ade20k_swin_tiny",
    # 查看所有 OneFormer 模型 https://huggingface.co/models?filter=oneformer
]

# 导入 scipy 可选模块
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 根据给定的模块和数量 N 返回克隆模块列表
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# 从 transformers.models.deformable_detr.modeling_deformable_detr.multi_scale_deformable_attention 复制来的函数
def multi_scale_deformable_attention(
    value: Tensor, value_spatial_shapes: Tensor, sampling_locations: Tensor, attention_weights: Tensor
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    # 将 value 分隔成相应尺寸的值列表
    value_list = value.split([height.item() * width.item() for height, width in value_spatial_shapes], dim=1)
    # 计算采样网格
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    # 遍历每个级别的空间形状和对应的索引
    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # 将值列表展平并进行转置，reshape成(batch_size*num_heads, hidden_dim, height, width)的形式
        value_l_ = (
            value_list[level_id].flatten(2).transpose(1, 2).reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # 从sampling_grids中获取采样网格数据，转置并展平
        sampling_grid_l_ = sampling_grids[:, :, :, level_id].transpose(1, 2).flatten(0, 1)
        # 使用双线性插值对value_l_进行采样，得到sampling_value_l_
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        # 将sampling_value_l_添加到sampling_value_list中
        sampling_value_list.append(sampling_value_l_)
    # 将注意力权重矩阵转置并reshape成(batch_size * num_heads, 1, num_queries, num_levels * num_points)的形式
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * num_heads, 1, num_queries, num_levels * num_points
    )
    # 对sampling_value_list和attention_weights进行加权求和运算，得到output
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    # 返回output并转置和保持连续性
    return output.transpose(1, 2).contiguous()
# 从 transformers.models.maskformer.modeling_maskformer.dice_loss 复制而来的函数，计算 DICE loss，用于计算两个掩码之间的相似度
def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    """
    计算 DICE loss，类似于掩码的广义 IOU，计算公式如下：

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x \cap y }{x \cup y + 1}} $$

    在实践中，由于 `labels` 是一个二进制掩码（只有 0 和 1），dice 可以按如下方式计算

    $$ \mathcal{L}_{\text{dice}(x, y) = 1 - \frac{2 * x * y }{x + y + 1}} $$

    参数：
        inputs (`torch.Tensor`)：
            表示一个掩码的张量。
        labels (`torch.Tensor`)：
            与 inputs 形状相同的张量。存储了 inputs 中每个元素的二进制分类标签（负类为 0，正类为 1）。
        num_masks (`int`)：
            当前批次中掩码的数量，用于归一化。

    返回：
        `torch.Tensor`：计算得到的损失。
    """
    # 对输入掩码应用 sigmoid 函数
    probs = inputs.sigmoid().flatten(1)
    # 计算分子部分，即输入掩码与标签的交集的两倍
    numerator = 2 * (probs * labels).sum(-1)
    # 计算分母部分，即输入掩码和标签的和
    denominator = probs.sum(-1) + labels.sum(-1)
    # 计算损失，公式为 1 - (分子 + 1) / (分母 + 1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 对损失进行求和并除以掩码数量，取平均
    loss = loss.sum() / num_masks
    return loss


# 从 transformers.models.mask2former.modeling_mask2former.sigmoid_cross_entropy_loss 复制而来的函数，计算 sigmoid 交叉熵损失
def sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor, num_masks: int) -> torch.Tensor:
    """
    参数：
        inputs (`torch.Tensor`)：
            任意形状的浮点张量。
        labels (`torch.Tensor`)：
            与 inputs 形状相同的张量。存储了 inputs 中每个元素的二进制分类标签（负类为 0，正类为 1）。

    返回：
        loss (`torch.Tensor`)：计算得到的损失。
    """
    # 使用 nn.BCEWithLogitsLoss() 函数创建损失函数
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    # 计算交叉熵损失
    cross_entropy_loss = criterion(inputs, labels)

    # 对损失进行求平均并除以掩码数量
    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss


# 从 transformers.models.maskformer.modeling_maskformer.pair_wise_dice_loss 复制而来的函数，计算两个掩码之间的相似度
def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    """
    Dice loss 的对比版本，参见 `dice_loss` 的用法。

    参数：
        inputs (`torch.Tensor`)：
            表示一个掩码的张量。
        labels (`torch.Tensor`)：
            与 inputs 形状相同的张量。存储了 inputs 中每个元素的二进制分类标签（负类为 0，正类为 1）。

    返回：
        `torch.Tensor`：每对掩码之间的计算损失。
    """
    # 对输入掩码应用 sigmoid 函数
    inputs = inputs.sigmoid().flatten(1)
    # 计算分子部分，即输入掩码与标签的点积的两倍
    numerator = 2 * torch.matmul(inputs, labels.T)
    # 使用广播运算获得一个 [num_queries, NUM_CLASSES] 矩阵
    denominator = inputs.sum(-1)[:, None] + labels.sum(-1)[None, :]
    # 计算损失，公式为 1 - (分子 + 1) / (分母 + 1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss
# 从transformers.models.mask2former.modeling_mask2former.pair_wise_sigmoid_cross_entropy_loss中复制而来的函数
def pair_wise_sigmoid_cross_entropy_loss(inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    交叉熵损失的一对一版本，参见`sigmoid_cross_entropy_loss`的用法。

    Args:
        inputs (`torch.Tensor`):
            表示掩码的张量。
        labels (`torch.Tensor`):
            与inputs具有相同形状的张量。存储inputs中每个元素的二元分类标签
            （0表示负类，1表示正类）。

    Returns:
        loss (`torch.Tensor`): 每对之间计算的损失。
    """

    # 获取输入的高度和宽度
    height_and_width = inputs.shape[1]

    # 创建二元交叉熵损失函数对象
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    # 计算正类别的交叉熵损失
    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    # 计算负类别的交叉熵损失
    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))

    # 计算正类别的损失
    loss_pos = torch.matmul(cross_entropy_loss_pos, labels.T)
    # 计算负类别的损失
    loss_neg = torch.matmul(cross_entropy_loss_neg, (1 - labels).T)
    # 计算总的损失
    loss = loss_pos + loss_neg
    # 对损失进行归一化
    loss = loss / height_and_width
    return loss


# 从transformers.models.mask2former.modeling_mask2former.sample_point中复制而来的函数
def sample_point(
    input_features: torch.Tensor, point_coordinates: torch.Tensor, add_dim=False, **kwargs
) -> torch.Tensor:
    """
    对`torch.nn.functional.grid_sample`的一个包装，以支持3D point_coordinates张量。

    Args:
        input_features (`torch.Tensor` of shape (batch_size, channels, height, width)):
            包含高*宽网格上特征映射的张量
        point_coordinates (`torch.Tensor` of shape (batch_size, num_points, 2) or (batch_size, grid_height, grid_width,
        2)):
            包含[0, 1] * [0, 1]归一化点坐标的张量
        add_dim (`bool`):
            用于跟踪添加的维度的布尔值

    Returns:
        point_features (`torch.Tensor` of shape (batch_size, channels, num_points) or (batch_size, channels,
        height_grid, width_grid)):
            包含`point_coordinates`中点的特征的张量。
    """
    # 如果point_coordinates的维度为3，则添加一个维度
    if point_coordinates.dim() == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)

    # 使用双线性插值从`point_coordinates`中获取点的特征
    point_features = torch.nn.functional.grid_sample(input_features, 2.0 * point_coordinates - 1.0, **kwargs)
    # 如果添加了维度，则压缩维度
    if add_dim:
        point_features = point_features.squeeze(3)

    return point_features


# 从https://github.com/SHI-Labs/OneFormer/blob/33ebb56ed34f970a30ae103e786c0cb64c653d9a/oneformer/modeling/matcher.py#L93中重构而来
class OneFormerHungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0, num_points: int = 12544
    ):
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
        # 调用父类初始化方法
        super().__init__()
        # 如果分类错误、掩模损失和Dice损失都为零，则抛出值错误异常
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs cant be 0")
        # 设置分类错误、掩模损失和Dice损失的相对权重以及采样点数
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
# 定义一个损失函数类 OneFormerLoss，用于计算 OneFormer 模型的损失
class OneFormerLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: OneFormerHungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        contrastive_temperature: float = None,
    ):
        # 初始化损失函数所需的参数
        # num_classes: 类别数量
        # matcher: 用于计算预测和标签之间匹配的模块
        # weight_dict: 不同损失函数的权重
        # eos_coef: 空类的权重
        # num_points: 用于计算点级损失的点数
        # oversample_ratio: 用于点级损失计算的过采样比例
        # importance_sample_ratio: 用于点级损失计算的重要性采样比例
        # contrastive_temperature: 用于缩放对比损失逻辑值的温度
        requires_backends(self, ["scipy"])
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        # 创建一个权重张量，最后一个元素为 eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # 初始化点级损失计算相关参数
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.contrastive_temperature = contrastive_temperature
        # 如果设置了对比损失温度，则创建一个可学习的缩放参数
        if self.contrastive_temperature is not None:
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / contrastive_temperature)))

    # 定义一个辅助函数，用于计算一组列表中每个位置上的最大值
    def _max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes
    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        # 获取批次中的最大尺寸
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        # 计算最终尺寸
        batch_shape = [batch_size] + max_size
        b, _, h, w = batch_shape
        # 获取元数据
        dtype = tensors[0].dtype
        device = tensors[0].device
        # 创建零填充的张量，形状为批次最大尺寸，与填充掩码
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((b, h, w), dtype=torch.bool, device=device)
        # 将张量填充至最大尺寸
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[: tensor.shape[0], : tensor.shape[1], : tensor.shape[2]].copy_(tensor)
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

        image_queries = contrastive_queries_logits.float()

        # 对图像查询进行归一化
        image_queries = nn.functional.normalize(image_queries.flatten(1), dim=-1)
        # 对文本查询进行归一化
        text_queries = nn.functional.normalize(text_queries.flatten(1), dim=-1)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)

        # 计算文本查询与图像查询之间的相似度
        logits_per_text = torch.matmul(text_queries, image_queries.t()) * logit_scale
        logits_per_img = logits_per_text.t()

        # 计算图像损失
        loss_img = nn.functional.cross_entropy(
            logits_per_img, torch.arange(len(logits_per_img), device=logits_per_text.device)
        )
        # 计算文本损失
        loss_text = nn.functional.cross_entropy(
            logits_per_text, torch.arange(len(logits_per_text), device=logits_per_text.device)
        )

        # 计算对比损失
        loss_contrastive = loss_img + loss_text

        losses = {"loss_contrastive": loss_contrastive}
        return losses

    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]
    ) -> Dict[str, Tensor]:
        """计算与标签相关的损失，使用交叉熵。

        Args:
            class_queries_logits (`torch.Tensor`):
                形状为 `batch_size, num_queries, num_labels` 的张量
            class_labels (`List[torch.Tensor]`):
                形状为 `(labels)` 的类标签列表
            indices (`Tuple[np.array])`:
                由匈牙利匹配器计算出的索引

        Returns:
            `Dict[str, Tensor]`: 包含以下键的 `torch.Tensor` 字典:
            - **loss_cross_entropy** -- 使用预测和真实标签计算的交叉熵损失
        """
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)

        # 形状为 (batch_size, num_queries) 的张量
        target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        # 形状为 (batch_size, num_queries) 的张量
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_classes, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        # 调换 pred_logits 的维度 (batch_size, num_queries, num_labels) -> (batch_size, num_labels, num_queries)
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self, masks_queries_logits: Tensor, mask_labels: List[Tensor], indices: Tuple[np.array], num_masks: int
    ) -> Dict[str, Tensor]:
        """计算与掩码相关的损失，使用焦点和骰子损失。

        Args:
            masks_queries_logits (`torch.Tensor`):
                形状为 `batch_size, num_queries, height, width` 的张量
            mask_labels (`torch.Tensor`):
                形状为 `(labels, height, width)` 的掩码标签列表。
            indices (`Tuple[np.array])`:
                由匈牙利匹配器计算的索引。
            num_masks (`int)`:
                掩码数量，用于归一化。

        Returns:
            `Dict[str, Tensor]`: 包含两个键的 `torch.Tensor` 字典:
            - **loss_mask** -- 使用预测值和真实掩码之间的 Sigmoid CE 损失计算的损失。
            - **loss_dice** -- 使用预测值和真实掩码之间的骰子损失计算的损失。
        """
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        # 形状为 (batch_size * num_queries, height, width)
        pred_masks = masks_queries_logits[src_idx]
        # 形状为 (batch_size, num_queries, height, width)
        # 填充所有并将目标叠加到 num_labels 维度
        # 将预测值上采样到目标尺寸，必须添加一个维度以使用 interpolate
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # 采样点坐标
            point_coords = self.sample_points_using_uncertainty(
                pred_masks,
                self.calculate_uncertainty,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # 获取真实标签
            point_labels = sample_point(target_masks, point_coords, align_corners=False).squeeze(1)

        point_logits = sample_point(pred_masks, point_coords, align_corners=False).squeeze(1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    # 从 transformers.models.mask2former.modeling_mask2former.Mask2FormerLoss.calculate_uncertainty 复制而来
    # 计算不确定性得分的函数
    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (`torch.Tensor`):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:
            the number of foreground classes. The values are logits.

        Returns:
            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most
            uncertain locations having the highest uncertainty score.
        """
        # 计算不确定性得分，使用负的logits的绝对值
        uncertainty_scores = -(torch.abs(logits))
        # 返回不确定性得分
        return uncertainty_scores

    # 从给定的不确定性函数中使用不确定性的采样点
    # 每个输入点都会生成一个独特的不确定性分数，帮助选择最不确定的点来进行采样
    def sample_points_using_uncertainty(
        self,
        logits: torch.Tensor,
        uncertainty_function,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """

        # Calculate the number of boxes
        num_boxes = logits.shape[0]
        # Calculate the number of points to be sampled with oversampling
        num_points_sampled = int(num_points * oversample_ratio)

        # Get random point coordinates within [0, 1] * [0, 1] coordinate space for each box
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        # Sample prediction values for the generated point coordinates
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        # Calculate uncertainties for the sampled points using the provided uncertainty function
        point_uncertainties = uncertainty_function(point_logits)

        # Calculate the number of uncertain points based on the importance sampling ratio
        num_uncertain_points = int(importance_sample_ratio * num_points)
        # Calculate the number of randomly sampled points
        num_random_points = num_points - num_uncertain_points

        # Select the indices of uncertain points based on their uncertainties
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        # Adjust the indices to match the flattened point_coordinates tensor
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        # Reshape the point_coordinates tensor to select uncertain points
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)

        # If there are random points to be sampled, generate and concatenate them
        if num_random_points > 0:
            point_coordinates = torch.cat(
                [point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)],
                dim=1,
            )
        # Return the final coordinates for sampled points
        return point_coordinates

    def _get_predictions_permutation_indices(self, indices):
        # Permute predictions according to the provided indices
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_targets_permutation_indices(self, indices):
        # Permute labels according to the provided indices
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices
    # 定义一个函数用于模型的前向传播
    def forward(
        self,
        masks_queries_logits: Tensor,  # 掩膜查询的logits
        class_queries_logits: Tensor,  # 类别查询的logits
        contrastive_queries_logits: Tensor,  # 对比查询的logits
        mask_labels: List[Tensor],  # 掩膜标签列表
        class_labels: List[Tensor],  # 类别标签列表
        text_queries: Tensor,  # 文本查询
        auxiliary_predictions: Optional[Dict[str, Tensor]] = None,  # 辅助预测结果的字典，可选参数
        calculate_contrastive_loss: bool = True,  # 是否计算对比损失的布尔值
    # 定义一个函数用于获取类别标签中目标掩膜的数量，用于归一化
    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        # 计算批次中目标掩膜的平均数量
        num_masks = sum([len(classes) for classes in class_labels])
        # 将目标掩膜数转化为张量，设置数据类型为浮点数并设置设备
        num_masks_pt = torch.as_tensor([num_masks], dtype=torch.float, device=device)
        # 返回目标掩膜数的张量
        return num_masks_pt
# 定义了一个数据类 OneFormerTransformerDecoderOutput，用于保存 Transformer 解码器的输出结果
@dataclass
class OneFormerTransformerDecoderOutput(BaseModelOutput):
    """
    Base class for outputs of the Transformer decoder. This class adds attributes for class predictions, mask
    predictions and contrastive logits to BaseModelOutputWithCrossAttentions.
    用于保存 Transformer 解码器的输出结果的基类，该类添加了用于类预测、蒙版预测和对比日志的属性到BaseModelOutputWithCrossAttentions。

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

    object_queries: torch.FloatTensor = None
    contrastive_logits: Optional[torch.FloatTensor] = None
    prediction_masks: torch.FloatTensor = None
    prediction_class: torch.FloatTensor = None
    auxiliary_predictions: Optional[Tuple[Dict[str, torch.FloatTensor]]] = None


@dataclass
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

    multi_scale_features: Tuple[torch.FloatTensor] = None
    mask_features: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class OneFormerPixelLevelModuleOutput(ModelOutput):
    """
    OneFormer's pixel level module output. It returns both the last and (optionally) the hidden states from the
    OneFormer's pixel level module output. It returns both the last and (optionally) the hidden states from the PixelDecoder.
    """
    # `encoder` and `decoder`. By default, the `encoder` is a Swin/Dinat Backbone and the `decoder` is a Multi-Scale
    # Deformable Attention based decoder.
    # 默认情况下，`encoder` 是一个 Swin/Dinat 骨干网络，而 `decoder` 是一个基于多尺度可变形注意力的解码器。
    
    # Args:
    #     encoder_features (List of `(torch.FloatTensor)`):
    #         List of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
    #         called feature maps) of the model at the output of each stage.
    # 参数：
    #     encoder_features (List of `(torch.FloatTensor)`):
    #         一个列表，其中每个元素是形状为`(batch_size, num_channels, height, width)`的`torch.FloatTensor`。模型在每个阶段输出的隐藏状态（也称为特征图）。
    
    #     decoder_features (List of `(torch.FloatTensor)`):
    #         List of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`. Hidden-states (also
    #         called feature maps) of the model at the output of each stage.
    #     decoder_features (List of `(torch.FloatTensor)`):
    #         一个列表，其中每个元素是形状为`(batch_size, num_channels, height, width)`的`torch.FloatTensor`。模型在每个阶段输出的隐藏状态（也称为特征图）。
    
    #     decoder_last_feature (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)):
    #         1/4 scale features from the last Pixel Decoder Layer.
    #     decoder_last_feature (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)):
    #         最后一个像素解码器层的1/4尺度特征。
# 这是一个 OneFormerModelOutput 类，它是一个 ModelOutput 的子类。它包含了 OneFormerModel 的输出结果，包括编码器隐藏状态、像素解码器隐藏状态、Transformer 解码器隐藏状态等。
@dataclass
class OneFormerModelOutput(ModelOutput):
    """
    Class for outputs of [`OneFormerModel`]. This class returns all the needed hidden states to compute the logits.
    """

    # 编码器隐藏状态，可选的元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 像素解码器隐藏状态，可选的元组
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # Transformer 解码器隐藏状态，可选的 Tensor
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    # Transformer 解码器目标查询，Tensor
    transformer_decoder_object_queries: torch.FloatTensor = None
    # Transformer 解码器对比查询，可选的 Tensor
    transformer_decoder_contrastive_queries: Optional[torch.FloatTensor] = None
    # Transformer 解码器掩码预测，Tensor
    transformer_decoder_mask_predictions: torch.FloatTensor = None
    # Transformer 解码器类别预测，Tensor
    transformer_decoder_class_predictions: torch.FloatTensor = None
    # Transformer 解码器辅助预测，可选的元组字典
    transformer_decoder_auxiliary_predictions: Optional[Tuple[Dict[str, torch.FloatTensor]]] = None
    # 文本查询，可选的 Tensor
    text_queries: Optional[torch.FloatTensor] = None
    # 任务令牌，Tensor
    task_token: torch.FloatTensor = None
    # 注意力权重，可选的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 这是一个 OneFormerForUniversalSegmentationOutput 类，它是一个 ModelOutput 的子类。它包含了 OneFormerForUniversalSegmentation 模型的输出结果，可以直接传递给 OneFormerImageProcessor 进行后处理。
@dataclass
class OneFormerForUniversalSegmentationOutput(ModelOutput):
    """
    Class for outputs of [`OneFormerForUniversalSegmentationOutput`].

    This output can be directly passed to [`~OneFormerImageProcessor.post_process_semantic_segmentation`] or
    [`~OneFormerImageProcessor.post_process_instance_segmentation`] or
    [`~OneFormerImageProcessor.post_process_panoptic_segmentation`] depending on the task. Please, see
    [`~OneFormerImageProcessor] for details regarding usage.
    """

    # 损失，可选的 Tensor
    loss: Optional[torch.FloatTensor] = None
    # 类别查询的逻辑输出，Tensor
    class_queries_logits: torch.FloatTensor = None
    # 掩码查询的逻辑输出，Tensor
    masks_queries_logits: torch.FloatTensor = None
    # 辅助预测，列表
    auxiliary_predictions: List[Dict[str, torch.FloatTensor]] = None
    # 编码器隐藏状态，可选的元组
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 像素解码器隐藏状态，可选的列表
    pixel_decoder_hidden_states: Optional[List[torch.FloatTensor]] = None
    # Transformer 解码器隐藏状态，可选的 Tensor
    transformer_decoder_hidden_states: Optional[torch.FloatTensor] = None
    # Transformer 解码器目标查询，Tensor
    transformer_decoder_object_queries: torch.FloatTensor = None
    # Transformer 解码器对比查询，可选的 Tensor
    transformer_decoder_contrastive_queries: Optional[torch.FloatTensor] = None
    # Transformer 解码器掩码预测，Tensor
    transformer_decoder_mask_predictions: torch.FloatTensor = None
    # Transformer 解码器类别预测，Tensor
    transformer_decoder_class_predictions: torch.FloatTensor = None
    # Transformer 解码器辅助预测，可选的列表字典
    transformer_decoder_auxiliary_predictions: Optional[List[Dict[str, torch.FloatTensor]]] = None
    # 文本查询，可选的 Tensor
    text_queries: Optional[torch.FloatTensor] = None
    # 任务令牌，Tensor
    task_token: torch.FloatTensor = None
    # 注意力权重，可选的元组
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


# 这是一个 OneFormerPixelDecoderFrozenBatchNorm2d 类，它是一个 nn.Module 的子类。它实现了一个冻结的 BatchNorm2d 层，其中批量统计和仿射参数是固定的。
# 这个类的实现是从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrFrozenBatchNorm2d 修改而来的，将 DeformableDetr 改为了 OneFormerPixelDecoder。
class OneFormerPixelDecoderFrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """
    def __init__(self, n):
        # 初始化函数，继承父类
        super().__init__()
        # 注册模型参数：权重、偏置、均值、方差，分别初始化为全1、全0、全0、全1
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 根据给定的前缀获取 num_batches_tracked_key
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # 如果 num_batches_tracked_key 在 state_dict 中，则删除该键值对
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类的_load_from_state_dict函数，更新模型参数
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # 将权重、偏置、均值、方差重塑为适合卷积操作的形状
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = 1e-5
        # 计算 scale 缩放参数
        scale = weight * (running_var + epsilon).rsqrt()
        # 计算 bias 偏置参数
        bias = bias - running_mean * scale
        # 返回经过 BatchNorm 处理后的数据
        return x * scale + bias
```  
# 从transformers.models.detr.modeling_deformable_detr.DeformableDetrMultiscaleDeformableAttention修改为OneFormerPixelDecoderEncoder
class OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
    """
    Multiscale deformable attention as proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int):
        super().__init__()
        # 检查embed_dim是否可以被num_heads整除
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        # 计算每个attention head的维度
        dim_per_head = embed_dim // num_heads
        # 检查dim_per_head是否是2的幂
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )
        
        self.im2col_step = 128

        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        # 初始化一些Linear层
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    # 添加位置编码到输入张量
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
        # 在将隐藏状态投影到查询和键之前，将位置嵌入添加到隐藏状态中
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # 获取隐藏状态的批量大小、查询数量和特征维度
        batch_size, num_queries, _ = hidden_states.shape
        # 获取编码器隐藏状态的批量大小、序列长度和特征维度
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        # 检查空间形状的总和是否等于编码器隐藏状态的序列长度
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        # 对编码器隐藏状态进行值投影
        value = self.value_proj(encoder_hidden_states)
        # 如果存在注意力掩码，则反转注意力掩码
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], float(0))
        # 重塑值张量的形状以便用于多头注意力
        value = value.view(batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads)
        # 获取采样偏移量
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        # 获取注意力权重
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        # 对注意力权重进行 softmax 操作
        attention_weights = nn.functional.softmax(attention_weights, -1).view(
            batch_size, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        # 如果参考点的形状的最后一个维度为2
        if reference_points.shape[-1] == 2:
            # 计算采样位置
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        # 如果参考点的形状的最后一个维度为4
        elif reference_points.shape[-1] == 4:
            # 计算采样位置
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        # 如果参考点的最后一个维度既不是2也不是4，则引发 ValueError
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
        # 使用 PyTorch 实现的多尺度可变形注意力机制
        output = multi_scale_deformable_attention(value, spatial_shapes, sampling_locations, attention_weights)
        # 对输出进行投影
        output = self.output_proj(output)

        return output, attention_weights
# 定义一个名为 OneFormerPixelDecoderEncoderLayer 的 PyTorch 模块类
class OneFormerPixelDecoderEncoderLayer(nn.Module):
    # 初始化函数，接受 OneFormerConfig 对象作为输入
    def __init__(self, config: OneFormerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置模型的隐藏层维度大小
        self.embed_dim = config.conv_dim
        # 创建一个 OneFormerPixelDecoderEncoderMultiscaleDeformableAttention 注意力层
        self.self_attn = OneFormerPixelDecoderEncoderMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            n_levels=3,
            n_points=4,
        )
        # 创建一个层归一化层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 设置 dropout 比例
        self.dropout = config.dropout
        # 设置激活函数为 ReLU
        self.activation_fn = nn.functional.relu
        # 设置激活函数的 dropout 比例
        self.activation_dropout = config.dropout
        # 创建两个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        # 创建另一个层归一化层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 设置训练标志
        self.is_training = config.is_training

    # 前向传播函数
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
                Input to the layer.  传入该层的输入hidden_states，形状为(batch_size, sequence_length, hidden_size)
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.  注意力掩码，形状为(batch_size, sequence_length)
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.  位置嵌入，添加到hidden_states中
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.  参考点
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.  主干特征图的空间形状
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.  级别开始索引
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.  是否返回所有注意力层的注意力张量
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps. 在多尺度特征图上应用多尺度可变形注意力模块
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

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.is_training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.is_training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.is_training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.is_training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 从transformers.models.detr.modeling_deformable_detr.DeformableDetrEncoder修改为OneFormerPixelDecoderEncoderOnly
class OneFormerPixelDecoderEncoderOnly(nn.Module):
    """
    由*config.encoder_layers*个可变形注意力层组成的Transformer编码器。 每一层都是一个[`OneFormerPixelDecoderEncoderLayer`]。

    编码器通过多个可变形注意力层更新多尺度特征图。

    Args:
        config: OneFormerConfig
    """

    def __init__(self, config: OneFormerConfig):
        # 初始化函数，接受OneFormerConfig类型的config参数
        super().__init__()

        self.config = config
        self.dropout = config.dropout
        # 创建一个由config.encoder_layers个OneFormerPixelDecoderEncoderLayer对象组成的模块列表
        self.layers = nn.ModuleList([OneFormerPixelDecoderEncoderLayer(config) for _ in range(config.encoder_layers)])

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        获取每个特征图的参考点。用于解码器。

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                每个特征图的空间形状。
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                每个特征图的有效比例。
            device (`torch.device`):
                创建张量的设备。
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
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
# 从transformers.models.mask2former.modeling_mask2former.Mask2FormerPixelDecoder修改为OneFormerPixelDecoder
class OneFormerPixelDecoder(nn.Module):
    # 获取所有特征图的有效比例
    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""
        # 获取掩码的形状信息
        _, height, width = mask.shape
        # 计算每行有效像素的数量
        valid_height = torch.sum(~mask[:, :, 0], 1)
        # 计算每列有效像素的数量
        valid_width = torch.sum(~mask[:, 0, :], 1)
        # 将有效像素的数量转换为指定数据类型，并计算高度和宽度的有效比例
        valid_ratio_heigth = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        # 沿着最后一个维度将宽度和高度的有效比例堆叠成张量
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        # 返回有效比例张量
        return valid_ratio

    def forward(
        self,
        features,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
# 从transformers.models.mask2former.modeling_mask2former中修改而来，用于定义OneFormerPixelLevelModule类
class OneFormerPixelLevelModule(nn.Module):
    def __init__(self, config: OneFormerConfig):
        """
        提出的像素级模块，见[Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)。
        它通过一个骨干网络和一个像素解码器对输入图像进行处理，生成多尺度特征图和像素嵌入。

        Args:
            config ([`OneFormerConfig`]):
                用于实例化此模型的配置。
        """
        super().__init__()
        # 获取骨干网络的配置
        backbone_config = config.backbone_config
        # 使用给定的配置创建骨干网络
        self.encoder = AutoBackbone.from_config(backbone_config)
        # 使用给定的配置创建像素解码器，并传入骨干网络的通道数作为特征通道数
        self.decoder = OneFormerPixelDecoder(config, feature_channels=self.encoder.channels)

    def forward(self, pixel_values: Tensor, output_hidden_states: bool = False) -> OneFormerPixelLevelModuleOutput:
        # 将像素值输入骨干网络，获取特征图列表
        features: List[Tensor] = self.encoder(pixel_values).feature_maps
        # 将特征图列表输入像素解码器，获取解码器输出
        decoder_output: OneFormerPixelDecoderOutput = self.decoder(features, output_hidden_states=output_hidden_states)
        # 返回像素级模块的输出，包括骨干网络的特征、解码器的多尺度特征以及解码器的最后一个特征
        return OneFormerPixelLevelModuleOutput(
            encoder_features=tuple(features),
            decoder_features=decoder_output.multi_scale_features,
            decoder_last_feature=decoder_output.mask_features,
        )


# 从transformers.models.detr.modeling_detr中修改而来，用于定义OneFormerAttention类
class OneFormerAttention(nn.Module):
    """
    'Attention Is All You Need'论文中的多头注意力机制。这里我们为查询和键添加位置嵌入（如DETR论文中所述）。
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
        # 初始化注意力机制的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        # 计算每个头的维度
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim必须可以被num_heads整除（得到`embed_dim`：{self.embed_dim}和`num_heads`：{num_heads}）。"
            )
        # 缩放因子
        self.scaling = self.head_dim**-0.5

        # 初始化线性变换层，用于将查询、键、值和输出投影到指定维度的空间
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        # 将张量重塑成合适的形状以便进行多头注意力计算
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        # 如果存在位置嵌入，则将其添加到张量中
        return tensor if position_embeddings is None else tensor + position_embeddings
    # 前向传播函数，用于Transformer的自注意力机制
    def forward(
        self,
        # 输入的隐藏状态张量，shape为(batch_size, sequence_length, hidden_size)
        hidden_states: torch.Tensor,
        # 注意力掩码张量，可选参数，默认为None，shape为(batch_size, sequence_length)
        attention_mask: Optional[torch.Tensor] = None,
        # 位置嵌入张量，可选参数，默认为None，shape为(batch_size, sequence_length, hidden_size)
        position_embeddings: Optional[torch.Tensor] = None,
        # 键值状态张量，可选参数，默认为None，shape为(batch_size, num_heads, sequence_length, head_dim)
        key_value_states: Optional[torch.Tensor] = None,
        # 键值位置嵌入张量，可选参数，默认为None，shape为(batch_size, num_heads, sequence_length, head_dim)
        key_value_position_embeddings: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，默认为False
        output_attentions: bool = False,
# 定义 OneFormerTransformerDecoderSelfAttentionLayer 类，继承自 nn.Module
class OneFormerTransformerDecoderSelfAttentionLayer(nn.Module):
    # 初始化函数，接收输入 embed_dim、num_heads、dropout、activation、normalize_before、layer_norm_eps
    def __init__(
        self, embed_dim, num_heads, dropout=0.0, activation="relu", normalize_before=False, layer_norm_eps=1e-05
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 创建 self_attn 层，通过 OneFormerAttention 类实现自注意力机制
        self.self_attn = OneFormerAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, is_decoder=True)
        # 创建 LayerNorm 层，用于归一化处理
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        # 创建 Dropout 层，实现随机失活
        self.dropout = nn.Dropout(dropout)
        # 激活函数选择，根据 activation 参数选择对应激活函数
        self.activation = ACT2FN[activation]
        # 归一化处理选择，根据 normalize_before 参数判断是否在进行自注意力之前归一化
        self.normalize_before = normalize_before

    # 对输入张量添加位置编码，如果 pos 为 None 则不添加，否则加上位置编码
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # 实现自注意力层的前处理（归一化->自注意力计算->残差连接->归一化）
    def forward_post(
        self,
        output,
        output_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 进行自注意力计算，得到 output2 和 attention_weights
        output2, attention_weights = self.self_attn(
            hidden_states=output, position_embeddings=query_pos, attention_mask=output_mask, output_attentions=True
        )
        # 残差连接和随机失活
        output = output + self.dropout(output2)
        # 归一化处理
        output = self.norm(output)
        # 返回处理结果和注意力权重
        return output, attention_weights

    # 实现自注意力层的后处理（归一化->自注意力计算->残差连接）
    def forward_pre(
        self,
        output,
        output_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 归一化处理
        output2 = self.norm(output)
        # 自注意力计算并得到 output2 和 attention_weights
        output2, attention_weights = self.self_attn(
            hidden_states=output2, position_embeddings=query_pos, attention_mask=output_mask, output_attentions=True
        )
        # 残差连接和随机失活
        output = output + self.dropout(output2)
        # 返回处理结果和注意力权重
        return output, attention_weights

    # 实现自注意力层的前向传播
    def forward(
        self,
        output,
        output_mask: Optional[Tensor] = None,
        output_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 根据 normalize_before 判断是否进行前处理或后处理
        if self.normalize_before:
            return self.forward_pre(output, output_mask, output_key_padding_mask, query_pos)
        return self.forward_post(output, output_mask, output_key_padding_mask, query_pos)


# 定义 OneFormerTransformerDecoderCrossAttentionLayer 类，继承自 nn.Module
class OneFormerTransformerDecoderCrossAttentionLayer(nn.Module):
    # 初始化函数，接收输入 embed_dim、num_heads、dropout、activation、normalize_before、layer_norm_eps
    def __init__(
        self, embed_dim, num_heads, dropout=0.0, activation="relu", normalize_before=False, layer_norm_eps=1e-05
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 创建 multihead_attn 层，通过 nn.MultiheadAttention 类实现跨注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # 创建 LayerNorm 层，用于归一化处理
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        # 创建 Dropout 层，实现随机失活
        self.dropout = nn.Dropout(dropout)
        # 激活函数选择，根据 activation 参数选择对应激活函数
        self.activation = ACT2FN[activation]
        # 归一化处理选择，根据 normalize_before 参数判断是否在进行跨注意力之前归一化
        self.normalize_before = normalize_before

    # 对输入张量添加位置编码，如果 pos 为 None 则不添加，否则加上位置编码
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    # 定义一个方法，用于实现多头自注意力机制
    def forward_post(
        self,
        output,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 将输出和查询位置嵌入结合起来，作为查询，进行多头自注意力计算，并返回结果和注意力权重
        output2, attention_weights = self.multihead_attn(
            query=self.with_pos_embed(output, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        # 将原输出与多头自注意力结果相加，并加上 dropout 后返回
        output = output + self.dropout(output2)
        # 对输出进行归一化
        output = self.norm(output)

        return output, attention_weights

    # 定义另一个方法，用于实现多头自注意力机制，不同之处在于归一化操作在注意力计算之前
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
        # 将归一化后的输出与查询位置嵌入结合起来，作为查询，进行多头自注意力计算，并返回结果和注意力权重
        output2, attention_weights = self.multihead_attn(
            query=self.with_pos_embed(output2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        # 将原输出与多头自注意力结果相加，并加上 dropout 后返回
        output = output + self.dropout(output2)

        return output, attention_weights

    # 定义一个方法，根据 normalize_before 参数选择调用 forward_post 或 forward_pre 方法
    def forward(
        self,
        output,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # 如果 normalize_before 为 True，则调用 forward_pre 方法，否则调用 forward_post 方法
        if self.normalize_before:
            return self.forward_pre(output, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
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
        # 初始化 Feedforward 层
        super().__init__()
        # 线性变换1，将输入维度 d_model 转换为 dim_feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # 添加 dropout
        self.dropout = nn.Dropout(dropout)
        # 线性变换2，将 dim_feedforward 转换为 d_model
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorm 层，标准化输出
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # 激活函数，根据 activation 参数选择相应激活函数
        self.activation = ACT2FN[activation]
        # 是否在 LayerNorm 之前标准化
        self.normalize_before = normalize_before

    # 添加位置编码
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # 后处理层
    def forward_post(self, output):
        # Feedforward 网络
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        # 输出加上残差连接后再标准化
        output = output + self.dropout(output2)
        output = self.norm(output)
        return output

    # 先处理层
    def forward_pre(self, output):
        # 先标准化
        output2 = self.norm(output)
        # Feedforward 网络
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2))))
        # 输出加上残差连接
        output = output + self.dropout(output2)
        return output

    # 前向传播方法
    def forward(self, output):
        # 根据 normalize_before 参数选择调用先处理还是后处理
        if self.normalize_before:
            return self.forward_pre(output)
        return self.forward_post(output)


# 预测头部的 MLP 模型
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
        # 初始化 MLP 预测头
        super().__init__()
        # 输入维度和隐藏层维度的列表
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        # 隐藏层维度和输出维度的列表
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        layers = []
        # 构建多层预测块
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(
                PredictionBlock(in_dim, out_dim, activation=nn.ReLU() if i < num_layers - 1 else nn.Identity())
            )

        # 使用 Sequential 封装多层预测块
        self.layers = nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)


# 从原始实现重构而来的 Transformer 解码层
class OneFormerTransformerDecoderLayer(nn.Module):
    # 初始化方法，接收一个 OneFormerConfig 对象作为参数
    def __init__(self, config: OneFormerConfig):
        # 调用父类初始化方法
        super().__init__()
        # 设置嵌入维度为配置中的隐藏维度
        self.embed_dim = config.hidden_dim
        # 设置特征级别数为3
        self.num_feature_levels = 3
    
        # 创建交叉注意力层对象，设置参数
        self.cross_attn = OneFormerTransformerDecoderCrossAttentionLayer(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=0.0,
            normalize_before=config.pre_norm,
            layer_norm_eps=config.layer_norm_eps,
        )
    
        # 创建自注意力层对象，设置参数
        self.self_attn = OneFormerTransformerDecoderSelfAttentionLayer(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=0.0,
            normalize_before=config.pre_norm,
            layer_norm_eps=config.layer_norm_eps,
        )
    
        # 创建前馈神经网络层对象，设置参数
        self.ffn = OneFormerTransformerDecoderFFNLayer(
            d_model=self.embed_dim,
            dim_feedforward=config.dim_feedforward,
            dropout=0.0,
            normalize_before=config.pre_norm,
            layer_norm_eps=config.layer_norm_eps,
        )
    
    # 前向传播方法，接收多个参数
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
            index (`int`): Transformer解码器中层的索引。
            output (`torch.FloatTensor`): 形状为`(N, batch, hidden_dim)`的对象查询。
            multi_stage_features (`List[torch.Tensor]`): 像素解码器中的多尺度特征。
            multi_stage_positional_embeddings (`List[torch.Tensor]`):
                多个multi_stage_features的位置嵌入。
            attention_mask (`torch.FloatTensor`): 用于遮蔽交叉注意力层的注意力遮罩。
            query_embeddings (`torch.FloatTensor`, *可选*):
                添加到自注意力层中的查询和键的位置嵌入。
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。有关更多细节，请参阅返回张量下的`attentions`。
        """

        level_index = index % self.num_feature_levels
        attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False

        # Masked Cross Attention
        output, cross_attn_weights = self.cross_attn(
            output,
            multi_stage_features[level_index],
            memory_mask=attention_mask,
            memory_key_padding_mask=None,  # 这里我们不在填充区域应用遮蔽
            pos=multi_stage_positional_embeddings[level_index],
            query_pos=query_embeddings,
        )

        # Self Attention
        output, self_attn_weights = self.self_attn(
            output,
            output_mask=None,
            output_key_padding_mask=None,
            query_pos=query_embeddings,
        )

        # Fully Connected
        output = self.ffn(output)

        outputs = (output,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
# 定义一个包含多个解码层的解码器类
class OneFormerTransformerDecoderQueryTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # 复制指定数量的解码层
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    # 解码器的前向传播函数
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
        # 存储中间结果
        intermediate = []

        # 遍历所有解码层
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
            # 如果需要返回中间结果，则将当前输出添加到中间结果列表
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # 如果存在归一化函数，则对最终输出进行归一化处理
        if self.norm is not None:
            output = self.norm(output)
            # 如果需要返回中间结果，则更新中间结果列表
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        # 如果需要返回中间结果，则将中间结果拼接后返回
        if self.return_intermediate:
            return torch.stack(intermediate)

        # 返回最终输出，并增加维度
        return output.unsqueeze(0)


# 定义一个解码层类
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
        # 多头注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 前馈网络实现
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = ACT2FN[activation]
        self.normalize_before = normalize_before

    # 如果存在位置编码，将位置编码与张量相加
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    # 该函数实现了Transformer编码器层的前向传播过程
    def forward_post(
        self,
        output,  # 输入的输出张量
        memory,  # 输入的记忆张量
        output_mask: Optional[Tensor] = None,  # 输出张量的mask张量（可选）
        memory_mask: Optional[Tensor] = None,  # 记忆张量的mask张量（可选）
        output_key_padding_mask: Optional[Tensor] = None,  # 输出张量的键padding mask张量（可选）
        memory_key_padding_mask: Optional[Tensor] = None,  # 记忆张量的键padding mask张量（可选）
        pos: Optional[Tensor] = None,  # 位置编码张量（可选）
        query_pos: Optional[Tensor] = None  # query的位置编码张量（可选）
    ):
        # 将位置编码添加到输出中
        q = k = self.with_pos_embed(output, query_pos)
        # 执行self-attention操作
        output2 = self.self_attn(q, k, value=output, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]
        # 将self-attention的输出与原始输出相加并归一化
        output = output + self.dropout1(output2)
        output = self.norm1(output)
        # 执行cross-attention操作
        output2 = self.multihead_attn(
            query=self.with_pos_embed(output, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        output2 = output2[0]
        # 将cross-attention的输出与原始输出相加并归一化
        output = output + self.dropout2(output2)
        output = self.norm2(output)
        # 执行前馈网络操作
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output))))
        # 将前馈网络的输出与原始输出相加并归一化
        output = output + self.dropout3(output2)
        output = self.norm3(output)
        return output
    
    # 该函数实现了Transformer编码器层的前向传播过程（pre-norm版本）
    def forward_pre(
        self,
        output,  # 输入的输出张量
        memory,  # 输入的记忆张量
        output_mask: Optional[Tensor] = None,  # 输出张量的mask张量（可选）
        memory_mask: Optional[Tensor] = None,  # 记忆张量的mask张量（可选）
        output_key_padding_mask: Optional[Tensor] = None,  # 输出张量的键padding mask张量（可选）
        memory_key_padding_mask: Optional[Tensor] = None,  # 记忆张量的键padding mask张量（可选）
        pos: Optional[Tensor] = None,  # 位置编码张量（可选）
        query_pos: Optional[Tensor] = None  # query的位置编码张量（可选）
    ):
        # 执行归一化操作
        output2 = self.norm1(output)
        # 将位置编码添加到输出中
        q = k = self.with_pos_embed(output2, query_pos)
        # 执行self-attention操作
        output2 = self.self_attn(q, k, value=output2, attn_mask=output_mask, key_padding_mask=output_key_padding_mask)
        output2 = output2[0]
        # 将self-attention的输出与原始输出相加并归一化
        output = output + self.dropout1(output2)
        # 执行归一化操作
        output2 = self.norm2(output)
        # 执行cross-attention操作
        output2 = self.multihead_attn(
            query=self.with_pos_embed(output2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        output2 = output2[0]
        # 将cross-attention的输出与原始输出相加并归一化
        output = output + self.dropout2(output2)
        # 执行归一化操作
        output2 = self.norm3(output)
        # 执行前馈网络操作
        output2 = self.linear2(self.dropout(self.activation(self.linear1(output2))))
        # 将前馈网络的输出与原始输出相加并返回
        output = output + self.dropout3(output2)
        return output
    
    # 该函数是forward_post和forward_pre的统一接口
    def forward(
        self,
        output,  # 输入的输出张量
        memory,  # 输入的记忆张量
        output_mask: Optional[Tensor] = None,  # 输出张量的mask张量（可选）
        memory_mask: Optional[Tensor] = None,  # 记忆张量的mask张量（可选）
        output_key_padding_mask: Optional[Tensor] = None,  # 输出张量的键padding mask张量（可选）
        memory_key_padding_mask: Optional[Tensor] = None,  # 记忆张量的键padding mask张量（可选）
        pos: Optional[Tensor] = None,  # 位置编码张量（可选）
        query_pos: Optional[Tensor] = None  # query的位置编码张量（可选）
    ):
        # 根据配置选择forward_post或forward_pre
        ...
    # 如果 normalize_before 为真，则执行 forward_pre() 方法进行前向传播
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
    # 否则执行 forward_post() 方法进行前向传播
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
    
    
    这段代码根据变量 `normalize_before` 的值来判断使用哪个前向传播方法。如果 `normalize_before` 为真，则调用 `forward_pre()` 方法；否则调用 `forward_post()` 方法。这段代码是一个条件语句，根据条件来选择执行不同的代码块。
class OneFormerTransformerDecoderQueryTransformer(nn.Module):
    # 定义一个名为OneFormerTransformerDecoderQueryTransformer的类，继承自nn.Module
    def __init__(
        self,
        # 初始化函数，接受一系列参数
        d_model=512,  # 模型维度，默认为512
        nhead=8,  # 注意力头的数量，默认为8
        num_decoder_layers=6,  # 解码器层数，默认为6
        dim_feedforward=2048,  # 前馈神经网络中间维度，默认为2048
        dropout=0.1,  # dropout概率，默认为0.1
        activation="relu",  # 激活函数类型，默认为ReLU
        normalize_before=False,  # 在层归一化之前是否进行归一化，默认为False
        return_intermediate_dec=False,  # 是否返回中间结果，默认为False
        layer_norm_eps=1e-05,  # 层归一化的eps值，默认为1e-05
    ):
        super().__init__()
        # 调用父类构造函数

        decoder_layer = OneFormerTransformerDecoderQueryTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before, layer_norm_eps
        )
        # 创建解码器层对象
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # 创建层归一化对象
        self.decoder = OneFormerTransformerDecoderQueryTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        # 创建解码器对象

        self.d_model = d_model  # 设置模型维度
        self.nhead = nhead  # 设置注意力头的数量

    def forward(self, src, mask, query_embed, pos_embed, task_token=None):
        # 定义前向传播函数，接受src、mask、query_embed、pos_embed和task_token参数
        batch_size = src.shape[0]  # 获取src的批量大小
        src = src.flatten(2).permute(2, 0, 1)  # 对src进行flatten操作并进行维度变换
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # 对pos_embed进行flatten操作并进行维度变换
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)  # 对query_embed进行维度扩展和复制
        if mask is not None:
            mask = mask.flatten(1)  # 如果mask不为None，则对mask进行flatten操作

        if task_token is None:
            queries = torch.zeros_like(query_embed)  # 如果task_token为None，则创建与query_embed相同大小的全零tensor
        else:
            queries = task_token.repeat(query_embed.shape[0], 1, 1)  # 否则，对task_token进行维度扩展和复制

        queries = self.decoder(queries, src, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        # 使用解码器处理queries、src以及相关的mask和位置编码
        return queries.transpose(1, 2)  # 返回结果进行维度置换


class OneFormerTransformerDecoder(nn.Module):
    """
    Transformer decoder
    """
    # 定义一个名为OneFormerTransformerDecoder的类，继承自nn.Module
    # 初始化函数，接收输入通道数和配置参数
    def __init__(self, in_channels: int, config: OneFormerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 将配置参数存储在self.config中
        self.config = config

        # 初始化一些成员变量
        self.dropout = config.dropout
        self.num_heads = config.num_attention_heads
        self.is_training = config.is_training
        self.use_task_norm = config.use_task_norm
        self.use_auxiliary_loss = config.use_auxiliary_loss

        # 初始化查询变换器
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

        # 初始化解码器标准化层
        self.decoder_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

        # 初始化特征级别数量
        self.num_feature_levels = 3

        # 初始化解码器层列表
        self.layers = nn.ModuleList(
            [OneFormerTransformerDecoderLayer(config) for _ in range(config.decoder_layers - 1)]
        )

        # 初始化查询输入投影层
        self.query_input_projection = nn.Conv2d(in_channels, config.hidden_dim, kernel_size=1)

        # 初始化类别嵌入层
        self.class_embed = nn.Linear(config.hidden_dim, config.num_labels + 1)
        
        # 初始化掩码嵌入层
        self.mask_embed = OneFormerMLPPredictionHead(
            config.hidden_dim,
            config.hidden_dim,
            config.mask_dim,
            3,
        )

    # 前向传播函数
    def forward(
        self,
        task_token=None, # 任务令牌
        multi_stage_features=None, # 多阶段特征
        multi_stage_positional_embeddings=None, # 多阶段位置嵌入
        mask_features=None, # 掩码特征
        query_features=None, # 查询特征
        query_embeddings=None, # 查询嵌入
        query_embedder=None, # 查询嵌入器
        size_list=None, # 大小列表
        output_attentions=None, # 输出注意力权重
        ):
        # 如果使用任务规范化，则对任务令牌进行解码
        if self.use_task_norm:
            task_token = self.decoder_norm(task_token)

        # 使用查询转换器对查询特征进行转换
        object_queries = self.query_transformer(
            query_features,
            None,
            query_embedder.weight[:-1],
            self.query_input_projection(mask_features),
            task_token if self.use_task_norm else None,
        )

        # 调整维度顺序
        object_queries = object_queries[0].permute(1, 0, 2)

        # 将对象查询和任务令牌拼接在一起
        queries = torch.cat([object_queries, task_token], dim=0)

        # 克隆查询
        output = queries.clone()

        intermediate_class_predictions = []
        intermediate_mask_predictions = []

        # 在可学习的查询特征上进行预测
        outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(
            output, mask_features, attention_mask_target_size=size_list[0]
        )
        intermediate_class_predictions.append(outputs_class)
        intermediate_mask_predictions.append(outputs_mask)

        attentions = ()

        # 遍历层并处理输出
        for index, layer in enumerate(self.layers):
            layer_outputs = layer(
                index=index,
                output=output,
                multi_stage_features=multi_stage_features,
                multi_stage_positional_embeddings=multi_stage_positional_embeddings,
                attention_mask=attention_mask,
                query_embeddings=query_embeddings,
                output_attentions=output_attentions,
            )

            output = layer_outputs[0]
            attentions += (layer_outputs[1:],)

            # 继续进行预测
            outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(
                output, mask_features, attention_mask_target_size=size_list[(index + 1) % self.num_feature_levels]
            )
            intermediate_class_predictions.append(outputs_class)
            intermediate_mask_predictions.append(outputs_mask)

        # 检查中间预测的数量是否与层数相同
        if not len(intermediate_mask_predictions) == len(self.layers) + 1:
            raise ValueError(
                "Intermediate predictions in the transformer decoder must have the same number of elements as number"
                " of layers"
            )

        # 调整维度顺序
        object_queries = layer_outputs[0].permute(1, 0, 2)

        # 调整维度顺序
        contrastive_logits = queries.permute(1, 0, 2)

        # 返回解码器的输出结果
        return OneFormerTransformerDecoderOutput(
            object_queries=object_queries,
            contrastive_logits=contrastive_logits,
            prediction_masks=intermediate_mask_predictions[-1],
            prediction_class=intermediate_class_predictions[-1],
            auxiliary_predictions=self._get_aux_predictions(
                intermediate_class_predictions, intermediate_mask_predictions
            )
            if self.use_auxiliary_loss
            else None,
            attentions=attentions,
        )
    # 前向传播函数，用于模型的前向预测
    def forward_prediction_heads(self, output, mask_features, attention_mask_target_size):
        # 对decoder的输出进行规范化
        decoder_output = self.decoder_norm(output)
        # 调换tensor的维度顺序
        decoder_output = decoder_output.transpose(0, 1)
        # 使用decoder输出进行分类预测
        outputs_class = self.class_embed(decoder_output)
        # 使用decoder输出进行mask预测
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # 对outputs_mask进行上采样，以匹配attention_mask_target_size的大小
        attention_mask = nn.functional.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        # 必须使用布尔类型
        # 如果提供了一个BoolTensor，则位置为“True”将不允许参与注意力，而“False”值将保持不变。
        attention_mask = (
            attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5
        ).bool()
        attention_mask = attention_mask.detach()

        # 返回分类结果、mask预测结果和注意力掩码
        return outputs_class, outputs_mask, attention_mask

    @torch.jit.unused
    # 获取辅助预测结果的函数，用于使torchscript正常工作
    def _get_aux_predictions(self, outputs_class, outputs_seg_masks):
        # 这是使torchscript正常工作的一个解决方案，因为torchscript不支持具有非同质值的字典，例如一个同时包含tensor和list的字典。
        aux_list = [
            {"class_queries_logits": a, "masks_queries_logits": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
        return tuple(aux_list)
class OneFormerTransformerModule(nn.Module):
    """
    The OneFormer's transformer module.
    """

    def __init__(self, in_features: int, config: OneFormerConfig):
        # 初始化函数，传入输入特征数量和配置信息
        super().__init__()
        hidden_dim = config.hidden_dim
        # 特征层级数量
        self.num_feature_levels = 3
        # 定义位置嵌入器
        self.position_embedder = OneFormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        # 定义查询嵌入器
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = []

        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_proj:
                # 如果输入特征数量不等于隐藏维度或需要强制输入投影，则进行卷积投影
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                # 否则为空对象
                self.input_projections.append(nn.Sequential())

        # 定义解码器
        self.decoder = OneFormerTransformerDecoder(in_channels=in_features, config=config)
        # 定义层级嵌入器
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(
        self,
        multi_scale_features: List[Tensor],
        mask_features: Tensor,
        task_token: Tensor,
        output_attentions: bool = False,
    ) -> OneFormerTransformerDecoderOutput:
        if not len(multi_scale_features) == self.num_feature_levels:
            # 检查多尺度特征数量是否与特征层级数量相匹配
            raise ValueError(
                f"Number of elements in multi_scale_features ({len(multi_scale_features)}) and num_feature_levels"
                f" ({self.num_feature_levels}) do not match!"
            )
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            # 生成多阶段位置嵌入
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            # 生成多阶段特征
            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(2)
                + self.level_embed.weight[i][None, :, None]
            )

            # 将 NxCxHxW 扁平化为 HWxNxC
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)

        _, batch_size, _ = multi_stage_features[0].shape

        # QxNxC
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        task_token = task_token.unsqueeze(0)

        query_features = self.position_embedder(mask_features, None)

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
        )
# 从 Transformers 库的 MaskFormerSinePositionEmbedding 类复制而来，将 Mask 替换为 One
class OneFormerSinePositionEmbedding(nn.Module):
    """
   这是一个更标准的位置编码版本，非常类似于 Attention is all you need 论文中使用的版本，并泛化为适用于图像。
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = (~mask).to(x.dtype)
        # 沿着特定维度计算 not_mask 的累积和
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            # 根据规范化标志对嵌入进行规范化
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 使用正弦和余弦函数来转换位置编码
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 拼接得到最终位置编码
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# 从 Transformers 库的 PredictionBlock 类复制而来
class PredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        # 使用线性层和激活函数作为网络层
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # 将子模块索引保持为 Sequential 块的一部分
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        # 依次经过网络层进行前向传播
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class OneFormerTextMapperAttention(nn.Module):
    # 初始化函数，设置注意力机制的参数
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # 设置缩放因子为 head_dim 的倒数，如果 qk_scale 未指定则使用默认值
        self.scale = qk_scale or head_dim**-0.5

        # 创建查询、键、值的线性投影层
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # 初始化注意力下降层和投影下降层
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    # 前向传播函数
    def forward(self, q, k, v):
        # 获取输入张量的维度信息
        batch_size, q_sequence_length, num_channels = q.shape
        # 如果键和值的形状不同，抛出异常
        if not k.shape == v.shape:
            raise ValueError(f"keys ({list(k.shape)}) and values ({list(v.shape)}) have different shapes!")
        # 获取输入张量的维度信息
        batch_size, k_sequence_length, num_channels = k.shape
        # 经过查询、键、值的线性投影，将结果重塑为“头”的形式
        q = self.q_proj(q).reshape(batch_size, q_sequence_length, self.num_heads, num_channels // self.num_heads)
        k = self.k_proj(k).reshape(batch_size, k_sequence_length, self.num_heads, num_channels // self.num_heads)
        v = self.v_proj(v).reshape(batch_size, k_sequence_length, self.num_heads, num_channels // self.num_heads)

        # 使用 einsum 计算注意力矩阵
        attn = torch.einsum("bnkc,bmkc->bknm", q, k) * self.scale

        # 对注意力矩阵进行 softmax 操作
        attn = attn.softmax(dim=-1)

        # 根据注意力矩阵计算输出
        output = torch.einsum("bknm,bmkc->bnkc", attn, v).reshape(batch_size, q_sequence_length, num_channels)

        # 通过投影层处理输出
        output = self.proj(output)
        output = self.proj_drop(output)
        return output
class OneFormerTextTransformerDecoderLayer(nn.Module):
    # 定义一个Transformer解码器层
    def __init__(
        # 初始化函数，接受参数d_model, nhead, dropout, layer_norm_eps
        self,
        d_model,
        nhead,
        dropout=0.1,
        layer_norm_eps=1e-05,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 定义自注意力层及跨注意力层
        self.self_attn = OneFormerTextMapperAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = OneFormerTextMapperAttention(d_model, nhead, proj_drop=dropout)

        # 定义LayerNorm层
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # 定义Dropout层
        self.dropout = nn.Dropout(dropout)

        # 定义MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, hidden_state, mem):
        # 向前传播函数，接受hidden_state和mem作为输入
        q = k = v = self.norm1(hidden_state)
        hidden_state = hidden_state + self.self_attn(q, k, v)
        q = self.norm2(hidden_state)
        hidden_state = hidden_state + self.cross_attn(q, mem, mem)
        hidden_state = hidden_state + self.dropout(self.mlp(self.norm3(hidden_state)))
        return hidden_state


class OneFormerTextContextDecoder(nn.Module):
    # 定义一个文本上下文解码器
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
        # 初始化函数，接受一系列参数
        super().__init__()

        # 定义内存映射层和文本映射层
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim, eps=layer_norm_eps),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width, eps=layer_norm_eps),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim, eps=layer_norm_eps),
            nn.Linear(visual_dim, transformer_width),
        )

        # 定义解码器层的列表
        self.decoder = nn.ModuleList(
            [
                OneFormerTextTransformerDecoderLayer(transformer_width, transformer_heads, dropout, layer_norm_eps)
                for _ in range(transformer_layers)
            ]
        )

        # 定义输出映射层
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width, eps=layer_norm_eps), nn.Linear(transformer_width, visual_dim)
        )

    def forward(self, text, visual):
        # 前向传播函数，接受文本和视觉特征作为输入
        visual = self.memory_proj(visual)
        hidden_state = self.text_proj(text)

        for layer in self.decoder:
            hidden_state = layer(hidden_state, visual)

        return self.out_proj(hidden_state)


class OneFormerTextMLP(nn.Module):
    # 定义一个MLP模型
    def __init__(
        self,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        output_size: Optional[int] = None,
    ):
        # 调用父类构造函数初始化对象
        super().__init__()
        # 设置激活函数为预定义的 "quick_gelu"
        self.activation_fn = ACT2FN["quick_gelu"]
        # 设置隐藏层大小
        hidden_size = hidden_size
        # 设置中间层大小
        intermediate_size = intermediate_size
        # 设置输出层大小
        output_size = output_size
        # 定义第一个全连接层，连接输入和中间层
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        # 定义第二个全连接层，连接中间层和输出
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用第一个全连接层进行前向计算
        hidden_states = self.fc1(hidden_states)
        # 使用预定义的激活函数进行激活
        hidden_states = self.activation_fn(hidden_states)
        # 使用第二个全连接层进行前向计算，得到输出
        hidden_states = self.fc2(hidden_states)
        # 返回最终的输出张量
        return hidden_states
class OneFormerTextTransformerLayer(nn.Module):
    def __init__(self, width: int, heads: int, attn_mask: torch.Tensor, layer_norm_eps=1e-05):
        super().__init__()
        # 定义一个多头注意力层
        self.self_attn = nn.MultiheadAttention(width, heads)
        # 定义第一个层归一化层
        self.layer_norm1 = nn.LayerNorm(width, eps=layer_norm_eps)
        # 定义一个MLP（多层感知机）用于对隐藏状态进行转换
        self.mlp = OneFormerTextMLP(width, width * 4, width)
        # 定义第二个层归一化层
        self.layer_norm2 = nn.LayerNorm(width, eps=layer_norm_eps)
        # 存储注意力掩码，以便在forward中使用
        self.attn_mask = attn_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        # 应用第一个层归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 使用多头自注意力机制计算注意力，得到新的隐藏状态
        hidden_states = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )[0]
        # 添加残差连接
        hidden_states = residual + hidden_states

        residual = hidden_states
        # 应用第二个层归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 应用MLP进行隐藏状态的转换
        hidden_states = self.mlp(hidden_states)
        # 添加残差连接
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
        # 创建多个OneFormerTextTransformerLayer层组成的序列
        self.layers = nn.Sequential(
            *[OneFormerTextTransformerLayer(width, heads, attn_mask, layer_norm_eps) for _ in range(layers)]
        )
        # 是否使用梯度检查点
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states: torch.Tensor):
        # 遍历所有层，应用变换
        for layer in self.layers:
            if self.use_checkpoint:
                # 使用梯度检查点进行前向传播
                hidden_states = self._gradient_checkpointing_func(layer, hidden_states)
            else:
                # 正常进行前向传播
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
        # 计算头数
        heads = width // 64
        self.context_length = context_length
        self.width = width
        # 创建一个OneFormerTextTransformer对象
        self.transformer = OneFormerTextTransformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(),  # 构建注意力掩码
            use_checkpoint=use_checkpoint,
            layer_norm_eps=layer_norm_eps,
        )

        # 定义位置嵌入参数
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        # 最终层归一化
        self.ln_final = nn.LayerNorm(width, eps=layer_norm_eps)
        # 定义词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, width)
    # 构建注意力掩码
    def build_attention_mask(self):
        # 延迟创建因果注意力掩码，在视觉令牌之间完全注意
        # PyTorch 使用加性注意力掩码; 用 -inf 填充
        mask = torch.empty(self.context_length, self.context_length)
        # 用 -inf 填充张量
        mask.fill_(float("-inf"))
        # 执行上三角矩阵化操作, 使下三角部分元素为 0
        mask.triu_(1)
        # 返回注意力掩码
        return mask
    
    # 前向传播
    def forward(self, text):
        # 通过词嵌入层得到隐藏状态
        hidden_state = self.token_embedding(text)
        # 将隐藏状态加上位置编码
        hidden_state = hidden_state + self.positional_embedding
        # 调整张量维度顺序
        hidden_state = hidden_state.permute(1, 0, 2)
        # 通过 Transformer 编码器
        hidden_state = self.transformer(hidden_state)
        # 调整回原来的维度顺序
        hidden_state = hidden_state.permute(1, 0, 2)
        # 通过最终 LayerNorm 层
        hidden_state = self.ln_final(hidden_state)
        # 根据输入 text 的 argmax 索引, 获取最终隐藏状态
        hidden_state = hidden_state[torch.arange(hidden_state.shape[0]), text.argmax(dim=-1)]
        
        # 返回最终隐藏状态
        return hidden_state
class OneFormerTextMapper(nn.Module):
    def __init__(self, config: OneFormerConfig):
        # 初始化函数，接收一个配置对象作为参数
        super().__init__()
        # 实例化一个 OneFormerTextEncoder 对象，用于文本编码
        self.text_encoder = OneFormerTextEncoder(
            context_length=config.text_encoder_context_length,
            width=config.text_encoder_width,
            layers=config.text_encoder_num_layers,
            vocab_size=config.text_encoder_vocab_size,
            layer_norm_eps=config.layer_norm_eps,
        )

        # 实例化一个 OneFormerMLPPredictionHead 对象，用于文本投影
        self.text_projector = OneFormerMLPPredictionHead(
            config.text_encoder_width,
            config.hidden_dim,
            config.hidden_dim,
            config.text_encoder_proj_layers,
        )
        # 如果配置中定义了上下文长度，则实例化一个嵌入层用于处理上下文信息
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
        # 对输入文本进行编码并返回编码后的结果
        text_queries = self.encode_text(inputs)

        return text_queries

    def encode_text(self, text):
        # 检查输入文本的维度
        if text.ndim is None:
            raise ValueError("text must not be NoneType")
        if text.ndim not in [2, 3]:
            raise ValueError("Number of dimensions in text must be 2 or 3")
        squeeze_dim = False
        num_text = 1
        # 如果输入文本维度为3，则需要进行处理
        if text.ndim == 3:
            num_text = text.shape[1]
            batch_size, num_text, hidden_dim = text.shape
            # 将文本维度重新整形为二维
            text = text.reshape(batch_size * num_text, hidden_dim)
            squeeze_dim = True

        # 对文本进行编码
        encoded_text = self.text_encoder(text)

        # 对编码后的文本进行投影
        text_queries = self.text_projector(encoded_text)

        # 如果之前进行了维度压缩，则需要还原维度
        if squeeze_dim:
            _, hidden_dim = text_queries.shape
            text_queries = text_queries.reshape(batch_size, num_text, hidden_dim)
            # 如果存在上下文信息，则将其添加到文本查询结果中
            if self.prompt_ctx is not None:
                text_queries_ctx = self.prompt_ctx.weight.unsqueeze(0).repeat(text_queries.shape[0], 1, 1)
                text_queries = torch.cat([text_queries, text_queries_ctx], dim=1)

        return text_queries


class OneFormerTaskModel(nn.Module):
    def __init__(self, config: OneFormerConfig):
        # 初始化函数，接收一个配置对象作为参数
        super().__init__()
        # 实例化一个 OneFormerMLPPredictionHead 对象，用于任务模型的预测
        self.task_mlp = OneFormerMLPPredictionHead(
            config.task_seq_len,
            config.hidden_dim,
            config.hidden_dim,
            2,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # 对输入进行任务模型的预测
        task_tokens = self.task_mlp(inputs)
        return task_tokens


ONEFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.


"""
    # 参数:
    #     config ([`OneFormerConfig`]): 包含模型所有参数的模型配置类。用一个配置文件初始化不会加载与模型关联的权重，只加载配置信息。
    #         可以查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
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
    config_class = OneFormerConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"  # 设置主要输入名称为"pixel_values"


@add_start_docstrings(    # 添加 Model 的文档字符串
    "The bare OneFormer Model outputting raw hidden-states without any specific head on top.",
    ONEFORMER_START_DOCSTRING,
)
class OneFormerModel(OneFormerPreTrainedModel):
    main_input_name = ["pixel_values", "task_inputs"]  # 设置主要输入名称为["pixel_values", "task_inputs"]

    def __init__(self, config: OneFormerConfig):
        super().__init__(config)
        self.pixel_level_module = OneFormerPixelLevelModule(config)  # 创建像素级模块
        self.transformer_module = OneFormerTransformerModule(in_features=config.conv_dim, config=config)  # 创建 Transformer 模块
        self.task_encoder = OneFormerTaskModel(config)  # 创建任务编码器
        self.is_training = config.is_training  # 设置是否训练状态

        if self.is_training:
            self.text_mapper = OneFormerTextMapper(config)  # 如果是训练状态，创建文本映射器
        else:
            self.text_mapper = None

        self.post_init()  # 调用后续初始化函数

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
@add_start_docstrings(    # 添加 Model 的文档字符串
    "OneFormer Model for instance, semantic and panoptic image segmentation.",
    ONEFORMER_START_DOCSTRING,
)
# 定义一个继承自 OneFormerPreTrainedModel 的类，用于通用分割任务
class OneFormerForUniversalSegmentation(OneFormerPreTrainedModel):
    # 定义主要输入的名称
    main_input_name = ["pixel_values", "task_inputs"]

    # 初始化方法，接收一个 OneFormerConfig 类型的参数
    def __init__(self, config: OneFormerConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 OneFormerModel 对象
        self.model = OneFormerModel(config)

        # 创建一个 OneFormerHungarianMatcher 对象，使用传入的配置参数对其进行初始化
        self.matcher = OneFormerHungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=config.train_num_points,
        )

        # 创建一个权重字典，键为损失名称，值为配置参数中对应的权重
        self.weight_dict: Dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
            "loss_contrastive": config.contrastive_weight,
        }

        # 创建一个 OneFormerLoss 对象，使用传入的配置参数对其进行初始化
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

        # 调用额外的初始化方法
        self.post_init()

    # 定义一个方法，用于计算损失的字典
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
        # 调用 OneFormerLoss 对象的损失计算方法，得到损失字典
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

        # 根据权重字典对每个损失进行加权
        for key, weight in self.weight_dict.items():
            for loss_key, loss in loss_dict.items():
                if key in loss_key:
                    loss *= weight

        return loss_dict

    # 定义一个方法，用于计算总损失
    def get_loss(self, loss_dict: Dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    # 为模型正向传播方法添加注释
    @add_start_docstrings_to_model_forward(ONEFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=OneFormerForUniversalSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    # 定义前向传播函数
    def forward(
        # 图像数据输入
        self,
        pixel_values: Tensor,
        # 任务相关的输入数据
        task_inputs: Tensor,
        # 文本输入数据，可选
        text_inputs: Optional[Tensor] = None,
        # 掩码标签数据，可选
        mask_labels: Optional[List[Tensor]] = None,
        # 类别标签数据，可选
        class_labels: Optional[List[Tensor]] = None,
        # 像素掩码数据，可选
        pixel_mask: Optional[Tensor] = None,
        # 是否输出辅助逻辑输出，可选
        output_auxiliary_logits: Optional[bool] = None,
        # 是否输出隐藏状态，可选
        output_hidden_states: Optional[bool] = None,
        # 是否输出注意力权重，可选
        output_attentions: Optional[bool] = None,
        # 是否返回字典格式的输出，可选
        return_dict: Optional[bool] = None,
    ):
```