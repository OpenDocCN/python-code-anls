# `.\yolov8\ultralytics\models\utils\loss.py`

```py
# Ultralytics YOLO , AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss  # 导入 FocalLoss 和 VarifocalLoss 损失函数
from ultralytics.utils.metrics import bbox_iou  # 导入 bbox_iou 函数

from .ops import HungarianMatcher  # 导入匈牙利匹配器

class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class. This class calculates and returns the different loss components for the
    DETR object detection model. It computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary
    losses.

    Attributes:
        nc (int): The number of classes.
        loss_gain (dict): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Use FocalLoss or not.
        use_vfl (bool): Use VarifocalLoss or not.
        use_uni_match (bool): Whether to use a fixed layer to assign labels for the auxiliary branch.
        uni_match_ind (int): The fixed indices of a layer to use if `use_uni_match` is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss or None): Focal Loss object if `use_fl` is True, otherwise None.
        vfl (VarifocalLoss or None): Varifocal Loss object if `use_vfl` is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=False, use_uni_match=False, uni_match_ind=0
    ):
        """
        DETR loss function.

        Args:
            nc (int): The number of classes.
            loss_gain (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_vfl (bool): Use VarifocalLoss or not.
            use_uni_match (bool): Whether to use a fixed layer to assign labels for auxiliary branch.
            uni_match_ind (int): The fixed indices of a layer.
        """
        super().__init__()

        # 如果 loss_gain 为 None，则使用默认的损失系数
        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        
        # 设置类别数目 nc，初始化匈牙利匹配器 matcher
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        
        # 设置损失系数 loss_gain，是否计算辅助损失 aux_loss，以及是否使用 FocalLoss 和 VarifocalLoss
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

        # 是否使用固定层来分配辅助分支的标签，以及固定层的索引
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None
    # 计算分类损失，基于预测值、目标值和实际得分
    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        """Computes the classification loss based on predictions, target values, and ground truth scores."""
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        
        # 创建一个全零的张量，形状为(bs, nq, self.nc + 1)，类型为int64，存储在与targets相同的设备上
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        # 使用scatter_函数将one_hot张量中的指定位置设为1，形成one-hot编码，排除最后一个类别
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        # 将每个类别的得分乘以对应的one-hot编码，得到每个样本每个查询点的分类得分
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        # 计算分类损失
        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            # 使用二分类交叉熵损失函数计算损失，mean(1)对每个查询点求均值，sum()对所有查询点求和
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        # 返回损失值，乘以分类损失增益系数
        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    # 计算边界框损失，包括预测边界框和实际边界框的L1损失和GIoU损失
    def _get_loss_bbox(self, pred_bboxes, gt_bboxes, postfix=""):
        """Calculates and returns the bounding box loss and GIoU loss for the predicted and ground truth bounding
        boxes.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            # 如果没有实际边界框，损失为0
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        # 计算边界框损失
        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        # 计算GIoU损失
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]

        # 返回损失字典，将值展平为标量
        return {k: v.squeeze() for k, v in loss.items()}
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        match_indices=None,
        postfix="",
        masks=None,
        gt_mask=None,
    ):
        """Get auxiliary losses."""
        # NOTE: loss class, bbox, giou, mask, dice
        # Initialize a tensor to hold different types of losses
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        
        # If match_indices is not provided and using uni_match, compute match_indices using matcher
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        
        # Iterate over predicted boxes and scores to compute losses
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            # Compute losses using _get_loss function
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            # Accumulate class, bbox, and giou losses
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            
            # Uncomment below section if handling mask and dice losses
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        # Construct a dictionary with computed losses
        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        
        # Uncomment below section if handling mask and dice losses
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]

        # Return the dictionary of computed losses
        return loss

    @staticmethod
    def _get_index(match_indices):
        """Returns batch indices, source indices, and destination indices from provided match indices."""
        # 生成一个批次索引，源索引和目标索引，从匹配索引中提取
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(self, pred_bboxes, gt_bboxes, match_indices):
        """Assigns predicted bounding boxes to ground truth bounding boxes based on the match indices."""
        # 根据匹配索引将预测的边界框分配给真实边界框
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes,
        pred_scores,
        gt_bboxes,
        gt_cls,
        gt_groups,
        masks=None,
        gt_mask=None,
        postfix="",
        match_indices=None,
    ):
        """Get losses."""
        # 如果没有提供匹配索引，则调用self.matcher计算匹配索引
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        # 调用_get_index函数获取索引和对应的真实边界框索引
        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        # 创建一个全是self.nc的张量，作为分类目标
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        # 如果gt_bboxes非空，则计算预测边界框与真实边界框的IoU作为得分
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        loss = {}
        # 调用_get_loss_class和_get_loss_bbox计算分类损失和边界框损失
        loss.update(self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix))
        # 如果masks和gt_mask都不为None，则调用_get_loss_mask计算掩码损失
        # if masks is not None and gt_mask is not None:
        #     loss.update(self._get_loss_mask(masks, gt_mask, match_indices, postfix))
        return loss
    def forward(self, pred_bboxes, pred_scores, batch, postfix="", **kwargs):
        """
        Args:
            pred_bboxes (torch.Tensor): [l, b, query, 4]
            pred_scores (torch.Tensor): [l, b, query, num_classes]
            batch (dict): A dict includes:
                gt_cls (torch.Tensor) with shape [num_gts, ],
                gt_bboxes (torch.Tensor): [num_gts, 4],
                gt_groups (List(int)): a list of batch size length includes the number of gts of each image.
            postfix (str): postfix of loss name.
        """
        # 设定当前设备为预测边界框的设备
        self.device = pred_bboxes.device
        # 获取匹配索引，如果未提供则为 None
        match_indices = kwargs.get("match_indices", None)
        # 从批次中获取真实类别、边界框和分组信息
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        # 计算总损失，传入最后一个预测结果的边界框和分数，真实边界框、类别、分组信息及后缀
        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        # 如果有辅助损失，则添加到总损失中
        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        # 返回计算得到的总损失
        return total_loss
    # 定义了一个 RT-DETR 检测损失类，继承自 DETRLoss 类
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    # 前向传播函数，用于计算检测损失
    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        # 解析预测的边界框和分数
        pred_bboxes, pred_scores = preds
        # 计算标准检测损失
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # 检查是否提供了去噪元数据以计算去噪训练损失
        if dn_meta is not None:
            # 提取去噪正样本索引和组数
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            # 断言批数据中的组数量与去噪正样本索引长度相等
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # 获取用于去噪的匹配索引
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # 计算去噪训练损失
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # 如果没有提供去噪元数据，则将去噪损失设置为零
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        # 返回总损失字典
        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        # 初始化一个空列表，用于存储匹配的索引
        dn_match_indices = []
        
        # 计算每个图像的累积索引组
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        
        # 遍历每个图像及其对应的ground truth数目
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                # 生成包含所有ground truth索引的张量
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                # 将ground truth索引重复dn_num_group次，以匹配denoising组的数目
                gt_idx = gt_idx.repeat(dn_num_group)
                # 断言：确保dn_pos_idx[i]和gt_idx长度相同
                assert len(dn_pos_idx[i]) == len(gt_idx), f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                # 将匹配的索引对加入到dn_match_indices列表中
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                # 如果ground truth数目为0，则创建一个空的张量对
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        
        # 返回包含所有匹配索引对的列表
        return dn_match_indices
```