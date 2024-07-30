# `.\yolov8\ultralytics\utils\loss.py`

```py
# 导入PyTorch库中需要的模块
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从Ultralytics工具包中导入一些特定的功能
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

# 从当前目录下的.metrics文件中导入bbox_iou和probiou函数
from .metrics import bbox_iou, probiou
# 从当前目录下的.tal文件中导入bbox2dist函数
from .tal import bbox2dist

# 定义一个名为VarifocalLoss的PyTorch模块，继承自nn.Module
class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        # 计算权重
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        # 禁用自动混合精度计算
        with autocast(enabled=False):
            # 计算损失
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)  # 沿着维度1取均值
                .sum()  # 求和
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        # 计算二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # 计算概率
        pred_prob = pred.sigmoid()  # logits转为概率
        # 计算p_t值
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        # 计算调节因子
        modulating_factor = (1.0 - p_t) ** gamma
        # 应用调节因子到损失上
        loss *= modulating_factor
        # 如果alpha大于0，则应用alpha因子
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max
    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        # 将目标张量限制在 [0, self.reg_max - 1 - 0.01] 的范围内
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        # 将目标张量转换为长整型（整数）
        tl = target.long()  # target left
        # 计算目标张量的右侧邻近值
        tr = tl + 1  # target right
        # 计算左侧权重
        wl = tr - target  # weight left
        # 计算右侧权重
        wr = 1 - wl  # weight right
        # 计算左侧损失（使用交叉熵损失函数）
        left_loss = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
        # 计算右侧损失（使用交叉熵损失函数）
        right_loss = F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        # 返回左侧损失和右侧损失的平均值（在最后一个维度上求均值，保持维度）
        return (left_loss + right_loss).mean(-1, keepdim=True)
# 定义了一个用于计算边界框损失的模块
class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        # 如果 reg_max 大于 1，则创建一个 DFLoss 对象，否则设为 None
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Compute IoU loss."""
        # 计算前景掩码中目标得分的总和，并扩展维度
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # 计算预测边界框和目标边界框之间的 IoU（Intersection over Union）
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # 计算 IoU 损失
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 计算 DFL loss
        if self.dfl_loss:
            # 将锚点和目标边界框转换成距离形式
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            # 计算 DFL 损失
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            # 如果没有 DFL loss，则设为 0
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


# 继承自 BboxLoss 类，用于处理旋转边界框损失
class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Compute IoU loss for rotated bounding boxes."""
        # 计算前景掩码中目标得分的总和，并扩展维度
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # 计算预测边界框和目标边界框之间的概率 IoU
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        # 计算 IoU 损失
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 计算 DFL loss
        if self.dfl_loss:
            # 将锚点和目标边界框转换成距离形式
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            # 计算 DFL 损失
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            # 如果没有 DFL loss，则设为 0
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


# 用于计算关键点损失的模块
class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        # 初始化关键点损失类，接收 sigmas 参数
        self.sigmas = sigmas
    # 定义一个方法，用于计算预测关键点和真实关键点之间的损失因子和欧氏距离损失。
    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        # 计算预测关键点与真实关键点在 x 和 y 方向上的平方差，得到欧氏距离的平方
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        # 计算关键点损失因子，用于调整不同关键点的重要性，避免稀疏区域对损失的过度影响
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # 计算欧氏距离损失，根据预设的尺度参数 self.sigmas 和区域面积 area 进行调整
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # 来自于 cocoeval 的公式
        # 返回加权后的关键点损失的均值，其中损失通过 kpt_mask 进行加权，确保只考虑有效关键点
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()
class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        # 获取模型的设备信息
        device = next(model.parameters()).device  # get model device
        # 从模型中获取超参数
        h = model.args  # hyperparameters

        # 获取最后一个模型组件，通常是 Detect() 模块
        m = model.model[-1]  # Detect() module
        # 使用 nn.BCEWithLogitsLoss 计算 BCE 损失，设置为不进行归约
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # 保存超参数
        self.hyp = h
        # 获取模型的步长信息
        self.stride = m.stride  # model strides
        # 获取模型的类别数
        self.nc = m.nc  # number of classes
        # 设置输出通道数，包括类别和回归目标
        self.no = m.nc + m.reg_max * 4
        # 获取模型的最大回归目标数量
        self.reg_max = m.reg_max
        # 保存模型的设备信息
        self.device = device

        # 判断是否使用 DFL（Distribution-based Focal Loss）
        self.use_dfl = m.reg_max > 1

        # 初始化任务对齐分配器，用于匹配目标框和锚点框
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        # 初始化边界框损失函数，使用指定数量的回归目标
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        # 创建一个张量，用于后续的数学运算，位于指定的设备上
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        # 获取目标张量的维度信息
        nl, ne = targets.shape
        # 如果目标张量为空，则返回一个零张量
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            # 获取图像索引和其对应的计数
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            # 创建零张量，用于存储预处理后的目标数据
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                # 获取与当前批次图像索引匹配的目标
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # 对输出的边界框坐标进行缩放和转换
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        # 如果使用 DFL，则对预测分布进行处理
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # 对预测分布进行解码，使用预定义的投影张量
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # 另外两种可能的解码方式，根据实际需求选择
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        # 返回解码后的边界框坐标
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # 初始化一个全零的张量，用于存储损失值，包括box、cls和dfl
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        
        # 如果preds是元组，则取第二个元素作为feats，否则直接使用preds
        feats = preds[1] if isinstance(preds, tuple) else preds
        
        # 将feats中的特征拼接并分割，得到预测的分布(pred_distri)和分数(pred_scores)
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # 调整张量维度，使得pred_scores和pred_distri的维度更适合后续计算
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # 获取pred_scores的数据类型和batch size
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        
        # 计算图像尺寸，以张量形式存储在imgsz中，单位是像素
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        
        # 使用make_anchors函数生成锚点(anchor_points)和步长张量(stride_tensor)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 处理目标数据，包括图像索引、类别和边界框，转换为Tensor并传输到设备上
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        
        # 将目标数据拆分为类别标签(gt_labels)和边界框(gt_bboxes)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        
        # 生成用于过滤的掩码(mask_gt)，判断边界框是否有效
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # 解码预测的边界框(pred_bboxes)，得到真实坐标
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # 使用分配器(assigner)计算匹配的目标边界框和分数
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # 计算目标分数之和，用于损失计算的归一化
        target_scores_sum = max(target_scores.sum(), 1)

        # 类别损失计算，使用二元交叉熵损失(BCE)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # 如果有前景掩码(fg_mask)存在，则计算边界框损失和分布损失
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # 损失值乘以超参数中的各自增益系数
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # 返回损失值的总和乘以batch size，以及分离的损失张量
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        # Compute predicted masks using prototype masks and coefficients
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, H, W) -> (n, H, W)
        # Compute binary cross entropy loss between predicted masks and ground truth masks
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        # Crop the loss using bounding boxes, then compute mean per instance and sum across instances
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape  # 获取原型掩模的高度和宽度

        loss = 0  # 初始化损失值为0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]  # 将目标边界框归一化到0-1范围

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)  # 计算目标边界框的面积

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)  # 将边界框归一化到掩模大小

        # 遍历每个样本中的前景掩模、目标索引、预测掩模、原型、归一化边界框、目标面积、掩模
        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i

            if fg_mask_i.any():  # 如果前景掩模中有任何True值
                mask_idx = target_gt_idx_i[fg_mask_i]  # 获取前景掩模对应的目标索引
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)  # 如果存在重叠，则获取真实掩模
                    gt_mask = gt_mask.float()  # 转换为浮点数
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]  # 否则直接获取真实掩模

                # 计算单个掩模损失
                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                # 防止在多GPU分布式数据并行处理中出现未使用梯度的错误
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # 将inf和相加可能导致nan损失

        # 返回平均每个前景掩模的损失
        return loss / fg_mask.sum()
class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        # Check if the model deals with pose keypoints (17 keypoints with 3 coordinates each)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        # Set sigmas for keypoint loss calculation based on whether it's pose or not
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        # Scale keypoints coordinates
        y[..., :2] *= 2.0
        # Translate keypoints to their anchor points
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
):
        """Calculate the keypoints loss."""
        # Implementation of keypoints loss calculation goes here
        pass  # Placeholder for actual implementation

class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            # If no targets, return zero tensor
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            # Initialize output tensor with proper dimensions
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    # Extract and scale bounding boxes, then concatenate with class labels
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out
    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        # 如果使用 DFL（Dynamic Feature Learning），对预测的距离进行处理
        if self.use_dfl:
            # 获取批量大小、锚点数、通道数
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # 重新调整预测距离的形状并进行 softmax 处理，然后乘以投影矩阵
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # 将解码后的旋转边界框坐标和预测的角度拼接在一起并返回
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)
# 定义一个名为 E2EDetectLoss 的类，用于计算训练损失
class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """初始化 E2EDetectLoss 类，使用提供的模型初始化一个一对多和一个对一检测损失。"""
        self.one2many = v8DetectionLoss(model, tal_topk=10)  # 初始化一对多检测损失对象
        self.one2one = v8DetectionLoss(model, tal_topk=1)    # 初始化一对一检测损失对象

    def __call__(self, preds, batch):
        """调用实例时计算框、类别和深度特征点损失的总和，乘以批次大小。"""
        preds = preds[1] if isinstance(preds, tuple) else preds  # 如果 preds 是元组，则使用第二个元素
        one2many = preds["one2many"]  # 获取预测结果中的一对多损失
        loss_one2many = self.one2many(one2many, batch)  # 计算一对多损失
        one2one = preds["one2one"]  # 获取预测结果中的一对一损失
        loss_one2one = self.one2one(one2one, batch)  # 计算一对一损失
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
        # 返回两个损失的总和，分别对应框和类别损失，以及深度特征点损失
```