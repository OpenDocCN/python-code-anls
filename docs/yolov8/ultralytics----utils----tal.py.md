# `.\yolov8\ultralytics\utils\tal.py`

```py
# 导入 PyTorch 库中的相关模块
import torch
import torch.nn as nn

# 从本地模块中导入必要的函数和类
from .checks import check_version
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy

# 检查当前使用的 PyTorch 版本是否符合最低要求
TORCH_1_10 = check_version(torch.__version__, "1.10.0")

# 定义一个任务对齐分配器的类，用于目标检测
class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        # 调用父类构造函数初始化模块
        super().__init__()
        # 设置对象属性，用于指定任务对齐分配器的超参数
        self.topk = topk  # 设置前k个候选框的数量
        self.num_classes = num_classes  # 设置目标类别的数量
        self.bg_idx = num_classes  # 设置背景类别的索引，默认为num_classes
        self.alpha = alpha  # 设置任务对齐度量中分类组件的参数alpha
        self.beta = beta  # 设置任务对齐度量中定位组件的参数beta
        self.eps = eps  # 设置一个极小值，用于避免除以零的情况

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment. Reference code is available at
        https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py.

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
                预测得分张量，形状为(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
                预测边界框张量，形状为(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
                锚点坐标张量，形状为(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
                真实标签张量，形状为(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
                真实边界框张量，形状为(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
                真实边界框掩码张量，形状为(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
                目标标签张量，形状为(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
                目标边界框张量，形状为(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
                目标得分张量，形状为(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
                前景掩码张量，形状为(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
                目标真实边界框索引张量，形状为(bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]  # 记录批次大小
        self.n_max_boxes = gt_bboxes.shape[1]  # 记录每个样本最大边界框数

        if self.n_max_boxes == 0:  # 如果没有真实边界框
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),  # 返回背景类索引
                torch.zeros_like(pd_bboxes).to(device),  # 返回零张量形状与预测边界框一致
                torch.zeros_like(pd_scores).to(device),  # 返回零张量形状与预测得分一致
                torch.zeros_like(pd_scores[..., 0]).to(device),  # 返回零张量形状与预测得分一致
                torch.zeros_like(pd_scores[..., 0]).to(device),  # 返回零张量形状与预测得分一致
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )  # 获取正样本掩码、对齐度量、重叠度量

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        # 选择最高重叠度的真实边界框索引、前景掩码、正样本掩码

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        # 获取分配的目标标签、目标边界框、目标得分

        # Normalize
        align_metric *= mask_pos  # 对齐度量乘以正样本掩码
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # 计算每个样本的最大对齐度量
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # 计算每个样本的最大重叠度
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        # 计算归一化后的对齐度量
        target_scores = target_scores * norm_align_metric  # 目标得分乘以归一化后的对齐度量

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        # Select candidates within ground truth bounding boxes
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        
        # Compute alignment metric and overlaps between predicted and ground truth boxes
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        
        # Select top-k candidates based on alignment metric
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        
        # Merge masks to get the final positive mask
        mask_pos = mask_topk * mask_in_gts * mask_gt
        
        # Return the final positive mask, alignment metric, and overlaps
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        
        # Initialize tensors for overlaps and bbox scores
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        
        # Create indices tensor for accessing scores based on ground truth labels
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        
        # Assign predicted scores to corresponding locations in bbox_scores
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w
        
        # Extract predicted and ground truth bounding boxes where mask_gt is True
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]  # (b, max_num_obj, 1, 4)
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]  # (b, 1, h*w, 4)
        
        # Compute IoU between selected boxes
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        
        # Calculate alignment metric using bbox_scores and overlaps
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
        # Calculate IoU using bbox_iou function with specified parameters
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
    # 根据给定的 metrics 张量，选择每个位置的前 self.topk 个候选项的指标值和索引
    topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
    
    # 如果 topk_mask 未提供，则根据 metrics 张量中的最大值确定 top-k 值，并扩展为布尔张量
    if topk_mask is None:
        topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
    
    # 根据 topk_mask，将 topk_idxs 中未选中的位置填充为 0
    topk_idxs.masked_fill_(~topk_mask, 0)

    # 创建一个与 metrics 张量形状相同的计数张量，用于统计每个位置被选择的次数
    count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
    ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
    
    # 遍历 topk 值，对每个 topk 索引位置添加计数值
    for k in range(self.topk):
        count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
    
    # 将计数张量中大于 1 的值（即超过一次被选择的位置），置为 0，用于过滤无效的候选项
    count_tensor.masked_fill_(count_tensor > 1, 0)

    # 将计数张量转换为与 metrics 相同类型的张量，并返回结果
    return count_tensor.to(metrics.dtype)
    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        # Create batch indices for indexing into gt_labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # Adjust target_gt_idx to point to the correct location in the flattened gt_labels
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        # Extract target labels from gt_labels using flattened indices
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        # Reshape gt_bboxes to (b * max_num_obj, 4) and then index using target_gt_idx
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)  # Clamp target_labels to ensure non-negative values

        # 10x faster than F.one_hot()
        # Initialize target_scores tensor with zeros and then scatter ones at target_labels indices
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        # Mask target_scores based on fg_mask to only keep scores for foreground anchor points
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select the positive anchor center in gt.

        Args:
            xy_centers (Tensor): shape(h*w, 2) - 存储锚点的中心坐标
            gt_bboxes (Tensor): shape(b, n_boxes, 4) - 存储每个图像中各个边界框的坐标信息

        Returns:
            (Tensor): shape(b, n_boxes, h*w) - 返回一个布尔值张量，指示哪些锚点与边界框有显著重叠
        """
        n_anchors = xy_centers.shape[0]  # 获取锚点的数量
        bs, n_boxes, _ = gt_bboxes.shape  # 获取边界框的数量和维度信息
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # 计算每个锚点与其对应边界框之间的距离，形成四个坐标差值并存储在bbox_deltas张量中
        return bbox_deltas.amin(3).gt_(eps)  # 判断距离是否大于阈值eps，并返回布尔值结果

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        If an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.

        Args:
            mask_pos (Tensor): shape(b, n_max_boxes, h*w) - 存储布尔值指示哪些锚点与边界框有重叠
            overlaps (Tensor): shape(b, n_max_boxes, h*w) - 存储每个锚点与所有边界框之间的IoU值

        Returns:
            target_gt_idx (Tensor): shape(b, h*w) - 返回每个锚点与其最佳匹配边界框的索引
            fg_mask (Tensor): shape(b, h*w) - 返回一个布尔值张量，指示哪些锚点被分配给了边界框
            mask_pos (Tensor): shape(b, n_max_boxes, h*w) - 返回更新后的锚点分配信息
        """
        # (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)  # 计算每个锚点分配给边界框的数量

        if fg_mask.max() > 1:  # 如果一个锚点被分配给多个边界框
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)  # 更新后的锚点分配数量

        # 找到每个网格服务的哪个gt（索引）
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos
class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (Tensor): shape(h*w, 2) - Anchor centers to consider.
            gt_bboxes (Tensor): shape(b, n_boxes, 5) - Ground-truth rotated bounding boxes.

        Returns:
            (Tensor): shape(b, n_boxes, h*w) - Boolean mask indicating positive anchor centers.
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2) - Rearrange bounding box coordinates.
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2) - Extract corner points a, b, and d from corners.
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a  # Compute vectors ab and ad from corner points.
        ad = d - a

        # (b, n_boxes, h*w, 2) - Calculate vector ap from anchor centers to point a.
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)  # Calculate norms and dot products for IoU calculation.
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None  # Ensure features are not None.
    dtype, device = feats[0].dtype, feats[0].device  # Determine data type and device from the first feature.
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape  # Retrieve height and width of feature map.
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # Generate x offsets.
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # Generate y offsets.
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)  # Create grid points.
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))  # Stack grid points into anchor points.
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))  # Create stride tensor.
    return torch.cat(anchor_points), torch.cat(stride_tensor)  # Concatenate anchor points and strides.


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)  # Split distance tensor into left-top and right-bottom.
    x1y1 = anchor_points - lt  # Compute top-left corner coordinates.
    x2y2 = anchor_points + rb  # Compute bottom-right corner coordinates.
    if xywh:
        c_xy = (x1y1 + x2y2) / 2  # Compute center coordinates.
        wh = x2y2 - x1y1  # Compute width and height.
        return torch.cat((c_xy, wh), dim)  # xywh bbox - Concatenate center and size.
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox - Concatenate top-left and bottom-right.


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)  # Split bbox tensor into x1y1 and x2y2.
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.
    """
    # Function not completed in provided snippet, further implementation required.
    # 将预测的旋转距离张量按照指定维度分割为左上角和右下角坐标偏移量
    lt, rb = pred_dist.split(2, dim=dim)
    # 计算预测的角度的余弦和正弦值
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # 计算中心点偏移量的 x 和 y 分量
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    # 根据旋转角度对中心点偏移量进行调整，得到旋转后的中心点坐标
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    # 将旋转后的中心点坐标与锚点相加，得到最终的旋转后的坐标
    xy = torch.cat([x, y], dim=dim) + anchor_points
    # 将左上角和右下角坐标偏移量相加，得到最终的旋转后的边界框坐标
    return torch.cat([xy, lt + rb], dim=dim)
```