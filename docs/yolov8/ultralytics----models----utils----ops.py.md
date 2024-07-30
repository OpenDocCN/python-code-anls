# `.\yolov8\ultralytics\models\utils\ops.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数模块
from scipy.optimize import linear_sum_assignment  # 导入SciPy库中的linear_sum_assignment函数

from ultralytics.utils.metrics import bbox_iou  # 导入Ultralytics工具包中的bbox_iou函数
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh  # 导入Ultralytics工具包中的坐标转换函数


class HungarianMatcher(nn.Module):
    """
    A module implementing the HungarianMatcher, which is a differentiable module to solve the assignment problem in an
    end-to-end fashion.

    HungarianMatcher performs optimal assignment over the predicted and ground truth bounding boxes using a cost
    function that considers classification scores, bounding box coordinates, and optionally, mask predictions.

    Attributes:
        cost_gain (dict): Dictionary of cost coefficients: 'class', 'bbox', 'giou', 'mask', and 'dice'.
        use_fl (bool): Indicates whether to use Focal Loss for the classification cost calculation.
        with_mask (bool): Indicates whether the model makes mask predictions.
        num_sample_points (int): The number of sample points used in mask cost calculation.
        alpha (float): The alpha factor in Focal Loss calculation.
        gamma (float): The gamma factor in Focal Loss calculation.

    Methods:
        forward(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None): Computes the
            assignment between predictions and ground truths for a batch.
        _cost_mask(bs, num_gts, masks=None, gt_mask=None): Computes the mask cost and dice cost if masks are predicted.
    """

    def __init__(self, cost_gain=None, use_fl=True, with_mask=False, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """Initializes HungarianMatcher with cost coefficients, Focal Loss, mask prediction, sample points, and alpha
        gamma factors.
        """
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain  # 设置成本系数字典，包括'class', 'bbox', 'giou', 'mask', 'dice'
        self.use_fl = use_fl  # 是否使用Focal Loss进行分类成本计算
        self.with_mask = with_mask  # 模型是否进行了掩模预测
        self.num_sample_points = num_sample_points  # 掩模成本计算中使用的样本点数目
        self.alpha = alpha  # Focal Loss计算中的alpha系数
        self.gamma = gamma  # Focal Loss计算中的gamma系数

    # This function is for future RT-DETR Segment models
    # def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
    #     assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
    #     # all masks share the same set of points for efficient matching
    #     sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
    #     sample_points = 2.0 * sample_points - 1.0
    #
    #     out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
    #     out_mask = out_mask.flatten(0, 1)
    #
    #     tgt_mask = torch.cat(gt_mask).unsqueeze(1)
    #     sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
    #     tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])
    #
    # 使用 torch.amp 自动混合精度，禁用 CUDA
    with torch.amp.autocast("cuda", enabled=False):
        # 计算二进制交叉熵损失
        pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
        neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
        cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
        cost_mask /= self.num_sample_points
        
        # 计算 Dice 损失
        out_mask = F.sigmoid(out_mask)
        numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
        denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
        cost_dice = 1 - (numerator + 1) / (denominator + 1)
        
        # 计算最终的损失函数 C，结合二进制交叉熵损失和 Dice 损失，根据设定的权重
        C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
    # 返回最终的损失 C
    return C
def get_cdn_group(
    batch, num_classes, num_queries, class_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False
):
    """
    Get contrastive denoising training group. This function creates a contrastive denoising training group with positive
    and negative samples from the ground truths (gt). It applies noise to the class labels and bounding box coordinates,
    and returns the modified labels, bounding boxes, attention mask and meta information.

    Args:
        batch (dict): A dict that includes 'gt_cls' (torch.Tensor with shape [num_gts, ]), 'gt_bboxes'
            (torch.Tensor with shape [num_gts, 4]), 'gt_groups' (List(int)) which is a list of batch size length
            indicating the number of gts of each image.
        num_classes (int): Number of classes.
        num_queries (int): Number of queries.
        class_embed (torch.Tensor): Embedding weights to map class labels to embedding space.
        num_dn (int, optional): Number of denoising. Defaults to 100.
        cls_noise_ratio (float, optional): Noise ratio for class labels. Defaults to 0.5.
        box_noise_scale (float, optional): Noise scale for bounding box coordinates. Defaults to 1.0.
        training (bool, optional): If it's in training mode. Defaults to False.

    Returns:
        (Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Dict]]): The modified class embeddings,
            bounding boxes, attention mask and meta information for denoising. If not in training mode or 'num_dn'
            is less than or equal to 0, the function returns None for all elements in the tuple.
    """

    # 如果不处于训练模式或者 num_dn 小于等于 0，则返回 None
    if (not training) or num_dn <= 0:
        return None, None, None, None

    # 从 batch 中获取 gt_groups，即每张图像中 gt 的数量列表
    gt_groups = batch["gt_groups"]
    # 计算总的 gt 数量
    total_num = sum(gt_groups)
    # 获取一个 batch 中最大的 gt 数量
    max_nums = max(gt_groups)
    
    # 如果最大的 gt 数量为 0，则返回 None
    if max_nums == 0:
        return None, None, None, None
    
    # 计算每个 group 中的数量，确保至少为 1
    num_group = num_dn // max_nums
    num_group = 1 if num_group == 0 else num_group
    
    # 获取 batch 的大小
    bs = len(gt_groups)
    
    # 从 batch 中获取 gt_cls 和 gt_bbox
    gt_cls = batch["cls"]  # (bs*num, )
    gt_bbox = batch["bboxes"]  # bs*num, 4
    b_idx = batch["batch_idx"]
    
    # 每个 group 包含正负样本
    dn_cls = gt_cls.repeat(2 * num_group)  # (2*num_group*bs*num, )
    dn_bbox = gt_bbox.repeat(2 * num_group, 1)  # 2*num_group*bs*num, 4
    dn_b_idx = b_idx.repeat(2 * num_group).view(-1)  # (2*num_group*bs*num, )
    
    # 创建负样本的索引
    neg_idx = torch.arange(total_num * num_group, dtype=torch.long, device=gt_bbox.device) + num_group * total_num
    
    # 如果 cls_noise_ratio 大于 0，则对 dn_cls 应用噪声
    if cls_noise_ratio > 0:
        # 生成一个掩码，以半概率应用于 bbox
        mask = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)
        idx = torch.nonzero(mask).squeeze(-1)
        # 随机生成新的标签
        new_label = torch.randint_like(idx, 0, num_classes, dtype=dn_cls.dtype, device=dn_cls.device)
        dn_cls[idx] = new_label
    # 如果盒子噪声比例大于0，则进行以下操作
    known_bbox = xywh2xyxy(dn_bbox)  # 将相对坐标转换为绝对坐标格式

    # 计算随机扰动的大小
    diff = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 2*num_group*bs*num, 4

    # 生成随机符号和随机部分
    rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0
    rand_part = torch.rand_like(dn_bbox)
    rand_part[neg_idx] += 1.0
    rand_part *= rand_sign

    # 添加随机扰动到已知的边界框
    known_bbox += rand_part * diff

    # 将边界框裁剪到0到1的范围内
    known_bbox.clip_(min=0.0, max=1.0)

    # 将绝对坐标格式的边界框转换回相对坐标格式
    dn_bbox = xyxy2xywh(known_bbox)

    # 对相对坐标进行逆sigmoid变换
    dn_bbox = torch.logit(dn_bbox, eps=1e-6)

num_dn = int(max_nums * 2 * num_group)  # 计算总的去噪查询数

# 创建填充的类别嵌入和边界框
dn_cls_embed = class_embed[dn_cls]  # bs*num * 2 * num_group, 256
padding_cls = torch.zeros(bs, num_dn, dn_cls_embed.shape[-1], device=gt_cls.device)
padding_bbox = torch.zeros(bs, num_dn, 4, device=gt_bbox.device)

# 构建映射索引用于对齐去噪后的查询与原始查询
map_indices = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups])
pos_idx = torch.stack([map_indices + max_nums * i for i in range(num_group)], dim=0)

map_indices = torch.cat([map_indices + max_nums * i for i in range(2 * num_group)])

# 将类别嵌入和边界框填充到填充张量中
padding_cls[(dn_b_idx, map_indices)] = dn_cls_embed
padding_bbox[(dn_b_idx, map_indices)] = dn_bbox

tgt_size = num_dn + num_queries  # 计算目标的总大小
attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)  # 创建注意力掩码

# 设定查询与重构之间的匹配不能看到
attn_mask[num_dn:, :num_dn] = True

# 设定重构之间相互不能看到
for i in range(num_group):
    if i == 0:
        attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
    if i == num_group - 1:
        attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * i * 2] = True
    else:
        attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), max_nums * 2 * (i + 1) : num_dn] = True
        attn_mask[max_nums * 2 * i : max_nums * 2 * (i + 1), : max_nums * 2 * i] = True

# 构建去噪任务的元信息字典
dn_meta = {
    "dn_pos_idx": [p.reshape(-1) for p in pos_idx.cpu().split(list(gt_groups), dim=1)],
    "dn_num_group": num_group,
    "dn_num_split": [num_dn, num_queries],
}

# 返回结果
return (
    padding_cls.to(class_embed.device),
    padding_bbox.to(class_embed.device),
    attn_mask.to(class_embed.device),
    dn_meta,
)
```