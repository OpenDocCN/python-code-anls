# `.\yolov8\ultralytics\models\sam\amg.py`

```
# Ultralytics YOLO 🚀, AGPL-3.0 license

# 导入标准数学库
import math
# 导入 itertools 中的 product 函数
from itertools import product
# 导入类型提示相关库
from typing import Any, Generator, List, Tuple

# 导入第三方库 numpy 和 torch
import numpy as np
import torch


def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> torch.Tensor:
    """Return a boolean tensor indicating if boxes are near the crop edge."""
    # 将 crop_box 和 orig_box 转换为 torch.Tensor，并使用与 boxes 相同的设备
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    # 调用 uncrop_boxes_xyxy 函数并将其结果转换为 float 类型
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    # 检查 boxes 是否在 crop 边缘附近，使用绝对容差 atol 进行比较
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    # 检查 boxes 是否在原始图像边缘附近，使用绝对容差 atol 进行比较
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    # 将 near_crop_edge 与 ~near_image_edge 逻辑与操作，以排除原始图像边缘的情况
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    # 检查是否有任何 boxes 在 crop 边缘附近，返回结果作为 boolean tensor
    return torch.any(near_crop_edge, dim=1)


def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    """Yield batches of data from the input arguments."""
    # 断言确保 args 不为空且每个参数的长度与第一个参数相同，用于批处理迭代
    assert args and all(len(a) == len(args[0]) for a in args), "Batched iteration must have same-size inputs."
    # 计算需要生成的批次数量
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    # 生成器函数，按批次生成输入参数的数据
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


def calculate_stability_score(masks: torch.Tensor, mask_threshold: float, threshold_offset: float) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks.

    The stability score is the IoU between the binary masks obtained by thresholding the predicted mask logits at high
    and low values.

    Notes:
        - One mask is always contained inside the other.
        - Save memory by preventing unnecessary cast to torch.int64
    """
    # 计算高阈值和低阈值下的二进制掩模的交集和并集
    intersections = (masks > (mask_threshold + threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > (mask_threshold - threshold_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    # 计算稳定性分数，即交集除以并集
    return intersections / unions


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generate a 2D grid of evenly spaced points in the range [0,1]x[0,1]."""
    # 计算每个边上均匀分布的点的偏移量
    offset = 1 / (2 * n_per_side)
    # 在 [offset, 1-offset] 区间内生成 n_per_side 个均匀分布的点
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    # 使用 np.tile 创建完整的网格
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    # 将点的 x 和 y 坐标堆叠起来，生成最终的点网格并返回
    return np.stack([points_x, points_y], axis=-1).reshape(-1, 2)


def build_all_layer_point_grids(n_per_side: int, n_layers: int, scale_per_layer: int) -> List[np.ndarray]:
    """Generate point grids for all crop layers."""
    # 生成所有裁剪层的点网格
    return [build_point_grid(int(n_per_side / (scale_per_layer**i))) for i in range(n_layers + 1)]


def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes.
    
    # 代码未完成，需要继续补充完整
    Each layer has (2**i)**2 boxes for the ith layer.
    """

    # 初始化空列表，用于存储裁剪框和图层索引
    crop_boxes, layer_idxs = [], []
    # 获取输入图像的高度和宽度
    im_h, im_w = im_size
    # 计算图像的较短边
    short_side = min(im_h, im_w)

    # 原始图像的裁剪框，表示整个图像
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        """Crops bounding boxes to the size of the input image."""
        # 根据输入图像的大小裁剪边界框
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    # 循环生成各层的裁剪框
    for i_layer in range(n_layers):
        # 每层的裁剪数量是2的(i_layer + 1)次方
        n_crops_per_side = 2 ** (i_layer + 1)
        # 计算重叠区域的大小
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        # 计算裁剪框的宽度和高度
        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        # 计算裁剪框左上角的坐标
        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # 以XYWH格式进行裁剪
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            # 根据左上角坐标和裁剪框的宽高计算裁剪框
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            # 将裁剪框添加到列表中
            crop_boxes.append(box)
            # 记录当前裁剪框属于的图层索引
            layer_idxs.append(i_layer + 1)

    # 返回裁剪框列表和图层索引列表作为结果
    return crop_boxes, layer_idxs
def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """Uncrop bounding boxes by adding the crop box offset."""
    # Extract the top-left corner coordinates of the crop box
    x0, y0, _, _ = crop_box
    # Create an offset tensor based on the crop box coordinates
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # Check if the boxes tensor has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    # Add the offset to the boxes tensor to uncrop them
    return boxes + offset


def uncrop_points(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    """Uncrop points by adding the crop box offset."""
    # Extract the top-left corner coordinates of the crop box
    x0, y0, _, _ = crop_box
    # Create an offset tensor based on the crop box coordinates
    offset = torch.tensor([[x0, y0]], device=points.device)
    # Check if the points tensor has a channel dimension
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    # Add the offset to the points tensor to uncrop them
    return points + offset


def uncrop_masks(masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int) -> torch.Tensor:
    """Uncrop masks by padding them to the original image size."""
    # Extract the crop box coordinates
    x0, y0, x1, y1 = crop_box
    # Check if the crop box covers the entire original image
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # Calculate the padding required to restore the masks to original size
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    # Pad the masks tensor to the original size with zeros
    return torch.nn.functional.pad(masks, pad, value=0)


def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str) -> Tuple[np.ndarray, bool]:
    """Remove small disconnected regions or holes in a mask, returning the mask and a modification indicator."""
    import cv2  # type: ignore

    # Ensure the mode is valid
    assert mode in {"holes", "islands"}, f"Provided mode {mode} is invalid"
    # Determine whether to correct holes or islands based on mode
    correct_holes = mode == "holes"
    # Convert mask to binary and invert if correcting holes
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    # Perform connected component analysis to find regions
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    # Extract region sizes
    sizes = stats[:, -1][1:]  # Row 0 is background label
    # Identify small regions based on area threshold
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    # If no small regions found, return original mask
    if not small_regions:
        return mask, False
    # Create list of labels to fill (small regions)
    fill_labels = [0] + small_regions
    # If not correcting holes, keep only the largest region if all are below threshold
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels] or [int(np.argmax(sizes)) + 1]
    # Generate mask with only specified fill labels
    mask = np.isin(regions, fill_labels)
    return mask, True


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks.

    Return [0,0,0,0] for an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # Return zeros if masks tensor is empty
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize masks to shape CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    masks = masks.flatten(0, -3) if len(shape) > 2 else masks.unsqueeze(0)
    # Compute top edges and their coordinates
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    # Calculate bottom edges based on top edges
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    # 计算输入高度坐标
    in_height_coords = in_height_coords + h * (~in_height)
    # 获取顶部边缘坐标
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # 获取左右边缘
    # 计算输入宽度
    in_width, _ = torch.max(masks, dim=-2)
    # 计算宽度坐标
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    # 获取右边缘坐标
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    # 更新宽度坐标，处理超出边界情况
    in_width_coords = in_width_coords + w * (~in_width)
    # 获取左边缘坐标
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # 如果掩码为空，则右边缘会在左边缘左侧，或者底部边缘在顶部边缘上方。
    # 将这些框替换为 [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    # 组合左上右下边缘坐标
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    # 根据空过滤器将不合规的框设置为零
    out = out * (~empty_filter).unsqueeze(-1)

    # 返回到原始形状
    return out.reshape(*shape[:-2], 4) if len(shape) > 2 else out[0]
```