# `.\segment-anything\segment_anything\utils\amg.py`

```
# 导入必要的库
import numpy as np
import torch

import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple

# 定义一个用于存储掩码及相关数据的批处理格式的结构
class MaskData:
    """
    A structure for storing masks and their related data in batched format.
    Implements basic filtering and concatenation.
    """

    # 初始化方法，接受关键字参数
    def __init__(self, **kwargs) -> None:
        # 遍历所有参数值，检查是否为列表、numpy数组或torch张量
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, torch.Tensor)
            ), "MaskData only supports list, numpy arrays, and torch tensors."
        # 使用关键字参数创建一个字典来存储数据
        self._stats = dict(**kwargs)

    # 设置指定键的值
    def __setitem__(self, key: str, item: Any) -> None:
        # 检查值是否为列表、numpy数组或torch张量
        assert isinstance(
            item, (list, np.ndarray, torch.Tensor)
        ), "MaskData only supports list, numpy arrays, and torch tensors."
        # 设置键对应的值
        self._stats[key] = item

    # 删除指定键及其对应的值
    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    # 获取指定键的值
    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    # 返回字典的键值对视图
    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    # 根据给定的掩码过滤数据
    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")
    # 将新的统计数据合并到当前对象中
    def cat(self, new_stats: "MaskData") -> None:
        # 遍历新的统计数据字典
        for k, v in new_stats.items():
            # 如果键不在当前对象的统计数据中，或者对应值为空
            if k not in self._stats or self._stats[k] is None:
                # 深拷贝新值并赋给当前对象的统计数据
                self._stats[k] = deepcopy(v)
            # 如果值是 torch.Tensor 类型
            elif isinstance(v, torch.Tensor):
                # 在指定维度上拼接当前对象的值和新值
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            # 如果值是 np.ndarray 类型
            elif isinstance(v, np.ndarray):
                # 在指定轴上拼接当前对象的值和新值
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            # 如果值是列表类型
            elif isinstance(v, list):
                # 将新值深拷贝后与当前对象的值相加
                self._stats[k] = self._stats[k] + deepcopy(v)
            # 如果值类型不受支持，抛出异常
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    # 将 torch.Tensor 类型的值转换为 numpy 数组
    def to_numpy(self) -> None:
        # 遍历当前对象的统计数据字典
        for k, v in self._stats.items():
            # 如果值是 torch.Tensor 类型
            if isinstance(v, torch.Tensor):
                # 将值从计算图中分离，转移到 CPU 上，并转换为 numpy 数组
                self._stats[k] = v.detach().cpu().numpy()
# 判断边界框是否靠近裁剪边缘，但不靠近原始图像边缘
def is_box_near_crop_edge(
    boxes: torch.Tensor, crop_box: List[int], orig_box: List[int], atol: float = 20.0
) -> torch.Tensor:
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    # 将裁剪边界框和原始边界框转换为 torch 张量
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    # 将边界框从裁剪坐标系转换为原始坐标系
    boxes = uncrop_boxes_xyxy(boxes, crop_box).float()
    # 判断边界框是否靠近裁剪边缘
    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    # 判断边界框是否靠近原始图像边缘
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    # 过滤掉靠近原始图像边缘的边界框
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    # 返回是否有边界框靠近裁剪边缘的结果
    return torch.any(near_crop_edge, dim=1)


# 将边界框从 (x1, y1, x2, y2) 格式转换为 (x, y, w, h) 格式
def box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    box_xywh = deepcopy(box_xyxy)
    # 计算边界框的宽度和高度
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh


# 批量迭代器，用于按批次处理数据
def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    # 检查输入参数是否都具有相同的大小
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    # 计算批次数量
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        # 按批次生成输入参数
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


# 将 PyTorch 张量的掩码转换为未压缩的 RLE 编码
def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # 将张量转置为 Fortran 顺序，并展平为 (h*w) 的形状
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # 计算像素值变化的索引
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # 编码运行长度
    out = []
    # 遍历范围为 b 的循环
    for i in range(b):
        # 获取当前索引 i 对应的变化索引
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        # 将当前索引拼接为 [0, cur_idxs + 1, h * w] 的张量
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        # 计算相邻元素之间的差值
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        # 如果 tensor[i, 0] 为 0，则 counts 为空列表，否则为 [0]
        counts = [] if tensor[i, 0] == 0 else [0]
        # 将 btw_idxs 转换为不需要梯度计算的 CPU 张量，并转换为列表后拼接到 counts 中
        counts.extend(btw_idxs.detach().cpu().tolist())
        # 将结果以字典形式添加到 out 列表中
        out.append({"size": [h, w], "counts": counts})
    # 返回 out 列表
    return out
# 从未压缩的 RLE 编码计算二进制掩码
def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    # 获取 RLE 编码中的高度和宽度
    h, w = rle["size"]
    # 创建一个空的布尔类型数组，用于存储掩码
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    # 根据 RLE 编码中的 counts 数组生成掩码
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    # 将掩码数组重塑为指定的宽度和高度
    mask = mask.reshape(w, h)
    # 返回转置后的掩码数组，以符合 C 顺序
    return mask.transpose()  # Put in C order


# 从 RLE 编码中计算面积
def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])


# 计算稳定性分数
def calculate_stability_score(
    masks: torch.Tensor, mask_threshold: float, threshold_offset: float
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # 一个掩码始终包含在另一个掩码中
    # 通过防止不必要的转换为 torch.int64 来节省内存
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / unions


# 生成均匀分布在 [0,1]x[0,1] 区域内的二维点网格
def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


# 为所有裁剪层生成点网格
def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    # 遍历每一层，包括输入层和隐藏层
    for i in range(n_layers + 1):
        # 计算每一层的点数，根据每层的缩放比例和每层的点数
        n_points = int(n_per_side / (scale_per_layer**i))
        # 调用函数构建每一层的点网格，并将结果添加到列表中
        points_by_layer.append(build_point_grid(n_points))
    # 返回每一层的点网格列表
    return points_by_layer
# 生成不同尺寸的裁剪框列表，每一层有(2**i)**2个裁剪框
def generate_crop_boxes(
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # 原始图像
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    # 计算裁剪长度
    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # 以XYWH格式裁剪
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


# 反裁剪XYXY格式的框
def uncrop_boxes_xyxy(boxes: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0, x0, y0]], device=boxes.device)
    # 检查框是否有通道维度
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    return boxes + offset


# 反裁剪点
def uncrop_points(points: torch.Tensor, crop_box: List[int]) -> torch.Tensor:
    x0, y0, _, _ = crop_box
    offset = torch.tensor([[x0, y0]], device=points.device)
    # 检查点是否有通道维度
    if len(points.shape) == 3:
        offset = offset.unsqueeze(1)
    return points + offset


# 反裁剪掩码
def uncrop_masks(
    # 定义函数参数，masks为torch.Tensor类型，crop_box为包含4个整数的列表，orig_h和orig_w为整数类型
    masks: torch.Tensor, crop_box: List[int], orig_h: int, orig_w: int
# 根据给定的裁剪框坐标对 masks 进行裁剪，如果裁剪框与原始大小相同则直接返回 masks
def crop_and_pad_masks(masks: torch.Tensor, crop_box: Tuple[int, int, int, int], orig_w: int, orig_h: int) -> torch.Tensor:
    x0, y0, x1, y1 = crop_box
    # 如果裁剪框与原始大小相同，则直接返回 masks
    if x0 == 0 and y0 == 0 and x1 == orig_w and y1 == orig_h:
        return masks
    # 计算裁剪框与原始大小之间的差值，用于坐标变换
    pad_x, pad_y = orig_w - (x1 - x0), orig_h - (y1 - y0)
    pad = (x0, pad_x - x0, y0, pad_y - y0)
    # 使用 torch.nn.functional.pad 进行坐标变换
    return torch.nn.functional.pad(masks, pad, value=0)


# 移除 mask 中面积小于阈值的孤立区域或孔洞，返回处理后的 mask 和是否修改的指示
def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str) -> Tuple[np.ndarray, bool]:
    """
    Removes small disconnected regions and holes in a mask. Returns the
    mask and an indicator of if the mask has been modified.
    """
    import cv2  # type: ignore

    # 检查 mode 是否为 "holes" 或 "islands"
    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    # 转换 mask 为工作 mask，用于处理孔洞或孤立区域
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    # 使用 cv2.connectedComponentsWithStats 获取连通区域信息
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)
    sizes = stats[:, -1][1:]  # 第一行是背景标签
    # 找出面积小于阈值的区域
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    # 如果没有小区域，则直接返回原始 mask
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # 如果每个区域都小于阈值，则保留最大的区域
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    # 根据填充标签更新 mask
    mask = np.isin(regions, fill_labels)
    return mask, True


# 对未压缩的 RLE 进行编码，返回编码后的 RLE
def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]
    # 使用 pycocotools.mask.frPyObjects 进行 RLE 编码
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # 必要的序列化操作
    return rle


# 将批量的 masks 转换为 XYXY 格式的框，对于空 mask 返回 [0,0,0,0]，输入形状为 C1xC2x...xHxW，输出形状为 C1xC2x...x4
def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # 如果输入的 masks 是空的，则返回一个全零的张量，避免 torch.max 报错
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # 将 masks 的形状标准化为 CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # 获取顶部和底部边缘
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # 获取左侧和右侧边缘
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # 如果 mask 是空的，则右边缘会在左边缘的左侧，将这些框替换为 [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # 返回到原始形状
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out
```