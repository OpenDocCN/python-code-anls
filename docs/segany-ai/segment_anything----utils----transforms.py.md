# `.\segment-anything\segment_anything\utils\transforms.py`

```
# 导入必要的库
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
# 导入深拷贝函数和元组类型
from copy import deepcopy
from typing import Tuple

# 定义一个类，用于将图像和坐标/框调整为最长边为'target_length'的大小，并提供相应的方法
class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        # 初始化函数，设置目标长度
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # 将图像调整为目标大小，返回调整后的图像
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        # 将坐标调整为目标大小，返回调整后的坐标
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        # 将框调整为目标大小，返回调整后的框
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)
    # 对输入的 torch.Tensor 图像进行处理，将其大小调整为目标大小
    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # 期望输入的图像为批量图像，形状为 BxCxHxW，数据类型为浮点数
        # 此转换可能不完全匹配 apply_image。apply_image 是模型期望的转换。
        # 获取预处理后的目标大小
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        # 使用双线性插值调整图像大小
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    # 对输入的 torch.Tensor 坐标进行处理，根据原始图像大小调整坐标
    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        # 期望输入的 torch.Tensor 在最后一个维度上长度为 2。需要原始图像大小的 (H, W) 格式。
        # 获取原始图像的高度和宽度
        old_h, old_w = original_size
        # 获取预处理后的目标大小
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        # 深拷贝坐标，并转换为浮点数类型
        coords = deepcopy(coords).to(torch.float)
        # 根据原始图像大小和目标大小调整坐标
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    # 对输入的 torch.Tensor 边界框进行处理，根据原始图像大小调整边界框
    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        # 期望输入的 torch.Tensor 形状为 Bx4。需要原始图像大小的 (H, W) 格式。
        # 调用 apply_coords_torch 函数对边界框进行处理
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    # 静态方法，计算给定输入大小和目标长边长度的输出大小
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        # 根据输入大小和目标长边长度计算输出大小
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
```