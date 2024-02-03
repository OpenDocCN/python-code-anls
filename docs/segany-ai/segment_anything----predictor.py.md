# `.\segment-anything\segment_anything\predictor.py`

```
# 导入必要的库
import numpy as np
import torch

# 从segment_anything.modeling模块中导入Sam类
from segment_anything.modeling import Sam

# 从typing模块中导入Optional和Tuple类型
from typing import Optional, Tuple

# 从当前目录下的utils.transforms模块中导入ResizeLongestSide类
from .utils.transforms import ResizeLongestSide

# 定义SamPredictor类
class SamPredictor:
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        # 检查图像格式是否为RGB或BGR
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        # 如果图像格式与模型要求的格式不同，则进行颜色通道转换
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # 将图像转换为模型所需的格式
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        self.set_torch_image(input_image_torch, image.shape[:2])

    # 禁止梯度计算
    @torch.no_grad()
    # 设置 Torch 图像的方法，计算提供图像的图像嵌入，允许使用 'predict' 方法预测掩模。期望输入图像已经转换为模型所期望的格式。
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        计算提供图像的图像嵌入，允许使用 'predict' 方法预测掩模。期望输入图像已经转换为模型所期望的格式。

        参数:
          transformed_image (torch.Tensor): 输入图像，形状为 1x3xHxW，已经使用 ResizeLongestSide 进行转换。
          original_image_size (tuple(int, int)): 转换前图像的大小，格式为 (H, W)。
        """
        # 断言输入图像的形状符合要求
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        # 重置图像
        self.reset_image()

        # 设置原始大小和输入大小
        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        # 对输入图像进行预处理
        input_image = self.model.preprocess(transformed_image)
        # 获取图像嵌入
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    # 预测方法，用于预测掩模
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    @torch.no_grad()
    # Torch 版本的预测方法
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    # 返回当前设置图像的图像嵌入，形状为1xCxHxW，其中C是嵌入维度，(H,W)是SAM的嵌入空间维度（通常为C=256，H=W=64）。
    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        # 如果未设置图像，则引发运行时错误
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        # 断言特征存在，如果设置了图像，则特征必须存在
        assert self.features is not None, "Features must exist if an image has been set."
        # 返回特征
        return self.features

    # 返回模型的设备
    @property
    def device(self) -> torch.device:
        return self.model.device

    # 重置当前设置的图像
    def reset_image(self) -> None:
        """Resets the currently set image."""
        # 将is_image_set标记为False
        self.is_image_set = False
        # 将特征设置为None
        self.features = None
        # 将原始高度设置为None
        self.orig_h = None
        # 将原始宽度设置为None
        self.orig_w = None
        # 将输入高度设置为None
        self.input_h = None
        # 将输入宽度设置为None
        self.input_w = None
```