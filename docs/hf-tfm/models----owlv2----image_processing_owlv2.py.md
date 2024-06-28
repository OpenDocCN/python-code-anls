# `.\models\owlv2\image_processing_owlv2.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for OWLv2."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
    center_to_corners_format,
    pad,
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import (
    TensorType,
    is_scipy_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)

# Set up logging
logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


if is_vision_available():
    import PIL

if is_scipy_available():
    from scipy import ndimage as ndi


# Copied from transformers.models.owlvit.image_processing_owlvit._upcast
def _upcast(t):
    """
    Protects from numerical overflows in multiplications by upcasting to the equivalent higher type.

    Args:
        t (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Upcasted tensor to float32 or float64 for floating point types,
                      or to int32 or int64 for integer types.
    """
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.owlvit.image_processing_owlvit.box_area
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes.

    Args:
        boxes (torch.FloatTensor): Bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        torch.FloatTensor: Tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.owlvit.image_processing_owlvit.box_iou
def box_iou(boxes1, boxes2):
    """
    Computes the intersection over union (IoU) of two sets of bounding boxes.

    Args:
        boxes1 (torch.FloatTensor): Bounding boxes set 1 in (x1, y1, x2, y2) format.
        boxes2 (torch.FloatTensor): Bounding boxes set 2 in (x1, y1, x2, y2) format.

    Returns:
        torch.FloatTensor: IoU for each pair of boxes from boxes1 and boxes2.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    # 计算框的宽度和高度，使用 clamp 函数确保宽度和高度不小于 0
    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    # 计算交集的面积，对每个框的宽度和高度进行逐元素乘法
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    # 计算并集的面积，area1 和 area2 分别为两组框的面积，使用广播机制扩展计算
    union = area1[:, None] + area2 - inter

    # 计算 IoU（Intersection over Union），即交集面积除以并集面积
    iou = inter / union
    # 返回 IoU 和并集面积
    return iou, union
def _preprocess_resize_output(image, output_shape):
    """Validate resize output shape according to input image.

    Args:
        image (`np.ndarray`):
         Image to be resized.
        output_shape (`iterable`):
            Size of the generated output image `(rows, cols[, ...][, dim])`. If `dim` is not provided, the number of
            channels is preserved.

    Returns
    -------
    image (`np.ndarray):
        The input image, but with additional singleton dimensions appended in the case where `len(output_shape) >
        input.ndim`.
    output_shape (`Tuple`):
        The output shape converted to tuple.

    Raises
    ------
    ValueError:
        If output_shape length is smaller than the image number of dimensions.

    Notes
    -----
    The input image is reshaped if its number of dimensions is not equal to output_shape_length.
    """
    output_shape = tuple(output_shape)  # Convert output_shape to a tuple
    output_ndim = len(output_shape)  # Get the number of dimensions of output_shape
    input_shape = image.shape  # Get the shape of the input image
    if output_ndim > image.ndim:
        # If output_ndim is greater than the number of dimensions of the input image,
        # append singleton dimensions to input_shape
        input_shape += (1,) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)  # Reshape the input image
    elif output_ndim == image.ndim - 1:
        # If output_ndim is equal to the number of dimensions of the input image minus one,
        # it's a multichannel case; append shape of last axis to output_shape
        output_shape = output_shape + (image.shape[-1],)
    elif output_ndim < image.ndim:
        # If output_ndim is less than the number of dimensions of the input image, raise an error
        raise ValueError("output_shape length cannot be smaller than the image number of dimensions")

    return image, output_shape


def _clip_warp_output(input_image, output_image):
    """Clip output image to range of values of input image.

    Note that this function modifies the values of *output_image* in-place.

    Taken from:
    https://github.com/scikit-image/scikit-image/blob/b4b521d6f0a105aabeaa31699949f78453ca3511/skimage/transform/_warps.py#L640.

    Args:
    ----
    input_image : ndarray
        Input image.
    output_image : ndarray
        Output image, which is modified in-place.
    """
    min_val = np.min(input_image)  # Get the minimum value of the input image
    if np.isnan(min_val):
        # If NaNs are detected in the input image, use NaN-safe min/max functions
        min_func = np.nanmin
        max_func = np.nanmax
        min_val = min_func(input_image)
    else:
        min_func = np.min
        max_func = np.max
    max_val = max_func(input_image)  # Get the maximum value of the input image

    output_image = np.clip(output_image, min_val, max_val)  # Clip output_image to the range [min_val, max_val]

    return output_image


class Owlv2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs an OWLv2 image processor.
    """
    pass  # Placeholder, no additional functionality added here
    # 参数定义部分，用于配置图像预处理的选项
    Args:
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否对图像进行重新缩放，缩放因子由 `rescale_factor` 指定。可以在 `preprocess` 方法中通过 `do_rescale` 参数进行覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果需要重新缩放图像，使用的缩放因子。可以在 `preprocess` 方法中通过 `rescale_factor` 参数进行覆盖。
        do_pad (`bool`, *optional*, defaults to `True`):
            是否对图像进行填充，使其变成正方形，并在右下角用灰色像素填充。可以在 `preprocess` 方法中通过 `do_pad` 参数进行覆盖。
        do_resize (`bool`, *optional*, defaults to `True`):
            控制是否将图像的（高度，宽度）尺寸调整为指定的 `size`。可以在 `preprocess` 方法中通过 `do_resize` 参数进行覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 960, "width": 960}`):
            要调整的图像大小。可以在 `preprocess` 方法中通过 `size` 参数进行覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            如果调整图像大小，要使用的重采样方法。可以在 `preprocess` 方法中通过 `resample` 参数进行覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行标准化。可以在 `preprocess` 方法中通过 `do_normalize` 参数进行覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_MEAN`):
            如果进行图像标准化，要使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在 `preprocess` 方法中通过 `image_mean` 参数进行覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `OPENAI_CLIP_STD`):
            如果进行图像标准化，要使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在 `preprocess` 方法中通过 `image_std` 参数进行覆盖。
    ```
    ) -> None:
        super().__init__(**kwargs)  # 调用父类的构造方法，传递所有的关键字参数给父类

        self.do_rescale = do_rescale  # 设置是否进行重新缩放的标志
        self.rescale_factor = rescale_factor  # 设置重新缩放的因子
        self.do_pad = do_pad  # 设置是否进行填充的标志
        self.do_resize = do_resize  # 设置是否进行调整大小的标志
        self.size = size if size is not None else {"height": 960, "width": 960}  # 设置图像的目标大小，默认为960x960
        self.resample = resample  # 设置重采样方法
        self.do_normalize = do_normalize  # 设置是否进行归一化的标志
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN  # 设置图像归一化的均值
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD  # 设置图像归一化的标准差
        self._valid_processor_keys = [  # 定义有效的处理器关键字列表，用于后续检查和处理
            "images",
            "do_pad",
            "do_resize",
            "size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    def pad(
        self,
        image: np.array,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Pad an image to a square with gray pixels on the bottom and the right, as per the original OWLv2
        implementation.

        Args:
            image (`np.ndarray`):
                Image to pad.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        height, width = get_image_size(image)  # 获取图像的高度和宽度
        size = max(height, width)  # 计算需要填充到的正方形大小
        image = pad(
            image=image,
            padding=((0, size - height), (0, size - width)),  # 设置填充的方式，向下和向右填充灰色像素
            constant_values=0.5,  # 设置填充的常数值
            data_format=data_format,  # 设置填充后图像的数据格式
            input_data_format=input_data_format,  # 设置输入图像的数据格式
        )

        return image

    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        anti_aliasing: bool = True,
        anti_aliasing_sigma=None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    def resize_image(
        image: np.ndarray,
        size: Dict[str, int],
        anti_aliasing: bool = True,
        anti_aliasing_sigma: Optional[float] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None
    ) -> np.ndarray:
        """
        Resize an image using scipy functions with optional anti-aliasing.
    
        Args:
            image (`np.ndarray`):
                The input image to resize.
            size (`Dict[str, int]`):
                Dictionary containing the target height and width of the resized image.
            anti_aliasing (`bool`, *optional*, defaults to `True`):
                Whether to apply anti-aliasing when downsampling the image.
            anti_aliasing_sigma (`float`, *optional*, defaults to `None`):
                Standard deviation for the Gaussian kernel used in anti-aliasing. Automatically calculated if `None`.
            data_format (`str` or `ChannelDimension`, *optional*):
                The desired channel dimension format of the output image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. Inferred from input image if not provided.
    
        Raises:
            ValueError: If anti_aliasing_sigma is negative or if it's greater than zero without downsampling along all axes.
            UserWarning: If anti_aliasing_sigma is greater than zero but not downsampling along all axes.
    
        Returns:
            np.ndarray:
                Resized image with dimensions specified by `size`.
        """
        # Check and ensure scipy is available
        requires_backends(self, "scipy")
    
        # Calculate the desired output shape based on `size`
        output_shape = (size["height"], size["width"])
    
        # Convert image to the last channel dimension format
        image = to_channel_dimension_format(image, ChannelDimension.LAST)
    
        # Preprocess: Adjust image and output shape based on resize operation
        image, output_shape = _preprocess_resize_output_shape(image, output_shape)
    
        # Determine the input shape of the image
        input_shape = image.shape
    
        # Compute scaling factors based on input and output shapes
        factors = np.divide(input_shape, output_shape)
    
        # Set parameters for np.pad translation to scipy.ndimage modes
        ndi_mode = "mirror"
        cval = 0
        order = 1
    
        # Apply anti-aliasing if specified
        if anti_aliasing:
            if anti_aliasing_sigma is None:
                anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
            else:
                anti_aliasing_sigma = np.atleast_1d(anti_aliasing_sigma) * np.ones_like(factors)
                if np.any(anti_aliasing_sigma < 0):
                    raise ValueError("Anti-aliasing standard deviation must be greater than or equal to zero")
                elif np.any((anti_aliasing_sigma > 0) & (factors <= 1)):
                    warnings.warn(
                        "Anti-aliasing standard deviation greater than zero but not down-sampling along all axes"
                    )
            # Apply Gaussian filter for anti-aliasing
            filtered = ndi.gaussian_filter(image, anti_aliasing_sigma, cval=cval, mode=ndi_mode)
        else:
            # No anti-aliasing: use original image
            filtered = image
    
        # Compute zoom factors based on scaling factors
        zoom_factors = [1 / f for f in factors]
    
        # Perform zooming operation on the filtered image
        out = ndi.zoom(filtered, zoom_factors, order=order, mode=ndi_mode, cval=cval, grid_mode=True)
    
        # Clip and warp output image based on input and output shapes
        image = _clip_warp_output(image, out)
    
        # Convert back to the input data format if specified
        image = to_channel_dimension_format(image, input_data_format, ChannelDimension.LAST)
    
        # Convert to the desired data format if specified
        image = (
            to_channel_dimension_format(image, data_format, input_data_format)
            if data_format is not None
            else image
        )
    
        # Return the resized image
        return image
    `
        # 定义一个方法 preprocess，接受多种参数，用于预处理图像数据
        def preprocess(
            self,
            images: ImageInput,  # 输入图像，类型为 ImageInput
            do_pad: bool = None,  # 是否进行填充，默认为 None
            do_resize: bool = None,  # 是否进行缩放，默认为 None
            size: Dict[str, int] = None,  # 图像尺寸，类型为字典，键为字符串，值为整数，默认为 None
            do_rescale: bool = None,  # 是否进行重缩放，默认为 None
            rescale_factor: float = None,  # 重缩放因子，默认为 None
            do_normalize: bool = None,  # 是否进行归一化，默认为 None
            image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，支持单一浮点数或浮点数列表，默认为 None
            image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，支持单一浮点数或浮点数列表，默认为 None
            return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型，可以是字符串或 TensorType，默认为 None
            data_format: ChannelDimension = ChannelDimension.FIRST,  # 数据格式，默认为 ChannelDimension.FIRST
            input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为 None
            **kwargs,  # 额外的关键字参数
        # 从 transformers.models.owlvit.image_processing_owlvit.OwlViTImageProcessor 导入的 post_process_object_detection 方法的签名
        def post_process_object_detection(
            self, outputs, threshold: float = 0.1, target_sizes: Union[TensorType, List[Tuple]] = None  # 定义 post_process_object_detection 方法，接收输出、阈值和目标尺寸，默认为 None
    ):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # TODO: (amy) add support for other frameworks

        # Extract logits and boxes from the model outputs
        logits, boxes = outputs.logits, outputs.pred_boxes

        # Validate target sizes if provided
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # Calculate probabilities, scores, and labels
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert bounding boxes to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates if target_sizes is provided
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        # Prepare results as a list of dictionaries containing scores, labels, and boxes
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    # Copied from transformers.models.owlvit.image_processing_owlvit.OwlViTImageProcessor.post_process_image_guided_detection
```