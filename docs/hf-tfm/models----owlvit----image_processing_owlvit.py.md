# `.\models\owlvit\image_processing_owlvit.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for OwlViT"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    center_crop,
    center_to_corners_format,
    rescale,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_torch_available, logging

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def _upcast(t):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.
    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    # Calculate the area of each box using the formula: (width) * (height)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Computes the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (`torch.FloatTensor` of shape `(N, 4)`): Bounding boxes in format (x1, y1, x2, y2).
        boxes2 (`torch.FloatTensor` of shape `(M, 4)`): Bounding boxes in format (x1, y1, x2, y2).

    Returns:
        `torch.FloatTensor`: IoU values of shape `(N, M)`.
        `torch.FloatTensor`: Union area of shape `(N, M)`.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Calculate the coordinates of the intersection boxes
    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # Calculate width and height of intersection area, clamping at zero
    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]

    # Calculate intersection area
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    # Calculate union area
    union = area1[:, None] + area2 - inter

    # Calculate IoU
    iou = inter / union
    return iou, union


class OwlViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs an OWL-ViT image processor.
    """
    pass  # Placeholder for future implementation
    """
    This image processor inherits from `ImageProcessingMixin` which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the shorter edge of the input to a certain `size`.
        size (`Dict[str, int]`, *optional*, defaults to {"height": 768, "width": 768}):
            The size to use for resizing the image. Only has an effect if `do_resize` is set to `True`. If `size` is a
            sequence like (h, w), output size will be matched to this. If `size` is an int, then image will be resized
            to (size, size).
        resample (`int`, *optional*, defaults to `Resampling.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
            `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
            `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
            to `True`.
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to {"height": 768, "width": 768}):
            The size to use for center cropping the image. Only has an effect if `do_center_crop` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input by a certain factor.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            The factor to use for rescaling the image. Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with `image_mean` and `image_std`. Desired output size when applying
            center-cropping. Only has an effect if `do_center_crop` is set to `True`.
        image_mean (`List[int]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """
        ):
            # 如果未指定尺寸，则默认为 768x768，并确保尺寸是一个字典格式
            size = size if size is not None else {"height": 768, "width": 768}
            # 使用函数将尺寸规范化为标准字典格式，确保宽高一致
            size = get_size_dict(size, default_to_square=True)

            # 如果未指定裁剪尺寸，则默认为 768x768，并确保尺寸是一个字典格式
            crop_size = crop_size if crop_size is not None else {"height": 768, "width": 768}
            # 使用函数将裁剪尺寸规范化为标准字典格式，确保宽高一致
            crop_size = get_size_dict(crop_size, default_to_square=True)

            # 在 OWL-ViT hub 上早期的配置中，使用了 "rescale" 作为标志位。
            # 这与视觉图像处理方法 `rescale` 冲突，因为它将在 super().__init__ 调用期间设置为属性。
            # 为了向后兼容，这里将其处理为 `do_rescale` 的键值对参数。
            if "rescale" in kwargs:
                rescale_val = kwargs.pop("rescale")
                kwargs["do_rescale"] = rescale_val

            # 调用父类的初始化方法，传递所有参数
            super().__init__(**kwargs)
            # 设置对象的各个属性
            self.do_resize = do_resize
            self.size = size
            self.resample = resample
            self.do_center_crop = do_center_crop
            self.crop_size = crop_size
            self.do_rescale = do_rescale
            self.rescale_factor = rescale_factor
            self.do_normalize = do_normalize
            # 如果未指定图像均值，则使用默认的 OpenAI Clip 均值
            self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
            # 如果未指定图像标准差，则使用默认的 OpenAI Clip 标准差
            self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
            # 初始化有效的处理器关键字列表，用于验证参数的完整性
            self._valid_processor_keys = [
                "images",
                "do_resize",
                "size",
                "resample",
                "do_center_crop",
                "crop_size",
                "do_rescale",
                "rescale_factor",
                "do_normalize",
                "image_mean",
                "image_std",
                "return_tensors",
                "data_format",
                "input_data_format",
            ]

        def resize(
            self,
            image: np.ndarray,
            size: Dict[str, int],
            resample: PILImageResampling.BICUBIC,
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
            **kwargs,
    def center_crop(
        self,
        image: np.ndarray,
        crop_size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Center crop an image to a certain size.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            crop_size (`Dict[str, int]`):
                The size to center crop the image to. Must contain height and width keys.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 将crop_size字典转换为包含高度和宽度的标准尺寸字典
        crop_size = get_size_dict(crop_size, default_to_square=True)
        # 检查crop_size字典是否包含必需的高度和宽度键
        if "height" not in crop_size or "width" not in crop_size:
            raise ValueError("crop_size dictionary must contain height and width keys")

        # 调用函数进行中心裁剪，并返回裁剪后的图像
        return center_crop(
            image,
            (crop_size["height"], crop_size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    def rescale(
        self,
        image: np.ndarray,
        rescale_factor: float,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Rescale the image by the given factor. image = image * rescale_factor.

        Args:
            image (`np.ndarray`):
                Image to rescale.
            rescale_factor (`float`):
                The value to use for rescaling.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input image. If unset, is inferred from the input image. Can be
                one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        # 调用内部的图像重缩放函数，返回重缩放后的图像
        return rescale(image, rescale_factor, data_format=data_format, input_data_format=input_data_format)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocess images with various transformations like resizing, cropping, rescaling, and normalization.

        Args:
            images (`ImageInput`):
                Input images to preprocess.
            do_resize (`bool`, *optional*):
                Whether to resize the images.
            size (`Dict[str, int]`, *optional*):
                Target size for resizing, as a dictionary with keys 'height' and 'width'.
            resample (`PILImageResampling`, *optional*):
                Resampling method for resizing images.
            do_center_crop (`bool`, *optional*):
                Whether to perform center cropping.
            crop_size (`Dict[str, int]`, *optional*):
                Size of the center crop, as a dictionary with keys 'height' and 'width'.
            do_rescale (`bool`, *optional*):
                Whether to rescale the images.
            rescale_factor (`float`, *optional*):
                Factor to use for rescaling the images.
            do_normalize (`bool`, *optional*):
                Whether to normalize the images.
            image_mean (`float` or `List[float]`, *optional*):
                Mean values for image normalization.
            image_std (`float` or `List[float]`, *optional*):
                Standard deviation values for image normalization.
            return_tensors (`TensorType` or `str`, *optional*):
                Desired tensor type for output images.
            data_format (`str` or `ChannelDimension`):
                The channel dimension format for the output images.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the input images.
            **kwargs:
                Additional keyword arguments for preprocessing.

        Returns:
            Preprocessed images.
        """
        # 省略了具体的预处理步骤，根据参数进行图像预处理并返回预处理后的结果
        pass
    def post_process(self, outputs, target_sizes):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation). For visualization, this should be the image size after data
                augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # TODO: (amy) add support for other frameworks
        # 发出警告信息，提醒用户该函数将在 Transformers 版本 v5 中被移除，并建议使用新函数
        warnings.warn(
            "`post_process` is deprecated and will be removed in v5 of Transformers, please use"
            " `post_process_object_detection` instead, with `threshold=0.` for equivalent results.",
            FutureWarning,
        )

        # 提取模型输出的分类 logits 和预测框 boxes
        logits, boxes = outputs.logits, outputs.pred_boxes

        # 检查 logits 和 target_sizes 的维度是否匹配
        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        # 检查 target_sizes 的每个元素是否包含正确的大小 (h, w)
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # 计算每个预测框的概率 scores、类别 labels
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # 将预测框转换为 [x0, y0, x1, y1] 格式
        boxes = center_to_corners_format(boxes)

        # 将相对坐标 [0, 1] 转换为绝对坐标 [0, height]，其中 height 和 width 分别来自 target_sizes
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        # 构建输出结果列表，每个元素是一个字典包含 scores、labels 和 boxes
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results

    def post_process_object_detection(
        self, outputs, threshold: float = 0.1, target_sizes: Union[TensorType, List[Tuple]] = None
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
        # Extract logits and bounding boxes from model outputs
        logits, boxes = outputs.logits, outputs.pred_boxes

        # Check if target_sizes is provided and validate its length
        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

        # Calculate probabilities and perform sigmoid activation
        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert bounding boxes from center format to corners format [x0, y0, x1, y1]
        boxes = center_to_corners_format(boxes)

        # Convert relative [0, 1] coordinates to absolute [0, height] coordinates if target_sizes is provided
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
            boxes = boxes * scale_fct[:, None, :]

        # Filter predictions based on score threshold and organize results into dictionaries
        results = []
        for s, l, b in zip(scores, labels, boxes):
            score = s[s > threshold]
            label = l[s > threshold]
            box = b[s > threshold]
            results.append({"scores": score, "labels": label, "boxes": box})

        return results

    # TODO: (Amy) Make compatible with other frameworks
```