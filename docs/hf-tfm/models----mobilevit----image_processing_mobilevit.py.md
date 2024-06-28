# `.\models\mobilevit\image_processing_mobilevit.py`

```py
# coding=utf-8
# 定义文件编码格式为 UTF-8

# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证授权

# you may not use this file except in compliance with the License.
# 除非符合许可证，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则软件依"原样"分发，不附任何明示或暗示的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证获取详细的权限和限制条款

"""Image processor class for MobileViT."""
# MobileViT 的图像处理器类

from typing import Dict, List, Optional, Tuple, Union
# 引入必要的类型提示模块

import numpy as np
# 导入 NumPy 库

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 从图像处理工具中导入基础图像处理器、批量特征和获取尺寸字典函数

from ...image_transforms import flip_channel_order, get_resize_output_image_size, resize, to_channel_dimension_format
# 从图像变换模块导入反转通道顺序、获取调整后图像大小、调整大小和转换通道维度格式函数

from ...image_utils import (
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
# 从图像工具模块导入相关函数和枚举类型

from ...utils import TensorType, is_torch_available, is_torch_tensor, is_vision_available, logging
# 从工具模块导入张量类型、检查是否有 Torch 库、是否为 Torch 张量、是否可用 Vision 模块和日志记录函数

if is_vision_available():
    import PIL
    # 如果 Vision 可用，导入 PIL 库

if is_torch_available():
    import torch
    # 如果 Torch 可用，导入 Torch 库

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象


class MobileViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MobileViT image processor.
    构建 MobileViT 图像处理器类
    """
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`.
            Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Controls the size of the output image after resizing.
            Can be overridden by the `size` parameter in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Defines the resampling filter to use if resizing the image.
            Can be overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
            Can be overridden by the `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
            Can be overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input at the center.
            Can be overridden by the `do_center_crop` parameter in the `preprocess` method.
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 256, "width": 256}`):
            Desired output size `(size["height"], size["width"])` when applying center-cropping.
            Can be overridden by the `crop_size` parameter in the `preprocess` method.
        do_flip_channel_order (`bool`, *optional*, defaults to `True`):
            Whether to flip the color channels from RGB to BGR.
            Can be overridden by the `do_flip_channel_order` parameter in the `preprocess` method.
    """

    # 定义模型输入的名称列表，这里只有一个元素 "pixel_values"
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_flip_channel_order: bool = True,
        **kwargs,
    ):
        # 初始化函数，设定图像预处理参数的默认值和类型，参数都可以在 preprocess 方法中被覆盖
        pass
    # 定义类的初始化方法，继承自父类
    ) -> None:
        # 调用父类的初始化方法，并传递关键字参数
        super().__init__(**kwargs)
        # 如果 size 参数不为 None，则设置为指定值；否则使用默认的 {"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 调用函数 get_size_dict，获取处理后的 size 字典，不强制为正方形
        size = get_size_dict(size, default_to_square=False)
        # 如果 crop_size 参数不为 None，则设置为指定值；否则使用默认的 {"height": 256, "width": 256}
        crop_size = crop_size if crop_size is not None else {"height": 256, "width": 256}
        # 调用函数 get_size_dict，获取处理后的 crop_size 字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 设置类的属性，指示是否执行 resize 操作
        self.do_resize = do_resize
        # 设置类的属性，指定 resize 操作的大小
        self.size = size
        # 设置类的属性，指定 resize 操作的插值方法
        self.resample = resample
        # 设置类的属性，指示是否执行 rescale 操作
        self.do_rescale = do_rescale
        # 设置类的属性，指定 rescale 操作的因子
        self.rescale_factor = rescale_factor
        # 设置类的属性，指示是否执行中心裁剪操作
        self.do_center_crop = do_center_crop
        # 设置类的属性，指定中心裁剪操作的大小
        self.crop_size = crop_size
        # 设置类的属性，指示是否执行通道顺序翻转操作
        self.do_flip_channel_order = do_flip_channel_order
        # 设置类的属性，有效的处理器键列表
        self._valid_processor_keys = [
            "images",
            "segmentation_maps",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_center_crop",
            "crop_size",
            "do_flip_channel_order",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 从 transformers.models.mobilenet_v1.image_processing_mobilenet_v1.MobileNetV1ImageProcessor.resize 复制而来，将 PILImageResampling.BICUBIC 替换为 PILImageResampling.BILINEAR
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        default_to_square = True  # 默认将图像调整为正方形
        if "shortest_edge" in size:  # 如果 `size` 中包含 "shortest_edge"
            size = size["shortest_edge"]  # 将 `size` 调整为最短边的大小
            default_to_square = False  # 不再默认将图像调整为正方形
        elif "height" in size and "width" in size:  # 如果 `size` 中包含 "height" 和 "width"
            size = (size["height"], size["width"])  # 将 `size` 调整为给定的高度和宽度
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")  # 抛出数值错误，要求 `size` 包含 'shortest_edge' 或 'height' 和 'width'

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )  # 获取调整后的图像大小
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )  # 返回调整后的图像

    def flip_channel_order(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Flip the color channels from RGB to BGR or vice versa.

        Args:
            image (`np.ndarray`):
                The image, represented as a numpy array.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        return flip_channel_order(image, data_format=data_format, input_data_format=input_data_format)
        # 调用函数 `flip_channel_order` 对图像颜色通道顺序进行翻转

    def __call__(self, images, segmentation_maps=None, **kwargs):
        """
        Preprocesses a batch of images and optionally segmentation maps.

        Overrides the `__call__` method of the `Preprocessor` class so that both images and segmentation maps can be
        passed in as positional arguments.
        """
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)
        # 调用父类的 `__call__` 方法，预处理一批图像和可选的分割地图
    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool,
        do_rescale: bool,
        do_center_crop: bool,
        do_flip_channel_order: bool,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        rescale_factor: Optional[float] = None,
        crop_size: Optional[Dict[str, int]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # 如果需要进行尺寸调整，则调用 resize 方法
        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        # 如果需要进行尺度重置，则调用 rescale 方法
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        # 如果需要进行中心裁剪，则调用 center_crop 方法
        if do_center_crop:
            image = self.center_crop(image=image, size=crop_size, input_data_format=input_data_format)

        # 如果需要反转通道顺序，则调用 flip_channel_order 方法
        if do_flip_channel_order:
            image = self.flip_channel_order(image, input_data_format=input_data_format)

        # 返回预处理后的图像
        return image

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_flip_channel_order: bool = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # 将输入图像转换为 numpy 数组
        image = to_numpy_array(image)

        # 如果图像已经被缩放且需要重新缩放，则发出警告
        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        # 推断图像通道维度格式，如果未指定则进行推断
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        # 调用 _preprocess 方法进行图像预处理
        image = self._preprocess(
            image=image,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_flip_channel_order=do_flip_channel_order,
            input_data_format=input_data_format,
        )

        # 将图像转换为指定的通道维度格式
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        # 返回处理后的图像
        return image

    def _preprocess_mask(
        self,
        segmentation_map: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        # 这个方法用于预处理分割图像（掩模），与 _preprocess_image 方法类似但不包含缩放和反转通道顺序的选项
        # 这里可以实现相应的分割图像预处理逻辑
        pass
    ) -> np.ndarray:
        """Preprocesses a single mask."""
        # 将分割地图转换为 NumPy 数组，确保数据类型一致
        segmentation_map = to_numpy_array(segmentation_map)
        
        # 如果分割地图的维度为2，则添加通道维度，某些变换需要这样做
        if segmentation_map.ndim == 2:
            added_channel_dim = True
            segmentation_map = segmentation_map[None, ...]  # 在第一个维度上添加一个维度
            input_data_format = ChannelDimension.FIRST  # 设置数据格式为通道维度在第一个位置
        else:
            added_channel_dim = False
            # 如果未指定输入数据格式，则推断通道维度格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(segmentation_map, num_channels=1)

        # 对分割地图进行预处理
        segmentation_map = self._preprocess(
            image=segmentation_map,
            do_resize=do_resize,
            size=size,
            resample=PILImageResampling.NEAREST,
            do_rescale=False,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_flip_channel_order=False,
            input_data_format=input_data_format,
        )
        
        # 如果之前添加了额外的通道维度，则去除它，恢复原始形状
        if added_channel_dim:
            segmentation_map = segmentation_map.squeeze(0)
        
        # 将分割地图转换为 int64 数据类型
        segmentation_map = segmentation_map.astype(np.int64)
        
        # 返回预处理后的分割地图
        return segmentation_map
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
        """
        Converts the output of [`MobileViTForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`MobileViTForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.

        Returns:
            semantic_segmentation: `List[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry of each `torch.Tensor` correspond to a semantic class id.
        """
        # TODO: add support for other frameworks

        # Extract logits from the model outputs
        logits = outputs.logits

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            # Check if the number of logits matches the number of target sizes
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            # Convert target_sizes to numpy array if it's a torch tensor
            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()

            semantic_segmentation = []

            # Iterate over each logits tensor
            for idx in range(len(logits)):
                # Resize logits using bilinear interpolation
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # Extract semantic segmentation map by taking the argmax along the channel dimension
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            # Compute semantic segmentation maps directly from logits
            semantic_segmentation = logits.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        # Return the list of semantic segmentation maps
        return semantic_segmentation
```