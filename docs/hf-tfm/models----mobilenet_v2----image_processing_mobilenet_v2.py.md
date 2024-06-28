# `.\models\mobilenet_v2\image_processing_mobilenet_v2.py`

```py
# coding=utf-8
# 设置文件编码为 UTF-8

# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证进行许可

# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下链接获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 根据许可证，软件按"原样"分发，不提供任何明示或暗示的保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可证，了解特定语言的权限和限制

"""Image processor class for MobileNetV2."""
# MobileNetV2 图像处理器类

from typing import Dict, List, Optional, Tuple, Union

import numpy as np  # 导入 NumPy 库

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入基本图像处理工具，批处理特征，获取大小字典函数

from ...image_transforms import (
    get_resize_output_image_size,  # 导入获取调整大小后图像尺寸的函数
    resize,  # 导入调整大小的函数
    to_channel_dimension_format,  # 导入转换通道维度格式的函数
)

from ...image_utils import (
    IMAGENET_STANDARD_MEAN,  # 导入 ImageNet 标准均值
    IMAGENET_STANDARD_STD,  # 导入 ImageNet 标准标准差
    ChannelDimension,  # 导入通道维度类
    ImageInput,  # 导入图像输入类
    PILImageResampling,  # 导入 PIL 图像重采样枚举
    infer_channel_dimension_format,  # 导入推断通道维度格式的函数
    is_scaled_image,  # 导入检查是否为缩放图像的函数
    make_list_of_images,  # 导入生成图像列表的函数
    to_numpy_array,  # 导入转换为 NumPy 数组的函数
    valid_images,  # 导入验证图像函数
    validate_kwargs,  # 导入验证关键字参数的函数
    validate_preprocess_arguments,  # 导入验证预处理参数的函数
)

from ...utils import TensorType, is_torch_available, is_torch_tensor, logging
# 导入张量类型，检查是否有 Torch 可用，检查是否为 Torch 张量，日志记录函数

if is_torch_available():  # 如果 Torch 可用
    import torch  # 导入 Torch 库

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class MobileNetV2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MobileNetV2 image processor.
    构建一个 MobileNetV2 图像处理器。
    """
    # 定义函数的参数和默认值，用于图像预处理
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的高度和宽度尺寸到指定的 `size`。可以在 `preprocess` 方法中通过 `do_resize` 参数进行覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 256}`):
            调整后的图像尺寸。图像的最短边被调整为 `size["shortest_edge"]`，保持输入的宽高比。可以在 `preprocess` 方法中通过 `size` 参数进行覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            调整图像尺寸时使用的重采样滤波器。可以在 `preprocess` 方法中通过 `resample` 参数进行覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            是否对图像进行中心裁剪。如果输入尺寸小于任何边缘的 `crop_size`，则用 0 填充图像，然后进行中心裁剪。可以在 `preprocess` 方法中通过 `do_center_crop` 参数进行覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            应用中心裁剪时的期望输出尺寸。仅在 `do_center_crop` 设置为 `True` 时生效。可以在 `preprocess` 方法中通过 `crop_size` 参数进行覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定的比例因子 `rescale_factor` 对图像进行重新缩放。可以在 `preprocess` 方法中通过 `do_rescale` 参数进行覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新缩放图像时使用的缩放因子。可以在 `preprocess` 方法中通过 `rescale_factor` 参数进行覆盖。
        do_normalize:
            是否对图像进行归一化。可以在 `preprocess` 方法中通过 `do_normalize` 参数进行覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            归一化图像时使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在 `preprocess` 方法中通过 `image_mean` 参数进行覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            归一化图像时使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在 `preprocess` 方法中通过 `image_std` 参数进行覆盖。
    """

    # 定义模型输入的名称列表
    model_input_names = ["pixel_values"]
    # 初始化方法，设置图像处理器的各种参数和默认值
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行大小调整，默认为True
        size: Optional[Dict[str, int]] = None,  # 图像大小的字典，可选，默认为None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        do_center_crop: bool = True,  # 是否进行中心裁剪，默认为True
        crop_size: Dict[str, int] = None,  # 裁剪尺寸的字典，可选，默认为None
        do_rescale: bool = True,  # 是否进行重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否进行归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，可选，默认为None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，可选，默认为None
        **kwargs,  # 其他参数
    ) -> None:
        super().__init__(**kwargs)  # 调用父类的初始化方法
        size = size if size is not None else {"shortest_edge": 256}  # 如果size为None，则设置默认最短边为256
        size = get_size_dict(size, default_to_square=False)  # 根据size字典获取图像尺寸的字典，不默认为正方形
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}  # 如果crop_size为None，则设置默认裁剪尺寸为224x224
        crop_size = get_size_dict(crop_size, param_name="crop_size")  # 根据crop_size字典获取裁剪尺寸的字典
        self.do_resize = do_resize  # 设置是否进行大小调整的属性
        self.size = size  # 设置图像大小的属性
        self.resample = resample  # 设置重采样方法的属性
        self.do_center_crop = do_center_crop  # 设置是否进行中心裁剪的属性
        self.crop_size = crop_size  # 设置裁剪尺寸的属性
        self.do_rescale = do_rescale  # 设置是否进行重新缩放的属性
        self.rescale_factor = rescale_factor  # 设置重新缩放因子的属性
        self.do_normalize = do_normalize  # 设置是否进行归一化的属性
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 设置图像均值的属性，如果为None则使用预设值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 设置图像标准差的属性，如果为None则使用预设值
        self._valid_processor_keys = [
            "images",  # 图像关键字
            "do_resize",  # 是否进行大小调整的关键字
            "size",  # 图像大小的关键字
            "resample",  # 重采样方法的关键字
            "do_center_crop",  # 是否进行中心裁剪的关键字
            "crop_size",  # 裁剪尺寸的关键字
            "do_rescale",  # 是否进行重新缩放的关键字
            "rescale_factor",  # 重新缩放因子的关键字
            "do_normalize",  # 是否进行归一化的关键字
            "image_mean",  # 图像均值的关键字
            "image_std",  # 图像标准差的关键字
            "return_tensors",  # 返回张量的关键字
            "data_format",  # 数据格式的关键字
            "input_data_format",  # 输入数据格式的关键字
        ]

    # 从transformers.models.mobilenet_v1.image_processing_mobilenet_v1.MobileNetV1ImageProcessor.resize复制而来
    def resize(
        self,
        image: np.ndarray,  # 输入图像的numpy数组
        size: Dict[str, int],  # 目标尺寸的字典
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，可选，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可选，默认为None
        **kwargs,  # 其他参数
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 默认将图像调整为正方形
        default_to_square = True
        # 如果输入的尺寸字典中包含 "shortest_edge" 键
        if "shortest_edge" in size:
            # 将 size 重置为 shortest_edge 的值
            size = size["shortest_edge"]
            # 取消默认将图像调整为正方形的设置
            default_to_square = False
        # 如果输入的尺寸字典中同时包含 "height" 和 "width" 键
        elif "height" in size and "width" in size:
            # 将 size 重置为 (height, width) 的元组
            size = (size["height"], size["width"])
        else:
            # 如果尺寸字典中既没有 "shortest_edge" 也没有同时包含 "height" 和 "width"，则抛出数值错误
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        # 获取调整后的输出图像尺寸
        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        # 返回调整大小后的图像
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocesses images according to specified operations.

        Args:
            images (`ImageInput`): Input images to preprocess.
            do_resize (`bool`, *optional*): Whether to resize the images.
            size (`Dict[str, int]`, *optional*): Target size of the images after resizing.
            resample (`PILImageResampling`, *optional*): Resampling filter for resizing.
            do_center_crop (`bool`, *optional*): Whether to perform center cropping.
            crop_size (`Dict[str, int]`, *optional*): Size of the crop.
            do_rescale (`bool`, *optional*): Whether to rescale the images.
            rescale_factor (`float`, *optional*): Scaling factor for rescaling.
            do_normalize (`bool`, *optional*): Whether to normalize the images.
            image_mean (`float` or `List[float]`, *optional*): Mean values for normalization.
            image_std (`float` or `List[float]`, *optional*): Standard deviation values for normalization.
            return_tensors (`str` or `TensorType`, *optional*): Desired tensor type for output.
            data_format (`str` or `ChannelDimension`): Channel dimension format of the images.
            input_data_format (`str` or `ChannelDimension`, *optional*): Channel dimension format of the input images.
            **kwargs: Additional keyword arguments.

        Returns:
            Preprocessed images according to the specified operations.
        """
        # 此处省略了具体的实现内容，根据函数定义，该方法对输入的图像进行预处理，并根据参数执行相应的操作。
        # 具体的预处理操作包括但不限于调整大小、中心裁剪、重新缩放、归一化等。
        pass
    def post_process_semantic_segmentation(self, outputs, target_sizes: List[Tuple] = None):
        """
        Converts the output of [`MobileNetV2ForSemanticSegmentation`] into semantic segmentation maps. Only supports PyTorch.

        Args:
            outputs ([`MobileNetV2ForSemanticSegmentation`]):
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

        # 获取输出中的 logits
        logits = outputs.logits

        # 如果指定了目标大小，则调整 logits 并计算语义分割图
        if target_sizes is not None:
            # 检查 logits 的数量与目标大小列表的长度是否一致
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            # 如果 target_sizes 是 torch tensor，则转换为 numpy 数组
            if is_torch_tensor(target_sizes):
                target_sizes = target_sizes.numpy()

            # 初始化语义分割结果列表
            semantic_segmentation = []

            # 遍历每个 logits
            for idx in range(len(logits)):
                # 使用双线性插值调整 logits 的尺寸
                resized_logits = torch.nn.functional.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                # 获取调整大小后的语义分割图
                semantic_map = resized_logits[0].argmax(dim=0)
                # 将语义分割图添加到结果列表中
                semantic_segmentation.append(semantic_map)
        else:
            # 如果未指定目标大小，则直接计算 logits 的每个样本的语义分割图
            semantic_segmentation = logits.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        # 返回语义分割结果列表
        return semantic_segmentation
```