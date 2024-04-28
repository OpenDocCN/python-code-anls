# `.\transformers\models\nougat\image_processing_nougat.py`

```py
# 这是 HuggingFace 的 Nougat 图像处理器类的代码
# 导入所需的库和模块
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

"""Image processor class for Nougat."""

from typing import Dict, List, Optional, Union

import numpy as np

# 导入基本的图像处理工具类和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,
    pad,
    resize,
    to_channel_dimension_format,
    to_pil_image,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, logging
from ...utils.import_utils import is_cv2_available, is_vision_available


logger = logging.get_logger(__name__)


if is_cv2_available():
    pass


if is_vision_available():
    import PIL


# 定义 NougatImageProcessor 类，继承自 BaseImageProcessor
class NougatImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Nougat image processor.
    Args:
        do_crop_margin (`bool`, *optional*, defaults to `True`):
            是否裁剪图像边缘。
        do_resize (`bool`, *optional*, defaults to `True`):
            是否将图像的（高度，宽度）尺寸调整为指定的 `size`。可以在 `preprocess` 方法中被 `do_resize` 覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 896, "width": 672}`):
            调整大小后的图像尺寸。可以在 `preprocess` 方法中被 `size` 覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            如果调整图像大小，要使用的重采样滤波器。可以在 `preprocess` 方法中被 `resample` 覆盖。
        do_thumbnail (`bool`, *optional*, defaults to `True`):
            是否使用缩略图方法调整图像大小。
        do_align_long_axis (`bool`, *optional*, defaults to `False`):
            是否通过旋转 90 度来将图像的长轴与 `size` 的长轴对齐。
        do_pad (`bool`, *optional*, defaults to `True`):
            是否将图像填充到批次中最大图像尺寸。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否通过指定的缩放比例 `rescale_factor` 调整图像。可以在 `preprocess` 方法中被 `do_rescale` 参数覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果调整图像大小，要使用的缩放因子。可以在 `preprocess` 方法中被 `rescale_factor` 参数覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化。可以在 `preprocess` 方法中被 `do_normalize` 覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_MEAN`):
            如果归一化图像，要使用的均值。这是一个浮点数或浮点数列表，长度与图像中通道数相同。可以在 `preprocess` 方法中被 `image_mean` 参数覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_DEFAULT_STD`):
            图像的标准差。
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_crop_margin: bool = True,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_thumbnail: bool = True,
        do_align_long_axis: bool = False,
        do_pad: bool = True,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    # 初始化图像预处理类
    def __init__(
        self,
        do_crop_margin: bool = True,
        do_resize: bool = True,
        size: Optional[Union[int, Dict[str, int]]] = None,
        resample: Resampling = Resampling.BICUBIC,
        do_thumbnail: bool = False,
        do_align_long_axis: bool = False,
        do_pad: bool = False,
        do_rescale: bool = False,
        rescale_factor: float = 1.0,
        do_normalize: bool = False,
        image_mean: Optional[Union[float, Tuple[float, ...]]] = None,
        image_std: Optional[Union[float, Tuple[float, ...]]] = None,
        **kwargs
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
    
        # 如果未指定尺寸，则设置默认尺寸
        size = size if size is not None else {"height": 896, "width": 672}
        # 获取尺寸字典
        size = get_size_dict(size)
    
        # 设置各种图像预处理操作的标志
        self.do_crop_margin = do_crop_margin
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_thumbnail = do_thumbnail
        self.do_align_long_axis = do_align_long_axis
        self.do_pad = do_pad
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        # 设置图像均值和标准差
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
    
    # 实现findNonZero函数的Python版本
    def python_find_non_zero(self, image: np.array):
        """This is a reimplementation of a findNonZero function equivalent to cv2."""
        # 找到图像中非零元素的索引
        non_zero_indices = np.column_stack(np.nonzero(image))
        # 交换行列索引
        idxvec = non_zero_indices[:, [1, 0]]
        # 将结果整理为(n, 1, 2)的形状
        idxvec = idxvec.reshape(-1, 1, 2)
        # 返回结果
        return idxvec
    
    # 实现BoundingRect函数的Python版本
    def python_bounding_rect(self, coordinates):
        """This is a reimplementation of a BoundingRect function equivalent to cv2."""
        # 找到坐标中的最小值和最大值
        min_values = np.min(coordinates, axis=(0, 1)).astype(int)
        max_values = np.max(coordinates, axis=(0, 1)).astype(int)
        # 计算边界框的左上角坐标和宽高
        x_min, y_min = min_values[0], min_values[1]
        width = max_values[0] - x_min + 1
        height = max_values[1] - y_min + 1
        # 返回边界框参数
        return x_min, y_min, width, height
    
    # 裁剪图像边缘的方法
    def crop_margin(
        self,
        image: np.array,
        gray_threshold: int = 200,
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    # 定义一个方法，用于裁剪图像边缘，灰色像素被视为边缘（即值低于阈值的像素）
    def crop_margin(
            self,
            image: np.array,
            gray_threshold: int = 200,
            data_format: ChannelDimension = None,
            input_data_format: ChannelDimension = None
    ) -> np.array:
            # 如果输入数据格式为空，则从图像推断通道维度格式
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(image)
    
            # 将图像转换为 PIL 图像，指定输入数据格式
            image = to_pil_image(image, input_data_format=input_data_format)
            # 将 PIL 图像转换为灰度图像数组，并转换为 uint8 类型
            data = np.array(image.convert("L")).astype(np.uint8)
            # 获取数组中的最大最小值
            max_val = data.max()
            min_val = data.min()
            # 如果最大值等于最小值，返回原始图像
            if max_val == min_val:
                image = np.array(image)
                # 如果数据格式已指定，转换为指定数据格式
                # 否则保持原始格式
                image = (
                    to_channel_dimension_format(image, data_format, input_data_format)
                    if data_format is not None
                    else image
                )
                return image
            # 对数据进行归一化以便裁剪
            data = (data - min_val) / (max_val - min_val) * 255
            # 将灰度值小于阈值的像素设置为 True
            gray = data < gray_threshold
            # 查找灰度值不为零的像素坐标
            coords = self.python_find_non_zero(gray)
            # 计算包含所有非零像素的最小矩形区域
            x_min, y_min, width, height = self.python_bounding_rect(coords)
            # 使用最小矩形区域裁剪图像
            image = image.crop((x_min, y_min, x_min + width, y_min + height))
            # 将裁剪后的图像转换为数组，并转换为 uint8 类型
            image = np.array(image).astype(np.uint8)
            # 将图像的数据格式转换为输入数据格式的通道维度格式
            image = to_channel_dimension_format(image, input_data_format, ChannelDimension.LAST)
    
            # 如果已指定数据格式，将图像转换为指定数据格式
            # 如果未指定，保持原始格式
            image = (
                to_channel_dimension_format(image, data_format, input_data_format) if data_format is not None else image
            )
    
            return image
    
        # 从 transformers.models.donut.image_processing_donut.DonutImageProcessor 复制的 align_long_axis 方法
        def align_long_axis(
            self,
            image: np.ndarray,
            size: Dict[str, int],
            data_format: Optional[Union[str, ChannelDimension]] = None,
            input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Align the long axis of the image to the longest axis of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be aligned.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to align the long axis to.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.

        Returns:
            `np.ndarray`: The aligned image.
        """
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 获取输出图像的高度和宽度
        output_height, output_width = size["height"], size["width"]

        # 检查是否需要旋转图像以匹配输出的长轴
        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            # 如果需要，将图像逆时针旋转90度
            image = np.rot90(image, 3)

        # 如果指定了输出数据格式，则转换图像的通道维度格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        # 返回对齐后的图像
        return image

    def pad_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad the image to the specified size at the top, bottom, left and right.

        Args:
            image (`np.ndarray`):
                The image to be padded.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取输出图像的高度和宽度
        output_height, output_width = size["height"], size["width"]
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        # 计算宽度和高度的差值
        delta_width = output_width - input_width
        delta_height = output_height - input_height

        # 计算上、下、左、右各自的填充量
        pad_top = delta_height // 2
        pad_left = delta_width // 2
        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        # 构造填充元组
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        # 执行填充操作
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format)

    # Copied from transformers.models.donut.image_processing_donut.DonutImageProcessor.thumbnail
    # 缩略图生成函数，将图像调整大小以生成缩略图，调整后的图像的任何尺寸都不会大于指定尺寸的对应尺寸。
    def thumbnail(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Args:
            image (`np.ndarray`): 图像数组，待调整大小的图像。
            size (`Dict[str, int]`): 调整后的尺寸 `{"height": h, "width": w}`。
            resample (`PILImageResampling`, *optional*, 默认为 `PILImageResampling.BICUBIC`):
                使用的重采样滤波器。
            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                输出图像的数据格式。如果未设置，则使用与输入图像相同的格式。
            input_data_format (`ChannelDimension` or `str`, *optional*):
                输入图像的通道维度格式。如果未提供，则将进行推断。
        """
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 获取输出的高度和宽度
        output_height, output_width = size["height"], size["width"]

        # 我们始终将大小调整为输入尺寸和输出尺寸中的最小值。
        height = min(input_height, output_height)
        width = min(input_width, output_width)

        # 如果输入图像的高度和宽度与要调整的高度和宽度相等，则直接返回输入图像
        if height == input_height and width == input_width:
            return image

        # 如果输入图像的高度大于宽度，则根据比例调整宽度
        if input_height > input_width:
            width = int(input_width * height / input_height)
        # 如果输入图像的宽度大于高度，则根据比例调整高度
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        # 调整图像大小
        return resize(
            image,
            size=(height, width),
            resample=resample,
            reducing_gap=2.0,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 从 transformers.models.donut.image_processing_donut.DonutImageProcessor.resize 复制而来的函数
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        定义一个函数，将输入的图像`image`调整大小到指定的`(height, width)`尺寸，使用PIL库进行操作。

        Args:
            image (`np.ndarray`):
                需要调整大小的图像。
            size (`Dict[str, int]`):
                输出图像的尺寸。
            resample (`PILImageResampling`, *可选*, 默认为`PILImageResampling.BICUBIC`):
                调整图像时使用的重采样滤波器。
            data_format (`str` or `ChannelDimension`, *可选*):
                图像的通道维度格式。如果未提供，将与输入图像相同。
            input_data_format (`ChannelDimension` or `str`, *可选*):
                输入图像的通道维度格式。如果未提供，将被推断。
        """
        size = get_size_dict(size)
        shortest_edge = min(size["height"], size["width"])
        output_size = get_resize_output_image_size(
            image, size=shortest_edge, default_to_square=False, input_data_format=input_data_format
        )
        resized_image = resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return resized_image

    def preprocess(
        self,
        images: ImageInput,
        do_crop_margin: bool = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        do_rescale: bool = None,
        rescale_factor: Union[int, float] = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```