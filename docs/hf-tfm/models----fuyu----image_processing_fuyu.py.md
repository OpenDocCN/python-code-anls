# `.\models\fuyu\image_processing_fuyu.py`

```py
# coding=utf-8
# 设置编码格式为 UTF-8，确保可以处理包含非英文字符的文本文件

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 版权声明，标明代码的版权归 HuggingFace Inc. 团队所有，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 引入 Apache 许可证 2.0 版本

# you may not use this file except in compliance with the License.
# 在符合许可证规定的情况下，才可以使用本文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则依据许可证提供的软件将按"原样"分发，不附带任何形式的保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证以了解权限和限制

"""Image processor class for Fuyu."""
# 文件描述：Fuyu 的图像处理类

import math
# 导入 math 库，用于数学运算

from typing import Dict, List, Optional, Union
# 导入类型提示所需的模块

import numpy as np
# 导入 NumPy 库，用于处理数组

from ...image_processing_utils import BaseImageProcessor, BatchFeature
# 导入本地的图像处理工具类和批处理特征类

from ...image_transforms import (
    pad,
    resize,
    to_channel_dimension_format,
)
# 从本地图像变换模块中导入 pad, resize, to_channel_dimension_format 函数

from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    validate_preprocess_arguments,
)
# 从本地图像工具模块中导入各种图像处理函数和工具函数

from ...utils import (
    TensorType,
    is_torch_available,
    is_torch_device,
    is_torch_dtype,
    logging,
    requires_backends,
)
# 从本地工具模块导入各种通用工具和 TensorFlow 相关函数

if is_torch_available():
    import torch
# 如果 TensorFlow 可用，则导入 TensorFlow 模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器

def make_list_of_list_of_images(
    images: Union[List[List[ImageInput]], List[ImageInput], ImageInput],
) -> List[List[ImageInput]]:
    # 定义函数 make_list_of_list_of_images，接受一个参数 images，可以是二维图像列表、一维图像列表或单个图像
    if is_valid_image(images):
        # 如果 images 是有效的图像，则返回一个包含 images 的嵌套列表
        return [[images]]

    if isinstance(images, list) and all(isinstance(image, list) for image in images):
        # 如果 images 是二维列表，且每个元素都是列表，则返回 images 本身
        return images

    if isinstance(images, list):
        # 如果 images 是一维列表，则将其中每个元素都转换为图像列表，再返回
        return [make_list_of_images(image) for image in images]

    raise ValueError("images must be a list of list of images or a list of images or an image.")
    # 如果 images 不符合上述条件，则抛出值错误异常，提示 images 参数必须符合指定的类型要求

class FuyuBatchFeature(BatchFeature):
    # 定义 FuyuBatchFeature 类，继承自 BatchFeature 类

    """
    BatchFeature class for Fuyu image processor and processor.

    The outputs dictionary from the processors contains a mix of tensors and lists of tensors.
    """
    # 类的说明文档：用于 Fuyu 图像处理器和处理器的批处理特征类
    # 处理器输出的字典包含张量和张量列表的混合内容
    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
        """
        # 如果 tensor_type 为 None，则直接返回当前对象自身，无需转换
        if tensor_type is None:
            return self

        # 根据 tensor_type 获取对应的判断和转换函数
        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type=tensor_type)

        def _convert_tensor(elem):
            # 如果 elem 已经是 tensor 类型，则直接返回
            if is_tensor(elem):
                return elem
            # 否则将 elem 转换成 tensor 类型并返回
            return as_tensor(elem)

        def _safe_convert_tensor(elem):
            try:
                return _convert_tensor(elem)
            except:  # noqa E722
                # 处理异常情况，根据 key 不同抛出不同的 ValueError
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        # 批量进行 tensor 转换
        for key, value in self.items():
            if isinstance(value, list) and isinstance(value[0], list):
                # 处理二维列表的情况，将其中每个元素转换为 tensor 类型
                self[key] = [[_safe_convert_tensor(elem) for elem in elems] for elems in value]
            elif isinstance(value, list):
                # 处理一维列表的情况，将每个元素转换为 tensor 类型
                self[key] = [_safe_convert_tensor(elem) for elem in value]
            else:
                # 处理单个元素的情况，将其转换为 tensor 类型
                self[key] = _safe_convert_tensor(value)
        # 返回转换后的对象本身
        return self
    def to(self, *args, **kwargs) -> "BatchFeature":
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])  # 要求当前环境支持 torch 库
        import torch  # noqa  # 导入 torch 库，忽略与 PEP 8 格式相关的警告

        new_data = {}  # 初始化一个空字典用于存储转换后的数据
        device = kwargs.get("device")  # 获取 kwargs 中的 device 参数

        # 检查 args 是否包含设备信息或数据类型信息
        if device is None and len(args) > 0:
            # 如果 device 参数为 None 且 args 不为空
            arg = args[0]  # 获取第一个参数
            if is_torch_dtype(arg):
                # 如果第一个参数是 torch 的数据类型
                pass  # 什么都不做
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                # 如果第一个参数是字符串、torch 设备对象或整数，则将其作为设备
                device = arg
            else:
                # 如果参数类型不符合预期，则抛出异常
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")

        def _to(elem):
            # 将元素转换并发送到指定设备
            if torch.is_floating_point(elem):
                # 检查元素是否为浮点数类型
                return elem.to(*args, **kwargs)  # 如果是浮点数，则按 args 和 kwargs 指定的设备和类型进行转换
            if device is not None:
                return elem.to(device=device)  # 如果有指定设备，则将元素发送到该设备

            return elem  # 返回未转换的元素

        # 仅将浮点数张量进行类型转换，以避免 tokenizer 将 `LongTensor` 转换为 `FloatTensor` 的问题
        for k, v in self.items():
            if isinstance(v, list) and isinstance(v[0], list):
                # 如果数据结构是列表的列表
                new_v = []
                for elems in v:
                    new_v.append([_to(elem) for elem in elems])  # 对列表中的每个元素进行转换操作
                new_data[k] = new_v  # 更新转换后的数据到 new_data 中
            elif isinstance(v, list):
                # 如果数据结构是列表
                new_data[k] = [_to(elem) for elem in v]  # 对列表中的每个元素进行转换操作
            else:
                new_data[k] = _to(v)  # 对单个元素进行转换操作
        self.data = new_data  # 更新对象的数据为转换后的数据
        return self  # 返回修改后的 BatchFeature 实例
# 继承自 BaseImageProcessor 类的 FuyuImageProcessor 类，用于处理 FuyuForCausalLM 主体之前的图像处理部分。
class FuyuImageProcessor(BaseImageProcessor):
    """
    This class should handle the image processing part before the main FuyuForCausalLM. In particular, it should
    handle:

    - Processing Images:
        Taking a batch of images as input. If the images are variable-sized, it resizes them based on the desired patch
        dimensions. The image output is always img_h, img_w of (1080, 1920)

        Then, it patches up these images using the patchify_image function.

    - Creating Image Input IDs:
        For each patch, a placeholder ID is given to identify where these patches belong in a token sequence. For
        variable-sized images, each line of patches is terminated with a newline ID.

    - Image Patch Indices:
        For each image patch, the code maintains an index where these patches should be inserted in a token stream.


    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image to `size`.
        size (`Dict[str, int]`, *optional*, defaults to `{"height": 1080, "width": 1920}`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image to `size`.
        padding_value (`float`, *optional*, defaults to 1.0):
            The value to pad the image with.
        padding_mode (`str`, *optional*, defaults to `"constant"`):
            The padding mode to use when padding the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float`, *optional*, defaults to 0.5):
            The mean to use when normalizing the image.
        image_std (`float`, *optional*, defaults to 0.5):
            The standard deviation to use when normalizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image.
        rescale_factor (`float`, *optional*, defaults to `1 / 255`):
            The factor to use when rescaling the image.
        patch_size (`Dict[str, int]`, *optional*, defaults to `{"height": 30, "width": 30}`):
            Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
    """

    # 定义模型输入的名称列表，包括图像、图像输入 ID、图像补丁等信息
    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]
    # 初始化函数，用于设置图像处理器的参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像大小调整的标志
        size: Optional[Dict[str, int]] = None,  # 图像大小的目标尺寸，如果未指定则默认为 {"height": 1080, "width": 1920}
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像调整大小时的重采样方法，默认为双线性插值
        do_pad: bool = True,  # 是否进行图像填充的标志
        padding_value: float = 1.0,  # 填充像素的数值，默认为1.0
        padding_mode: str = "constant",  # 填充像素的模式，默认为常数填充
        do_normalize: bool = True,  # 是否进行图像归一化的标志
        image_mean: Union[float, List[float]] = 0.5,  # 图像归一化的均值，默认为0.5
        image_std: Union[float, List[float]] = 0.5,  # 图像归一化的标准差，默认为0.5
        do_rescale: bool = True,  # 是否进行图像像素值缩放的标志
        rescale_factor: float = 1 / 255,  # 图像像素值缩放的因子，默认为1/255
        patch_size: Optional[Dict[str, int]] = None,  # 图像处理中的补丁尺寸，如果未指定则默认为 {"height": 30, "width": 30}
        **kwargs,  # 其他可能的参数，以字典形式接收
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 初始化各个图像处理器的参数
        self.do_resize = do_resize
        self.size = size if size is not None else {"height": 1080, "width": 1920}
        self.resample = resample
        self.do_pad = do_pad
        self.padding_value = padding_value
        self.padding_mode = padding_mode
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.patch_size = patch_size if patch_size is not None else {"height": 30, "width": 30}
        # 定义有效的图像处理参数键列表，用于验证和筛选参数
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_pad",
            "padding_value",
            "padding_mode",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_rescale",
            "rescale_factor",
            "patch_size",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    def resize_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        # Obtain the height and width of the input image based on its channel dimension format
        image_height, image_width = get_image_size(image, input_data_format)
        
        # Extract the target height and width from the provided size dictionary
        target_height, target_width = size["height"], size["width"]

        # Check if the input image already meets or exceeds the target dimensions
        if image_width <= target_width and image_height <= target_height:
            return image

        # Calculate scaling factors to resize the image while preserving aspect ratio
        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        # Compute new dimensions based on the optimal scaling factor
        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        # Resize the image using the specified parameters
        scaled_image = resize(
            image=image,
            size=(new_height, new_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return scaled_image
    ) -> np.ndarray:
        """
        Pad an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to pad.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            data_format (`ChannelDimension` or `str`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取输入图像的高度和宽度
        image_height, image_width = get_image_size(image, input_data_format)
        # 获取目标填充后的高度和宽度
        target_height, target_width = size["height"], size["width"]
        # 计算上、左、下、右填充的像素数
        padding_top = 0
        padding_left = 0
        padding_bottom = target_height - image_height
        padding_right = target_width - image_width
        # 对图像进行填充操作
        padded_image = pad(
            image,
            padding=((padding_top, padding_bottom), (padding_left, padding_right)),
            mode=mode,  # 填充模式，如常量填充、边缘复制等
            constant_values=constant_values,  # 常量填充的值
            data_format=data_format,  # 输出图像的数据格式
            input_data_format=input_data_format,  # 输入图像的通道维度格式
        )
        # 返回填充后的图像
        return padded_image

    def preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_pad: Optional[bool] = None,
        padding_value: Optional[float] = None,
        padding_mode: Optional[str] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[float] = None,
        image_std: Optional[float] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        patch_size: Optional[Dict[str, int]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        return_tensors: Optional[TensorType] = None,
    def get_num_patches(self, image_height: int, image_width: int, patch_size: Dict[str, int] = None) -> int:
        """
        Calculate number of patches required to encode an image.

        Args:
            image_height (`int`):
                Height of the image.
            image_width (`int`):
                Width of the image.
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
        # 如果未提供 patch_size 参数，则使用对象的默认 patch_size
        patch_size = patch_size if patch_size is not None else self.patch_size
        # 从 patch_size 字典中获取 patch 的高度和宽度
        patch_height, patch_width = self.patch_size["height"], self.patch_size["width"]

        # 检查图像高度是否可以被 patch 高度整除，否则抛出错误
        if image_height % patch_height != 0:
            raise ValueError(f"{image_height=} must be divisible by {patch_height}")
        # 检查图像宽度是否可以被 patch 宽度整除，否则抛出错误
        if image_width % patch_width != 0:
            raise ValueError(f"{image_width=} must be divisible by {patch_width}")

        # 计算每个维度中的 patch 数量
        num_patches_per_dim_h = image_height // patch_height
        num_patches_per_dim_w = image_width // patch_width
        # 计算总的 patch 数量
        num_patches = num_patches_per_dim_h * num_patches_per_dim_w
        return num_patches

    def patchify_image(self, image: "torch.Tensor", patch_size: Optional[Dict[str, int]] = None) -> "torch.Tensor":
        """
        Convert an image into a tensor of patches.

        Args:
            image (`torch.Tensor`):
                Image to convert. Shape: [batch, channels, height, width]
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
        # 确保 torch 被正确导入
        requires_backends(self, ["torch"])
        # 如果未提供 patch_size 参数，则使用对象的默认 patch_size
        patch_size = patch_size if patch_size is not None else self.patch_size
        # 从 patch_size 字典中获取 patch 的高度和宽度
        patch_height, patch_width = patch_size["height"], patch_size["width"]

        # 使用 unfold 方法对图像进行展开操作，按照 patch 的高度和宽度展开
        batch_size, channels, _, _ = image.shape
        unfolded_along_height = image.unfold(2, patch_height, patch_height)
        patches = unfolded_along_height.unfold(3, patch_width, patch_width)
        # 将展开后的 tensor 进行重塑和转置操作，以生成 patches
        patches = patches.contiguous()
        patches = patches.view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 3, 4, 1)
        patches = patches.reshape(batch_size, -1, channels * patch_height * patch_width)
        return patches

    def preprocess_with_tokenizer_info(
        self,
        image_input: "torch.Tensor",
        image_present: "torch.Tensor",
        image_unpadded_h: "torch.Tensor",
        image_unpadded_w: "torch.Tensor",
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
        patch_size: Optional[Dict[str, int]] = None,
    ):
        """
        Preprocess image tensors along with tokenizer information.

        Args:
            image_input (`torch.Tensor`):
                Input image tensor.
            image_present (`torch.Tensor`):
                Present image tensor.
            image_unpadded_h (`torch.Tensor`):
                Unpadded image height tensor.
            image_unpadded_w (`torch.Tensor`):
                Unpadded image width tensor.
            image_placeholder_id (`int`):
                Placeholder ID for the image.
            image_newline_id (`int`):
                Newline ID for the image.
            variable_sized (`bool`):
                Whether the image size varies.
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
        """
        # 这里需要实现函数的详细注释，请根据具体的函数功能添加注释。
        pass
```