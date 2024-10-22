# `.\diffusers\pipelines\blip_diffusion\blip_image_processing.py`

```py
# coding=utf-8  # 指定源代码文件的编码为 UTF-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.  # 版权声明，标明版权归 HuggingFace Inc. 团队所有
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 指明该文件遵循 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 除非遵守许可证，否则不得使用该文件
# You may obtain a copy of the License at  # 说明如何获取许可证
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证的 URL
#
# Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
# distributed under the License is distributed on an "AS IS" BASIS,  # 根据许可证分发的软件按“原样”提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何形式的保证或条件
# See the License for the specific language governing permissions and  # 查看许可证以了解有关权限的具体语言
# limitations under the License.  # 以及许可证下的限制
"""Image processor class for BLIP."""  # 该模块是 BLIP 图像处理器类的定义

from typing import Dict, List, Optional, Union  # 导入用于类型注解的模块

import numpy as np  # 导入 NumPy 库，通常用于数组和矩阵操作
import torch  # 导入 PyTorch 库，用于深度学习
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 从 transformers 导入图像处理相关的基类和工具函数
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format  # 导入图像转换函数
from transformers.image_utils import (  # 导入图像工具函数
    OPENAI_CLIP_MEAN,  # OpenAI CLIP 的均值
    OPENAI_CLIP_STD,  # OpenAI CLIP 的标准差
    ChannelDimension,  # 通道维度相关的定义
    ImageInput,  # 图像输入类型定义
    PILImageResampling,  # PIL 图像重采样功能
    infer_channel_dimension_format,  # 推断通道维度格式的函数
    is_scaled_image,  # 判断是否为缩放图像的函数
    make_list_of_images,  # 将图像转换为图像列表的函数
    to_numpy_array,  # 将数据转换为 NumPy 数组的函数
    valid_images,  # 检查有效图像的函数
)
from transformers.utils import TensorType, is_vision_available, logging  # 导入工具函数和类型定义

from diffusers.utils import numpy_to_pil  # 从 diffusers 导入将 NumPy 数组转换为 PIL 图像的函数


if is_vision_available():  # 如果视觉库可用
    import PIL.Image  # 导入 PIL 图像处理库


logger = logging.get_logger(__name__)  # 创建一个日志记录器，用于记录当前模块的日志


# We needed some extra functions on top of the ones in transformers.image_processing_utils.BaseImageProcessor, namely center crop
# Copy-pasted from transformers.models.blip.image_processing_blip.BlipImageProcessor  # 说明该类在 transformers.image_processing_utils.BaseImageProcessor 的基础上增加了一些额外的功能，如中心裁剪，且复制自 BLIP 图像处理器
class BlipImageProcessor(BaseImageProcessor):  # 定义 BlipImageProcessor 类，继承自 BaseImageProcessor
    r"""  # 开始文档字符串，描述该类的用途
    Constructs a BLIP image processor.  # 构造一个 BLIP 图像处理器
    # 参数说明文档
    Args:
        # 是否调整图像的（高度，宽度）尺寸到指定的 `size`，可通过 `preprocess` 方法中的 `do_resize` 参数覆盖
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        # 输出图像调整大小后的尺寸，默认为 {"height": 384, "width": 384}，可通过 `preprocess` 方法中的 `size` 参数覆盖
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        # 如果调整图像大小，使用的重采样滤波器，仅在 `do_resize` 设置为 True 时有效，且可通过 `preprocess` 方法中的 `resample` 参数覆盖
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        # 是否通过指定的缩放因子 `rescale_factor` 对图像进行重新缩放，默认为 True，可通过 `preprocess` 方法中的 `do_rescale` 参数覆盖
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        # 如果对图像进行重新缩放时使用的缩放因子，仅在 `do_rescale` 设置为 True 时有效，且可通过 `preprocess` 方法中的 `rescale_factor` 参数覆盖
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        # 是否对图像进行归一化处理，默认为 True，可通过 `preprocess` 方法中的 `do_normalize` 参数覆盖
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        # 归一化图像时使用的均值，可以是一个浮点数或浮点数列表，其长度与图像通道数相等，可通过 `preprocess` 方法中的 `image_mean` 参数覆盖
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        # 归一化图像时使用的标准差，可以是一个浮点数或浮点数列表，其长度与图像通道数相等，可通过 `preprocess` 方法中的 `image_std` 参数覆盖
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        # 是否将图像转换为 RGB 格式
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """  # 文档字符串结束

    # 定义模型输入的名称列表，包含 "pixel_values"
    model_input_names = ["pixel_values"]
    # 初始化方法，用于设置类的基本属性
        def __init__(
            self,
            do_resize: bool = True,  # 是否进行图像缩放，默认为 True
            size: Dict[str, int] = None,  # 图像尺寸，默认为 None
            resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
            do_rescale: bool = True,  # 是否进行像素值重缩放，默认为 True
            rescale_factor: Union[int, float] = 1 / 255,  # 像素值重缩放因子，默认为 1/255
            do_normalize: bool = True,  # 是否对图像进行归一化处理，默认为 True
            image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，默认为 None
            image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，默认为 None
            do_convert_rgb: bool = True,  # 是否将图像转换为 RGB 格式，默认为 True
            do_center_crop: bool = True,  # 是否进行中心裁剪，默认为 True
            **kwargs,  # 其他可选参数
        ) -> None:
            super().__init__(**kwargs)  # 调用父类初始化方法，传入其他参数
            size = size if size is not None else {"height": 224, "width": 224}  # 如果 size 为 None，则设置为默认尺寸
            size = get_size_dict(size, default_to_square=True)  # 获取尺寸字典，默认转换为正方形
    
            self.do_resize = do_resize  # 设置实例属性 do_resize
            self.size = size  # 设置实例属性 size
            self.resample = resample  # 设置实例属性 resample
            self.do_rescale = do_rescale  # 设置实例属性 do_rescale
            self.rescale_factor = rescale_factor  # 设置实例属性 rescale_factor
            self.do_normalize = do_normalize  # 设置实例属性 do_normalize
            self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN  # 设置实例属性 image_mean，默认使用 OPENAI_CLIP_MEAN
            self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD  # 设置实例属性 image_std，默认使用 OPENAI_CLIP_STD
            self.do_convert_rgb = do_convert_rgb  # 设置实例属性 do_convert_rgb
            self.do_center_crop = do_center_crop  # 设置实例属性 do_center_crop
    
        # 从 transformers.models.vit.image_processing_vit.ViTImageProcessor.resize 复制而来，重采样方法由 PILImageResampling.BILINEAR 修改为 PILImageResampling.BICUBIC
        def resize(
            self,
            image: np.ndarray,  # 输入图像，类型为 numpy.ndarray
            size: Dict[str, int],  # 指定的新尺寸，类型为字典
            resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
            data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，默认为 None
            input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为 None
            **kwargs,  # 其他可选参数
    ) -> np.ndarray:  # 指定该函数返回一个 numpy 数组
        """  # 开始函数文档字符串
        Resize an image to `(size["height"], size["width"])`.  # 描述函数功能：调整图像大小
        Args:  # 参数说明部分
            image (`np.ndarray`):  # 输入参数：待调整大小的图像，类型为 numpy 数组
                Image to resize.  # 图像的说明
            size (`Dict[str, int]`):  # 输入参数：字典，包含目标图像的高度和宽度
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.  # 字典格式的描述
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):  # 可选参数：指定重采样的方法
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.  # 重采样过滤器的说明
            data_format (`ChannelDimension` or `str`, *optional*):  # 可选参数：输出图像的通道维度格式
                The channel dimension format for the output image. If unset, the channel dimension format of the input  # 描述输入图像通道格式的使用
                image is used. Can be one of:  # 可能的通道格式选项
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.  # 第一种格式的说明
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.  # 第二种格式的说明
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.  # 第三种格式的说明
            input_data_format (`ChannelDimension` or `str`, *optional*):  # 可选参数：输入图像的通道维度格式
                The channel dimension format for the input image. If unset, the channel dimension format is inferred  # 描述输入图像通道格式的推断
                from the input image. Can be one of:  # 可能的输入格式选项
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.  # 第一种格式的说明
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.  # 第二种格式的说明
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.  # 第三种格式的说明
        Returns:  # 返回值说明部分
            `np.ndarray`: The resized image.  # 返回一个调整大小后的 numpy 数组图像
        """  # 结束函数文档字符串
        size = get_size_dict(size)  # 获取标准化的大小字典
        if "height" not in size or "width" not in size:  # 检查字典中是否包含 'height' 和 'width' 键
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")  # 抛出错误，提示缺少必要的键
        output_size = (size["height"], size["width"])  # 根据 size 字典获取输出图像的尺寸元组
        return resize(  # 调用 resize 函数进行图像调整大小
            image,  # 传入待调整大小的图像
            size=output_size,  # 传入目标大小
            resample=resample,  # 传入重采样选项
            data_format=data_format,  # 传入数据格式选项
            input_data_format=input_data_format,  # 传入输入数据格式选项
            **kwargs,  # 传入其他关键字参数
        )  # 返回调整大小后的图像

    def preprocess(  # 定义 preprocess 函数
        self,  # 类实例本身
        images: ImageInput,  # 输入参数：待处理的图像，类型为 ImageInput
        do_resize: Optional[bool] = None,  # 可选参数：是否执行调整大小操作
        size: Optional[Dict[str, int]] = None,  # 可选参数：调整大小时的目标尺寸
        resample: PILImageResampling = None,  # 可选参数：重采样过滤器
        do_rescale: Optional[bool] = None,  # 可选参数：是否执行重新缩放操作
        do_center_crop: Optional[bool] = None,  # 可选参数：是否执行中心裁剪操作
        rescale_factor: Optional[float] = None,  # 可选参数：重新缩放的比例因子
        do_normalize: Optional[bool] = None,  # 可选参数：是否执行归一化操作
        image_mean: Optional[Union[float, List[float]]] = None,  # 可选参数：图像的均值，用于归一化
        image_std: Optional[Union[float, List[float]]] = None,  # 可选参数：图像的标准差，用于归一化
        return_tensors: Optional[Union[str, TensorType]] = None,  # 可选参数：指定返回的张量类型
        do_convert_rgb: bool = None,  # 可选参数：是否将图像转换为 RGB 格式
        data_format: ChannelDimension = ChannelDimension.FIRST,  # 指定数据格式，默认为通道优先
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 可选参数：输入图像的通道格式
        **kwargs,  # 传入其他关键字参数
    # 继承自 diffusers.VaeImageProcessor.postprocess 方法
    def postprocess(self, sample: torch.Tensor, output_type: str = "pil"):
        # 检查输出类型是否在支持的类型列表中
        if output_type not in ["pt", "np", "pil"]:
            # 如果不在列表中，抛出一个值错误
            raise ValueError(
                f"output_type={output_type} is not supported. Make sure to choose one of ['pt', 'np', or 'pil']"
            )
    
        # 等价于 diffusers.VaeImageProcessor.denormalize，将样本归一化到 [0, 1] 范围
        sample = (sample / 2 + 0.5).clamp(0, 1)
        # 如果输出类型是 'pt'，直接返回处理后的样本
        if output_type == "pt":
            return sample
    
        # 等价于 diffusers.VaeImageProcessor.pt_to_numpy，将样本从 PyTorch 张量转换为 NumPy 数组
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        # 如果输出类型是 'np'，返回 NumPy 数组
        if output_type == "np":
            return sample
        # 否则，输出类型必须是 'pil'
        sample = numpy_to_pil(sample)
        # 返回 PIL 图像对象
        return sample
```