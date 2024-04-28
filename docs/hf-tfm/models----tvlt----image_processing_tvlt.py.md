# `.\transformers\models\tvlt\image_processing_tvlt.py`

```
# 定义编码格式为 utf-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证，只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 只有在适用法律要求或书面同意的情况下，才能基于此许可证分发软件
# 软件按"原样"的基础分发，不提供任何明示或暗示的担保或条件
# 请查看许可证了解具体的语言和限制条件

# 引入所需的库
from typing import Dict, List, Optional, Union
import numpy as np

# 引入项目中的其它模块和函数
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import get_resize_output_image_size, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ChannelDimension, ImageInput,
    PILImageResampling, infer_channel_dimension_format, is_scaled_image, is_valid_image,
    to_numpy_array, valid_images
)
from ...utils import TensorType, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义批处理视频的函数
def make_batched(videos) -> List[List[ImageInput]]:
    # 如果 videos 是列表或元组，并且第一个元素也是列表或元组，则返回 videos
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)):
        return videos
    # 如果 videos 是列表或元组，并且第一个元素是有效图像，则根据维度返回 videos
    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        videos_dim = np.array(videos[0]).ndim
        if videos_dim == 3:
            return [videos]
        elif videos_dim == 4:
            return videos
    # 如果 videos 是有效图像，则根据维度返回 videos
    elif is_valid_image(videos):
        videos_dim = np.array(videos).ndim
        if videos_dim == 3:
            return [[videos]]
        elif videos_dim == 4:
            return [videos]
        elif videos_dim == 5:
            return videos
    # 抛出异常，表示无法从 videos 创建批处理视频
    raise ValueError(f"Could not make batched video from {videos}")

# TVLT 图像处理器类，继承自 BaseImageProcessor
class TvltImageProcessor(BaseImageProcessor):
    r"""
    构造一个 TVLT 图像处理器。

    此处理器可用于通过将图像转换为 1 帧视频来准备模型要使用的视频或图像。
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否将图像的（高度，宽度）尺寸调整为指定的 `size`。可以通过 `preprocess` 方法中的 `do_resize` 参数进行覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            调整大小后的输出图像尺寸。图像的最短边将调整为 `size["shortest_edge"]`，同时保持原始图像的纵横比。可以通过
            `preprocess` 方法中的 `size` 进行覆盖。
        patch_size (`List[int]` *optional*, defaults to [16,16]):
            图像补丁嵌入的补丁大小。
        num_frames (`int` *optional*, defaults to 8):
            视频帧的最大数量。
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            如果调整图像大小，则要使用的重采样滤波器。可以通过 `preprocess` 方法中的 `resample` 参数进行覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            是否对图像进行中心裁剪至指定的 `crop_size`。可以通过 `preprocess` 方法中的 `do_center_crop` 参数进行覆盖。
        crop_size (`Dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            应用中心裁剪后的图像尺寸。可以通过 `preprocess` 方法中的 `crop_size` 参数进行覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定比例 `rescale_factor` 对图像进行重新缩放。可以通过 `preprocess` 方法中的 `do_rescale` 参数进行覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to 1/255):
            如果重新缩放图像，则定义要使用的比例因子。可以通过 `preprocess` 方法中的 `rescale_factor` 参数进行覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化。可以通过 `preprocess` 方法中的 `do_normalize` 参数进行覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果对图像进行归一化，则使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以通过 `preprocess` 方法中的 `image_mean` 参数进行覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果对图像进行归一化，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以通过 `preprocess` 方法中的 `image_std` 参数进行覆盖。
    """

    model_input_names = [
        "pixel_values",
        "pixel_mask",
        "pixel_values_mixed",
        "pixel_mask_mixed",
    ]
    # 初始化方法，接受多个参数并设置默认值
    def __init__(
        self,
        # 是否执行大小调整
        do_resize: bool = True,
        # 图像大小的字典
        size: Dict[str, int] = None,
        # patch 的大小
        patch_size: List[int] = [16, 16],
        # 帧数量
        num_frames: int = 8,
        # 重取样方法
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        # 是否执行中心裁剪
        do_center_crop: bool = True,
        # 裁剪后的大小
        crop_size: Dict[str, int] = None,
        # 是否重新缩放
        do_rescale: bool = True,
        # 重新缩放因子
        rescale_factor: Union[int, float] = 1 / 255,
        # 是否标准化
        do_normalize: bool = True,
        # 图像均值
        image_mean: Optional[Union[float, List[float]]] = IMAGENET_STANDARD_MEAN,
        # 图像标准差
        image_std: Optional[Union[float, List[float]]] = IMAGENET_STANDARD_STD,
        # 是否初始化掩模生成器
        init_mask_generator=False,
        **kwargs,
    ) -> None:
        # 调用父类的初始化方法，传入关键字参数
        super().__init__(**kwargs)
        # 如果 size 为 None，则设置默认值为 {"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 根据输入参数生成标准的 size 字典
        size = get_size_dict(size, default_to_square=False)
        # 如果 crop_size 为 None，则设置默认值为 {"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 根据输入参数生成标准的 crop_size 字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 设置各个属性的初始值
        self.do_resize = do_resize
        self.size = size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    # 图像调整大小的方法
    def resize(
        self,
        # 输入图像数组
        image: np.ndarray,
        # 调整后的大小字典
        size: Dict[str, int],
        # 重取样方法
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        # 数据格式参数
        data_format: Optional[Union[str, ChannelDimension]] = None,
        # 输入数据格式参数
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```  
    ) -> np.ndarray:
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image. If `size` is of the form `{"height": h, "width": w}`, the output image will
                have the size `(h, w)`. If `size` is of the form `{"shortest_edge": s}`, the output image will have its
                shortest edge of length `s` while keeping the aspect ratio of the original image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 将size字典转换为指定输出图像大小的字典
        size = get_size_dict(size, default_to_square=False)
        
        # 如果size字典中有"shortest_edge"键，计算输出图像大小
        if "shortest_edge" in size:
            output_size = get_resize_output_image_size(
                image, size["shortest_edge"], default_to_square=False, input_data_format=input_data_format
            )
        # 如果size字典中有"height"和"width"键，直接取指定大小作为输出图像大小
        elif "height" in size and "width" in size:
            output_size = (size["height"], size["width"])
        else:
            # 抛出异常，要求size字典必须包含"height"和"width"或"shortest_edge"键
            raise ValueError(f"Size must have 'height' and 'width' or 'shortest_edge' as keys. Got {size.keys()}")
        
        # 调用resize函数进行图像尺寸调整，并返回结果
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,


注释：
    ) -> np.ndarray:
        """处理单个图像的预处理。"""
        if do_resize and size is None or resample is None:
            raise ValueError("如果 do_resize 为 True，则必须指定 size 和 resample。")

        if do_center_crop and crop_size is None:
            raise ValueError("如果 do_center_crop 为 True，则必须指定 crop_size。")

        if do_rescale and rescale_factor is None:
            raise ValueError("如果 do_rescale 为 True，则必须指定 rescale_factor。")

        if do_normalize and (image_mean is None or image_std is None):
            raise ValueError("如果 do_normalize 为 True，则必须指定 image_mean 和 image_std。")

        # 所有的转换都希望接收 numpy 数组。
        image = to_numpy_array(image)

        if is_scaled_image(image) and do_rescale:
            logger.warning_once(
                "看起来您正在尝试重新缩放已经缩放过的图像。如果输入图像的像素值在 0 到 1 之间，请设置 `do_rescale=False`，以避免再次对其进行缩放。"
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        if do_resize:
            image = self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)

        if do_center_crop:
            image = self.center_crop(image, size=crop_size, input_data_format=input_data_format)

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def preprocess(
        self,
        videos: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        patch_size: List[int] = None,
        num_frames: int = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        is_mixed: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```