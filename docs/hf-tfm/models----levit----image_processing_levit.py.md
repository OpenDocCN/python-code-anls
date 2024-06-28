# `.\models\levit\image_processing_levit.py`

```
# 引入必要的库和模块
from typing import Dict, Iterable, Optional, Union

import numpy as np  # 导入 NumPy 库

# 导入图像处理相关的工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,  # 导入图像处理的常量，如默认均值和标准差
    IMAGENET_DEFAULT_STD,
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
from ...utils import TensorType, logging  # 导入工具函数和日志记录器

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


class LevitImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LeViT image processor.
    """
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整输入图像的最短边至 int(256/224 * size)，可以在 `preprocess` 方法中的 `do_resize` 参数中覆盖。
        size (`Dict[str, int]`, *optional*, defaults to `{"shortest_edge": 224}`):
            调整后的输出图像尺寸。如果 `size` 是一个包含 "width" 和 "height" 键的字典，图像将被调整至 `(size["height"], size["width"])`。如果 `size` 是一个包含 "shortest_edge" 键的字典，最短边的值 `c` 将被重新缩放为 `int(c * (256/224))`。图像的较小边将被匹配到此值，例如，如果 height > width，则图像将被缩放至 `(size["shortest_edge"] * height / width, size["shortest_edge"])`。可以在 `preprocess` 方法中的 `size` 参数中覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            如果调整图像大小，使用的重采样滤波器。可以在 `preprocess` 方法中的 `resample` 参数中覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            是否对输入图像进行中心裁剪至 `(crop_size["height"], crop_size["width"])`。可以在 `preprocess` 方法中的 `do_center_crop` 参数中覆盖。
        crop_size (`Dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
            `center_crop` 后的期望图像尺寸。可以在 `preprocess` 方法中的 `crop_size` 参数中覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            控制是否按指定的比例因子 `rescale_factor` 重新缩放图像。可以在 `preprocess` 方法中的 `do_rescale` 参数中覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新缩放图像，要使用的缩放因子。可以在 `preprocess` 方法中的 `rescale_factor` 参数中覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            控制是否对图像进行归一化。可以在 `preprocess` 方法中的 `do_normalize` 参数中覆盖。
        image_mean (`List[int]`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
            如果归一化图像，要使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在 `preprocess` 方法中的 `image_mean` 参数中覆盖。
        image_std (`List[int]`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
            如果归一化图像，要使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以在 `preprocess` 方法中的 `image_std` 参数中覆盖。
    """

    model_input_names = ["pixel_values"]
    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, Iterable[float]]] = IMAGENET_DEFAULT_MEAN,
        image_std: Optional[Union[float, Iterable[float]]] = IMAGENET_DEFAULT_STD,
        **kwargs,
    ) -> None:
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 如果 size 参数为 None，则设置默认的最短边为 224
        size = size if size is not None else {"shortest_edge": 224}
        # 根据给定的 size 参数获取大小的字典，确保不会默认为正方形
        size = get_size_dict(size, default_to_square=False)
        # 如果 crop_size 参数为 None，则设置默认的高度和宽度均为 224
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 根据给定的 crop_size 参数获取裁剪大小的字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")

        # 初始化类成员变量
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        # 设置有效的处理器关键字列表，包括图像处理相关的参数和数据格式参数
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
    ) -> np.ndarray:
        """
        Resize an image.

        If size is a dict with keys "width" and "height", the image will be resized to `(size["height"],
        size["width"])`.

        If size is a dict with key "shortest_edge", the shortest edge value `c` is rescaled to `int(c * (256/224))`.
        The smaller edge of the image will be matched to this value i.e, if height > width, then image will be rescaled
        to `(size["shortest_egde"] * height / width, size["shortest_egde"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image after resizing. If size is a dict with keys "width" and "height", the image
                will be resized to (height, width). If size is a dict with key "shortest_edge", the shortest edge value
                `c` is rescaled to int(`c` * (256/224)). The smaller edge of the image will be matched to this value
                i.e, if height > width, then image will be rescaled to (size * height / width, size).
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # Determine the target size dictionary based on input size parameters
        size_dict = get_size_dict(size, default_to_square=False)

        # Check if 'shortest_edge' is specified in size dictionary
        if "shortest_edge" in size:
            # Calculate the length of the shortest edge based on the scaling factor (256/224)
            shortest_edge = int((256 / 224) * size["shortest_edge"])
            # Determine the output size after resizing based on the calculated shortest edge
            output_size = get_resize_output_image_size(
                image, size=shortest_edge, default_to_square=False, input_data_format=input_data_format
            )
            # Update size_dict to reflect the height and width after resizing
            size_dict = {"height": output_size[0], "width": output_size[1]}

        # Ensure that the size_dict contains both 'height' and 'width' keys
        if "height" not in size_dict or "width" not in size_dict:
            # Raise an error if the size_dict does not have the required keys
            raise ValueError(
                f"Size dict must have keys 'height' and 'width' or 'shortest_edge'. Got {size_dict.keys()}"
            )

        # Resize the image to the specified dimensions using the resize function
        return resize(
            image,
            size=(size_dict["height"], size_dict["width"]),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    # 定义一个预处理方法，用于处理图像数据
    def preprocess(
        self,
        images: ImageInput,  # 图像输入，可以是单张图像或图像列表
        do_resize: Optional[bool] = None,  # 是否调整大小的标志，默认为None
        size: Optional[Dict[str, int]] = None,  # 调整大小的目标尺寸，字典形式，包含宽和高
        resample: PILImageResampling = None,  # 调整大小时使用的重采样方法，默认为None
        do_center_crop: Optional[bool] = None,  # 是否进行中心裁剪的标志，默认为None
        crop_size: Optional[Dict[str, int]] = None,  # 中心裁剪的目标尺寸，字典形式，包含宽和高
        do_rescale: Optional[bool] = None,  # 是否进行重新缩放的标志，默认为None
        rescale_factor: Optional[float] = None,  # 重新缩放的因子，默认为None
        do_normalize: Optional[bool] = None,  # 是否进行归一化的标志，默认为None
        image_mean: Optional[Union[float, Iterable[float]]] = None,  # 图像归一化的均值，默认为None
        image_std: Optional[Union[float, Iterable[float]]] = None,  # 图像归一化的标准差，默认为None
        return_tensors: Optional[TensorType] = None,  # 返回的张量类型，默认为None
        data_format: ChannelDimension = ChannelDimension.FIRST,  # 数据的通道格式，默认为第一通道
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的通道格式，默认为None
        **kwargs,  # 其他可能的关键字参数，以字典形式接收
```