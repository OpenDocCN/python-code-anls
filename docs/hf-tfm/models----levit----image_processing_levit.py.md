# `.\transformers\models\levit\image_processing_levit.py`

```py
# 定义了 LeViT 的图像处理器类
from typing import Dict, Iterable, Optional, Union  # 引入需要的类型提示模块

import numpy as np  # 引入 numpy 模块

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict  # 引入需要的图像处理工具模块
from ...image_transforms import (
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
)  # 引入图像变换相关函数
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)  # 引入图像相关工具
from ...utils import TensorType, logging  # 引入需要的工具模块

logger = logging.get_logger(__name__) # 获取日志记录器
    # 定义函数参数:
    # 是否调整输入图像的最短边以匹配指定大小。可以被`preprocess`方法中的`do_resize`参数覆盖。
    # 默认为`True`
    do_resize (`bool`, *optional*, defaults to `True`):
    
    # 调整大小后的输出图像大小。如果大小是一个具有键"width"和"height"的字典，那么图像将被调整大小为`(size["height"], size["width"])`。
    # 如果大小是具有键"shortest_edge"的字典，则最短边值`c`将被重新缩放为`int(c * (256/224))`。
    # 图像的较小边将与此值匹配，即，如果height>width，则图像将被缩放为`(size["shortest_egde"] * height / width, size["shortest_egde"])`。
    # 可被`preprocess`方法中的`size`参数覆盖。
    size (`Dict[str, int]`, *optional*, defaults to `{"shortest_edge": 224}`):
    
    # 如果调整图像大小，则使用的重采样滤波器。可以被`preprocess`方法中的`resample`参数覆盖。
    resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
    
    # 是否居中裁剪输入为`(crop_size["height"], crop_size["width"])`。可以被`preprocess`方法中的`do_center_crop`参数覆盖。
    do_center_crop (`bool`, *optional*, defaults to `True`):
    
    # `center_crop`之后的期望图像大小。可以被`preprocess`方法中的`crop_size`参数覆盖。
    crop_size (`Dict`, *optional*, defaults to `{"height": 224, "width": 224}`):
    
    # 控制是否通过指定的缩放因子`rescale_factor`对图像进行重新缩放。可以被`preprocess`方法中的`do_rescale`参数覆盖。
    do_rescale (`bool`, *optional*, defaults to `True`):
    
    # 如果重新缩放图像，则使用的缩放因子。可以被`preprocess`方法中的`rescale_factor`参数覆盖。
    rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
    
    # 控制是否对图像进行标准化。可以被`preprocess`方法中的`do_normalize`参数覆盖。
    do_normalize (`bool`, *optional*, defaults to `True`):
    
    # 如果对图像进行标准化，则使用的均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以被`preprocess`方法中的`image_mean`参数覆盖。
    image_mean (`List[int]`, *optional*, defaults to `[0.485, 0.456, 0.406]`):
    
    # 如果对图像进行标准化，则使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以被`preprocess`方法中的`image_std`参数覆盖。
    image_std (`List[int]`, *optional*, defaults to `[0.229, 0.224, 0.225]`):
    
    # 模型输入的名称列表
    model_input_names = ["pixel_values"]
    # 初始化方法，用于初始化图像预处理器对象
    def __init__(
        self,
        do_resize: bool = True,  # 是否执行调整大小操作，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，默认为None
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
        do_center_crop: bool = True,  # 是否执行中心裁剪，默认为True
        crop_size: Dict[str, int] = None,  # 裁剪大小的字典，默认为None
        do_rescale: bool = True,  # 是否执行重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 重新缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否执行归一化，默认为True
        image_mean: Optional[Union[float, Iterable[float]]] = IMAGENET_DEFAULT_MEAN,  # 图像均值，默认为ImageNet的均值
        image_std: Optional[Union[float, Iterable[float]]] = IMAGENET_DEFAULT_STD,  # 图像标准差，默认为ImageNet的标准差
        **kwargs,  # 其他参数
    ) -> None:
        # 调用父类初始化方法
        super().__init__(**kwargs)
        # 如果size为None，则将size设置为{"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 获取调整后的size字典
        size = get_size_dict(size, default_to_square=False)
        # 如果crop_size为None，则将crop_size设置为{"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 获取调整后的crop_size字典
        crop_size = get_size_dict(crop_size, param_name="crop_size")
    
        # 初始化各属性
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
    
    # 图像调整大小方法
    def resize(
        self,
        image: np.ndarray,  # 输入图像的numpy数组表示
        size: Dict[str, int],  # 调整后的图像大小字典
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为None
        **kwargs,  # 其他参数
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
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取尺寸字典，不默认使用正方形
        size_dict = get_size_dict(size, default_to_square=False)
        # 如果尺寸字典中含有键 "shortest_edge"
        if "shortest_edge" in size:
            # 计算最短边的大小
            shortest_edge = int((256 / 224) * size["shortest_edge"])
            # 获取调整后的图像大小
            output_size = get_resize_output_image_size(
                image, size=shortest_edge, default_to_square=False, input_data_format=input_data_format
            )
            # 将调整后的大小组成尺寸字典
            size_dict = {"height": output_size[0], "width": output_size[1]}
        # 如果尺寸字典中不含有 "height" 或 "width" 键
        if "height" not in size_dict or "width" not in size_dict:
            # 抛出值错误异常
            raise ValueError(
                f"Size dict must have keys 'height' and 'width' or 'shortest_edge'. Got {size_dict.keys()}"
            )
        # 返回调整大小后的图像
        return resize(
            image,
            size=(size_dict["height"], size_dict["width"]),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    # 图像预处理函数，对输入的图像进行预处理
    def preprocess(
        # 输入的图像数据，可以是单个图像或图像列表
        self,
        images: ImageInput,
        # 是否进行调整图像大小的操作
        do_resize: Optional[bool] = None,
        # 调整图像大小的目标尺寸
        size: Optional[Dict[str, int]] = None,
        # 重新采样方法
        resample: PILImageResampling = None,
        # 是否进行中心裁剪操作
        do_center_crop: Optional[bool] = None,
        # 中心裁剪的目标尺寸
        crop_size: Optional[Dict[str, int]] = None,
        # 是否进行图像尺度调整
        do_rescale: Optional[bool] = None,
        # 图像尺度调整的尺度因子
        rescale_factor: Optional[float] = None,
        # 是否进行图像归一化
        do_normalize: Optional[bool] = None,
        # 图像归一化的均值
        image_mean: Optional[Union[float, Iterable[float]]] = None,
        # 图像归一化的标准差
        image_std: Optional[Union[float, Iterable[float]]] = None,
        # 返回的数据类型，如张量
        return_tensors: Optional[TensorType] = None,
        # 数据格式，通道维度的顺序
        data_format: ChannelDimension = ChannelDimension.FIRST,
        # 输入数据的格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他参数
        **kwargs,
```