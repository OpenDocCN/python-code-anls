# `.\models\donut\image_processing_donut.py`

```py
# 设置编码格式为 utf-8
# 版权声明，保留所有权利
# 根据 Apache 许可证，除非遵守许可证规定，否则不能使用该文件
# 查看许可证的副本，请访问 http://www.apache.org/licenses/LICENSE-2.0

# 如果适用法律要求或书面同意，那么根据许可证分发的软件基于“原样”基础分发，不附带任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以了解特定语言的权限和限制
"""Image processor class for Donut.""" 模块，用于 Donut 的图像处理器类。

from typing import Dict, List, Optional, Union  # 引入必要的类型提示

import numpy as np  # 引入 numpy 库

# 引入相关的图像处理工具
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    get_resize_output_image_size,
    pad,
    resize,
    to_channel_dimension_format,
)
from ...image_utils import (  # 引入图像处理所需的工具
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
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
from ...utils import TensorType, logging  # 引入必要的工具和日志
from ...utils.import_utils import is_vision_available  # 对于图像处理是否可用的判断

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果图像处理工具可用
if is_vision_available():
    # 导入 PIL 库
    import PIL


# 创建 Donut 图像处理器类
class DonutImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Donut image processor.
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to resize the image using thumbnail method.
        do_align_long_axis (`bool`, *optional*, defaults to `False`):
            Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
        do_pad (`bool`, *optional*, defaults to `True`):
            Whether to pad the image. If `random_padding` is set to `True` in `preprocess`, each image is padded with a
            random amount of padding on each side, up to the largest image size in the batch. Otherwise, all images are
            padded to the largest image size in the batch.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Image standard deviation.
    """

    # 定义模型输入的名称列表
    model_input_names = ["pixel_values"]
    # 初始化函数，设置各种参数并初始化特征提取器类
    def __init__(
        self,
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
        image_std: Optional[Union[float, List[float]] = None,
        **kwargs,
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 设置size参数默认值，如果size为空则设置为{"height": 2560, "width": 1920}
        size = size if size is not None else {"height": 2560, "width": 1920}
        # 如果size是元组或列表形式，则转换成(height, width)格式
        if isinstance(size, (tuple, list)):
            size = size[::-1]
        # 获取有效的size字典
        size = get_size_dict(size)

        # 初始化各类参数
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_thumbnail = do_thumbnail
        self.do_align_long_axis = do_align_long_axis
        self.do_pad = do_pad
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    # 将图像的长轴与指定尺寸的长轴对齐
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

        # 如果输出宽度小于高度并且输入宽度大于高度，或者输出宽度大于高度并且输入宽度小于高度，则旋转图像
        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3)

        # 如果设置了data_format参数，则将图像转换为指定数据格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        # 返回对齐后的图像
        return image
    # 定义一个方法，用于给定的图像添加填充，使其达到指定的大小
    def pad_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        random_padding: bool = False,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        填充图像到指定的大小。

        Args:
            image (`np.ndarray`):
                要进行填充的图像。
            size (`Dict[str, int]`):
                指定图像填充后的大小，格式为 `{"height": h, "width": w}`。
            random_padding (`bool`, *optional*, defaults to `False`):
                是否使用随机填充，默认为否。
            data_format (`str` 或 `ChannelDimension`, *optional*):
                输出图像的数据格式。若未设置，则使用输入图像的相同格式。
            input_data_format (`ChannelDimension` 或 `str`, *optional*):
                输入图像的通道维度格式。若未提供，则会被推断。
        """
        # 获取输出大小的高度和宽度
        output_height, output_width = size["height"], size["width"]
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        # 计算需要填充的高度和宽度差距
        delta_width = output_width - input_width
        delta_height = output_height - input_height

        if random_padding:
            # 使用随机填充时，随机确定顶部和左侧的填充大小
            pad_top = np.random.randint(low=0, high=delta_height + 1)
            pad_left = np.random.randint(low=0, high=delta_width + 1)
        else:
            # 使用对称填充时，计算顶部和左侧的填充大小
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        # 计算底部和右侧的填充大小
        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        # 定义填充参数，分别为垂直方向和水平方向的填充量
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        # 调用 pad 方法对图像进行填充，并返回填充后的图像
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format)

    # 定义一个已被弃用的方法，用于填充图像
    def pad(self, *args, **kwargs):
        # 显示一条日志消息，指示该方法已被弃用，并将在未来版本中移除
        logger.info("pad is deprecated and will be removed in version 4.27. Please use pad_image instead.")
        # 调用新的 pad_image 方法，替代旧的 pad 方法
        return self.pad_image(*args, **kwargs)

    # 定义一个方法，用于生成缩略图
    def thumbnail(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
        ...
    ) -> np.ndarray:
        """
        调整图像大小以生成缩略图。调整图像大小使得没有任何维度大于指定大小的相应维度。

        Args:
            image (`np.ndarray`):
                要调整大小的图像。
            size (`Dict[str, int]`):
                要调整图像大小为的 `{"height": h, "width": w}` 大小。
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                要使用的重采样滤波器。
            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                输出图像的数据格式。如果未设置，则使用与输入图像相同的格式。
            input_data_format (`ChannelDimension` or `str`, *optional*):
                输入图像的通道维度格式。如果未提供，将进行推断。
        """
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        # 获取输出图像的高度和宽度
        output_height, output_width = size["height"], size["width"]

        # 我们始终调整大小为输入或输出大小中较小的一个。
        height = min(input_height, output_height)
        width = min(input_width, output_width)

        # 如果输入图像的尺寸与输出尺寸相同，则直接返回输入图像
        if height == input_height and width == input_width:
            return image

        # 如果输入图像的高度大于宽度，则根据高度调整宽度
        if input_height > input_width:
            width = int(input_width * height / input_height)
        # 如果输入图像的宽度大于高度，则根据宽度调整高度
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
        Resizes `image` to `(height, width)` specified by `size` using the PIL library.

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
        # 根据给定的大小参数获取要调整的图片的大小
        size = get_size_dict(size)
        # 获取最短边
        shortest_edge = min(size["height"], size["width"])
        # 根据最短边获取调整后的输出图像大小
        output_size = get_resize_output_image_size(
            image, size=shortest_edge, default_to_square=False, input_data_format=input_data_format
        )
        # 调整图像大小
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
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        random_padding: bool = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```