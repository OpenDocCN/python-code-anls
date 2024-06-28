# `.\models\imagegpt\image_processing_imagegpt.py`

```py
# 导入所需的模块和函数
from typing import Dict, List, Optional, Union  # 导入类型提示相关的模块和函数

import numpy as np  # 导入 NumPy 库，用于数值计算

# 导入所需的图像处理工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import rescale, resize, to_channel_dimension_format  # 导入图像变换相关函数
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
)  # 导入图像处理和验证相关函数

from ...utils import TensorType, is_vision_available, logging  # 导入其他工具函数和变量

# 如果视觉相关功能可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 获取 logger 实例，用于记录日志
logger = logging.get_logger(__name__)


def squared_euclidean_distance(a, b):
    """
    计算两组向量之间的平方欧几里得距离。

    参数：
    - a: 第一组向量，形状为 (m, d)
    - b: 第二组向量，形状为 (n, d)

    返回：
    - d: 形状为 (m, n) 的距离矩阵
    """
    b = b.T  # 将 b 转置为 (d, n)，以便计算点积
    a2 = np.sum(np.square(a), axis=1)  # 计算 a 中每个向量的平方和
    b2 = np.sum(np.square(b), axis=0)  # 计算 b 中每个向量的平方和
    ab = np.matmul(a, b)  # 计算 a 和 b 之间的点积
    d = a2[:, None] - 2 * ab + b2[None, :]  # 计算平方欧几里得距离矩阵
    return d


def color_quantize(x, clusters):
    """
    对输入的像素值进行颜色量化，将每个像素映射到最近的颜色簇。

    参数：
    - x: 输入的像素值数组，形状为 (h*w, 3)，其中 h 是高度，w 是宽度
    - clusters: 颜色簇的数组，形状为 (k, 3)，其中 k 是颜色簇的数量

    返回：
    - 颜色簇索引数组，形状为 (h*w,)，每个元素表示对应像素所属的颜色簇索引
    """
    x = x.reshape(-1, 3)  # 将输入的像素值重塑为 (h*w, 3) 的二维数组
    d = squared_euclidean_distance(x, clusters)  # 计算每个像素值与各个颜色簇的距离
    return np.argmin(d, axis=1)  # 返回每个像素值所属的最近颜色簇的索引


class ImageGPTImageProcessor(BaseImageProcessor):
    """
    ImageGPT 的图像处理器类，用于将图像调整大小到较小的分辨率（如 32x32 或 64x64），归一化并进行颜色量化，
    以获取像素值序列（颜色簇）。
    """
    def __init__(self):
        """
        初始化 ImageGPTImageProcessor 类。
        """
        super().__init__()  # 调用父类 BaseImageProcessor 的初始化方法

    def process(self, images: List[ImageInput], size: Optional[Union[int, Tuple[int, int]]] = None) -> List[BatchFeature]:
        """
        对输入的图像列表进行处理，包括调整大小、归一化和颜色量化。

        参数：
        - images: 输入的图像列表，每个元素是 ImageInput 类型的对象
        - size: 要调整的目标大小，可以是单个整数（将图像调整为正方形）或包含两个整数的元组（宽度，高度）

        返回：
        - 处理后的 BatchFeature 列表，每个 BatchFeature 包含处理后的特征和元数据
        """
        if size is not None:
            validate_preprocess_arguments(size)  # 验证预处理参数是否合法

        # 将图像转换为 NumPy 数组列表
        np_images = make_list_of_images(images)

        if size is not None:
            # 调整图像大小到指定尺寸
            np_images = resize(np_images, size, resampling=PILImageResampling.BICUBIC)

        # 将图像转换为适合通道维度格式
        np_images = to_channel_dimension_format(np_images, ChannelDimension.LAST)

        # 归一化图像像素值到 [0, 1] 范围内
        np_images = rescale(np_images)

        # 对归一化后的图像进行颜色量化，得到颜色簇序列
        clusters = get_size_dict(size)
        pixel_values = [color_quantize(im, clusters) for im in np_images]

        # 将处理后的特征和元数据封装为 BatchFeature 对象并返回
        batch_features = [
            BatchFeature(
                pixel_values=im_quantized.tolist(),  # 转换为列表形式的颜色簇序列
                metadata={"original_size": im.shape[:2]}  # 记录原始图像的尺寸
            )
            for im, im_quantized in zip(images, pixel_values)
        ]

        return batch_features  # 返回处理后的 BatchFeature 列表
    Args:
        clusters (`np.ndarray` or `List[List[int]]`, *optional*):
            The color clusters to use, of shape `(n_clusters, 3)` when color quantizing. Can be overriden by `clusters`
            in `preprocess`.
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's dimensions to `(size["height"], size["width"])`. Can be overridden by
            `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 256, "width": 256}`):
            Size of the image after resizing. Can be overridden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image pixel value to between [-1, 1]. Can be overridden by `do_normalize` in
            `preprocess`.
        do_color_quantize (`bool`, *optional*, defaults to `True`):
            Whether to color quantize the image. Can be overridden by `do_color_quantize` in `preprocess`.
    """

    # List of input names expected by the model
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        # clusters is a first argument to maintain backwards compatibility with the old ImageGPTImageProcessor
        clusters: Optional[Union[List[List[int]], np.ndarray]] = None,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_normalize: bool = True,
        do_color_quantize: bool = True,
        **kwargs,
    ) -> None:
        # Call the constructor of the superclass with additional keyword arguments
        super().__init__(**kwargs)
        # If size argument is None, set default size to {"height": 256, "width": 256}
        size = size if size is not None else {"height": 256, "width": 256}
        # Normalize size dictionary to ensure it contains both "height" and "width" keys
        size = get_size_dict(size)
        # Convert clusters to a numpy array if not None, else set it to None
        self.clusters = np.array(clusters) if clusters is not None else None
        # Initialize instance variables with provided or default values
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.do_color_quantize = do_color_quantize
        # List of valid keys for the processor configuration
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_normalize",
            "do_color_quantize",
            "clusters",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # Copied from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize
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
        # 获取规范化后的尺寸字典
        size = get_size_dict(size)
        # 检查尺寸字典中是否包含有效的"height"和"width"键
        if "height" not in size or "width" not in size:
            # 如果缺少任一键，抛出值错误异常
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 设置输出尺寸为(height, width)
        output_size = (size["height"], size["width"])
        # 调用resize函数进行图像调整大小操作，返回调整后的图像
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
    def normalize_image(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None
    ) -> np.ndarray:
        """
        Normalizes an image's pixel values to between [-1, 1].

        Args:
            image (`np.ndarray`):
                Image to normalize.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        
        Returns:
            np.ndarray:
                Normalized image with pixel values scaled to [-1, 1].
        """
        # Rescale the image pixel values to the range [-1, 1]
        image = rescale(image=image, scale=1 / 127.5, data_format=data_format, input_data_format=input_data_format)
        # Adjust the image values to fit the range [-1, 1] by subtracting 1
        image = image - 1
        # Return the normalized image
        return image

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_normalize: bool = None,
        do_color_quantize: Optional[bool] = None,
        clusters: Optional[Union[List[List[int]], np.ndarray]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocesses images based on specified operations and parameters.

        Args:
            images (`ImageInput`):
                Input images to be preprocessed.
            do_resize (`bool`, *optional*):
                Flag indicating whether to resize images.
            size (`Dict[str, int]`, *optional*):
                Dictionary specifying target sizes for resizing.
            resample (`PILImageResampling`, *optional*):
                Resampling method for resizing images.
            do_normalize (`bool`, *optional*):
                Flag indicating whether to normalize images.
            do_color_quantize (`Optional[bool]`, *optional*):
                Flag indicating whether to perform color quantization.
            clusters (`Optional[Union[List[List[int]], np.ndarray]]`, *optional*):
                Clusters for color quantization.
            return_tensors (`Optional[Union[str, TensorType]]`, *optional*):
                Desired format for output tensors.
            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                Format of the image data channels.
            input_data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                Format of the input image data channels.

        Returns:
            Preprocessed images according to the specified operations and parameters.
        """
```