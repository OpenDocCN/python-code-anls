# `.\models\imagegpt\image_processing_imagegpt.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，保留所有权利
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""ImageGPT"的图像处理器类

从 typing 模块导入类型提示
从 ...image_processing_utils 模块导入 BaseImageProcessor、BatchFeature 和 get_size_dict 函数
从 ...image_transforms 模块导入 rescale、resize 和 to_channel_dimension_format 函数
从 ...image_utils 模块导入 ChannelDimension、ImageInput、PILImageResampling、infer_channel_dimension_format、is_scaled_image、make_list_of_images、to_numpy_array 和 valid_images 函数
从 ...utils 模块导入 TensorType、is_vision_available 和 logging 函数

如果视觉模块可用:
    从 PIL 模块导入

获取日志记录器
"""
def squared_euclidean_distance(a, b):
    # 将 b 转置
    b = b.T
    # 计算 a 的平方和
    a2 = np.sum(np.square(a), axis=1)
    # 计算 b 的平方和
    b2 = np.sum(np.square(b), axis=0)
    # 计算 ab 的点积
    ab = np.matmul(a, b)
    # 计算欧氏距离的平方
    d = a2[:, None] - 2 * ab + b2[None, :]
    return d

def color_quantize(x, clusters):
    # 重塑 x 为二维数组
    x = x.reshape(-1, 3)
    # 计算 x 与 clusters 之间的欧氏距离
    d = squared_euclidean_distance(x, clusters)
    # 返回每个像素点对应的最近颜色簇索引
    return np.argmin(d, axis=1)

class ImageGPTImageProcessor(BaseImageProcessor):
    r"""
    构造一个 ImageGPT 图像处理器。此图像处理器可用于将图像调整为较小分辨率（如 32x32 或 64x64），对其进行归一化，最后对其进行颜色量化以获得"像素值"序列（颜色簇）。
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
        # Call the superclass constructor with additional keyword arguments
        super().__init__(**kwargs)
        # Set default size if not provided
        size = size if size is not None else {"height": 256, "width": 256}
        # Get the size dictionary
        size = get_size_dict(size)
        # Convert clusters to numpy array if not None
        self.clusters = np.array(clusters) if clusters is not None else None
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.do_color_quantize = do_color_quantize

    # Copied from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
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
        # Convert size to a dictionary
        size = get_size_dict(size)
        # Check if the dictionary contains both "height" and "width" keys
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # Get the output size from the dictionary
        output_size = (size["height"], size["width"])
        # Call the resize function with the specified parameters
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def normalize(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    def normalize_image(
        self,
        image: np.ndarray,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None
    ) -> np.ndarray:
        """
        Normalizes an images' pixel values to between [-1, 1].

        Args:
            image (`np.ndarray`):
                Image to normalize.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # Rescale the image pixel values to be between [-1, 1]
        image = rescale(image=image, scale=1 / 127.5, data_format=data_format, input_data_format=input_data_format)
        # Subtract 1 from the image pixel values
        image = image - 1
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
```