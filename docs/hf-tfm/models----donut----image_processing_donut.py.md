# `.\models\donut\image_processing_donut.py`

```
# 如果视觉处理库可用，则导入PIL库
if is_vision_available():
    import PIL

# 定义一个名为DonutImageProcessor的类，继承自BaseImageProcessor类
class DonutImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Donut image processor.
    """
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
    # 定义模型输入的名称列表，仅包含像素值
    model_input_names = ["pixel_values"]
    # 初始化函数，用于设置图像处理的各项参数和默认值
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像尺寸调整，默认为True
        size: Dict[str, int] = None,  # 图像的目标尺寸，字典形式表示，包含高度和宽度，默认为None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像调整大小时的重采样方法，默认为双线性插值
        do_thumbnail: bool = True,  # 是否生成缩略图，默认为True
        do_align_long_axis: bool = False,  # 是否在长轴上对齐图像，默认为False
        do_pad: bool = True,  # 是否进行图像填充，默认为True
        do_rescale: bool = True,  # 是否对图像进行重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 图像重新缩放的因子，默认为1/255
        do_normalize: bool = True,  # 是否对图像进行归一化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像的均值用于归一化，默认为None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像的标准差用于归一化，默认为None
        **kwargs,  # 其他可选的关键字参数
    ) -> None:
        # 调用父类的初始化方法，传入其他的关键字参数
        super().__init__(**kwargs)

        # 如果size为None，则设定默认的高度和宽度
        size = size if size is not None else {"height": 2560, "width": 1920}
        # 如果size是元组或列表形式，则转换为字典形式，表示高度和宽度
        if isinstance(size, (tuple, list)):
            # The previous feature extractor size parameter was in (width, height) format
            size = size[::-1]
        # 使用函数get_size_dict处理size，确保返回的是一个标准化的尺寸字典
        size = get_size_dict(size)

        # 设置对象的属性值，将初始化函数的参数赋值给对象的属性
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_thumbnail = do_thumbnail
        self.do_align_long_axis = do_align_long_axis
        self.do_pad = do_pad
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 如果image_mean为None，则使用IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 如果image_std为None，则使用IMAGENET_STANDARD_STD
        # 验证处理器的关键字列表，用于后续处理
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_thumbnail",
            "do_align_long_axis",
            "do_pad",
            "random_padding",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
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

        # 如果输出宽度小于高度且输入宽度大于高度，或者输出宽度大于高度且输入宽度小于高度，则须旋转图像
        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3)

        # 如果指定了输出数据格式，则转换图像数据格式
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)

        # 返回对齐后的图像
        return image

    def pad_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        random_padding: bool = False,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Pad the image to the specified size.

        Args:
            image (`np.ndarray`):
                The image to be padded.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to pad the image to.
            random_padding (`bool`, *optional*, defaults to `False`):
                Whether to use random padding or not.
            data_format (`str` or `ChannelDimension`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # Extract output height and width from the size dictionary
        output_height, output_width = size["height"], size["width"]
        
        # Obtain input height and width from the input image
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        # Calculate the difference between output and input dimensions
        delta_width = output_width - input_width
        delta_height = output_height - input_height

        # Determine padding amounts based on random_padding flag
        if random_padding:
            pad_top = np.random.randint(low=0, high=delta_height + 1)
            pad_left = np.random.randint(low=0, high=delta_width + 1)
        else:
            pad_top = delta_height // 2
            pad_left = delta_width // 2

        # Calculate remaining padding amounts to complete the pad
        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        # Construct the padding tuple for np.pad function
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        
        # Apply padding to the image using np.pad
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format)

    def pad(self, *args, **kwargs):
        # Log a deprecation warning for the `pad` method
        logger.info("pad is deprecated and will be removed in version 4.27. Please use pad_image instead.")
        # Redirect to `pad_image` method
        return self.pad_image(*args, **kwargs)

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
        Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
        corresponding dimension of the specified size.

        Args:
            image (`np.ndarray`):
                The image to be resized.
            size (`Dict[str, int]`):
                The size `{"height": h, "width": w}` to resize the image to.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                The resampling filter to use.
            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                The data format of the output image. If unset, the same format as the input image is used.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        # 获取输入图像的高度和宽度
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)
        
        # 获取输出图像的目标高度和宽度
        output_height, output_width = size["height"], size["width"]

        # 始终调整图像大小为输入或输出大小中较小的那一个
        height = min(input_height, output_height)
        width = min(input_width, output_width)

        # 如果输入图像已经符合要求的大小，则直接返回原图像
        if height == input_height and width == input_width:
            return image

        # 根据输入图像的长宽比例调整目标高度或宽度
        if input_height > input_width:
            width = int(input_width * height / input_height)
        elif input_width > input_height:
            height = int(input_height * width / input_width)

        # 调用 resize 函数，进行图像的实际调整
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
        ):
        """
        Resize the input image to the specified size.

        Args:
            image (`np.ndarray`):
                The image to be resized.
            size (`Dict[str, int]`):
                The target size `{"height": h, "width": w}` to resize the image to.
            resample (`PILImageResampling`, *optional*):
                The resampling filter to use.
            data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                The data format of the output image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image.

        Returns:
            np.ndarray: The resized image.
        """
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
        # 调整 `size` 参数，确保其为大小字典
        size = get_size_dict(size)
        # 计算 `size` 中较短的边长
        shortest_edge = min(size["height"], size["width"])
        # 获取调整大小后的输出图像尺寸
        output_size = get_resize_output_image_size(
            image, size=shortest_edge, default_to_square=False, input_data_format=input_data_format
        )
        # 调整图像大小并返回调整后的图像
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
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```