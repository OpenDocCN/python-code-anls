# `.\models\chinese_clip\image_processing_chinese_clip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，声明代码版权归 OFA-Sys 团队作者和 HuggingFace 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件
# 除非符合许可证的条款，否则不得使用本文件
# 可以在以下网址获取许可证的
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
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
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """
    # 定义模型输入名称列表，用于模型输入的像素值
    model_input_names = ["pixel_values"]
    # 初始化函数，用于设置图像处理器的各种参数和属性
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像尺寸调整的标志
        size: Dict[str, int] = None,  # 图像调整后的尺寸字典，包含宽和高
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图像调整时的重采样方法，默认为双三次插值
        do_center_crop: bool = True,  # 是否进行中心裁剪的标志
        crop_size: Dict[str, int] = None,  # 裁剪后的尺寸字典，包含裁剪后的宽和高
        do_rescale: bool = True,  # 是否进行图像像素值缩放的标志
        rescale_factor: Union[int, float] = 1 / 255,  # 像素值缩放的因子，默认为1/255
        do_normalize: bool = True,  # 是否进行图像归一化的标志
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像归一化的均值，默认使用OpenAI的均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像归一化的标准差，默认使用OpenAI的标准差
        do_convert_rgb: bool = True,  # 是否进行RGB格式转换的标志
        **kwargs,  # 其他可能的参数
    ) -> None:
        # 调用父类的初始化方法，传递可能的其他参数
        super().__init__(**kwargs)
        # 如果未提供图像调整后的尺寸字典，则设置默认最短边为224像素
        size = size if size is not None else {"shortest_edge": 224}
        # 获取标准化后的图像尺寸字典，确保不强制为正方形
        size = get_size_dict(size, default_to_square=False)
        # 如果未提供裁剪后的尺寸字典，则设置默认裁剪尺寸为224x224像素
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 获取标准化后的裁剪尺寸字典
        crop_size = get_size_dict(crop_size)

        # 将参数赋值给对象的属性
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        # 定义有效的图像处理器键列表，用于后续验证
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
            "do_convert_rgb",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]
    def resize_image(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """
        Resize an image maintaining aspect ratio based on the shortest edge.

        Args:
            image (`np.ndarray`):
                The input image to be resized.
            size (`Dict[str, int]`):
                Dictionary containing target height and width.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter used during resizing.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the output image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image.

        Returns:
            np.ndarray:
                Resized image as a NumPy array.
        """
        # Obtain the resized size dictionary ensuring aspect ratio preservation
        size = get_size_dict(size, default_to_square=False)
        
        # Calculate the output size based on the input image and target size
        output_size = get_resize_output_image_size(
            image, size=(size["height"], size["width"]), default_to_square=False, input_data_format=input_data_format
        )
        
        # Perform the resizing operation
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocesses a batch of images based on specified parameters.

        Args:
            images (`ImageInput`):
                Input images to be preprocessed.
            do_resize (`bool`, *optional*):
                Whether to resize the images.
            size (`Dict[str, int]`, *optional*):
                Target size for resizing the images.
            resample (`PILImageResampling`, *optional*):
                Resampling filter used during image resizing.
            do_center_crop (`bool`, *optional*):
                Whether to perform center cropping.
            crop_size (`int`, *optional*):
                Size of the center crop.
            do_rescale (`bool`, *optional*):
                Whether to rescale the images.
            rescale_factor (`float`, *optional*):
                Factor by which to rescale the images.
            do_normalize (`bool`, *optional*):
                Whether to normalize the images.
            image_mean (`Optional[Union[float, List[float]]]`, *optional*):
                Mean value(s) for image normalization.
            image_std (`Optional[Union[float, List[float]]]`, *optional*):
                Standard deviation value(s) for image normalization.
            do_convert_rgb (`bool`, *optional*):
                Whether to convert images to RGB format.
            return_tensors (`Optional[Union[str, TensorType]]`, *optional*):
                Format of output tensors.
            data_format (`Optional[ChannelDimension]`, *optional*):
                Channel dimension format of the images.
            input_data_format (`Optional[Union[str, ChannelDimension]]`, *optional*):
                Channel dimension format of the input images.

        Returns:
            Processed images according to specified preprocessing steps.
        """
```