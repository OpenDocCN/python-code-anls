# `.\models\vit_hybrid\image_processing_vit_hybrid.py`

```py
    r"""
    Constructs a ViT Hybrid image processor.
    """

    def __init__(self, image_size: Union[int, List[int]], mean: Optional[List[float]] = OPENAI_CLIP_MEAN,
                 std: Optional[List[float]] = OPENAI_CLIP_STD, resampling: PILImageResampling = PILImageResampling.BILINEAR,
                 channel_dim: ChannelDimension = ChannelDimension.LAST, dtype: TensorType = np.float32):
        """
        Initialize the ViT Hybrid image processor.

        Parameters:
        - image_size (Union[int, List[int]]): Desired size of the output image.
        - mean (Optional[List[float]]): Mean values for normalization, defaults to OPENAI_CLIP_MEAN.
        - std (Optional[List[float]]): Standard deviation values for normalization, defaults to OPENAI_CLIP_STD.
        - resampling (PILImageResampling): Resampling method for image resizing, defaults to PILImageResampling.BILINEAR.
        - channel_dim (ChannelDimension): Channel dimension format, defaults to ChannelDimension.LAST.
        - dtype (TensorType): Data type of the processed images, defaults to np.float32.
        """
        super().__init__()
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.resampling = resampling
        self.channel_dim = channel_dim
        self.dtype = dtype

    def __call__(self, images: Union[ImageInput, List[ImageInput]], return_tensors: bool = True,
                 **kwargs) -> Union[BatchFeature, List[BatchFeature]]:
        """
        Process a single image or a batch of images.

        Parameters:
        - images (Union[ImageInput, List[ImageInput]]): Input image(s) to be processed.
        - return_tensors (bool): Whether to return tensors (True) or numpy arrays (False), defaults to True.
        - **kwargs: Additional keyword arguments for preprocessing.

        Returns:
        - Union[BatchFeature, List[BatchFeature]]: Processed image(s) as tensors or numpy arrays.
        """
        # Validate and preprocess input arguments
        images = make_list_of_images(images)
        validate_kwargs(kwargs)
        validate_preprocess_arguments(self.mean, self.std, self.image_size, self.channel_dim)

        # Resize images to the desired size
        resized_images = [resize(image, self.image_size, self.resampling) for image in images]

        # Convert images to RGB format if needed
        rgb_images = [convert_to_rgb(image) for image in resized_images]

        # Ensure images have the correct channel dimension format
        formatted_images = [to_channel_dimension_format(image, self.channel_dim) for image in rgb_images]

        # Normalize images
        normalized_images = [self._normalize_image(image) for image in formatted_images]

        # Convert images to numpy arrays or tensors based on return_tensors flag
        if return_tensors:
            return np.stack(normalized_images).astype(self.dtype)
        else:
            return normalized_images

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image data using mean and standard deviation.

        Parameters:
        - image (np.ndarray): Input image data.

        Returns:
        - np.ndarray: Normalized image data.
        """
        mean = np.array(self.mean)
        std = np.array(self.std)
        return (image.astype(np.float32) - mean) / std
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
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
        do_normalize:
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

    # 定义模型输入的名称列表，包含一个元素 "pixel_values"
    model_input_names = ["pixel_values"]
    # 初始化方法，用于设置图像处理器的各种参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像大小调整的标志
        size: Dict[str, int] = None,  # 图像调整后的尺寸字典，默认最短边为224
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图像调整时的重采样方法，默认为双三次插值
        do_center_crop: bool = True,  # 是否进行中心裁剪的标志
        crop_size: Dict[str, int] = None,  # 中心裁剪后的尺寸字典，默认为224x224
        do_rescale: bool = True,  # 是否进行图像数值缩放的标志
        rescale_factor: Union[int, float] = 1 / 255,  # 图像数值缩放的因子，默认为1/255
        do_normalize: bool = True,  # 是否进行图像标准化的标志
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像标准化的均值，默认为OpenAI CLIP模型的均值
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准化的标准差，默认为OpenAI CLIP模型的标准差
        do_convert_rgb: bool = True,  # 是否进行RGB格式转换的标志
        **kwargs,  # 其他可选参数
    ) -> None:
        # 调用父类初始化方法，传递额外参数
        super().__init__(**kwargs)
        # 如果size为None，则设置默认尺寸字典，最短边为224
        size = size if size is not None else {"shortest_edge": 224}
        # 根据参数获取尺寸字典，不默认为正方形
        size = get_size_dict(size, default_to_square=False)
        # 如果crop_size为None，则设置默认的裁剪尺寸字典，高度和宽度均为224
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 根据参数获取裁剪尺寸字典，默认为正方形
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        # 将参数赋值给实例变量
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
        # 设置有效的处理器关键字列表
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

    # 从transformers.models.clip.image_processing_clip.CLIPImageProcessor.resize复制而来的方法
    def resize(
        self,
        image: np.ndarray,  # 待处理的图像数据，NumPy数组格式
        size: Dict[str, int],  # 目标尺寸字典，包含高度和宽度信息
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 重采样方法，默认为双三次插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，可选参数
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，可选参数
        **kwargs,  # 其他可选参数
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

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
        # 默认情况下将图片调整为正方形
        default_to_square = True
        if "shortest_edge" in size:
            # 如果指定了最短边的长度，则按照最短边调整图片大小
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            # 如果指定了高度和宽度，则按照这两个尺寸调整图片大小
            size = (size["height"], size["width"])
        else:
            # 如果大小参数中既没有指定最短边，也没有指定高度和宽度，则抛出数值错误
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        # 获得调整后的图片尺寸
        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        # 调整图片大小并返回调整后的图片数据
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
```