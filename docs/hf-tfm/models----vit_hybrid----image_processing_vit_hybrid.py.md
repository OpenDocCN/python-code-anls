# `.\transformers\models\vit_hybrid\image_processing_vit_hybrid.py`

```
# 设置文件编码格式为UTF-8
# 版权声明信息
#
# 载入所需的类型提示
从numpy库中导入数组操作函数
从image_processing_utils库中导入BaseImageProcessor类、BatchFeature类和get_size_dict函数
从image_transforms库中导入convert_to_rgb函数、get_resize_output_image_size函数、resize函数和to_channel_dimension_format函数
从image_utils库中导入OPENAI_CLIP_MEAN常量、OPENAI_CLIP_STD常量、ChannelDimension枚举类、ImageInput类、PILImageResampling枚举类、infer_channel_dimension_format函数、is_scaled_image函数、make_list_of_images函数、to_numpy_array函数、valid_images函数
从utils库中导入TensorType类、is_vision_available函数和logging模块

获取名为__name__的logger对象
如果视觉库可用
    从PIL库中导入全部模块
构建一个ViT混合图像处理器
    # 定义参数说明
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的（高度，宽度）尺寸以符合指定的`size`值。 可以在`preprocess`方法中通过`do_resize`进行覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            调整大小后的图像大小。 图像的最短边将调整为`size["shortest_edge"]`，最长边将调整以保持输入纵横比。 可以在`preprocess`方法中通过`size`进行覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            如果调整图像大小，要使用的重采样滤波器。 可以在`preprocess`方法中通过`resample`进行覆盖。
        do_center_crop (`bool`, *optional*, defaults to `True`):
            是否对图像进行中心裁剪以符合指定的`crop_size`。 可以在`preprocess`方法中通过`do_center_crop`进行覆盖。
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            在应用`center_crop`后输出图像的大小。 可以在`preprocess`方法中通过`crop_size`进行覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按指定的比例`rescale_factor`重新缩放图像。 可以在`preprocess`方法中通过`do_rescale`进行覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新缩放图像，要使用的缩放因子。 可以在`preprocess`方法中通过`rescale_factor`进行覆盖。
        do_normalize:
            是否将图像标准化。 可以在`preprocess`方法中通过`do_normalize`进行覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果标准化图像要使用的平均值。 这是一个浮点数或与图像通道数相同长度的浮点数列表。 可以在`preprocess`方法中通过`image_mean`参数进行覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果标准化图像要使用的标准差。 这是一个浮点数或与图像通道数相同长度的浮点数列表。 可以在`preprocess`方法中通过`image_std`参数进行覆盖。
            可以在`preprocess`方法中通过`image_std`参数进行覆盖。
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            是否将图像转换为RGB。
    """
    
    # 定义模型输入名称
    model_input_names = ["pixel_values"]
    # 初始化方法，用于设置图像处理器的参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像缩放，默认为True
        size: Dict[str, int] = None,  # 图像大小参数，默认为None
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图像重采样方法，默认为BICUBIC
        do_center_crop: bool = True,  # 是否进行中心裁剪，默认为True
        crop_size: Dict[str, int] = None,  # 裁剪尺寸参数，默认为None
        do_rescale: bool = True,  # 是否进行图像重新缩放，默认为True
        rescale_factor: Union[int, float] = 1 / 255,  # 图像重新缩放因子，默认为1/255
        do_normalize: bool = True,  # 是否进行图像标准化，默认为True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像均值，默认为None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像标准差，默认为None
        do_convert_rgb: bool = True,  # 是否转换图像为RGB格式，默认为True
        **kwargs,  # 其他参数
    ) -> None:
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果size参数为None，则设置默认的图像大小参数为{"shortest_edge": 224}
        size = size if size is not None else {"shortest_edge": 224}
        # 获取最终的图像大小参数
        size = get_size_dict(size, default_to_square=False)
        # 如果crop_size参数为None，则设置默认的裁剪尺寸参数为{"height": 224, "width": 224}
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        # 获取最终的裁剪尺寸参数
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        # 初始化各参数
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

    # 从transformers.models.clip.image_processing_clip.CLIPImageProcessor.resize方法复制过来的方法
    def resize(
        self,
        image: np.ndarray,  # 待处理的图像数据
        size: Dict[str, int],  # 图像大小参数
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图像重采样方法，默认为BICUBIC
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据格式，默认为None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式，默认为None
        **kwargs,  # 其他参数
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
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        default_to_square = True
        如果 size 中包含 "shortest_edge" 键
            将 size 设置为 size["shortest_edge"]
            将 default_to_square 设置为 False
        elif size 中包含 "height" 和 "width" 键
            将 size 设置为 (size["height"], size["width"])
        else:
            抛出数值错误 "Size must contain either 'shortest_edge' or 'height' and 'width'."
        
        根据输入的参数获取输出图像的大小
        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
         调整图片的大小
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
        image_std: Optional[Union[float, List[float]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```