# `.\models\convnext\image_processing_convnext.py`

```py
# coding=utf-8
# 版权声明，指定该文件的编码格式为UTF-8
# 引入必要的模块和函数
# 导入所需的模块和函数的包，如果这些模块和函数不存在，则会抛出相应的错误
# 定义了一些常量，如“License”，“IMAGENET_STANDARD_MEAN”等
# 导入一些模块和函数，如“BaseImageProcessor”、“get_size_dict”、“center_crop”等
# 函数内部调用的模块和函数的导入
# 定义了一个类，继承自“BaseImageProcessor”，用于实现图像处理的功能
    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            控制是否将图像的（高度，宽度）尺寸调整为指定的 `size`。可以被 `preprocess` 方法中的 `do_resize` 覆盖。
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 384}`):
            在应用 `resize` 后输出图像的分辨率。如果 `size["shortest_edge"]` >= 384，图像将被调整为 `(size["shortest_edge"], size["shortest_edge"])`。否则，图像的较小边将匹配到 `int(size["shortest_edge"]/crop_pct)`，然后图像将被裁剪到 `(size["shortest_edge"], size["shortest_edge"])`。仅在 `do_resize` 设置为 `True` 时有效。可以被 `preprocess` 方法中的 `size` 覆盖。
        crop_pct (`float` *optional*, defaults to 224 / 256):
            要裁剪图像的百分比。仅在 `do_resize` 为 `True` 且 size < 384 时有效。可以被 `preprocess` 方法中的 `crop_pct` 覆盖。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            如果调整图像大小，则使用的重采样滤波器。可以被 `preprocess` 方法中的 `resample` 覆盖。
        do_rescale (`bool`, *optional*, defaults to `True`):
            是否按照指定的比例 `rescale_factor` 重新缩放图像。可以被 `preprocess` 方法中的 `do_rescale` 覆盖。
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            如果重新缩放图像，则使用的缩放因子。可以被 `preprocess` 方法中的 `rescale_factor` 覆盖。
        do_normalize (`bool`, *optional*, defaults to `True`):
            是否对图像进行归一化。可以被 `preprocess` 方法中的 `do_normalize` 覆盖。
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            如果规范化图像，则使用的均值。这是一个长度等于图像通道数的浮点数或浮点数列表。可以被 `preprocess` 方法中的 `image_mean` 覆盖。
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            如果规范化图像，则使用的标准差。这是一个长度等于图像通道数的浮点数或浮点数列表。可以被 `preprocess` 方法中的 `image_std` 覆盖。
    """

    model_input_names = ["pixel_values"]
    # 图像处理类的初始化函数
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行调整大小的操作，默认为 True
        size: Dict[str, int] = None,  # 调整大小的目标尺寸，默认为 None
        crop_pct: float = None,  # 按比例裁剪图像的比例，默认为 None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像调整大小时的插值方法，默认为 BILINEAR
        do_rescale: bool = True,  # 是否进行图像的重新缩放操作，默认为 True
        rescale_factor: Union[int, float] = 1 / 255,  # 图像重新缩放的因子，默认为 1 / 255
        do_normalize: bool = True,  # 是否进行图像的归一化操作，默认为 True
        image_mean: Optional[Union[float, List[float]]] = None,  # 图像归一化的均值，默认为 None
        image_std: Optional[Union[float, List[float]]] = None,  # 图像归一化的标准差，默认为 None
        **kwargs,
    ) -> None:
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        # 若给定的 size 为 None，则设置默认的目标尺寸为 {"shortest_edge": 384}
        size = size if size is not None else {"shortest_edge": 384}
        # 调整 size 的格式，确保 size 是字典类型，并设置 default_to_square 为 False
        size = get_size_dict(size, default_to_square=False)
    
        # 初始化各个参数
        self.do_resize = do_resize
        self.size = size  # 调整大小的目标尺寸
        self.crop_pct = crop_pct if crop_pct is not None else 224 / 256  # 按比例裁剪图像的比例
        self.resample = resample  # 图像调整大小时的插值方法
        self.do_rescale = do_rescale  # 是否进行图像的重新缩放操作
        self.rescale_factor = rescale_factor  # 图像重新缩放的因子
        self.do_normalize = do_normalize  # 是否进行图像的归一化操作
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 图像归一化的均值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 图像归一化的标准差
    
    # 对图像进行调整大小的函数
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数组
        size: Dict[str, int],  # 调整大小的目标尺寸
        crop_pct: float,  # 按比例裁剪图像的比例
        resample: PILImageResampling = PILImageResampling.BICUBIC,  # 图像调整大小时的插值方法
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 数据的格式，默认为 None
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据的格式，默认为 None
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary of the form `{"shortest_edge": int}`, specifying the size of the output image. If
                `size["shortest_edge"]` >= 384 image is resized to `(size["shortest_edge"], size["shortest_edge"])`.
                Otherwise, the smaller edge of the image will be matched to `int(size["shortest_edge"] / crop_pct)`,
                after which the image is cropped to `(size["shortest_edge"], size["shortest_edge"])`.
            crop_pct (`float`):
                Percentage of the image to crop. Only has an effect if size < 384.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred from the input
                image.
        """
        # 获取大小字典，确保不默认为正方形
        size = get_size_dict(size, default_to_square=False)
        # 如果大小字典中不包含 "shortest_edge" 键，则引发 ValueError
        if "shortest_edge" not in size:
            raise ValueError(f"Size dictionary must contain 'shortest_edge' key. Got {size.keys()}")
        # 获取最短边的长度
        shortest_edge = size["shortest_edge"]

        # 如果最短边长度小于 384
        if shortest_edge < 384:
            # 保持相同的比例，将最短边调整为 shortest_edge/crop_pct
            resize_shortest_edge = int(shortest_edge / crop_pct)
            # 获取调整大小后的图像大小
            resize_size = get_resize_output_image_size(
                image, size=resize_shortest_edge, default_to_square=False, input_data_format=input_data_format
            )
            # 调整图像大小
            image = resize(
                image=image,
                size=resize_size,
                resample=resample,
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )
            # 然后裁剪到 (shortest_edge, shortest_edge)
            return center_crop(
                image=image,
                size=(shortest_edge, shortest_edge),
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )
        else:
            # 当最短边大于或等于 384 时，执行拉伸操作（不进行裁剪）
            return resize(
                image,
                size=(shortest_edge, shortest_edge),
                resample=resample,
                data_format=data_format,
                input_data_format=input_data_format,
                **kwargs,
            )
    # 对输入的图像进行预处理的函数
    def preprocess(
        self,
        # 输入的图像数据
        images: ImageInput,
        # 是否进行调整大小
        do_resize: bool = None,
        # 图像调整大小的尺寸
        size: Dict[str, int] = None,
        # 裁剪比例
        crop_pct: float = None,
        # 重新采样方法
        resample: PILImageResampling = None,
        # 是否进行尺度调整
        do_rescale: bool = None,
        # 尺度调整因子
        rescale_factor: float = None,
        # 是否进行标准化
        do_normalize: bool = None,
        # 图像均值
        image_mean: Optional[Union[float, List[float]]] = None,
        # 图像标准差
        image_std: Optional[Union[float, List[float]]] = None,
        # 返回的张量类型
        return_tensors: Optional[Union[str, TensorType]] = None,
        # 数据格式
        data_format: ChannelDimension = ChannelDimension.FIRST,
        # 输入数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        # 其他参数
        **kwargs,
```