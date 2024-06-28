# `.\models\layoutlmv2\image_processing_layoutlmv2.py`

```
# 定义一个名为 normalize_box 的函数，用于将边界框归一化为相对于图像宽高的千分比
def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),    # 左上角 x 坐标的归一化值
        int(1000 * (box[1] / height)),   # 左上角 y 坐标的归一化值
        int(1000 * (box[2] / width)),    # 右下角 x 坐标的归一化值
        int(1000 * (box[3] / height)),   # 右下角 y 坐标的归一化值
    ]

# 定义一个名为 apply_tesseract 的函数，用于在文档图像上应用 Tesseract OCR，并返回识别的单词和归一化的边界框
def apply_tesseract(
    image: np.ndarray,
    lang: Optional[str],
    tesseract_config: Optional[str] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""
    tesseract_config = tesseract_config if tesseract_config is not None else ""

    # 将 numpy 数组的图像转换为 PIL 图像格式
    pil_image = to_pil_image(image, input_data_format=input_data_format)
    # 获取 PIL 图像的宽度和高度
    image_width, image_height = pil_image.size
    # 使用 pytesseract 对 PIL 图像进行 OCR，返回识别的文本和详细信息字典
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type="dict", config=tesseract_config)
    # 解析识别结果中的文本、左上角坐标、宽度和高度信息
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # 过滤掉空白文本和对应的坐标
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]
    # 将坐标转换为 (左, 上, 左+宽, 上+高) 的格式
    actual_boxes = []
    for x, y, w, h in zip(left, top, width, height):
        actual_box = [x, y, x + w, y + h]  # 计算每个边界框的左、上、右、下坐标
        actual_boxes.append(actual_box)  # 将计算得到的边界框添加到列表中

    # 最后，对边界框进行归一化处理
    normalized_boxes = []
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))  # 调用归一化函数处理边界框

    # 断言确保单词列表的长度与归一化后的边界框列表长度相等
    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"

    # 返回处理后的单词列表和归一化后的边界框列表
    return words, normalized_boxes
    r"""
    Constructs a LayoutLMv2 image processor.
    构造一个 LayoutLMv2 图像处理器。

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to `(size["height"], size["width"])`. Can be
            overridden by `do_resize` in `preprocess`.
            是否将图像的 (height, width) 尺寸调整为 `(size["height"], size["width"])`。可以在 `preprocess` 中通过 `do_resize` 覆盖。

        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in `preprocess`.
            调整大小后的图像尺寸。可以在 `preprocess` 中通过 `size` 覆盖。

        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
            如果调整图像大小，要使用的重采样滤波器。可以在 `preprocess` 方法中通过 `resample` 参数覆盖。

        apply_ocr (`bool`, *optional*, defaults to `True`):
            Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
            `apply_ocr` in `preprocess`.
            是否应用 Tesseract OCR 引擎来获取单词 + 标准化边界框。可以在 `preprocess` 中通过 `apply_ocr` 覆盖。

        ocr_lang (`str`, *optional*):
            The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
            used. Can be overridden by `ocr_lang` in `preprocess`.
            Tesseract OCR 引擎使用的语言，使用 ISO 代码指定。默认使用英语。可以在 `preprocess` 中通过 `ocr_lang` 覆盖。

        tesseract_config (`str`, *optional*, defaults to `""`):
            Any additional custom configuration flags that are forwarded to the `config` parameter when calling
            Tesseract. For example: '--psm 6'. Can be overridden by `tesseract_config` in `preprocess`.
            调用 Tesseract 时转发给 `config` 参数的任何额外自定义配置标志。例如：'--psm 6'。可以在 `preprocess` 中通过 `tesseract_config` 覆盖。
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        apply_ocr: bool = True,
        ocr_lang: Optional[str] = None,
        tesseract_config: Optional[str] = "",
        **kwargs,
    ) -> None:
        # Call the constructor of the superclass (BaseImageProcessor) with any additional keyword arguments (**kwargs)
        super().__init__(**kwargs)

        # Set default size if not provided
        size = size if size is not None else {"height": 224, "width": 224}
        size = get_size_dict(size)  # Normalize size to a dictionary format

        # Initialize instance variables with provided or default values
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.apply_ocr = apply_ocr
        self.ocr_lang = ocr_lang
        self.tesseract_config = tesseract_config

        # Define a list of valid keys for the processor configuration
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "apply_ocr",
            "ocr_lang",
            "tesseract_config",
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
        size = get_size_dict(size)  # 调用函数 `get_size_dict` 将 `size` 参数转换为标准格式的字典
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        output_size = (size["height"], size["width"])  # 根据 `size` 字典确定输出图像的尺寸
        return resize(
            image,  # 调用 `resize` 函数对输入图像进行调整大小操作
            size=output_size,  # 设置调整后的图像尺寸
            resample=resample,  # 设置图像调整大小时使用的重采样方法
            data_format=data_format,  # 设置输出图像的通道格式
            input_data_format=input_data_format,  # 设置输入图像的通道格式，如果未指定则从输入图像推断
            **kwargs,  # 允许传递其他关键字参数给 `resize` 函数
        )
```