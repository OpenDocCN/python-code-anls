# `.\models\layoutlmv3\image_processing_layoutlmv3.py`

```py
# 设置编码格式为 UTF-8
# 版权声明及许可证明，此代码受 Apache License, Version 2.0 许可，详见链接
"""LayoutLMv3 的图像处理器类。"""

# 导入必要的模块和类型定义
from typing import Dict, Iterable, Optional, Union

import numpy as np  # 导入 NumPy 库

# 导入自定义的图像处理工具函数和类
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
# 导入图像变换函数
from ...image_transforms import resize, to_channel_dimension_format, to_pil_image
# 导入图像相关的实用工具函数和常量
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
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
)
# 导入通用实用函数和类型定义
from ...utils import TensorType, is_pytesseract_available, is_vision_available, logging, requires_backends

# 如果 Vision 相关库可用，则导入 PIL 库
if is_vision_available():
    import PIL

# 如果安装了 pytesseract 库，则导入该库
if is_pytesseract_available():
    import pytesseract

# 获取日志记录器
logger = logging.get_logger(__name__)


def normalize_box(box, width, height):
    """将边界框的坐标归一化为 [0, 1000] 的范围内。

    Args:
        box (list): 边界框的坐标 [left, top, right, bottom]。
        width (int): 图像宽度。
        height (int): 图像高度。

    Returns:
        list: 归一化后的边界框坐标 [left_norm, top_norm, right_norm, bottom_norm]。
    """
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def apply_tesseract(
    image: np.ndarray,
    lang: Optional[str],
    tesseract_config: Optional[str],
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """对文档图像应用 Tesseract OCR，并返回识别的单词及归一化的边界框。

    Args:
        image (np.ndarray): 输入的图像数据。
        lang (Optional[str]): OCR 使用的语言设置。
        tesseract_config (Optional[str]): Tesseract 配置选项。
        input_data_format (Optional[Union[ChannelDimension, str]]): 输入图像的通道格式。

    Returns:
        None
    """
    # 将 NumPy 数组转换为 PIL 图像对象
    pil_image = to_pil_image(image, input_data_format=input_data_format)
    # 获取 PIL 图像的宽度和高度
    image_width, image_height = pil_image.size
    # 使用 pytesseract 库进行 OCR，返回识别的单词及其相关数据
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type="dict", config=tesseract_config)
    # 解析 OCR 结果中的单词、左上角坐标、宽度和高度信息
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # 过滤掉空单词及其对应的坐标信息
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    # 将坐标转换为 (left, top, right, bottom) 格式
    # 初始化空列表，用于存储计算得到的实际边界框坐标
    actual_boxes = []
    # 使用 zip 函数迭代 left, top, width, height 四个列表，并依次取出对应的 x, y, w, h 值
    for x, y, w, h in zip(left, top, width, height):
        # 计算每个边界框的实际坐标，格式为 [左上角 x 坐标, 左上角 y 坐标, 右下角 x 坐标, 右下角 y 坐标]
        actual_box = [x, y, x + w, y + h]
        # 将计算得到的实际边界框坐标添加到 actual_boxes 列表中
        actual_boxes.append(actual_box)
    
    # 最终，对边界框进行归一化处理
    normalized_boxes = []
    # 遍历每个实际边界框，调用 normalize_box 函数对其进行归一化处理，并将处理后的结果添加到 normalized_boxes 列表中
    for box in actual_boxes:
        normalized_boxes.append(normalize_box(box, image_width, image_height))
    
    # 断言确保识别出的单词数量与归一化后的边界框数量相等，否则抛出异常信息 "Not as many words as there are bounding boxes"
    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"
    
    # 返回识别出的单词列表和归一化后的边界框列表作为结果
    return words, normalized_boxes
class LayoutLMv3ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LayoutLMv3 image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to `(size["height"], size["width"])`. Can be
            overridden by `do_resize` in `preprocess`.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in `preprocess`.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image's pixel values by the specified `rescale_value`. Can be overridden by
            `do_rescale` in `preprocess`.
        rescale_factor (`float`, *optional*, defaults to 1 / 255):
            Value by which the image's pixel values are rescaled. Can be overridden by `rescale_factor` in
            `preprocess`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`Iterable[float]` or `float`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`Iterable[float]` or `float`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        apply_ocr (`bool`, *optional*, defaults to `True`):
            Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
            the `apply_ocr` parameter in the `preprocess` method.
        ocr_lang (`str`, *optional*):
            The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
            used. Can be overridden by the `ocr_lang` parameter in the `preprocess` method.
        tesseract_config (`str`, *optional*):
            Any additional custom configuration flags that are forwarded to the `config` parameter when calling
            Tesseract. For example: '--psm 6'. Can be overridden by the `tesseract_config` parameter in the
            `preprocess` method.
    """

    # 定义 LayoutLMv3 图像处理器类，继承自 BaseImageProcessor 类

    # 模型输入的名称列表，仅包含 "pixel_values"
    model_input_names = ["pixel_values"]
    # 初始化函数，用于初始化图像处理器对象的各种参数和属性
    def __init__(
        self,
        do_resize: bool = True,  # 是否进行图像大小调整的标志，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，包含高度和宽度，默认为None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像重采样方法，默认为双线性插值
        do_rescale: bool = True,  # 是否进行图像像素值缩放的标志，默认为True
        rescale_value: float = 1 / 255,  # 图像像素值缩放的因子，默认为1/255
        do_normalize: bool = True,  # 是否进行图像标准化的标志，默认为True
        image_mean: Union[float, Iterable[float]] = None,  # 图像标准化的均值，默认为IMAGENET_STANDARD_MEAN
        image_std: Union[float, Iterable[float]] = None,  # 图像标准化的标准差，默认为IMAGENET_STANDARD_STD
        apply_ocr: bool = True,  # 是否应用OCR识别的标志，默认为True
        ocr_lang: Optional[str] = None,  # OCR识别使用的语言，默认为None
        tesseract_config: Optional[str] = "",  # Tesseract OCR的配置参数，默认为空字符串
        **kwargs,  # 其他关键字参数
    ) -> None:
        # 调用父类的初始化函数，传入其他关键字参数
        super().__init__(**kwargs)
        # 如果size参数为None，则设置默认的图像大小为{"height": 224, "width": 224}
        size = size if size is not None else {"height": 224, "width": 224}
        # 调用get_size_dict函数，确保size参数是符合要求的字典形式
        size = get_size_dict(size)

        # 初始化各个属性
        self.do_resize = do_resize  # 是否进行图像大小调整
        self.size = size  # 图像大小的字典
        self.resample = resample  # 图像重采样方法
        self.do_rescale = do_rescale  # 是否进行图像像素值缩放
        self.rescale_factor = rescale_value  # 图像像素值缩放因子
        self.do_normalize = do_normalize  # 是否进行图像标准化
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN  # 图像标准化的均值
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD  # 图像标准化的标准差
        self.apply_ocr = apply_ocr  # 是否应用OCR识别
        self.ocr_lang = ocr_lang  # OCR识别使用的语言
        self.tesseract_config = tesseract_config  # Tesseract OCR的配置参数
        # 图像处理器对象的有效键列表，包括各个属性名称和其他通用参数
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "apply_ocr",
            "ocr_lang",
            "tesseract_config",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    # 从transformers.models.vit.image_processing_vit.ViTImageProcessor.resize中复制而来的函数
    def resize(
        self,
        image: np.ndarray,  # 输入的图像数据，为NumPy数组
        size: Dict[str, int],  # 调整后的图像大小字典，包含高度和宽度
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 图像重采样方法，默认为双线性插值
        data_format: Optional[Union[str, ChannelDimension]] = None,  # 输出数据格式
        input_data_format: Optional[Union[str, ChannelDimension]] = None,  # 输入数据格式
        **kwargs,  # 其他关键字参数
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
        # 获取标准化后的尺寸字典
        size = get_size_dict(size)
        # 检查尺寸字典中是否包含必要的键
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 将尺寸字典转换为输出尺寸元组
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

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample=None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Union[float, Iterable[float]] = None,
        image_std: Union[float, Iterable[float]] = None,
        apply_ocr: bool = None,
        ocr_lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```