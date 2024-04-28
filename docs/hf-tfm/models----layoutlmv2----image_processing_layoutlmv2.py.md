# `.\models\layoutlmv2\image_processing_layoutlmv2.py`

```py
# 设置编码格式为 utf-8
# 版权声明
# 根据 Apache License, Version 2.0 许可获取许可证副本
# 如果未满足适用法律的要求或没有书面同意，在"AS IS"的基础上分发软件，不提供任何明示或暗示的担保和条件
# 请查阅具体语言管理权限和限制的许可证
"""LayoutLMv2的图像处理类的注释。"""

from typing import Dict, Optional, Union    # 从typing库中导入Dict, Optional, Union类型

import numpy as np    # 导入numpy库

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict    # 从...image_processing_utils库中导入BaseImageProcessor, BatchFeature, get_size_dict函数
from ...image_transforms import flip_channel_order, resize, to_channel_dimension_format, to_pil_image    # 从...image_transforms库中导入flip_channel_order, resize, to_channel_dimension_format, to_pil_image函数
from ...image_utils import (    # 从...image_utils库中导入下列函数和类型
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_pytesseract_available, is_vision_available, logging, requires_backends    # 从...utils库中导入TensorType, is_pytesseract_available, is_vision_available, logging, requires_backends

if is_vision_available():    # 如果is_vision_available返回True
    import PIL    # 导入PIL库

# 软依赖
if is_pytesseract_available():    # 如果is_pytesseract_available返回True
    import pytesseract    # 导入pytesseract库

logger = logging.get_logger(__name__)    # 生成logger对象

def normalize_box(box, width, height):    # 定义normalize_box函数，参数为box, width, height
    return [    # 返回列表
        int(1000 * (box[0] / width)),    # 列表的第一项
        int(1000 * (box[1] / height)),    # 列表的第二项
        int(1000 * (box[2] / width)),    # 列表的第三项
        int(1000 * (box[3] / height)),    # 列表的第四项
    ]


def apply_tesseract(    # 定义apply_tesseract函数
    image: np.ndarray,    # 参数：image，类型为numpy数组
    lang: Optional[str],    # 参数：lang，类型为str或None
    tesseract_config: Optional[str] = None,    # 参数：tesseract_config，类型为str或None，默认为None
    input_data_format: Optional[Union[str, ChannelDimension]] = None,    # 参数：input_data_format，类型为str, ChannelDimension或None，默认为None
):    # 函数初始化
    """在文档图像上应用Tesseract OCR，并返回识别的单词+归一化的边界框。"""

    tesseract_config = tesseract_config if tesseract_config is not None else ""    # 如果tesseract_config不为None，则tesseract_config保持不变，否则赋值为空字符串

    # 应用OCR
    pil_image = to_pil_image(image, input_data_format=input_data_format)    # 将numpy数组转换成PIL图像
    image_width, image_height = pil_image.size    # 获取PIL图像的宽度和高度
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type="dict", config=tesseract_config)    # 将PIL图像应用于Tesseract OCR
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]    # 分别获取文本、左边界、顶边界、宽度、高度

    # 过滤空字符以及其对应的坐标
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]    # 筛选出空字符对应的索引
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]    # 筛选出非空字符
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]    # 筛选出非空字符的左边界坐标
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]    # 筛选出非空字符的顶边界坐标
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]    # 筛选出非空字符的宽度
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]    # 筛选出非空字符的高度

    # 将坐标转换为 (left, top, left+width, top+height) 格式
    actual_boxes = []    # 初始化actual_boxes列表
    # 遍历 left, top, width, height 四个列表，并将它们打包成元组，每次取一个元组赋值给 x, y, w, h
    for x, y, w, h in zip(left, top, width, height):
        # 根据 x, y, w, h 计算出实际的边界框坐标，形成一个列表
        actual_box = [x, y, x + w, y + h]
        # 将实际的边界框坐标添加到 actual_boxes 列表中
        actual_boxes.append(actual_box)

    # 最后，规范化边界框
    normalized_boxes = []
    for box in actual_boxes:
        # 调用 normalize_box 函数，将边界框坐标规范化，并添加到 normalized_boxes 列表
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    # 使用断言检查 words 列表的长度是否与 normalized_boxes 列表的长度相等，如果不相等则抛出异常
    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"

    # 返回 words 列表和 normalized_boxes 列表作为结果
    return words, normalized_boxes
# LayoutLMv2 图像处理器类，继承自 BaseImageProcessor 类
class LayoutLMv2ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LayoutLMv2 image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            是否调整图像的 (height, width) 尺寸为 `(size["height"], size["width"])`。可以被 `preprocess` 方法中的 `do_resize` 参数重写。
        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            调整尺寸后的图像尺寸。可以被 `preprocess` 方法中的 `size` 参数重写。
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            如果调整图像大小，使用的采样滤波器。可以被 `preprocess` 方法中的 `resample` 参数重写。
        apply_ocr (`bool`, *optional*, defaults to `True`):
            是否使用 Tesseract OCR 引擎获取单词 + 标准化边界框。可以被 `preprocess` 方法中的 `apply_ocr` 参数重写。
        ocr_lang (`str`, *optional*):
            Tesseract OCR 引擎使用的语言，以 ISO 代码指定。默认为英语。可以被 `preprocess` 方法中的 `ocr_lang` 参数重写。
        tesseract_config (`str`, *optional*, defaults to `""`):
            Tesseract 调用时传递给 `config` 参数的任何其他自定义配置标志。例如：'--psm 6'。可以被 `preprocess` 方法中的 `tesseract_config` 参数重写。
    """

    # 模型所需的输入名称
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
        # 调用父类 BaseImageProcessor 的初始化方法
        super().__init__(**kwargs)
        # 如果 size 为 None，则设为默认 {"height": 224, "width": 224}
        size = size if size is not None else {"height": 224, "width": 224}
        # 获取标准化的尺寸字典
        size = get_size_dict(size)

        # 初始化参数
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.apply_ocr = apply_ocr
        self.ocr_lang = ocr_lang
        self.tesseract_config = tesseract_config

    # 从 transformers.models.vit.image_processing_vit.ViTImageProcessor 中复制的 resize 方法
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    # 重新定义一个函数，用来改变输入图像的大小
    def resize_image(
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Union[ChannelDimension, str] = None,
        input_data_format: Union[ChannelDimension, str] = None,
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
        # 将 size 转换成标准格式的字典
        size = get_size_dict(size)
        # 如果 size 字典中不包含 "height" 或 "width" 键，则抛出异常
        if "height" not in size or "width" not in size:
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # 获取输出图像的大小
        output_size = (size["height"], size["width"])
        # 调用 resize 函数来改变图像大小，传入指定的参数
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    # 预处理函数，用来预处理输入的图像
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        apply_ocr: bool = None,
        ocr_lang: Optional[str] = None,
        tesseract_config: Optional[str] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
```