# `.\models\layoutlmv3\image_processing_layoutlmv3.py`

```
# 设置字符编码为 UTF-8
# 版权声明
# 在 Apache 许可证 2.0 下授权使用此代码
# 只有在合规使用许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，根据这个许可证分发的软件都是基于"按现状提供"分发的
# ，没有任何种类的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取更多关于权限和限制的详细信息
"""LayoutLMv3 的图像处理类。"""

# 引入必要的类型注解和模块
from typing import Dict, Iterable, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import resize, to_channel_dimension_format, to_pil_image
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
)
from ...utils import TensorType, is_pytesseract_available, is_vision_available, logging, requires_backends

# 如果有视觉处理模块可用，导入 PIL 模块
if is_vision_available():
    import PIL

# 有 pytesseract 的软依赖，导入 pytesseract 模块
if is_pytesseract_available():
    import pytesseract

# 获取 logger 对象
logger = logging.get_logger(__name__)


# 定义归一化边界框的函数
def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


# 应用 Tesseract OCR 在文档图像上，并返回识别的单词和归一化的边界框
def apply_tesseract(
    image: np.ndarray,
    lang: Optional[str],
    tesseract_config: Optional[str],
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    """Applies Tesseract OCR on a document image, and returns recognized words + normalized bounding boxes."""

    # 将图像转为 PIL 图像格式
    pil_image = to_pil_image(image, input_data_format=input_data_format)
    image_width, image_height = pil_image.size
    # 使用 Tesseract 进行 OCR
    data = pytesseract.image_to_data(pil_image, lang=lang, output_type="dict", config=tesseract_config)
    words, left, top, width, height = data["text"], data["left"], data["top"], data["width"], data["height"]

    # 过滤掉空白单词和对应的坐标
    irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
    words = [word for idx, word in enumerate(words) if idx not in irrelevant_indices]
    left = [coord for idx, coord in enumerate(left) if idx not in irrelevant_indices]
    top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
    width = [coord for idx, coord in enumerate(width) if idx not in irrelevant_indices]
    height = [coord for idx, coord in enumerate(height) if idx not in irrelevant_indices]

    # 将坐标转为 (left, top, left+width, top+height) 格式
    actual_boxes = []
    # 使用 zip 函数同时迭代 left、top、width 和 height 这四个列表，得到每个矩形框的左上角坐标和宽高
    for x, y, w, h in zip(left, top, width, height):
        # 计算当前矩形框的实际坐标，右下角坐标 = 左上角坐标 + 宽高
        actual_box = [x, y, x + w, y + h]
        # 将当前矩形框的实际坐标加入到 actual_boxes 列表中
        actual_boxes.append(actual_box)

    # 最终，对边界框进行归一化处理
    normalized_boxes = []
    # 遍历所有实际坐标框
    for box in actual_boxes:
        # 调用 normalize_box 函数，对当前矩形框进行归一化处理，传入参数为矩形框的坐标和图片的宽高
        normalized_boxes.append(normalize_box(box, image_width, image_height))

    # 使用断言确保识别出的单词数量与归一化后的边界框数量相同
    assert len(words) == len(normalized_boxes), "Not as many words as there are bounding boxes"

    # 返回识别出的单词列表和归一化后的边界框列表
    return words, normalized_boxes
class LayoutLMv3ImageProcessor(BaseImageProcessor):
    r"""
    Constructs a LayoutLMv3 image processor.
    构造一个 LayoutLMv3 图像处理器。

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to `(size["height"], size["width"])`. Can be
            overridden by `do_resize` in `preprocess`.
            是否将图像的 (height, width) 尺寸调整为 `(size["height"], size["width"])`。可以通过 `preprocess` 中的 `do_resize` 参数进行覆盖。

        size (`Dict[str, int]` *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after resizing. Can be overridden by `size` in `preprocess`.
            调整大小后图像的尺寸。可以通过 `preprocess` 中的 `size` 参数进行覆盖。

        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
            如果调整图像大小，要使用的重采样滤波器。可以通过 `preprocess` 中的 `resample` 参数进行覆盖。

        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image's pixel values by the specified `rescale_value`. Can be overridden by
            `do_rescale` in `preprocess`.
            是否按指定的 `rescale_value` 缩放图像的像素值。可以通过 `preprocess` 中的 `do_rescale` 参数进行覆盖。

        rescale_factor (`float`, *optional*, defaults to 1 / 255):
            Value by which the image's pixel values are rescaled. Can be overridden by `rescale_factor` in
            `preprocess`.
            图像的像素值缩放的值。可以通过 `preprocess` 中的 `rescale_factor` 参数进行覆盖。

        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
            是否对图像进行归一化。可以通过 `preprocess` 方法中的 `do_normalize` 参数进行覆盖。

        image_mean (`Iterable[float]` or `float`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
            如果对图像进行归一化要使用的平均值。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以通过 `preprocess` 方法中的 `image_mean` 参数进行覆盖。

        image_std (`Iterable[float]` or `float`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            如果对图像进行归一化要使用的标准差。这是一个浮点数或与图像通道数相同长度的浮点数列表。可以通过 `preprocess` 方法中的 `image_std` 参数进行覆盖。

        apply_ocr (`bool`, *optional*, defaults to `True`):
            Whether to apply the Tesseract OCR engine to get words + normalized bounding boxes. Can be overridden by
            the `apply_ocr` parameter in the `preprocess` method.
            是否应用 Tesseract OCR 引擎以获取单词和归一化的边界框。可以通过 `preprocess` 方法中的 `apply_ocr` 参数进行覆盖。

        ocr_lang (`str`, *optional*):
            The language, specified by its ISO code, to be used by the Tesseract OCR engine. By default, English is
            used. Can be overridden by the `ocr_lang` parameter in the `preprocess` method.
            Tesseract OCR 引擎要使用的语言，使用其 ISO 代码指定。默认情况下使用英语。可以通过 `preprocess` 方法中的 `ocr_lang` 参数进行覆盖。

        tesseract_config (`str`, *optional*):
            Any additional custom configuration flags that are forwarded to the `config` parameter when calling
            Tesseract. For example: '--psm 6'. Can be overridden by the `tesseract_config` parameter in the
            `preprocess` method.
            调用 Tesseract 时转发到 `config` 参数的任何其他自定义配置标志。例如：'--psm 6'。可以通过 `preprocess` 方法中的 `tesseract_config` 参数进行覆盖。
    """

    model_input_names = ["pixel_values"]
    # 定义初始化函数，设置类的属性值和默认参数
    def __init__(
        self,
        do_resize: bool = True,  # 是否执行图像调整，默认为True
        size: Dict[str, int] = None,  # 图像大小的字典，默认为None
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        do_rescale: bool = True,  # 是否执行图像缩放，默认为True
        rescale_value: float = 1 / 255,  # 缩放因子，默认值为1/255
        do_normalize: bool = True,  # 是否进行归一化，默认为True
        image_mean: Union[float, Iterable[float]] = None,  # 图像均值，默认为None
        image_std: Union[float, Iterable[float]] = None,  # 图像标准差，默认为None
        apply_ocr: bool = True,  # 是否执行OCR，默认为True
        ocr_lang: Optional[str] = None,  # OCR语言，默认为None
        tesseract_config: Optional[str] = "",  # Tesseract配置，默认为""
        **kwargs,  # 其他关键字参数
    ) -> None:  # 返回空值
        super().__init__(**kwargs)  # 调用父类初始化函数
        size = size if size is not None else {"height": 224, "width": 224}  # 如果size为None，则设置默认大小为224x224
        size = get_size_dict(size)  # 转换size为字典形式

        # 设置各属性的初始值
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_value
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD
        self.apply_ocr = apply_ocr
        self.ocr_lang = ocr_lang
        self.tesseract_config = tesseract_config

    # 从transformers.models.vit.image_processing_vit.ViTImageProcessor中复制的resize函数
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],  # 图像大小的字典
        resample: PILImageResampling = PILImageResampling.BILINEAR,  # 重采样方法，默认为双线性插值
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
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
        # Convert `size` to a dictionary if not already
        size = get_size_dict(size)
        # Check if both "height" and "width" keys are present in the size dictionary
        if "height" not in size or "width" not in size:
            # Raise a ValueError if any of the keys is missing
            raise ValueError(f"The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}")
        # Extract the height and width from the size dictionary
        output_size = (size["height"], size["width"])
        # Call the `resize` function with specified arguments
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