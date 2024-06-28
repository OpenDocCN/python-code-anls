# `.\models\layoutlmv3\processing_layoutlmv3.py`

```
"""
Processor class for LayoutLMv3.
"""

# 导入警告模块
import warnings
# 引入类型提示模块中的相关类型
from typing import List, Optional, Union

# 导入处理工具的混合处理器
from ...processing_utils import ProcessorMixin
# 导入基础的令牌化工具相关模块
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
# 导入张量类型
from ...utils import TensorType

# 定义 LayoutLMv3Processor 类，继承自 ProcessorMixin
class LayoutLMv3Processor(ProcessorMixin):
    r"""
    Constructs a LayoutLMv3 processor which combines a LayoutLMv3 image processor and a LayoutLMv3 tokenizer into a
    single processor.

    [`LayoutLMv3Processor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv3ImageProcessor`] to resize and normalize document images, and optionally applies OCR to
    get words and normalized bounding boxes. These are then provided to [`LayoutLMv3Tokenizer`] or
    [`LayoutLMv3TokenizerFast`], which turns the words and bounding boxes into token-level `input_ids`,
    `attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
    into token-level `labels` for token classification tasks (such as FUNSD, CORD).

    Args:
        image_processor (`LayoutLMv3ImageProcessor`, *optional*):
            An instance of [`LayoutLMv3ImageProcessor`]. The image processor is a required input.
        tokenizer (`LayoutLMv3Tokenizer` or `LayoutLMv3TokenizerFast`, *optional*):
            An instance of [`LayoutLMv3Tokenizer`] or [`LayoutLMv3TokenizerFast`]. The tokenizer is a required input.
    """

    # 定义类属性 attributes
    attributes = ["image_processor", "tokenizer"]
    # 定义图像处理器类的名称
    image_processor_class = "LayoutLMv3ImageProcessor"
    # 定义令牌化器类的名称，可以是 LayoutLMv3Tokenizer 或 LayoutLMv3TokenizerFast
    tokenizer_class = ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast")
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 初始化函数，用于创建类的实例
        feature_extractor = None
        if "feature_extractor" in kwargs:
            # 如果传入了 `feature_extractor` 参数，发出警告，此参数在 v5 版本中将被移除，请使用 `image_processor` 替代
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            # 将 `feature_extractor` 参数的值从 `kwargs` 中弹出并保存
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果未指定 `image_processor`，则尝试使用 `feature_extractor`
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果最终 `image_processor` 仍然为 None，则抛出数值错误
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果未指定 `tokenizer`，则抛出数值错误
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化函数，传入 `image_processor` 和 `tokenizer`
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        # 调用对象时执行的函数，支持多种参数组合，具体含义参见参数列表
        ...

    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # 获取溢出样本对应的图像数据，确保每个 `input_ids` 样本都映射到相应的图像
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            # 根据溢出样本映射，添加对应的图像数据到列表中
            images_with_overflow.append(images[sample_idx])

        # 检查溢出图像列表长度是否与映射长度一致，若不一致则抛出数值错误
        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        # 返回带有溢出图像的列表
        return images_with_overflow

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 批量解码方法，将所有参数转发给 `PreTrainedTokenizer` 的 `batch_decode` 方法
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发到 PreTrainedTokenizer 的 `decode` 方法，并返回结果
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表
    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "attention_mask", "pixel_values"]

    # 返回特征提取器的类。警告：`feature_extractor_class` 将在 v5 中移除，建议使用 `image_processor_class`
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回特征提取器。警告：`feature_extractor` 将在 v5 中移除，建议使用 `image_processor`
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```