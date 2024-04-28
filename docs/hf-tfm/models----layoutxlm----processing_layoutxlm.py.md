# `.\models\layoutxlm\processing_layoutxlm.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指明使用的是 Apache 许可证 2.0 版本
# 如果符合许可证条件，可以使用此文件，详细条件可查看链接 http://www.apache.org/licenses/LICENSE-2.0
"""
Processor class for LayoutXLM.
"""
# 导入警告模块
import warnings
# 导入必要的类型提示
from typing import List, Optional, Union
# 导入处理工具类 ProcessorMixin
from ...processing_utils import ProcessorMixin
# 导入基础的标记化处理工具和类型
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
# 导入张量类型
from ...utils import TensorType


class LayoutXLMProcessor(ProcessorMixin):
    r"""
    Constructs a LayoutXLM processor which combines a LayoutXLM image processor and a LayoutXLM tokenizer into a single
    processor.

    [`LayoutXLMProcessor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv2ImageProcessor`] to resize document images to a fixed size, and optionally applies OCR to
    get words and normalized bounding boxes. These are then provided to [`LayoutXLMTokenizer`] or
    [`LayoutXLMTokenizerFast`], which turns the words and bounding boxes into token-level `input_ids`,
    `attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
    into token-level `labels` for token classification tasks (such as FUNSD, CORD).

    Args:
        image_processor (`LayoutLMv2ImageProcessor`, *optional*):
            An instance of [`LayoutLMv2ImageProcessor`]. The image processor is a required input.
        tokenizer (`LayoutXLMTokenizer` or `LayoutXLMTokenizerFast`, *optional*):
            An instance of [`LayoutXLMTokenizer`] or [`LayoutXLMTokenizerFast`]. The tokenizer is a required input.
    """

    # 定义类的属性，指定图像处理器和分词器
    attributes = ["image_processor", "tokenizer"]
    # 指定图像处理器的类名
    image_processor_class = "LayoutLMv2ImageProcessor"
    # 指定分词器的类名
    tokenizer_class = ("LayoutXLMTokenizer", "LayoutXLMTokenizerFast")
```  
    # 初始化方法，接受图像处理器和标记器等参数
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 如果参数中包含 `feature_extractor`，发出警告，该参数将在v5中移除，请使用 `image_processor` 替代
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            # 从参数中移除 `feature_extractor`
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果没有指定 `image_processor`，使用 `feature_extractor`
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果 `image_processor` 仍为 None，抛出数值错误
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果没有指定 `tokenizer`，抛出数值错误
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，传入图像处理器和标记器
        super().__init__(image_processor, tokenizer)

    # 调用方法，接受多个参数
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
    # 获取包含溢出的图像，确保每个 `input_ids` 样本都映射到相应的图像
    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        return images_with_overflow

    # 批量解码方法，将参数传递给 PreTrainedTokenizer 的 `batch_decode` 方法，并返回结果
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数传递给PreTrainedTokenizer的`~PreTrainedTokenizer.decode`方法。有关更多信息，请参考此方法的文档字符串。
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表
    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "attention_mask", "image"]

    # 返回特征提取器的类
    @property
    def feature_extractor_class(self):
        # 发出警告，提示`feature_extractor_class`已过时，并将在v5中移除。请改用`image_processor_class`。
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回特征提取器
    @property
    def feature_extractor(self):
        # 发出警告，提示`feature_extractor`已过时，并将在v5中移除。请改用`image_processor`。
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```