# `.\models\layoutlmv3\processing_layoutlmv3.py`

```py
# coding=utf-8
# 版权归 2022 年 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证 2.0 版（“许可证”）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据“原样”分发，不附带任何明示或暗示的保证
# 或适用于特定目的的条件。查看许可证
# 获取特定语言的权限和限制。
"""
LayoutLMv3 的 Processor 类。
"""

import warnings
from typing import List, Optional, Union

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType

class LayoutLMv3Processor(ProcessorMixin):
    r"""
    构建一个 LayoutLMv3 处理器，将 LayoutLMv3 图像处理器和 LayoutLMv3 分词器结合为单个处理器。

    [`LayoutLMv3Processor`] 提供了准备模型数据所需的所有功能。

    首先使用 [`LayoutLMv3ImageProcessor`] 调整和标准化文档图像，可选择应用 OCR 来获取单词和标准化边界框。
    然后将它们提供给 [`LayoutLMv3Tokenizer`] 或 [`LayoutLMv3TokenizerFast`]，将单词和边界框转换为标记级的 `input_ids`、`attention_mask`、`token_type_ids`、`bbox`。
    可选地，可以提供整数 `word_labels`，它们会转换为用于标记分类任务（如 FUNSD、CORD）的标记级 `labels`。

    Args:
        image_processor (`LayoutLMv3ImageProcessor`, *可选*):
            [`LayoutLMv3ImageProcessor`] 的实例。图像处理器是必需的输入。
        tokenizer (`LayoutLMv3Tokenizer` 或 `LayoutLMv3TokenizerFast`, *可选*):
            [`LayoutLMv3Tokenizer`] 或 [`LayoutLMv3TokenizerFast`] 的实例。分词器是必需的输入。
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LayoutLMv3ImageProcessor"
    tokenizer_class = ("LayoutLMv3Tokenizer", "LayoutLMv3TokenizerFast")
```  
    # 初始化方法，用于初始化对象的属性和参数
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 如果参数中包含"feature_extractor"，则发出警告并将其赋值给feature_extractor
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")
        
        # 根据优先级确定image_processor的取值
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 检查是否指定了image_processor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 检查是否指定了tokenizer
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
    
        # 调用父类初始化方法，传入image_processor和tokenizer
        super().__init__(image_processor, tokenizer)
    
    # 调用方法，根据输入参数处理返回结果
    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]] = None,
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
    
    # 获取溢出图片数据的方法
    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # 确保每个`input_ids`样本都映射到相应的图像，以防溢出
        images_with_overflow = []
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])
    
        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )
    
        return images_with_overflow
    
    # 批处理解码方法，将所有��数传递给PreTrainedTokenizer的batch_decode方法
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发给 PreTrainedTokenizer 的 `~PreTrainedTokenizer.decode` 方法
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，包括 input_ids、bbox、attention_mask 和 pixel_values
    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "attention_mask", "pixel_values"]

    # 返回特征提取器的类，同时发出警告提示该属性在 v5 版本中将被移除，建议使用 `image_processor_class` 替代
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回特征提取器，同时发出警告提示该属性在 v5 版本中将被移除，建议使用 `image_processor` 替代
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```