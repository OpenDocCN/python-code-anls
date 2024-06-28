# `.\models\layoutlmv2\processing_layoutlmv2.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for LayoutLMv2.
"""

import warnings
from typing import List, Optional, Union

# 导入处理工具和数据结构定义
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class LayoutLMv2Processor(ProcessorMixin):
    r"""
    Constructs a LayoutLMv2 processor which combines a LayoutLMv2 image processor and a LayoutLMv2 tokenizer into a
    single processor.

    [`LayoutLMv2Processor`] offers all the functionalities you need to prepare data for the model.

    It first uses [`LayoutLMv2ImageProcessor`] to resize document images to a fixed size, and optionally applies OCR to
    get words and normalized bounding boxes. These are then provided to [`LayoutLMv2Tokenizer`] or
    [`LayoutLMv2TokenizerFast`], which turns the words and bounding boxes into token-level `input_ids`,
    `attention_mask`, `token_type_ids`, `bbox`. Optionally, one can provide integer `word_labels`, which are turned
    into token-level `labels` for token classification tasks (such as FUNSD, CORD).

    Args:
        image_processor (`LayoutLMv2ImageProcessor`, *optional*):
            An instance of [`LayoutLMv2ImageProcessor`]. The image processor is a required input.
        tokenizer (`LayoutLMv2Tokenizer` or `LayoutLMv2TokenizerFast`, *optional*):
            An instance of [`LayoutLMv2Tokenizer`] or [`LayoutLMv2TokenizerFast`]. The tokenizer is a required input.
    """

    # 定义类属性，这些属性用于标识 processor 的特征
    attributes = ["image_processor", "tokenizer"]
    # 指定图片处理器类的名称
    image_processor_class = "LayoutLMv2ImageProcessor"
    # 指定 tokenizer 类的名称，支持两种类型
    tokenizer_class = ("LayoutLMv2Tokenizer", "LayoutLMv2TokenizerFast")
    # 初始化方法，接受图像处理器（image_processor）、分词器（tokenizer）等参数
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 如果参数中包含 'feature_extractor'，发出警告并将其移除，建议使用 'image_processor' 替代
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果未显式指定图像处理器，则尝试使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果最终图像处理器仍为 None，则抛出数值错误
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果分词器为 None，则抛出数值错误
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，传递图像处理器和分词器作为参数
        super().__init__(image_processor, tokenizer)

    # 调用实例时执行的方法，用于将输入的图像及相关信息转换为模型可接受的格式
    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
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
        """
        批量处理图像及其相关信息，将其转换为模型可以处理的格式。参数详细说明可以参考 `PreTrainedTokenizer.batch_decode` 方法的文档字符串。
        """
        # 实际调用分词器的 batch_decode 方法来处理输入数据
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 获取溢出图像的方法，确保每个 `input_ids` 样本都对应其相应的图像
    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        images_with_overflow = []
        # 根据溢出到样本映射，将相应索引的图像加入到结果列表中
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        # 检查结果列表的长度与溢出映射的长度是否一致，否则抛出数值错误
        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        # 返回包含溢出图像的列表
        return images_with_overflow
    # 将所有参数转发到 PreTrainedTokenizer 的 `decode` 方法中，并返回结果
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，包括 input_ids、bbox、token_type_ids、attention_mask 和 image
    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "token_type_ids", "attention_mask", "image"]

    # 返回特征提取器的类。显示警告，告知 `feature_extractor_class` 将在 v5 版本中删除，建议使用 `image_processor_class` 替代
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回特征提取器。显示警告，告知 `feature_extractor` 将在 v5 版本中删除，建议使用 `image_processor` 替代
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```