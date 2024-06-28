# `.\models\layoutxlm\processing_layoutxlm.py`

```
# coding=utf-8
# 设置文件编码为 UTF-8

# Copyright 2021 The HuggingFace Inc. team.
# 版权声明：2021 年由 HuggingFace 公司团队拥有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 进行许可

# you may not use this file except in compliance with the License.
# 除非遵循许可协议，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可协议副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，本软件按"原样"分发，不附带任何形式的明示或暗示担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可协议以了解特定语言的许可权限和限制

"""
Processor class for LayoutXLM.
"""
# 用于 LayoutXLM 的处理器类

import warnings
# 导入警告模块
from typing import List, Optional, Union
# 导入类型提示的必要模块

from ...processing_utils import ProcessorMixin
# 从父级目录的 processing_utils 模块中导入 ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
# 从父级目录的 tokenization_utils_base 模块中导入多个类和策略
from ...utils import TensorType
# 从父级目录的 utils 模块中导入 TensorType 类型

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
    # LayoutXLM 处理器类，结合 LayoutXLM 图像处理器和 LayoutXLM 分词器成为一个单独的处理器

    attributes = ["image_processor", "tokenizer"]
    # 类属性列表包括 "image_processor" 和 "tokenizer"

    image_processor_class = "LayoutLMv2ImageProcessor"
    # 图像处理器类为 "LayoutLMv2ImageProcessor"

    tokenizer_class = ("LayoutXLMTokenizer", "LayoutXLMTokenizerFast")
    # 分词器类包括 "LayoutXLMTokenizer" 和 "LayoutXLMTokenizerFast"
    # 初始化方法，用于创建一个新的实例对象
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # 如果 kwargs 中包含 'feature_extractor' 参数，则发出警告并将其移除
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 将 image_processor 设置为传入的 image_processor 或从 kwargs 中取出的 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果 image_processor 仍然为 None，则抛出数值错误异常
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        # 如果 tokenizer 为 None，则抛出数值错误异常
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # 调用父类的初始化方法，传入 image_processor 和 tokenizer 参数
        super().__init__(image_processor, tokenizer)

    # 调用实例对象时执行的方法，用于处理输入的图片和文本数据
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
        # 此处省略了方法体，用于处理传入的多个参数并进行相关处理

    # 获取溢出图片的方法，根据溢出映射返回对应的溢出图片列表
    def get_overflowing_images(self, images, overflow_to_sample_mapping):
        # 创建空列表，用于存放溢出的图片数据
        images_with_overflow = []
        # 遍历溢出映射，将对应索引的图片添加到列表中
        for sample_idx in overflow_to_sample_mapping:
            images_with_overflow.append(images[sample_idx])

        # 如果 images_with_overflow 列表长度与溢出映射长度不一致，则抛出数值错误异常
        if len(images_with_overflow) != len(overflow_to_sample_mapping):
            raise ValueError(
                "Expected length of images to be the same as the length of `overflow_to_sample_mapping`, but got"
                f" {len(images_with_overflow)} and {len(overflow_to_sample_mapping)}"
            )

        # 返回包含溢出图片的列表
        return images_with_overflow

    # 批量解码方法，将参数传递给 PreTrainedTokenizer 的 batch_decode 方法进行批量解码
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数传递给 PreTrainedTokenizer 的 `decode` 方法，并返回其结果
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，包括输入的标识符、边界框、注意力掩码和图像
    @property
    def model_input_names(self):
        return ["input_ids", "bbox", "attention_mask", "image"]

    # 返回特征提取器的类名，并发出未来警告，建议使用 `image_processor_class` 替代
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回特征提取器，并发出未来警告，建议使用 `image_processor` 替代
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```