# `.\models\bridgetower\processing_bridgetower.py`

```py
# coding=utf-8
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
Processor class for BridgeTower.
"""

from typing import List, Optional, Union

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class BridgeTowerProcessor(ProcessorMixin):
    r"""
    Constructs a BridgeTower processor which wraps a Roberta tokenizer and BridgeTower image processor into a single
    processor.

    [`BridgeTowerProcessor`] offers all the functionalities of [`BridgeTowerImageProcessor`] and
    [`RobertaTokenizerFast`]. See the docstring of [`~BridgeTowerProcessor.__call__`] and
    [`~BridgeTowerProcessor.decode`] for more information.

    Args:
        image_processor (`BridgeTowerImageProcessor`):
            An instance of [`BridgeTowerImageProcessor`]. The image processor is a required input.
        tokenizer (`RobertaTokenizerFast`):
            An instance of ['RobertaTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BridgeTowerImageProcessor"
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
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
        # Callable method to preprocess inputs combining images and text for model input
        """
        Process images and optionally text into model input.

        Args:
            images: Input images to be processed.
            text: Optional text input, can be either TextInput or PreTokenizedInput format.
            add_special_tokens: Whether to add special tokens (like [CLS], [SEP]) to the inputs.
            padding: Padding strategy. Can be a bool, str, or PaddingStrategy enum.
            truncation: Truncation strategy. Can be a bool, str, or TruncationStrategy enum.
            max_length: Maximum length of the returned sequences.
            stride: Stride to use when overflowing tokens.
            pad_to_multiple_of: Pad to a multiple of specified value.
            return_token_type_ids: Whether to return token type ids.
            return_attention_mask: Whether to return attention mask.
            return_overflowing_tokens: Whether to return overflowing tokens.
            return_special_tokens_mask: Whether to return special tokens mask.
            return_offsets_mapping: Whether to return offsets mapping.
            return_length: Whether to return the lengths of processed inputs.
            verbose: Whether to output detailed logs during processing.
            return_tensors: Return tensors format (e.g., "pt" for PyTorch tensors).
            **kwargs: Additional keyword arguments for processing.

        Returns:
            BatchEncoding: Processed inputs formatted as BatchEncoding.

        Notes:
            This method processes images and optionally text into a format suitable for model input,
            handling tokenization, padding, truncation, and special token additions as specified.
        """
        pass  # Placeholder for actual implementation or further logic
    ) -> BatchEncoding:
        """
        使用 [`BridgeTowerImageProcessor.__call__`] 方法准备图像以供模型使用，
        使用 [`RobertaTokenizerFast.__call__`] 方法准备文本以供模型使用。

        更多信息请参考上述两个方法的文档字符串。
        """
        # 使用指定参数调用 tokenizer 方法，生成编码结果
        encoding = self.tokenizer(
            text=text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )
        # 使用 image_processor 方法处理图像，获取处理后的编码结果
        encoding_image_processor = self.image_processor(
            images, return_tensors=return_tensors, do_normalize=True, do_center_crop=True, **kwargs
        )
        # 将图像处理的编码结果更新到文本处理的编码结果中
        encoding.update(encoding_image_processor)

        # 返回合并了文本和图像编码结果的最终编码结果
        return encoding

    def batch_decode(self, *args, **kwargs):
        """
        将所有参数转发给 RobertaTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`] 方法。
        更多信息请参考该方法的文档字符串。
        """
        # 调用 tokenizer 的 batch_decode 方法，并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        将所有参数转发给 RobertaTokenizerFast 的 [`~PreTrainedTokenizer.decode`] 方法。
        更多信息请参考该方法的文档字符串。
        """
        # 调用 tokenizer 的 decode 方法，并返回结果
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        # 获取 tokenizer 和 image_processor 的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        # 合并去重后的模型输入名称列表，并返回
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```