# `.\models\pix2struct\processing_pix2struct.py`

```
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for Pix2Struct.
"""

from typing import List, Optional, Union

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class Pix2StructProcessor(ProcessorMixin):
    r"""
    Constructs a PIX2STRUCT processor which wraps a BERT tokenizer and PIX2STRUCT image processor into a single
    processor.

    [`Pix2StructProcessor`] offers all the functionalities of [`Pix2StructImageProcessor`] and [`T5TokenizerFast`]. See
    the docstring of [`~Pix2StructProcessor.__call__`] and [`~Pix2StructProcessor.decode`] for more information.

    Args:
        image_processor (`Pix2StructImageProcessor`):
            An instance of [`Pix2StructImageProcessor`]. The image processor is a required input.
        tokenizer (Union[`T5TokenizerFast`, `T5Tokenizer`]):
            An instance of ['T5TokenizerFast`] or ['T5Tokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Pix2StructImageProcessor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    def __init__(self, image_processor, tokenizer):
        # Disable token type IDs as they are not used in this processor
        tokenizer.return_token_type_ids = False
        # Initialize the processor with the provided image processor and tokenizer
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images=None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        max_patches: Optional[int] = 2048,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        """
        Process input images and text into a format suitable for PIX2STRUCT tasks.

        Args:
            images (optional): Input images to process.
            text (Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]): Input text data.
            add_special_tokens (bool): Whether to add special tokens (like [CLS], [SEP]) or not.
            padding (Union[bool, str, PaddingStrategy]): Padding strategy for text sequences.
            truncation (Union[bool, str, TruncationStrategy]): Truncation strategy for text sequences.
            max_length (Optional[int]): Maximum sequence length to enforce.
            max_patches (Optional[int]): Maximum number of patches to consider.
            stride (int): Stride length for patch extraction.
            pad_to_multiple_of (Optional[int]): Pad the sequence length to a multiple of this value.
            return_attention_mask (Optional[bool]): Whether to return attention masks.
            return_overflowing_tokens (bool): Whether to return overflowing tokens.
            return_special_tokens_mask (bool): Whether to return special tokens mask.
            return_offsets_mapping (bool): Whether to return offsets mapping.
            return_token_type_ids (bool): Whether to return token type IDs (not used in this processor).
            return_length (bool): Whether to return sequence length.
            verbose (bool): Whether to print verbose information.
            return_tensors (Optional[Union[str, TensorType]]): Desired tensor type for returned tensors.

        Returns:
            BatchEncoding: Processed inputs in a batch encoding format.

        Notes:
            This method processes both images and text to prepare them for PIX2STRUCT tasks.
            It incorporates functionality from both `Pix2StructImageProcessor` and `T5TokenizerFast`.
        """
        # Implementation of input processing logic goes here
        pass
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        # 调用内部的 `batch_decode` 方法，将所有参数传递给 Pix2StructTokenizerFast 的 `batch_decode` 方法
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Pix2StructTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 调用内部的 `decode` 方法，将所有参数传递给 Pix2StructTokenizerFast 的 `decode` 方法
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        This property returns a list of unique model input names by combining tokenizer's and image_processor's input names.
        """
        # 获取 tokenizer 和 image_processor 的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        # 使用集合去除重复项，然后转换为列表并返回
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```