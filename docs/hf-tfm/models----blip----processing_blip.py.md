# `.\models\blip\processing_blip.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Processor class for Blip.
"""

from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class BlipProcessor(ProcessorMixin):
    r"""
    Constructs a BLIP processor which wraps a BERT tokenizer and BLIP image processor into a single processor.

    [`BlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`BertTokenizerFast`]. See the
    docstring of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`BertTokenizerFast`):
            An instance of ['BertTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, image_processor, tokenizer):
        tokenizer.return_token_type_ids = False  # 禁用 tokenizer 的返回 token 类型 ID 功能
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor  # 设置当前处理器为图像处理器

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
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
        Process images and text inputs using the BLIP processor.

        Args:
            images (ImageInput, optional): Input images to be processed.
            text (Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]], optional):
                Input text or pre-tokenized text inputs to be processed.
            add_special_tokens (bool, optional): Whether to add special tokens to the inputs.
            padding (Union[bool, str, PaddingStrategy], optional): Padding strategy for inputs.
            truncation (Union[bool, str, TruncationStrategy], optional): Truncation strategy for inputs.
            max_length (int, optional): Maximum length of the processed inputs.
            stride (int, optional): Stride used for overflowing tokens.
            pad_to_multiple_of (int, optional): Pad inputs to a multiple of this number.
            return_attention_mask (bool, optional): Whether to return attention masks.
            return_overflowing_tokens (bool, optional): Whether to return overflowing tokens.
            return_special_tokens_mask (bool, optional): Whether to return special tokens mask.
            return_offsets_mapping (bool, optional): Whether to return offsets mapping.
            return_token_type_ids (bool, optional): Whether to return token type IDs.
            return_length (bool, optional): Whether to return the length of the processed inputs.
            verbose (bool, optional): Whether to print verbose information.
            return_tensors (Union[str, TensorType], optional): Type of tensor to return.

        Returns:
            BatchEncoding: Processed inputs in batch encoding format.
        """
        pass
    ) -> BatchEncoding:
        """
        使用 [`BlipImageProcessor.__call__`] 方法准备图像数据以供模型使用，
        并使用 [`BertTokenizerFast.__call__`] 方法准备文本数据以供模型使用。

        更多信息请参考上述两个方法的文档字符串。
        """
        # 如果未提供图像和文本，则抛出数值错误
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        # 仅处理文本情况
        if images is None:
            # 设置当前处理器为分词器
            self.current_processor = self.tokenizer
            # 使用分词器处理文本数据，返回编码结果
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            return text_encoding

        # 处理包含图像的情况
        # 使用图像处理器处理图像数据，返回编码结果
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors)

        if text is not None:
            # 如果同时提供了文本，使用分词器处理文本数据，返回编码结果
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            # 如果未提供文本，将文本编码设置为 None
            text_encoding = None

        if text_encoding is not None:
            # 如果存在文本编码结果，则更新图像处理器的编码结果
            encoding_image_processor.update(text_encoding)

        return encoding_image_processor

    def batch_decode(self, *args, **kwargs):
        """
        将所有参数转发给 BertTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`] 方法。
        请参考该方法的文档字符串获取更多信息。
        """
        # 调用分词器的批量解码方法，并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发给 BertTokenizerFast 的 `~PreTrainedTokenizer.decode` 方法，并返回结果
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，这里包括了 tokenizer 和 image_processor 的输入名称，并去除重复项
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```