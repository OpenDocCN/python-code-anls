# `.\models\blip_2\processing_blip_2.py`

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
Processor class for BLIP-2.
"""

from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class Blip2Processor(ProcessorMixin):
    r"""
    Constructs a BLIP-2 processor which wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.

    [`BlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the docstring
    of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.__init__
    def __init__(self, image_processor, tokenizer):
        # 禁用 tokenizer 的 token_type_ids 返回功能
        tokenizer.return_token_type_ids = False
        # 调用父类 ProcessorMixin 的构造函数，传入 image_processor 和 tokenizer
        super().__init__(image_processor, tokenizer)
        # 将当前处理器设置为图像处理器 image_processor
        self.current_processor = self.image_processor

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.__call__
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
        # 根据参数设置调用处理器的各种功能
        pass
    ) -> BatchEncoding:
        """
        使用 BlipImageProcessor.__call__ 方法准备模型的图像输入，
        使用 BertTokenizerFast.__call__ 方法准备模型的文本输入。

        详细信息请参考上述两个方法的文档字符串。
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        # 只处理文本情况
        if images is None:
            self.current_processor = self.tokenizer
            # 使用 tokenizer 处理文本编码
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

        # 添加像素值处理
        encoding_image_processor = self.image_processor(images, return_tensors=return_tensors)

        if text is not None:
            # 使用 tokenizer 处理文本编码
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
            text_encoding = None

        if text_encoding is not None:
            # 更新图像处理器的文本编码结果
            encoding_image_processor.update(text_encoding)

        return encoding_image_processor

    # 从 transformers.models.blip.processing_blip.BlipProcessor.batch_decode 复制，并替换为 BertTokenizerFast->PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 PreTrainedTokenizer 的 batch_decode 方法。请参考该方法的文档字符串获取详细信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 从 `transformers.models.blip.processing_blip.BlipProcessor.decode` 复制的方法。
    # 将所有参数和关键字参数转发给 `PreTrainedTokenizer` 的 `decode` 方法。
    # 请参阅 `PreTrainedTokenizer.decode` 方法的文档字符串以获取更多信息。
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # 从 `transformers.models.blip.processing_blip.BlipProcessor.model_input_names` 复制的属性。
    # 获取 `tokenizer` 和 `image_processor` 的模型输入名称，并返回去重后的列表。
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```