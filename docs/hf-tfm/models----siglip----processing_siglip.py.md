# `.\models\siglip\processing_siglip.py`

```py
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Image/Text processor class for SigLIP.
"""

from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class SiglipProcessor(ProcessorMixin):
    r"""
    Constructs a Siglip processor which wraps a Siglip image processor and a Siglip tokenizer into a single processor.

    [`SiglipProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`SiglipTokenizer`]. See the
    [`~SiglipProcessor.__call__`] and [`~SiglipProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`SiglipTokenizer`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "SiglipTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: int = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ):
        """
        This method combines the functionalities of both image processing and tokenization. It processes input text
        and/or images according to specified padding, truncation, and max length parameters, and returns processed data
        in a format based on the return_tensors argument.
        """
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SiglipTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SiglipTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Property method that provides model input names. It is copied from transformers.models.clip.processing_clip.
        CLIPProcessor.model_input_names with modifications for Siglip and T5.
        """
        return {
            "text": "inputs",
            "image": "pixel_values",
            "padding": "padding",
            "max_length": "max_length",
            "truncation": "truncation",
            "return_tensors": "return_tensors",
        }
    # 定义方法：获取模型输入的名称列表
    def model_input_names(self):
        # 获取分词器的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取图像处理器的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 将两个列表合并，并去除重复的元素，以列表形式返回结果
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```