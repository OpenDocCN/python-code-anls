# `.\models\llava_next\processing_llava_next.py`

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
Processor class for LLaVa-NeXT.
"""


from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType


class LlavaNextProcessor(ProcessorMixin):
    r"""
    Constructs a LLaVa-NeXT processor which wraps a LLaVa-NeXT image processor and a LLaMa tokenizer into a single processor.

    [`LlavaNextProcessor`] offers all the functionalities of [`LlavaNextImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~LlavaNextProcessor.__call__`] and [`~LlavaNextProcessor.decode`] for more information.

    Args:
        image_processor ([`LlavaNextImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "LlavaNextImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ):
        """
        This method is the entry point for preprocessing textual and image inputs using the LLaVa-NeXT processor.

        Args:
            text (Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]):
                Input text or pre-tokenized text to be processed.
            images (ImageInput, optional):
                Input images to be processed.
            padding (Union[bool, str, PaddingStrategy], optional):
                Argument specifying if and how to pad the sequences.
            truncation (Union[bool, str, TruncationStrategy], optional):
                Argument specifying if and how to truncate sequences.
            max_length (int, optional):
                Maximum length of the sequences after tokenization.
            return_tensors (Optional[Union[str, TensorType]], optional):
                Desired framework tensors (PyTorch, TensorFlow) for the returned data.

        Returns:
            dict: A dictionary containing processed inputs suitable for model ingestion.
        """
        # Implementation of processing logic omitted for brevity
        pass

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.

        Returns:
            List[str]: Decoded texts corresponding to the input tokens or IDs.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # 调用 LlamaTokenizerFast 的 decode 方法，将所有参数传递给它
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # 从 transformers.models.clip.processing_clip.CLIPProcessor.model_input_names 复制而来
    def model_input_names(self):
        # 获取 tokenizer 的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取 image_processor 的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 合并去重 tokenizer 和 image_processor 的模型输入名称，返回列表
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```