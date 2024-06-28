# `.\models\git\processing_git.py`

```
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
Image/Text processor class for GIT
"""

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class GitProcessor(ProcessorMixin):
    r"""
    Constructs a GIT processor which wraps a CLIP image processor and a BERT tokenizer into a single processor.

    [`GitProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`BertTokenizerFast`]. See the
    [`~GitProcessor.__call__`] and [`~GitProcessor.decode`] for more information.

    Args:
        image_processor ([`AutoImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`AutoTokenizer`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        # 将传入的图像处理器设为当前处理器

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
        # 调用 tokenizer 对象的 batch_decode 方法，并将参数传递给 BertTokenizerFast 对象的 batch_decode 方法

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
        # 调用 tokenizer 对象的 decode 方法，并将参数传递给 BertTokenizerFast 对象的 decode 方法

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values"]
        # 返回模型输入的名称列表，包括 input_ids、attention_mask 和 pixel_values
```