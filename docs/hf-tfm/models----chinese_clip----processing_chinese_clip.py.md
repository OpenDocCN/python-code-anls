# `.\models\chinese_clip\processing_chinese_clip.py`

```
# coding=utf-8
# Copyright 2022 The OFA-Sys Team Authors and The HuggingFace Team. All rights reserved.
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
Image/Text processor class for Chinese-CLIP
"""

import warnings

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class ChineseCLIPProcessor(ProcessorMixin):
    r"""
    Constructs a Chinese-CLIP processor which wraps a Chinese-CLIP image processor and a Chinese-CLIP tokenizer into a
    single processor.

    [`ChineseCLIPProcessor`] offers all the functionalities of [`ChineseCLIPImageProcessor`] and [`BertTokenizerFast`].
    See the [`~ChineseCLIPProcessor.__call__`] and [`~ChineseCLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`ChineseCLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`BertTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ChineseCLIPImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # Deprecated feature_extractor warning and migration
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # Determine the image_processor from feature_extractor or provided argument
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # Initialize the processor with image_processor and tokenizer
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # Delegate batch decoding to the underlying tokenizer
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发给 BertTokenizerFast 的 `~PreTrainedTokenizer.decode` 方法，并返回其结果
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入名称的列表，这些名称由分词器和图像处理器的模型输入名称合并而成，且保持唯一性
    @property
    def model_input_names(self):
        # 获取分词器的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取图像处理器的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 将两个列表合并并去除重复项，返回结果列表
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # 返回图像处理器的类名，并发出关于该属性即将移除的警告
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回图像处理器的类名作为特征提取器类的代理
        return self.image_processor_class
```