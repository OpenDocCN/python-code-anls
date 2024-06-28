# `.\models\altclip\processing_altclip.py`

```py
# coding=utf-8
# Copyright 2022 WenXiang ZhongzhiCheng LedellWu LiuGuang BoWenZhang The HuggingFace Inc. team. All rights reserved.
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
Image/Text processor class for AltCLIP
"""
import warnings

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding

class AltCLIPProcessor(ProcessorMixin):
    r"""
    Constructs a AltCLIP processor which wraps a CLIP image processor and a XLM-Roberta tokenizer into a single
    processor.

    [`AltCLIPProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`XLMRobertaTokenizerFast`]. See
    the [`~AltCLIPProcessor.__call__`] and [`~AltCLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`XLMRobertaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # Deprecated feature_extractor handling
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # Set image_processor to feature_extractor if image_processor is None
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # Initialize with image_processor and tokenizer
        super().__init__(image_processor, tokenizer)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to XLMRobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        # Delegate batch decoding to tokenizer
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发给 XLMRobertaTokenizerFast 的 `~PreTrainedTokenizer.decode` 方法
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to XLMRobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 调用 tokenizer 对象的 decode 方法，将参数传递给它，并返回结果
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，合并并去重 tokenizer 和 image_processor 的输入名称
    @property
    def model_input_names(self):
        # 获取 tokenizer 的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取 image_processor 的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 合并两个列表，使用字典去重后转换为列表，并返回结果
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
```