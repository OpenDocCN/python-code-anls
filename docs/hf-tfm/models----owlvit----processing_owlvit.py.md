# `.\models\owlvit\processing_owlvit.py`

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
Image/Text processor class for OWL-ViT
"""

import warnings
from typing import List

import numpy as np

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import is_flax_available, is_tf_available, is_torch_available

class OwlViTProcessor(ProcessorMixin):
    r"""
    Constructs an OWL-ViT processor which wraps [`OwlViTImageProcessor`] and [`CLIPTokenizer`]/[`CLIPTokenizerFast`]
    into a single processor that interits both the image processor and tokenizer functionalities. See the
    [`~OwlViTProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more information.

    Args:
        image_processor ([`OwlViTImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    # 类属性，包含了要初始化的属性名称列表
    image_processor_class = "OwlViTImageProcessor"
    # 类属性，指定图像处理器的类名
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")
    # 类属性，指定了两种可能的标记化器类名

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 初始化特征提取器为 None
        if "feature_extractor" in kwargs:
            # 如果在参数中有 'feature_extractor'，发出警告
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")
            # 弹出 'feature_extractor' 参数并将其赋给特征提取器

        image_processor = image_processor if image_processor is not None else feature_extractor
        # 如果没有指定图像处理器，则使用特征提取器（如果有的话）
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
            # 如果图像处理器为空，则抛出值错误异常
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
            # 如果标记化器为空，则抛出值错误异常

        super().__init__(image_processor, tokenizer)
        # 调用父类的初始化方法，传入图像处理器和标记化器作为参数

    def post_process(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OwlViTImageProcessor.post_process`]. Please refer to the docstring
        of this method for more information.
        """
        return self.image_processor.post_process(*args, **kwargs)
        # 调用图像处理器的后处理方法，并将所有参数转发给它
    def post_process_object_detection(self, *args, **kwargs):
        """
        将所有参数转发到 `OwlViTImageProcessor.post_process_object_detection` 方法中。
        请参阅该方法的文档字符串获取更多信息。
        """
        return self.image_processor.post_process_object_detection(*args, **kwargs)

    def post_process_image_guided_detection(self, *args, **kwargs):
        """
        将所有参数转发到 `OwlViTImageProcessor.post_process_one_shot_object_detection` 方法中。
        请参阅该方法的文档字符串获取更多信息。
        """
        return self.image_processor.post_process_image_guided_detection(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        将所有参数转发到 CLIPTokenizerFast 的 `~PreTrainedTokenizer.batch_decode` 方法中。
        请参阅该方法的文档字符串获取更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        将所有参数转发到 CLIPTokenizerFast 的 `~PreTrainedTokenizer.decode` 方法中。
        请参阅该方法的文档字符串获取更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def feature_extractor_class(self):
        """
        警告：`feature_extractor_class` 已弃用，并将在 v5 版本中移除。请使用 `image_processor_class` 替代。
        返回 `image_processor_class`。
        """
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        """
        警告：`feature_extractor` 已弃用，并将在 v5 版本中移除。请使用 `image_processor` 替代。
        返回 `image_processor`。
        """
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```