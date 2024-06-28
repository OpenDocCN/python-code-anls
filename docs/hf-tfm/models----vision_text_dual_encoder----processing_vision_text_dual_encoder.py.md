# `.\models\vision_text_dual_encoder\processing_vision_text_dual_encoder.py`

```
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Processor class for VisionTextDualEncoder
"""

import warnings

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class VisionTextDualEncoderProcessor(ProcessorMixin):
    r"""
    Constructs a VisionTextDualEncoder processor which wraps an image processor and a tokenizer into a single
    processor.

    [`VisionTextDualEncoderProcessor`] offers all the functionalities of [`AutoImageProcessor`] and [`AutoTokenizer`].
    See the [`~VisionTextDualEncoderProcessor.__call__`] and [`~VisionTextDualEncoderProcessor.decode`] for more
    information.

    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

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

        # Set image_processor to feature_extractor if not provided separately
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You have to specify an image_processor.")
        if tokenizer is None:
            raise ValueError("You have to specify a tokenizer.")

        # Initialize the processor with image_processor and tokenizer
        super().__init__(image_processor, tokenizer)
        # Set the current_processor to image_processor
        self.current_processor = self.image_processor

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VisionTextDualEncoderTokenizer's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more information.
        """
        # Delegate batch_decode to the underlying tokenizer
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 将所有参数转发给 VisionTextDualEncoderTokenizer 的 `PreTrainedTokenizer.decode` 方法。
    # 请参考该方法的文档字符串获取更多信息。
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 返回模型输入的名称列表，合并并去除重复的 Tokenizer 和图像处理器的输入名称。
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # 返回图像处理器的类。
    # 发出警告，提示`feature_extractor_class`将在v5中移除，建议使用`image_processor_class`代替。
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    # 返回图像处理器。
    # 发出警告，提示`feature_extractor`将在v5中移除，建议使用`image_processor`代替。
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```