# `.\models\x_clip\processing_x_clip.py`

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
Image/Text processor class for XCLIP
"""

import warnings

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class XCLIPProcessor(ProcessorMixin):
    r"""
    Constructs an X-CLIP processor which wraps a VideoMAE image processor and a CLIP tokenizer into a single processor.

    [`XCLIPProcessor`] offers all the functionalities of [`VideoMAEImageProcessor`] and [`CLIPTokenizerFast`]. See the
    [`~XCLIPProcessor.__call__`] and [`~XCLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`VideoMAEImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "VideoMAEImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        # Check for deprecated argument and warn the user
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # Use `feature_extractor` if `image_processor` is not provided directly
        image_processor = image_processor if image_processor is not None else feature_extractor
        # Raise an error if `image_processor` or `tokenizer` is not provided
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        # Initialize the processor mixin with `image_processor` and `tokenizer`
        super().__init__(image_processor, tokenizer)
        # Set the current processor to `image_processor`
        self.current_processor = self.image_processor

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # Forward batch decoding request to `tokenizer`
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # Forward decoding request to `tokenizer`
        return self.tokenizer.decode(*args, **kwargs)
    # 返回模型输入的名称列表，包括输入的标识符、注意力掩码、位置标识和像素值
    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "position_ids", "pixel_values"]
    
    # 返回特征提取器的类。警告已弃用，将在v5中移除。建议使用`image_processor_class`替代。
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class
    
    # 返回特征提取器。警告已弃用，将在v5中移除。建议使用`image_processor`替代。
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor
```