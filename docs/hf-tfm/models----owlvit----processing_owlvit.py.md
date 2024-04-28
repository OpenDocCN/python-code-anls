# `.\transformers\models\owlvit\processing_owlvit.py`

```py
# coding=utf-8
# 版权所有 2022 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 按“原样”分发，不提供任何形式的担保
# 或条件，明示或暗示。
# 有关许可证的更多信息，请参见
# 许可证。
"""
OWL-ViT 的图像/文本处理器类
"""

import warnings
from typing import List

import numpy as np

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import is_flax_available, is_tf_available, is_torch_available

class OwlViTProcessor(ProcessorMixin):
    r"""
    构建一个 OWL-ViT 处理器，将 [`OwlViTImageProcessor`] 和 [`CLIPTokenizer`]/[`CLIPTokenizerFast`] 封装到一个单一的处理器中，
    继承了图像处理器和分词器的功能。有关更多信息，请参阅 [`~OwlViTProcessor.__call__`] 和 [`~OwlViTProcessor.decode`]。

    Args:
        image_processor ([`OwlViTImageProcessor`], *optional*):
            图像处理器是必需的输入。
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`], *optional*):
            分词器是必需的输入。
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "OwlViTImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def post_process(self, *args, **kwargs):
        """
        此方法将其所有参数转发给 [`OwlViTImageProcessor.post_process`]。有关更多信息，请参见此方法的文档字符串。
        """
        return self.image_processor.post_process(*args, **kwargs)
    def post_process_object_detection(self, *args, **kwargs):
        """
        This method forwards all its arguments to `OwlViTImageProcessor.post_process_object_detection`. Please refer
        to the docstring of this method for more information.
        """
        # 调用`OwlViTImageProcessor.post_process_object_detection`方法并传递所有参数，返回结果
        return self.image_processor.post_process_object_detection(*args, **kwargs)

    def post_process_image_guided_detection(self, *args, **kwargs):
        """
        This method forwards all its arguments to `OwlViTImageProcessor.post_process_one_shot_object_detection`.
        Please refer to the docstring of this method for more information.
        """
        # 调用`OwlViTImageProcessor.post_process_image_guided_detection`方法并传递所有参数，返回结果
        return self.image_processor.post_process_image_guided_detection(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's `PreTrainedTokenizer.batch_decode`. Please
        refer to the docstring of this method for more information.
        """
        # 调用`CLIPTokenizerFast`的`PreTrainedTokenizer.batch_decode`方法并传递所有参数，返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's `PreTrainedTokenizer.decode`. Please refer to
        the docstring of this method for more information.
        """
        # 调用`CLIPTokenizerFast`的`PreTrainedTokenizer.decode`方法并传递所有参数，返回结果
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def feature_extractor_class(self):
        """
        This property is deprecated and will be removed in v5. Use `image_processor_class` instead.
        """
        # 发出警告，提示`feature_extractor_class`已过时，建议使用`image_processor_class`
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回`image_processor_class`
        return self.image_processor_class

    @property
    def feature_extractor(self):
        """
        This property is deprecated and will be removed in v5. Use `image_processor` instead.
        """
        # 发出警告，提示`feature_extractor`已过时，建议使用`image_processor`
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        # 返回`image_processor`
        return self.image_processor
```