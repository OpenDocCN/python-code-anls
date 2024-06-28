# `.\models\owlv2\processing_owlv2.py`

```
# coding=utf-8
# 版权所有 2023 年 HuggingFace Inc. 团队
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据“原样”分发的软件
# 无任何担保或条件，包括但不限于，适销性和特定用途适用性的保证。
# 有关许可证的详细信息，请参阅许可证。
"""
OWLv2 的图像/文本处理器类
"""

from typing import List

import numpy as np

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import is_flax_available, is_tf_available, is_torch_available


class Owlv2Processor(ProcessorMixin):
    r"""
    构建 OWLv2 处理器，将 [`Owlv2ImageProcessor`] 和 [`CLIPTokenizer`] / [`CLIPTokenizerFast`] 包装成一个处理器，
    继承了图像处理器和分词器的功能。详细信息请参阅 [`~OwlViTProcessor.__call__`] 和 [`~OwlViTProcessor.decode`]。

    Args:
        image_processor ([`Owlv2ImageProcessor`]):
            必需的图像处理器输入。
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            必需的分词器输入。
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Owlv2ImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor, tokenizer, **kwargs):
        super().__init__(image_processor, tokenizer)

    # 从 transformers.models.owlvit.processing_owlvit.OwlViTProcessor.__call__ 复制，将 OWLViT->OWLv2
    # 从 transformers.models.owlvit.processing_owlvit.OwlViTProcessor.post_process_object_detection 复制，将 OWLViT->OWLv2
    def post_process_object_detection(self, *args, **kwargs):
        """
        此方法将所有参数转发给 [`OwlViTImageProcessor.post_process_object_detection`]。
        有关更多信息，请参阅此方法的文档字符串。
        """
        return self.image_processor.post_process_object_detection(*args, **kwargs)

    # 从 transformers.models.owlvit.processing_owlvit.OwlViTProcessor.post_process_image_guided_detection 复制，将 OWLViT->OWLv2
    def post_process_image_guided_detection(self, *args, **kwargs):
        """
        此方法将所有参数转发给 [`OwlViTImageProcessor.post_process_one_shot_object_detection`]。
        有关更多信息，请参阅此方法的文档字符串。
        """
        return self.image_processor.post_process_image_guided_detection(*args, **kwargs)

    # 从 transformers.models.owlvit.processing_owlvit.OwlViTProcessor.batch_decode 复制
    # 将所有参数转发给 CLIPTokenizerFast 的 batch_decode 方法
    # 请参考该方法的文档字符串获取更多信息
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 调用 CLIPTokenizerFast 的 batch_decode 方法，并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 从 transformers.models.owlvit.processing_owlvit.OwlViTProcessor.decode 复制而来
    # 将所有参数转发给 CLIPTokenizerFast 的 decode 方法
    # 请参考该方法的文档字符串获取更多信息
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # 调用 CLIPTokenizerFast 的 decode 方法，并返回结果
        return self.tokenizer.decode(*args, **kwargs)
```