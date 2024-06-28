# `.\models\clip\processing_clip.py`

```
# coding=utf-8
# 版权所有 2021 年 HuggingFace Inc. 团队
#
# 根据 Apache 许可证 2.0 版本进行许可；除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件按“原样”分发，不提供任何形式的担保或条件，
# 无论是明示的还是默示的。详细信息请参阅许可证。
"""
CLIP 的图像/文本处理类
"""

import warnings

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class CLIPProcessor(ProcessorMixin):
    r"""
    构建一个 CLIP 处理器，将 CLIP 图像处理器和 CLIP 分词器包装成一个单一处理器。

    [`CLIPProcessor`] 提供了 [`CLIPImageProcessor`] 和 [`CLIPTokenizerFast`] 的所有功能。参见
    [`~CLIPProcessor.__call__`] 和 [`~CLIPProcessor.decode`] 获取更多信息。

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            图像处理器，必需输入。
        tokenizer ([`CLIPTokenizerFast`], *optional*):
            分词器，必需输入。
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        feature_extractor = None
        # 如果 kwargs 中包含 `feature_extractor`，则发出警告并将其弹出
        if "feature_extractor" in kwargs:
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        # 如果未提供 image_processor，则尝试使用 feature_extractor
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发到 CLIPTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`]。请参阅该方法的文档字符串以获取更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        此方法将所有参数转发到 CLIPTokenizerFast 的 [`~PreTrainedTokenizer.decode`]。请参阅该方法的文档字符串以获取更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # 返回模型输入的名称列表，合并并去重来自于分词器和图像处理器的输入名称
    def model_input_names(self):
        # 获取分词器的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取图像处理器的模型输入名称列表
        image_processor_input_names = self.image_processor.model_input_names
        # 返回合并并去重后的输入名称列表
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # 返回特征提取器的类，发出关于该属性即将在 v5 版本中删除的警告
    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        # 返回图像处理器的类
        return self.image_processor_class

    # 返回特征提取器，发出关于该属性即将在 v5 版本中删除的警告
    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        # 返回图像处理器
        return self.image_processor
```