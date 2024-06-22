# `.\transformers\models\clap\processing_clap.py`

```py
# coding=utf-8
# 版权 2023 年 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 无论是明示的还是暗示的。请参阅许可证了解特定语言的权限
# 和限制。
"""
CLAP 的音频/文本处理器类
"""

from ...processing_utils import ProcessorMixin  # 导入处理器混合类
from ...tokenization_utils_base import BatchEncoding  # 导入批量编码类


class ClapProcessor(ProcessorMixin):
    r"""
    构造一个 CLAP 处理器，将 CLAP 特征提取器和 RoBerta 分词器包装成一个单一处理器。

    [`ClapProcessor`] 提供了 [`ClapFeatureExtractor`] 和 [`RobertaTokenizerFast`] 的所有功能。请参阅
    [`~ClapProcessor.__call__`] 和 [`~ClapProcessor.decode`] 获取更多信息。

    Args:
        feature_extractor ([`ClapFeatureExtractor`]):
            音频处理器是必需的输入。
        tokenizer ([`RobertaTokenizerFast`]):
            分词器是必需的输入。
    """

    feature_extractor_class = "ClapFeatureExtractor"  # 特征提取器类名
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")  # 分词器类名

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)  # 调用父类的初始化方法

    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 RobertaTokenizerFast 的 [`~PreTrainedTokenizer.batch_decode`]。请参阅此方法的文档字符串了解更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)  # 调用分词器的批量解码方法

    def decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给 RobertaTokenizerFast 的 [`~PreTrainedTokenizer.decode`]。请参阅此方法的文档字符串了解更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)  # 调用分词器的解码方法

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names  # 获取分词器的模型输入名称
        feature_extractor_input_names = self.feature_extractor.model_input_names  # 获取特征提取器的模型输入名称
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))  # 返回模型输入名称列表并去重
```