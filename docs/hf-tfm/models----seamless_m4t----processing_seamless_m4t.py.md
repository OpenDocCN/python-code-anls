# `.\models\seamless_m4t\processing_seamless_m4t.py`

```py
"""
Audio/Text processor class for SeamlessM4T
"""

from ...processing_utils import ProcessorMixin

class SeamlessM4TProcessor(ProcessorMixin):
    r"""
    Constructs a SeamlessM4T processor which wraps a SeamlessM4T feature extractor and a SeamlessM4T tokenizer into a
    single processor.

    [`SeamlessM4TProcessor`] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and
    [`SeamlessM4TTokenizerFast`]. See the [`~SeamlessM4TProcessor.__call__`] and [`~SeamlessM4TProcessor.decode`] for
    more information.

    Args:
        feature_extractor ([`SeamlessM4TFeatureExtractor`]):
            The audio processor is a required input.
        tokenizer ([`SeamlessM4TTokenizerFast`]):
            The tokenizer is a required input.
    """

    # 设定特征提取器类名
    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    # 设定分词器类名
    tokenizer_class = ("SeamlessM4TTokenizer", "SeamlessM4TTokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        # 调用父类构造函数，初始化特征提取器和分词器
        super().__init__(feature_extractor, tokenizer)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SeamlessM4TTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        # 调用分词器的批量解码方法，将参数传递给其[`~PreTrainedTokenizer.batch_decode`]方法
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SeamlessM4TTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 调用分词器的解码方法，将参数传递给其[`~PreTrainedTokenizer.decode`]方法
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        # 获取分词器和特征提取器的模型输入名称列表，去重后返回
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))
```