# `.\transformers\models\seamless_m4t\processing_seamless_m4t.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，告知代码的版权归属和使用许可
"""
Audio/Text processor class for SeamlessM4T
"""

# 导入处理工具类 ProcessorMixin
from ...processing_utils import ProcessorMixin

# 定义 SeamlessM4TProcessor 类，继承自 ProcessorMixin
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
    # 类属性，指定特征提取器类名
    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    # 类属性，指定标记器类名
    tokenizer_class = ("SeamlessM4TTokenizer", "SeamlessM4TTokenizerFast")

    # 初始化方法，接受特征提取器和标记器作为参数
    def __init__(self, feature_extractor, tokenizer):
        # 调用父类的初始化方法
        super().__init__(feature_extractor, tokenizer)

    # 批量解码方法，将所有参数转发给 SeamlessM4TTokenizerFast 的 batch_decode 方法
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SeamlessM4TTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 解码方法，将所有参数转发给 SeamlessM4TTokenizerFast 的 decode 方法
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SeamlessM4TTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    # 模型输入名称属性，返回特征提取器和标记器的模型输入名称的并集
    @property
    def model_input_names(self):
        # 获取标记器的模型输入名称列表
        tokenizer_input_names = self.tokenizer.model_input_names
        # 获取特征提取器的模型输入名称列表
        feature_extractor_input_names = self.feature_extractor.model_input_names
        # 将两个列表合并并去重，返回结果
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))
```