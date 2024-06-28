# `.\models\clap\processing_clap.py`

```
"""
Audio/Text processor class for CLAP
"""

from ...processing_utils import ProcessorMixin  # 导入ProcessorMixin，用于处理混合功能
from ...tokenization_utils_base import BatchEncoding  # 导入BatchEncoding，用于批量编码

class ClapProcessor(ProcessorMixin):
    r"""
    Constructs a CLAP processor which wraps a CLAP feature extractor and a RoBerta tokenizer into a single processor.

    [`ClapProcessor`] offers all the functionalities of [`ClapFeatureExtractor`] and [`RobertaTokenizerFast`]. See the
    [`~ClapProcessor.__call__`] and [`~ClapProcessor.decode`] for more information.

    Args:
        feature_extractor ([`ClapFeatureExtractor`]):
            The audio processor is a required input.
        tokenizer ([`RobertaTokenizerFast`]):
            The tokenizer is a required input.
    """

    feature_extractor_class = "ClapFeatureExtractor"  # 设定特征提取器的类名为"ClapFeatureExtractor"
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")  # 设定标记器的类名为"RobertaTokenizer"和"RobertaTokenizerFast"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)  # 调用父类的初始化方法，传入特征提取器和标记器

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)  # 调用标记器的批量解码方法，并将所有参数传递给它

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)  # 调用标记器的解码方法，并将所有参数传递给它

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names  # 获取标记器的模型输入名称列表
        feature_extractor_input_names = self.feature_extractor.model_input_names  # 获取特征提取器的模型输入名称列表
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))  # 返回合并并去重后的模型输入名称列表
```