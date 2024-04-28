# `.\transformers\models\wav2vec2_bert\processing_wav2vec2_bert.py`

```
# 设置代码文件的字符编码为 UTF-8
# 版权声明，指明代码的版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件；
# 除非符合许可证的规定，否则不得使用本文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件按“原样”提供，
# 不提供任何明示或暗示的担保或条件。
# 请参阅许可证以获取特定语言的权限和限制。
"""
Wav2Vec2-BERT 的语音处理器类
"""
# 导入警告模块
import warnings

# 导入处理工具类 ProcessorMixin
from ...processing_utils import ProcessorMixin

# 导入特征提取器 SeamlessM4TFeatureExtractor
from ..seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor

# 导入 Wav2Vec2CTCTokenizer
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer

# 定义 Wav2Vec2BertProcessor 类，继承 ProcessorMixin 类
class Wav2Vec2BertProcessor(ProcessorMixin):
    r"""
    构造一个 Wav2Vec2-BERT 处理器，将 Wav2Vec2-BERT 特征提取器和 Wav2Vec2 CTC 分词器封装成一个单独的处理器。

    [`Wav2Vec2Processor`] 提供了 [`SeamlessM4TFeatureExtractor`] 和 [`PreTrainedTokenizer`] 的所有功能。
    更多信息请参阅 [`~Wav2Vec2Processor.__call__`] 和 [`~Wav2Vec2Processor.decode`] 的文档字符串。

    Args:
        feature_extractor (`SeamlessM4TFeatureExtractor`):
            [`SeamlessM4TFeatureExtractor`] 的一个实例。特征提取器是必需的输入。
        tokenizer ([`PreTrainedTokenizer`]):
            [`PreTrainedTokenizer`] 的一个实例。分词器是必需的输入。
    """

    # 类属性，特征提取器类名
    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    # 类属性，分词器类名
    tokenizer_class = "AutoTokenizer"

    # 初始化方法
    def __init__(self, feature_extractor, tokenizer):
        # 调用父类 ProcessorMixin 的初始化方法
        super().__init__(feature_extractor, tokenizer)

    # 类方法，从预训练模型中加载处理器
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 尝试从预训练模型加载处理器
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 如果加载失败，则执行以下操作
        except OSError:
            # 发出警告，提示即将废弃的功能
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            # 从预训练模型加载特征提取器
            feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            # 从预训练模型加载分词器
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            # 返回 Wav2Vec2BertProcessor 的实例
            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
    # 定义一个方法，用于准备模型的一个或多个序列和音频数据。该方法将`audio`和`kwargs`参数传递给SeamlessM4TFeatureExtractor的[`~SeamlessM4TFeatureExtractor.__call__`]，如果`audio`不是`None`，以预处理音频。为了准备目标序列，该方法将`text`和`kwargs`参数传递给PreTrainedTokenizer的[`~PreTrainedTokenizer.__call__`]，如果`text`不是`None`。更多信息，请参考上述两种方法的文档字符串。

    def __call__(self, audio=None, text=None, **kwargs):
        sampling_rate = kwargs.pop("sampling_rate", None)  # 从`kwargs`中弹出"sampling_rate"参数，并将其值赋给`sampling_rate`

        # 如果`audio`和`text`都是`None`, 抛出异常
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        # 如果`audio`不是`None`
        if audio is not None:
            # 将`audio`传递给特征提取器`feature_extractor`，使用`sampling_rate`和`kwargs`参数预处理音频，结果赋给`inputs`
            inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, **kwargs)
        
        # 如果`text`不是`None`
        if text is not None:
            # 将`text`传递给分词器`tokenizer`，使用`kwargs`参数进行编码，结果赋给`encodings`
            encodings = self.tokenizer(text, **kwargs)

        # 如果`text`是`None`，返回`inputs`
        if text is None:
            return inputs
        
        # 如果`audio`是`None`，返回`encodings`
        elif audio is None:
            return encodings
        
        # 如果`audio`和`text`都不是`None`
        else:
            # 将`encodings`中的"input_ids"赋值给`inputs`中的"labels"
            inputs["labels"] = encodings["input_ids"]
            return inputs
    # 定义一个方法用于填充输入特征和标签
    def pad(self, input_features=None, labels=None, **kwargs):
        """
        如果 `input_features` 不是 `None`，则将 `input_features` 和 `kwargs` 参数传递给 SeamlessM4TFeatureExtractor 的 [`~SeamlessM4TFeatureExtractor.pad`] 方法来填充输入特征。
        如果 `labels` 不是 `None`，则将 `labels` 和 `kwargs` 参数传递给 PreTrainedTokenizer 的 [`~PreTrainedTokenizer.pad`] 方法来填充标签。
        有关以上两个方法的更多信息，请参考其文档字符串。
        """
        # 如果既没有输入特征也没有标签，则抛出数值错误
        if input_features is None and labels is None:
            raise ValueError("You need to specify either an `input_features` or `labels` input to pad.")

        # 如果有输入特征，则使用特征提取器的 pad 方法填充输入特征
        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, **kwargs)
        # 如果有标签，则使用分词器的 pad 方法填充标签
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        # 如果没有标签，则返回输入特征
        if labels is None:
            return input_features
        # 如果没有输入特征，则返回标签
        elif input_features is None:
            return labels
        # 如果既有输入特征又有标签，则将标签的 input_ids 填充至输入特征的标签字段中
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    # 定义一个方法用于批量解码
    def batch_decode(self, *args, **kwargs):
        """
        此方法将其所有参数转发给 PreTrainedTokenizer 的 [`~PreTrainedTokenizer.batch_decode`] 方法。有关此方法的更多信息，请参考其文档字符串。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 定义一个方法用于解码
    def decode(self, *args, **kwargs):
        """
        此方法将其所有参数转发给 PreTrainedTokenizer 的 [`~PreTrainedTokenizer.decode`] 方法。有关此方法的更多信息，请参考其文档字符串。
        """
        return self.tokenizer.decode(*args, **kwargs)
```