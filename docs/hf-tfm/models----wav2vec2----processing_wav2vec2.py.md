# `.\transformers\models\wav2vec2\processing_wav2vec2.py`

```py
# 设置文件编码为 utf-8
# 版权声明为 2021 年 HuggingFace 公司团队所有
# 使用 Apache 许可证 2.0 版本进行许可
# 只能在遵守许可证的情况下使用这个文件
# 你可以在以下链接找到许可证的拷贝: http://www.apache.org/licenses/LICENSE-2.0
#
# 除非必须法律要求或书面同意，否则依据许可证进行分发的软件在 "按原样" 基础上分发，
# 没有任何担保或条件，无论明示或暗示的，包括但不限于，特定用途的担保隐含担保
# 参阅许可证以了解特定的语言限制和条件
"""
Wav2Vec2 的语音处理器类
"""
# 导入警告模块
import warnings
# 导入上下文管理器
from contextlib import contextmanager
# 导入处理工具模块
from ...processing_utils import ProcessorMixin
# 导入特征提取模块
from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
# 导入标记化模块
from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer

class Wav2Vec2Processor(ProcessorMixin):
    r"""
    构建一个 Wav2Vec2 处理器，该处理器封装了一个 Wav2Vec2 特征提取器和一个 Wav2Vec2 CTC 标记化器

    [`Wav2Vec2Processor`] 提供了所有 [`Wav2Vec2FeatureExtractor`] 和 [`PreTrainedTokenizer`] 的功能。
    有关更多信息，请参阅 [`~Wav2Vec2Processor.__call__`] 和 [`~Wav2Vec2Processor.decode`] 的文档字符串

    Args:
        feature_extractor (`Wav2Vec2FeatureExtractor`):
            [`Wav2Vec2FeatureExtractor`] 的实例。特征提取器是必需的输入
        tokenizer ([`PreTrainedTokenizer`]):
            [`PreTrainedTokenizer`] 的实例。标记化器是必需的输入
    """

    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
    def __call__(self, *args, **kwargs):
        """
        当在正常模式下使用时，此方法将所有参数转发到Wav2Vec2FeatureExtractor的[`~Wav2Vec2FeatureExtractor.__call__`]并返回其输出。
        如果在[`~Wav2Vec2Processor.as_target_processor`]上下文中使用此方法，此方法将所有参数转发到PreTrainedTokenizer的[`~PreTrainedTokenizer.__call__`]。
        有关更多信息，请参考上述两种方法的文档字符串。
        """
        # 为了向后兼容
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        if "raw_speech" in kwargs:
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")
        else:
            audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def pad(self, *args, **kwargs):
        """
        当在正常模式下使用时，此方法将所有参数转发到Wav2Vec2FeatureExtractor的[`~Wav2Vec2FeatureExtractor.pad`]并返回其输出。
        如果在[`~Wav2Vec2Processor.as_target_processor`]上下文中使用此方法，此方法将所有参数转发到PreTrainedTokenizer的[`~PreTrainedTokenizer.pad`]。
        有关更多信息，请参考上述两种方法的文档字符串。
        """
        # 为了向后兼容
        if self._in_target_context_manager:
            return self.current_processor.pad(*args, **kwargs)

        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]

        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features
    # 将所有参数转发给PreTrainedTokenizer的`~PreTrainedTokenizer.batch_decode`方法
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 将所有参数转发给PreTrainedTokenizer的`~PreTrainedTokenizer.decode`方法
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 临时设置处理输入的分词器，用于在微调Wav2Vec2时编码标签
    @contextmanager
    def as_target_processor(self):
        # 发出警告，表示`as_target_processor`方法已过时，并将在Transformers的v5版本中删除
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        # 设置当前上下文为目标处理器
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        # 重置当前处理器为特征提取器
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
```