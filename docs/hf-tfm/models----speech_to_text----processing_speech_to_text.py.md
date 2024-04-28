# `.\transformers\models\speech_to_text\processing_speech_to_text.py`

```
# 设置编码为 UTF-8
# 版权声明及许可协议
"""
Speech2Text 的语音处理类
"""
# 导入警告模块
import warnings
# 导入上下文管理器模块
from contextlib import contextmanager
# 导入处理工具类
from ...processing_utils import ProcessorMixin

# 定义 Speech2TextProcessor 类，继承自 ProcessorMixin
class Speech2TextProcessor(ProcessorMixin):
    r"""
    构建一个 Speech2Text 处理器，将 Speech2Text 特征提取器和 Speech2Text 分词器包装成一个单独的处理器。

    [`Speech2TextProcessor`] 提供了 [`Speech2TextFeatureExtractor`] 和 [`Speech2TextTokenizer`] 的所有功能。
    有关更多信息，请参阅 [`~Speech2TextProcessor.__call__`] 和 [`~Speech2TextProcessor.decode`]。

    Args:
        feature_extractor (`Speech2TextFeatureExtractor`):
            一个 [`Speech2TextFeatureExtractor`] 实例。特征提取器是必需的输入。
        tokenizer (`Speech2TextTokenizer`):
            一个 [`Speech2TextTokenizer`] 实例。分词器是必需的输入。
    """

    # 特征提取器类名
    feature_extractor_class = "Speech2TextFeatureExtractor"
    # 分词器类名
    tokenizer_class = "Speech2TextTokenizer"

    # 初始化方法
    def __init__(self, feature_extractor, tokenizer):
        # 调用父类的初始化方法
        super().__init__(feature_extractor, tokenizer)
        # 当前处理器设置为特征提取器
        self.current_processor = self.feature_extractor
        # 是否在目标上下文管理器中的标志
        self._in_target_context_manager = False
``` 
    def __call__(self, *args, **kwargs):
        """
        当在普通模式下使用时，此方法将所有参数转发到 Speech2TextFeatureExtractor 的 [`~Speech2TextFeatureExtractor.__call__`]，并返回其输出。
        如果在上下文 [`~Speech2TextProcessor.as_target_processor`] 中使用此方法，则将所有参数转发到 Speech2TextTokenizer 的 
        [`~Speech2TextTokenizer.__call__`]。请参考上述两个方法的文档字符串了解更多信息。
        """
        # 为了向后兼容性
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        # 如果 kwargs 中存在 "raw_speech" 键
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

    def batch_decode(self, *args, **kwargs):
        """
        将所有参数转发到 Speech2TextTokenizer 的 [`~PreTrainedTokenizer.batch_decode`] 方法。请参考此方法的文档字符串了解更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        将所有参数转发到 Speech2TextTokenizer 的 [`~PreTrainedTokenizer.decode`] 方法。请参考此方法的文档字符串了解更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        暂时设置用于处理输入的标记器。在 fine-tuning Speech2Text 时编码标签非常有用。
        """
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
```