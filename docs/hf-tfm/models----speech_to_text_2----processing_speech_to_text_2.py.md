# `.\transformers\models\speech_to_text_2\processing_speech_to_text_2.py`

```
# 设置编码格式为 UTF-8
# 版权声明
"""
Speech2Text2 的语音处理类
"""
# 引入警告模块
import warnings
# 引入上下文管理器模块
from contextlib import contextmanager

# 引入处理工具模块 ProcessorMixin
from ...processing_utils import ProcessorMixin


# 定义 Speech2Text2Processor 类，并继承 ProcessorMixin 类
class Speech2Text2Processor(ProcessorMixin):
    r"""
    构建 Speech2Text2 处理器，将 Speech2Text2 特征提取器和 Speech2Text2 分词器封装到单个处理器中。

    [`Speech2Text2Processor`] 提供了 [`AutoFeatureExtractor`] 和 [`Speech2Text2Tokenizer`] 的所有功能。
    有关更多信息，请参阅 [`~Speech2Text2Processor.__call__`] 和 [`~Speech2Text2Processor.decode`]。

    Args:
        feature_extractor (`AutoFeatureExtractor`):
            [`AutoFeatureExtractor`] 的实例。特征提取器是必需的输入。
        tokenizer (`Speech2Text2Tokenizer`):
            [`Speech2Text2Tokenizer`] 的实例。分词器是必需的输入。
    """

    # 特征提取器类名
    feature_extractor_class = "AutoFeatureExtractor"
    # 分词器类名
    tokenizer_class = "Speech2Text2Tokenizer"

    # 初始化方法
    def __init__(self, feature_extractor, tokenizer):
        # 调用父类的初始化方法
        super().__init__(feature_extractor, tokenizer)
        # 将当前处理器设置为特征提取器
        self.current_processor = self.feature_extractor
        # 设置目标上下文管理器的标志为 False
        self._in_target_context_manager = False
    def __call__(self, *args, **kwargs):
        """
        当在普通模式下使用时，该方法将所有参数转发到 AutoFeatureExtractor 的 [`~AutoFeatureExtractor.__call__`]，并返回其输出。
        如果在上下文中使用 [`~Speech2Text2Processor.as_target_processor`]，则该方法将所有参数转发到 Speech2Text2Tokenizer 的 [`~Speech2Text2Tokenizer.__call__`]。
        请参考上述两个方法的文档以获取更多信息。
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

    def batch_decode(self, *args, **kwargs):
        """
        该方法将所有参数转发到 Speech2Text2Tokenizer 的 [`~PreTrainedTokenizer.batch_decode`]。请参考该方法的文档以获取更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        该方法将所有参数转发到 Speech2Text2Tokenizer 的 [`~PreTrainedTokenizer.decode`]。请参考该方法的文档以获取更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        临时设置用于处理输入的分词器。在 Fine-tuning Speech2Text2 时编码标签非常有用。
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