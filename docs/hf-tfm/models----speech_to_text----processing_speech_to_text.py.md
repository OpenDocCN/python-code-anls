# `.\models\speech_to_text\processing_speech_to_text.py`

```
# coding=utf-8
# 设置文件编码格式为 UTF-8

# 版权声明：2021 年由 HuggingFace Inc. 团队版权所有。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”提供软件，
# 无论是明示还是默示的保证或条件。
# 有关更多详细信息，请参见许可证。

"""
Speech2Text 的语音处理器类
"""
# 引入警告模块
import warnings
# 引入上下文管理器
from contextlib import contextmanager

# 从本地引入处理工具函数 ProcessorMixin
from ...processing_utils import ProcessorMixin


class Speech2TextProcessor(ProcessorMixin):
    """
    构造一个 Speech2Text 处理器，将 Speech2Text 特征提取器和 Speech2Text 分词器封装到单个处理器中。

    [`Speech2TextProcessor`] 提供了 [`Speech2TextFeatureExtractor`] 和 [`Speech2TextTokenizer`] 的所有功能。
    查看 [`~Speech2TextProcessor.__call__`] 和 [`~Speech2TextProcessor.decode`] 获取更多信息。

    Args:
        feature_extractor (`Speech2TextFeatureExtractor`):
            [`Speech2TextFeatureExtractor`] 的一个实例。特征提取器是必需的输入。
        tokenizer (`Speech2TextTokenizer`):
            [`Speech2TextTokenizer`] 的一个实例。分词器是必需的输入。
    """

    # 类属性，特征提取器的类名
    feature_extractor_class = "Speech2TextFeatureExtractor"
    # 类属性，分词器的类名
    tokenizer_class = "Speech2TextTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        # 调用父类 ProcessorMixin 的构造方法
        super().__init__(feature_extractor, tokenizer)
        # 将特征提取器作为当前处理器
        self.current_processor = self.feature_extractor
        # 内部标志，用于目标上下文管理器
        self._in_target_context_manager = False
    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Speech2TextFeatureExtractor's
        [`~Speech2TextFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~Speech2TextProcessor.as_target_processor`] this method forwards all its arguments to Speech2TextTokenizer's
        [`~Speech2TextTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # 对于向后兼容性
        if self._in_target_context_manager:
            # 如果处于目标处理器上下文管理器中，将参数转发给当前处理器（tokenizer）
            return self.current_processor(*args, **kwargs)

        if "raw_speech" in kwargs:
            # 如果使用了过时的关键字参数 `raw_speech`，发出警告并改用 `audio`
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")
        else:
            # 否则，使用关键字参数 `audio`，默认为 None
            audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            # 如果位置参数 args 不为空，将第一个位置参数作为 audio，并将其余位置参数赋给 args
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            # 如果既没有 audio 也没有 text 输入，则抛出 ValueError
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            # 如果有 audio 输入，则使用特征提取器（feature_extractor）处理
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            # 如果有 text 输入，则使用 tokenizer 处理
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            # 如果只有 audio 输入，则返回处理后的 inputs
            return inputs
        elif audio is None:
            # 如果只有 text 输入，则返回处理后的 encodings
            return encodings
        else:
            # 如果既有 audio 又有 text 输入，则将 encodings 的 input_ids 作为 labels 放入 inputs，并返回
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2TextTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 将所有参数转发给 tokenizer 的 batch_decode 方法，并返回其输出
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2TextTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 将所有参数转发给 tokenizer 的 decode 方法，并返回其输出
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Speech2Text.
        """
        # 发出警告，说明该方法即将被移除，并建议使用 __call__ 方法中的 text 参数来处理标签
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer  # 设置当前处理器为 tokenizer
        yield  # 执行代码块直到 yield
        self.current_processor = self.feature_extractor  # 恢复当前处理器为 feature_extractor
        self._in_target_context_manager = False  # 结束目标处理器上下文管理器
```