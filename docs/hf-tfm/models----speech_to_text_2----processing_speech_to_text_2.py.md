# `.\models\speech_to_text_2\processing_speech_to_text_2.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Speech processor class for Speech2Text2
"""
import warnings
from contextlib import contextmanager

from ...processing_utils import ProcessorMixin

# 导入 ProcessorMixin 类，该类提供了处理器的基本功能

class Speech2Text2Processor(ProcessorMixin):
    r"""
    Constructs a Speech2Text2 processor which wraps a Speech2Text2 feature extractor and a Speech2Text2 tokenizer into
    a single processor.

    [`Speech2Text2Processor`] offers all the functionalities of [`AutoFeatureExtractor`] and [`Speech2Text2Tokenizer`].
    See the [`~Speech2Text2Processor.__call__`] and [`~Speech2Text2Processor.decode`] for more information.

    Args:
        feature_extractor (`AutoFeatureExtractor`):
            An instance of [`AutoFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`Speech2Text2Tokenizer`):
            An instance of [`Speech2Text2Tokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "AutoFeatureExtractor"
    # 类属性：特征提取器类名，值为字符串 "AutoFeatureExtractor"
    tokenizer_class = "Speech2Text2Tokenizer"
    # 类属性：分词器类名，值为字符串 "Speech2Text2Tokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        # 调用父类 ProcessorMixin 的构造函数，初始化特征提取器和分词器
        self.current_processor = self.feature_extractor
        # 设置当前处理器为特征提取器对象
        self._in_target_context_manager = False
        # 私有属性：标志当前不处于目标上下文管理器中
    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoFeatureExtractor's
        [`~AutoFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~Speech2Text2Processor.as_target_processor`] this method forwards all its arguments to
        Speech2Text2Tokenizer's [`~Speech2Text2Tokenizer.__call__`]. Please refer to the doctsring of the above two
        methods for more information.
        """
        # 对于向后兼容性
        if self._in_target_context_manager:
            # 如果处于目标处理器上下文管理器中，则调用当前处理器的方法并返回其输出
            return self.current_processor(*args, **kwargs)

        if "raw_speech" in kwargs:
            # 如果 kwargs 中包含 "raw_speech" 关键字参数，则发出警告并将其替换为 "audio"
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")
        else:
            # 否则，从 kwargs 中弹出 "audio" 参数，如果没有则设为 None
            audio = kwargs.pop("audio", None)
        # 从 kwargs 中弹出 "sampling_rate" 和 "text" 参数，如果没有则设为 None
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            # 如果传入了额外的位置参数，则将第一个参数视为 audio，并将剩余参数放入 args 中
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            # 如果既没有传入 audio，也没有传入 text，则抛出 ValueError
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            # 如果传入了 audio，则使用特征提取器处理 audio，将结果存储在 inputs 中
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            # 如果传入了 text，则使用 tokenizer 处理 text，将结果存储在 encodings 中
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            # 如果没有传入 text，则返回处理后的 inputs
            return inputs
        elif audio is None:
            # 如果没有传入 audio，则返回处理后的 encodings
            return encodings
        else:
            # 如果既有 audio 又有 text，则将 encodings 的 "input_ids" 存入 inputs 的 "labels" 中，然后返回 inputs
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2Text2Tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 将所有参数转发给 tokenizer 的 batch_decode 方法，并返回其输出
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Speech2Text2Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 将所有参数转发给 tokenizer 的 decode 方法，并返回其输出
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        Speech2Text2.
        """
        # 发出警告，说明此方法将在 Transformers v5 中移除，推荐在普通 __call__ 方法的 text 参数中处理标签
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        # 将 _in_target_context_manager 标志设为 True，当前处理器设为 tokenizer
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        # 在退出上下文管理器后，将当前处理器设回 feature_extractor，并将 _in_target_context_manager 标志设为 False
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
```