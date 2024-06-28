# `.\models\deprecated\mctct\processing_mctct.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
Speech processor class for M-CTC-T
"""
import warnings
from contextlib import contextmanager

from ....processing_utils import ProcessorMixin


class MCTCTProcessor(ProcessorMixin):
    r"""
    Constructs a MCTCT processor which wraps a MCTCT feature extractor and a MCTCT tokenizer into a single processor.

    [`MCTCTProcessor`] offers all the functionalities of [`MCTCTFeatureExtractor`] and [`AutoTokenizer`]. See the
    [`~MCTCTProcessor.__call__`] and [`~MCTCTProcessor.decode`] for more information.

    Args:
        feature_extractor (`MCTCTFeatureExtractor`):
            An instance of [`MCTCTFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of [`AutoTokenizer`]. The tokenizer is a required input.
    """

    # 类属性，指定特征提取器的类名
    feature_extractor_class = "MCTCTFeatureExtractor"
    # 类属性，指定分词器的类名
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        # 调用父类的构造函数，传入特征提取器和分词器实例
        super().__init__(feature_extractor, tokenizer)
        # 设置当前处理器为特征提取器实例
        self.current_processor = self.feature_extractor
        # 内部标志，指示是否处于目标上下文管理器中的状态，默认为 False
        self._in_target_context_manager = False
    # 当对象被调用时执行的方法，根据当前上下文执行不同的操作
    """
    当对象在正常模式下使用时，此方法将所有参数转发给 MCTCTFeatureExtractor 的 [`~MCTCTFeatureExtractor.__call__`] 并返回其输出。
    如果在 [`~MCTCTProcessor.as_target_processor`] 上下文中使用，则将所有参数转发给 AutoTokenizer 的 [`~AutoTokenizer.__call__`]。
    更多信息请参考上述两个方法的文档字符串。
    """
    # 对于向后兼容性
    if self._in_target_context_manager:
        # 如果处于目标处理器上下文中，直接调用当前处理器的方法，并返回其输出
        return self.current_processor(*args, **kwargs)

    # 如果关键字参数中包含 "raw_speech"，发出警告并使用 "audio" 替代
    if "raw_speech" in kwargs:
        warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
        audio = kwargs.pop("raw_speech")
    else:
        # 否则，尝试从关键字参数中获取 "audio"，默认为 None
        audio = kwargs.pop("audio", None)
    # 尝试从关键字参数中获取 "sampling_rate"，默认为 None
    sampling_rate = kwargs.pop("sampling_rate", None)
    # 尝试从关键字参数中获取 "text"，默认为 None
    text = kwargs.pop("text", None)
    # 如果位置参数 args 不为空，则将第一个参数作为 audio，并从 args 中移除
    if len(args) > 0:
        audio = args[0]
        args = args[1:]

    # 如果既没有 audio 也没有 text 输入，则抛出 ValueError
    if audio is None and text is None:
        raise ValueError("You need to specify either an `audio` or `text` input to process.")

    # 如果有音频输入，则使用 feature_extractor 处理音频数据
    if audio is not None:
        inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
    # 如果有文本输入，则使用 tokenizer 处理文本数据
    if text is not None:
        encodings = self.tokenizer(text, **kwargs)

    # 根据输入情况返回相应的结果
    if text is None:
        return inputs  # 如果只有音频输入，则返回音频处理的结果
    elif audio is None:
        return encodings  # 如果只有文本输入，则返回文本处理的结果
    else:
        inputs["labels"] = encodings["input_ids"]  # 如果既有音频又有文本输入，则将文本处理结果作为标签添加到音频处理结果中
        return inputs  # 返回整合后的结果字典

# 批量解码方法，将所有参数转发给 AutoTokenizer 的 [`~PreTrainedTokenizer.batch_decode`] 方法
def batch_decode(self, *args, **kwargs):
    """
    此方法将所有参数转发给 AutoTokenizer 的 [`~PreTrainedTokenizer.batch_decode`]。请参考该方法的文档字符串以获取更多信息。
    """
    return self.tokenizer.batch_decode(*args, **kwargs)
    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to MCTCTFeatureExtractor's
        [`~MCTCTFeatureExtractor.pad`] and returns its output. If used in the context
        [`~MCTCTProcessor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.pad`]. Please refer to the docstring of the above two methods for more information.
        """
        # 对于向后兼容性
        if self._in_target_context_manager:
            # 如果处于目标处理器上下文管理器中，则调用当前处理器的 pad 方法
            return self.current_processor.pad(*args, **kwargs)

        # 获取特定参数
        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)
        if len(args) > 0:
            # 如果有位置参数，将第一个位置参数作为 input_features，其余作为 args
            input_features = args[0]
            args = args[1:]

        # 如果 input_features 不为 None，则调用特征提取器的 pad 方法
        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        # 如果 labels 不为 None，则调用标记器的 pad 方法
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        # 根据输入是否为 None，返回相应结果
        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to AutoTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        # 将所有参数转发给 tokenizer 的 decode 方法
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning MCTCT.
        """
        # 发出警告信息，因为该方法即将在 v5 版本中移除
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        # 设置标志位，指示当前处于目标处理器上下文管理器中
        self._in_target_context_manager = True
        # 将当前处理器设置为 tokenizer
        self.current_processor = self.tokenizer
        yield
        # 在退出上下文管理器后，将当前处理器设置回特征提取器
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
```