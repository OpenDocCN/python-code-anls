# `.\models\wav2vec2\processing_wav2vec2.py`

```
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
Speech processor class for Wav2Vec2
"""
import warnings
from contextlib import contextmanager

# 导入处理工具模块中的混合处理器类
from ...processing_utils import ProcessorMixin
# 导入Wav2Vec2的特征提取器和CTC标记器
from .feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor
from .tokenization_wav2vec2 import Wav2Vec2CTCTokenizer

# Wav2Vec2处理器类，继承自混合处理器类
class Wav2Vec2Processor(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

    [`Wav2Vec2Processor`] offers all the functionalities of [`Wav2Vec2FeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~Wav2Vec2Processor.__call__`] and [`~Wav2Vec2Processor.decode`] for more information.

    Args:
        feature_extractor (`Wav2Vec2FeatureExtractor`):
            An instance of [`Wav2Vec2FeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    # 类属性，指定特征提取器的类名
    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    # 类属性，指定标记器的类名
    tokenizer_class = "AutoTokenizer"

    # 初始化方法，接受特征提取器和标记器作为参数
    def __init__(self, feature_extractor, tokenizer):
        # 调用父类的初始化方法
        super().__init__(feature_extractor, tokenizer)
        # 设置当前处理器为特征提取器
        self.current_processor = self.feature_extractor
        # 初始化目标上下文管理器状态为False
        self._in_target_context_manager = False

    # 类方法，从预训练模型加载处理器
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            # 尝试从预训练模型加载处理器
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            # 如果加载失败，发出警告
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            # 从预训练模型加载Wav2Vec2的特征提取器和CTC标记器
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            # 返回加载后的处理器对象
            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.__call__`] and returns its output. If used in the context
        [`~Wav2Vec2Processor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.__call__`]. Please refer to the docstring of the above two methods for more information.
        """
        # 对于向后兼容性
        # 如果当前处于目标上下文管理器中，则调用当前处理器的方法并传递所有参数
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        # 如果关键字参数中包含 "raw_speech"，发出警告并将其替换为 "audio"
        if "raw_speech" in kwargs:
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")
        else:
            audio = kwargs.pop("audio", None)

        # 获取采样率和文本，如果提供的话
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)

        # 如果位置参数 args 中有值，则将第一个值视为 audio，其余的值保留在 args 中
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        # 如果既没有提供 audio 也没有提供 text，则抛出 ValueError
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        # 如果提供了 audio，则使用 feature_extractor 处理输入
        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)

        # 如果提供了 text，则使用 tokenizer 处理输入
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        # 根据输入的类型返回相应的处理结果
        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            # 将 encodings 中的 "input_ids" 赋值给 inputs 中的 "labels" 键
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.pad`] and returns its output. If used in the context
        [`~Wav2Vec2Processor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.pad`]. Please refer to the docstring of the above two methods for more information.
        """
        # 对于向后兼容性
        # 如果当前处于目标上下文管理器中，则调用当前处理器的 pad 方法并传递所有参数
        if self._in_target_context_manager:
            return self.current_processor.pad(*args, **kwargs)

        # 获取输入特征和标签，如果提供的话
        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)

        # 如果位置参数 args 中有值，则将第一个值视为 input_features，其余的值保留在 args 中
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]

        # 如果提供了 input_features，则使用 feature_extractor 的 pad 方法进行填充
        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)

        # 如果提供了 labels，则使用 tokenizer 的 pad 方法进行填充
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        # 根据输入的类型返回相应的填充结果
        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            # 将 labels 中的 "input_ids" 赋值给 input_features 中的 "labels" 键
            input_features["labels"] = labels["input_ids"]
            return input_features
    # 此方法将所有参数转发给 PreTrainedTokenizer 的 `batch_decode` 方法。请参考该方法的文档字符串获取更多信息。
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 此方法将所有参数转发给 PreTrainedTokenizer 的 `decode` 方法。请参考该方法的文档字符串获取更多信息。
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 上下文管理器：临时设置处理输入的 tokenizer。在微调 Wav2Vec2 时，用于编码标签。
    def as_target_processor(self):
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        # 标记当前处于目标处理器上下文管理器中
        self._in_target_context_manager = True
        # 将当前处理器设置为 tokenizer
        self.current_processor = self.tokenizer
        yield  # 返回上下文
        # 恢复当前处理器为特征提取器
        self.current_processor = self.feature_extractor
        # 标记目标处理器上下文管理器结束
        self._in_target_context_manager = False
```