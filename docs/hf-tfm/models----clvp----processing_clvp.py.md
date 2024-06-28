# `.\models\clvp\processing_clvp.py`

```py
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Processor class for CLVP
"""


from ...processing_utils import ProcessorMixin

# 导入处理工具类 ProcessorMixin


class ClvpProcessor(ProcessorMixin):
    """
    Constructs a CLVP processor which wraps a CLVP Feature Extractor and a CLVP Tokenizer into a single processor.

    [`ClvpProcessor`] offers all the functionalities of [`ClvpFeatureExtractor`] and [`ClvpTokenizer`]. See the
    [`~ClvpProcessor.__call__`], [`~ClvpProcessor.decode`] and [`~ClvpProcessor.batch_decode`] for more information.

    Args:
        feature_extractor (`ClvpFeatureExtractor`):
            An instance of [`ClvpFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`ClvpTokenizer`):
            An instance of [`ClvpTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "ClvpFeatureExtractor"
    tokenizer_class = "ClvpTokenizer"
    model_input_names = [
        "input_ids",
        "input_features",
        "attention_mask",
    ]

    def __init__(self, feature_extractor, tokenizer):
        # 初始化方法，接收 CLVP Feature Extractor 和 CLVP Tokenizer 实例，并调用父类构造函数
        super().__init__(feature_extractor, tokenizer)

    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` and `sampling_rate` arguments to [`~ClvpFeatureExtractor.__call__`] and the `text`
        argument to [`~ClvpTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        
        raw_speech = kwargs.pop("raw_speech", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)

        if raw_speech is None and text is None:
            # 如果既没有原始语音输入也没有文本输入，则抛出数值错误
            raise ValueError("You need to specify either an `raw_speech` or `text` input to process.")

        if raw_speech is not None:
            # 如果有原始语音输入，则调用 CLVP Feature Extractor 处理原始语音和采样率参数
            inputs = self.feature_extractor(raw_speech, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            # 如果有文本输入，则调用 CLVP Tokenizer 处理文本参数
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            # 如果只有原始语音输入，则返回 CLVP Feature Extractor 的输出
            return inputs
        elif raw_speech is None:
            # 如果只有文本输入，则返回 CLVP Tokenizer 的输出
            return encodings
        else:
            # 如果同时有原始语音和文本输入，则将 CLVP Tokenizer 的输出合并到 CLVP Feature Extractor 的输出中，并返回
            inputs["input_ids"] = encodings["input_ids"]
            inputs["attention_mask"] = encodings["attention_mask"]
            return inputs

    # Copied from transformers.models.whisper.processing_whisper.WhisperProcessor.batch_decode with Whisper->Clvp
    # 此方法将所有参数转发给 ClvpTokenizer 的 `batch_decode` 方法。
    # 请参考该方法的文档字符串获取更多信息。
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 从 transformers.models.whisper.processing_whisper.WhisperProcessor.decode 复制而来，
    # 将 Whisper 替换为 Clvp
    # 此方法将所有参数转发给 ClvpTokenizer 的 `decode` 方法。
    # 请参考该方法的文档字符串获取更多信息。
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
```