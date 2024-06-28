# `.\models\musicgen\processing_musicgen.py`

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
Text/audio processor class for MusicGen
"""
from typing import List, Optional

import numpy as np

from ...processing_utils import ProcessorMixin
from ...utils import to_numpy


class MusicgenProcessor(ProcessorMixin):
    r"""
    Constructs a MusicGen processor which wraps an EnCodec feature extractor and a T5 tokenizer into a single processor
    class.

    [`MusicgenProcessor`] offers all the functionalities of [`EncodecFeatureExtractor`] and [`TTokenizer`]. See
    [`~MusicgenProcessor.__call__`] and [`~MusicgenProcessor.decode`] for more information.

    Args:
        feature_extractor (`EncodecFeatureExtractor`):
            An instance of [`EncodecFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`T5Tokenizer`):
            An instance of [`T5Tokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "EncodecFeatureExtractor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        # 调用父类构造函数初始化特征提取器和分词器
        super().__init__(feature_extractor, tokenizer)
        # 将特征提取器设为当前处理器
        self.current_processor = self.feature_extractor
        # 设定目标上下文管理器标志为假
        self._in_target_context_manager = False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        # 调用分词器的方法获取解码器提示 ID
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)
    # 实现对象的可调用行为，将 `audio` 参数传递给 `EncodecFeatureExtractor` 的 [`~EncodecFeatureExtractor.__call__`] 方法，
    # 将 `text` 参数传递给 [`~T5Tokenizer.__call__`] 方法。更多信息请参考上述两个方法的文档字符串。
    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to EncodecFeatureExtractor's [`~EncodecFeatureExtractor.__call__`] and the `text`
        argument to [`~T5Tokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # 为了向后兼容
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        # 从 kwargs 中弹出 `audio`、`sampling_rate` 和 `text` 参数
        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)

        # 如果有位置参数，则将第一个参数作为 `audio`，其余的作为 `args`
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        # 如果 `audio` 和 `text` 均为 None，则抛出 ValueError
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        # 如果存在 `text` 参数，则使用 tokenizer 处理文本
        if text is not None:
            inputs = self.tokenizer(text, **kwargs)

        # 如果存在 `audio` 参数，则使用 feature_extractor 处理音频
        if audio is not None:
            audio_inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)

        # 如果只有 `audio` 参数，则返回处理后的文本输入
        if audio is None:
            return inputs

        # 如果只有 `text` 参数，则返回处理后的音频输入
        elif text is None:
            return audio_inputs

        # 如果既有 `audio` 又有 `text` 参数，则合并处理结果并返回
        else:
            inputs["input_values"] = audio_inputs["input_values"]
            if "padding_mask" in audio_inputs:
                inputs["padding_mask"] = audio_inputs["padding_mask"]
            return inputs

    # 批量解码方法，用于解码来自 MusicGen 模型的音频输出批次或来自 tokenizer 的 token ids 批次
    def batch_decode(self, *args, **kwargs):
        """
        This method is used to decode either batches of audio outputs from the MusicGen model, or batches of token ids
        from the tokenizer. In the case of decoding token ids, this method forwards all its arguments to T5Tokenizer's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more information.
        """
        # 从 kwargs 中弹出 `audio` 和 `padding_mask` 参数
        audio_values = kwargs.pop("audio", None)
        padding_mask = kwargs.pop("padding_mask", None)

        # 如果有位置参数，则将第一个参数作为 `audio_values`，其余的作为 `args`
        if len(args) > 0:
            audio_values = args[0]
            args = args[1:]

        # 如果存在 `audio_values` 参数，则调用 `_decode_audio` 方法解码音频
        if audio_values is not None:
            return self._decode_audio(audio_values, padding_mask=padding_mask)
        else:
            # 否则调用 tokenizer 的 batch_decode 方法解码 token ids
            return self.tokenizer.batch_decode(*args, **kwargs)

    # 解码方法，将所有参数转发给 T5Tokenizer 的 [`~PreTrainedTokenizer.decode`] 方法
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to T5Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
    def _decode_audio(self, audio_values, padding_mask: Optional = None) -> List[np.ndarray]:
        """
        This method strips any padding from the audio values to return a list of numpy audio arrays.
        """
        # 将输入的音频值转换为 numpy 数组
        audio_values = to_numpy(audio_values)
        
        # 获取批量大小、通道数和序列长度
        bsz, channels, seq_len = audio_values.shape

        # 如果没有提供填充掩码，则直接返回音频值的列表
        if padding_mask is None:
            return list(audio_values)

        # 将填充掩码也转换为 numpy 数组
        padding_mask = to_numpy(padding_mask)

        # 计算填充掩码的序列长度差，以便与生成的音频数组匹配
        difference = seq_len - padding_mask.shape[-1]

        # 根据填充值（非填充标记）填充填充掩码，确保生成的音频值不被视为填充标记
        padding_value = 1 - self.feature_extractor.padding_value
        padding_mask = np.pad(padding_mask, ((0, 0), (0, difference)), "constant", constant_values=padding_value)

        # 将音频值转换为列表形式以便后续处理
        audio_values = audio_values.tolist()
        for i in range(bsz):
            # 根据填充掩码切片音频数组，去除填充部分，并重新整形为通道数和变长序列
            sliced_audio = np.asarray(audio_values[i])[
                padding_mask[i][None, :] != self.feature_extractor.padding_value
            ]
            audio_values[i] = sliced_audio.reshape(channels, -1)

        # 返回处理后的音频值列表
        return audio_values
```