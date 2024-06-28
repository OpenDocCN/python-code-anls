# `.\models\musicgen_melody\processing_musicgen_melody.py`

```py
# coding=utf-8
# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
Text/audio processor class for MusicGen Melody
"""
from typing import List, Optional

import numpy as np

from ...processing_utils import ProcessorMixin
from ...utils import to_numpy


class MusicgenMelodyProcessor(ProcessorMixin):
    r"""
    Constructs a MusicGen Melody processor which wraps a Wav2Vec2 feature extractor - for raw audio waveform processing - and a T5 tokenizer into a single processor
    class.

    [`MusicgenProcessor`] offers all the functionalities of [`MusicgenMelodyFeatureExtractor`] and [`T5Tokenizer`]. See
    [`~MusicgenProcessor.__call__`] and [`~MusicgenProcessor.decode`] for more information.

    Args:
        feature_extractor (`MusicgenMelodyFeatureExtractor`):
            An instance of [`MusicgenMelodyFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`T5Tokenizer`):
            An instance of [`T5Tokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "MusicgenMelodyFeatureExtractor"
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    # Copied from transformers.models.musicgen.processing_musicgen.MusicgenProcessor.get_decoder_prompt_ids
    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        """
        Retrieves decoder prompt IDs from the tokenizer.
        
        Args:
            task (str, optional): Task identifier. Defaults to None.
            language (str, optional): Language identifier. Defaults to None.
            no_timestamps (bool, optional): Flag indicating whether to exclude timestamps. Defaults to True.
        
        Returns:
            List[int]: List of decoder prompt IDs based on the provided task, language, and timestamps preferences.
        """
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)
    # 定义一个方法，用于处理模型的输入，支持音频和文本的预处理
    def __call__(self, audio=None, text=None, **kwargs):
        """
        主方法，用于准备模型的一个或多个序列和音频。如果 `audio` 不为 `None`，则将 `audio` 和 `kwargs` 参数传递给 MusicgenMelodyFeatureExtractor 的 [`~MusicgenMelodyFeatureExtractor.__call__`] 来预处理音频。如果 `text` 不为 `None`，则将 `text` 和 `kwargs` 参数传递给 PreTrainedTokenizer 的 [`~PreTrainedTokenizer.__call__`]。请参考上述两个方法的文档字符串获取更多信息。

        Args:
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                要准备的音频或音频批处理。每个音频可以是 NumPy 数组或 PyTorch 张量。如果是 NumPy 数组/PyTorch 张量，则每个音频应为形状为 (T) 的单声道或立体声信号，其中 T 是音频的样本长度。
            text (`str`, `List[str]`, `List[List[str]]`):
                要编码的序列或序列批处理。每个序列可以是字符串或字符串列表（预分词字符串）。如果作为字符串列表（预分词）提供序列，则必须设置 `is_split_into_words=True`（以消除与序列批处理的歧义）。
            kwargs (*optional*):
                剩余的关键字参数字典，将传递给特征提取器和/或标记器。
        Returns:
            [`BatchEncoding`]: 一个 [`BatchEncoding`]，具有以下字段：
            - **input_ids** -- 要输入模型的令牌 ID 列表。在 `text` 不为 `None` 时返回。
            - **input_features** -- 要输入模型的音频输入特征。在 `audio` 不为 `None` 时返回。
            - **attention_mask** -- 列表的令牌索引，指定模型在 `text` 不为 `None` 时应注意哪些令牌。
            当仅指定 `audio` 时，返回时间戳的注意力蒙版。
        """

        # 从 `kwargs` 中弹出 `sampling_rate` 参数
        sampling_rate = kwargs.pop("sampling_rate", None)

        # 如果 `audio` 和 `text` 均为 `None`，则抛出 ValueError
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        # 如果 `text` 不为 `None`，则使用标记器处理文本输入
        if text is not None:
            inputs = self.tokenizer(text, **kwargs)
        
        # 如果 `audio` 不为 `None`，则使用特征提取器处理音频输入
        if audio is not None:
            audio_inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, **kwargs)

        # 如果仅有音频输入，则返回音频处理结果
        if text is None:
            return audio_inputs
        # 如果仅有文本输入，则返回文本处理结果
        elif audio is None:
            return inputs
        # 如果同时有文本和音频输入，则将音频特征添加到文本处理结果中后返回
        else:
            inputs["input_features"] = audio_inputs["input_features"]
            return inputs

    # 从 transformers.models.musicgen.processing_musicgen.MusicgenProcessor.batch_decode 复制，将 padding_mask 改为 attention_mask
    # 从关键字参数中弹出音频输出值，如果没有则为 None
    audio_values = kwargs.pop("audio", None)
    # 从关键字参数中弹出注意力掩码，如果没有则为 None
    attention_mask = kwargs.pop("attention_mask", None)

    # 如果位置参数的数量大于 0
    if len(args) > 0:
        # 将第一个位置参数作为音频输出值
        audio_values = args[0]
        # 剩余位置参数重新赋值为 args 去掉第一个元素后的部分
        args = args[1:]

    # 如果音频输出值不为 None
    if audio_values is not None:
        # 调用 _decode_audio 方法解码音频输出值，并传入注意力掩码
        return self._decode_audio(audio_values, attention_mask=attention_mask)
    else:
        # 否则调用 tokenizer 对象的 batch_decode 方法，传入剩余的位置参数和关键字参数
        return self.tokenizer.batch_decode(*args, **kwargs)


# 从 transformers.models.musicgen.processing_musicgen.MusicgenProcessor.decode 复制过来
# 此方法将所有参数转发给 T5Tokenizer 的 decode 方法
def decode(self, *args, **kwargs):
    """
    This method forwards all its arguments to T5Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
    docstring of this method for more information.
    """
    return self.tokenizer.decode(*args, **kwargs)


# 从 transformers.models.musicgen.processing_musicgen.MusicgenProcessor._decode_audio 复制过来
# 此方法将音频输出值解码并去除任何填充，返回一个 numpy 音频数组列表
def _decode_audio(self, audio_values, attention_mask: Optional = None) -> List[np.ndarray]:
    """
    This method strips any padding from the audio values to return a list of numpy audio arrays.
    """
    # 将音频输出值转换为 numpy 数组
    audio_values = to_numpy(audio_values)
    # 获取批次大小、通道数和序列长度
    bsz, channels, seq_len = audio_values.shape

    # 如果注意力掩码为 None，则直接返回音频数组列表
    if attention_mask is None:
        return list(audio_values)

    # 将注意力掩码转换为 numpy 数组
    attention_mask = to_numpy(attention_mask)

    # 匹配填充掩码的序列长度与生成的音频数组长度，使用非填充标记进行填充
    difference = seq_len - attention_mask.shape[-1]
    padding_value = 1 - self.feature_extractor.padding_value
    attention_mask = np.pad(attention_mask, ((0, 0), (0, difference)), "constant", constant_values=padding_value)

    # 将音频输出值转换为列表形式
    audio_values = audio_values.tolist()
    # 遍历批次中的每个样本
    for i in range(bsz):
        # 根据注意力掩码切片生成音频数组，去除填充标记
        sliced_audio = np.asarray(audio_values[i])[
            attention_mask[i][None, :] != self.feature_extractor.padding_value
        ]
        # 将处理后的音频数组重新整形，并存入音频值列表中
        audio_values[i] = sliced_audio.reshape(channels, -1)

    # 返回处理后的音频值列表
    return audio_values
    # Helper function to generate null inputs for unconditional generation, which allows using the model without
    # the feature extractor or tokenizer.
    def get_unconditional_inputs(self, num_samples=1, return_tensors="pt"):
        """
        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.

        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.

        Example:
        ```
        >>> from transformers import MusicgenMelodyForConditionalGeneration, MusicgenMelodyProcessor

        >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")

        >>> # get the unconditional (or 'null') inputs for the model
        >>> processor = MusicgenMelodyProcessor.from_pretrained("facebook/musicgen-melody")
        >>> unconditional_inputs = processor.get_unconditional_inputs(num_samples=1)

        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
        ```"""
        # Use the tokenizer to encode empty strings for the specified number of samples,
        # returning tensors suitable for PyTorch (or other frameworks).
        inputs = self.tokenizer([""] * num_samples, return_tensors=return_tensors, return_attention_mask=True)
        # Set attention mask to zero for all samples to indicate that no tokens should be attended to.
        inputs["attention_mask"][:] = 0

        # Return the prepared inputs dictionary.
        return inputs
```