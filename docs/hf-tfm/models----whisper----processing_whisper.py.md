# `.\models\whisper\processing_whisper.py`

```py
# coding=utf-8
# 文件编码声明，使用 UTF-8 编码
# Copyright 2022 The HuggingFace Inc. team.
# 版权声明，标明代码版权归 HuggingFace Inc. 团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 遵循 Apache License, Version 2.0 许可协议，允许在特定条件下使用本代码
# you may not use this file except in compliance with the License.
# 除非符合许可协议的条件，否则禁止使用本文件
# You may obtain a copy of the License at
# 可以在上述许可协议链接处获取完整的许可协议文本
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 根据适用法律或书面同意，本软件按“原样”分发，不提供任何担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何担保或条件，包括但不限于特定用途的适销性或适用性
# See the License for the specific language governing permissions and
# limitations under the License.
# 请参阅许可协议以了解权限和限制的具体条款

"""
Speech processor class for Whisper
"""
# 注释: Whisper 的语音处理器类

from ...processing_utils import ProcessorMixin
# 导入处理工具类 ProcessorMixin

class WhisperProcessor(ProcessorMixin):
    r"""
    Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single
    processor.
    
    构建 Whisper 处理器，将 Whisper 特征提取器和 Whisper 分词器封装到一个处理器中

    [`WhisperProcessor`] offers all the functionalities of [`WhisperFeatureExtractor`] and [`WhisperTokenizer`]. See
    the [`~WhisperProcessor.__call__`] and [`~WhisperProcessor.decode`] for more information.
    
    WhisperProcessor 提供了所有 WhisperFeatureExtractor 和 WhisperTokenizer 的功能。查看 `~WhisperProcessor.__call__` 和 `~WhisperProcessor.decode` 获取更多信息

    Args:
        feature_extractor (`WhisperFeatureExtractor`):
            An instance of [`WhisperFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`WhisperTokenizer`):
            An instance of [`WhisperTokenizer`]. The tokenizer is a required input.
    """
    # 参数说明：feature_extractor 是 WhisperFeatureExtractor 实例，tokenizer 是 WhisperTokenizer 实例

    feature_extractor_class = "WhisperFeatureExtractor"
    # 类属性：特征提取器类名为 WhisperFeatureExtractor
    tokenizer_class = "WhisperTokenizer"
    # 类属性：分词器类名为 WhisperTokenizer

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        # 调用父类的初始化方法，使用 feature_extractor 和 tokenizer 进行初始化
        self.current_processor = self.feature_extractor
        # 设置当前处理器为特征提取器对象
        self._in_target_context_manager = False
        # 设置目标上下文管理器状态为 False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)
        # 调用分词器的方法获取解码器提示符的 ID
    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` argument to WhisperFeatureExtractor's [`~WhisperFeatureExtractor.__call__`] and the `text`
        argument to [`~WhisperTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """
        # 如果在目标上下文管理器中，则调用当前处理器并返回结果
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        # 从 kwargs 中弹出 `audio`, `sampling_rate`, `text` 参数
        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        # 如果有额外的位置参数，将第一个作为 `audio` 参数处理，其余作为 args
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        # 如果 `audio` 和 `text` 都为 None，则抛出 ValueError
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        # 如果有 `audio`，调用特征提取器处理音频输入
        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        # 如果有 `text`，调用分词器处理文本输入
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        # 如果 `text` 为 None，返回特征提取器处理的结果
        if text is None:
            return inputs
        # 如果 `audio` 为 None，返回分词器处理的结果
        elif audio is None:
            return encodings
        else:
            # 将分词器的编码结果作为特征提取器结果的标签
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        # 调用分词器的批量解码方法并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to WhisperTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # 调用分词器的解码方法并返回结果
        return self.tokenizer.decode(*args, **kwargs)

    def get_prompt_ids(self, text: str, return_tensors="np"):
        # 调用分词器的获取提示符编号方法并返回结果
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)
```