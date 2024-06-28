# `.\models\speecht5\processing_speecht5.py`

```
# coding=utf-8
# 声明文件编码为 UTF-8

# 导入处理工具的混合类 ProcessorMixin
from ...processing_utils import ProcessorMixin

# 定义一个名为 SpeechT5Processor 的类，继承自 ProcessorMixin 类
class SpeechT5Processor(ProcessorMixin):
    """
    构造一个 SpeechT5Processor 类，将特征提取器和分词器封装成一个单一的处理器。

    [`SpeechT5Processor`] 提供了 [`SpeechT5FeatureExtractor`] 和 [`SpeechT5Tokenizer`] 的所有功能。查看
    [`~SpeechT5Processor.__call__`] 和 [`~SpeechT5Processor.decode`] 的文档字符串以获取更多信息。

    Args:
        feature_extractor (`SpeechT5FeatureExtractor`):
            [`SpeechT5FeatureExtractor`] 的实例。特征提取器是必需的输入。
        tokenizer (`SpeechT5Tokenizer`):
            [`SpeechT5Tokenizer`] 的实例。分词器是必需的输入。
    """

    # 类属性：特征提取器类名
    feature_extractor_class = "SpeechT5FeatureExtractor"
    # 类属性：分词器类名
    tokenizer_class = "SpeechT5Tokenizer"

    # 初始化方法，接收特征提取器和分词器作为参数
    def __init__(self, feature_extractor, tokenizer):
        # 调用父类 ProcessorMixin 的初始化方法
        super().__init__(feature_extractor, tokenizer)
    def __call__(self, *args, **kwargs):
        """
        Processes audio and text input, as well as audio and text targets.

        You can process audio by using the argument `audio`, or process audio targets by using the argument
        `audio_target`. This forwards the arguments to SpeechT5FeatureExtractor's
        [`~SpeechT5FeatureExtractor.__call__`].

        You can process text by using the argument `text`, or process text labels by using the argument `text_target`.
        This forwards the arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.__call__`].

        Valid input combinations are:

        - `text` only
        - `audio` only
        - `text_target` only
        - `audio_target` only
        - `text` and `audio_target`
        - `audio` and `audio_target`
        - `text` and `text_target`
        - `audio` and `text_target`

        Please refer to the docstring of the above two methods for more information.
        """
        # 从关键字参数中取出 `audio`，如果不存在则为 None
        audio = kwargs.pop("audio", None)
        # 从关键字参数中取出 `text`，如果不存在则为 None
        text = kwargs.pop("text", None)
        # 从关键字参数中取出 `text_target`，如果不存在则为 None
        text_target = kwargs.pop("text_target", None)
        # 从关键字参数中取出 `audio_target`，如果不存在则为 None
        audio_target = kwargs.pop("audio_target", None)
        # 从关键字参数中取出 `sampling_rate`，如果不存在则为 None
        sampling_rate = kwargs.pop("sampling_rate", None)

        # 如果同时有音频输入和文本输入，则抛出 ValueError
        if audio is not None and text is not None:
            raise ValueError(
                "Cannot process both `audio` and `text` inputs. Did you mean `audio_target` or `text_target`?"
            )
        # 如果同时有音频目标和文本目标输入，则抛出 ValueError
        if audio_target is not None and text_target is not None:
            raise ValueError(
                "Cannot process both `audio_target` and `text_target` inputs. Did you mean `audio` or `text`?"
            )
        # 如果没有指定任何输入或目标，则抛出 ValueError
        if audio is None and audio_target is None and text is None and text_target is None:
            raise ValueError(
                "You need to specify either an `audio`, `audio_target`, `text`, or `text_target` input to process."
            )

        # 根据有无音频输入来选择调用特征提取器或者分词器处理输入数据
        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        elif text is not None:
            inputs = self.tokenizer(text, **kwargs)
        else:
            inputs = None

        # 根据有无音频目标来选择调用特征提取器或者分词器处理目标数据
        if audio_target is not None:
            targets = self.feature_extractor(audio_target=audio_target, *args, sampling_rate=sampling_rate, **kwargs)
            labels = targets["input_values"]
        elif text_target is not None:
            targets = self.tokenizer(text_target, **kwargs)
            labels = targets["input_ids"]
        else:
            targets = None

        # 如果输入为空，则直接返回目标
        if inputs is None:
            return targets

        # 如果目标不为空，则将标签添加到输入中，并且根据目标的注意力掩码设置解码器的注意力掩码
        if targets is not None:
            inputs["labels"] = labels

            decoder_attention_mask = targets.get("attention_mask")
            if decoder_attention_mask is not None:
                inputs["decoder_attention_mask"] = decoder_attention_mask

        # 返回处理后的输入数据
        return inputs
    def pad(self, *args, **kwargs):
        """
        Collates the audio and text inputs, as well as their targets, into a padded batch.

        Audio inputs are padded by SpeechT5FeatureExtractor's [`~SpeechT5FeatureExtractor.pad`]. Text inputs are padded
        by SpeechT5Tokenizer's [`~SpeechT5Tokenizer.pad`].

        Valid input combinations are:

        - `input_ids` only
        - `input_values` only
        - `labels` only, either log-mel spectrograms or text tokens
        - `input_ids` and log-mel spectrogram `labels`
        - `input_values` and text `labels`

        Please refer to the docstring of the above two methods for more information.
        """
        # 从 kwargs 中取出对应的输入数据
        input_values = kwargs.pop("input_values", None)
        input_ids = kwargs.pop("input_ids", None)
        labels = kwargs.pop("labels", None)

        # 如果既有 input_values 又有 input_ids，则抛出数值错误
        if input_values is not None and input_ids is not None:
            raise ValueError("Cannot process both `input_values` and `input_ids` inputs.")
        # 如果 input_values、input_ids 和 labels 均为 None，则抛出数值错误
        if input_values is None and input_ids is None and labels is None:
            raise ValueError(
                "You need to specify either an `input_values`, `input_ids`, or `labels` input to be padded."
            )

        # 根据输入数据类型选择相应的填充方法
        if input_values is not None:
            inputs = self.feature_extractor.pad(input_values, *args, **kwargs)
        elif input_ids is not None:
            inputs = self.tokenizer.pad(input_ids, **kwargs)
        else:
            inputs = None

        # 如果存在 labels，则处理目标数据
        if labels is not None:
            # 如果 labels 包含 "input_ids" 或其第一个元素是包含 "input_ids" 的列表，则使用 tokenizer 进行填充
            if "input_ids" in labels or (isinstance(labels, list) and "input_ids" in labels[0]):
                targets = self.tokenizer.pad(labels, **kwargs)
                labels = targets["input_ids"]
            else:
                # 否则，进行特征提取器的填充，针对 log-mel spectrograms
                feature_size_hack = self.feature_extractor.feature_size
                self.feature_extractor.feature_size = self.feature_extractor.num_mel_bins
                targets = self.feature_extractor.pad(labels, *args, **kwargs)
                self.feature_extractor.feature_size = feature_size_hack
                labels = targets["input_values"]
        else:
            targets = None

        # 如果 inputs 为 None，则直接返回 targets
        if inputs is None:
            return targets

        # 如果 targets 存在，则将 labels 添加到 inputs 中，并处理 decoder_attention_mask
        if targets is not None:
            inputs["labels"] = labels

            decoder_attention_mask = targets.get("attention_mask")
            if decoder_attention_mask is not None:
                inputs["decoder_attention_mask"] = decoder_attention_mask

        # 返回处理后的 inputs
        return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        # 将所有参数传递给 tokenizer 的 batch_decode 方法，并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)
    def decode(self, *args, **kwargs):
        """
        这个方法将所有参数转发给 SpeechT5Tokenizer 的 [`~SpeechT5Tokenizer.decode`] 方法。
        请参考该方法的文档字符串以获取更多信息。
        """
        # 调用 tokenizer 对象的 decode 方法，并返回其结果
        return self.tokenizer.decode(*args, **kwargs)
```