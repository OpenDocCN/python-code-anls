# `.\transformers\models\speecht5\processing_speecht5.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可协议，此处使用 Apache License 2.0
"""SpeechT5 的语音处理器类。"""

# 导入 ProcessorMixin 类，用于构建 SpeechT5Processor 类
from ...processing_utils import ProcessorMixin

# 定义 SpeechT5Processor 类，继承自 ProcessorMixin 类
class SpeechT5Processor(ProcessorMixin):
    r"""
    构建一个 SpeechT5 处理器，将特征提取器和分词器封装到一个单一的处理器中。

    [`SpeechT5Processor`] 提供了 [`SpeechT5FeatureExtractor`] 和 [`SpeechT5Tokenizer`] 的所有功能。查看 [`~SpeechT5Processor.__call__`] 和 [`~SpeechT5Processor.decode`] 的文档字符串以获取更多信息。

    Args:
        feature_extractor (`SpeechT5FeatureExtractor`):
            [`SpeechT5FeatureExtractor`] 的一个实例。特征提取器是必需的输入。
        tokenizer (`SpeechT5Tokenizer`):
            [`SpeechT5Tokenizer`] 的一个实例。分词器是必需的输入。
    """

    # 特征提取器类的名称
    feature_extractor_class = "SpeechT5FeatureExtractor"
    # 分词器类的名称
    tokenizer_class = "SpeechT5Tokenizer"

    # 构造函数，初始化 SpeechT5Processor 实例
    def __init__(self, feature_extractor, tokenizer):
        # 调用父类 ProcessorMixin 的构造函数
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
        # 从参数中提取音频输入、文本输入、文本目标和音频目标
        audio = kwargs.pop("audio", None)
        text = kwargs.pop("text", None)
        text_target = kwargs.pop("text_target", None)
        audio_target = kwargs.pop("audio_target", None)
        sampling_rate = kwargs.pop("sampling_rate", None)

        # 检查输入组合是否有效，并引发相应的异常
        if audio is not None and text is not None:
            raise ValueError(
                "Cannot process both `audio` and `text` inputs. Did you mean `audio_target` or `text_target`?"
            )
        if audio_target is not None and text_target is not None:
            raise ValueError(
                "Cannot process both `audio_target` and `text_target` inputs. Did you mean `audio` or `text`?"
            )
        if audio is None and audio_target is None and text is None and text_target is None:
            raise ValueError(
                "You need to specify either an `audio`, `audio_target`, `text`, or `text_target` input to process."
            )

        # 根据不同输入情况调用相应的特征提取器或标记器
        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        elif text is not None:
            inputs = self.tokenizer(text, **kwargs)
        else:
            inputs = None

        # 根据不同目标情况调用相应的特征提取器或标记器，并提取标签
        if audio_target is not None:
            targets = self.feature_extractor(audio_target=audio_target, *args, sampling_rate=sampling_rate, **kwargs)
            labels = targets["input_values"]
        elif text_target is not None:
            targets = self.tokenizer(text_target, **kwargs)
            labels = targets["input_ids"]
        else:
            targets = None

        # 如果输入为空，则返回目标
        if inputs is None:
            return targets

        # 如果目标不为空，则将标签添加到输入中，并检查是否有解码器注意力掩码
        if targets is not None:
            inputs["labels"] = labels

            decoder_attention_mask = targets.get("attention_mask")
            if decoder_attention_mask is not None:
                inputs["decoder_attention_mask"] = decoder_attention_mask

        # 返回输入
        return inputs
    def pad(self, *args, **kwargs):
        """
        将音频和文本输入以及它们的目标组合成一个填充的批次。

        音频输入由SpeechT5FeatureExtractor的[`~SpeechT5FeatureExtractor.pad`]进行填充。文本输入由SpeechT5Tokenizer的[`~SpeechT5Tokenizer.pad`]进行填充。

        有效的输入组合包括：

        - 仅 `input_ids`
        - 仅 `input_values`
        - 仅 `labels`，可以是对数梅尔频谱图或文本标记
        - `input_ids` 和对数梅尔频谱图 `labels`
        - `input_values` 和文本 `labels`

        更多信息请参考上述两种方法的文档字符串。
        """
        # 从kwargs中获取input_values、input_ids和labels
        input_values = kwargs.pop("input_values", None)
        input_ids = kwargs.pop("input_ids", None)
        labels = kwargs.pop("labels", None)

        # 如果input_values和input_ids同时存在，则引发ValueError
        if input_values is not None and input_ids is not None:
            raise ValueError("Cannot process both `input_values` and `input_ids` inputs.")
        # 如果input_values、input_ids和labels均为None，则引发ValueError
        if input_values is None and input_ids is None and labels is None:
            raise ValueError(
                "You need to specify either an `input_values`, `input_ids`, or `labels` input to be padded."
            )

        # 如果存在input_values，则使用feature_extractor的pad方法进行填充
        if input_values is not None:
            inputs = self.feature_extractor.pad(input_values, *args, **kwargs)
        # 如果存在input_ids，则使用tokenizer的pad方法进行填充
        elif input_ids is not None:
            inputs = self.tokenizer.pad(input_ids, **kwargs)
        else:
            inputs = None

        # 如果存在labels
        if labels is not None:
            # 如果labels中包含"input_ids"或者labels是一个列表且labels[0]中包含"input_ids"
            if "input_ids" in labels or (isinstance(labels, list) and "input_ids" in labels[0]):
                # 使用tokenizer的pad方法进行填充，并获取标签中的input_ids
                targets = self.tokenizer.pad(labels, **kwargs)
                labels = targets["input_ids"]
            else:
                # 获取feature_extractor的feature_size，并将其设为num_mel_bins
                feature_size_hack = self.feature_extractor.feature_size
                self.feature_extractor.feature_size = self.feature_extractor.num_mel_bins
                # 使用feature_extractor的pad方法进行填充，并获取标签中的input_values
                targets = self.feature_extractor.pad(labels, *args, **kwargs)
                # 还原feature_extractor的feature_size
                self.feature_extractor.feature_size = feature_size_hack
                labels = targets["input_values"]
        else:
            targets = None

        # 如果inputs为None，则返回targets
        if inputs is None:
            return targets

        # 如果targets不为None
        if targets is not None:
            # 将标签添加到inputs中
            inputs["labels"] = labels

            # 获取targets中的decoder_attention_mask
            decoder_attention_mask = targets.get("attention_mask")
            # 如果decoder_attention_mask不为None，则将其添加到inputs中
            if decoder_attention_mask is not None:
                inputs["decoder_attention_mask"] = decoder_attention_mask

        # 返回inputs
        return inputs

    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发给SpeechT5Tokenizer的[`~SpeechT5Tokenizer.batch_decode`]。请参考此方法的文档字符串以获取更多信息。
        """
        # 调用tokenizer的batch_decode方法，并返回结果
        return self.tokenizer.batch_decode(*args, **kwargs)
    # 定义一个方法decode，接受任意位置参数和关键字参数
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to SpeechT5Tokenizer's [`~SpeechT5Tokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        # 调用SpeechT5Tokenizer的decode方法，并将所有参数转发给它
        return self.tokenizer.decode(*args, **kwargs)
```