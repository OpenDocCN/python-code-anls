# `.\transformers\models\musicgen\processing_musicgen.py`

```py
# 定义编码音乐的处理器类，继承了ProcessorMixin，其中包含了一些处理音乐数据的方法
class MusicgenProcessor(ProcessorMixin):
    """
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
    # 设置特征提取器类名为 "EncodecFeatureExtractor"
    feature_extractor_class = "EncodecFeatureExtractor"
    # 设置标记器类名为 ("T5Tokenizer", "T5TokenizerFast")
    tokenizer_class = ("T5Tokenizer", "T5TokenizerFast")

    # 初始化方法，接受特征提取器和标记器作为参数
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        # 将当前处理器设置为特征提取器
        self.current_processor = self.feature_extractor
        # 初始化目标上下文管理器标志为 False
        self._in_target_context_manager = False

    # 获取解码器提示 ID 的方法，接受任务、语言和是否包含时间戳等参数
    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        # 调用标记器的获取解码器提示 ID 方法，并返回结果
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)
    # 将 `audio` 参数传递给 EncodecFeatureExtractor 的 [`~EncodecFeatureExtractor.__call__`] 方法，将 `text` 参数传递给 [`~T5Tokenizer.__call__`] 方法。请参考上述两个方法的文档字符串获取更多信息。
    def __call__(self, *args, **kwargs):
        # 为了向后兼容
        if self._in_target_context_manager:
            # 返回当前处理器的结果
            return self.current_processor(*args, **kwargs)

        # 从关键字参数中提取 `audio` 和 `sampling_rate`，并将其移出 kwargs
        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        # 如果存在位置参数，将第一个位置参数作为 `audio`，并将其从 args 中移出
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        # 如果 `audio` 和 `text` 均为 None，则抛出 ValueError
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        # 如果存在文本输入，则使用 tokenizer 处理文本输入
        if text is not None:
            inputs = self.tokenizer(text, **kwargs)

        # 如果存在音频输入，则使用 feature_extractor 处理音频输入
        if audio is not None:
            audio_inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)

        # 如果只有音频输入，则返回 inputs
        if audio is None:
            return inputs

        # 如果只有文本输入，则返回 audio_inputs
        elif text is None:
            return audio_inputs

        # 如果既有文本输入又有音频输入，则将音频输入的 "input_values" 添加到 inputs 中，并根据情况添加 "padding_mask"
        else:
            inputs["input_values"] = audio_inputs["input_values"]
            if "padding_mask" in audio_inputs:
                inputs["padding_mask"] = audio_inputs["padding_mask"]
            return inputs

    # 用于批量解码音频输出或 tokenizer 的 token ids 批量输出
    def batch_decode(self, *args, **kwargs):
        # 从关键字参数中提取 `audio` 和 `padding_mask`，并将其移出 kwargs
        audio_values = kwargs.pop("audio", None)
        padding_mask = kwargs.pop("padding_mask", None)

        # 如果存在位置参数，则将第一个位置参数作为 `audio_values`，并将其从 args 中移出
        if len(args) > 0:
            audio_values = args[0]
            args = args[1:]

        # 如果存在音频输出，则调用 _decode_audio 方法解码音频输出
        if audio_values is not None:
            return self._decode_audio(audio_values, padding_mask=padding_mask)
        # 否则，调用 tokenizer 的 batch_decode 方法
        else:
            return self.tokenizer.batch_decode(*args, **kwargs)

    # 将参数全部转发给 T5Tokenizer 的 [`~PreTrainedTokenizer.decode`] 方法
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
    # 解码音频数据，去除音频值中的任何填充，返回numpy音频数组的列表
    def _decode_audio(self, audio_values, padding_mask: Optional = None) -> List[np.ndarray]:
        """
        此方法从音频值中去除任何填充，返回numpy音频数组的列表。
        """
        # 将音频值转换为numpy数组
        audio_values = to_numpy(audio_values)
        bsz, channels, seq_len = audio_values.shape

        if padding_mask is None:
            return list(audio_values)

        # 将填充掩码转换为numpy数组
        padding_mask = to_numpy(padding_mask)

        # 将填充掩码的序列长度匹配到生成的音频数组，用**非填充**的标记进行填充（以确保生成的音频值不被视为填充标记）
        difference = seq_len - padding_mask.shape[-1]
        padding_value = 1 - self.feature_extractor.padding_value
        padding_mask = np.pad(padding_mask, ((0, 0), (0, difference)), "constant", constant_values=padding_value)

        # 将音频值转换为列表类型
        audio_values = audio_values.tolist()
        for i in range(bsz):
            # 通过对掩码应用条件索引，去除填充部分的音频数据
            sliced_audio = np.asarray(audio_values[i])[
                padding_mask[i][None, :] != self.feature_extractor.padding_value
            ]
            audio_values[i] = sliced_audio.reshape(channels, -1)

        return audio_values
```