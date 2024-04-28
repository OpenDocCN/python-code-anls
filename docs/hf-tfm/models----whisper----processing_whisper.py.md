# `.\transformers\models\whisper\processing_whisper.py`

```py
# 设置文件编码为 utf-8

# 版权声明

# 导入处理工具类的ProcessorMixin

# 定义WhisperProcessor类，继承ProcessorMixin

    # 构造WhisperProcessor类，将Whisper特征提取器和Whisper标记器封装成单个处理器

    # ['WhisperProcessor'] 提供了['WhisperFeatureExtractor']和['WhisperTokenizer']的所有功能。详细信息请参阅['~WhisperProcessor.__call__']和['~WhisperProcessor.decode']

    # 初始化函数
    # 参数
        # feature_extractor（'WhisperFeatureExtractor'）：WhisperFeatureExtractor的实例。特征提取器为必需输入。
        # tokenizer（'WhisperTokenizer'）：WhisperTokenizer的实例。标记器为必需输入。

    # 设置feature_extractor_class为"WhisperFeatureExtractor"
    # 设置tokenizer_class为"WhisperTokenizer"

    # 初始化函数
    # 参数
        # feature_extractor：特征提取器
        # tokenizer：标记器
    # 调用父类的初始化函数，传入feature_extractor和tokenizer

    # 设置当前处理器为feature_extractor
    # 设置_in_target_context_manager为False

    # 获取解码器提示的ID
    # 参数
        # task：任务
        # language：语言
        # no_timestamps：是否包含时间戳
        # 调用tokenizer的get_decoder_prompt_ids方法，传入对应参数
    # 定义一个类方法，将`audio`参数传递给WhisperFeatureExtractor的`__call__`方法，将`text`参数传递给WhisperTokenizer的`__call__`方法。请参考上述两个方法的文档字符串以获取更多信息。
    def __call__(self, *args, **kwargs):
        # 为了向后兼容
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        # 从kwargs中获取`audio`、`sampling_rate`和`text`参数
        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        
        # 如果args中有参数，将第一个参数视为audio，其余参数视为args
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        # 如果既没有audio也没有text输入，则抛出异常
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        # 如果有audio输入，则使用feature_extractor处理
        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        # 如果有text输入，则使用tokenizer处理
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        # 如果只有text输入，则返回inputs
        if text is None:
            return inputs
        # 如果只有audio输入，则返回encodings
        elif audio is None:
            return encodings
        else:
            # 将encodings中的"input_ids"作为labels添加到inputs中，然后返回inputs
            inputs["labels"] = encodings["input_ids"]
            return inputs

    # 定义一个方法，将其参数全部转发给WhisperTokenizer的`batch_decode`方法。请参考该方法的文档字符串以获取更多信息。
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 定义一个方法，将其参数全部转发给WhisperTokenizer的`decode`方法。请参考该方法的文档字符串以获取更多信息。
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    # 定义一个方法，将文本输入转换为prompt_ids。返回结果可以是numpy数组或torch张量，取决于return_tensors参数。
    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)
```