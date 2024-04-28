# `.\transformers\models\clvp\processing_clvp.py`

```
# 导入ProcessorMixin类
from ...processing_utils import ProcessorMixin

# 定义ClvpProcessor类，继承自ProcessorMixin类
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

    # 定义类属性
    feature_extractor_class = "ClvpFeatureExtractor"
    tokenizer_class = "ClvpTokenizer"
    model_input_names = [
        "input_ids",
        "input_features",
        "attention_mask",
    ]

    # 初始化方法
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    # __call__方法，用于处理输入
    def __call__(self, *args, **kwargs):
        """
        Forwards the `audio` and `sampling_rate` arguments to [`~ClvpFeatureExtractor.__call__`] and the `text`
        argument to [`~ClvpTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more
        information.
        """

        # 从kwargs中弹出raw_speech、sampling_rate和text参数
        raw_speech = kwargs.pop("raw_speech", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)

        # 如果raw_speech和text都为None，则抛出错误
        if raw_speech is None and text is None:
            raise ValueError("You need to specify either an `raw_speech` or `text` input to process.")

        # 如果raw_speech不为None
        if raw_speech is not None:
            # 调用feature_extractor处理raw_speech，并传递sampling_rate和kwargs
            inputs = self.feature_extractor(raw_speech, sampling_rate=sampling_rate, **kwargs)
        
        # 如果text不为None
        if text is not None:
            # 调用tokenizer处理text，并传递kwargs
            encodings = self.tokenizer(text, **kwargs)

        # 如果text为None，则返回inputs
        if text is None:
            return inputs
        # 如果raw_speech为None，则返回encodings
        elif raw_speech is None:
            return encodings
        # 如果raw_speech和text都不为None
        else:
            # 将encodings中的input_ids和attention_mask赋值给inputs对应的键
            inputs["input_ids"] = encodings["input_ids"]
            inputs["attention_mask"] = encodings["attention_mask"]
            return inputs

    # 从transformers.models.whisper.processing_whisper.WhisperProcessor.batch_decode复制而来，将Whisper替换为Clvp
    # 这个方法将所有参数转发到ClvpTokenizer的`~PreTrainedTokenizer.batch_decode`方法。请参考此方法的文档字符串以获取更多信息。
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 从transformers.models.whisper.processing_whisper.WhisperProcessor.decode复制并替换Whisper为Clvp
    # 此方法将所有参数转发到ClvpTokenizer的`~PreTrainedTokenizer.decode`方法。请参考此方法的文档字符串以获取更多信息。
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
```