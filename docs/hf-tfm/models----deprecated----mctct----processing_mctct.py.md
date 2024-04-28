# `.\models\deprecated\mctct\processing_mctct.py`

```
# 设置字符编码为 UTF-8
# 版权声明
# 版权所有2022年HuggingFace Inc.团队。保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；您除非遵守许可证，否则不得使用此文件。
# 您可以在以下位置获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不对License下分布的软件进行分发，以“原样”为基础，
# 没有任何形式的担保或条件，无论是明示的还是暗示的。
# 对特定语言的特定条件下权限和限制，请查看许可证。
"""
M-CTC-T的语音处理器类
"""
# 引入警告库
import warnings
# 引入上下文管理器
from contextlib import contextmanager
# 从processing_utils中引入ProcessorMixin
from ....processing_utils import ProcessorMixin

# 创建MCTCTProcessor类，它将MCTCT特征提取器和MCTCT标记器封装为一个处理器
class MCTCTProcessor(ProcessorMixin):
    r"""
    构造一个MCTCT处理器，将MCTCT特征提取器和MCTCT标记器包装成一个单独的处理器。

    [`MCTCTProcessor`]提供了[`MCTCTFeatureExtractor`]和[`AutoTokenizer`]的所有功能。查看
    [`~MCTCTProcessor.__call__`]和[`~MCTCTProcessor.decode`]以获取更多信息。

    参数:
        feature_extractor (`MCTCTFeatureExtractor`):
            [`MCTCTFeatureExtractor`]的一个实例。特征提取器是一个必需的输入。
        tokenizer (`AutoTokenizer`):
            [`AutoTokenizer`]的一个实例。标记器是一个必需的输入。
    """

    feature_extractor_class = "MCTCTFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    # 初始化方法，接受特征提取器和标记器作为输入
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        # 将当前处理器设置为特征提取器
        self.current_processor = self.feature_extractor
        # 标记当前上下文管理器的状态为假
        self._in_target_context_manager = False
    def __call__(self, *args, **kwargs):
        """
        当在常规模式下使用时，此方法将所有参数转发到MCTCTFeatureExtractor的[`~MCTCTFeatureExtractor.__call__`]，并返回其输出。
        如果在上下文[`~MCTCTProcessor.as_target_processor`]中使用，此方法将所有参数转发到AutoTokenizer的[`~AutoTokenizer.__call__`]。
        有关更多信息，请参阅上述两种方法的文档字符串。
        """
        # 为了向后兼容
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        if "raw_speech" in kwargs:
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")
        else:
            audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        此方法将所有参数转发到AutoTokenizer的[`~PreTrainedTokenizer.batch_decode`]。请参阅此方法的文档字符串以获取更多信息。
        """
        return self.tokenizer.batch_decode(*args, **kwargs)
    def pad(self, *args, **kwargs):
        """
        当在正常模式下使用时，该方法将所有参数转发到MCTCTFeatureExtractor的[`~MCTCTFeatureExtractor.pad`]并返回其输出。
        如果在[`~MCTCTProcessor.as_target_processor`]上下文中使用该方法，它将所有参数转发到PreTrainedTokenizer的[`~PreTrainedTokenizer.pad`]。
        请参考上述两个方法的文档字符串获取更多信息。
        """
        # 为了向后兼容
        if self._in_target_context_manager:
            return self.current_processor.pad(*args, **kwargs)

        # 获取额外的输入特征和标签
        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]

        # 如果有输入特征，则对输入特征进行填充
        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        # 如果有标签，则对标签进行填充
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        # 如果标签为空，则返回输入特征；如果输入特征为空，则返回标签；否则将标签添加到输入特征中
        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    def decode(self, *args, **kwargs):
        """
        该方法将所有参数转发到AutoTokenizer的[`~PreTrainedTokenizer.decode`]。请参考该方法的文档字符串获取更多信息。
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        临时设置用于处理输入的分词器。在微调MCTCT时编码标签时很有用。
        """
        warnings.warn(
            "`as_target_processor`已弃用，将在Transformers的v5中移除。您可以通过使用常规`__call__`方法的参数`text`来处理标签
            (无论是在与音频输入相同的调用中还是在单独的调用中)。
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
```