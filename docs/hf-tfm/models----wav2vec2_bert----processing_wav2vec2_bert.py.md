# `.\models\wav2vec2_bert\processing_wav2vec2_bert.py`

```py
# 设置编码格式为 UTF-8
# 版权声明：2024 年 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“现状”提供软件，
# 没有任何明示或暗示的担保或条件。请参阅许可证以了解详细信息。
"""
Wav2Vec2-BERT 的语音处理器类
"""
# 引入警告模块
import warnings

# 导入处理工具函数
from ...processing_utils import ProcessorMixin
# 导入特征提取模块
from ..seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
# 导入 Wav2Vec2 CTC 分词器
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer


class Wav2Vec2BertProcessor(ProcessorMixin):
    r"""
    构建一个 Wav2Vec2-BERT 处理器，将 Wav2Vec2-BERT 特征提取器和 Wav2Vec2 CTC 分词器封装为单个处理器。

    [`Wav2Vec2Processor`] 提供了 [`SeamlessM4TFeatureExtractor`] 和 [`PreTrainedTokenizer`] 的所有功能。
    有关更多信息，请参阅 [`~Wav2Vec2Processor.__call__`] 和 [`~Wav2Vec2Processor.decode`] 的文档字符串。

    Args:
        feature_extractor (`SeamlessM4TFeatureExtractor`):
            [`SeamlessM4TFeatureExtractor`] 的实例。特征提取器是必需的输入。
        tokenizer ([`PreTrainedTokenizer`]):
            [`PreTrainedTokenizer`] 的实例。分词器是必需的输入。
    """

    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            # 尝试从预训练模型加载
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except OSError:
            # 若加载失败，则发出警告
            warnings.warn(
                f"Loading a tokenizer inside {cls.__name__} from a config that does not"
                " include a `tokenizer_class` attribute is deprecated and will be "
                "removed in v5. Please add `'tokenizer_class': 'Wav2Vec2CTCTokenizer'`"
                " attribute to either your `config.json` or `tokenizer_config.json` "
                "file to suppress this warning: ",
                FutureWarning,
            )

            # 从预训练模型加载特征提取器和分词器
            feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

            # 返回处理器实例
            return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)
    def __call__(self, audio=None, text=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and audio(s). This method forwards the `audio`
        and `kwargs` arguments to SeamlessM4TFeatureExtractor's [`~SeamlessM4TFeatureExtractor.__call__`] if `audio` is not
        `None` to pre-process the audio. To prepare the target sequences(s), this method forwards the `text` and `kwargs` arguments to
        PreTrainedTokenizer's [`~PreTrainedTokenizer.__call__`] if `text` is not `None`. Please refer to the docstring of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as a list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            audio (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case
                of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels,
                and T the sample length of the audio.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to the feature extractor and/or the
                tokenizer.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_features** -- Audio input features to be fed to a model. Returned when `audio` is not `None`.
            - **attention_mask** -- List of indices specifying which timestamps should be attended to by the model when `audio` is not `None`.
              When only `text` is specified, returns the token attention mask.
            - **labels** -- List of token ids to be fed to a model. Returned when both `text` and `audio` are not `None`.
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None` and `audio` is `None`.
        """

        # Pop the 'sampling_rate' from kwargs, if present
        sampling_rate = kwargs.pop("sampling_rate", None)

        # Raise an error if both audio and text inputs are None
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        # If audio input is provided, call feature_extractor to preprocess audio
        if audio is not None:
            inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, **kwargs)

        # If text input is provided, call tokenizer to encode the text
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        # If only text input is provided, return the processed inputs
        if text is None:
            return inputs
        # If only audio input is provided, return the encoded text
        elif audio is None:
            return encodings
        # If both audio and text inputs are provided, merge inputs and encodings, and return
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs
    # 如果 `input_features` 不为 `None`，则将 `input_features` 和 `kwargs` 参数传递给 `SeamlessM4TFeatureExtractor` 的 `pad` 方法进行填充。
    # 如果 `labels` 不为 `None`，则将 `labels` 和 `kwargs` 参数传递给 `PreTrainedTokenizer` 的 `pad` 方法进行填充。
    # 更多信息请参考上述两个方法的文档字符串。
    def pad(self, input_features=None, labels=None, **kwargs):
        if input_features is None and labels is None:
            raise ValueError("You need to specify either an `input_features` or `labels` input to pad.")
        
        # 如果 `input_features` 不为 `None`，调用 `feature_extractor` 的 `pad` 方法进行填充
        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, **kwargs)
        
        # 如果 `labels` 不为 `None`，调用 `tokenizer` 的 `pad` 方法进行填充
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)
        
        # 如果 `labels` 为 `None`，返回 `input_features`
        if labels is None:
            return input_features
        # 如果 `input_features` 为 `None`，返回 `labels`
        elif input_features is None:
            return labels
        else:
            # 将 `labels` 的 `input_ids` 赋值给 `input_features` 的 `"labels"` 键
            input_features["labels"] = labels["input_ids"]
            return input_features

    # 将所有参数转发给 `PreTrainedTokenizer` 的 `batch_decode` 方法
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # 将所有参数转发给 `PreTrainedTokenizer` 的 `decode` 方法
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
```