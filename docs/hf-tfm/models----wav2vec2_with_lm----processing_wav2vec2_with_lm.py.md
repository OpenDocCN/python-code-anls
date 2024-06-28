# `.\models\wav2vec2_with_lm\processing_wav2vec2_with_lm.py`

```py
# coding=utf-8
# 定义了代码文件的编码格式为 UTF-8

# 版权声明，指出该代码由 HuggingFace Inc. 团队编写
# 根据 Apache 许可证 2.0 版本发布
# 您可以在符合许可证条件下使用此文件，详见许可证文档
"""
Speech processor class for Wav2Vec2
"""
# 导入必要的库和模块
import os  # 导入操作系统功能模块
import warnings  # 导入警告处理模块
from contextlib import contextmanager, nullcontext  # 导入上下文管理器和空上下文
from dataclasses import dataclass  # 导入用于定义数据类的装饰器
from multiprocessing import Pool, get_context, get_start_method  # 导入多进程相关模块
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union  # 导入类型提示相关模块

import numpy as np  # 导入 NumPy 库

# 导入相关的自定义模块和类
from ...processing_utils import ProcessorMixin
from ...utils import ModelOutput, logging, requires_backends  # 导入模型输出、日志和后端要求

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果是类型检查阶段
if TYPE_CHECKING:
    from pyctcdecode import BeamSearchDecoderCTC  # 导入 BeamSearchDecoderCTC 类

    from ...feature_extraction_utils import FeatureExtractionMixin  # 导入特征提取混合类
    from ...tokenization_utils import PreTrainedTokenizerBase  # 导入预训练分词器基类

# 定义一个列表字典类型的别名
ListOfDict = List[Dict[str, Union[int, str]]]


@dataclass
class Wav2Vec2DecoderWithLMOutput(ModelOutput):
    """
    Output type of [`Wav2Vec2DecoderWithLM`], with transcription.

    Args:
        text (list of `str` or `str`):
            Decoded logits in text from. Usually the speech transcription.
        logit_score (list of `float` or `float`):
            Total logit score of the beams associated with produced text.
        lm_score (list of `float`):
            Fused lm_score of the beams associated with produced text.
        word_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`):
            Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets
            can be used to compute time stamps for each word.
    """

    text: Union[List[List[str]], List[str], str]  # 文本结果，可以是列表的列表、列表或字符串形式
    logit_score: Union[List[List[float]], List[float], float] = None  # 对数得分，可以是列表的列表、列表或浮点数形式，默认为 None
    lm_score: Union[List[List[float]], List[float], float] = None  # 语言模型得分，可以是列表的列表、列表或浮点数形式，默认为 None
    word_offsets: Union[List[List[ListOfDict]], List[ListOfDict], ListOfDict] = None  # 单词偏移量，可以是列表的列表的列表、列表的列表或列表字典形式，默认为 None


class Wav2Vec2ProcessorWithLM(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a decoder
    with language model support into a single processor for language model boosted speech recognition decoding.
    """
    Args:
        feature_extractor ([`Wav2Vec2FeatureExtractor`]):
            An instance of [`Wav2Vec2FeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`Wav2Vec2CTCTokenizer`]):
            An instance of [`Wav2Vec2CTCTokenizer`]. The tokenizer is a required input.
        decoder (`pyctcdecode.BeamSearchDecoderCTC`):
            An instance of [`pyctcdecode.BeamSearchDecoderCTC`]. The decoder is a required input.
    """
    
    # 定义字符串常量，表示特征提取器和分词器的类名
    feature_extractor_class = "Wav2Vec2FeatureExtractor"
    tokenizer_class = "Wav2Vec2CTCTokenizer"

    def __init__(
        self,
        feature_extractor: "FeatureExtractionMixin",
        tokenizer: "PreTrainedTokenizerBase",
        decoder: "BeamSearchDecoderCTC",
    ):
        from pyctcdecode import BeamSearchDecoderCTC
        
        # 调用父类的初始化方法，传入特征提取器和分词器实例
        super().__init__(feature_extractor, tokenizer)
        
        # 检查解码器是否为正确的类型，若不是则抛出异常
        if not isinstance(decoder, BeamSearchDecoderCTC):
            raise ValueError(f"`decoder` has to be of type {BeamSearchDecoderCTC.__class__}, but is {type(decoder)}")

        # 确保解码器的字母表与分词器的词汇表内容匹配
        missing_decoder_tokens = self.get_missing_alphabet_tokens(decoder, tokenizer)
        if len(missing_decoder_tokens) > 0:
            raise ValueError(
                f"The tokens {missing_decoder_tokens} are defined in the tokenizer's "
                "vocabulary, but not in the decoder's alphabet. "
                f"Make sure to include {missing_decoder_tokens} in the decoder's alphabet."
            )

        # 将解码器、当前处理器和目标上下文管理器的初始状态设置为属性
        self.decoder = decoder
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    # 保存预训练模型至指定目录
    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)  # 调用父类方法保存预训练模型
        self.decoder.save_to_dir(save_directory)  # 调用解码器的保存方法保存至指定目录

    # 设置语言模型属性的静态方法，用于设置解码器的模型属性
    @classmethod
    @staticmethod
    def _set_language_model_attribute(decoder: "BeamSearchDecoderCTC", attribute: str, value: float):
        setattr(decoder.model_container[decoder._model_key], attribute, value)

    # 返回解码器的语言模型属性作为属性方法
    @property
    def language_model(self):
        return self.decoder.model_container[self.decoder._model_key]

    @staticmethod
    def get_missing_alphabet_tokens(decoder, tokenizer):
        from pyctcdecode.alphabet import BLANK_TOKEN_PTN, UNK_TOKEN, UNK_TOKEN_PTN

        # 确保解码器的字母表中包含所有除特殊标记外的标记，检索缺失的字母表标记
        tokenizer_vocab_list = list(tokenizer.get_vocab().keys())

        # 替换特殊标记
        for i, token in enumerate(tokenizer_vocab_list):
            if BLANK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = ""
            if token == tokenizer.word_delimiter_token:
                tokenizer_vocab_list[i] = " "
            if UNK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = UNK_TOKEN

        # 检查哪些额外标记不是特殊的标记
        missing_tokens = set(tokenizer_vocab_list) - set(decoder._alphabet.labels)

        return missing_tokens

    def __call__(self, *args, **kwargs):
        """
        在普通模式下使用时，该方法将所有参数转发到Wav2Vec2FeatureExtractor的[`~Wav2Vec2FeatureExtractor.__call__`]，并返回其输出。
        如果在上下文[`~Wav2Vec2ProcessorWithLM.as_target_processor`]中使用，则将所有参数转发到Wav2Vec2CTCTokenizer的[`~Wav2Vec2CTCTokenizer.__call__`]。
        有关更多信息，请参阅上述两个方法的文档字符串。
        """
        # 为了向后兼容性
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
    # 定义一个方法 `pad`，用于数据填充
    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.pad`] and returns its output. If used in the context
        [`~Wav2Vec2ProcessorWithLM.as_target_processor`] this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's [`~Wav2Vec2CTCTokenizer.pad`]. Please refer to the docstring of the above two methods
        for more information.
        """
        # 如果在目标处理器的上下文中使用，则调用当前处理器的 `pad` 方法
        if self._in_target_context_manager:
            return self.current_processor.pad(*args, **kwargs)

        # 从 `kwargs` 中弹出 `input_features` 和 `labels` 参数
        input_features = kwargs.pop("input_features", None)
        labels = kwargs.pop("labels", None)

        # 如果有额外的位置参数，将第一个位置参数作为 `input_features`，其余作为 `args`
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]

        # 如果 `input_features` 不为 `None`，使用特征提取器的 `pad` 方法进行填充
        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        # 如果 `labels` 不为 `None`，使用标记器的 `pad` 方法进行填充
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        # 根据是否有 `labels` 和 `input_features` 返回不同的结果
        if labels is None:
            return input_features
        elif input_features is None:
            return labels
        else:
            # 如果两者都有，将 `labels` 的 `input_ids` 添加到 `input_features` 的 `"labels"` 键中
            input_features["labels"] = labels["input_ids"]
            return input_features

    # 定义一个方法 `batch_decode`，用于批量解码 logits
    def batch_decode(
        self,
        logits: np.ndarray,
        pool: Optional[Pool] = None,
        num_processes: Optional[int] = None,
        beam_width: Optional[int] = None,
        beam_prune_logp: Optional[float] = None,
        token_min_logp: Optional[float] = None,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        unk_score_offset: Optional[float] = None,
        lm_score_boundary: Optional[bool] = None,
        output_word_offsets: bool = False,
        n_best: int = 1,
    ):
        # 方法用于批量解码 logits 并返回结果
        pass

    # 定义一个方法 `decode`，用于解码 logits
    def decode(
        self,
        logits: np.ndarray,
        beam_width: Optional[int] = None,
        beam_prune_logp: Optional[float] = None,
        token_min_logp: Optional[float] = None,
        hotwords: Optional[Iterable[str]] = None,
        hotword_weight: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        unk_score_offset: Optional[float] = None,
        lm_score_boundary: Optional[bool] = None,
        output_word_offsets: bool = False,
        n_best: int = 1,
    ):
        # 方法用于解码 logits 并返回结果
        pass

    @contextmanager
    # 定义一个方法 `as_target_processor`，用于临时设置处理目标的处理器。在微调 Wav2Vec2 模型时，用于对标签进行编码。
    def as_target_processor(self):
        """
        Temporarily sets the processor for processing the target. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        """
        # 发出警告信息，提醒用户 `as_target_processor` 方法将在 Transformers v5 中移除，建议使用 `__call__` 方法的 `text` 参数处理标签。
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        # 设置目标处理上下文管理器为真
        self._in_target_context_manager = True
        # 将当前处理器设置为分词器 tokenizer
        self.current_processor = self.tokenizer
        # 返回一个生成器，用于临时设置目标处理器
        yield
        # 在生成器中，将当前处理器设置为特征提取器 feature_extractor
        self.current_processor = self.feature_extractor
        # 设置目标处理上下文管理器为假，表示处理结束
        self._in_target_context_manager = False
```