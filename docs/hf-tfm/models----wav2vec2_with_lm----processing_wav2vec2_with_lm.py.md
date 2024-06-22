# `.\transformers\models\wav2vec2_with_lm\processing_wav2vec2_with_lm.py`

```py
# 设置代码文件编码格式为 UTF-8
# 版权声明
# 2021年 HuggingFace Inc. 团队
#
# 根据 Apache 许可证 2.0 版 ("许可证")授权；
# 除非符合许可证的规定，否则您不得使用此文件
# 您可以在以下位置获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律另有规定或以书面形式同意，否则不得
# 基于"按原样"的方式分发软件
# 无论是显式的还是暗示的保证或条件都不包括在内
# 有关详细的语言，请参见许可证

"""
用于 Wav2Vec2 的语音处理器类
"""
# 导入模块
import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import Pool, get_context, get_start_method
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

import numpy as np

from ...processing_utils import ProcessorMixin
from ...utils import ModelOutput, logging, requires_backends

# 获取日志记录器
logger = logging.get_logger(__name__)

# 类型检查
if TYPE_CHECKING:
    from pyctcdecode import BeamSearchDecoderCTC
    from ...feature_extraction_utils import FeatureExtractionMixin
    from ...tokenization_utils import PreTrainedTokenizerBase

# 定义一个字典列表的别名
ListOfDict = List[Dict[str, Union[int, str]]]

# 定义一个数据类
@dataclass
class Wav2Vec2DecoderWithLMOutput(ModelOutput):
    """
    [`Wav2Vec2DecoderWithLM`] 的输出类型，包括转录内容。

    Args:
        text (list of `str` or `str`):
            文本中的解码对数。通常是语音转录文本。
        logit_score (list of `float` or `float`):
            与产生的文本相关的束的总对数分数。
        lm_score (list of `float`):
            与产生的文本相关的融合 lm_score。
        word_offsets (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]`):
            解码单词的偏移量。结合采样率和模型降采样率，单词偏移量可用于计算每个单词的时间戳。
    """

    text: Union[List[List[str]], List[str], str]
    logit_score: Union[List[List[float]], List[float], float] = None
    lm_score: Union[List[List[float]], List[float], float] = None
    word_offsets: Union[List[List[ListOfDict]], List[ListOfDict], ListOfDict] = None


class Wav2Vec2ProcessorWithLM(ProcessorMixin):
    r"""
    构建一个 Wav2Vec2 处理器，将 Wav2Vec2 特征提取器、Wav2Vec2 CTC 分词器和具有语言模型支持的解码器
    封装到一个统一的处理器中，用于语言模型增强的语音识别解码。
    # 引入所需的库和模块
    from pyctcdecode import BeamSearchDecoderCTC
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, PreTrainedTokenizerBase, FeatureExtractionMixin
    
    
    # 定义 Wav2Vec2CTC 类，继承 FeatureExtractionMixin 类
    class Wav2Vec2CTC(FeatureExtractionMixin):
        
        # 初始化方法
        def __init__(
            self,
            feature_extractor: "Wav2Vec2FeatureExtractor",  # 特征提取器的实例
            tokenizer: "PreTrainedTokenizerBase",  # 分词器的实例
            decoder: "BeamSearchDecoderCTC",  # 解码器的实例
        ):
            super().__init__(feature_extractor, tokenizer)  # 调用父类的初始化方法
            
            # 如果解码器不是 BeamSearchDecoderCTC 类型的实例，抛出 ValueError
            if not isinstance(decoder, BeamSearchDecoderCTC):
                raise ValueError(f"`decoder` has to be of type {BeamSearchDecoderCTC.__class__}, but is {type(decoder)}")
            
            # 检查解码器的字母表和分词器的词汇表内容是否匹配
            missing_decoder_tokens = self.get_missing_alphabet_tokens(decoder, tokenizer)
            if len(missing_decoder_tokens) > 0:
                raise ValueError(
                    f"The tokens {missing_decoder_tokens} are defined in the tokenizer's "
                    "vocabulary, but not in the decoder's alphabet. "
                    f"Make sure to include {missing_decoder_tokens} in the decoder's alphabet."
                )
            
            # 将解码器和当前的处理器设为特征提取器
            self.decoder = decoder
            self.current_processor = self.feature_extractor
            self._in_target_context_manager = False
    
        # 保存预训练模型
        def save_pretrained(self, save_directory):
            super().save_pretrained(save_directory)  # 调用父类的保存预训练模型方法
            self.decoder.save_to_dir(save_directory)  # 保存解码器的模型
    
        # 设置语言模型属性
        @classmethod
        @staticmethod
        def _set_language_model_attribute(decoder: "BeamSearchDecoderCTC", attribute: str, value: float):
            setattr(decoder.model_container[decoder._model_key], attribute, value)
    
        # 获取语言模型
        @property
        def language_model(self):
            return self.decoder.model_container[self.decoder._model_key]
        # 导入所需的模块和类
        from pyctcdecode.alphabet import BLANK_TOKEN_PTN, UNK_TOKEN, UNK_TOKEN_PTN

        # 检查decoder的字母表中是否包含tokenizer的除特殊标记之外的所有标记
        # 获取tokenizer的词汇表列表
        tokenizer_vocab_list = list(tokenizer.get_vocab().keys())

        # 替换特殊标记
        for i, token in enumerate(tokenizer_vocab_list):
            if BLANK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = ""
            if token == tokenizer.word_delimiter_token:
                tokenizer_vocab_list[i] = " "
            if UNK_TOKEN_PTN.match(token):
                tokenizer_vocab_list[i] = UNK_TOKEN

        # 找到decoder字母表中缺失的标记
        missing_tokens = set(tokenizer_vocab_list) - set(decoder._alphabet.labels)

        return missing_tokens

    def __call__(self, *args, **kwargs):
        """
        当以普通模式使用时，将所有参数转发给Wav2Vec2FeatureExtractor的
        [`~Wav2Vec2FeatureExtractor.__call__`]并返回其输出。如果在上下文中使用
        [`~Wav2Vec2ProcessorWithLM.as_target_processor`]，则将所有参数转发给
        Wav2Vec2CTCTokenizer的[`~Wav2Vec2CTCTokenizer.__call__`]。更多信息请参考上述两个
        方法的文档字符串。
        """
        # 为了向后兼容性
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        # 检查是否使用了“raw_speech”作为关键字参数，如果是，将其替换为“audio”
        if "raw_speech" in kwargs:
            warnings.warn("Using `raw_speech` as a keyword argument is deprecated. Use `audio` instead.")
            audio = kwargs.pop("raw_speech")
        else:
            audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", None)
        text = kwargs.pop("text", None)
        
        # 检查是否有参数传入，如果有，将audio设为第一个参数，并将args中去掉该参数
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        # 检查是否同时传入了audio和text参数
        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        # 根据参数情况进行处理并返回相应结果
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
    # 定义一个名为pad的方法，接受任意数量的位置参数和关键字参数
    def pad(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor's
        [`~Wav2Vec2FeatureExtractor.pad`] and returns its output. If used in the context
        [`~Wav2Vec2ProcessorWithLM.as_target_processor`] this method forwards all its arguments to
        Wav2Vec2CTCTokenizer's [`~Wav2Vec2CTCTokenizer.pad`]. Please refer to the docstring of the above two methods
        for more information.
        """
        # 为了向后兼容
        if self._in_target_context_manager:
            return self.current_processor.pad(*args, **kwargs)

        # 从kwargs中弹出"input_features"参数，如果不存在则为None
        input_features = kwargs.pop("input_features", None)
        # 从kwargs中弹出"labels"参数，如果不存在则为None
        labels = kwargs.pop("labels", None)
        # 如果args的长度大于0，则将args的第一个元素赋值给input_features，并截取args的剩余部分
        if len(args) > 0:
            input_features = args[0]
            args = args[1:]

        # 如果input_features不为None，则使用feature_extractor对其进行填充，使用剩余的args和kwargs
        if input_features is not None:
            input_features = self.feature_extractor.pad(input_features, *args, **kwargs)
        # 如果labels不为None，则使用tokenizer对其进行填充，使用kwargs
        if labels is not None:
            labels = self.tokenizer.pad(labels, **kwargs)

        # 如果labels为None，则返回input_features
        if labels is None:
            return input_features
        # 如果input_features为None，则返回labels
        elif input_features is None:
            return labels
        # 否则将labels的"input_ids"赋值给input_features的"labels"，然后返回input_features
        else:
            input_features["labels"] = labels["input_ids"]
            return input_features

    # 定义一个名为batch_decode的方法，接受一个名为logits的numpy数组和一系列可选参数
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
    # 定义一个名为decode的方法，接受一个名为logits的numpy数组和一系列可选参数
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
    @contextmanager
    # 将当前处理器设置为处理目标的处理器。在微调Wav2Vec2时用于编码标签。
    def as_target_processor(self):
        """
        Temporarily sets the processor for processing the target. Useful for encoding the labels when fine-tuning
        Wav2Vec2.
        """
        # 发出警告，`as_target_processor`已被弃用，将在Transformers的v5中移除。可以通过在正常`__call__`方法的`text`参数中处理标签来处理标签（可以在与音频输入相同的调用中，也可以在单独的调用中）。
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        # 设置在目标上下文管理器中
        self._in_target_context_manager = True
        # 将当前处理器设置为分词器
        self.current_processor = self.tokenizer
        # 生成器中的代码块
        yield
        # 将当前处理器设置为特征提取器
        self.current_processor = self.feature_extractor
        # 结束目标上下文管理器
        self._in_target_context_manager = False
```