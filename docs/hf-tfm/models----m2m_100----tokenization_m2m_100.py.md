# `.\models\m2m_100\tokenization_m2m_100.py`

```py
# 版权声明和许可声明，说明代码的版权和使用条款
# 请注意，这部分代码不会执行，仅作为声明性文本存在

"""Tokenization classes for M2M100."""
# 引入所需的模块和库，包括json、os、Path、copyfile和typing等
import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

# 引入sentencepiece库，用于处理分词
import sentencepiece

# 引入日志记录模块
from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 定义句子片段的连接符，用于后续的分词处理
SPIECE_UNDERLINE = "▁"

# 定义词汇文件名的映射关系
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "spm_file": "sentencepiece.bpe.model",
    "tokenizer_config_file": "tokenizer_config.json",
}

# 预训练模型的词汇文件映射关系，包括不同模型对应的文件地址
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/m2m100_418M": "https://huggingface.co/facebook/m2m100_418M/resolve/main/vocab.json",
        "facebook/m2m100_1.2B": "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/vocab.json",
    },
    "spm_file": {
        "facebook/m2m100_418M": "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model",
        "facebook/m2m100_1.2B": "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_config_file": {
        "facebook/m2m100_418M": "https://huggingface.co/facebook/m2m100_418M/resolve/main/tokenizer_config.json",
        "facebook/m2m100_1.2B": "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/tokenizer_config.json",
    },
}

# 预训练位置嵌入的大小，对应不同模型
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/m2m100_418M": 1024,
}

# 定义Fairseq的语言代码，包括m2m100和wmt21模型的支持语言列表
# fmt: off
FAIRSEQ_LANGUAGE_CODES = {
    "m2m100": ["af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"],
    "wmt21": ['en', 'ha', 'is', 'ja', 'cs', 'ru', 'zh', 'de']
}
# fmt: on


class M2M100Tokenizer(PreTrainedTokenizer):
    """
    构造一个M2M100分词器。基于SentencePiece实现。
    """
    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        spm_file (`str`):
            Path to [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary.
        src_lang (`str`, *optional*):
            A string representing the source language.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        language_codes (`str`, *optional*, defaults to `"m2m100"`):
            What language codes to use. Should be one of `"m2m100"` or `"wmt21"`.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Examples:

    ```
    >>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    >>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    >>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", tgt_lang="ro")
    >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
    >>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
    # 使用给定的src_text和tgt_text以及tokenizer对象，生成模型输入
    model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
    # 使用生成的模型输入调用模型，返回模型输出
    outputs = model(**model_inputs)  # 应该正常工作

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    # 初始化函数，设置各种属性和参数
    def __init__(
        self,
        vocab_file,
        spm_file,
        src_lang=None,
        tgt_lang=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        language_codes="m2m100",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        num_madeup_words=8,
        **kwargs,
    ) -> None:
        # 如果未提供spm_file参数，则使用空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置语言代码和Fairseq语言代码映射，用于生成特殊标记
        self.language_codes = language_codes
        fairseq_language_code = FAIRSEQ_LANGUAGE_CODES[language_codes]
        self.lang_code_to_token = {lang_code: f"__{lang_code}__" for lang_code in fairseq_language_code}

        # 处理额外的特殊标记，确保每种语言的特殊标记都在额外特殊标记列表中
        additional_special_tokens = kwargs.pop("additional_special_tokens", [])
        for lang_code in fairseq_language_code:
            token = self.get_lang_token(lang_code)
            if token not in additional_special_tokens and lang_code not in str(token) not in self.added_tokens_encoder:
                additional_special_tokens.append(token)

        # 设置词汇文件和解码器，从词汇文件加载词汇映射
        self.vocab_file = vocab_file
        self.encoder = load_json(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.spm_file = spm_file
        # 加载SPM模型，使用给定的spm_file和参数
        self.sp_model = load_spm(spm_file, self.sp_model_kwargs)

        # 设置编码器的大小为词汇表大小
        self.encoder_size = len(self.encoder)

        # 创建语言标记到ID的映射，使用Fairseq语言代码
        self.lang_token_to_id = {
            self.get_lang_token(lang_code): self.encoder_size + i for i, lang_code in enumerate(fairseq_language_code)
        }
        self.lang_code_to_id = {lang_code: self.encoder_size + i for i, lang_code in enumerate(fairseq_language_code)}
        self.id_to_lang_token = {v: k for k, v in self.lang_token_to_id.items()}

        # 设置源语言和目标语言，默认源语言为英语
        self._src_lang = src_lang if src_lang is not None else "en"
        self.tgt_lang = tgt_lang
        # 获取当前语言的ID，使用源语言设置
        self.cur_lang_id = self.get_lang_id(self._src_lang)

        # 设置虚构词数量
        self.num_madeup_words = num_madeup_words

        # 调用父类的初始化方法，设置其他参数
        super().__init__(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            language_codes=language_codes,
            sp_model_kwargs=self.sp_model_kwargs,
            additional_special_tokens=additional_special_tokens,
            num_madeup_words=num_madeup_words,
            **kwargs,
        )
        # 设置源语言的特殊标记
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    # 返回编码器中的词汇量大小
    def vocab_size(self) -> int:
        return len(self.encoder)

    # 获取词汇表，并将索引与词汇一一对应的字典返回
    def get_vocab(self) -> Dict:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 添加自定义的特殊标记到词汇表中
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 返回源语言代码
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 设置源语言代码，并更新相关特殊标记
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    # 使用句子分段模型对文本进行分词，并返回结果
    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    # 将词汇转换为对应的 ID
    def _convert_token_to_id(self, token):
        if token in self.lang_token_to_id:
            return self.lang_token_to_id[token]
        # 如果未找到对应词汇，则使用未知标记的 ID
        return self.encoder.get(token, self.encoder[self.unk_token])

    # 将 ID 转换为对应的词汇
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        if index in self.id_to_lang_token:
            return self.id_to_lang_token[index]
        # 如果未找到对应 ID，则使用未知标记的词汇
        return self.decoder.get(index, self.unk_token)

    # 将一系列的 tokens 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # 确保特殊标记不会被句子分段模型解码
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    # 获取特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            # If the tokens already have special tokens, delegate to superclass method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Initialize a list of ones corresponding to prefix special tokens
        prefix_ones = [1] * len(self.prefix_tokens)
        # Initialize a list of ones corresponding to suffix special tokens
        suffix_ones = [1] * len(self.suffix_tokens)

        if token_ids_1 is None:
            # If there is only one sequence (token_ids_1 is None), return prefix tokens + sequence tokens + suffix tokens
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones

        # If there are two sequences, return prefix tokens + sequence 1 tokens + sequence 2 tokens + suffix tokens
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An MBART sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `X [eos, src_lang_code]`
        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # If there is only one sequence (token_ids_1 is None), return prefix tokens + token_ids_0 + suffix tokens
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens

        # If there are two sequences, return prefix tokens + token_ids_0 + token_ids_1 + suffix tokens
        # We maintain pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def __getstate__(self) -> Dict:
        # Serialize the object state excluding the sp_model attribute
        state = self.__dict__.copy()
        state["sp_model"] = None  # Ensure sp_model is set to None during serialization
        return state

    def __setstate__(self, d: Dict) -> None:
        # Deserialize the object state
        self.__dict__ = d

        # Ensure backward compatibility by setting sp_model_kwargs if it doesn't exist
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # Load the sp_model attribute using the existing attributes spm_file and sp_model_kwargs
        self.sp_model = load_spm(self.spm_file, self.sp_model_kwargs)
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 将保存目录路径转换为Path对象
        save_dir = Path(save_directory)
        # 如果保存目录不存在，则抛出异常
        if not save_dir.is_dir():
            raise OSError(f"{save_directory} should be a directory")
        
        # 构建词汇表文件保存路径
        vocab_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        # 构建序列化模型文件保存路径
        spm_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["spm_file"]
        )

        # 保存编码器对象到JSON文件
        save_json(self.encoder, vocab_save_path)

        # 如果当前的序列模型文件路径与目标路径不同且存在有效的序列模型文件，则复制序列模型文件
        if os.path.abspath(self.spm_file) != os.path.abspath(spm_save_path) and os.path.isfile(self.spm_file):
            copyfile(self.spm_file, spm_save_path)
        # 否则，如果当前序列模型文件路径无效，则将序列化模型内容写入目标路径
        elif not os.path.isfile(self.spm_file):
            with open(spm_save_path, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回保存的词汇表文件路径和序列模型文件路径的元组
        return (str(vocab_save_path), str(spm_save_path))

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言，并配置源语言特殊标记
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self.src_lang)
        # 调用父类方法，准备序列到序列任务的批处理编码
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _build_translation_inputs(self, raw_inputs, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        # 检查源语言和目标语言是否为空，若为空则抛出值错误异常
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        # 设置当前实例的源语言
        self.src_lang = src_lang
        # 使用模型处理原始输入，并添加特殊标记
        inputs = self(raw_inputs, add_special_tokens=True, **extra_kwargs)
        # 获取目标语言对应的语言ID，并设置为强制BOS标记ID
        tgt_lang_id = self.get_lang_id(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def _switch_to_input_mode(self):
        # 切换为输入模式，设置源语言的特殊标记
        self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        # 切换为目标模式，设置目标语言的特殊标记
        self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        # 获取源语言对应的语言标记，并设置当前语言ID
        lang_token = self.get_lang_token(src_lang)
        self.cur_lang_id = self.lang_token_to_id[lang_token]
        # 设置前缀特殊标记为当前语言ID，后缀特殊标记为结束标记ID
        self.prefix_tokens = [self.cur_lang_id]
        self.suffix_tokens = [self.eos_token_id]
    # 设置目标语言的特殊标记。无前缀，后缀包含[eos, tgt_lang_code]。
    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        # 获取目标语言对应的语言特殊标记
        lang_token = self.get_lang_token(tgt_lang)
        # 将当前语言ID设置为目标语言特殊标记对应的ID
        self.cur_lang_id = self.lang_token_to_id[lang_token]
        # 将前缀标记设置为当前语言ID
        self.prefix_tokens = [self.cur_lang_id]
        # 将后缀标记设置为包含结束符(eos)和目标语言特殊标记对应的ID
        self.suffix_tokens = [self.eos_token_id]

    # 根据语言名称获取语言特殊标记
    def get_lang_token(self, lang: str) -> str:
        return self.lang_code_to_token[lang]

    # 根据语言名称获取语言ID
    def get_lang_id(self, lang: str) -> int:
        # 获取语言特殊标记
        lang_token = self.get_lang_token(lang)
        # 返回语言特殊标记对应的ID
        return self.lang_token_to_id[lang_token]
# 根据指定的参数加载 SentencePieceProcessor 对象
def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    # 使用传入的参数初始化 SentencePieceProcessor 对象
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    # 加载指定路径下的 SentencePiece 模型文件
    spm.Load(str(path))
    # 返回加载后的 SentencePieceProcessor 对象
    return spm


# 加载指定路径的 JSON 文件并返回其内容，可以是字典或列表
def load_json(path: str) -> Union[Dict, List]:
    # 打开指定路径的 JSON 文件作为只读模式
    with open(path, "r") as f:
        # 使用 json 模块加载 JSON 文件内容并返回
        return json.load(f)


# 将数据以 JSON 格式保存到指定路径的文件中
def save_json(data, path: str) -> None:
    # 打开指定路径的文件以写入模式
    with open(path, "w") as f:
        # 使用 json 模块将数据以可读性更好的缩进格式保存到文件中
        json.dump(data, f, indent=2)
```