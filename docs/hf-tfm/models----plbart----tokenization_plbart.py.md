# `.\models\plbart\tokenization_plbart.py`

```
# 设置文件编码为 UTF-8
# Copyright 2022, UCLA NLP, The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# 根据 Apache License, Version 2.0 许可，除非符合许可要求，否则禁止使用此文件
# 可在以下链接获取许可的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 如果法律要求或书面同意，本软件按"原样"分发，不提供任何明示或暗示的担保或条件
# 详细信息请查看许可证
import os  # 导入操作系统相关功能模块
from shutil import copyfile  # 导入 shutil 库中的文件复制函数 copyfile
from typing import Any, Dict, List, Optional, Tuple  # 导入类型提示相关模块

import sentencepiece as spm  # 导入 sentencepiece 库，用于处理文本分词

from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer  # 导入自定义的 tokenization_utils 模块中的类和函数
from ...utils import logging  # 导入自定义的 logging 模块中的日志功能

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

SPIECE_UNDERLINE = "▁"  # 定义一个特殊符号常量 SPIECE_UNDERLINE

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}
# 定义一个字典常量 VOCAB_FILES_NAMES，用于存储词汇文件和分词器文件的名称

PRETRAINED_VOCAB_FILES_MAP = {
    # 定义一个预训练词汇文件映射的字典常量 PRETRAINED_VOCAB_FILES_MAP
    # 定义一个包含多个模型的字典，每个模型关联一个 URL，用于获取其对应的词汇表文件
    "vocab_file": {
        "uclanlp/plbart-base": "https://huggingface.co/uclanlp/plbart-base/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-c-cpp-defect-detection": (
            "https://huggingface.co/uclanlp/plbart-c-cpp-defect-detection/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-cs-java": "https://huggingface.co/uclanlp/plbart-cs-java/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-en_XX-java": (
            "https://huggingface.co/uclanlp/plbart-en_XX-java/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-go-en_XX": (
            "https://huggingface.co/uclanlp/plbart-go-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-java-clone-detection": (
            "https://huggingface.co/uclanlp/plbart-java-clone-detection/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-java-cs": "https://huggingface.co/uclanlp/plbart-java-cs/resolve/main/sentencepiece.bpe.model",
        "uclanlp/plbart-java-en_XX": (
            "https://huggingface.co/uclanlp/plbart-java-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-javascript-en_XX": (
            "https://huggingface.co/uclanlp/plbart-javascript-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-php-en_XX": (
            "https://huggingface.co/uclanlp/plbart-php-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-python-en_XX": (
            "https://huggingface.co/uclanlp/plbart-python-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-refine-java-medium": (
            "https://huggingface.co/uclanlp/plbart-refine-java-medium/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-refine-java-small": (
            "https://huggingface.co/uclanlp/plbart-refine-java-small/resolve/main/sentencepiece.bpe.model"
        ),
        "uclanlp/plbart-ruby-en_XX": (
            "https://huggingface.co/uclanlp/plbart-ruby-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
    }
# 定义预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "uclanlp/plbart-base": 1024,
    "uclanlp/plbart-c-cpp-defect-detection": 1024,
    "uclanlp/plbart-cs-java": 1024,
    "uclanlp/plbart-en_XX-java": 1024,
    "uclanlp/plbart-go-en_XX": 1024,
    "uclanlp/plbart-java-clone-detection": 1024,
    "uclanlp/plbart-java-cs": 1024,
    "uclanlp/plbart-java-en_XX": 1024,
    "uclanlp/plbart-javascript-en_XX": 1024,
    "uclanlp/plbart-php-en_XX": 1024,
    "uclanlp/plbart-python-en_XX": 1024,
    "uclanlp/plbart-refine-java-medium": 1024,
    "uclanlp/plbart-refine-java-small": 1024,
    "uclanlp/plbart-ruby-en_XX": 1024,
}
# 定义 Fairseq 语言代码
FAIRSEQ_LANGUAGE_CODES = {
    "base": ["__java__", "__python__", "__en_XX__"],
    "multi": ["__java__", "__python__", "__en_XX__", "__javascript__", "__php__", "__ruby__", "__go__"],
}
# 定义 Fairseq 语言代码的映射
FAIRSEQ_LANGUAGE_CODES_MAP = {
    "java": "__java__",
    "python": "__python__",
    "en_XX": "__en_XX__",
    "javascript": "__javascript__",
    "php": "__php__",
    "ruby": "__ruby__",
    "go": "__go__",
}
# 定义 PLBartTokenizer 类
class PLBartTokenizer(PreTrainedTokenizer):
    """
    Construct an PLBART tokenizer.

    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.
    Args:
        vocab_file (`str`):
            Path to the vocabulary file. This specifies the location of the vocabulary file to be used by the tokenizer.
        src_lang (`str`, *optional*):
            A string representing the source language. If provided, specifies the source language for the tokenizer.
        tgt_lang (`str`, *optional*):
            A string representing the target language. If provided, specifies the target language for the tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The start of sequence token. Defines the token used to mark the beginning of a sequence.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token. Defines the token used to mark the end of a sequence.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token. Used in scenarios like sequence classification or question answering to separate sequences.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classification token. This token is used as the first token for all tasks.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. If a token is not found in the vocabulary, it is replaced with this token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The padding token. Used to pad sequences to the same length during batching.
        mask_token(`str`, *optional*, defaults to `"<mask>"`):
            The mask token. Used in masking tasks during training. Not used in multi-tokenizer scenarios.
        language_codes (`str`, *optional*, defaults to `"base"`):
            Specifies what language codes to use. Can be `"base"` or `"multi"`.
        sp_model_kwargs (`dict`, *optional*):
            Additional arguments passed to the `SentencePieceProcessor.__init__()` method. These parameters can configure
            subword regularization and other SentencePiece settings like `enable_sampling`, `nbest_size`, and `alpha`.
            See the [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) for details.
    Examples:

    ```python
    >>> from transformers import PLBartTokenizer

    >>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX")
    # 定义示例的 Python 代码短语和其对应的英文翻译，用于模型的输入
    example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
    expected_translation_english = "Returns the maximum value of a b c."
    # 使用预训练模型的 tokenizer 处理示例的 Python 代码和其对应的英文翻译，返回 PyTorch 张量
    inputs = tokenizer(example_python_phrase, text_target=expected_translation_english, return_tensors="pt")
    
    vocab_files_names = VOCAB_FILES_NAMES  # 加载预训练模型的词汇文件名列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 加载预训练模型的最大输入尺寸列表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 加载预训练模型的词汇文件映射表
    model_input_names = ["input_ids", "attention_mask"]  # 定义模型输入的名称列表
    
    prefix_tokens: List[int] = []  # 初始化前缀 tokens 列表
    suffix_tokens: List[int] = []  # 初始化后缀 tokens 列表
    
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        language_codes="base",
        tokenizer_file=None,
        src_lang=None,
        tgt_lang=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        additional_special_tokens=None,
        **kwargs,
    ):
        # 初始化函数，设置各种参数和属性
    
    def __getstate__(self):
        # 序列化对象状态时调用，返回对象的字典形式状态
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state
    
    def __setstate__(self, d):
        # 反序列化对象状态时调用，恢复对象的状态
        self.__dict__ = d
    
        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
    
        # 加载 SentencePiece 模型并设置状态
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
    
    @property
    def vocab_size(self):
        # 计算词汇表的大小，考虑语言编码和偏移量
        if self.language_codes == "base":
            return (
                len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1
            )  # 加 1 用于 mask token
        else:
            return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset
    
    @property
    def src_lang(self) -> str:
        # 获取源语言代码
        return self._src_lang
    
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        # 设置源语言代码，并更新特殊 token
        new_src_lang = self._convert_lang_code_special_format(new_src_lang)
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)
    
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        # 获取特殊 token 的掩码，用于处理输入 token 的特殊性
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

        # If the token list already has special tokens, delegate to superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Create lists of 1s corresponding to prefix and suffix tokens
        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)

        # If token_ids_1 is None, return tokens with prefix, sequence tokens (0s), and suffix
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones

        # If token_ids_1 is provided, return tokens with prefix, token_ids_0, token_ids_1, and suffix
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An PLBART sequence has the following format, where `X` represents the sequence:

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

        # If token_ids_1 is None, concatenate prefix, token_ids_0, and suffix tokens
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens

        # Otherwise, concatenate prefix, token_ids_0, token_ids_1, and suffix tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs from a sequence or a pair of sequences for sequence classification tasks. This is used
        to distinguish between the two sequences in a model that supports sequence pairs.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs representing the second sequence in a pair.

        Returns:
            `List[int]`: List of token type IDs (0 or 1) indicating the sequence type for each token.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. PLBart does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """

        # Separator token ID used in sequence pairs
        sep = [self.sep_token_id]
        # CLS token ID used in sequence pairs
        cls = [self.cls_token_id]

        # If only one sequence is provided, return the mask for that sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # If two sequences are provided, return the mask for both sequences concatenated
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""

        # Ensure source and target languages are provided
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")

        # Convert source and target language codes to special format
        self.src_lang = self._convert_lang_code_special_format(src_lang)
        self.tgt_lang = self._convert_lang_code_special_format(tgt_lang)

        # Generate model inputs with special tokens and specified return type
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)

        # Convert target language to its corresponding token ID
        tgt_lang_id = self.convert_tokens_to_ids(self.tgt_lang)

        # Add forced beginning-of-sequence token ID to inputs
        inputs["forced_bos_token_id"] = tgt_lang_id

        return inputs

    def get_vocab(self):
        # Create a vocabulary dictionary mapping token strings to their IDs
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # Include any additional tokens introduced during model training
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize input text using SentencePiece model and return as list of strings
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) into an ID using the vocabulary."""
        
        # Check if the token exists in the fairseq mapping
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        
        # Obtain token ID from SentencePiece model
        spm_id = self.sp_model.PieceToId(token)

        # Return unknown token ID if SentencePiece returns 0 (unknown token)
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocabulary."""
        
        # Check if the index exists in the fairseq mapping
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        
        # Convert index to token using SentencePiece model
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) into a single string."""
        
        # Concatenate tokens into a single string, replacing special sub-word marker with space
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构建输出词汇表文件的路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不同且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化后的特殊模型内容写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出文件路径的元组
        return (out_vocab_file,)

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "python",
        **kwargs,
    ) -> BatchEncoding:
        # 将源语言代码转换为特殊格式
        self.src_lang = self._convert_lang_code_special_format(src_lang)
        # 将目标语言代码转换为特殊格式
        self.tgt_lang = self._convert_lang_code_special_format(tgt_lang)
        # 调用父类方法，准备序列到序列的批处理数据
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        # 切换到输入模式，设置源语言特殊标记
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        # 切换到目标模式，设置目标语言特殊标记
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        # 将源语言代码转换为特殊格式
        src_lang = self._convert_lang_code_special_format(src_lang)
        # 根据转换后的源语言代码获取其对应的语言代码 ID
        self.cur_lang_code = self.lang_code_to_id[src_lang] if src_lang is not None else None
        # 清空前缀标记
        self.prefix_tokens = []
        # 如果当前语言代码不为 None，则后缀标记为[eos, 当前语言代码]；否则后缀标记为[eos]
        if self.cur_lang_code is not None:
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        # 将目标语言代码转换为特殊格式
        lang = self._convert_lang_code_special_format(lang)
        # 根据转换后的目标语言代码获取其对应的语言代码 ID
        self.cur_lang_code = self.lang_code_to_id[lang] if lang is not None else None
        # 清空前缀标记
        self.prefix_tokens = []
        # 如果当前语言代码不为 None，则后缀标记为[eos, 当前语言代码]；否则后缀标记为[eos]
        if self.cur_lang_code is not None:
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            self.suffix_tokens = [self.eos_token_id]

    def _convert_lang_code_special_format(self, lang: str) -> str:
        """Convert Language Codes to format tokenizer uses if required"""
        # 如果输入的语言代码在映射表中，则转换为对应的格式，否则保持不变
        lang = FAIRSEQ_LANGUAGE_CODES_MAP[lang] if lang in FAIRSEQ_LANGUAGE_CODES_MAP.keys() else lang
        return lang
```