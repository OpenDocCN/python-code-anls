# `.\transformers\models\plbart\tokenization_plbart.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，包括作者和许可证信息
# 根据 Apache 许可证 2.0 版本规定，使用此文件需要遵守许可证规定
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非法律要求或书面同意，否则不得使用此文件
# 根据许可证规定，分发的软件基于“原样”分发，没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言的权限和限制

# 导入所需的库
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库
import sentencepiece as spm

# 导入所需的模块和函数
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义句子片段的下划线符号
SPIECE_UNDERLINE = "▁"

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    # 定义一个字典，包含不同模型对应的词汇文件链接
    "vocab_file": {
        # uclanlp/plbart-base 模型的词汇文件链接
        "uclanlp/plbart-base": "https://huggingface.co/uclanlp/plbart-base/resolve/main/sentencepiece.bpe.model",
        # uclanlp/plbart-c-cpp-defect-detection 模型的词汇文件链接
        "uclanlp/plbart-c-cpp-defect-detection": (
            "https://huggingface.co/uclanlp/plbart-c-cpp-defect-detection/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-cs-java 模型的词汇文件链接
        "uclanlp/plbart-cs-java": "https://huggingface.co/uclanlp/plbart-cs-java/resolve/main/sentencepiece.bpe.model",
        # uclanlp/plbart-en_XX-java 模型的词汇文件链接
        "uclanlp/plbart-en_XX-java": (
            "https://huggingface.co/uclanlp/plbart-en_XX-java/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-go-en_XX 模型的词汇文件链接
        "uclanlp/plbart-go-en_XX": (
            "https://huggingface.co/uclanlp/plbart-go-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-java-clone-detection 模型的词汇文件链接
        "uclanlp/plbart-java-clone-detection": (
            "https://huggingface.co/uclanlp/plbart-java-clone-detection/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-java-cs 模型的词汇文件链接
        "uclanlp/plbart-java-cs": "https://huggingface.co/uclanlp/plbart-java-cs/resolve/main/sentencepiece.bpe.model",
        # uclanlp/plbart-java-en_XX 模型的词汇文件链接
        "uclanlp/plbart-java-en_XX": (
            "https://huggingface.co/uclanlp/plbart-java-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-javascript-en_XX 模型的词汇文件链接
        "uclanlp/plbart-javascript-en_XX": (
            "https://huggingface.co/uclanlp/plbart-javascript-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-php-en_XX 模型的词汇文件链接
        "uclanlp/plbart-php-en_XX": (
            "https://huggingface.co/uclanlp/plbart-php-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-python-en_XX 模型的词汇文件链接
        "uclanlp/plbart-python-en_XX": (
            "https://huggingface.co/uclanlp/plbart-python-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-refine-java-medium 模型的词汇文件链接
        "uclanlp/plbart-refine-java-medium": (
            "https://huggingface.co/uclanlp/plbart-refine-java-medium/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-refine-java-small 模型的词汇文件链接
        "uclanlp/plbart-refine-java-small": (
            "https://huggingface.co/uclanlp/plbart-refine-java-small/resolve/main/sentencepiece.bpe.model"
        ),
        # uclanlp/plbart-ruby-en_XX 模型的词汇文件链接
        "uclanlp/plbart-ruby-en_XX": (
            "https://huggingface.co/uclanlp/plbart-ruby-en_XX/resolve/main/sentencepiece.bpe.model"
        ),
    }
# 预训练位置嵌入的大小，以字典形式存储不同模型的大小
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

# Fairseq语言代码，以字典形式存储不同模型的语言代码列表
FAIRSEQ_LANGUAGE_CODES = {
    "base": ["__java__", "__python__", "__en_XX__"],
    "multi": ["__java__", "__python__", "__en_XX__", "__javascript__", "__php__", "__ruby__", "__go__"],
}

# Fairseq语言代码映射，将语言名称映射为对应的语言代码
FAIRSEQ_LANGUAGE_CODES_MAP = {
    "java": "__java__",
    "python": "__python__",
    "en_XX": "__en_XX__",
    "javascript": "__javascript__",
    "php": "__php__",
    "ruby": "__ruby__",
    "go": "__go__",
}

# 定义PLBartTokenizer类，继承自PreTrainedTokenizer类
class PLBartTokenizer(PreTrainedTokenizer):
    """
    Construct an PLBART tokenizer.

    Adapted from [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [SentencePiece](https://github.com/google/sentencepiece).

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        src_lang (`str`, *optional*):
            A string representing the source language.
        tgt_lang (`str`, *optional*):
            A string representing the target language.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The start of sequence token.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The cls token, which is a special token used as the first token for all tasks.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token(`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masking tasks. This
            is only used in the `"base"` tokenizer type. For `"multi"` tokenizer, masking is never done for the
            downstream tasks.
        language_codes (`str`, *optional*, defaults to `"base"`):
            What language codes to use. Should be one of `"base"` or `"multi"`.
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

    ```python
    >>> from transformers import PLBartTokenizer

    >>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX")
    # 定义一个示例的 Python 代码短语和其对应的英文翻译
    example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
    expected_translation_english = "Returns the maximum value of a b c."
    # 使用 tokenizer 函数对示例 Python 代码短语进行标记化处理，指定目标文本为英文翻译，返回 PyTorch 张量
    inputs = tokenizer(example_python_phrase, text_target=expected_translation_english, return_tensors="pt")
    
    # 初始化一些变量和参数
    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []
    
    # 定义 Tokenizer 类
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
        
    # 序列化 Tokenizer 对象的状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state
    
    # 反序列化 Tokenizer 对象的状态
    def __setstate__(self, d):
        self.__dict__ = d
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
    
    # 获取词汇表大小
    @property
    def vocab_size(self):
        if self.language_codes == "base":
            return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1  # 加 1 是为了 mask token
        else:
            return len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset
    
    # 获取源语言
    @property
    def src_lang(self) -> str:
        return self._src_lang
    
    # 设置源语言
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        new_src_lang = self._convert_lang_code_special_format(new_src_lang)
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)
    
    # 获取特殊 token 的 mask
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
            # If the token list already has special tokens, return the special tokens mask
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            # If there is no second list of IDs, return the token list with added special tokens
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        # If there is a second list of IDs, return both lists with added special tokens
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
        if token_ids_1 is None:
            # If there is no second list of IDs, return the token list with added special tokens
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # If there is a second list of IDs, return both lists with added special tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传入的两个序列创建一个用于序列对分类任务的掩码。PLBart 不使用 token type ids，因此返回一个全为零的列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 全为零的列表。
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """用于翻译流水线，准备用于 generate 函数的输入"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = self._convert_lang_code_special_format(src_lang)
        self.tgt_lang = self._convert_lang_code_special_format(tgt_lang)
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(self.tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """使用词汇表将一个 token（str）转换为 id。"""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # 如果 SP 模型返回 0，则需要返回未知 token
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """使用词汇表将一个索引（整数）转换为 token（str）。"""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """将一系列 token（子词的字符串）转换为单个字符串。"""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string
    # 保存词汇表到指定目录下，可指定文件名前缀，默认为 None
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，若不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 若词汇表文件路径与当前词汇表路径不同且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 若当前词汇表文件不存在，则将当前词汇表序列化后写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        # 返回输出文件路径
        return (out_vocab_file,)

    # 准备用于序列到序列模型的批处理数据
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "python",
        **kwargs,
    ) -> BatchEncoding:
        # 转换源语言代码格式
        self.src_lang = self._convert_lang_code_special_format(src_lang)
        # 转换目标语言代码格式
        self.tgt_lang = self._convert_lang_code_special_format(tgt_lang)
        # 调用父类方法准备数据批
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    # 切换到输入模式
    def _switch_to_input_mode(self):
        # 设置源语言的特殊标记
        return self.set_src_lang_special_tokens(self.src_lang)

    # 切换到目标模式
    def _switch_to_target_mode(self):
        # 设置目标语言的特殊标记
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    # 设置源语言的特殊标记
    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        # 将语言代码转换为特殊格式（若需要）
        src_lang = self._convert_lang_code_special_format(src_lang)
        # 获取语言代码对应的 ID
        self.cur_lang_code = self.lang_code_to_id[src_lang] if src_lang is not None else None
        # 设置前缀标记为空
        self.prefix_tokens = []
        # 如果语言代码不为空，则设置后缀标记为 [eos, 语言代码]
        if self.cur_lang_code is not None:
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            # 否则，只设置后缀标记为 [eos]
            self.suffix_tokens = [self.eos_token_id]

    # 设置目标语言的特殊标记
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        # 将语言代码转换为特殊格式（若需要）
        lang = self._convert_lang_code_special_format(lang)
        # 获取语言代码对应的 ID
        self.cur_lang_code = self.lang_code_to_id[lang] if lang is not None else None
        # 设置前缀标记为空
        self.prefix_tokens = []
        # 如果语言代码不为空，则设置后缀标记为 [eos, 语言代码]
        if self.cur_lang_code is not None:
            self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]
        else:
            # 否则，只设置后缀标记为 [eos]
            self.suffix_tokens = [self.eos_token_id]

    # 将语言代码转换为特殊格式（若需要）
    def _convert_lang_code_special_format(self, lang: str) -> str:
        """Convert Language Codes to format tokenizer uses if required"""
        # 若语言代码在映射表中，则转换格式，否则保持原样
        lang = FAIRSEQ_LANGUAGE_CODES_MAP[lang] if lang in FAIRSEQ_LANGUAGE_CODES_MAP.keys() else lang
        return lang
```