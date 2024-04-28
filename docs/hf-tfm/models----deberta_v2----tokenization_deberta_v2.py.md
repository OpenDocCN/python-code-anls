# `.\models\deberta_v2\tokenization_deberta_v2.py`

```py
# coding=utf-8
# 版权 2020 年 Microsoft 和 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证第 2.0 版（"许可证"）获得许可;
# 你不得使用此文件，除非符合许可的规定。
# 你可以在许可证下获得一份许可证副本。
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据 "原样" 分发，
# 无任何明示或暗示的担保或条件。
# 请参阅许可证以了解许可下的特定语言的权限和限制。
"""DeBERTa 模型的标记化类。

import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as sp

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

logger = logging.get_logger(__name__)

# 指定预训练词汇文件的路径
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/spm.model"
        ),
        "microsoft/deberta-v2-xxlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/spm.model"
        ),
    }
}

# 每个预训练模型的定位嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-v2-xlarge": 512,
    "microsoft/deberta-v2-xxlarge": 512,
    "microsoft/deberta-v2-xlarge-mnli": 512,
    "microsoft/deberta-v2-xxlarge-mnli": 512,
}

# 每个预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-v2-xlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xlarge-mnli": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge-mnli": {"do_lower_case": False},
}

# 预训练词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "spm.model"}


class DebertaV2Tokenizer(PreTrainedTokenizer):
    """
    构建一个 DeBERTa-v2 标记化器。基于 [SentencePiece](https://github.com/google/sentencepiece)。

    """

    # 指定词汇文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 指定预训练词汇文件的映射关系
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 指定预训练模型的初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 指定每个模型能够处理的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        split_by_punct=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    )
    # 定义一个类，用于处理文本的分词器
    class PreTrainedTokenizer(SpecialTokensMixin, ABC):
        # 创建一个实例时可以传入 sp_model_kwargs 参数，如果未指定则为空字典
        def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            add_prefix_space=False,
            sp_model_kwargs=None,
            **kwargs,
        ) -> None:
            self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
    
            # 如果指定的词汇文件不存在，则抛出异常
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                    " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
            self.do_lower_case = do_lower_case
            self.split_by_punct = split_by_punct
            self.vocab_file = vocab_file
            # 创建一个 SPMTokenizer 实例
            self._tokenizer = SPMTokenizer(
                vocab_file, None, split_by_punct=split_by_punct, sp_model_kwargs=self.sp_model_kwargs
            )
            # 如果 unk_token 是字符串，则创建一个 AddedToken 实例
            unk_token = AddedToken(unk_token, normalized=True, special=True) if isinstance(unk_token, str) else unk_token
            # 调用父类的初始化方法，传入各种参数和 kwargs
            super().__init__(
                do_lower_case=do_lower_case,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                mask_token=mask_token,
                split_by_punct=split_by_punct,
                sp_model_kwargs=self.sp_model_kwargs,
                **kwargs,
            )
            # 将特殊 token 设置为所有的特殊 token
            self._tokenizer.special_tokens = self.all_special_tokens
    
        # 用于返回词汇表的大小
        @property
        def vocab_size(self):
            return len(self.vocab)
    
        # 用于返回词汇表
        @property
        def vocab(self):
            return self._tokenizer.vocab
    
        # 用于返回扩展后的词汇表
        def get_vocab(self):
            vocab = self.vocab.copy()
            vocab.update(self.get_added_vocab())
            return vocab
    
        # 用于将文本分词成 tokens
        def _tokenize(self, text: str) -> List[str]:
            """Take as input a string and return a list of strings (tokens) for words/sub-words"""
            if self.do_lower_case:
                text = text.lower()
            return self._tokenizer.tokenize(text)
    
        # 用于将 token 转换为对应的 id
        def _convert_token_to_id(self, token):
            """Converts a token (str) in an id using the vocab."""
            return self._tokenizer.spm.PieceToId(token)
    
        # 用于将 id 转换为对应的 token
        def _convert_id_to_token(self, index):
            """Converts an index (integer) in a token (str) using the vocab."""
            return self._tokenizer.spm.IdToPiece(index) if index < self.vocab_size else self.unk_token
    
        # 用于将 tokens 转换为字符串
        def convert_tokens_to_string(self, tokens):
            """Converts a sequence of tokens (string) in a single string."""
            return self._tokenizer.decode(tokens)
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        # If only one sequence is provided, add special tokens and return
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        
        # Initialize special tokens for classification
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # Concatenate tokens for a pair of sequences with special tokens and return
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

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

        # If token list already has special tokens, directly return the mask
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If a pair of sequences is provided, create a mask with special tokens added, otherwise for a single sequence
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # For a single sequence, create a mask with special tokens added
        return [1] + ([0] * len(token_ids_0)) + [1]
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```py

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # If token_ids_1 is None, return list of 0s for the first portion of the mask
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # Return concatenated list of 0s and 1s for the mask based on the two sequences
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # Pop the "add_prefix_space" parameter from the kwargs with default value False
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        # If text is already split into words or add_prefix_space is True, add a space at the beginning
        if is_split_into_words or add_prefix_space:
            text = " " + text
        # Return updated text and kwargs
        return (text, kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Save the vocabulary using the tokenizer's save_pretrained method
        return self._tokenizer.save_pretrained(save_directory, filename_prefix=filename_prefix)
# 定义一个基于 SentencePiece 的分词器类
class SPMTokenizer:
    r"""
    基于 [SentencePiece](https://github.com/google/sentencepiece) 构建分词器。

    Args:
        vocab_file (`str`):
            包含实例化分词器所需词汇的 [SentencePiece](https://github.com/google/sentencepiece) 文件
            (通常具有 *.spm* 扩展名)。
        sp_model_kwargs (`dict`, *optional*):
            将传递给 `SentencePieceProcessor.__init__()` 方法的参数。[SentencePiece 的 Python 封装](https://github.com/google/sentencepiece/tree/master/python)
            可用于设置：

            - `enable_sampling`: 启用子词正则化。
            - `nbest_size`: 用于单字词的采样参数。对于 BPE-Dropout 无效。

              - `nbest_size = {0,1}`: 不执行采样。
              - `nbest_size > 1`: 从 nbest_size 结果中采样。
              - `nbest_size < 0`: 假设 nbest_size 为无限大，并使用前向-后向采样算法从所有假设(格)中采样。

            - `alpha`: 单字词采样的平滑参数，以及 BPE-dropout 的合并操作的丢失概率。
    """

    def __init__(
        self, vocab_file, special_tokens, split_by_punct=False, sp_model_kwargs: Optional[Dict[str, Any]] = None
    ):
        # 是否通过标点符号进行分割
        self.split_by_punct = split_by_punct
        # 词汇文件路径
        self.vocab_file = vocab_file
        # SentencePiece 参数
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 初始化 SentencePiece 处理器
        spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        # 检查词汇文件是否存在
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"{vocab_file} does not exist!")
        # 加载词汇文件
        spm.load(vocab_file)
        # 获取词汇表大小
        bpe_vocab_size = spm.GetPieceSize()
        # 创建词汇表映射
        # <unk> 0+1
        # <s> 1+1
        # </s> 2+1
        self.vocab = {spm.IdToPiece(i): i for i in range(bpe_vocab_size)}
        self.ids_to_tokens = [spm.IdToPiece(i) for i in range(bpe_vocab_size)]
        # 初始化 SentencePiece 对象
        self.spm = spm
        # 特殊标记
        self.special_tokens = special_tokens

    # 获取对象状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["spm"] = None
        return state

    # 设置对象状态
    def __setstate__(self, d):
        self.__dict__ = d

        # 向后兼容
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 加载 SentencePiece
        self.spm = sp.SentencePieceProcessor(**self.sp_model_kwargs)
        self.spm.Load(self.vocab_file)

    # 分词
    def tokenize(self, text):
        return self._encode_as_pieces(text)

    # 将 id 转换为标记
    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens
    # 将 tokens 解码为文本，返回解码后的字符串
    def decode(self, tokens, start=-1, end=-1, raw_text=None):
        # 如果未提供原始文本，则创建空列表和空字符串以存储解码后的文本
        if raw_text is None:
            current_sub_tokens = []
            out_string = ""
            prev_is_special = False
            for token in tokens:
                # 确保特殊字符不使用 sentencepiece 模型解码
                if token in self.special_tokens:
                    if not prev_is_special:
                        out_string += " "
                    out_string += self.spm.decode_pieces(current_sub_tokens) + token
                    prev_is_special = True
                    current_sub_tokens = []
                else:
                    current_sub_tokens.append(token)
                    prev_is_special = False
            out_string += self.spm.decode_pieces(current_sub_tokens)
            return out_string.strip()
        else:
            words = self.split_to_words(raw_text)
            word_tokens = [self.tokenize(w) for w in words]
            token2words = [0] * len(tokens)
            tid = 0
            for i, w in enumerate(word_tokens):
                for k, t in enumerate(w):
                    token2words[tid] = i
                    tid += 1
            word_start = token2words[start]
            word_end = token2words[end] if end < len(tokens) else len(words)
            text = "".join(words[word_start:word_end])
            return text

    # 向特殊 tokens 列表中添加一个特殊字符
    def add_special_token(self, token):
        if token not in self.special_tokens:
            self.special_tokens.append(token)
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) - 1
                self.ids_to_tokens.append(token)
        return self.id(token)

    # 判断 token 是否为整个单词的一部分
    def part_of_whole_word(self, token, is_bos=False):
        logger.warning_once(
            "The `DebertaTokenizer.part_of_whole_word` method is deprecated and will be removed in `transformers==4.35`"
        )
        if is_bos:
            return True
        if (
            len(token) == 1
            and (_is_whitespace(list(token)[0]) or _is_control(list(token)[0]) or _is_punctuation(list(token)[0]))
        ) or token in self.special_tokens:
            return False

        word_start = b"\xe2\x96\x81".decode("utf-8")
        return not token.startswith(word_start)

    # 返回填充符
    def pad(self):
        return "[PAD]"

    # 返回开始符
    def bos(self):
        return "[CLS]"

    # 返回结束符
    def eos(self):
        return "[SEP]"

    # 返回未知符
    def unk(self):
        return "[UNK]"

    # 返回掩码符
    def mask(self):
        return "[MASK]"

    # 根据 id 返回符号
    def sym(self, id):
        return self.ids_to_tokens[id]

    # 根据符号返回 id
    def id(self, sym):
        logger.warning_once(
            "The `DebertaTokenizer.id` method is deprecated and will be removed in `transformers==4.35`"
        )
        return self.vocab[sym] if sym in self.vocab else 1
    # 将文本编码为片段
    def _encode_as_pieces(self, text):
        # 转换文本为Unicode编码
        text = convert_to_unicode(text)
        # 如果按标点符号分割
        if self.split_by_punct:
            # 对文本根据标点符号进行分割
            words = self._run_split_on_punc(text)
            # 使用训练好的SPM模型对每个分割后的单词编码
            pieces = [self.spm.encode(w, out_type=str) for w in words]
            # 展平编码结果
            return [p for w in pieces for p in w]
        else:
            # 直接使用SPM模型对文本进行编码
            return self.spm.encode(text, out_type=str)

    # 将文本拆分为单词
    def split_to_words(self, text):
        pieces = self._encode_as_pieces(text)
        # 定义单词的起始标志
        word_start = b"\xe2\x96\x81".decode("utf-8")
        words = []
        offset = 0
        prev_end = 0
        for i, p in enumerate(pieces):
            if p.startswith(word_start):
                if offset > prev_end:
                    words.append(text[prev_end:offset])
                prev_end = offset
                w = p.replace(word_start, "")
            else:
                w = p
            try:
                s = text.index(w, offset)
                pn = ""
                k = i + 1
                while k < len(pieces):
                    pn = pieces[k].replace(word_start, "")
                    if len(pn) > 0:
                        break
                    k += 1

                if len(pn) > 0 and pn in text[offset:s]:
                    offset = offset + 1
                else:
                    offset = s + len(w)
            except Exception:
                offset = offset + 1

        if prev_end < offset:
            words.append(text[prev_end:offset])

        return words

    # 在文本中运行标点符号分割
    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):  # 判断是否为标点符号
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:  # 开始新单词
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    # 保存预训练模型
    def save_pretrained(self, path: str, filename_prefix: str = None):
        # 获取vocab文件名
        filename = VOCAB_FILES_NAMES[list(VOCAB_FILES_NAMES.keys())[0]]
        # 如果有前缀，则使用前缀+文件名
        if filename_prefix is not None:
            filename = filename_prefix + "-" + filename
        full_path = os.path.join(path, filename)
        # 将序列化后的模型写入文件
        with open(full_path, "wb") as fs:
            fs.write(self.spm.serialized_model_proto())
        return (full_path,)
# 检查字符是否为空格字符
def _is_whitespace(char):
    # 四种空格字符：空格、制表符、换行符、回车符
    # 将其视为空格字符

    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    # 获取字符的 Unicode 分类
    cat = unicodedata.category(char)
    # "Zs" 为 Unicode 分类中的空格字符
    if cat == "Zs":
        return True
    return False


# 检查字符是否为控制字符
def _is_control(char):
    # 制表符、换行符、回车符不被视为控制字符
    if char == "\t" or char == "\n" or char == "\r":
        return False
    # 获取字符的 Unicode 分类
    cat = unicodedata.category(char)
    # 以 C 开头的 Unicode 分类为控制字符
    if cat.startswith("C"):
        return True
    return False


# 检查字符是否为标点符号字符
def _is_punctuation(char):
    # ASCII 中的非字母/数字字符视为标点符号字符
    # "^", "$", "`" 虽然不属于 Unicode 标点符号类别，但为了一致性也将其视为标点符号
    cp = ord(char)
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    # 获取字符的 Unicode 分类
    cat = unicodedata.category(char)
    # 以 P 开头的 Unicode 分类为标点符号类别
    if cat.startswith("P"):
        return True
    return False


# 将文本转换为 Unicode 编码（如果尚未转换），假定输入为 utf-8 编码
def convert_to_unicode(text):
    # 如果输入已经是字符串，则直接返回
    if isinstance(text, str):
        return text
    # 如果输入是字节流，则将其以 utf-8 编码解码为字符串
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        # 抛出异常，表示不支持的字符串类型
        raise ValueError(f"Unsupported string type: {type(text)}")
```