# `.\models\distilbert\tokenization_distilbert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""DistilBERT 的 Tokenization 类。"""

# 导入所需的模块
import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/vocab.txt",
        "distilbert-base-uncased-distilled-squad": (
            "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/vocab.txt"
        ),
        "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/vocab.txt",
        "distilbert-base-cased-distilled-squad": (
            "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/vocab.txt"
        ),
        "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/vocab.txt",
        "distilbert-base-multilingual-cased": (
            "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/vocab.txt"
        ),
    }
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "distilbert-base-uncased": 512,
    "distilbert-base-uncased-distilled-squad": 512,
    "distilbert-base-cased": 512,
    "distilbert-base-cased-distilled-squad": 512,
    "distilbert-base-german-cased": 512,
    "distilbert-base-multilingual-cased": 512,
}

# 预训练模型的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "distilbert-base-uncased": {"do_lower_case": True},
    "distilbert-base-uncased-distilled-squad": {"do_lower_case": True},
    "distilbert-base-cased": {"do_lower_case": False},
    "distilbert-base-cased-distilled-squad": {"do_lower_case": False},
    "distilbert-base-german-cased": {"do_lower_case": False},
    "distilbert-base-multilingual-cased": {"do_lower_case": False},
}

# 从 transformers.models.bert.tokenization_bert.load_vocab 复制的函数
def load_vocab(vocab_file):
    """加载词汇文件到字典中。"""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab
# 从transformers.models.bert.tokenization_bert.whitespace_tokenize中复制的whitespace_tokenize函数
def whitespace_tokenize(text):
    """对文本进行基本的空格清理和分割。"""
    # 去除文本两端的空格
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，得到token列表
    tokens = text.split()
    return tokens

# DistilBertTokenizer类，继承自PreTrainedTokenizer
class DistilBertTokenizer(PreTrainedTokenizer):
    r"""
    构建一个DistilBERT分词器。基于WordPiece。

    这个分词器继承自[`PreTrainedTokenizer`]，其中包含大部分主要方法。用户应该参考
    这个超类以获取有关这些方法的更多信息。

    Args:
        vocab_file (`str`):
            包含词汇表的文件。
        do_lower_case (`bool`, *optional*, 默认为 `True`):
            在分词时是否将输入转换为小写。
        do_basic_tokenize (`bool`, *optional*, 默认为 `True`):
            在WordPiece之前是否进行基本分词。
        never_split (`Iterable`, *optional*):
            在分词时永远不会被分割的token集合。仅在`do_basic_tokenize=True`时有效。
        unk_token (`str`, *optional*, 默认为 `"[UNK]"`):
            未知token。词汇表中不存在的token无法转换为ID，而是设置为此token。
        sep_token (`str`, *optional*, 默认为 `"[SEP]"`):
            分隔符token，在从多个序列构建序列时使用，例如用于序列分类的两个序列
            或用于文本和问题的问题回答。还用作使用特殊token构建的序列的最后一个token。
        pad_token (`str`, *optional*, 默认为 `"[PAD]"`):
            用于填充的token，例如在批处理不同长度的序列时使用。
        cls_token (`str`, *optional*, 默认为 `"[CLS]"`):
            分类器token，在进行序列分类时使用（整个序列的分类而不是每个token的分类）。
            它是使用特殊token构建的序列的第一个token。
        mask_token (`str`, *optional*, 默认为 `"[MASK]"`):
            用于掩码值的token。这是在使用掩码语言建模训练此模型时使用的token。
            这是模型将尝试预测的token。
        tokenize_chinese_chars (`bool`, *optional*, 默认为 `True`):
            是否对中文字符进行分词。

            对于日语，这可能应该被禁用（参见此
            [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将由
            `lowercase`的值确定（与原始BERT相同）。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 如果词汇文件不存在，则抛出异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = DistilBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 将词汇表转换为有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要基本分词，则初始化基本分词器
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 初始化词片段分词器
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类初始化方法
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    @property
    # 获取是否小写属性
    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.do_lower_case 复制而来
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    # 获取词汇表大小属性
    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.vocab_size 复制而来
    def vocab_size(self):
        return len(self.vocab)

    # 获取词汇表方法
    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.get_vocab 复制而来
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 分词方法
    # 从 transformers.models.bert.tokenization_bert.BertTokenizer._tokenize 复制而来
    # 将文本进行分词处理，返回分词后的结果
    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        # 如果需要进行基本的分词处理
        if self.do_basic_tokenize:
            # 使用基本分词器对文本进行分词处理
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果分词结果在不分割的特殊标记集合中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 使用 WordPiece 分词器对 token 进行分词处理
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 使用 WordPiece 分词器对整个文本进行分词处理
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将一系列 tokens 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建带有特殊标记的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 获取特殊标记的掩码
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

        # Check if the token list already has special tokens added
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If there are two token lists for sequence pairs, add special tokens accordingly
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # If there is only one token list, add special tokens accordingly
        return [1] + ([0] * len(token_ids_0)) + [1]

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.create_token_type_ids_from_sequences
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # If there is only one token list, return the token type IDs for the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # If there are two token lists for sequence pairs, return the token type IDs for both sequences
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.save_vocabulary
    # 保存词汇表到指定目录，返回保存的文件路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引
        index = 0
        # 判断保存目录是否存在
        if os.path.isdir(save_directory):
            # 构建词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 构建词汇表文件路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件，写入词汇表内容
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表，按索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 检查索引是否连续
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                # 写入词汇表中的词汇
                writer.write(token + "\n")
                index += 1
        # 返回保存的文件路径
        return (vocab_file,)
# 从transformers.models.bert.tokenization_bert.BasicTokenizer复制而来的类
class BasicTokenizer(object):
    """
    构造一个BasicTokenizer，用于运行基本的分词（标点符号分割，小写化等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在分词时将输入小写化。
        never_split (`Iterable`, *optional*):
            在分词时永远不会被分割的标记集合。仅在`do_basic_tokenize=True`时有效。
        tokenize_chinese_chars (`bool`, *optional* defaults to `True`):
            是否分词中文字符。

            对于日语，这可能需要禁用（参见此[问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将由`lowercase`的值确定（与原始BERT相同）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词可以捕获单词的完整上下文，例如缩略词。
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果never_split未指定，则设为一个空列表
        if never_split is None:
            never_split = []
        # 初始化各个参数
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    # 对文本进行基本的分词处理。对于子词分词，请参考WordPieceTokenizer。
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 将never_split和self.never_split的并集赋值给never_split，如果never_split为None，则直接使用self.never_split
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本内容
        text = self._clean_text(text)

        # 该部分是在2018年11月1日为多语言和中文模型添加的。现在也应用于英语模型，但由于英语模型没有在任何中文数据上训练，
        # 并且通常不包含任何中文数据（英语维基百科中有一些中文词汇，因此词汇表中有中文字符）。
        if self.tokenize_chinese_chars:
            # 对中文字符进行分词处理
            text = self._tokenize_chinese_chars(text)
        # 防止将具有不同Unicode代码点的相同字符视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空格进行分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    # 如果需要转换为小写，则将token转换为小写
                    token = token.lower()
                    if self.strip_accents is not False:
                        # 如果需要去除重音符号，则运行去除重音符号的方法
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果需要去除重音符号，则运行去除重音符号的方法
                    token = self._run_strip_accents(token)
            # 运行基于标点符号的分词方法，并将结果添加到split_tokens中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空格进行分词，并返回结果
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    # 从文本中去除重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 将文本中的字符进行Unicode标准化
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """在文本上分割标点符号。"""
        # 如果不需要在标点符号上分割或者文本在不需要分割的列表中，则返回原文本
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """在任何中日韩字符周围添加空格。"""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """检查 CP 是否是中日韩字符的代码点。"""
        # 这里将“中日韩字符”定义为CJK Unicode块中的任何字符：
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # 请注意，CJK Unicode块并不包括所有日语和韩语字符，
        # 尽管其名称如此。现代韩语Hangul字母是一个不同的块，
        # 日语平假名和片假名也是如此。这些字母用于书写以空格分隔的单词，
        # 因此它们不会被特殊对待，而是像其他所有语言一样处理。
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """对文本执行无效字符删除和空格清理。"""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
# 从transformers.models.bert.tokenization_bert中复制了WordpieceTokenizer类
class WordpieceTokenizer(object):
    """运行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer类，传入词汇表、未知标记和单词的最大输入字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将文本标记化为其单词片段。这使用一种贪婪的最长匹配算法来使用给定的词汇表进行分词。

        例如，`input = "unaffable"`将返回`["un", "##aff", "##able"]`。

        Args:
            text: 一个单个标记或以空格分隔的标记。这应该已经通过*BasicTokenizer*。

        Returns:
            一个单词片段标记的列表。
        """

        # 初始化空的输出单词片段标记列表
        output_tokens = []
        # 循环遍历分词后的每个标记
        for token in whitespace_tokenize(text):
            # 将标记转换成字符列表
            chars = list(token)
            # 如果标记字符数超过了最大输入字符数，将未知标记加入输出列表并继续下一个标记
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            # 当开始位置小于字符长度时，执行循环
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                # 当开始位置小于结束位置时，执行循环
                while start < end:
                    # 获取从开始位置到结束位置的子字符串
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    # 如果子字符串在词汇表中，则当前子字符串即为词汇的一部分，退出循环
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                # 如果没有找到匹配的词汇，则设置为坏词汇，退出外层循环
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            # 如果存在坏词汇，则将未知标记加入输出列表，否则将分词结果添加到输出列表
            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
```