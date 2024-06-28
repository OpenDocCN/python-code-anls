# `.\models\roformer\tokenization_roformer.py`

```py
# coding=utf-8
# 版权所有 2021 年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证版本 2.0（“许可证”）许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证详细信息，请参阅许可证。

"""RoFormer 的标记化类。"""

import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 定义词汇文件名映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/vocab.txt",
        "junnyu/roformer_chinese_char_small": (
            "https://huggingface.co/junnyu/roformer_chinese_char_small/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_chinese_char_base": (
            "https://huggingface.co/junnyu/roformer_chinese_char_base/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_discriminator": (
            "https://huggingface.co/junnyu/roformer_small_discriminator/resolve/main/vocab.txt"
        ),
        "junnyu/roformer_small_generator": (
            "https://huggingface.co/junnyu/roformer_small_generator/resolve/main/vocab.txt"
        ),
    }
}

# 预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "junnyu/roformer_chinese_small": 1536,
    "junnyu/roformer_chinese_base": 1536,
    "junnyu/roformer_chinese_char_small": 512,
    "junnyu/roformer_chinese_char_base": 512,
    "junnyu/roformer_small_discriminator": 128,
    "junnyu/roformer_small_generator": 128,
}

# 预训练模型的初始化配置映射
PRETRAINED_INIT_CONFIGURATION = {
    "junnyu/roformer_chinese_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_base": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_small": {"do_lower_case": True},
    "junnyu/roformer_chinese_char_base": {"do_lower_case": True},
    "junnyu/roformer_small_discriminator": {"do_lower_case": True},
    "junnyu/roformer_small_generator": {"do_lower_case": True},
}

# 从transformers.models.bert.tokenization_bert.load_vocab复制而来的函数
def load_vocab(vocab_file):
    """加载一个词汇文件到一个有序字典中。"""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    # 使用 enumerate 函数遍历 tokens 列表，同时获取索引 index 和每个 token
    for index, token in enumerate(tokens):
        # 去除 token 字符串末尾的换行符 "\n"
        token = token.rstrip("\n")
        # 将 token 添加到 vocab 字典中，键为 token，值为 index
        vocab[token] = index
    # 返回填充完毕的 vocab 字典
    return vocab
# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制而来
def whitespace_tokenize(text):
    """对文本进行基本的空格清理和分割。"""
    # 去除文本两端的空白字符
    text = text.strip()
    # 如果文本为空字符串，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，得到分词结果
    tokens = text.split()
    # 返回分词结果列表
    return tokens


# 从 transformers.models.bert.tokenization_bert.BasicTokenizer 复制而来
class BasicTokenizer(object):
    """
    构造一个 BasicTokenizer 对象，执行基本的分词（分割标点符号、转换为小写等）。

    Args:
        do_lower_case (`bool`, *可选*, 默认为 `True`):
            是否在分词时转换为小写。
        never_split (`Iterable`, *可选*):
            在分词过程中永远不会分割的词汇集合。仅在 `do_basic_tokenize=True` 时有效。
        tokenize_chinese_chars (`bool`, *可选*, 默认为 `True`):
            是否分割中文字符。

            对于日语，这可能需要禁用（参见此处
            [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *可选*):
            是否去除所有重音符号。如果未指定，则由 `lowercase` 的值决定（与原始 BERT 一致）。
        do_split_on_punc (`bool`, *可选*, 默认为 `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词可以捕获单词的完整上下文，例如缩写。

    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果 never_split 为 None，则设为空列表
        if never_split is None:
            never_split = []
        # 设置是否转换为小写
        self.do_lower_case = do_lower_case
        # 设置永远不会分割的词汇集合
        self.never_split = set(never_split)
        # 设置是否分割中文字符
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设置是否去除重音符号
        self.strip_accents = strip_accents
        # 设置是否在标点符号上进行分割
        self.do_split_on_punc = do_split_on_punc
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        # 如果提供了 `never_split` 参数，则将其与对象自身的 `never_split` 集合取并集
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，包括一些预处理步骤
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # 如果设置了 `tokenize_chinese_chars` 为 True，则执行中文字符的特殊处理
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 对文本进行 Unicode 规范化，确保统一字符的表示形式
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 将文本按空格分割成原始 token 列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历每个 token 进行处理
        for token in orig_tokens:
            # 如果 token 不在 `never_split` 集合中，则考虑是否进行小写处理和去除重音符号处理
            if token not in never_split:
                if self.do_lower_case:
                    # 如果设置了小写处理，则将 token 转换为小写
                    token = token.lower()
                    # 如果不是明确设置为 False，则执行去除重音符号的处理
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 如果设置了去除重音符号，则执行去除重音符号的处理
                    token = self._run_strip_accents(token)
            # 根据标点符号分割 token，并添加到分割后的 token 列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 将分割后的 token 列表按空格连接成字符串，并再次按空格分割为最终输出的 token 列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号拆分，或者指定的文本不应该被拆分，则直接返回原始文本列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号，则将其作为一个单独的列表项添加到输出中，并标记可以开始新单词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，根据当前是否应该开始新单词来添加到上一个列表项或新建一个列表项
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        # 将列表中的每个列表项（字符列表）转换为字符串，并返回最终的拆分结果列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是中文字符，则在其前后添加空格，并添加到输出中；否则直接添加到输出
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将输出列表中的所有元素合并为一个字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的 Unicode 码点是否属于CJK字符范围内的任何一个范围，返回布尔值
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
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或控制字符，则跳过不添加到输出；如果是空白字符则替换为单个空格
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将输出列表中的所有元素合并为一个字符串并返回
        return "".join(output)
# Copied from transformers.models.bert.tokenization_bert.WordpieceTokenizer
# 从 transformers.models.bert.tokenization_bert.WordpieceTokenizer 复制而来的类定义

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""
    # 执行 WordPiece 分词的类

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化函数，接收词汇表、未知标记和每个单词最大输入字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.

        Returns:
            A list of wordpiece tokens.
        """
        # 将文本分词成 WordPiece 格式的片段。使用贪婪的最长匹配算法，根据给定的词汇表进行分词。

        output_tokens = []
        for token in whitespace_tokenize(text):
            # 对文本中的每个 token 进行处理，使用 whitespace_tokenize 进行分割
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                # 如果 token 长度超过设定的最大输入字符数，则使用未知标记
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    # 从当前位置开始向后逐渐减小窗口，尝试匹配最长的词片段
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        # 如果找到了在词汇表中的词片段，则记录下来
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    # 如果未找到匹配的词片段，则标记为无效
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                # 如果整个 token 都无法分割成有效的词片段，则使用未知标记
                output_tokens.append(self.unk_token)
            else:
                # 否则将分割得到的子词片段添加到输出 tokens 中
                output_tokens.extend(sub_tokens)
        return output_tokens
    # 导入四个常量，分别是词汇表文件名列表、预训练词汇表文件映射、预训练模型输入最大长度列表和预训练模型初始化配置
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 初始化函数，用于创建一个新的 Tokenizer 对象
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
        # 检查给定的词汇文件是否存在，如果不存在则抛出 ValueError 异常
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表文件，并存储到 self.vocab 中
        self.vocab = load_vocab(vocab_file)
        # 创建一个从词汇映射到 ID 的有序字典 self.ids_to_tokens
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否执行基本的分词操作的标志
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本分词操作，则创建 BasicTokenizer 对象
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 创建 WordpieceTokenizer 对象，用于执行 WordPiece 分词
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        
        # 尝试导入 rjieba 库，用于中文分词
        try:
            import rjieba
        except ImportError:
            # 如果导入失败，抛出 ImportError 异常并提供安装提示信息
            raise ImportError(
                "You need to install rjieba to use RoFormerTokenizer. "
                "See https://pypi.org/project/rjieba/ for installation."
            )
        # 将 rjieba 模块赋值给 self.jieba，以便后续中文分词使用

        self.jieba = rjieba

        # 调用父类的初始化方法，传递相同的参数和额外的关键字参数
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

    # 返回当前 Tokenizer 是否执行小写处理的属性
    @property
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    # 返回词汇表大小的属性
    @property
    def vocab_size(self):
        return len(self.vocab)

    # 获取当前对象的状态以进行序列化
    def __getstate__(self):
        state = self.__dict__.copy()
        # 将 self.jieba 置为 None，避免序列化时引入额外依赖
        state["jieba"] = None
        return state

    # 根据给定的状态设置当前对象的状态以进行反序列化
    def __setstate__(self, d):
        self.__dict__ = d
        # 重新导入 rjieba 模块，以便反序列化后能够继续使用中文分词功能
        import rjieba

        self.jieba = rjieba

    # 返回词汇表和额外添加的标记的字典表示
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)
    # 将输入文本 `text` 分词为 token 序列，支持使用结巴分词库
    def _tokenize(self, text, use_jieba=True):
        split_tokens = []
        if use_jieba:
            # 使用结巴分词器分词，不进行全模式切分
            for wholword in self.jieba.cut(text, False):
                if wholword in self.vocab:
                    # 如果分词结果在词汇表中，直接添加到分词列表中
                    split_tokens.append(wholword)
                else:
                    # 否则使用 bert 分词器进行进一步分词处理
                    char_list = self._tokenize(wholword, use_jieba=False)
                    split_tokens.extend(char_list)
        else:
            if self.do_basic_tokenize:
                # 如果需要进行基础分词处理，则使用基础分词器进行处理
                for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                    if token in self.basic_tokenizer.never_split:
                        # 如果 token 在不分割集合中，直接添加到分词列表中
                        split_tokens.append(token)
                    else:
                        # 否则使用 wordpiece 分词器进行进一步处理
                        split_tokens += self.wordpiece_tokenizer.tokenize(token)
            else:
                # 否则直接使用 wordpiece 分词器进行处理
                split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 将 token 转换为对应的 ID，使用 vocab 进行映射，未知 token 使用 unk_token 处理
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 将 ID 转换为对应的 token，使用 ids_to_tokens 进行映射，未知 ID 使用 unk_token 处理
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将 tokens 序列转换为单个字符串，移除特殊 token 标记 (" ##")
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建包含特殊 token 的模型输入序列，用于序列分类任务
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A RoFormer sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # 如果只有一个输入序列，添加起始 ([CLS]) 和结束 ([SEP]) 特殊 token
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 如果有两个输入序列，添加起始 ([CLS])，序列 1，中间分隔 ([SEP])，序列 2，以及结束 ([SEP]) 特殊 token
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 获取包含特殊 token 的遮蔽掩码，用于序列对任务
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
            # If the tokens already have special tokens, delegate to the superclass method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            # Generate a mask for sequence pairs with special tokens added
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # Generate a mask for a single sequence with special tokens added
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A RoFormer
        sequence pair mask has the following format:

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
        # Define special tokens for separation and classification
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        if token_ids_1 is None:
            # If only one sequence is provided, return a mask for the first sequence only
            return len(cls + token_ids_0 + sep) * [0]
        
        # If two sequences are provided, return a mask for both sequences concatenated with special tokens
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 定义一个方法用于保存词汇表到指定的目录和文件名前缀
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引为0，用于检查词汇表索引是否连续
        index = 0
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 如果目录存在，构造词汇表文件的完整路径，包括文件名前缀和默认的文件名
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 如果目录不存在，将目录作为文件名处理，构造完整的文件路径，包括文件名前缀
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开文件以写入模式，使用 UTF-8 编码
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的 token 和其对应的索引，按索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果当前索引不等于预期的索引，记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    # 更新索引为当前 token 的索引
                    index = token_index
                # 将 token 写入文件，并在末尾添加换行符
                writer.write(token + "\n")
                # 更新索引以确保连续性
                index += 1
        # 返回保存的词汇表文件路径
        return (vocab_file,)
```