# `.\transformers\models\mobilebert\tokenization_mobilebert.py`

```py
# coding=utf-8
#
# 版权 2020 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 依“原样”提供，不提供任何形式的担保或条件，
# 无论是明示的还是暗示的。
# 有关许可证的详细信息，请参阅许可证。
"""MobileBERT 的标记化类。"""


import collections
import os
import unicodedata
from typing import List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import logging


logger = logging.get_logger(__name__)

# 词汇文件名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"mobilebert-uncased": "https://huggingface.co/google/mobilebert-uncased/resolve/main/vocab.txt"}
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"mobilebert-uncased": 512}

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {}


# 从 transformers.models.bert.tokenization_bert.load_vocab 复制的函数
def load_vocab(vocab_file):
    """将词汇文件加载到字典中。"""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制的函数
def whitespace_tokenize(text):
    """对文本进行基本的空白字符清理和分割。"""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


# 从 transformers.models.bert.tokenization_bert.BertTokenizer 复制并修改为 MobileBERT 的类
class MobileBertTokenizer(PreTrainedTokenizer):
    r"""
    构造一个 MobileBERT 分词器。基于 WordPiece。

    此分词器继承自 [`PreTrainedTokenizer`]，其中包含大多数主要方法。用户应参考
    此超类以获取有关这些方法的更多信息。
```  
    # 这个类是一个用于创建tokenizer的配置类，它包含了用于初始化tokenizer的参数和默认值
    # 参数:
    #     vocab_file (`str`): 包含词汇表的文件。
    #     do_lower_case (`bool`, *optional*, defaults to `True`): 在分词时是否将输入变为小写。
    #     do_basic_tokenize (`bool`, *optional*, defaults to `True`): 在进行WordPiece分词之前是否进行基本分词。
    #     never_split (`Iterable`, *optional*): 在进行基本分词时不会被分割的标记的集合，只有在`do_basic_tokenize=True`时生效。
    #     unk_token (`str`, *optional*, defaults to `"[UNK]"`): 未知标记。词汇表中不存在的词将无法转换为ID，会被替换为该标记。
    #     sep_token (`str`, *optional*, defaults to `"[SEP]"`): 分隔符标记，用于构建多个序列的序列，例如序列分类或问题回答中的文本和问题的序列。同时也是使用特殊标记构建的序列的最后一个标记。
    #     pad_token (`str`, *optional*, defaults to `"[PAD]"`): 用于padding的标记，例如对不同长度的序列进行批处理。
    #     cls_token (`str`, *optional*, defaults to `"[CLS]"`): 分类器标记，在进行序列分类（整个序列的分类，而不是每个标记的分类）时使用。特殊标记构建的序列的第一个标记。
    #     mask_token (`str`, *optional*, defaults to `"[MASK]"`): 用于mask值的标记。这是在使用masked语言建模训练此模型时使用的标记。模型将尝试预测这个标记。
    #     tokenize_chinese_chars (`bool`, *optional*, defaults to `True`): 是否分词汉字。这对于日语可能需要禁用。
    #     strip_accents (`bool`, *optional*): 是否去除所有的重音符号。如果未指定此选项，则根据`lowercase`的值来决定是否去除重音（如原始MobileBERT中）。
    # """
    # 设置类变量，指定词汇表文件名、预训练词汇表文件映射、预训练初始化配置和最大模型输入尺寸映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
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
        # 如果词汇文件不存在，抛出值错误
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = MobileBertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇表
        self.vocab = load_vocab(vocab_file)
        # 构建从 ids 到 tokens 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 设置是否进行基本的分词操作
        self.do_basic_tokenize = do_basic_tokenize
        # 如果需要进行基本分词，则初始化 BasicTokenizer 类
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 初始化 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))
        
        # 调用父类的初始化方法，并传入相应参数
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
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    # 获取词汇表
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # 分词方法，将文本拆分成 tokens
    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        # 如果需要进行基本分词操作
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果 token 是不能拆分的特殊符号
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 使用 WordpieceTokenizer 进行 token 的拆分
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        # 否则直接使用 WordpieceTokenizer 进行拆分
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 将 token 转换为 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 将 id 转换为 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将 tokens 转换为字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建包含特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或序列对中构建模型输入的 token type IDs，通过连接和添加特殊的标记。
        MobileBERT 的序列具有以下格式：
        - 单个序列: `[CLS] X [SEP]`
        - 序列对: `[CLS] A [SEP] B [SEP]`

        参数:
            token_ids_0 (`List[int]`):
                将添加特殊标记的 ID 的列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于表示序列对。

        返回:
            `List[int]`: 包含适当的特殊标记的 [token type IDs](../glossary#token-type-ids) 的列表。
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的 token 列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        参数:
            token_ids_0 (`List[int]`):
                ID 的列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于表示序列对。
            already_has_special_tokens (`bool`, *optional*, 默认值为 `False`):
                token 列表是否已经格式化为模型的特殊标记。

        返回:
            `List[int]`: 一个整数列表，范围在 [0, 1] 之间：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A MobileBERT sequence
        pair mask has the following format:

        ```py
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
        # 定义 [SEP] 标记的 ID 列表，用于表示序列的分隔
        sep = [self.sep_token_id]
        # 定义 [CLS] 标记的 ID 列表，用于表示序列的开头
        cls = [self.cls_token_id]
        # 如果第二个序列的 ID 列表为 None，则返回只包含第一个序列的标记类型 ID 的列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回包含两个序列标记类型 ID 的列表，第一个序列的标记类型 ID 全部为 0，第二个序列的标记类型 ID 全部为 1
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引为 0
        index = 0
        # 如果保存目录已存在
        if os.path.isdir(save_directory):
            # 构建词汇表文件路径
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 如果保存目录不存在，则词汇表文件路径直接为保存目录路径
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇表文件，以 utf-8 编码写入
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表字典，按词汇表索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 如果当前索引与词汇表索引不相等，则发出警告
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    # 更新当前索引为词汇表索引
                    index = token_index
                # 写入词汇到文件中
                writer.write(token + "\n")
                # 更新索引
                index += 1
        # 返回保存的词汇表文件路径
        return (vocab_file,)
# 从transformers.models.bert.tokenization_bert中复制BasicTokenizer类
class BasicTokenizer(object):
    """
    构造一个BasicTokenizer，用于运行基本的标记化（标点符号拆分，转换为小写等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在标记化时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在标记化过程中永远不会被拆分的标记集合。仅在`do_basic_tokenize=True`时生效
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否在标记化过程中拆分中文字符。

            对于日文，应该禁用此选项（参见此[issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否删除所有重音符号。如果未指定此选项，则会由`lowercase`的值确定（与原始BERT相同）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号拆分，以便后续的标记化可以捕捉单词的完整上下文，比如缩略词。

    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    # 对文本进行基本的分词处理。对于子词分词，请参见WordPieceTokenizer。
    
    # 合并自定义的不分割词列表和已有的不分割词集合，如果不存在自定义的不分割词列表，则直接使用已有的不分割词集合
    never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
    # 清洗文本，去除一些特殊字符或格式
    text = self._clean_text(text)
    
    # 以下部分主要处理中文字符和unicode字符的一些处理，包括是否进行小写处理、是否去除重音符号以及中文字符的处理
    if self.tokenize_chinese_chars:
        text = self._tokenize_chinese_chars(text)
    # 将文本中的unicode字符进行标准化处理
    unicode_normalized_text = unicodedata.normalize("NFC", text)
    # 对标准化后的unicode文本进行空格分割处理
    orig_tokens = whitespace_tokenize(unicode_normalized_text)
    split_tokens = []
    # 循环处理每个分割后的token
    for token in orig_tokens:
        # 如果token不在不分割词集合中，则处理
        if token not in never_split:
            # 如果需要进行小写处理，则将token转换为小写
            if self.do_lower_case:
                token = token.lower()
                # 如果需要去除重音符号，则执行去除重音符号的处理
                if self.strip_accents is not False:
                    token = self._run_strip_accents(token)
            # 如果不需要小写处理，但需要去重音符号处理，则执行去重音符号处理
            elif self.strip_accents:
                token = self._run_strip_accents(token)
        # 将处理后的token加入到split_tokens列表中
        split_tokens.extend(self._run_split_on_punc(token, never_split))
    
    # 对分割后的token进行空格分割处理，并返回结果
    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens
    
    # 从文本中去除重音符号的处理函数
    def _run_strip_accents(self, text):
        # 将文本中的字符标准化为NFD格式
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            # 获取字符的unicode类型
            cat = unicodedata.category(char)
            # 如果类型为"Mn"，即重音符号，则跳过
            if cat == "Mn":
                continue
            # 否则将字符加入到结果列表中
            output.append(char)
        # 将处理后的字符列表连接为字符串并返回
        return "".join(output)
    # 在给定文本上运行分隔标点符号的操作
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号处分割文本，或者指定的文本在never_split中，直接返回原文本的列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号，将其作为单独的字符列表项添加到输出中，并标记开始一个新词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，且标记为开始一个新词，则将一个空列表添加到输出中
                if start_new_word:
                    output.append([])
                start_new_word = False
                # 将当前字符添加到输出的最后一个列表项中
                output[-1].append(char)
            i += 1

        # 将列表中的字符列表拼接成字符串，并返回字符串列表
        return ["".join(x) for x in output]

    # 对中文字符进行标记
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是中文字符，将其两侧添加空格并添加到输出中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    # 检查字符是否是中文字符
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 这里将“中文字符”定义为CJK Unicode块中的字符：
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # 注意，CJK Unicode块并不包含所有的日文和韩文字符，
        # 现代韩文Hangul字母是一个不同的块，
        # 日文平假名和片假名也是不同的块。
        # 这些字母用于书写以空格分隔的单词，因此它们不被特殊处理，处理方式与其他语言相同。
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

    # 对文本进行清理，删除无效字符并清理空白
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符为0或0xFFFD，或者是控制字符，跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果是空白字符，替换为空格；否则保留字符
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer复制而来
class WordpieceTokenizer(object):
    """运行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordPiece分词器，设置词汇表、未识别标记和每个词的最大输入字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将一段文本标记为其词片段。这使用贪婪的最长匹配算法来使用给定词汇表进行标记化。

        例如，`input = "unaffable"`会返回输出`["un", "##aff", "##able"]`。

        Args:
            text: 单个令牌或以空格分隔的令牌。这应该已经通过*BasicTokenizer*。

        Returns:
            一个词片段令牌列表。
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
```