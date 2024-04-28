# `.\models\funnel\tokenization_funnel.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 版本 2.0 许可证，您可以使用此文件，但需要符合许可证规定
# 您可以通过链接获取许可证的副本
# 在适用法律允许的情况下，软件按 "AS IS" 基础分发，没有任何明示或暗示的保证或条件
# 请查看许可证以了解具体语言规定和限制
""" Funnel Transformer 的分词类 """

# 导入模块
import collections
import os
import unicodedata
from typing import List, Optional, Tuple
# 从 tokenization_utils 模块中导入 PreTrainedTokenizer、_is_control、_is_punctuation、_is_whitespace
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 从工具包中导入 logging 模块
from ...utils import logging


# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义模型名称的列表
_model_names = [
    "small",
    "small-base",
    "medium",
    "medium-base",
    "intermediate",
    "intermediate-base",
    "large",
    "large-base",
    "xlarge",
    "xlarge-base",
]

# 定义预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/vocab.txt",
        "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/vocab.txt",
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/vocab.txt",
        "funnel-transformer/medium-base": (
            "https://huggingface.co/funnel-transformer/medium-base/resolve/main/vocab.txt"
        ),
        "funnel-transformer/intermediate": (
            "https://huggingface.co/funnel-transformer/intermediate/resolve/main/vocab.txt"
        ),
        "funnel-transformer/intermediate-base": (
            "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/vocab.txt"
        ),
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/vocab.txt",
        "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/vocab.txt",
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/vocab.txt",
        "funnel-transformer/xlarge-base": (
            "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/vocab.txt"
        ),
    }
}

# 定义预训练位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {f"funnel-transformer/{name}": 512 for name in _model_names}

# 定义预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {f"funnel-transformer/{name}": {"do_lower_case": True} for name in _model_names}


# 从 transformers.models.bert.tokenization_bert.load_vocab 复制的代码
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 使用有序字典保存词汇
    vocab = collections.OrderedDict()
    # 以 UTF-8 编码打开词汇文件
    with open(vocab_file, "r", encoding="utf-8") as reader:
        # 逐行读取词汇文件
        tokens = reader.readlines()
    # 遍历 tokens 列表，同时返回索引和元素
    for index, token in enumerate(tokens):
        # 去除 token 结尾的换行符
        token = token.rstrip("\n")
        # 将去除换行符的 token 添加到 vocab 字典中，索引为 index
        vocab[token] = index
    # 返回填充了 token 和对应索引的 vocab 字典
    return vocab
# 从transformers.models.bert.tokenization_bert.whitespace_tokenize复制而来的代码
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本两侧的空白字符
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 用空格分割文本，得到token列表
    tokens = text.split()
    return tokens


class FunnelTokenizer(PreTrainedTokenizer):
    r"""
    构建一个Funnel Transformer的分词器。基于WordPiece。

    这个分词器继承自PreTrainedTokenizer，包含大部分主要方法。用户应参考父类来了解这些方法更多信息。
    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在进行标记化时将输入转换为小写。
        do_basic_tokenize (`bool`, *optional*, defaults to `True`):
            是否在进行 WordPiece 标记化之前进行基本标记化。
        never_split (`Iterable`, *optional*):
            在标记化过程中永远不会被分割的标记集合。仅在`do_basic_tokenize=True`时有效。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。如果一个标记不在词汇表中，无法被转换为 ID，就会被设置为该标记。
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            分隔标记，在构建来自多个序列的序列时使用，例如用于序列分类或用于文本和问题的问答。它也用作带有特殊标记的序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如在批处理不同长度的序列时。
        cls_token (`str`, *optional*, defaults to `"<cls>"`):
            用于序列分类的分类器标记（整个序列而不是每个标记的分类）。它是使用特殊标记构建的序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            用于掩码值的标记。这是在使用掩码语言建模训练此模型时使用的标记。这是模型将尝试预测的标记。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            句子开头的标记。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            句子结尾的标记。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否标记化中文字符。

            这可能对日语应该停用（参见此[问题](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否删除所有重音符号。如果未指定此选项，则将由`lowercase`的值确定（与原始 BERT 相同）。
    """

    # 词汇表文件名字典
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练的词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 分类器标记类型 ID，默认为2
    cls_token_type_id: int = 2
    # 初始化方法，设置词汇文件路径，是否转化为小写，是否进行基本的分词，特殊词汇，中文字符分词，去除重音符号
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        bos_token="<s>",
        eos_token="</s>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 如果词汇文件不存在，抛出数值错误
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = FunnelTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # 加载词汇
        self.vocab = load_vocab(vocab_file)
        # 构建从 ids 到 tokens 的有序字典
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        # 是否进行基本分词
        self.do_basic_tokenize = do_basic_tokenize
        # 如果进行基本分词，创建基本分词器
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        # 创建 WordpieceTokenizer 对象
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

        # 调用父类的初始化方法
        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

    @property
    # 获取是否转化为小写的属性
    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.do_lower_case
    def do_lower_case(self):
        return self.basic_tokenizer.do_lower_case

    @property
    # 获取词汇数量的属性
    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.vocab_size
    def vocab_size(self):
        return len(self.vocab)

    # 获取词汇
    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer.get_vocab
    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    # Copied from transformers.models.bert.tokenization_bert.BertTokenizer._tokenize
    # 根据输入的文本进行分词，如果需要分割特殊标记则进行分割
    def _tokenize(self, text, split_special_tokens=False):
        split_tokens = []
        # 如果需要进行基本分词处理
        if self.do_basic_tokenize:
            # 对文本进行基本分词处理，并根据需要进行特殊标记分割
            for token in self.basic_tokenizer.tokenize(
                text, never_split=self.all_special_tokens if not split_special_tokens else None
            ):
                # 如果分词结果是不可分割的特殊标记，则直接添加到最终结果中
                if token in self.basic_tokenizer.never_split:
                    split_tokens.append(token)
                else:
                    # 否则将分词结果按照 wordpiece 进行分词处理
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果不需要进行基本分词处理，则直接使用 wordpiece 进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    # 通过词汇表将 token 转换为对应的 id
    # 复制自 transformers.models.bert.tokenization_bert.BertTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    # 通过词汇表将 id 转换为对应的 token
    # 复制自 transformers.models.bert.tokenization_bert.BertTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    # 将 tokens 转换为单个字符串
    # 复制自 transformers.models.bert.tokenization_bert.BertTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # 构建包含特殊标记的输入
    # 复制自 transformers.models.bert.tokenization_bert.BertTokenizer.build_inputs_with_special_tokens
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
    # 复制自 transformers.models.bert.tokenization_bert.BertTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的令牌列表中检索序列 ID。在使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                序列对的第二个 ID 列表（可选）。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                令牌列表是否已经使用特殊标记格式化为模型所需。

        Returns:
            `List[int]`: 在范围 [0, 1] 中的整数列表：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。Funnel Transformer 序列对掩码的格式如下：

        ```
        2 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | 第一个序列         | 第二个序列      |
        ```py

        如果 `token_ids_1` 是 `None`，则此方法只返回掩码的第一部分（0）。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                序列对的第二个 ID 列表（可选）。

        Returns:
            `List[int]`: 根据给定序列返回 [令牌类型 ID](../glossary#token-type-ids) 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0]
        return len(cls) * [self.cls_token_type_id] + len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 从 transformers.models.bert.tokenization_bert.BertTokenizer.save_vocabulary 复制过来的
    # 定义一个方法，用于保存词汇表到指定的目录和文件名前缀下
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 初始化索引值
        index = 0
        # 检查保存目录是否存在
        if os.path.isdir(save_directory):
            # 构建词汇文件路径，包括文件名前缀和常量中的词汇文件名
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            # 构建词汇文件路径，包括文件名前缀
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        # 打开词汇文件，以UTF-8编码写入
        with open(vocab_file, "w", encoding="utf-8") as writer:
            # 遍历词汇表中的 token 和对应的索引，按索引排序
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                # 检查索引是否连续，如果不连续则记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    # 更新索引值
                    index = token_index
                # 将 token 写入文件，并添加换行符
                writer.write(token + "\n")
                # 更新索引值
                index += 1
        # 返回保存的词汇文件路径
        return (vocab_file,)
# 从transformers.models.bert.tokenization_bert.BasicTokenizer复制而来，定义了BasicTokenizer类
class BasicTokenizer(object):
    """
    构造一个BasicTokenizer，用于执行基本的标记化（标点分割、小写化等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在标记化时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            无论何时都不会在标记化过程中分割的标记集合。仅在`do_basic_tokenize=True`时才会生效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否标记化中文字符。

            这应该对于日文被停用（参见这个 [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定此选项，则将由`lowercase`的值确定（与原始BERT相同）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本标点分割，以便后续标记化可以捕获单词的完整上下文，例如缩略词。
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果never_split未指定，将其设置为一个空列表
        if never_split is None:
            never_split = []
        # 初始化BasicTokenizer对象的属性
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)  # 将never_split转换为集合，以便快速检查标记是否应该永远不被分割
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    # 对文本进行基本的分词。有关子词分词，请参阅WordPieceTokenizer。
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 如果有指定不分割的词汇列表，则将其与默认的不分割词汇列表合并
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本数据
        text = self._clean_text(text)

        # 以下代码段是为多语言和中文模型添加的（2018年11月1日）。现在也应用于英文模型，
        # 但并不重要，因为英文模型没有在任何中文数据上训练，通常也不包含任何中文数据（因为维基百科
        # 中的英文维基百科确实包含一些中文词汇）。
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 对文本进行Unicode规范化，以避免将具有不同Unicode代码点的同一字符视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白分词函数对规范化后的文本进行分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历原始分词结果
        for token in orig_tokens:
            # 如果分词结果不在不分割的词汇列表中
            if token not in never_split:
                # 如果需要将文本全部转换为小写
                if self.do_lower_case:
                    # 将分词结果转换为小写
                    token = token.lower()
                    # 如果需要移除重音符号
                    if self.strip_accents is not False:
                        # 执行移除重音符号的操作
                        token = self._run_strip_accents(token)
                # 如果需要移除重音符号
                elif self.strip_accents:
                    # 执行移除重音符号的操作
                    token = self._run_strip_accents(token)
            # 将处理后的分词结果加入到split_tokens列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空白分词函数对拼接后的分词结果进行分词，得到最终的输出分词结果
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回输出分词结果
        return output_tokens

    # 从文本中移除重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 对文本进行Unicode规范化，以分解组合字符
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取当前字符的Unicode分类
            cat = unicodedata.category(char)
            # 如果当前字符为重音符号
            if cat == "Mn":
                # 跳过该字符
                continue
            # 将当前字符加入到输出列表中
            output.append(char)
        # 将输出列表中的字符拼接成字符串并返回
        return "".join(output)
    # 在给定文本上分割标点符号
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号处分割，或者文本在never_split列表中，则返回原始文本
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        chars = list(text)  # 将文本转换为字符列表
        i = 0  # 初始化索引
        start_new_word = True  # 初始化一个标志，表示是否开始一个新的单词
        output = []  # 初始化输出列表
        while i < len(chars):  # 循环遍历字符列表
            char = chars[i]  # 获取当前字符
            if _is_punctuation(char):  # 如果当前字符是标点符号
                output.append([char])  # 在输出列表中添加一个新的列表，用来存放标点符号
                start_new_word = True  # 设置标志为True，表示开始一个新的单词
            else:
                if start_new_word:  # 如果标志为True
                    output.append([])  # 在输出列表中添加一个新的空列表
                start_new_word = False  # 设置标志为False
                output[-1].append(char)  # 在输出列表的最后一个子列表中添加当前字符
            i += 1

        return ["".join(x) for x in output]  # 返回合并后的输出列表

    # 在中文字符周围添加空格
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []  # 初始化输出列表
        for char in text:  # 遍历文本中的每个字符
            cp = ord(char)  # 获取字符的Unicode码点
            if self._is_chinese_char(cp):  # 如果是中文字符
                output.append(" ")  # 在输出列表中添加空格
                output.append(char)  # 在输出列表中添加当前字符
                output.append(" ")  # 在输出列表中再次添加空格
            else:
                output.append(char)  # 在输出列表中添加当前字符
        return "".join(output)  # 返回合并后的输出列表

    # 检查字符是否是CJK字符
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 定义CJK字符的Unicode范围
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)
            or (cp >= 0x20000 and cp <= 0x2A6DF)
            or (cp >= 0x2A700 and cp <= 0x2B73F)
            or (cp >= 0x2B740 and cp <= 0x2B81F)
            or (cp >= 0x2B820 and cp <= 0x2CEAF)
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)
        ):
            return True  # 在CJK字符范围内，返回True
        else:
            return False  # 不在CJK字符范围内，返回False

    # 清理文本中的无效字符和空白符
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []  # 初始化输出列表
        for char in text:  # 遍历文本中的每个字符
            cp = ord(char)  # 获取字符的Unicode码点
            if cp == 0 or cp == 0xFFFD or _is_control(char):  # 如果是无效字符或控制字符
                continue  # 跳过当前字符
            if _is_whitespace(char):  # 如果是空白符
                output.append(" ")  # 在输出列表中添加空格
            else:
                output.append(char)  # 在输出列表中添加当前字符
        return "".join(output)  # 返回合并后的输出列表
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer复制而来的类
class WordpieceTokenizer(object):
    """运行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer类实例
        self.vocab = vocab  # 词汇表
        self.unk_token = unk_token  # 未知标记
        self.max_input_chars_per_word = max_input_chars_per_word  # 每个单词的最大输入字符数

    def tokenize(self, text):
        """
        将文本分词为其单词片段。这使用贪婪的最长匹配算法，使用给定的词汇表进行标记化。

        例如，`input = "unaffable"` 将返回输出 `["un", "##aff", "##able"]`。

        Args:
            text: 一个单个标记或以空格分隔的标记。这应该已经通过*BasicTokenizer*。

        Returns:
            一组wordpiece标记。
        """

        output_tokens = []  # 初始化输出token列表
        for token in whitespace_tokenize(text):  # 通过空格分隔文本并遍历每个单词片段
            chars = list(token)  # 将单词片段转换为字符列表
            if len(chars) > self.max_input_chars_per_word:  # 如果单词片段的字符数超过最大输入字符数
                output_tokens.append(self.unk_token)  # 将未知标记添加到输出token列表
                continue

            is_bad = False  # 初始化标记，表明是否存在无法标记的情况
            start = 0  # 记录起始位置
            sub_tokens = []  # 初始化子token列表
            while start < len(chars):  # 当起始位置小于字符列表的长度时执行循环
                end = len(chars)  # 结束位置等于字符列表的长度
                cur_substr = None  # 初始化当前子字符串
                while start < end:  # 当起始位置小于结束位置时执行循环
                    substr = "".join(chars[start:end])  # 将字符列表的子串连接起来
                    if start > 0:  # 如果起始位置大于0
                        substr = "##" + substr  # 在子串前添加"##"
                    if substr in self.vocab:  # 如果子串在词汇表中
                        cur_substr = substr  # 当前子字符串为该子串
                        break
                    end -= 1  # 结束位置减一
                if cur_substr is None:  # 如果当前子字符串为空
                    is_bad = True  # 设置标记为True
                    break
                sub_tokens.append(cur_substr)  # 将当前子字符串添加到子token列表
                start = end  # 起始位置更新为结束位置

            if is_bad:  # 如果存在无法标记的情况
                output_tokens.append(self.unk_token)  # 将未知标记添加到输出token列表
            else:  # 如果不存在无法标记的情况
                output_tokens.extend(sub_tokens)  # 将子token列表扩展到输出token列表
        return output_tokens  # 返回输出token列表
```