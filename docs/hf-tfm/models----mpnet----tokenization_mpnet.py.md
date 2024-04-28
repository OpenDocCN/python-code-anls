# `.\transformers\models\mpnet\tokenization_mpnet.py`

```
# 设置文件编码为 UTF-8
# 版权声明和许可证信息
import collections  # 导入 collections 模块
import os  # 导入 os 模块
import unicodedata  # 导入 unicodedata 模块
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

from ...tokenization_utils import AddedToken, PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace  # 导入所需的 tokenizer 相关模块
from ...utils import logging  # 导入 logging 模块

logger = logging.get_logger(__name__)  # 获取 logger 对象

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 定义预训练模型所需的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/vocab.txt",  # MPNet 模型的词汇文件链接
    }
}

# 定义预训练模型的位置编码大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/mpnet-base": 512,  # MPNet 模型的位置编码大小
}

# 定义预训练模型的初始配置
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/mpnet-base": {"do_lower_case": True},  # MPNet 模型的初始配置
}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    # 创建一个空的有序字典
    vocab = collections.OrderedDict()
    # 打开词汇文件并逐行读取
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    # 将每个词汇及其索引添加到字典中
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")  # 去除末尾的换行符
        vocab[token] = index  # 将词汇及其索引添加到字典中
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()  # 去除文本的前后空格
    if not text:
        return []  # 如果文本为空则返回空列表
    tokens = text.split()  # 使用空格分割文本，得到词汇列表
    return tokens  # 返回词汇列表


class MPNetTokenizer(PreTrainedTokenizer):
    """
    This tokenizer inherits from [`BertTokenizer`] which contains most of the methods.
    Users should refer to the superclass for more information regarding methods.
    """

    # 定义词汇文件的名称、预训练模型所需的词汇文件映射、预训练模型的初始配置等属性
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化方法
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="[UNK]",
        pad_token="<pad>",
        mask_token="<mask>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
        ):
            # 如果 bos_token 是字符串，则将其转换为 AddedToken 对象，并标记为特殊 token
            bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
            # 如果 eos_token 是字符串，则将其转换为 AddedToken 对象，并标记为特殊 token
            eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
            # 如果 sep_token 是字符串，则将其转换为 AddedToken 对象，并标记为特殊 token
            sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
            # 如果 cls_token 是字符串，则将其转换为 AddedToken 对象，并标记为特殊 token
            cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
            # 如果 unk_token 是字符串，则将其转换为 AddedToken 对象，并标记为特殊 token
            unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
            # 如果 pad_token 是字符串，则将其转换为 AddedToken 对象，并标记为特殊 token
            pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token

            # 如果 mask_token 是字符串，则将其转换为 AddedToken 对象，并标记为特殊 token，并且在之前包含空格
            mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token

            # 如果给定的 vocab_file 不是文件路径，则引发 ValueError 异常
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
                    " model use `tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                )
            # 从给定的 vocab_file 加载词汇表
            self.vocab = load_vocab(vocab_file)
            # 创建一个从 token id 到 token 的有序字典
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
            # 是否执行基本的 tokenization
            self.do_basic_tokenize = do_basic_tokenize
            # 如果执行基本的 tokenization，则创建 BasicTokenizer 对象
            if do_basic_tokenize:
                self.basic_tokenizer = BasicTokenizer(
                    do_lower_case=do_lower_case,
                    never_split=never_split,
                    tokenize_chinese_chars=tokenize_chinese_chars,
                    strip_accents=strip_accents,
                )
            # 创建 WordpieceTokenizer 对象
            self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=str(unk_token))

            # 调用父类的构造函数初始化 tokenizer
            super().__init__(
                do_lower_case=do_lower_case,
                do_basic_tokenize=do_basic_tokenize,
                never_split=never_split,
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                sep_token=sep_token,
                cls_token=cls_token,
                pad_token=pad_token,
                mask_token=mask_token,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
                **kwargs,
            )

        @property
        def do_lower_case(self):
            # 返回是否执行小写化的标志
            return self.basic_tokenizer.do_lower_case

        @property
        def vocab_size(self):
            # 返回词汇表大小
            return len(self.vocab)

        def get_vocab(self):
            # "<mask>" 是词汇表的一部分，但是在快速保存版本中错误地添加到了错误的索引
            # 复制已添加 token 的编码器并更新词汇表
            vocab = self.added_tokens_encoder.copy()
            vocab.update(self.vocab)
            return vocab
    def _tokenize(self, text):
        # 用于将文本分词成标记的私有方法
        split_tokens = []
        # 如果需要进行基本的分词处理
        if self.do_basic_tokenize:
            # 使用基本分词器对文本进行分词处理，同时保留所有特殊标记
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                # 如果标记属于永不分割的特殊标记集合
                if token in self.basic_tokenizer.never_split:
                    # 将标记添加到分割标记列表中
                    split_tokens.append(token)
                else:
                    # 否则，使用 WordPiece 分词器对标记进行进一步分割
                    split_tokens += self.wordpiece_tokenizer.tokenize(token)
        else:
            # 如果不需要基本分词处理，则直接使用 WordPiece 分词器进行分词
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        # 返回分割后的标记列表
        return split_tokens

    def _convert_token_to_id(self, token):
        """将标记（字符串）转换为其在词汇表中的ID。"""
        # 使用词汇表获取标记对应的ID，如果标记不存在，则使用未知标记的ID
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """将索引（整数）转换为其在词汇表中的标记（字符串）。"""
        # 使用ID到标记的映射获取ID对应的标记，如果ID不存在，则使用未知标记
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """将一系列标记（字符串）转换为单个字符串。"""
        # 将标记列表连接成一个字符串，并去除其中的 ## 符号，然后去除首尾空白字符
        out_string = " ".join(tokens).replace(" ##", "").strip()
        # 返回拼接后的字符串
        return out_string

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊标记，从序列或序列对中构建用于序列分类任务的模型输入。一个 MPNet 序列的格式如下：

        - 单个序列：`
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
# 定义 BasicTokenizer 类，用于运行基本的分词（标点符号拆分，转化为小写等）
class BasicTokenizer(object):
    """
    构建一个 BasicTokenizer 实例，用于运行基本的分词（标点符号拆分，转化为小写等）。

    参数:
        do_lower_case (`bool`, *可选*, 默认为 `True`):
            是否在分词时将输入转换为小写。
        never_split (`Iterable`, *可选*):
            不应在分词过程中拆分的标记集合。仅在`do_basic_tokenize=True`时起作用
        tokenize_chinese_chars (`bool`, *可选*, 默认为 `True`):
            是否在分词中拆分中文字符。

            对于日文，可能应该关闭此选项（参见此问题
            [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *可选*):
            是否去除所有音调。如果没有指定此选项，那么将由`lowercase`的值决定（与原始BERT相同）。
        do_split_on_punc (`bool`, *可选*, 默认为 `True`):
            在某些情况下，我们希望跳过基本的标点符号拆分，以便后续的分词可以捕捉单词的完整上下文，例如缩略词。
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果 never_split 为 None，则设为一个空列表
        if never_split is None:
            never_split = []
        # 初始化各个属性
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.do_split_on_punc = do_split_on_punc
    # 定义一个方法，用于对给定文本进行基本的分词处理，用于生成单词片段。对于子词分词，请参见WordPieceTokenizer。
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.
    
        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 将传入的never_split参数与类属性never_split合并为一个新的集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本数据
        text = self._clean_text(text)
    
        # 当前部分是2018年11月1日为多语言和中文模型添加的内容，现在应用于英语模型，但不影响英语模型，因为英语模型没有在任何中文数据上训练
        if self.tokenize_chinese_chars:
            # 对中文字符进行分词处理
            text = self._tokenize_chinese_chars(text)
        # 防止将使用不同unicode代码点的同一字符视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空格分隔的文本进行tokenize
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    # 将token转换为小写
                    token = token.lower()
                    if self.strip_accents is not False:
                        # 根据strip_accents的值处理token
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    # 根据strip_accents的值处理token
                    token = self._run_strip_accents(token)
            # 对token进行拆分处理
            split_tokens.extend(self._run_split_on_punc(token, never_split))
    
        # 使用空格分隔的文本进行tokenize生成最终输出的tokens
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens
    
    # 方法用于去除文本中的重音符号
    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 对文本进行NFD规范化
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            # 判断字符是否为"Mark, Nonspacing"
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要在标点符号处分割或者该文本在不需要分割的列表中，直接返回原文本作为单词列表的一个元素
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换成字符列表
        chars = list(text)
        # 初始化索引和标志
        i = 0
        start_new_word = True
        # 初始化输出列表
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果是标点符号
            if _is_punctuation(char):
                # 将标点符号作为一个单独的列表加入输出列表
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号
                if start_new_word:
                    # 如果是一个新词的开始，则在输出列表中添加一个空列表
                    output.append([])
                start_new_word = False
                # 将字符加入当前词的列表中
                output[-1].append(char)
            i += 1

        # 将列表中的字符列表连接成字符串并返回
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 码点
            cp = ord(char)
            # 如果是中日韩字符
            if self._is_chinese_char(cp):
                # 在字符前后加入空格并添加到输出列表
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                # 如果不是中日韩字符，直接添加到输出列表
                output.append(char)
        # 将列表中的字符连接成字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 这里将 "中日韩字符" 定义为位于 CJK Unicode 块中的字符：
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # 注意，CJK Unicode 块并不包括所有的日语和韩语字符，
        # 韩文的现代 Hangul 字母是一个独立的块，
        # 日语的平假名和片假名也是如此。
        # 这些字母用于书写空格分隔的词语，因此不会被特别处理，
        # 而是像其他所有语言一样处理。
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
        # 初始化输出列表
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的 Unicode 码点
            cp = ord(char)
            # 如果字符是空字符或无效字符或控制字符，则跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则用空格替换
            if _is_whitespace(char):
                output.append(" ")
            else:
                # 否则直接添加到输出列表
                output.append(char)
        # 将列表中的字符连接成字符串并返回
        return "".join(output)
# 从transformers.models.bert.tokenization_bert.WordpieceTokenizer复制而来的类
class WordpieceTokenizer(object):
    """运行WordPiece标记化。"""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        # 初始化WordpieceTokenizer对象的词汇表、未知标记和每个单词的最大输入字符数
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        将一段文本标记化为它的WordPiece标记。这里使用贪婪的最长匹配算法来使用给定的词汇表进行标记化。

        例如，`input = "unaffable"`的输出将会是`["un", "##aff", "##able"]`。

        Args:
            text: 一个单独的标记或以空格分隔的标记。这应该已经通过*BasicTokenizer*。

        Returns:
            一个Wordpiece标记的列表。
        """

        output_tokens = []
        # 对每个标记进行标记化
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 如果标记长度超过了设定的最大输入字符数，则用未知标记替换
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