# `.\models\herbert\tokenization_herbert.py`

```py
# 导入必要的库和模块：json、os、re、unicodedata以及从typing模块导入List、Optional和Tuple
import json
import os
import re
import unicodedata
from typing import List, Optional, Tuple

# 从tokenization_utils中导入PreTrainedTokenizer、_is_control、_is_punctuation、_is_whitespace函数
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
# 从utils中导入logging模块
from ...utils import logging

# 获取logger对象用于日志记录
logger = logging.get_logger(__name__)

# 定义词汇文件的名称字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",   # 词汇表文件名
    "merges_file": "merges.txt",  # 合并文件名
}

# 预训练模型的词汇文件映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/vocab.json"
    },  # allegro/herbert-base-cased模型的词汇表下载地址
    "merges_file": {
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/merges.txt"
    },  # allegro/herbert-base-cased模型的合并文件下载地址
}

# 预训练模型的位置嵌入大小映射字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"allegro/herbert-base-cased": 514}

# 预训练模型初始化配置空字典
PRETRAINED_INIT_CONFIGURATION = {}

# 从transformers.models.xlm.tokenization_xlm中复制的函数：获取词中的符号对
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# 从transformers.models.xlm.tokenization_xlm中复制的函数：替换Unicode标点符号
def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    text = text.replace("，", ",")     # 替换中文逗号
    text = re.sub(r"。\s*", ". ", text)   # 替换中文句号并确保后面有空格
    text = text.replace("、", ",")     # 替换中文顿号
    text = text.replace("”", '"')      # 替换右双引号
    text = text.replace("“", '"')      # 替换左双引号
    text = text.replace("∶", ":")      # 替换中文分号
    text = text.replace("：", ":")      # 替换中文冒号
    text = text.replace("？", "?")      # 替换中文问号
    text = text.replace("《", '"')      # 替换中文书名号左
    text = text.replace("》", '"')      # 替换中文书名号右
    text = text.replace("）", ")")      # 替换右括号
    text = text.replace("！", "!")      # 替换中文感叹号
    text = text.replace("（", "(")      # 替换左括号
    text = text.replace("；", ";")      # 替换中文分号
    text = text.replace("１", "1")      # 替换全角数字1
    text = text.replace("」", '"')      # 替换中文引号右
    text = text.replace("「", '"')      # 替换中文引号左
    text = text.replace("０", "0")      # 替换全角数字0
    text = text.replace("３", "3")      # 替换全角数字3
    text = text.replace("２", "2")      # 替换全角数字2
    text = text.replace("５", "5")      # 替换全角数字5
    text = text.replace("６", "6")      # 替换全角数字6
    text = text.replace("９", "9")      # 替换全角数字9
    text = text.replace("７", "7")      # 替换全角数字7
    text = text.replace("８", "8")      # 替换全角数字8
    text = text.replace("４", "4")      # 替换全角数字4
    # 将全角句号后的空白替换为一个标准的英文句号加空格
    text = re.sub(r"．\s*", ". ", text)
    # 替换全角的波浪号为标准的波浪号
    text = text.replace("～", "~")
    # 替换单引号的全角形式为标准的单引号
    text = text.replace("’", "'")
    # 替换省略号的全角形式为标准的省略号
    text = text.replace("…", "...")
    # 替换全角的破折号为标准的破折号
    text = text.replace("━", "-")
    # 替换全角的左尖括号为标准的左尖括号
    text = text.replace("〈", "<")
    # 替换全角的右尖括号为标准的右尖括号
    text = text.replace("〉", ">")
    # 替换全角的左方括号为标准的左方括号
    text = text.replace("【", "[")
    # 替换全角的右方括号为标准的右方括号
    text = text.replace("】", "]")
    # 替换全角的百分号为标准的百分号
    text = text.replace("％", "%")
    # 返回处理后的文本
    return text
# 从transformers.models.xlm.tokenization_xlm.remove_non_printing_char复制而来
def remove_non_printing_char(text):
    """
    这个函数用于移除文本中的非打印字符。
    """
    output = []
    for char in text:
        # 获取字符的Unicode类别
        cat = unicodedata.category(char)
        # 如果字符的类别以"C"开头（表示控制字符），则跳过
        if cat.startswith("C"):
            continue
        # 否则将字符添加到输出列表中
        output.append(char)
    # 将列表中的字符连接成字符串并返回
    return "".join(output)


# 从transformers.models.bert.tokenization_bert.whitespace_tokenize复制而来
def whitespace_tokenize(text):
    """对文本进行基本的空白符号清理和分割。"""
    # 去除文本两端的空白符
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空白符分割文本，得到分词结果
    tokens = text.split()
    # 返回分词结果列表
    return tokens


# 从transformers.models.bert.tokenization_bert.BasicTokenizer复制而来
class BasicTokenizer(object):
    """
    构造一个BasicTokenizer对象，用于运行基本的分词（标点符号分割、小写化等）。

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在分词时将输入转换为小写。
        never_split (`Iterable`, *optional*):
            在分词时不会被拆分的token集合。仅在`do_basic_tokenize=True`时生效。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否对中文字符进行分词。

            对于日语，应该将其停用（见此问题）。
        strip_accents (`bool`, *optional*):
            是否去除所有的重音符号。如果未指定此选项，则将根据`lowercase`的值确定（与原始BERT相同）。
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便稍后的分词可以捕捉到单词的完整上下文，如缩写。
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
        # 是否在分词时转换为小写
        self.do_lower_case = do_lower_case
        # 在分词时不会被拆分的token集合
        self.never_split = set(never_split)
        # 是否对中文字符进行分词
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 是否去除所有的重音符号
        self.strip_accents = strip_accents
        # 是否进行标点符号分割
        self.do_split_on_punc = do_split_on_punc
    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. For sub-word tokenization, see WordPieceTokenizer.

        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # 使用 union() 方法将 self.never_split 和输入的 never_split 合并成一个新的集合，如果 never_split 为 None 则默认为空集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        
        # 清洁文本数据，处理特殊字符等
        text = self._clean_text(text)

        # 如果设置了 tokenize_chinese_chars 标志，对中文字符进行特殊处理，主要用于多语言和中文模型
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        
        # 使用 NFC 规范对文本进行 Unicode 规范化，主要是为了避免不同 Unicode 编码的相同字符被视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        
        # 使用空格分隔符进行基本的 tokenization，得到原始 token 列表
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        
        split_tokens = []
        # 遍历每个原始 token 进行处理
        for token in orig_tokens:
            # 如果 token 不在 never_split 中，则进行以下处理
            if token not in never_split:
                # 如果设置了 do_lower_case 标志，则将 token 转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果 strip_accents 不为 False，则移除 token 中的重音符号
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果 strip_accents 标志为 True，则移除 token 中的重音符号
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            
            # 使用 _run_split_on_punc 方法进行标点符号的拆分处理，并将结果添加到 split_tokens 中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空格分隔符重新组合 split_tokens 中的 token，得到最终的输出 token 列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 使用 NFD 规范对文本进行 Unicode 规范化，将重音符号分离出来
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取当前字符的 Unicode 分类
            cat = unicodedata.category(char)
            # 如果分类为 Mn（Mark, Nonspacing），则跳过当前字符，即跳过重音符号
            if cat == "Mn":
                continue
            # 将不包含重音符号的字符添加到 output 中
            output.append(char)
        # 将列表中的字符重新组合成字符串并返回
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果不需要根据标点符号分割文本，或者文本在never_split列表中，则直接返回原始文本作为列表的单个元素
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号，则将其作为新的列表项加入到输出列表中，并标记为开始一个新词
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                # 如果当前字符不是标点符号
                if start_new_word:
                    output.append([])  # 在输出列表中添加一个空列表作为新词的开始
                start_new_word = False
                output[-1].append(char)  # 将当前字符添加到当前词的列表项中
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            # 如果字符是CJK字符，则在其前后添加空格，并加入到输出列表中
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 检查给定的码点是否是CJK字符的码点范围内的值
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
            # 如果字符是无效字符或者控制字符，则跳过不处理
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，则将其替换为一个空格
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    # 定义一个名为 HerbertTokenizer 的类，继承自 PreTrainedTokenizer 类
    """
    Construct a BPE tokenizer for HerBERT.

    Peculiarities:

    - uses BERT's pre-tokenizer: BaseTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of a
      punctuation character will be treated separately.

    - Such pretokenized input is BPE subtokenized

    This tokenizer inherits from [`XLMTokenizer`] which contains most of the methods. Users should refer to the
    superclass for more information regarding methods.
    """

    # 定义类级别变量，指定各种文件的名称
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 初始化方法，构造函数
    def __init__(
        self,
        vocab_file,
        merges_file,
        tokenizer_file=None,
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sep_token="</s>",
        bos_token="<s>",
        do_lowercase_and_remove_accent=False,
        additional_special_tokens=[
            "<special0>",
            "<special1>",
            "<special2>",
            "<special3>",
            "<special4>",
            "<special5>",
            "<special6>",
            "<special7>",
            "<special8>",
            "<special9>",
        ],
        lang2id=None,
        id2lang=None,
        **kwargs,
    ):
    ):
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use HerbertTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        self.sm = sacremoses

        # cache of sm.MosesPunctNormalizer instance
        self.cache_moses_punct_normalizer = {}
        # cache of sm.MosesTokenizer instance
        self.cache_moses_tokenizer = {}
        self.lang_with_custom_tokenizer = {"zh", "th", "ja"}
        # True for current supported model (v1.2.0), False for XLM-17 & 100
        self.do_lowercase_and_remove_accent = do_lowercase_and_remove_accent
        self.lang2id = lang2id
        self.id2lang = id2lang
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)

        self.ja_word_tokenizer = None
        self.zh_word_tokenizer = None

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            lang2id=lang2id,
            id2lang=id2lang,
            do_lowercase_and_remove_accent=do_lowercase_and_remove_accent,
            tokenizer_file=None,
            **kwargs,
        )

        self.bert_pre_tokenizer = BasicTokenizer(
            do_lower_case=False,
            never_split=self.all_special_tokens,
            tokenize_chinese_chars=False,
            strip_accents=False,
        )

    @property
    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.do_lower_case
    def do_lower_case(self):
        # 返回当前对象的 do_lowercase_and_remove_accent 属性值
        return self.do_lowercase_and_remove_accent

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_punct_norm
    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            # 如果语言在缓存中不存在，则创建一个新的 MosesPunctNormalizer 实例
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            # 如果语言在缓存中已存在，则从缓存中获取 MosesPunctNormalizer 实例
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        # 使用 punct_normalizer 对文本进行标点符号规范化处理
        return punct_normalizer.normalize(text)

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_tokenize
    # 如果指定语言的 Moses 分词器不在缓存中
    if lang not in self.cache_moses_tokenizer:
        # 创建一个新的 Moses 分词器并添加到缓存中
        moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
        self.cache_moses_tokenizer[lang] = moses_tokenizer
    else:
        # 否则，从缓存中获取已存在的 Moses 分词器
        moses_tokenizer = self.cache_moses_tokenizer[lang]
    
    # 使用 Moses 分词器对文本进行分词处理，返回分词结果的列表
    return moses_tokenizer.tokenize(text, return_str=False, escape=False)

# 从 XLMTokenizer 中复制的方法，执行一系列的文本预处理步骤
def moses_pipeline(self, text, lang):
    # 替换文本中的 Unicode 标点符号
    text = replace_unicode_punct(text)
    # 对文本进行 Moses 标点符号规范化处理
    text = self.moses_punct_norm(text, lang)
    # 移除文本中的非打印字符
    text = remove_non_printing_char(text)
    # 返回处理后的文本
    return text

# 从 XLMTokenizer 中复制的方法，用于日语文本分词
def ja_tokenize(self, text):
    # 如果尚未初始化日语词汇分词器
    if self.ja_word_tokenizer is None:
        try:
            # 尝试导入 Mykytea 库并创建 Mykytea 对象
            import Mykytea
            self.ja_word_tokenizer = Mykytea.Mykytea(
                f"-model {os.path.expanduser('~')}/local/share/kytea/model.bin"
            )
        except (AttributeError, ImportError):
            # 如果导入失败，则记录错误信息并引发异常
            logger.error(
                "Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper"
                " (https://github.com/chezou/Mykytea-python) with the following steps"
            )
            logger.error("1. git clone git@github.com:neubig/kytea.git && cd kytea")
            logger.error("2. autoreconf -i")
            logger.error("3. ./configure --prefix=$HOME/local")
            logger.error("4. make && make install")
            logger.error("5. pip install kytea")
            raise
        
    # 使用日语词汇分词器对文本进行分词处理，返回分词结果的列表
    return list(self.ja_word_tokenizer.getWS(text))

@property
# 从 XLMTokenizer 中复制的属性，返回词汇表的大小
def vocab_size(self):
    return len(self.encoder)

# 从 XLMTokenizer 中复制的方法，返回词汇表的字典形式，包括添加的特殊标记
def get_vocab(self):
    return dict(self.encoder, **self.added_tokens_encoder)

# 从 XLMTokenizer 中复制的方法，用于 BPE（字节对编码）处理，但代码截断未完整提供
def bpe
    def bpe(self, token):
        # 将单词转换为 BPE 编码
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 如果缓存中已有该编码，直接返回缓存结果
        if token in self.cache:
            return self.cache[token]
        # 获取所有可能的符号对
        pairs = get_pairs(word)

        # 如果没有符号对，则直接返回原始单词加上结束符号
        if not pairs:
            return token + "</w>"

        # 开始迭代合并符号对
        while True:
            # 找到优先级最低的符号对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果找到的符号对不在预定义的符号对中，则停止合并
            if bigram not in self.bpe_ranks:
                break
            # 合并符号对
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果合并后的单词长度为1，则停止合并
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        # 将元组单词转换为字符串
        word = " ".join(word)
        # 替换特殊字符
        if word == "\n  </w>":
            word = "\n</w>"
        # 将结果存入缓存并返回
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        # 使用 BPE 对文本进行分词预处理
        pre_tokens = self.bert_pre_tokenizer.tokenize(text)

        split_tokens = []
        # 将每个预处理的 token 进行 BPE 分词处理
        for token in pre_tokens:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        return split_tokens

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将 token 转换为对应的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将 id 转换为对应的 token
        return self.decoder.get(index, self.unk_token)

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将一系列 token 转换为单个字符串
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        # 构建带有特殊 token 的输入
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate token type IDs (segment IDs) from a pair of token ID lists for sequence classification tasks. Each token
        ID list represents a sequence (or a pair of sequences).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs representing the second sequence in a pair.

        Returns:
            `List[int]`: List of token type IDs (segment IDs) where each element corresponds to a token in the input
            sequences. Typically, `0` is used for the first sequence and `1` for the second sequence in a pair.
        """

        # Define special tokens for beginning of sequence (BOS) and separator (SEP)
        bos = [self.bos_token_id]  # Get the ID of the beginning of sequence token
        sep = [self.sep_token_id]  # Get the ID of the separator token

        # Check if token_ids_1 is provided (indicating a pair of sequences)
        if token_ids_1 is None:
            # If only one sequence (token_ids_0), return token type IDs with BOS, sequence tokens, and SEP
            return [0] * len(bos + token_ids_0 + sep)
        
        # If two sequences are provided (token_ids_0 and token_ids_1), return token type IDs with BOS, sequence 1 tokens,
        # SEP, sequence 2 tokens, and SEP
        return [0] * len(bos + token_ids_0 + sep) + [1] * len(token_ids_1 + sep)
    # 返回用于序列对分类任务的序列对掩码。XLM 序列对掩码格式如下：
    #
    # ```
    # 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
    # | first sequence    | second sequence |
    # ```
    #
    # 如果 `token_ids_1` 是 `None`，则仅返回掩码的第一部分（全为0）。
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]  # 获取分隔符 token 的 ID
        cls = [self.cls_token_id]  # 获取类别标识符 token 的 ID
        if token_ids_1 is None:
            # 如果第二个序列的 token IDs 是 None，返回只包含第一个序列和分隔符的掩码（全为0）
            return len(cls + token_ids_0 + sep) * [0]
        else:
            # 否则返回两个序列加上分隔符的掩码，第一个序列部分为0，第二个序列部分为1
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.save_vocabulary 复制而来
    # 保存词汇表到指定的目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构建词汇表文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将词典编码器以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将 BPE 标记及其索引写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.__getstate__ 复制而来
    # 返回当前对象的状态，用于序列化
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sm"] = None  # 设置 sm 属性为 None
        return state

    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.__setstate__ 复制而来
    # 设置当前对象的状态，用于反序列化
    # 定义一个特殊方法 __setstate__，用于从序列化状态恢复对象的属性
    def __setstate__(self, d):
        # 将对象的 __dict__ 属性更新为给定的字典 d，用于恢复对象状态
        self.__dict__ = d

        # 尝试导入 sacremoses 库，用于处理文本的分词和正规化
        try:
            import sacremoses
        # 如果导入失败，抛出 ImportError 异常
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 如果导入成功，将 sacremoses 赋值给对象的属性 self.sm
        self.sm = sacremoses
```