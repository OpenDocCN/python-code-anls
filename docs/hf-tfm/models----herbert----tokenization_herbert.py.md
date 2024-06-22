# `.\models\herbert\tokenization_herbert.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制
import json  # 导入 json 模块
import os  # 导入 os 模块
import re  # 导入 re 模块
import unicodedata  # 导入 unicodedata 模块
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace  # 导入相关模块和函数
from ...utils import logging  # 从 utils 模块中导入 logging 函数

logger = logging.get_logger(__name__)  # 获取当前模块的 logger

VOCAB_FILES_NAMES = {  # 定义词汇文件的名称字典
    "vocab_file": "vocab.json",  # 词汇文件名
    "merges_file": "merges.txt",  # 合并文件名
}

PRETRAINED_VOCAB_FILES_MAP = {  # 预训练词汇文件映射
    "vocab_file": {  # 词汇文件映射
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/vocab.json"
    },
    "merges_file": {  # 合并文件映射
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/merges.txt"
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"allegro/herbert-base-cased": 514}  # 预训练位置嵌入大小
PRETRAINED_INIT_CONFIGURATION = {}  # 预训练初始化配置为空字典

# 从 transformers.models.xlm.tokenization_xlm.get_pairs 复制的函数
def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()  # 创建空集合
    prev_char = word[0]  # 获取单词的第一个字符
    for char in word[1:]:  # 遍历单词的每个字符
        pairs.add((prev_char, char))  # 将前一个字符和当前字符作为一对添加到集合中
        prev_char = char  # 更新前一个字符为当前字符
    return pairs  # 返回符号对集合

# 从 transformers.models.xlm.tokenization_xlm.replace_unicode_punct 复制的函数
def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    text = text.replace("，", ",")  # 替换中文逗号为英文逗号
    text = re.sub(r"。\s*", ". ", text)  # 替换中文句号为英文句号
    text = text.replace("、", ",")  # 替换中文顿号为英文逗号
    text = text.replace("”", '"')  # 替换中文右双引号为英文双引号
    text = text.replace("“", '"')  # 替换中文左双引号为英文双引号
    text = text.replace("∶", ":")  # 替换中文冒号为英文冒号
    text = text.replace("：", ":")  # 替换中文冒号为英文冒号
    text = text.replace("？", "?")  # 替换中文问号为英文问号
    text = text.replace("《", '"')  # 替换中文左书名号为英文双引号
    text = text.replace("》", '"')  # 替换中文右书名号为英文双引号
    text = text.replace("）", ")")  # 替换中文右括号为英文右括号
    text = text.replace("！", "!")  # 替换中文感叹号为英文感叹号
    text = text.replace("（", "(")  # 替换中文左括号为英文左括号
    text = text.replace("；", ";")  # 替换中文分号为英文分号
    text = text.replace("１", "1")  # 替换全角数字1为半角数字1
    text = text.replace("」", '"')  # 替换中文右引号为英文双引号
    text = text.replace("「", '"')  # 替换中文左引号为英文双引号
    text = text.replace("０", "0")  # 替换全角数字0为半角数字0
    text = text.replace("３", "3")  # 替换全角数字3为半角数字3
    text = text.replace("２", "2")  # 替换全角数字2为半角数字2
    text = text.replace("５", "5")  # 替换全角数字5为半角数字5
    text = text.replace("６", "6")  # 替换全角数字6为半角数字6
    text = text.replace("９", "9")  # 替换全角数字9为半角数字9
    text = text.replace("７", "7")  # 替换全角数字7为半角数字7
    text = text.replace("８", "8")  # 替换全角数字8为半角数字8
    text = text.replace("４", "4")  # 替换全角数字4为半角数字4
    # 使用正则表达式将全角句号后的空格替换为半角句号和空格
    text = re.sub(r"．\s*", ". ", text)
    # 将全角波浪号替换为半角波浪号
    text = text.replace("～", "~")
    # 将全角右单引号替换为半角右单引号
    text = text.replace("’", "'")
    # 将全角省略号替换为半角省略号
    text = text.replace("…", "...")
    # 将全角破折号替换为半角破折号
    text = text.replace("━", "-")
    # 将全角左尖括号替换为半角左尖括号
    text = text.replace("〈", "<")
    # 将全角右尖括号替换为半角右尖括号
    text = text.replace("〉", ">")
    # 将全角左方括号替换为半角左方括号
    text = text.replace("【", "[")
    # 将全角右方括号替换为半角右方括号
    text = text.replace("】", "]")
    # 将全角百分号替换为半角百分号
    text = text.replace("％", "%")
    # 返回处理后的文本
    return text
# 从transformers.models.xlm.tokenization_xlm.remove_non_printing_char中复制过来的函数，用于移除非打印字符
def remove_non_printing_char(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    # 初始化一个空列表用于存储处理后的文本
    output = []
    # 遍历文本中的每个字符
    for char in text:
        # 获取字符的Unicode分类
        cat = unicodedata.category(char)
        # 如果字符的分类以"C"开头，表示为控制字符，跳过处理
        if cat.startswith("C"):
            continue
        # 将非控制字符添加到输出列表中
        output.append(char)
    # 将处理后的字符列表连接成字符串并返回
    return "".join(output)


# 从transformers.models.bert.tokenization_bert.whitespace_tokenize中复制过来的函数，用于基本的空格分词处理
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    # 去除文本两端的空格
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空格分割文本，得到分词结果
    tokens = text.split()
    return tokens


# 从transformers.models.bert.tokenization_bert.BasicTokenizer中复制过来的类，用于运行基本的分词处理（如标点符号分割、小写处理等）
class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
        do_split_on_punc (`bool`, *optional*, defaults to `True`):
            In some instances we want to skip the basic punctuation splitting so that later tokenization can capture
            the full context of the words, such as contractions.
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果never_split为None，则初始化为空列表
        if never_split is None:
            never_split = []
        # 初始化BasicTokenizer对象的属性
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
class HerbertTokenizer(PreTrainedTokenizer):
    """
    Construct a BPE tokenizer for HerBERT.

    Peculiarities:

    - uses BERT's pre-tokenizer: BaseTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of a
      punctuation character will be treated separately.

    - Such pretokenized input is BPE subtokenized

    This tokenizer inherits from [`XLMTokenizer`] which contains most of the methods. Users should refer to the
    superclass for more information regarding methods.
    """

    # 定义类属性，指定词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义类属性，指定预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义类属性，指定预训练初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 定义类属性，指定最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # 初始化方法
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
        # 尝试导入sacremoses库
        try:
            import sacremoses
        except ImportError:
            # 如果导入失败，抛出ImportError异常，提示需要安装sacremoses库
            raise ImportError(
                "You need to install sacremoses to use HerbertTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 将sacremoses库赋值给self.sm
        self.sm = sacremoses

        # 缓存sm.MosesPunctNormalizer实例
        self.cache_moses_punct_normalizer = {}
        # 缓存sm.MosesTokenizer实例
        self.cache_moses_tokenizer = {}
        # 包含自定义分词器的语言集合
        self.lang_with_custom_tokenizer = {"zh", "th", "ja"}
        # 当前支持的模型（v1.2.0）为True，XLM-17 & 100为False
        self.do_lowercase_and_remove_accent = do_lowercase_and_remove_accent
        self.lang2id = lang2id
        self.id2lang = id2lang
        # 如果lang2id和id2lang都不为None，则断言它们的长度相等
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)

        self.ja_word_tokenizer = None
        self.zh_word_tokenizer = None

        # 从vocab_file中读取编码器内容
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 根据encoder创建解码器
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 从merges_file中读取合并内容
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        # 创建bpe_ranks字典，将合并内容与索引对应
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

        # 调用父类的初始化方法，传入各种特殊token和参数
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

        # 创建BasicTokenizer实例，用于BERT预处理
        self.bert_pre_tokenizer = BasicTokenizer(
            do_lower_case=False,
            never_split=self.all_special_tokens,
            tokenize_chinese_chars=False,
            strip_accents=False,
        )

    @property
    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.do_lower_case中复制
    def do_lower_case(self):
        # 返回do_lowercase_and_remove_accent的值
        return self.do_lowercase_and_remove_accent

    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_punct_norm中复制
    def moses_punct_norm(self, text, lang):
        # 如果lang不在缓存中，则创建一个新的MosesPunctNormalizer实例
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            # 否则从缓存中获取MosesPunctNormalizer实例
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        # 返回规范化后的文本
        return punct_normalizer.normalize(text)

    # 从transformers.models.xlm.tokenization_xlm.XLMTokenizer.moses_tokenize中复制
    # 使用 Moses 分词器对文本进行分词处理
    def moses_tokenize(self, text, lang):
        # 如果指定语言的 Moses 分词器不在缓存中，则创建并存储
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            # 如果已经存在于缓存中，则直接使用缓存中的 Moses 分词器
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        # 调用 Moses 分词器的 tokenize 方法对文本进行分词处理
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    # 从 XLMTokenizer 中复制的 Moses 处理流程
    def moses_pipeline(self, text, lang):
        # 替换文本中的 Unicode 标点符号
        text = replace_unicode_punct(text)
        # 对文本进行 Moses 标点规范化处理
        text = self.moses_punct_norm(text, lang)
        # 移除文本中的非打印字符
        text = remove_non_printing_char(text)
        return text

    # 从 XLMTokenizer 中复制的日语分词方法
    def ja_tokenize(self, text):
        # 如果日语分词器尚未初始化，则尝试导入 Mykytea 库并初始化分词器
        if self.ja_word_tokenizer is None:
            try:
                import Mykytea

                self.ja_word_tokenizer = Mykytea.Mykytea(
                    f"-model {os.path.expanduser('~')}/local/share/kytea/model.bin"
                )
            except (AttributeError, ImportError):
                # 如果导入失败，则输出错误信息并抛出异常
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
        # 调用日语分词器的 getWS 方法对文本进行分词处理并返回结果列表
        return list(self.ja_word_tokenizer.getWS(text))

    @property
    # 获取词汇表大小
    def vocab_size(self):
        return len(self.encoder)

    # 获取词汇表
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 从 XLMTokenizer 中复制的 BPE 处理方法
    # 对输入的 token 进行 BPE 编码
    def bpe(self, token):
        # 将 token 转换为元组形式，并在最后一个字符后添加 "</w>"
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 获取 token 的所有可能的 bigram 组合
        pairs = get_pairs(word)

        # 如果没有 bigram 组合，则直接返回 token 加上结束符 "</w>"
        if not pairs:
            return token + "</w>"

        # 循环处理 bigram 组合
        while True:
            # 找到当前最小的 bigram，并根据其在 bpe_ranks 中的值进行排序
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果当前 bigram 不在 bpe_ranks 中，则跳出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 处理 word 中的字符，根据 bigram 进行合并
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
            # 如果 word 的长度为 1，则跳出循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将 word 转换为字符串形式
        word = " ".join(word)
        # 如果 word 为特殊字符，则替换为指定字符
        if word == "\n  </w>":
            word = "\n</w>"
        # 将结果存入缓存并返回
        self.cache[token] = word
        return word

    # 对文本进行分词处理
    def _tokenize(self, text):
        # 使用 BERT 预处理分词器对文本进行预处理
        pre_tokens = self.bert_pre_tokenizer.tokenize(text)

        split_tokens = []
        # 遍历预处理后的 token，使用 bpe 方法进行分词
        for token in pre_tokens:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        return split_tokens

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    # 将一系列 token 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

    # 构建带有特殊标记的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一个序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。XLM 序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 序列对: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表。

        Returns:
            `List[int]`: 具有适当特殊标记的输入 ID 列表。

        """
        bos = [self.bos_token_id]  # 开始标记的 ID 列表
        sep = [self.sep_token_id]  # 分隔标记的 ID 列表

        if token_ids_1 is None:
            return bos + token_ids_0 + sep  # 如果没有第二个序列，返回连接后的 ID 列表
        return bos + token_ids_0 + sep + token_ids_1 + sep  # 如果有第二个序列，返回连接后的 ID 列表

    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.get_special_tokens_mask 复制而来
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                序列对的第二个 ID 列表。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    # 从 transformers.models.xlm.tokenization_xlm.XLMTokenizer.create_token_type_ids_from_sequences 复制而来
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLM sequence
        pair mask has the following format:

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
            `List[int`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Define the separator and classification tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # If token_ids_1 is None, return a mask with only the first portion (0s)
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # Return a mask with both sequences separated by 0s and 1s
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Check if save_directory is a valid directory
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # Define the paths for vocabulary and merges files
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        
        # Save vocabulary to vocab_file
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # Save merges to merge_file
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

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.__getstate__
    def __getstate__(self):
        # Create a copy of the object's state and set 'sm' attribute to None
        state = self.__dict__.copy()
        state["sm"] = None
        return state

    # Copied from transformers.models.xlm.tokenization_xlm.XLMTokenizer.__setstate__
    # 定义特殊方法 __setstate__，用于设置对象的状态
    def __setstate__(self, d):
        # 将对象的属性字典设置为传入的参数字典
        self.__dict__ = d

        # 尝试导入 sacremoses 模块
        try:
            import sacremoses
        # 如果导入失败，抛出 ImportError 异常
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 将导入的 sacremoses 模块赋值给对象的属性 sm
        self.sm = sacremoses
```