# `.\transformers\models\xlm\tokenization_xlm.py`

```py
# Tokenization classes for XLM.
用于XLM的标记化类。


import json
import os
import re
import sys
import unicodedata
from typing import List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

导入所需的库和模块。


logger = logging.get_logger(__name__)

获取logger对象。


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

定义词汇文件和合并文件的文件名。


PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "xlm-mlm-en-2048": "https://huggingface.co/xlm-mlm-en-2048/resolve/main/vocab.json",
        "xlm-mlm-ende-1024": "https://huggingface.co/xlm-mlm-ende-1024/resolve/main/vocab.json",
        "xlm-mlm-enfr-1024": "https://huggingface.co/xlm-mlm-enfr-1024/resolve/main/vocab.json",
        "xlm-mlm-enro-1024": "https://huggingface.co/xlm-mlm-enro-1024/resolve/main/vocab.json",
        "xlm-mlm-tlm-xnli15-1024": "https://huggingface.co/xlm-mlm-tlm-xnli15-1024/resolve/main/vocab.json",
        "xlm-mlm-xnli15-1024": "https://huggingface.co/xlm-mlm-xnli15-1024/resolve/main/vocab.json",
        "xlm-clm-enfr-1024": "https://huggingface.co/xlm-clm-enfr-1024/resolve/main/vocab.json",
        "xlm-clm-ende-1024": "https://huggingface.co/xlm-clm-ende-1024/resolve/main/vocab.json",
        "xlm-mlm-17-1280": "https://huggingface.co/xlm-mlm-17-1280/resolve/main/vocab.json",
        "xlm-mlm-100-1280": "https://huggingface.co/xlm-mlm-100-1280/resolve/main/vocab.json",
    },

定义预训练模型的词汇文件映射。

以上是给定代码的注释。
    # merges_file 字典包含了多个键值对，每个键表示一个模型名称，每个值表示对应模型的 merges.txt 文件的下载链接
    "merges_file": {
        "xlm-mlm-en-2048": "https://huggingface.co/xlm-mlm-en-2048/resolve/main/merges.txt",  # 模型 xlm-mlm-en-2048 对应的 merges.txt 文件的下载链接
        "xlm-mlm-ende-1024": "https://huggingface.co/xlm-mlm-ende-1024/resolve/main/merges.txt",  # 模型 xlm-mlm-ende-1024 对应的 merges.txt 文件的下载链接
        "xlm-mlm-enfr-1024": "https://huggingface.co/xlm-mlm-enfr-1024/resolve/main/merges.txt",  # 模型 xlm-mlm-enfr-1024 对应的 merges.txt 文件的下载链接
        "xlm-mlm-enro-1024": "https://huggingface.co/xlm-mlm-enro-1024/resolve/main/merges.txt",  # 模型 xlm-mlm-enro-1024 对应的 merges.txt 文件的下载链接
        "xlm-mlm-tlm-xnli15-1024": "https://huggingface.co/xlm-mlm-tlm-xnli15-1024/resolve/main/merges.txt",  # 模型 xlm-mlm-tlm-xnli15-1024 对应的 merges.txt 文件的下载链接
        "xlm-mlm-xnli15-1024": "https://huggingface.co/xlm-mlm-xnli15-1024/resolve/main/merges.txt",  # 模型 xlm-mlm-xnli15-1024 对应的 merges.txt 文件的下载链接
        "xlm-clm-enfr-1024": "https://huggingface.co/xlm-clm-enfr-1024/resolve/main/merges.txt",  # 模型 xlm-clm-enfr-1024 对应的 merges.txt 文件的下载链接
        "xlm-clm-ende-1024": "https://huggingface.co/xlm-clm-ende-1024/resolve/main/merges.txt",  # 模型 xlm-clm-ende-1024 对应的 merges.txt 文件的下载链接
        "xlm-mlm-17-1280": "https://huggingface.co/xlm-mlm-17-1280/resolve/main/merges.txt",  # 模型 xlm-mlm-17-1280 对应的 merges.txt 文件的下载链接
        "xlm-mlm-100-1280": "https://huggingface.co/xlm-mlm-100-1280/resolve/main/merges.txt",  # 模型 xlm-mlm-100-1280 对应的 merges.txt 文件的下载链接
    },
}

# 预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "xlm-mlm-en-2048": 512,
    "xlm-mlm-ende-1024": 512,
    "xlm-mlm-enfr-1024": 512,
    "xlm-mlm-enro-1024": 512,
    "xlm-mlm-tlm-xnli15-1024": 512,
    "xlm-mlm-xnli15-1024": 512,
    "xlm-clm-enfr-1024": 512,
    "xlm-clm-ende-1024": 512,
    "xlm-mlm-17-1280": 512,
    "xlm-mlm-100-1280": 512,
}

# 预训练初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "xlm-mlm-en-2048": {"do_lowercase_and_remove_accent": True},
    "xlm-mlm-ende-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {0: "de", 1: "en"},
        "lang2id": {"de": 0, "en": 1},
    },
    "xlm-mlm-enfr-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {0: "en", 1: "fr"},
        "lang2id": {"en": 0, "fr": 1},
    },
    "xlm-mlm-enro-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {0: "en", 1: "ro"},
        "lang2id": {"en": 0, "ro": 1},
    },
    "xlm-mlm-tlm-xnli15-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {
            0: "ar",
            1: "bg",
            2: "de",
            3: "el",
            4: "en",
            5: "es",
            6: "fr",
            7: "hi",
            8: "ru",
            9: "sw",
            10: "th",
            11: "tr",
            12: "ur",
            13: "vi",
            14: "zh",
        },
        "lang2id": {
            "ar": 0,
            "bg": 1,
            "de": 2,
            "el": 3,
            "en": 4,
            "es": 5,
            "fr": 6,
            "hi": 7,
            "ru": 8,
            "sw": 9,
            "th": 10,
            "tr": 11,
            "ur": 12,
            "vi": 13,
            "zh": 14,
        },
    },
    "xlm-mlm-xnli15-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {
            0: "ar",
            1: "bg",
            2: "de",
            3: "el",
            4: "en",
            5: "es",
            6: "fr",
            7: "hi",
            8: "ru",
            9: "sw",
            10: "th",
            11: "tr",
            12: "ur",
            13: "vi",
            14: "zh",
        },
        "lang2id": {
            "ar": 0,
            "bg": 1,
            "de": 2,
            "el": 3,
            "en": 4,
            "es": 5,
            "fr": 6,
            "hi": 7,
            "ru": 8,
            "sw": 9,
            "th": 10,
            "tr": 11,
            "ur": 12,
            "vi": 13,
            "zh": 14,
        },
    },
    "xlm-clm-enfr-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {0: "en", 1: "fr"},
        "lang2id": {"en": 0, "fr": 1},
    },
    "xlm-clm-ende-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {0: "de", 1: "en"},
        "lang2id": {"de": 0, "en": 1},
    },
    # 定义一个字典，键为模型名称 "xlm-mlm-17-1280"
    # 对应值为另一个字典，包含了几个键值对
    "xlm-mlm-17-1280": {
        # 是否执行小写处理和去除重音符号的操作，默认为 False
        "do_lowercase_and_remove_accent": False,
        # 一个字典，将语言ID映射到语言代码的字符串
        "id2lang": {
            # 语言ID 0 对应的语言代码是 "ar"（阿拉伯语）
            0: "ar",
            # 语言ID 1 对应的语言代码是 "de"（德语）
            1: "de",
            # 语言ID 2 对应的语言代码是 "en"（英语）
            2: "en",
            # 以此类推，映射了多种语言
        },
        # 另一个字典，将语言代码的字符串映射到语言ID
        "lang2id": {
            # 语言代码 "ar"（阿拉伯语）对应的语言ID是 0
            "ar": 0,
            # 语言代码 "de"（德语）对应的语言ID是 1
            "de": 1,
            # 语言代码 "en"（英语）对应的语言ID是 2
            # 以此类推，映射了多种语言
        },
    },
# 定义一个函数，用于获取单词中的符号对
def get_pairs(word):
    """
    返回单词中的符号对集合。单词以元组形式表示，元组中的符号可以是变长字符串。
    """
    # 创建一个空集合，用于存储符号对
    pairs = set()
    # 获取单词的第一个字符
    prev_char = word[0]
    # 遍历单词中的字符（从第二个字符开始）
    for char in word[1:]:
        # 将前一个字符和当前字符组成的元组加入到集合中
        pairs.add((prev_char, char))
        # 将当前字符赋值给前一个字符，以供下一个循环使用
        prev_char = char
    # 返回符号对集合
    return pairs


# 定义一个函数，将文本小写化并去除重音符
def lowercase_and_remove_accent(text):
    """
    将文本小写化并去除重音符，参考 https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
    """
    # 将文本按空格连接成一个字符串
    text = " ".join(text)
    # 将文本转换为小写
    text = text.lower()
    # 使用 NFD 正规化形式，分解字符以去除重音符
    text = unicodedata.normalize("NFD", text)
    # 创建一个空列表，用于存储处理后的字符
    output = []
    # 遍历文本中的字符
    for char in text:
        # 获取字符的 Unicode 类别
        cat = unicodedata.category(char)
        # 如果字符是非标记字符（Mn 表示非间隔符号），则跳过
        if cat == "Mn":
            continue
        # 将字符加入到输出列表中
        output.append(char)
    # 将输出列表转换为字符串，转换为小写，并根据空格拆分为单词列表
    return "".join(output).lower().split(" ")


# 定义一个函数，用于替换 Unicode 标点符号
def replace_unicode_punct(text):
    """
    对 https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl 的翻译
    """
    # 替换中文逗号为英文逗号
    text = text.replace("，", ",")
    # 使用正则表达式替换句号，并在其后添加空格
    text = re.sub(r"。\s*", ". ", text)
    # 替换各种 Unicode 标点符号
    text = text.replace("、", ",")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("∶", ":")
    text = text.replace("：", ":")
    text = text.replace("？", "?")
    text = text.replace("《", '"')
    text = text.replace("》", '"')
    text = text.replace("）", ")")
    text = text.replace("！", "!")
    text = text.replace("（", "(")
    text = text.replace("；", ";")
    text = text.replace("１", "1")
    text = text.replace("」", '"')
    text = text.replace("「", '"')
    text = text.replace("０", "0")
    text.replace("３", "3")
    text.replace("２", "2")
    text.replace("５", "5")
    text.replace("６", "6")
    text.replace("９", "9")
    text替换"７"、"８"、"４"
    # 使用正则表达式替换句号，并在其后添加空格
    text = re.sub(r"．\s*", ". ", text)
    text = text.replace("～", "~")
    text = text.replace("’", "'")
    text = text.replace("…", "...")
    text = text.replace("━", "-")
    text替换 "〈" 与 "〉"
    text.replace("【", "[")
    text.replace("】", "]")
    text.replace "％"
    # 返回处理后的文本
    return text


# 定义一个函数，用于删除不可打印的字符
def remove_non_printing_char(text):
    """
    对 https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl 的翻译
    """
    # 创建一个空列表，用于存储处理后的字符
    output = []
    # 遍历文本中的字符
    for char in text:
        # 获取字符的 Unicode 类别
        cat = unicodedata.category(char)
        # 如果字符类别以 "C" 开头，则跳过（不可打印字符）
        if cat.startswith("C"):
            continue
        # 将字符加入到输出列表中
        output.append(char)
    # 返回处理后的文本，转换为字符串
    return "".join(output)


# 定义一个函数，用于罗马尼亚语的预处理
def romanian_preprocessing(text):
    """Sennrich 的 WMT16 脚本，用于罗马尼亚语的预处理，使用于模型 `xlm-mlm-enro-1024`"""
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/normalise-romanian.py
    # 将特定的罗马尼亚语字符替换为其他字符
    text = text.replace("\u015e", "\u0218").replace("\u015f", "\u0219")
    text = text.replace("\u0162", "\u021a").replace("\u0163", "\u021b")
    # 返回处理后的文本
    return text
    # 替换文本中的特殊字符，将 S-comma 替换成 S，s-comma 替换成 s
    text = text.replace("\u0218", "S").replace("\u0219", "s")  # s-comma
    # 将 T-comma 替换成 T，t-comma 替换成 t
    text = text.replace("\u021a", "T").replace("\u021b", "t")  # t-comma
    # 将 Ă 替换成 A，ă 替换成 a
    text = text.replace("\u0102", "A").replace("\u0103", "a")
    # 将 Â 替换成 A，â 替换成 a
    text = text.replace("\u00C2", "A").replace("\u00E2", "a")
    # 将 Î 替换成 I，î 替换成 i
    text = text.replace("\u00CE", "I").replace("\u00EE", "i")
    # 返回替换后的文本
    return text
# XLMTokenizer 类是一个继承自 PreTrainedTokenizer 的类，用于构造 XLM tokenizer。
class XLMTokenizer(PreTrainedTokenizer):
    """
    Construct an XLM tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    # 构造一个 XLM tokenizer，基于 Byte-Pair Encoding。其 Tokenization 过程如下所示：
    
    - Moses preprocessing and tokenization for most supported languages.
    - 对于大多数支持的语言，使用 Moses 进行预处理和分词。
    
    - Language specific tokenization for Chinese (Jieba), Japanese (KyTea) and Thai (PyThaiNLP).
    - 对于中文（使用 Jieba）、日文（使用 KyTea）和泰文（使用 PyThaiNLP）等特定语言，进行特定的分词处理。
    
    - Optionally lowercases and normalizes all inputs text.
    - 可选的将所有输入文本转换为小写，并进行归一化处理。
    
    - The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like "__classify__") to a vocabulary.
    - 参数 special_tokens 和函数 set_special_tokens 可以用于向词汇表中添加额外的符号（比如 "__classify__"）。
    
    - The `lang2id` attribute maps the languages supported by the model with their IDs if provided (automatically set for pretrained vocabularies).
    - 如果提供了与语言对应的 ID，`lang2id` 属性会将模型支持的语言与其对应的 ID 进行映射（预训练的词汇表会自动设置该属性）。
    
    - The `id2lang` attributes does reverse mapping if provided (automatically set for pretrained vocabularies).
    - 如果提供了与 ID 对应的语言，`id2lang` 属性会进行反向映射（预训练的词汇表会自动设置该属性）。

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    
    # 该 tokenizer 类继承自 [`PreTrainedTokenizer`]，该父类中包含了大多数主要的方法。用户应该参考该父类以了解有关这些方法的更多信息。
    """
    Args:
        vocab_file (`str`):
            Vocabulary file. # 词汇表文件名
        merges_file (`str`):
            Merges file. # 合并文件
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. # 未知标记
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip> # 序列开始标记
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens. # 分隔符标记
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths. # 填充标记
        cls_token (`str`, *optional*, defaults to `"</s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens. # 分类器标记
        mask_token (`str`, *optional*, defaults to `"<special1>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict. # 掩码标记
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']`):
            List of additional special tokens. # 附加的特殊标记列表
        lang2id (`Dict[str, int]`, *optional*):
            Dictionary mapping languages string identifiers to their IDs. # 语言标识符到ID的映射字典
        id2lang (`Dict[int, str]`, *optional*):
            Dictionary mapping language IDs to their string identifiers. # 语言ID到标识符的映射字典
        do_lowercase_and_remove_accent (`bool`, *optional*, defaults to `True`):
            Whether to lowercase and remove accents when tokenizing. # 在标记化时是否转换为小写并移除重音符号
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义 XLMTokenizer 类的初始化方法
    def __init__(
        self,
        vocab_file, # 词汇表文件路径
        merges_file, # BPE 合并词文件路径
        unk_token="<unk>", # 未知标记
        bos_token="<s>", # 句子开始标记
        sep_token="</s>", # 句子结束标记
        pad_token="<pad>", # 填充标记
        cls_token="</s>", # 分类标记
        mask_token="<special1>", # 掩码标记
        additional_special_tokens=[ # 其他特殊标记
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
        lang2id=None, # 语言到 ID 的映射
        id2lang=None, # ID 到语言的映射
        do_lowercase_and_remove_accent=True, # 是否进行小写转换和去重音
        **kwargs,
    ):
        # 导入 sacremoses 库，用于处理文本
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )
        
        self.sm = sacremoses
    
        # 缓存 Moses 标点符号归一化和分词实例
        self.cache_moses_punct_normalizer = {}
        self.cache_moses_tokenizer = {}
        
        # 设置自定义分词的语言
        self.lang_with_custom_tokenizer = {"zh", "th", "ja"}
        
        # 设置是否进行小写转换和去重音
        self.do_lowercase_and_remove_accent = do_lowercase_and_remove_accent
        
        # 设置语言到 ID 和 ID 到语言的映射
        self.lang2id = lang2id
        self.id2lang = id2lang
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)
    
        # 初始化 Japanese 和 Chinese 分词器
        self.ja_word_tokenizer = None
        self.zh_word_tokenizer = None
    
        # 读取词汇表和 BPE 合并词
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        
        # 初始化缓存
        self.cache = {}
        
        # 调用父类的初始化方法
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
            **kwargs,
        )
    
    # 返回是否进行小写转换和去重音的标志
    @property
    def do_lower_case(self):
        return self.do_lowercase_and_remove_accent
    
    # 使用 Moses 标点符号归一化器来归一化给定语言的文本
    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        return punct_normalizer.normalize(text)
    # 使用 Moses 分词器对文本进行分词，根据语言选择相应的分词器
    def moses_tokenize(self, text, lang):
        # 如果指定语言的 Moses 分词器不在缓存中
        if lang not in self.cache_moses_tokenizer:
            # 创建指定语言的 Moses 分词器
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            # 将创建的分词器存储在缓存中
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            # 如果指定语言的 Moses 分词器已在缓存中，则从缓存中获取
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        # 使用获取到的 Moses 分词器对文本进行分词
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    # 对文本进行 Moses 处理管道处理
    def moses_pipeline(self, text, lang):
        # 替换文本中的 Unicode 标点符号
        text = replace_unicode_punct(text)
        # 对文本进行 Moses 标点规范化处理
        text = self.moses_punct_norm(text, lang)
        # 移除文本中的非打印字符
        text = remove_non_printing_char(text)
        # 返回处理后的文本
        return text

    # 对日语文本进行分词处理
    def ja_tokenize(self, text):
        # 如果日语分词器尚未初始化
        if self.ja_word_tokenizer is None:
            try:
                # 尝试导入 Mykytea 库
                import Mykytea
                # 使用 Mykytea 创建日语分词器，并指定模型路径
                self.ja_word_tokenizer = Mykytea.Mykytea(
                    f"-model {os.path.expanduser('~')}/local/share/kytea/model.bin"
                )
            except (AttributeError, ImportError):
                # 如果导入 Mykytea 失败，输出错误信息并引导安装 Mykytea
                logger.error(
                    "Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper"
                    " (https://github.com/chezou/Mykytea-python) with the following steps"
                )
                logger.error("1. git clone git@github.com:neubig/kytea.git && cd kytea")
                logger.error("2. autoreconf -i")
                logger.error("3. ./configure --prefix=$HOME/local")
                logger.error("4. make && make install")
                logger.error("5. pip install kytea")
                # 抛出异常中断程序执行
                raise
        # 使用日语分词器对文本进行分词，并返回结果列表
        return list(self.ja_word_tokenizer.getWS(text))

    # 返回词汇表大小
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 获取词汇表
    def get_vocab(self):
        # 将编码器和附加的编码器合并成一个字典，并返回
        return dict(self.encoder, **self.added_tokens_encoder)
    # 对给定的标记进行字节对编码处理
    def bpe(self, token):
        # 将标记转换为元组形式，并在最后一个元素后添加特殊标记
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 如果标记已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 获取标记的字节对
        pairs = get_pairs(word)
    
        # 如果没有字节对，则直接返回标记后添加特殊标记的结果
        if not pairs:
            return token + "</w>"
    
        # 处理字节对
        while True:
            # 选择出现频率最小的字节对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该字节对不在预训练字节对集合中，则终止循环
            if bigram not in self.bpe_ranks:
                break
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
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        # 替换特殊标记
        if word == "\n  </w>":
            word = "\n</w>"
        # 将结果存入缓存并返回
        self.cache[token] = word
        return word
    
    # 将标记转换为其对应的ID
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))
    
    # 将ID转换为对应的标记
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)
    
    # 将一系列标记转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string
    
    # 构建包含特殊标记的模型输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM sequence has the following format:
    
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`
    
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
    
        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
    
        """
        bos = [self.bos_token_id]
        sep = [self.sep_token_id]
    
        # 如果只有一个输入序列，则添加首部和尾部特殊标记后返回
        if token_ids_1 is None:
            return bos + token_ids_0 + sep
        # 如果有两个输入序列，则在第一个序列尾部和第二个序列尾部各添加特殊标记后返回
        return bos + token_ids_0 + sep + token_ids_1 + sep
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

        # 如果输入的 token 已经包含特殊 tokens，则直接返回
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果 token_ids_1 存在，则返回特殊 token mask
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        # 如果 token_ids_1 不存在，则只返回包含 token_ids_0 的特殊 token mask
        return [1] + ([0] * len(token_ids_0)) + [1]

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
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # 预定义特殊 tokens：[SEP] 和 [CLS]
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        # 如果 token_ids_1 不存在，则返回只包含第一个序列的 token type ids
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 如果 token_ids_1 存在，则返回包含两个序列的 token type ids
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
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

        # 将编码器的内容以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引变量
        index = 0
        # 将 BPE 标记和标记索引按索引值升序排列，写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 检查 BPE 合并索引是否连续，如果不连续则记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回词汇表文件路径和合并文件路径
        return vocab_file, merge_file

    # 获取对象状态
    def __getstate__(self):
        # 复制对象字典
        state = self.__dict__.copy()
        # 将 SM 设置为 None
        state["sm"] = None
        return state

    # 设置对象状态
    def __setstate__(self, d):
        # 将对象字典设置为给定的状态
        self.__dict__ = d

        # 捕获异常，尝试导入 sacremoses 库
        try:
            import sacremoses
        except ImportError:
            # 如果失败，则抛出 ImportError 异常并提供安装链接
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 将 SM 设置为 sacremoses 库
        self.sm = sacremoses
```