# `.\models\xlm\tokenization_xlm.py`

```py
# coding=utf-8
# Copyright 2019 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for XLM."""


import json  # 导入处理 JSON 格式的库
import os    # 导入操作系统相关功能的库
import re    # 导入正则表达式的库
import sys   # 导入系统相关功能的库
import unicodedata  # 导入 Unicode 数据库
from typing import List, Optional, Tuple  # 导入类型提示相关功能

from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练 Tokenizer 的工具类
from ...utils import logging  # 导入日志记录功能


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",   # 词汇表文件名
    "merges_file": "merges.txt",  # 合并文件名
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "FacebookAI/xlm-mlm-en-2048": "https://huggingface.co/FacebookAI/xlm-mlm-en-2048/resolve/main/vocab.json",
        "FacebookAI/xlm-mlm-ende-1024": "https://huggingface.co/FacebookAI/xlm-mlm-ende-1024/resolve/main/vocab.json",
        "FacebookAI/xlm-mlm-enfr-1024": "https://huggingface.co/FacebookAI/xlm-mlm-enfr-1024/resolve/main/vocab.json",
        "FacebookAI/xlm-mlm-enro-1024": "https://huggingface.co/FacebookAI/xlm-mlm-enro-1024/resolve/main/vocab.json",
        "FacebookAI/xlm-mlm-tlm-xnli15-1024": "https://huggingface.co/FacebookAI/xlm-mlm-tlm-xnli15-1024/resolve/main/vocab.json",
        "FacebookAI/xlm-mlm-xnli15-1024": "https://huggingface.co/FacebookAI/xlm-mlm-xnli15-1024/resolve/main/vocab.json",
        "FacebookAI/xlm-clm-enfr-1024": "https://huggingface.co/FacebookAI/xlm-clm-enfr-1024/resolve/main/vocab.json",
        "FacebookAI/xlm-clm-ende-1024": "https://huggingface.co/FacebookAI/xlm-clm-ende-1024/resolve/main/vocab.json",
        "FacebookAI/xlm-mlm-17-1280": "https://huggingface.co/FacebookAI/xlm-mlm-17-1280/resolve/main/vocab.json",
        "FacebookAI/xlm-mlm-100-1280": "https://huggingface.co/FacebookAI/xlm-mlm-100-1280/resolve/main/vocab.json",
    },
    # merges_file 字典，包含多个键值对，每个键值对表示一个模型名称和其对应的 merges.txt 文件链接
    "merges_file": {
        "FacebookAI/xlm-mlm-en-2048": "https://huggingface.co/FacebookAI/xlm-mlm-en-2048/resolve/main/merges.txt",
        "FacebookAI/xlm-mlm-ende-1024": "https://huggingface.co/FacebookAI/xlm-mlm-ende-1024/resolve/main/merges.txt",
        "FacebookAI/xlm-mlm-enfr-1024": "https://huggingface.co/FacebookAI/xlm-mlm-enfr-1024/resolve/main/merges.txt",
        "FacebookAI/xlm-mlm-enro-1024": "https://huggingface.co/FacebookAI/xlm-mlm-enro-1024/resolve/main/merges.txt",
        "FacebookAI/xlm-mlm-tlm-xnli15-1024": "https://huggingface.co/FacebookAI/xlm-mlm-tlm-xnli15-1024/resolve/main/merges.txt",
        "FacebookAI/xlm-mlm-xnli15-1024": "https://huggingface.co/FacebookAI/xlm-mlm-xnli15-1024/resolve/main/merges.txt",
        "FacebookAI/xlm-clm-enfr-1024": "https://huggingface.co/FacebookAI/xlm-clm-enfr-1024/resolve/main/merges.txt",
        "FacebookAI/xlm-clm-ende-1024": "https://huggingface.co/FacebookAI/xlm-clm-ende-1024/resolve/main/merges.txt",
        "FacebookAI/xlm-mlm-17-1280": "https://huggingface.co/FacebookAI/xlm-mlm-17-1280/resolve/main/merges.txt",
        "FacebookAI/xlm-mlm-100-1280": "https://huggingface.co/FacebookAI/xlm-mlm-100-1280/resolve/main/merges.txt",
    },
}

# 预训练位置嵌入大小的字典，每个模型名称对应其位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "FacebookAI/xlm-mlm-en-2048": 512,
    "FacebookAI/xlm-mlm-ende-1024": 512,
    "FacebookAI/xlm-mlm-enfr-1024": 512,
    "FacebookAI/xlm-mlm-enro-1024": 512,
    "FacebookAI/xlm-mlm-tlm-xnli15-1024": 512,
    "FacebookAI/xlm-mlm-xnli15-1024": 512,
    "FacebookAI/xlm-clm-enfr-1024": 512,
    "FacebookAI/xlm-clm-ende-1024": 512,
    "FacebookAI/xlm-mlm-17-1280": 512,
    "FacebookAI/xlm-mlm-100-1280": 512,
}

# 预训练模型初始化配置的字典，每个模型名称对应其特定的初始化配置
PRETRAINED_INIT_CONFIGURATION = {
    "FacebookAI/xlm-mlm-en-2048": {"do_lowercase_and_remove_accent": True},
    "FacebookAI/xlm-mlm-ende-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {0: "de", 1: "en"},
        "lang2id": {"de": 0, "en": 1},
    },
    "FacebookAI/xlm-mlm-enfr-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {0: "en", 1: "fr"},
        "lang2id": {"en": 0, "fr": 1},
    },
    "FacebookAI/xlm-mlm-enro-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {0: "en", 1: "ro"},
        "lang2id": {"en": 0, "ro": 1},
    },
    "FacebookAI/xlm-mlm-tlm-xnli15-1024": {
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
    "FacebookAI/xlm-mlm-xnli15-1024": {
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
    "FacebookAI/xlm-clm-enfr-1024": {
        "do_lowercase_and_remove_accent": True,
        "id2lang": {0: "en", 1: "fr"},
        "lang2id": {"en": 0, "fr": 1},
    },
    "FacebookAI/xlm-clm-ende-1024": {
        # 执行小写化和去除重音符号的操作，设为 True
        "do_lowercase_and_remove_accent": True,
        # ID 到语言的映射字典
        "id2lang": {0: "de", 1: "en"},
        # 语言到ID的映射字典
        "lang2id": {"de": 0, "en": 1},
    },
    "FacebookAI/xlm-mlm-17-1280": {
        # 执行小写化和去除重音符号的操作，设为 False
        "do_lowercase_and_remove_accent": False,
        # ID 到语言的映射字典，包含17种语言
        "id2lang": {
            0: "ar",
            1: "de",
            2: "en",
            3: "es",
            4: "fr",
            5: "hi",
            6: "it",
            7: "ja",
            8: "ko",
            9: "nl",
            10: "pl",
            11: "pt",
            12: "ru",
            13: "sv",
            14: "tr",
            15: "vi",
            16: "zh",
        },
        # 语言到ID的映射字典，与上面的ID到语言对应
        "lang2id": {
            "ar": 0,
            "de": 1,
            "en": 2,
            "es": 3,
            "fr": 4,
            "hi": 5,
            "it": 6,
            "ja": 7,
            "ko": 8,
            "nl": 9,
            "pl": 10,
            "pt": 11,
            "ru": 12,
            "sv": 13,
            "tr": 14,
            "vi": 15,
            "zh": 16,
        },
    },
}

# 定义函数结束，这是一个空的函数定义，没有具体的实现内容

def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        # 将当前字符与前一个字符作为一个符号对加入到集合中
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def lowercase_and_remove_accent(text):
    """
    Lowercase and strips accents from a piece of text based on
    https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
    """
    # 将文本以空格连接，然后转换为小写
    text = " ".join(text)
    text = text.lower()
    # 使用NFD规范将文本进行Unicode标准化
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        # 获取Unicode字符的分类
        cat = unicodedata.category(char)
        # 如果字符是非spacing mark，则加入到输出列表中
        if cat == "Mn":
            continue
        output.append(char)
    # 将输出列表连接成字符串并按空格分割后返回
    return "".join(output).lower().split(" ")


def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    # 替换文本中的Unicode标点符号为ASCII符号
    text = text.replace("，", ",")
    text = re.sub(r"。\s*", ". ", text)
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
    text = text.replace("３", "3")
    text = text.replace("２", "2")
    text = text.replace("５", "5")
    text = text.replace("６", "6")
    text = text.replace("９", "9")
    text = text.replace("７", "7")
    text = text.replace("８", "8")
    text = text.replace("４", "4")
    text = re.sub(r"．\s*", ". ", text)
    text = text.replace("～", "~")
    text = text.replace("’", "'")
    text = text.replace("…", "...")
    text = text.replace("━", "-")
    text = text.replace("〈", "<")
    text = text.replace("〉", ">")
    text = text.replace("【", "[")
    text = text.replace("】", "]")
    text = text.replace("％", "%")
    return text


def remove_non_printing_char(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    output = []
    for char in text:
        # 获取Unicode字符的分类
        cat = unicodedata.category(char)
        # 如果字符以C开头，表示是不可打印字符，跳过
        if cat.startswith("C"):
            continue
        output.append(char)
    # 将输出列表连接成字符串后返回
    return "".join(output)


def romanian_preprocessing(text):
    """Sennrich's WMT16 scripts for Romanian preprocessing, used by model `FacebookAI/xlm-mlm-enro-1024`"""
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/normalise-romanian.py
    # 替换文本中的特定Unicode字符为另一组Unicode字符
    text = text.replace("\u015e", "\u0218").replace("\u015f", "\u0219")
    text = text.replace("\u0162", "\u021a").replace("\u0163", "\u021b")
    # 替换文本中的特定 Unicode 字符为对应的 ASCII 字符
    text = text.replace("\u0218", "S").replace("\u0219", "s")  # 将 '\u0218' 替换为 'S'，'\u0219' 替换为 's'（s-comma）
    text = text.replace("\u021a", "T").replace("\u021b", "t")  # 将 '\u021a' 替换为 'T'，'\u021b' 替换为 't'（t-comma）
    text = text.replace("\u0102", "A").replace("\u0103", "a")  # 将 '\u0102' 替换为 'A'，'\u0103' 替换为 'a'
    text = text.replace("\u00C2", "A").replace("\u00E2", "a")  # 将 '\u00C2' 替换为 'A'，'\u00E2' 替换为 'a'
    text = text.replace("\u00CE", "I").replace("\u00EE", "i")  # 将 '\u00CE' 替换为 'I'，'\u00EE' 替换为 'i'
    # 返回替换后的文本
    return text
class XLMTokenizer(PreTrainedTokenizer):
    """
    Construct an XLM tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization for most supported languages.
    - Language specific tokenization for Chinese (Jieba), Japanese (KyTea) and Thai (PyThaiNLP).
    - Optionally lowercases and normalizes all inputs text.
    - The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
      "__classify__") to a vocabulary.
    - The `lang2id` attribute maps the languages supported by the model with their IDs if provided (automatically set
      for pretrained vocabularies).
    - The `id2lang` attributes does reverse mapping if provided (automatically set for pretrained vocabularies).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 定义函数的参数说明文档字符串，描述了每个参数的含义和默认值
    Args:
        vocab_file (`str`):
            Vocabulary file.
        merges_file (`str`):
            Merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (`str`, *optional*, defaults to `"</s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (`str`, *optional*, defaults to `"<special1>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*, defaults to `['<special0>', '<special1>', '<special2>', '<special3>', '<special4>', '<special5>', '<special6>', '<special7>', '<special8>', '<special9>']`):
            List of additional special tokens.
        lang2id (`Dict[str, int]`, *optional*):
            Dictionary mapping languages string identifiers to their IDs.
        id2lang (`Dict[int, str]`, *optional*):
            Dictionary mapping language IDs to their string identifiers.
        do_lowercase_and_remove_accent (`bool`, *optional*, defaults to `True`):
            Whether to lowercase and remove accents when tokenizing.
    ```

    # 初始化一些预定义的常量和映射，用于模型预训练时使用
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化函数，用于实例化一个 XLMTokenizer 对象
    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<unk>",
        bos_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="</s>",
        mask_token="<special1>",
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
        do_lowercase_and_remove_accent=True,
        **kwargs,
    ):
        # 尝试导入 sacremoses 库，如果导入失败则抛出 ImportError
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 将 sacremoses 模块赋值给 self.sm
        self.sm = sacremoses

        # 缓存 sm.MosesPunctNormalizer 实例的字典
        self.cache_moses_punct_normalizer = {}
        # 缓存 sm.MosesTokenizer 实例的字典
        self.cache_moses_tokenizer = {}

        # 支持自定义分词器的语言集合，包括中文、泰语和日语
        self.lang_with_custom_tokenizer = {"zh", "th", "ja"}

        # 是否执行小写化和去除重音，用于当前支持的模型（v1.2.0）和 XLM-17 & 100 模型的区分
        self.do_lowercase_and_remove_accent = do_lowercase_and_remove_accent
        self.lang2id = lang2id
        self.id2lang = id2lang

        # 如果 lang2id 和 id2lang 都不为 None，则断言它们的长度相等
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)

        # 日语分词器和中文分词器初始化为 None
        self.ja_word_tokenizer = None
        self.zh_word_tokenizer = None

        # 从 vocab_file 中读取编码器（encoder）的 JSON 格式数据
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        # 构建解码器（decoder），将编码器的键值对反转
        self.decoder = {v: k for k, v in self.encoder.items()}

        # 从 merges_file 中读取 BPE merges 数据并处理成字典形式的 bpe_ranks
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        # 缓存对象
        self.cache = {}

        # 调用父类的初始化方法，传递各种参数和关键字参数
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

    # do_lower_case 属性的 getter 方法，返回 do_lowercase_and_remove_accent 的值
    @property
    def do_lower_case(self):
        return self.do_lowercase_and_remove_accent

    # 使用 sacremoses 库的 MosesPunctNormalizer 进行标点符号规范化处理
    def moses_punct_norm(self, text, lang):
        # 如果 lang 不在 cache_moses_punct_normalizer 的键中，则创建一个新的 MosesPunctNormalizer 实例
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            # 否则从缓存中获取现有的 MosesPunctNormalizer 实例
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        # 调用 normalize 方法对文本进行标点符号规范化处理并返回结果
        return punct_normalizer.normalize(text)
    # 使用 Moses 分词器对文本进行分词处理，根据语言选择缓存的分词器实例或创建新的实例
    def moses_tokenize(self, text, lang):
        # 如果指定语言的分词器不在缓存中，则创建并存储
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            # 否则，从缓存中获取已存储的分词器实例
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        # 使用选定的分词器对文本进行分词处理，返回分词结果
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    # 执行一系列预处理步骤对输入文本进行规范化处理，不返回字符串格式的文本
    def moses_pipeline(self, text, lang):
        # 替换文本中的 Unicode 标点符号
        text = replace_unicode_punct(text)
        # 使用指定语言的 Moses 标点规范化函数处理文本
        text = self.moses_punct_norm(text, lang)
        # 移除文本中的非打印字符
        text = remove_non_printing_char(text)
        # 返回处理后的文本
        return text

    # 使用 Mykytea 进行日语文本的分词处理，若实例未初始化，则进行初始化
    def ja_tokenize(self, text):
        # 如果日语分词器尚未初始化
        if self.ja_word_tokenizer is None:
            try:
                # 尝试导入 Mykytea 库进行初始化
                import Mykytea
                # 使用 Mykytea 初始化日语分词器
                self.ja_word_tokenizer = Mykytea.Mykytea(
                    f"-model {os.path.expanduser('~')}/local/share/kytea/model.bin"
                )
            except (AttributeError, ImportError):
                # 若导入失败，则记录错误信息并引发异常
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
        # 使用日语分词器对文本进行分词处理，返回分词结果列表
        return list(self.ja_word_tokenizer.getWS(text))

    # 返回当前词汇表的大小，即编码器中条目的数量
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 返回词汇表的字典表示，包括编码器和添加的特殊标记编码器
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
    def bpe(self, token):
        # 将输入的 token 转换为特定格式的元组 word，以便后续处理
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 获取 token 中所有可能的 bigram 对
        pairs = get_pairs(word)

        # 如果没有找到任何 bigram 对，则在 token 后面加上结束符 "</w>" 并返回
        if not pairs:
            return token + "</w>"

        # 开始迭代处理 bigram 对，直到无法再合并
        while True:
            # 找到当前 word 中频率最小的 bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该 bigram 不在预先计算的频率表中，则停止合并
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 遍历 word 中的每个字符，根据找到的 bigram 进行合并或保留
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
            # 更新 word 为新的合并结果，并转换为元组
            new_word = tuple(new_word)
            word = new_word
            # 如果已经无法继续合并，则停止循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        # 将处理后的 word 转换为字符串形式
        word = " ".join(word)
        # 如果转换后的 word 是特定格式，则做相应替换处理
        if word == "\n  </w>":
            word = "\n</w>"
        # 将处理结果缓存起来，并返回
        self.cache[token] = word
        return word

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将 token 转换为对应的 ID，如果 token 不存在则使用未知词符号
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将 ID 转换为对应的 token，如果 ID 不存在则使用未知词符号
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将一系列 token 组合成一个字符串，替换特定结束符后返回
        out_string = "".join(tokens).replace("</w>", " ").strip()
        return out_string

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
        # 定义起始符 `<s>` 和分隔符 `</s>`
        bos = [self.bos_token_id]
        sep = [self.sep_token_id]

        # 如果没有提供 token_ids_1，则返回单个序列的输入 ID 列表
        if token_ids_1 is None:
            return bos + token_ids_0 + sep
        
        # 如果提供了 token_ids_1，则返回双序列的输入 ID 列表
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

        # If tokens already have special tokens, delegate to superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Calculate special tokens mask based on whether there is a second sequence
        if token_ids_1 is not None:
            # Case for sequence pair: [CLS] token_ids_0 [SEP] token_ids_1 [SEP]
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        else:
            # Case for single sequence: [CLS] token_ids_0 [SEP]
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
        # Define [SEP] and [CLS] tokens
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If only one sequence is provided
        if token_ids_1 is None:
            # Return token type IDs for single sequence: [CLS] token_ids_0 [SEP]
            return len(cls + token_ids_0 + sep) * [0]
        else:
            # Return token type IDs for sequence pair: [CLS] token_ids_0 [SEP] token_ids_1 [SEP]
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 将词汇表保存到指定目录下的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，若不存在则记录错误并返回
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

        # 将编码器(encoder)中的内容以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 初始化索引
        index = 0
        # 打开合并文件，以 UTF-8 编码写入
        with open(merge_file, "w", encoding="utf-8") as writer:
            # 遍历并排序 self.bpe_ranks 中的 BPE 标记及其索引，按索引排序
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 若索引不连续，则记录警告信息
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                # 将 BPE 标记写入文件，并以换行符结尾
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回词汇表文件路径和合并文件路径
        return vocab_file, merge_file

    # 返回对象的序列化状态，用于 pickle 保存
    def __getstate__(self):
        # 复制对象的字典属性
        state = self.__dict__.copy()
        # 将 sm 属性设为 None，避免 pickle 时出现不必要的引用
        state["sm"] = None
        return state

    # 恢复对象的状态，用于 pickle 加载
    def __setstate__(self, d):
        # 将对象的字典属性恢复为给定的状态
        self.__dict__ = d

        # 尝试导入 sacremoses 库，如果失败则抛出 ImportError
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 将 sacremoses 库赋给对象的 sm 属性
        self.sm = sacremoses
```