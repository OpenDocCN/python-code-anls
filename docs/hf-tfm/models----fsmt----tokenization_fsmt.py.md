# `.\models\fsmt\tokenization_fsmt.py`

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
"""Tokenization classes for FSMT."""


import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

# 定义词汇文件名的映射字典
VOCAB_FILES_NAMES = {
    "src_vocab_file": "vocab-src.json",  # 源语言词汇文件名
    "tgt_vocab_file": "vocab-tgt.json",  # 目标语言词汇文件名
    "merges_file": "merges.txt",          # 合并文件名
}

# 定义预训练模型的词汇文件映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "src_vocab_file": {
        "stas/tiny-wmt19-en-de": "https://huggingface.co/stas/tiny-wmt19-en-de/resolve/main/vocab-src.json"
    },  # 源语言词汇文件的预训练模型映射
    "tgt_vocab_file": {
        "stas/tiny-wmt19-en-de": "https://huggingface.co/stas/tiny-wmt19-en-de/resolve/main/vocab-tgt.json"
    },  # 目标语言词汇文件的预训练模型映射
    "merges_file": {
        "stas/tiny-wmt19-en-de": "https://huggingface.co/stas/tiny-wmt19-en-de/resolve/main/merges.txt"
    },  # 合并文件的预训练模型映射
}

# 定义预训练位置嵌入大小的字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"stas/tiny-wmt19-en-de": 1024}

# 定义预训练初始化配置的字典
PRETRAINED_INIT_CONFIGURATION = {
    "stas/tiny-wmt19-en-de": {
        "langs": ["en", "de"],                 # 支持的语言列表
        "model_max_length": 1024,              # 模型最大长度
        "special_tokens_map_file": None,       # 特殊标记映射文件路径
        "full_tokenizer_file": None,           # 完整分词器文件路径
    }
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))           # 将每对相邻字符添加到集合中
        prev_char = char
    return pairs


def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    text = text.replace("，", ",")              # 替换中文逗号为英文逗号
    text = re.sub(r"。\s*", ". ", text)        # 替换中文句号为英文句号并去除其后的空格
    text = text.replace("、", ",")              # 替换中文顿号为英文逗号
    text = text.replace("”", '"')               # 替换中文右双引号为英文双引号
    text = text.replace("“", '"')               # 替换中文左双引号为英文双引号
    text = text.replace("∶", ":")               # 替换中文冒号为英文冒号
    text = text.replace("：", ":")               # 替换中文冒号为英文冒号
    text = text.replace("？", "?")               # 替换中文问号为英文问号
    text = text.replace("《", '"')               # 替换中文书名号为英文双引号
    text = text.replace("》", '"')               # 替换中文书名号为英文双引号
    text = text.replace("）", ")")               # 替换中文右括号为英文右括号
    text = text.replace("！", "!")               # 替换中文感叹号为英文感叹号
    text = text.replace("（", "(")               # 替换中文左括号为英文左括号
    text = text.replace("；", ";")               # 替换中文分号为英文分号
    text = text.replace("１", "1")               # 替换全角数字１为半角数字1
    text = text.replace("」", '"')               # 替换中文右双引号为英文双引号
    text = text.replace("「", '"')               # 替换中文左双引号为英文双引号
    text = text.replace("０", "0")               # 替换全角数字０为半角数字0
    text = text.replace("３", "3")               # 替换全角数字３为半角数字3
    text = text.replace("２", "2")               # 替换全角数字２为半角数字2
    text = text.replace("５", "5")               # 替换全角数字５为半角数字5
    text = text.replace("６", "6")               # 替换全角数字６为半角数字6
    # 将全角数字９替换为半角数字9
    text = text.replace("９", "9")
    # 将全角数字７替换为半角数字7
    text = text.replace("７", "7")
    # 将全角数字８替换为半角数字8
    text = text.replace("８", "8")
    # 将全角数字４替换为半角数字4
    text = text.replace("４", "4")
    # 将中文句号后面的空白字符（包括全角和半角）替换为一个半角空格
    text = re.sub(r"．\s*", ". ", text)
    # 将全角波浪号～替换为半角波浪号~
    text = text.replace("～", "~")
    # 将全角右单引号’替换为半角右单引号'
    text = text.replace("’", "'")
    # 将全角省略号…替换为半角省略号...
    text = text.replace("…", "...")
    # 将全角长破折号━替换为半角破折号-
    text = text.replace("━", "-")
    # 将全角左尖括号〈替换为半角左尖括号<
    text = text.replace("〈", "<")
    # 将全角右尖括号〉替换为半角右尖括号>
    text = text.replace("〉", ">")
    # 将全角左方括号【替换为半角左方括号[
    text = text.replace("【", "[")
    # 将全角右方括号】替换为半角右方括号]
    text = text.replace("】", "]")
    # 将全角百分号％替换为半角百分号%
    text = text.replace("％", "%")
    # 返回处理后的文本
    return text
def remove_non_printing_char(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    # 初始化一个空列表用于存储处理后的文本字符
    output = []
    # 遍历输入的文本中的每个字符
    for char in text:
        # 使用 unicodedata 获取字符的分类信息
        cat = unicodedata.category(char)
        # 如果字符的分类以 "C" 开头（表示控制字符），则跳过该字符
        if cat.startswith("C"):
            continue
        # 将非控制字符添加到输出列表中
        output.append(char)
    # 将处理后的字符列表连接成一个字符串并返回
    return "".join(output)


# Porting notes:
# this one is modeled after XLMTokenizer
#
# added:
# - src_vocab_file,
# - tgt_vocab_file,
# - langs,


class FSMTTokenizer(PreTrainedTokenizer):
    """
    Construct an FAIRSEQ Transformer tokenizer. Based on Byte-Pair Encoding. The tokenization process is the following:

    - Moses preprocessing and tokenization.
    - Normalizing all inputs text.
    - The arguments `special_tokens` and the function `set_special_tokens`, can be used to add additional symbols (like
      "__classify__") to a vocabulary.
    - The argument `langs` defines a pair of languages.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        langs (`List[str]`, *optional*):
            A list of two languages to translate from and to, for instance `["en", "ru"]`.
        src_vocab_file (`str`, *optional*):
            File containing the vocabulary for the source language.
        tgt_vocab_file (`st`, *optional*):
            File containing the vocabulary for the target language.
        merges_file (`str`, *optional*):
            File containing the merges.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
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

    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 获取预训练模型的初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 获取预训练位置编码大小的配置
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # XLMTokenizer 类的构造函数
    def __init__(
        self,
        langs=None,
        src_vocab_file=None,
        tgt_vocab_file=None,
        merges_file=None,
        do_lower_case=False,
        unk_token="<unk>",
        bos_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        try:
            import sacremoses
        except ImportError:
            # 如果导入失败，抛出 ImportError 异常
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 导入 sacremoses 成功后，将其保存到实例属性中
        self.sm = sacremoses

        # 设置实例属性，保存传入的参数
        self.src_vocab_file = src_vocab_file
        self.tgt_vocab_file = tgt_vocab_file
        self.merges_file = merges_file
        self.do_lower_case = do_lower_case

        # 实例属性，缓存 sacremoses 的 MosesPunctNormalizer 实例
        self.cache_moses_punct_normalizer = {}
        # 实例属性，缓存 sacremoses 的 MosesTokenizer 实例
        self.cache_moses_tokenizer = {}
        # 实例属性，缓存 sacremoses 的 MosesDetokenizer 实例
        self.cache_moses_detokenizer = {}

        # 如果指定了语言列表，并且长度为 2
        if langs and len(langs) == 2:
            # 将第一个语言和第二个语言分别保存到实例属性中
            self.src_lang, self.tgt_lang = langs
        else:
            # 如果语言列表不符合要求，抛出 ValueError 异常
            raise ValueError(
                f"arg `langs` needs to be a list of 2 langs, e.g. ['en', 'ru'], but got {langs}. "
                "Usually that means that tokenizer can't find a mapping for the given model path "
                "in PRETRAINED_VOCAB_FILES_MAP, and other maps of this tokenizer."
            )

        # 使用 utf-8 编码打开源语料库词汇文件，并加载为 JSON 格式，保存到实例属性中
        with open(src_vocab_file, encoding="utf-8") as src_vocab_handle:
            self.encoder = json.load(src_vocab_handle)
        # 使用 utf-8 编码打开目标语料库词汇文件，并加载为 JSON 格式，创建反向映射字典，保存到实例属性中
        with open(tgt_vocab_file, encoding="utf-8") as tgt_vocab_handle:
            tgt_vocab = json.load(tgt_vocab_handle)
            self.decoder = {v: k for k, v in tgt_vocab.items()}
        # 使用 utf-8 编码打开 BPE 合并文件，读取内容并处理成元组列表，保存到实例属性中
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 实例属性，缓存
        self.cache = {}

        # 调用父类的构造函数，传入相同的参数和关键字参数
        super().__init__(
            langs=langs,
            src_vocab_file=src_vocab_file,
            tgt_vocab_file=tgt_vocab_file,
            merges_file=merges_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            **kwargs,
        )

    # hack override，重写父类方法，获取词汇表
    def get_vocab(self) -> Dict[str, int]:
        return self.get_src_vocab()

    # hack override，重写父类属性，返回源语言词汇表大小
    @property
    def vocab_size(self) -> int:
        return self.src_vocab_size
    # 使用 MosesPunctNormalizer 对象规范化文本中的标点符号，根据语言缓存对象以提高效率
    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = self.sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        return self.cache_moses_punct_normalizer[lang].normalize(text)

    # 使用 MosesTokenizer 对象对文本进行标记化，根据语言缓存对象以提高效率
    def moses_tokenize(self, text, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        return self.cache_moses_tokenizer[lang].tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=True
        )

    # 使用 MosesDetokenizer 对象对标记化的 tokens 进行反标记化，根据语言缓存对象以提高效率
    def moses_detokenize(self, tokens, lang):
        if lang not in self.cache_moses_detokenizer:
            moses_detokenizer = self.sm.MosesDetokenizer(lang=lang)
            self.cache_moses_detokenizer[lang] = moses_detokenizer
        return self.cache_moses_detokenizer[lang].detokenize(tokens)

    # 使用一系列预处理步骤处理文本，包括替换Unicode标点、标准化标点符号和移除非打印字符
    def moses_pipeline(self, text, lang):
        text = replace_unicode_punct(text)
        text = self.moses_punct_norm(text, lang)
        text = remove_non_printing_char(text)
        return text

    # 返回源语言词汇表的大小，即编码器的长度
    @property
    def src_vocab_size(self):
        return len(self.encoder)

    # 返回目标语言词汇表的大小，即解码器的长度
    @property
    def tgt_vocab_size(self):
        return len(self.decoder)

    # 返回源语言的词汇表，包括编码器和附加的特殊标记
    def get_src_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 返回目标语言的词汇表，包括解码器和附加的特殊标记
    def get_tgt_vocab(self):
        return dict(self.decoder, **self.added_tokens_decoder)

    # 使用 BPE（字节对编码）算法对单词进行分段处理，根据缓存提高处理速度
    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            # 找到 BPE 算法中频率最低的字节对，根据事先定义的排序规则选择
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
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
        # 将处理后的单词转换为字符串形式，并进行缓存以提高后续处理效率
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word
    def _tokenize(self, text, lang="en", bypass_tokenizer=False):
        """
        Tokenize a string given language code using Moses.

        Details of tokenization:

            - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`

        Args:
            - lang: ISO language code (default = 'en') (string). Languages should belong of the model supported
              languages. However, we don't enforce it.
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)
              (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
        # 忽略当前没有显式传递的 `lang` 参数，tokenization_utils.py 中总是结果 lang=en
        # if lang != self.src_lang:
        #     raise ValueError(f"Expected lang={self.src_lang}, but got {lang}")
        # 将 lang 参数设置为 self.src_lang
        lang = self.src_lang

        # 如果 do_lower_case 为 True，则将文本转换为小写
        if self.do_lower_case:
            text = text.lower()

        # 如果 bypass_tokenizer 为 True，则将文本按空格分割成列表
        if bypass_tokenizer:
            text = text.split()
        else:
            # 使用 Moses 处理管道处理文本
            text = self.moses_pipeline(text, lang=lang)
            # 使用 Moses 分词函数对文本进行分词
            text = self.moses_tokenize(text, lang=lang)

        # 初始化空列表 split_tokens 用于存放最终的分词结果
        split_tokens = []
        # 遍历每个 token
        for token in text:
            # 如果 token 存在
            if token:
                # 将 BPE 分词后的结果以空格分隔，加入到 split_tokens 列表中
                split_tokens.extend(list(self.bpe(token).split(" ")))

        # 返回最终的分词结果列表
        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将 token 转换为对应的 id，如果找不到则返回 unk_token 对应的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将 index 转换为对应的 token，如果找不到则返回 unk_token
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""

        # 去除 tokens 中的 BPE 标记
        tokens = [t.replace(" ", "").replace("</w>", " ") for t in tokens]
        # 将 tokens 列表合并成一个字符串
        tokens = "".join(tokens).split()
        # 使用 Moses 的 detokenize 方法将 tokens 转换为单个字符串
        text = self.moses_detokenize(tokens, self.tgt_lang)
        # 返回最终的字符串文本
        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.

        A RoBERTa sequence has the following format:
        single sequence: [CLS] X [SEP]
        pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs corresponding to the first sequence.
            token_ids_1 (:obj:`List[int]`, `optional`):
                List of IDs corresponding to the second sequence.

        Returns:
            :obj:`List[int]`: List of IDs with the appropriate special tokens.
        """
        # 初始化输入 tokens 列表，并加入第一个特殊 token [CLS]
        input_ids = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果有第二个序列的 token IDs，加入第二个特殊 token [SEP] 和第二个序列的 token IDs
        if token_ids_1 is not None:
            input_ids += token_ids_1 + [self.sep_token_id]

        # 返回包含特殊 token 的输入 token IDs 列表
        return input_ids
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A FAIRSEQ Transformer sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """
        sep = [self.sep_token_id]

        # no bos used in fairseq
        # If token_ids_1 is not provided, return token_ids_0 concatenated with sep tokens
        if token_ids_1 is None:
            return token_ids_0 + sep
        # Otherwise, concatenate token_ids_0, sep, token_ids_1, and sep
        return token_ids_0 + sep + token_ids_1 + sep

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

        # If already_has_special_tokens is True, delegate to the superclass's method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        
        # no bos used in fairseq
        # If token_ids_1 is not None, create a mask with 0s for token_ids_0, 1 for sep, 0s for token_ids_1, and 1 for sep
        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        
        # Otherwise, create a mask with 0s for token_ids_0 and 1 for sep
        return ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs tensor from a list of token ids. In a sequence pair, A and B would have different types (0 and 1).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs.
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in
    # 定义 __getstate__ 方法，用于返回对象的状态字典
    def __getstate__(self):
        # 复制对象的 __dict__ 属性，获取当前对象的状态
        state = self.__dict__.copy()
        # 将对象的 "sm" 属性设为 None，可能是为了清除敏感信息或重置状态
        state["sm"] = None
        # 返回对象的状态字典
        return state

    # 定义 __setstate__ 方法，用于设置对象的状态
    def __setstate__(self, d):
        # 将传入的状态字典 d 直接赋给对象的 __dict__ 属性，以恢复对象的状态
        self.__dict__ = d

        # 尝试导入 sacremoses 库，如果导入失败则抛出 ImportError
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 将导入的 sacremoses 库赋给对象的 "sm" 属性，可能用于后续的操作
        self.sm = sacremoses
```