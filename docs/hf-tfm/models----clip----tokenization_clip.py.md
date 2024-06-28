# `.\models\clip\tokenization_clip.py`

```
# coding=utf-8
# Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for CLIP."""

import json                # 导入处理 JSON 格式的模块
import os                  # 导入操作系统功能的模块
import unicodedata         # 导入 Unicode 数据处理模块
from functools import lru_cache  # 导入 functools 模块中的 lru_cache 装饰器
from typing import List, Optional, Tuple  # 导入类型提示相关的功能

import regex as re         # 导入正则表达式库 regex

from ...tokenization_utils import AddedToken, PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
                            # 导入上级目录中的 tokenization_utils 模块的部分功能
from ...utils import logging   # 导入上级目录中的 logging 模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",     # 定义词汇表文件名对应的常量
    "merges_file": "merges.txt",    # 定义合并规则文件名对应的常量
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
    },  # 预训练模型词汇表文件的映射
    "merges_file": {
        "openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
    },  # 预训练模型合并规则文件的映射
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai/clip-vit-base-patch32": 77,  # 预训练模型位置嵌入的尺寸映射
}

PRETRAINED_INIT_CONFIGURATION = {
    "openai/clip-vit-base-patch32": {},  # 预训练模型的初始化配置信息
}

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )  # 定义包含不同范围 Unicode 字节的列表
    cs = bs[:]  # 复制 bs 列表到 cs
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]  # 将 cs 列表中的整数转换为对应的 Unicode 字符
    return dict(zip(bs, cs))   # 返回由 utf-8 字节到 Unicode 字符的映射表

def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()          # 创建一个空集合用于存储符号对
    prev_char = word[0]    # 获取单词中的第一个字符作为上一个字符
    for char in word[1:]:  # 遍历单词中的每个字符，从第二个字符开始
        pairs.add((prev_char, char))  # 将前一个字符和当前字符作为符号对添加到集合中
        prev_char = char    # 更新前一个字符为当前字符
    return pairs           # 返回符号对集合

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)  # 使用正则表达式将多个连续空白字符替换为单个空格
    text = text.strip()               # 去除字符串两端的空白字符
    return text                       # 返回清理后的文本
# 从 transformers.models.bert.tokenization_bert.whitespace_tokenize 复制而来
def whitespace_tokenize(text):
    """对文本进行基本的空白符清理和分割。"""
    # 去除文本两端的空白符
    text = text.strip()
    # 如果文本为空，则返回空列表
    if not text:
        return []
    # 使用空白符分割文本，生成 token 列表
    tokens = text.split()
    # 返回分割后的 token 列表
    return tokens


# 从 transformers.models.bert.tokenization_bert.BasicTokenizer 复制而来
class BasicTokenizer(object):
    """
    构建一个 BasicTokenizer 对象，执行基本的分词操作（标点符号分割、小写处理等）。

    Args:
        do_lower_case (`bool`, *可选*, 默认为 `True`):
            是否在分词时将输入转换为小写。
        never_split (`Iterable`, *可选*):
            在分词时不应该分割的 token 集合。仅在 `do_basic_tokenize=True` 时有效。
        tokenize_chinese_chars (`bool`, *可选*, 默认为 `True`):
            是否对中文字符进行分词处理。

            对于日语，应该将此选项禁用（参见这个
            [issue](https://github.com/huggingface/transformers/issues/328)）。
        strip_accents (`bool`, *可选*):
            是否去除所有的重音符号。如果未指定此选项，则根据 `lowercase` 的值决定（与原始 BERT 一致）。
        do_split_on_punc (`bool`, *可选*, 默认为 `True`):
            在某些情况下，我们希望跳过基本的标点符号分割，以便后续的分词可以捕获单词的完整上下文，如缩略词。

    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
        do_split_on_punc=True,
    ):
        # 如果 never_split 为 None，则初始化为空列表
        if never_split is None:
            never_split = []
        # 设定是否将输入转换为小写
        self.do_lower_case = do_lower_case
        # 设置不进行分割的 token 集合
        self.never_split = set(never_split)
        # 设置是否对中文字符进行分词处理
        self.tokenize_chinese_chars = tokenize_chinese_chars
        # 设置是否去除所有的重音符号
        self.strip_accents = strip_accents
        # 设置是否进行基本的标点符号分割
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
        # 如果传入了新的不分割的词汇列表（never_split），则将其与类属性中的never_split集合进行合并，否则直接使用类属性的never_split集合
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        # 清理文本，去除不必要的字符
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        # 如果开启了中文字符的分词处理，则对文本进行中文字符的特殊处理
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        # 使用Unicode NFC规范化文本，防止不同Unicode编码的相同字符被视为不同字符
        unicode_normalized_text = unicodedata.normalize("NFC", text)
        # 使用空白字符进行分词
        orig_tokens = whitespace_tokenize(unicode_normalized_text)
        split_tokens = []
        # 遍历每个原始token
        for token in orig_tokens:
            # 如果token不在不分割的词汇列表中
            if token not in never_split:
                # 如果需要转换为小写
                if self.do_lower_case:
                    token = token.lower()
                    # 如果需要去除重音符号，则进行相应处理
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                # 如果需要去除重音符号，则进行相应处理
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            # 将分割后的token加入到split_tokens列表中
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        # 使用空白字符再次分割合并后的tokens，得到最终的输出tokens列表
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        # 返回最终的输出tokens列表
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        # 使用NFD规范化文本，将重音符号分离出来
        text = unicodedata.normalize("NFD", text)
        output = []
        # 遍历文本中的每个字符
        for char in text:
            # 获取字符的Unicode类别
            cat = unicodedata.category(char)
            # 如果字符是重音符号（Mn类别），则跳过该字符
            if cat == "Mn":
                continue
            # 否则将字符加入到输出列表中
            output.append(char)
        # 将输出列表中的字符连接成字符串，返回去除重音符号后的文本
        return "".join(output)
    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        # 如果禁止分割标点或者给定的 text 在 never_split 中，则直接返回包含整个 text 的列表
        if not self.do_split_on_punc or (never_split is not None and text in never_split):
            return [text]
        # 将文本转换为字符列表
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        # 遍历字符列表
        while i < len(chars):
            char = chars[i]
            # 如果当前字符是标点符号
            if _is_punctuation(char):
                # 添加一个新的列表，该列表包含当前标点符号
                output.append([char])
                start_new_word = True
            else:
                # 如果不是标点符号，检查是否需要开始一个新单词
                if start_new_word:
                    output.append([])
                start_new_word = False
                # 将当前字符添加到最后一个列表中
                output[-1].append(char)
            i += 1

        # 将每个子列表中的字符连接起来，形成最终的分割后的字符串列表
        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果是中文字符，添加前后空格
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接成一个字符串并返回
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # 判断是否是中文字符的条件
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
        # 遍历文本中的每个字符
        for char in text:
            cp = ord(char)
            # 如果字符是无效字符或者控制字符，跳过
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            # 如果字符是空白字符，用单个空格替换
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        # 将列表中的字符连接成一个字符串并返回
        return "".join(output)
    """
    构造一个 CLIP 分词器。基于字节级别的 Byte-Pair-Encoding。

    这个分词器继承自 `PreTrainedTokenizer`，其中包含大部分主要方法。用户应该参考这个超类以获取有关这些方法的更多信息。

    Args:
        vocab_file (`str`):
            词汇文件的路径。
        merges_file (`str`):
            合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            将字节解码为 UTF-8 时的错误处理模式。参见
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) 获取更多信息。
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            未知标记。词汇表中不存在的标记无法转换为 ID，因此将被设置为这个标记。
        bos_token (`str`, *optional*, defaults to `"<|startoftext|>"`):
            序列的起始标记。
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            序列的结束标记。
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            用于填充的标记，例如在对不同长度的序列进行批处理时使用。

    """
    vocab_files_names = VOCAB_FILES_NAMES  # 词汇文件的名称
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇文件映射表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练位置嵌入的最大输入尺寸
    model_input_names = ["input_ids", "attention_mask"]  # 模型输入名称列表

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",  # 用于填充的标记，用来启用填充的一个小技巧
        **kwargs,
        ):
            # 如果 bos_token 是字符串，则创建一个 AddedToken 对象，用于表示序列的开头
            bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
            # 如果 eos_token 是字符串，则创建一个 AddedToken 对象，用于表示序列的结尾
            eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
            # 如果 unk_token 是字符串，则创建一个 AddedToken 对象，用于表示未知词
            unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
            try:
                import ftfy
                
                # 尝试导入 ftfy 库，若成功则设定修复文本的函数
                self.fix_text = ftfy.fix_text
            except ImportError:
                # 若导入失败，记录日志并使用自定义的 BasicTokenizer 替代 ftfy
                logger.info("ftfy or spacy is not installed using custom BasicTokenizer instead of ftfy.")
                self.nlp = BasicTokenizer(strip_accents=False, do_split_on_punc=False)
                self.fix_text = None

            # 打开并加载词汇文件到 self.encoder 中
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                self.encoder = json.load(vocab_handle)
            # 创建 self.decoder，用于从编码到原始词汇的反向映射
            self.decoder = {v: k for k, v in self.encoder.items()}
            self.errors = errors  # 记录在解码时如何处理错误
            self.byte_encoder = bytes_to_unicode()
            # 创建字节到 Unicode 的反向映射
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
            # 打开并读取 BPE merges 文件，并处理成适合使用的格式
            with open(merges_file, encoding="utf-8") as merges_handle:
                bpe_merges = merges_handle.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
            bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
            # 创建 BPE merges 的排名字典
            self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
            # 初始化缓存，用于存储特殊 token
            self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}

            # 编译正则表达式模式，用于分词和处理文本
            self.pat = re.compile(
                r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
                re.IGNORECASE,
            )

            # 调用父类的初始化方法，设置模型的各种参数和特殊 token
            super().__init__(
                errors=errors,
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                **kwargs,
            )

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Generate token type IDs from a list of token IDs representing sequences. This is typically used in sequence pair
        tasks to differentiate between the first and the second sequence.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs representing the second sequence in a pair task.

        Returns:
            `List[int]`: List of token type IDs where each ID corresponds to a token in the input sequences.
        """

        # Initialize token type ID lists for the special tokens
        if token_ids_1 is None:
            # If there is only one sequence, all tokens belong to that sequence (token type ID 0)
            return [0] * len(token_ids_0)
        
        # For two sequences, differentiate between them using token type IDs
        # Start with token type 0 for the first sequence, then switch to token type 1 for the second sequence
        token_type_ids = [0] * len(token_ids_0) + [1] * len(token_ids_1)
        
        return token_type_ids
    ) -> List[int]:
        """
        Create a mask from the two sequences passed. CLIP does not make use of token type ids, therefore a list of
        zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        bos_token = [self.bos_token_id]  # Initialize list with beginning of sequence token ID
        eos_token = [self.eos_token_id]  # Initialize list with end of sequence token ID

        if token_ids_1 is None:
            return len(bos_token + token_ids_0 + eos_token) * [0]  # Return a list of zeros of length equal to the sum of the lengths of bos_token, token_ids_0, and eos_token
        return len(bos_token + token_ids_0 + eos_token + eos_token + token_ids_1 + eos_token) * [0]  # Return a list of zeros of length equal to the sum of the lengths of bos_token, token_ids_0, eos_token, another eos_token, token_ids_1, and eos_token

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]  # Return cached value if token exists in cache
        word = tuple(token[:-1]) + (token[-1] + "</w>",)  # Append "</w>" to the last character of the token and convert it to a tuple
        pairs = get_pairs(word)  # Get all pairs of characters in the token

        if not pairs:
            return token + "</w>"  # Append "</w>" to token if no character pairs are found

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))  # Find the pair with the lowest rank according to self.bpe_ranks
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram  # Separate the first and second characters of the bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)  # Find the index of the first character in word starting from index i
                except ValueError:
                    new_word.extend(word[i:])  # Extend new_word with remaining characters if first character is not found
                    break
                else:
                    new_word.extend(word[i:j])  # Extend new_word with characters from i to j (excluding j)
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)  # Append the bigram to new_word if it matches first and second characters in sequence
                    i += 2
                else:
                    new_word.append(word[i])  # Append current character to new_word
                    i += 1
            new_word = tuple(new_word)  # Convert new_word to tuple
            word = new_word  # Update word with new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)  # Get new pairs for updated word
        word = " ".join(word)  # Join characters of word with spaces
        self.cache[token] = word  # Cache the token and its corresponding word
        return word  # Return the final word after BPE encoding

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []  # Initialize list to store BPE tokens
        if self.fix_text is None:
            text = " ".join(self.nlp.tokenize(text))  # Tokenize text using self.nlp if fix_text is None
        else:
            text = whitespace_clean(self.fix_text(text)).lower()  # Clean and lowercase text using whitespace_clean function if fix_text is defined

        for token in re.findall(self.pat, text):  # Iterate through tokens found using regex pattern self.pat in text
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Encode each character of token into byte_encoder and join them into a string
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))  # Extend bpe_tokens with BPE tokens split by space
        return bpe_tokens  # Return list of BPE tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))  # Return ID of token from encoder or return ID of unk_token if token is not found
    def _convert_id_to_token(self, index):
        """Converts an index (integer) to a token (str) using the vocab."""
        # 使用词汇表将索引转换为对应的标记字符串
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings) into a single string."""
        # 将一系列标记字符串连接成一个字符串
        text = "".join(tokens)
        # 根据字节解码器将字符串转换为字节数组
        byte_array = bytearray([self.byte_decoder[c] for c in text])
        # 使用 utf-8 解码字节数组，并替换特定字符串，去除首尾空格
        text = byte_array.decode("utf-8", errors=self.errors).replace("</w>", " ").strip()
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        
        # 构建词汇表文件路径和合并文件路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 写入词汇表到文件中，使用 UTF-8 编码，保证非 ASCII 字符的正确性
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入合并文件的版本信息
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历并写入 BPE merges 到文件中，按照索引排序
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 记录警告，如果 BPE 合并的索引不是连续的，可能意味着分词器出现问题
                    logger.warning(
                        "Saving vocabulary to {}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!".format(merge_file)
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file
```