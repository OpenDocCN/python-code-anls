# `.\transformers\models\biogpt\tokenization_biogpt.py`

```py
# 导入所需的模块和类
# coding=utf-8
# 版权声明
# Copyright 2022 The HuggingFace Team and Microsoft Research AI4Science. All rights reserved.
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
"""Tokenization classes for BioGPT."""
import json  # 导入处理 JSON 格式的模块
import os  # 导入处理文件路径的模块
from typing import List, Optional, Tuple  # 导入类型提示相关的类和函数

from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器的基类
from ...utils import logging  # 导入日志记录模块

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义词汇文件的文件名
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 定义预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/biogpt": "https://huggingface.co/microsoft/biogpt/resolve/main/vocab.json",
    },
    "merges_file": {"microsoft/biogpt": "https://huggingface.co/microsoft/biogpt/resolve/main/merges.txt"},
}

# 定义预训练位置嵌入的尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/biogpt": 1024,
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    # 返回单词中的符号对集合，单词表示为符号元组（符号是变长字符串）
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BioGptTokenizer(PreTrainedTokenizer):
    """
    Construct an FAIRSEQ Transformer tokenizer. Moses tokenization followed by Byte-Pair Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
```  
    # 初始化一个Tokenizer对象，用于将文本转换为模型可以理解的输入格式
    class Tokenizer:
        """
        Args:
            vocab_file (`str`):
                词汇表文件的路径。
            merges_file (`str`):
                合并文件。
            unk_token (`str`, *optional*, defaults to `"<unk>"`):
                未知标记。词汇表中不存在的标记无法转换为 ID，并被设置为此标记。
            bos_token (`str`, *optional*, defaults to `"<s>"`):
                在预训练期间使用的序列开始标记。可用作序列分类器标记。
    
                <Tip>
    
                在使用特殊标记构建序列时，这不是用于序列开始的标记。用于序列开始的标记是 `cls_token`。
    
                </Tip>
    
            eos_token (`str`, *optional*, defaults to `"</s>"`):
                序列结束标记。
    
                <Tip>
    
                在使用特殊标记构建序列时，这不是用于序列结束的标记。用于序列结束的标记是 `sep_token`。
    
                </Tip>
    
            sep_token (`str`, *optional*, defaults to `"</s>"`):
                分隔标记，在从多个序列构建序列时使用，例如用于序列分类或用于文本和问题的问题回答。它还用作使用特殊标记构建的序列的最后一个标记。
            pad_token (`str`, *optional*, defaults to `"<pad>"`):
                用于填充的标记，例如当批处理不同长度的序列时。
        """
    
        # 定义预定义的文件名和映射
        vocab_files_names = VOCAB_FILES_NAMES
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        model_input_names = ["input_ids", "attention_mask"]
    
        def __init__(
            self,
            vocab_file,
            merges_file,
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            pad_token="<pad>",
            **kwargs,
    ):
        # 尝试导入sacremoses模块，如果失败则抛出ImportError
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use BioGptTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 设置语言为英语
        self.lang = "en"
        # 将sacremoses模块赋值给self.sm
        self.sm = sacremoses
        # sm.MosesTokenizer实例的缓存
        self.cache_moses_tokenizer = {}
        # sm.MosesDetokenizer实例的缓存
        self.cache_moses_detokenizer = {}

        """ Initialisation"""
        # 以UTF-8编码打开词汇文件，并加载词汇表到self.encoder中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建反向映射，从编码到词汇的映射
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 以UTF-8编码打开合并文件，加载合并操作并创建BPE排名
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 缓存
        self.cache = {}

        # 调用父类的初始化方法，并传入参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        """返回词汇表大小"""
        return len(self.encoder)

    # 返回词汇表
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 使用sacremoses进行分词
    def moses_tokenize(self, text, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        return self.cache_moses_tokenizer[lang].tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=True
        )

    # 使用sacremoses进行词汇还原
    def moses_detokenize(self, tokens, lang):
        if lang not in self.cache_moses_detokenizer:
            moses_detokenizer = self.sm.MosesDetokenizer(lang=lang)
            self.cache_moses_detokenizer[lang] = moses_detokenizer
        return self.cache_moses_detokenizer[lang].detokenize(tokens)
    def bpe(self, token):
        # 将 token 转换为元组形式，最后一个字符加上 "</w>"
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 获取 token 的所有字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则直接返回 token 加上 "</w>"
        if not pairs:
            return token + "</w>"

        # 循环处理字符对
        while True:
            # 找到当前字符对中频率最低的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果当前字符对不在字符对频率字典中，则跳出循环
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
            # 如果新的 word 长度为 1，则跳出循环
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将 word 转换为字符串形式
        word = " ".join(word)
        # 如果 word 是 "\n  </w>"，则替换为 "\n</w>"
        if word == "\n  </w>":
            word = "\n</w>"
        # 将结果存入缓存中
        self.cache[token] = word
        return word

    def _tokenize(self, text, bypass_tokenizer=False):
        """Returns a tokenized string."""
        # 如果绕过分词器，则直接按空格分割文本
        if bypass_tokenizer:
            text = text.split()
        else:
            # 否则使用 moses_tokenize 方法对文本进行分词
            text = self.moses_tokenize(text, self.lang)

        split_tokens = []
        for token in text:
            if token:
                # 对每个 token 进行 BPE 处理，并按空格分割后加入结果列表
                split_tokens.extend(list(self.bpe(token).split(" ")))

        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将 token 转换为对应的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将 id 转换为对应的 token
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 去除 BPE 标记，将 tokens 拼接为字符串
        tokens = [t.replace(" ", "").replace("</w>", " ") for t in tokens]
        tokens = "".join(tokens).split()
        # 反分词化
        text = self.moses_detokenize(tokens, self.lang)
        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。一个 BioGPT 序列的格式如下：

        - 单个序列：`</s> X `
        - 一对序列：`</s> A </s> B `

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        if token_ids_1 is None:
            return [self.sep_token_id] + token_ids_0
        sep = [self.sep_token_id]
        return sep + token_ids_0 + sep + token_ids_1

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *可选*，默认为 `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        # no bos used in fairseq
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
        return [1] + ([0] * len(token_ids_0))

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A FAIRSEQ
        Transformer sequence pair mask has the following format:

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
            `List[int`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]

        # no bos used in fairseq
        # 如果 token_ids_1 为 None，则只返回 mask 的第一部分（全为 0）
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入合并文件
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

    def __getstate__(self):
        # 复制对象的状态，将 sm 置为 None
        state = self.__dict__.copy()
        state["sm"] = None
        return state

    def __setstate__(self, d):
        # 恢复对象的状态
        self.__dict__ = d

        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        self.sm = sacremoses
```