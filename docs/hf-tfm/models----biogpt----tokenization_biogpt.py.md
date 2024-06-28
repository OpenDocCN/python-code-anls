# `.\models\biogpt\tokenization_biogpt.py`

```py
# coding=utf-8
# 设定文件编码为 UTF-8

# Copyright 2022 The HuggingFace Team and Microsoft Research AI4Science. All rights reserved.
# 版权声明，版权归属于 HuggingFace Team 和 Microsoft Research AI4Science，保留所有权利。

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证授权，除非符合许可证，否则不得使用此文件。

# you may not use this file except in compliance with the License.
# 除非符合许可证，否则不得使用此文件。

# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则软件

# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 按“原样”分发，无论是明示还是暗示的，不附带任何形式的担保或条件。

# See the License for the specific language governing permissions and
# 查看许可证获取特定语言的权限

# limitations under the License.
# 许可证下的限制。

"""Tokenization classes for BioGPT."""
# 用于 BioGPT 的分词类

import json
# 导入 json 库
import os
# 导入 os 库
from typing import List, Optional, Tuple
# 导入类型提示，List、Optional 和 Tuple

from ...tokenization_utils import PreTrainedTokenizer
# 从 tokenization_utils 模块中导入 PreTrainedTokenizer 类
from ...utils import logging
# 从 utils 模块中导入 logging 模块

logger = logging.get_logger(__name__)
# 获取当前模块的 logger 对象

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}
# 定义词汇文件和合并文件的名称映射字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/biogpt": "https://huggingface.co/microsoft/biogpt/resolve/main/vocab.json",
    },
    "merges_file": {"microsoft/biogpt": "https://huggingface.co/microsoft/biogpt/resolve/main/merges.txt"},
}
# 预训练模型的词汇文件和合并文件的映射字典，指定了 Microsoft 的 BioGPT 模型的文件位置

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/biogpt": 1024,
}
# 预训练模型的位置嵌入尺寸字典，指定了 Microsoft 的 BioGPT 模型的位置嵌入大小为 1024


def get_pairs(word):
    """
    Return set of symbol pairs in a word. word is represented as tuple of symbols (symbols being variable-length
    strings)
    """
    # 返回单词中的符号对集合，单词表示为符号元组（符号是可变长度字符串）
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs
    # 遍历单词中的字符，将相邻的字符作为一个对加入到集合中，并返回最终的符号对集合


class BioGptTokenizer(PreTrainedTokenizer):
    """
    Construct an FAIRSEQ Transformer tokenizer. Moses tokenization followed by Byte-Pair Encoding.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 构建一个 FAIRSEQ Transformer 分词器，使用 Moses 分词后接字节对编码

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        **kwargs
    ):
        # 初始化方法，接收词汇文件、合并文件以及其他参数
        super().__init__(
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs
        )
        # 调用父类的初始化方法，设置未知符号、结束符号和填充符号

        self.vocab_file = vocab_file
        self.merges_file = merges_file
        # 设置词汇文件和合并文件属性

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 构建包含特殊符号的输入方法，用于处理包含两个序列的输入
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens.
        """
        # 从序列或序列对构建模型输入，用于序列分类任务，通过连接和添加特殊符号

        if token_ids_1 is None:
            return token_ids_0
        else:
            return token_ids_0 + token_ids_1
        # 如果只有一个序列，则直接返回该序列；如果有两个序列，则连接它们后返回

    def get_vocab(self):
        # 获取词汇表方法
        with open(self.vocab_file, "r", encoding="utf-8") as reader:
            vocab = json.load(reader)
        return vocab
        # 打开词汇文件，加载词汇表并返回

    def tokenize(self, text):
        # 分词方法
        return text.split()
        # 使用空格分割文本并返回分词结果

    def convert_tokens_to_ids(self, tokens):
        # 将分词转换为 ID 方法
        vocab = self.get_vocab()
        return [vocab[token] for token in tokens]
        # 获取词汇表，将分词列表中的每个分词转换为对应的 ID 并返回

    def convert_ids_to_tokens(self, ids):
        # 将 ID 转换为分词方法
        vocab = self.get_vocab()
        return [list(vocab.keys())[list(vocab.values()).index(id)] for id in ids]
        # 获取词汇表，将 ID 列表中的每个 ID 转换为对应的分词并返回
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
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
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
            <Tip>
            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.
            </Tip>
        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
    """

    # 确定词汇文件名的键名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入的最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
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
        try:
            import sacremoses  # 尝试导入 sacremoses 库
        except ImportError:
            # 如果导入失败，抛出 ImportError 异常，提醒需要安装 sacremoses 库
            raise ImportError(
                "You need to install sacremoses to use BioGptTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        self.lang = "en"
        self.sm = sacremoses  # 将 sacremoses 赋值给 self.sm
        # 缓存 sm.MosesTokenizer 实例
        self.cache_moses_tokenizer = {}
        self.cache_moses_detokenizer = {}

        """ Initialisation"""
        # 用 utf-8 编码打开 vocab_file 文件，并将其内容加载为 JSON 格式，赋值给 self.encoder
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建 self.decoder 字典，将 self.encoder 的键值对反转
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 用 utf-8 编码打开 merges_file 文件，读取内容并按行分割，去除最后一个空行，赋值给 merges 列表
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        # 将 merges 列表中的每个元素按空格分割成元组，构成 merges 列表
        merges = [tuple(merge.split()[:2]) for merge in merges]
        # 创建 self.bpe_ranks 字典，将 merges 列表中的元素与其在列表中的索引号配对
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

        # 调用父类的初始化方法，并传入指定参数
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
        """Returns vocab size"""
        # 返回 self.encoder 的长度，即词汇表的大小
        return len(self.encoder)

    def get_vocab(self):
        # 返回包含 self.encoder 和 self.added_tokens_encoder 的字典
        return dict(self.encoder, **self.added_tokens_encoder)

    def moses_tokenize(self, text, lang):
        # 如果 lang 不在 self.cache_moses_tokenizer 中，则创建一个新的 sm.MosesTokenizer 实例并缓存
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = self.sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        # 使用缓存的 moses_tokenizer 对象对文本进行分词处理并返回结果
        return self.cache_moses_tokenizer[lang].tokenize(
            text, aggressive_dash_splits=True, return_str=False, escape=True
        )

    def moses_detokenize(self, tokens, lang):
        # 如果 lang 不在 self.cache_moses_detokenizer 中，则创建一个新的 sm.MosesDetokenizer 实例并缓存
        if lang not in self.cache_moses_detokenizer:
            moses_detokenizer = self.sm.MosesDetokenizer(lang=lang)
            self.cache_moses_detokenizer[lang] = moses_detokenizer
        # 使用缓存的 moses_detokenizer 对象对 tokens 进行反向处理并返回结果
        return self.cache_moses_detokenizer[lang].detokenize(tokens)
    def bpe(self, token):
        # 将输入的token进行BPE处理，生成新的词形
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        # 如果token已经被处理过，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 获取token中的所有字符对
        pairs = get_pairs(word)

        # 如果没有字符对，直接返回token后加上结束符的形式
        if not pairs:
            return token + "</w>"

        # 循环处理字符对，直到无法继续合并
        while True:
            # 找到优先级最低的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果这个字符对不在BPE词频表中，停止合并
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

                # 合并找到的字符对
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果只剩一个词元，停止合并
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        # 将词元列表转换为字符串形式返回
        word = " ".join(word)
        # 处理特殊情况下的结束符
        if word == "\n  </w>":
            word = "\n</w>"
        # 将处理后的结果缓存起来
        self.cache[token] = word
        return word

    def _tokenize(self, text, bypass_tokenizer=False):
        """Returns a tokenized string."""
        # 根据bypass_tokenizer的值选择是否使用分词器进行处理文本
        if bypass_tokenizer:
            text = text.split()
        else:
            text = self.moses_tokenize(text, self.lang)

        split_tokens = []
        # 对文本中的每个token进行BPE处理并拆分结果
        for token in text:
            if token:
                split_tokens.extend(list(self.bpe(token).split(" ")))

        return split_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据词汇表将token转换为对应的ID，未知token返回未知token的ID
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据词汇表将ID转换为对应的token，未知ID返回未知token
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 移除BPE处理过程中的空格和结束符，将tokens拼接成文本
        tokens = [t.replace(" ", "").replace("</w>", " ") for t in tokens]
        tokens = "".join(tokens).split()
        # 使用Moses库的反分词函数，将tokens还原为文本
        text = self.moses_detokenize(tokens, self.lang)
        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ):
        # 构建输入序列，添加特殊token以及可能的第二个序列的token
    def build_model_inputs(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BioGPT sequence has the following format:

        - single sequence: `</s> X `
        - pair of sequences: `</s> A </s> B `

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # If only one sequence provided, return with a single separator token added at the beginning
        if token_ids_1 is None:
            return [self.sep_token_id] + token_ids_0
        # If two sequences provided, construct the input with a separator token between and at the ends of each sequence
        sep = [self.sep_token_id]
        return sep + token_ids_0 + sep + token_ids_1

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
        # For sequences without special tokens, return a mask with 1s for special tokens and 0s for sequence tokens
        # no bos used in fairseq
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))
        return [1] + ([0] * len(token_ids_0))
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A FAIRSEQ
        Transformer sequence pair mask has the following format:

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
        sep = [self.sep_token_id]

        # 如果没有传入第二个序列的 token_ids_1，则只返回第一个部分的 mask（全为 0）
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        # 返回一个由 token_ids_0 和 token_ids_1 组成的 mask，其中 token_ids_0 部分全为 0，token_ids_1 部分全为 1
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and merges files to the specified directory.

        Args:
            save_directory (str):
                Directory path where the vocabulary files will be saved.
            filename_prefix (str, optional):
                Prefix to be added to the filenames of vocabulary and merges files.

        Returns:
            Tuple[str]: Tuple containing the paths to the saved vocabulary and merges files.
        """
        # 如果保存目录不存在，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构造保存的词汇表文件名和合并文件名的路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将词汇表写入到 JSON 格式的文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将 BPE token 和其索引写入到合并文件中
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

        # 返回保存的词汇表文件和合并文件的路径
        return vocab_file, merge_file

    def __getstate__(self):
        """
        Get the state of the XLMTokenizer object for pickling.
        """
        # 复制对象的状态字典，并设置 'sm' 为 None，以便序列化时忽略 'sm'
        state = self.__dict__.copy()
        state["sm"] = None
        return state

    def __setstate__(self, d):
        """
        Set the state of the XLMTokenizer object from a dictionary.
        """
        # 从字典中恢复对象的状态
        self.__dict__ = d

        # 检查是否安装了 sacremoses 库，如果没有则抛出 ImportError
        try:
            import sacremoses
        except ImportError:
            raise ImportError(
                "You need to install sacremoses to use XLMTokenizer. "
                "See https://pypi.org/project/sacremoses/ for installation."
            )

        # 将 sacremoses 模块引入 self.sm
        self.sm = sacremoses
```