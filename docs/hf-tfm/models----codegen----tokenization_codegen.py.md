# `.\models\codegen\tokenization_codegen.py`

```py
# coding=utf-8
# Copyright 2022 The Salesforce authors, The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for CodeGen"""

# 导入必要的库和模块
import json
import os
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import regex as re

# 导入日志模块和其它辅助函数
from ...utils import is_tf_available, is_torch_available, logging, to_py_obj

# 检查类型，根据可用的深度学习框架导入相关模块
if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义用于存储词汇表和合并文件名的字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 预训练模型的词汇表文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json",
    },
    "merges_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt",
    },
}

# 预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Salesforce/codegen-350M-mono": 2048,
}

# 用 LRU 缓存装饰器定义函数，将字节映射为 Unicode 字符
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
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# 定义函数，返回单词中的符号对集合
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# 定义一个类，继承自 PreTrainedTokenizer，用于代码生成的分词器
class CodeGenTokenizer(PreTrainedTokenizer):
    # 定义了 CodeGenTokenizer 类，基于字节级的 Byte-Pair-Encoding（BPE）进行分词。
    
    # 该分词器训练时将空格视为标记的一部分（类似 sentencepiece），因此同一个词在句首和非句首会有不同的编码：
    # - 如果词在句首没有空格，编码不同；
    # - 通过示例展示了不同编码的效果。
    
    # 可以通过在实例化或调用时传递 `add_prefix_space=True` 来修改这种行为，但是因为模型不是以此方式预训练的，
    # 可能会导致性能下降。
    
    # 当设置 `is_split_into_words=True` 时，分词器会在每个词之前添加一个空格，即使是第一个词也是如此。
    
    # CodeGenTokenizer 继承自 PreTrainedTokenizer，该类包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    
    # Args: 定义了构造函数的参数说明，包括：
    # - vocab_file (`str`): 词汇文件的路径。
    # - merges_file (`str`): 合并文件的路径。
    # - errors (`str`, *optional*, defaults to `"replace"`): 解码字节为 UTF-8 时的错误处理方式。
    # - unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`): 未知标记，用于不在词汇中的标记。
    # - bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`): 序列开始标记。
    # - eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`): 序列结束标记。
    # - pad_token (`str`, *optional*): 用于填充的标记，例如在批处理不同长度序列时使用。
    # - add_prefix_space (`bool`, *optional*, defaults to `False`): 是否在输入的开头添加空格，以便将首个词视为普通词。
    # - add_bos_token (`bool`, *optional*, defaults to `False`): 是否在序列开始处添加一个开始标记。
    vocab_files_names = VOCAB_FILES_NAMES  # 词汇文件名列表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练位置嵌入的最大模型输入大小
    model_input_names = ["input_ids", "attention_mask"]  # 模型输入的名称列表
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token
        self.add_bos_token = add_bos_token

        # 打开并读取词汇文件，使用 UTF-8 编码
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建词汇的反向映射，从索引到词汇的映射
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 用于处理解码中的错误
        # 创建字节到 Unicode 的编码器和解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 打开并读取 BPE 合并文件，使用 UTF-8 编码
        with open(merges_file, encoding="utf-8") as merges_handle:
            # 读取文件内容并解析 BPE 合并规则
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建 BPE 合并的排序字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # 设置正则表达式模式，用于标记化文本
        # 应该添加 re.IGNORECASE 以便对大小写不敏感的情况进行 BPE 合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # 调用父类的初始化方法，传递参数和关键字参数
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表的大小
        return len(self.encoder)

    def get_vocab(self):
        # 返回词汇表及其扩展标记的编码器
        return dict(self.encoder, **self.added_tokens_encoder)
    def _tokenize(self, text):
        """Tokenize a string."""
        # 初始化空列表，用于存储分词后的结果
        bpe_tokens = []
        # 使用正则表达式找到文本中的所有匹配项，并遍历每个匹配到的 token
        for token in re.findall(self.pat, text):
            # 将 token 编码为 UTF-8 字节，并通过 byte_encoder 映射为 Unicode 字符串，
            # 避免 BPE 中的控制标记（在这里是空格）
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            # 对经过 BPE 处理后的 token 进行拆分，并添加到 bpe_tokens 中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        # 返回 BPE 处理后的 token 列表
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用 encoder 字典将 token 转换为对应的 id，如果 token 不存在则使用 unk_token
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用 decoder 字典将 index 转换为对应的 token
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 tokens 列表中的所有 token 连接成一个字符串
        text = "".join(tokens)
        # 将 UTF-8 字节序列转换为字符串，使用 byte_decoder 进行解码，处理可能的编码错误
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        # 返回转换后的字符串
        return text
    # 保存词汇表到指定目录下的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构建词汇表文件路径，包括可选的前缀和文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建合并文件路径，包括可选的前缀和文件名
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器中的内容以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将 BPE 标记和它们的索引写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 记录警告信息，提示 BPE 合并索引不是连续的，可能的损坏
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的词汇表文件路径和合并文件路径
        return vocab_file, merge_file


    # 准备文本以便进行标记化处理
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 获取是否添加前缀空格的设置，若未提供则使用默认设置
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经分割成单词或需要添加前缀空格，则在文本前加空格
        if is_split_into_words or add_prefix_space:
            text = " " + text
        # 返回处理后的文本和可能更新的参数
        return (text, kwargs)


    # 解码操作，将标记 ID 转换为文本
    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        truncate_before_pattern: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            truncate_before_pattern (`List[str]`, *optional*, defaults to `None`):
                A list of regular expression strings that will be used to truncate the returned string. This can be
                used to remove extra pieces of code (e.g. truncate if observing a comment symbol "#" at the beginning
                of a new line). An example pattern could be `["^#", re.escape("<|endoftext|>"), "^'''", "\n\n\n"]`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """

        # Convert `token_ids` to Python object
        token_ids = to_py_obj(token_ids)

        # Decode the `token_ids` into text using inherited `_decode` method
        decoded_text = super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        # Truncate the decoded text based on `truncate_before_pattern` if specified
        if truncate_before_pattern is not None and len(truncate_before_pattern) > 0:
            decoded_text = self.truncate(decoded_text, truncate_before_pattern)

        # Return the decoded text as a string
        return decoded_text

    def truncate(self, completion, truncate_before_pattern):
        """
        Truncates `completion` text based on specified `truncate_before_pattern`.

        Args:
            completion (str): The text to truncate.
            truncate_before_pattern (List[str]): List of regular expressions to determine truncation points.

        Returns:
            str: The truncated text.
        """

        def find_re(string, pattern, start_pos):
            """
            Helper function to find the position of a pattern in a string.

            Args:
                string (str): The string to search within.
                pattern (Pattern): The compiled regular expression pattern.
                start_pos (int): The starting position of the search.

            Returns:
                int: The position of the pattern in the string, or -1 if not found.
            """
            m = pattern.search(string, start_pos)
            return m.start() if m else -1

        # Compile regular expression patterns for each element in `truncate_before_pattern`
        terminals = [re.compile(pattern, re.MULTILINE) for pattern in truncate_before_pattern]

        # Find all occurrences of "^print" in `completion` and limit to the second occurrence
        prints = list(re.finditer("^print", completion, re.MULTILINE))

        if len(prints) > 1:
            completion = completion[: prints[1].start()]

        # Find all occurrences of "^def" in `completion` and limit to the second occurrence
        defs = list(re.finditer("^def", completion, re.MULTILINE))

        if len(defs) > 1:
            completion = completion[: defs[1].start()]

        start_pos = 0

        # Find positions of all patterns in `truncate_before_pattern` within `completion`
        terminals_pos = [
            pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1
        ]

        # Return `completion` truncated before the smallest found position, or as is if no positions found
        if len(terminals_pos) > 0:
            return completion[: min(terminals_pos)]
        else:
            return completion
```