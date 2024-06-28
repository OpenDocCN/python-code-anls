# `.\models\deberta\tokenization_deberta.py`

```py
# coding=utf-8
# Copyright 2020 Microsoft and the HuggingFace Inc. team.
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
""" Tokenization class for model DeBERTa."""

import json
import os
from typing import List, Optional, Tuple

import regex as re  # 导入 regex 库，用于支持正则表达式的操作

from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入基础的标记化工具和预训练的标记器
from ...utils import logging  # 导入日志记录工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名的常量字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# 定义预训练模型与词汇文件映射的常量字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/vocab.json",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/vocab.json",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/vocab.json",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/vocab.json",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/vocab.json",
        "microsoft/deberta-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/merges.txt",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/merges.txt",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/merges.txt",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/merges.txt",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/merges.txt",
        "microsoft/deberta-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/merges.txt"
        ),
    },
}

# 定义预训练模型位置编码大小的常量字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
    "microsoft/deberta-xlarge": 512,
    "microsoft/deberta-base-mnli": 512,
    "microsoft/deberta-large-mnli": 512,
    "microsoft/deberta-xlarge-mnli": 512,
}

# 定义预训练模型初始化配置的常量字典
PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-base": {"do_lower_case": False},
    "microsoft/deberta-large": {"do_lower_case": False},
}

# 从transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode函数复制过来的函数
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    # 定义一个列表，包含所有可打印ASCII字符的Unicode码点范围
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    
    # 复制bs列表到cs列表
    cs = bs[:]
    # 初始化计数器n为0
    n = 0
    # 遍历0到255的所有字节值
    for b in range(2**8):
        # 如果b不在bs列表中，则将b添加到bs列表，将2**8 + n添加到cs列表，并增加n计数器
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    
    # 将cs列表中的每个数值转换为对应的Unicode字符，并组成新的列表
    cs = [chr(n) for n in cs]
    
    # 返回一个将utf-8字节映射到Unicode字符串的字典
    return dict(zip(bs, cs))
# 从 transformers.models.gpt2.tokenization_gpt2.get_pairs 复制而来的函数，用于生成单词中的符号对集合
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # 初始化一个空集合，用于存放符号对
    pairs = set()
    # 从单词的第一个字符开始遍历到倒数第二个字符
    prev_char = word[0]
    for char in word[1:]:
        # 将当前字符与前一个字符组成一个符号对，并添加到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符，为下一次迭代做准备
        prev_char = char
    # 返回生成的符号对集合
    return pairs


class DebertaTokenizer(PreTrainedTokenizer):
    """
    Construct a DeBERTa tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import DebertaTokenizer

    >>> tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    >>> tokenizer("Hello world")["input_ids"]
    [1, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [1, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # DeBERTa 分词器的构造函数，基于字节级字节对编码
    # 此分词器已经训练成将空格视为标记的一部分（类似 sentencepiece），因此一个单词的编码取决于它是否在句子开头（没有空格）

    # 示例代码块结束
    # 定义一个类，用于处理特定的词汇和标记化文件
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    
    # 初始化方法，设置类的基本属性和参数
    def __init__(
        self,
        vocab_file,            # 词汇文件的路径
        merges_file,           # 合并文件的路径
        errors="replace",      # 解码字节到UTF-8时出现错误的处理策略
        bos_token="[CLS]",     # 序列开始标记
        eos_token="[SEP]",     # 序列结束标记
        sep_token="[SEP]",     # 分隔标记，用于多序列构建或特殊标记序列的最后一个标记
        cls_token="[CLS]",     # 分类器标记，在序列分类时使用
        unk_token="[UNK]",     # 未知标记，用于处理不在词汇表中的词汇
        pad_token="[PAD]",     # 填充标记，用于处理不同长度序列的批处理
        mask_token="[MASK]",   # 掩码标记，用于掩码语言建模
        add_prefix_space=False,  # 是否在输入前添加初始空格，用于Deberta分词器
        add_bos_token=False,   # 是否在输入前添加初始序列结束标记
        **kwargs,              # 其他可能的参数
    ):
    ):
        bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        self.add_bos_token = add_bos_token

        # 使用指定的词汇文件打开并加载词汇表，使用 UTF-8 编码
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 根据编码表生成解码表
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 处理解码时的错误策略
        # 初始化字节编码器和解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 使用指定的合并文件打开并读取 BPE 合并规则
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建 BPE 合并规则到索引的映射字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}  # 初始化缓存字典
        self.add_prefix_space = add_prefix_space  # 控制是否在添加特殊标记时加入前置空格

        # 使用正则表达式定义 tokenization 的模式，支持对缩写单词的大小写不敏感处理
        # 这里添加 re.IGNORECASE 标记，以便支持合并首字母大写的缩写单词
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化方法，传入参数设置
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            **kwargs,
        )

    @property
    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.vocab_size 复制而来
    def vocab_size(self):
        return len(self.encoder)

    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.get_vocab 复制而来
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe 复制而来
    def bpe(self, token):
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        
        # 将 token 转换成元组形式
        word = tuple(token)
        # 获取 token 的所有可能的 bigram 对
        pairs = get_pairs(word)

        # 如果不存在 bigram 对，直接返回 token
        if not pairs:
            return token

        # 开始进行 BPE 合并操作，直到无法再合并为止
        while True:
            # 找到当前权重最小的 bigram 对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果找到的 bigram 不在预先定义的 BPE 权重中，停止合并
            if bigram not in self.bpe_ranks:
                break
            # 分解当前的 word，将符合条件的 bigram 合并为一个 token
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
            # 如果 word 只剩一个 token，停止合并
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        # 将合并后的 word 转换为字符串形式
        word = " ".join(word)
        # 将结果存入缓存
        self.cache[token] = word
        return word

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过连接和添加特殊 token 构建用于序列分类任务的模型输入。DeBERTa 的序列格式如下：

        - 单个序列: [CLS] X [SEP]
        - 序列对: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊 token 的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表（用于序列对）。

        Returns:
            `List[int]`: 带有适当特殊 token 的输入 ID 列表。
        """
        if token_ids_1 is None:
            # 返回只有一个序列的情况下的输入列表
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 构建包含两个序列的输入列表，包括特殊 token
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        """
        返回一个 mask 列表，指示哪些 token 是特殊 token。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的 token ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 token ID 列表（用于序列对）。
            already_has_special_tokens (`bool`):
                是否已经包含了特殊 token。

        Returns:
            `List[int]`: mask 列表，每个元素为 1（特殊 token）或 0（普通 token）。
        """
        # 初始化 mask 列表，默认为全 0
        special_tokens_mask = [0] * len(token_ids_0)

        # 如果已经有特殊 token，直接返回全 1 的 mask 列表
        if already_has_special_tokens:
            return special_tokens_mask

        # 设置开始和结束的特殊 token 位置为 1
        special_tokens_mask[0] = 1  # CLS token
        special_tokens_mask[-1] = 1  # SEP token

        # 如果有第二个序列，则将第二个序列的 SEP token 位置也设置为 1
        if token_ids_1 is not None:
            special_tokens_mask += [1] * len(token_ids_1)  # SEP token for second sequence
        
        return special_tokens_mask
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

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
        if already_has_special_tokens:
            # If the token list already has special tokens, delegate the masking to the base class method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If no special tokens exist in the token lists, construct a mask with appropriate positions
        if token_ids_1 is None:
            # For a single sequence, prepend and append with special tokens
            return [1] + ([0] * len(token_ids_0)) + [1]
        else:
            # For sequence pairs, prepend and append special tokens for both sequences
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A DeBERTa
        sequence pair mask has the following format:

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
        sep = [self.sep_token_id]  # Separator token ID
        cls = [self.cls_token_id]  # Classification token ID

        if token_ids_1 is None:
            # If there's only one sequence, return a mask with zeros for its tokens
            return len(cls + token_ids_0 + sep) * [0]
        else:
            # For sequence pairs, return a mask with zeros for the first sequence and ones for the second
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._tokenize
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # Encode each token byte and split it using BPE
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_token_to_id
    # 使用词汇表将一个 token（字符串）转换为对应的 id
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 从词汇表将一个 id（整数）转换为对应的 token（字符串）
    def _convert_id_to_token(self, index):
        return self.decoder.get(index)

    # 将一个 token 序列（字符串列表）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        # 将 tokens 列表连接成一个字符串
        text = "".join(tokens)
        # 将字符串转换为字节数组，并使用 byte_decoder 进行解码成 utf-8 编码的文本
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    # 将词汇表保存到指定目录下的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接词汇表文件和合并文件的保存路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将词汇表编码（encoder）以 JSON 格式写入到 vocab_file 中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将 BPE 合并信息写入到 merge_file 中
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历 bpe_ranks 中的项目，按 token_index 排序，并将每个 BPE token 列表写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果 BPE 合并索引不是连续的，记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的 vocab_file 和 merge_file 的路径
        return vocab_file, merge_file

    # 准备文本进行 tokenization 前的预处理操作
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 获取是否需要添加前缀空格的参数，默认使用 self.add_prefix_space
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经分成单词或者需要添加前缀空格，并且第一个字符不是空白字符，则在文本前添加空格
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        # 返回处理后的文本和额外的参数
        return (text, kwargs)
```