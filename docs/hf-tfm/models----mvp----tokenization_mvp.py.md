# `.\models\mvp\tokenization_mvp.py`

```py
# coding=utf-8
# Copyright 2022 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

import json  # 导入处理 JSON 格式数据的模块
import os  # 导入操作系统相关功能的模块
from functools import lru_cache  # 导入用于缓存函数结果的装饰器
from typing import List, Optional, Tuple  # 导入用于类型注解的模块

import regex as re  # 导入正则表达式模块

from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入特定的 tokenization_utils 模块
from ...utils import logging  # 导入日志记录工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}  # 词汇表文件名的映射字典

# See all MVP models at https://huggingface.co/models?filter=mvp
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "RUCAIBox/mvp": "https://huggingface.co/RUCAIBox/mvp/resolve/main/vocab.json",
    },
    "added_tokens.json": {
        "RUCAIBox/mvp": "https://huggingface.co/RUCAIBox/mvp/resolve/main/added_tokens.json",
    },
    "merges_file": {
        "RUCAIBox/mvp": "https://huggingface.co/RUCAIBox/mvp/resolve/main/merges.txt",
    },
}  # 预训练模型的词汇文件映射字典

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "RUCAIBox/mvp": 1024,
}  # 预训练模型的位置编码大小映射字典

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
    )  # 定义包含 UTF-8 字节范围的列表 bs
    cs = bs[:]  # 复制 bs 到 cs
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]  # 将 cs 中的数值转换为对应的 Unicode 字符
    return dict(zip(bs, cs))  # 返回 UTF-8 字节到 Unicode 字符的映射字典

def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()  # 创建一个空集合 pairs，用于存放单词中的符号对
    prev_char = word[0]  # 获取单词的第一个字符
    for char in word[1:]:  # 遍历单词的其余字符
        pairs.add((prev_char, char))  # 将前一个字符和当前字符作为符号对添加到集合中
        prev_char = char  # 更新前一个字符为当前字符
    return pairs  # 返回符号对的集合

class MvpTokenizer(PreTrainedTokenizer):
    """
    Constructs a MVP tokenizer, which is smilar to the RoBERTa tokenizer, using byte-level Byte-Pair-Encoding.
    """
    # 构造函数，初始化 MVP tokenizer 对象
    # 从transformers库导入MvpTokenizer类
    >>> from transformers import MvpTokenizer
    
    # 使用预训练模型"RUCAIBox/mvp"实例化一个tokenizer对象
    >>> tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
    
    # 对文本"Hello world"进行tokenization，并获取其input_ids（输入ID）
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]
    
    # 对文本" Hello world"进行tokenization（在单词前加上空格），并获取其input_ids（输入ID）
    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
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
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (MVP tokenizer detect beginning of words by the preceding space).
    """

    # 定义一些常量和映射，用于处理预训练模型的输入和输出
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs,
    ):
        # 初始化方法，用于创建一个新的对象实例，并初始化其属性
        bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token

        # 如果 mask_token 是字符串类型，则创建一个特殊的 AddedToken 对象，左去空格，表示它在标记化时不带空格
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
        
        # 使用 UTF-8 编码打开词汇文件，并加载其中的 JSON 数据到 self.encoder 字典中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        
        # 创建 self.decoder 字典，将 self.encoder 的键值对反转，用于从索引解码到词汇
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # 设置处理解码错误的方法（默认是替换错误）
        self.errors = errors
        
        # 初始化字节到 Unicode 编码的映射
        self.byte_encoder = bytes_to_unicode()
        
        # 创建字节解码器，将字节到 Unicode 编码的映射反转，用于从 Unicode 解码到字节
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 使用 UTF-8 编码打开 merges_file 文件，并读取其中的 BPE 合并规则
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        
        # 将每行 BPE 合并规则转换为元组列表
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        
        # 创建 self.bpe_ranks 字典，将 BPE 合并规则和其索引值构成键值对
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        
        # 创建一个空的缓存字典
        self.cache = {}
        
        # 设置是否在词汇前添加空格的选项
        self.add_prefix_space = add_prefix_space
        
        # 使用正则表达式创建 self.pat 模式，用于词汇分割和合并
        # 包括缩略词、字母、数字、非空格非字母非数字字符、空格等的匹配
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # 调用父类的初始化方法，传递相应参数和关键字参数
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
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表的大小，即 self.encoder 字典的键值对数量
        return len(self.encoder)

    def get_vocab(self):
        # 获取词汇表，包括已添加的特殊 token
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab
    def _tokenize(self, text):
        """Tokenize a string."""
        # 定义一个空列表，用于存储经过 BPE 处理后的 token
        bpe_tokens = []
        # 使用正则表达式找到文本中所有符合模式的 token，并进行处理
        for token in re.findall(self.pat, text):
            # 将每个 token 编码成字节，并通过字节编码器映射为 unicode 字符串，避免 BPE 中的控制标记（在本例中是空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 将经过 BPE 处理后的 token 拆分成多个子 token，并加入到 bpe_tokens 列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        # 返回经过 BPE 处理后的所有 token
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据 token 在词汇表中查找对应的 id，如果找不到则返回未知 token 对应的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据 id 查找词汇表中对应的 token
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 tokens 列表中的所有 token 拼接成一个字符串
        text = "".join(tokens)
        # 将字符串转换为字节，并根据字节解码器将其解码为 utf-8 编码的字符串，处理可能的错误情况
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        # 返回转换后的字符串
        return text
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
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

        # 将编码器内容以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将 BPE 标记和对应的索引写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
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

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A MVP sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果只有一个序列，则添加开头和结尾的特殊标记
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果有两个序列，则按照格式添加特殊标记
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        """
        Return a mask indicating the positions of special tokens in the input sequences.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs corresponding to the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs corresponding to the second sequence.
            already_has_special_tokens (`bool`):
                Whether the inputs already include special tokens or not.

        Returns:
            `List[int]`: Mask indicating the positions of special tokens (1 for special token, 0 for regular token).
        """
        # 如果输入已经包含特殊标记，则直接返回全零的掩码
        if already_has_special_tokens:
            return [0] * len(token_ids_0)

        # 初始化一个全零掩码
        special_tokens_mask = [0] * len(token_ids_0)

        # 设置第一个序列的开头和结尾标记位置
        special_tokens_mask[0] = 1  # CLS token
        special_tokens_mask[-1] = 1  # SEP token

        # 如果有第二个序列，设置第二个序列的开头和结尾标记位置
        if token_ids_1 is not None:
            special_tokens_mask.extend([1] * len(token_ids_1) + [1])  # two SEP tokens

        return special_tokens_mask
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
        # 如果已经包含特殊标记，则调用父类方法获取特殊标记的掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果没有第二个 token_ids_1，则返回一个列表，表示有特殊标记的序列
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 否则返回一个列表，表示有特殊标记的序列对
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. MVP does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 分隔符和类别标记的 ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果没有第二个 token_ids_1，则返回一个全零列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则返回一个全零列表，用于序列对的掩码
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经分割成单词或者需要在文本前加空格，并且文本的第一个字符不是空格，则在文本前加一个空格
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)
```