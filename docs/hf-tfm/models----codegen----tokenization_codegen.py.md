# `.\models\codegen\tokenization_codegen.py`

```
# 设置文件编码为 UTF-8
# 版权声明
#
#
# 版权声明，根据 Apache License, Version 2.0 授权
# 用户可以在遵守许可协议的前提下使用该文件
# 可以在以下网址获取许可协议的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得使用本文件
# 根据协议分发的软件将基于“原样”分发，
# 没有任何明示或暗示条件，包括但不限于
# 声明或保证的条件
# 为特定语言约束权限和限制条件，请参见协议
"""Tokenization classes for CodeGen"""

# 导入必要的库
import json
import os
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import numpy as np
import regex as re
from ...utils import is_tf_available, is_torch_available, logging

# 根据环境加载相应的库
if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf

# 导入必要的代码
from ...tokenization_utils import AddedToken, PreTrainedTokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 词汇文件名和对应的文件名
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json",
    },
    "merges_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt",
    },
}

# 预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Salesforce/codegen-350M-mono": 2048,
}

# 使用缓存来存储结果，减少计算时间
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

# 获取单词中的符号对
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

# CodeGenTokenizer 类
class CodeGenTokenizer(PreTrainedTokenizer):
    """
    # 构建一个 CodeGen 分词器。基于字节级别的 Byte-Pair-Encoding。
    
    # 此分词器经过训练，将空格视为标记的一部分（有点像 sentencepiece），因此一个单词会根据是否在句子开头（没有空格）而被编码为不同的形式：
    
    >>> from transformers import CodeGenTokenizer
    
    >>> tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]
    
    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    
    # 通过在实例化此分词器时或在调用某些文本时传递 `add_prefix_space=True`，可以避免此行为，但由于模型未经过此方式预训练，可能会导致性能降低。
    
    <Tip>
    
    # 当与 `is_split_into_words=True` 一起使用时，此分词器将在每个单词之前添加一个空格（即使是第一个单词）。
    
    </Tip>
    
    # 此分词器继承自 [`PreTrainedTokenizer`]，其中包含大部分主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    
    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        merges_file (`str`):
            合并文件的路径。
        errors (`str`, *可选*, 默认为 `"replace"`):
            解码字节为 UTF-8 时要遵循的范例。有关更多信息，请参阅 [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode)。
        unk_token (`str`, *可选*, 默认为 `"<|endoftext|>"`):
            未知标记。词汇表中没有的标记无法转换为 ID，并被设置为此标记。
        bos_token (`str`, *可选*, 默认为 `"<|endoftext|>"`):
            序列的开头标记。
        eos_token (`str`, *可选*, 默认为 `"<|endoftext|>"`):
            序列的结尾标记。
        pad_token (`str`, *可选*):
            用于填充的标记，例如在批处理不同长度的序列时。
        add_prefix_space (`bool`, *可选*, 默认为 `False`):
            是否在输入前添加一个初始空格。这允许将前导词视为任何其他词。（CodeGen 分词器通过前导空格检测单词的开头）。
        add_bos_token (`bool`, *可选*, 默认为 `False`):
            是否在序列开头添加一个序列开始标记。
    """
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化方法，接受参数：词汇文件、合并文件、错误处理方式、未知词特殊标记、开始标记、结束标记、填充标记、是否添加前缀空格、是否添加开始标记
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
        # 如果开始标记是字符串，将其转换为特殊标记
        bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
        # 如果结束标记是字符串，将其转换为特殊标记
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        # 如果未知词标记是字符串，将其转换为特殊标记
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        # 如果填充标记是字符串，将其转换为特殊标记
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token
        # 设置是否添加开始标记
        self.add_bos_token = add_bos_token

        # 打开词汇文件，使用 UTF-8 编码读取文件内容，并加载到 encoder 字典中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建 decoder 字典，键值对互换，用于解码
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置错误处理方式
        self.errors = errors  # how to handle errors in decoding
        # 创建字节到 Unicode 的编码转换器
        self.byte_encoder = bytes_to_unicode()
        # 创建 Unicode 到字节的解码转换器
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 打开合并文件，使用 UTF-8 编码读取文件内容，提取 BPE 合并规则
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建 BPE 合并规则字典，记录合并规则及其索引
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存
        self.cache = {}
        # 设置是否添加前缀空格
        self.add_prefix_space = add_prefix_space

        # 编译正则表达式，匹配单词、数字、非空白字符，以及不以空白字符结尾的空白字符
        # 添加了 re.IGNORECASE 以便 BPE 合并可以针对大小写形式的缩写
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # 调用父类的初始化方法
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

    # 获取词汇表大小的属性方法
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 获取词汇表的方法，返回词汇表及其扩展词汇的字典
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
    def bpe(self, token):
        # 如果标记已经在缓存中，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将标记转换为字符元组
        word = tuple(token)
        # 获取标记的所有字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则直接返回原始标记
        if not pairs:
            return token

        # 不断地将标记中的字符对替换为 BPE 编码，直到无法再替换为止
        while True:
            # 选择出现频率最低的字符对进行替换
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该字符对不在 BPE 编码表中，则停止替换
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 遍历标记中的字符，根据字符对进行替换
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
            # 如果替换后的标记长度为1，则停止替换
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将标记转换为字符串形式，并加入缓存中
        word = " ".join(word)
        self.cache[token] = word
        return word

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 如果需要添加起始符号，则加入起始符号对应的标记 ID
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        # 将第一个标记序列与起始符号标记序列合并
        output = bos_token_ids + token_ids_0

        # 如果存在第二个标记序列，则将其与起始符号标记序列和第一个标记序列合并
        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def _tokenize(self, text):
        """Tokenize a string."""
        # 使用正则表达式找出文本中的所有标记，并将其进行 BPE 编码
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # 将所有字节映射为 Unicode 字符串，避免 BPE 中的控制标记（在我们的情况下是空格）
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将标记转换为对应的标记 ID，若标记不存在于词汇表中，则返回未知标记 ID
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将标记 ID 转换为对应的标记，若标记 ID 不存在，则返回 None
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将标记序列转换为单个字符串
        text = "".join(tokens)
        # 将标记字符串中的每个字节按照词汇表中的映射转换为 Unicode 字符，并解码为字符串形式
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 将词汇表保存到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则报错
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

        # 写入词汇表到文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入合并信息到文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            # 写入合并文件的版本信息
            writer.write("#version: 0.2\n")
            # 遍历 BPE 字典，将 BPE 令牌及其索引写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                # 检查 BPE 合并索引是否连续
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                # 写入 BPE 令牌
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回词汇表文件路径和合并文件路径
        return vocab_file, merge_file

    # 准备进行分词的文本预处理
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # 获取是否在文本前添加空格的设置，默认使用对象的设置
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经分词或需要在文本前添加空格，则在文本前添加一个空格
        if is_split_into_words or add_prefix_space:
            text = " " + text
        # 返回处理后的文本和可能存在的其他参数
        return (text, kwargs)

    # 解码器，将 token 转换为文本
    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        truncate_before_pattern: Optional[List[str]] = None,
        **kwargs,
    def decode(self, token_ids) -> str:
        """
        将一系列ID转换为字符串，使用标记器和词汇表，具有删除特殊令牌和清理标记化空格的选项。

        类似于执行`self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`。

        参数：
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                标记化的输入ID列表。可以使用`__call__`方法获得。
            skip_special_tokens (`bool`, *optional*, 默认为`False`):
                是否删除解码中的特殊令牌。
            clean_up_tokenization_spaces (`bool`, *optional*):
                是否清理标记化空格。如果`None`，则默认为`self.clean_up_tokenization_spaces`（在`tokenizer_config`中可用）。
            truncate_before_pattern (`List[str]`, *optional*, 默认为`None`):
                一个正则表达式字符串列表，将用于截断返回的字符串。这可以用于移除额外的代码片段（例如，如果在新行的开头观察到注释符号"#"，则截断）。一个示例模式可能是`["^#", re.escape("<|endoftext|>"), "^'''", "\n\n\n"]`。
            kwargs (其他关键字参数，*optional*):
                将传递给底层模型特定的解码方法。

        返回：
            `str`: 解码后的句子。
        """
        decoded_text = super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        if truncate_before_pattern is not None and len(truncate_before_pattern) > 0:
            decoded_text = self.truncate(decoded_text, truncate_before_pattern)

        return decoded_text

    def truncate(self, completion, truncate_before_pattern):
        def find_re(string, pattern, start_pos):
            m = pattern.search(string, start_pos)
            return m.start() if m else -1

        terminals = [re.compile(pattern, re.MULTILINE) for pattern in truncate_before_pattern]

        prints = list(re.finditer("^print", completion, re.MULTILINE))

        if len(prints) > 1:
            completion = completion[: prints[1].start()]

        defs = list(re.finditer("^def", completion, re.MULTILINE))

        if len(defs) > 1:
            completion = completion[: defs[1].start()]

        start_pos = 0

        terminals_pos = [
            pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1
        ]

        if len(terminals_pos) > 0:
            return completion[: min(terminals_pos)]
        else:
            return completion
```