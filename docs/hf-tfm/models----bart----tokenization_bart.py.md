# `.\transformers\models\bart\tokenization_bart.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入所需的库
import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import regex as re

# 从 Hugging Face 库中导入相关模块
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/vocab.json",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/vocab.json",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/vocab.json",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/vocab.json",
    },
    "merges_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/merges.txt",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/merges.txt",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/merges.txt",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/merges.txt",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/merges.txt",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/merges.txt",
    },
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/bart-base": 1024,
    "facebook/bart-large": 1024,
    "facebook/bart-large-mnli": 1024,
    "facebook/bart-large-cnn": 1024,
    "facebook/bart-large-xsum": 1024,
    "yjernite/bart_eli5": 1024,
}

# 使用 lru_cache 装饰器缓存函数的结果
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    # 定义一个函数，用于生成 utf-8 字节和 Unicode 字符之间的查找表
    def bytes_to_unicode():
        # ASCII 范围内的可打印字符以及常见的扩展字符组成的字节列表
        bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        )
        # 将 bs 复制给 cs
        cs = bs[:]
        # n 初始化为 0，用于记录添加的字符数量
        n = 0
        # 遍历 0 到 255 的所有字节
        for b in range(2**8):
            # 如果字节 b 不在 bs 中，则将其添加到 bs 和 cs 中
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        # 将 cs 中的整数转换为对应的 Unicode 字符
        cs = [chr(n) for n in cs]
        # 将 bs 和 cs 中的元素一一对应组成字典，并返回
        return dict(zip(bs, cs))
# 定义一个函数用于获取单词中的符号对集合
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    单词表示为符号元组（符号为可变长度字符串）。
    """
    # 初始化一个空集合用于存储符号对
    pairs = set()
    # 获取单词中的第一个符号
    prev_char = word[0]
    # 遍历单词中的每个符号
    for char in word[1:]:
        # 将当前符号与前一个符号组成一个符号对，并添加到符号对集合中
        pairs.add((prev_char, char))
        # 更新前一个符号为当前符号，为下一轮循环做准备
        prev_char = char
    # 返回符号对集合
    return pairs


# 定义一个 BART 分词器类，继承自 PreTrainedTokenizer
class BartTokenizer(PreTrainedTokenizer):
    """
    Constructs a BART tokenizer, which is smilar to the ROBERTa tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import BartTokenizer

    >>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```py

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.


    """
    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        merges_file (`str`):
            合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            将字节解码为 UTF-8 时采用的范例。有关更多信息，请参见[bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode)。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            在预训练期间用作序列开头的标记。可用作序列分类器标记。

            <Tip>

            在使用特殊标记构建序列时，此标记并非用于序列开头的标记。用于开头的标记是 `cls_token`。

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            序列的结束标记。

            <Tip>

            在使用特殊标记构建序列时，此标记并非用于序列结束的标记。用于结束的标记是 `sep_token`。

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            分隔标记，在从多个序列构建序列时使用，例如用于序列分类或用于文本和问题的问答。还用作使用特殊标记构建的序列的最后一个标记。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            分类器标记，用于序列分类（对整个序列进行分类而不是对每个标记进行分类）。在使用特殊标记构建时，它是序列的第一个标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。不在词汇表中的标记无法转换为 ID，而是设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如在对不同长度的序列进行批处理时。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            用于屏蔽值的标记。这是在使用掩码语言模型进行训练时使用的标记。这是模型将尝试预测的标记。
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            是否在输入前添加初始空格。这允许将前导词视为任何其他词。（BART 分词器通过前面的空格检测单词的开头）。
    """

    # 词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练的词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入的最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化方法，接受词汇文件、合并文件等参数，并设置默认值
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
        # 如果 bos_token 是字符串，则创建一个 AddedToken 对象，否则直接使用传入的对象
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 同上，处理 eos_token、sep_token、cls_token、unk_token、pad_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 处理 mask_token，使其包含前面的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 从 vocab_file 中加载词汇表到 encoder 中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建 decoder，将 encoder 的键值对颠倒
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置错误处理方式
        self.errors = errors  # how to handle errors in decoding
        # 创建 byte_encoder 和 byte_decoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 从 merges_file 中加载合并信息到 bpe_merges 中
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建 bpe_ranks 字典，记录合并信息的顺序
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges)))
        # 初始化缓存
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # 创建正则表达式模式，用于处理文本
        # 添加 re.IGNORECASE 以便对缩写的大写形式进行 BPE 合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化方法，传入参数
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

    # 返回词汇表大小
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 返回词汇表和额外 token 的字典
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
    # 实现 BPE（Byte Pair Encoding）算法，将给定的 token 进行编码
    def bpe(self, token):
        # 如果 token 已经在缓存中，则直接返回其对应的编码结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转换为元组形式
        word = tuple(token)
        # 获取 token 的所有可能的字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则返回原始 token
        if not pairs:
            return token

        # 循环处理字符对，直到无法再进行替换
        while True:
            # 根据字符对的编码频率选择最小的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该字符对不存在于预定义的编码中，则停止处理
            if bigram not in self.bpe_ranks:
                break
            # 获取字符对的两个字符
            first, second = bigram
            new_word = []
            i = 0
            # 遍历 token 的每个字符
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到字符对的第一个字符，则直接将剩余字符加入新 token
                    new_word.extend(word[i:])
                    break
                else:
                    # 将字符对之间的字符加入新 token
                    new_word.extend(word[i:j])
                    i = j

                # 判断字符是否为字符对的第一个字符，并且后面还有字符，且紧接着的字符是字符对的第二个字符
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    # 如果是，则将字符对替换为一个新的字符
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则，保留原字符，并移动到下一个字符
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果新 token 的长度为1，则停止处理
            if len(word) == 1:
                break
            else:
                # 否则，继续获取新的字符对
                pairs = get_pairs(word)
        # 将处理后的 token 转换为字符串形式
        word = " ".join(word)
        # 将处理后的结果添加到缓存中
        self.cache[token] = word
        # 返回处理后的结果
        return word

    # 将输入文本进行分词处理，并返回分词结果
    def _tokenize(self, text):
        """Tokenize a string."""
        # 初始化 BPE 分词结果列表
        bpe_tokens = []
        # 使用正则表达式找到文本中符合条件的 token，并进行处理
        for token in re.findall(self.pat, text):
            # 将每个字符转换为对应的字节编码，并使用 BPE 编码处理
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 将处理后的结果进行拆分，并添加到分词结果列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        # 返回 BPE 分词结果列表
        return bpe_tokens

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将 token 转换为对应的 id，如果不存在则使用未知标记的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 id 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将 id 转换为对应的 token
        return self.decoder.get(index)

    # 将 token 序列转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将 token 序列连接成一个字符串
        text = "".join(tokens)
        # 将字符串中的每个字符转换为对应的字节编码，并再次转换为字符串形式
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        # 返回转换后的字符串
        return text
    # 保存词汇表到指定目录，返回保存的文件路径
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

        # 写入词汇表到文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历 BPE 标记和索引，写入合并文件
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

    # 构建带有特殊标记的模型输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BART sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果只��一个序列，添加特殊标记并返回
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 如果有两个序列，添加特殊标记并返回
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # 获取特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从不含特殊标记的标记列表中检索序列ID。当使用标记器的 `prepare_for_model` 方法添加特殊标记时，调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经用模型的特殊标记格式化。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：特殊标记为 1，序列标记为 0。
        """
        # 如果已经包含特殊标记，则调用父类方法获取特殊标记掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果没有第二个序列，返回带有 CLS 和 SEP 标记的单个序列掩码
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 否则，返回带有 CLS、SEP 和两个序列的掩码
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建用于序列对分类任务的掩码。BART 不使用 token 类型 ID，因此返回零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 零列表。
        """
        # CLS 和 SEP 标记的 ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果没有第二个序列，返回零列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回带有 CLS、SEP 和两个序列的零列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果已经分割成单词或需要在文本前加空格，并且文本长度大于0且第一个字符不是空白字符，则在文本前加空格
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)
```