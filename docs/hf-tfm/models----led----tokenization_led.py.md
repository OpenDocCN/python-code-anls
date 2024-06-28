# `.\models\led\tokenization_led.py`

```py
# coding=utf-8
# 版权所有 2021 Iz Beltagy，Matthew E. Peters，Arman Cohan 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证版本 2.0 进行许可；
# 除非符合许可证的条款，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 没有任何形式的明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
"""LED 的分词类。"""

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding, EncodedInput
from ...utils import PaddingStrategy, logging

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/vocab.json",
    },
    "merges_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/tokenizer.json",
    },
}

# 预训练模型的位置编码嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allenai/led-base-16384": 16384,
}


@lru_cache()
# 从 transformers.models.bart.tokenization_bart.bytes_to_unicode 复制的函数
def bytes_to_unicode():
    """
    返回 utf-8 字节列表和 Unicode 字符串的映射。避免映射到空白字符或控制字符，以免引起 bpe 代码错误。
    
    可逆的 bpe 代码适用于 Unicode 字符串。这意味着如果要避免 UNK（未知）符号，词汇表中需要大量的 Unicode 字符。
    当数据集达到约 100 亿个标记时，您需要大约 5000 个 Unicode 字符以获得良好的覆盖率。
    这在普通的 32K bpe 词汇表中占有相当大的比例。为了避免这种情况，我们需要 utf-8 字节和 Unicode 字符串之间的查找表。
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


# 从 transformers.models.bart.tokenization_bart.get_pairs 复制的函数
def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词表示为符号元组（符号是长度可变的字符串）。
    """
    # 创建一个空的集合用于存储字符对
    pairs = set()
    # 取单词的第一个字符作为前一个字符
    prev_char = word[0]
    # 遍历单词中除第一个字符外的每个字符
    for char in word[1:]:
        # 将前一个字符和当前字符作为一个元组，添加到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符为当前字符，为下一次循环做准备
        prev_char = char
    # 返回存储了单词中相邻字符对的集合
    return pairs
class LEDTokenizer(PreTrainedTokenizer):
    """
    Constructs a LED tokenizer, which is similar to the ROBERTa tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import LEDTokenizer

    >>> tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # LEDTokenizer 类，继承自 PreTrainedTokenizer
    # 用于构建 LED 分词器，其使用字节级字节对编码（Byte-Pair-Encoding）
    
    def __init__(self, vocab_file, merges_file, **kwargs):
        """
        Initializes the LEDTokenizer with the provided vocabulary and merges files.

        Args:
            vocab_file (str): Path to the vocabulary file.
            merges_file (str): Path to the merges file.
            kwargs: Additional arguments passed to the tokenizer initialization.
        """
        # 调用父类的初始化方法，传入词汇表文件和合并文件路径，以及其他可选参数
        super().__init__(**kwargs)
        # 使用给定的词汇表文件和合并文件初始化 LEDTokenizer
        
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        # 设置词汇表文件和合并文件的属性
        
        self.encoder = json.load(open(vocab_file))
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 从词汇表文件中加载编码器和解码器，用于将词汇映射到整数和反向映射

        with open(merges_file, encoding="utf-8") as f:
            bpe_data = f.read().split("\n")[1:-1]
        # 打开合并文件，读取 BPE 数据
        
        merges = [(tuple(merge.split()[0:2]), int(merge.split()[2])) for merge in bpe_data]
        # 解析 BPE 数据并转换为元组的列表
        
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # 创建 BPE 合并的排名字典，以便快速查找合并的顺序

    def _tokenize(self, text):
        """
        Tokenizes a given text into subwords.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of subwords representing the tokenized text.
        """
        # 将给定的文本分词成子词（subwords）
        merges = self._split_to_subwords(text)
        # 调用 _split_to_subwords 方法，将文本拆分为子词
        
        return merges
        # 返回分词后的子词列表

    def _split_to_subwords(self, text):
        """
        Splits the text into subwords based on the BPE merges.

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: A list of subwords.
        """
        # 根据 BPE 合并将文本拆分为子词
        return []
        # 返回空的子词列表，实际应该是根据 BPE 算法进行子词拆分并返回结果
    # 词汇文件的名称映射，用于指定预训练模型的词汇文件
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射，指定每个预训练模型对应的词汇文件路径
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入大小的映射，指定每个预训练模型的最大输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入的名称列表，指定输入数据的名称，通常为 "input_ids" 和 "attention_mask"
    model_input_names = ["input_ids", "attention_mask"]

    # 以下代码段是从 transformers.models.bart.tokenization_bart.BartTokenizer.__init__ 中复制的
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
        # 如果 `bos_token`, `eos_token`, `sep_token`, `cls_token`, `unk_token`, `pad_token` 是字符串类型，则将它们封装为 `AddedToken` 对象，否则保持原样
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 如果 `mask_token` 是字符串类型，则将它封装为 `AddedToken` 对象，并且在左侧去掉空格，保持右侧不变；否则保持原样
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 使用 UTF-8 编码打开 `vocab_file` 文件，并将其加载为 JSON 格式，存储到 `self.encoder` 中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        
        # 创建 `self.decoder` 字典，将 `self.encoder` 中的键值对反转，以便根据索引获取词汇
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # 指定解码过程中处理错误的策略，存储到 `self.errors` 中
        self.errors = errors  # how to handle errors in decoding
        
        # 转换字节到 Unicode 编码的映射表，通过调用 `bytes_to_unicode()` 函数实现
        self.byte_encoder = bytes_to_unicode()
        
        # 创建 `self.byte_decoder` 字典，将 `self.byte_encoder` 中的键值对反转，以便根据编码获取原始字节
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 使用 UTF-8 编码打开 `merges_file` 文件，读取 BPE 合并操作，存储到 `bpe_merges` 中
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        
        # 将每行 BPE 合并操作转换为元组形式，存储到 `bpe_merges` 列表中
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        
        # 创建 `self.bpe_ranks` 字典，将 BPE 合并操作与其在列表中的索引关联起来
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        
        # 初始化缓存 `self.cache`，用于存储临时数据
        self.cache = {}
        
        # 设置是否在词前添加空格的标志，存储到 `self.add_prefix_space` 中
        self.add_prefix_space = add_prefix_space

        # 使用正则表达式创建 `self.pat` 模式，用于分词时处理合并和大小写
        # 添加 `re.IGNORECASE` 标志以支持对大写版本的缩写进行 BPE 合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的初始化方法，传递参数并完成初始化
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
    # 返回 `self.encoder` 中的条目数量，即词汇表的大小
    # 从 `transformers.models.bart.tokenization_bart.BartTokenizer.vocab_size` 处复制
    def vocab_size(self):
        return len(self.encoder)

    # 返回包含 `self.encoder` 和 `self.added_tokens_encoder` 的词汇表字典
    # 从 `transformers.models.bart.tokenization_bart.BartTokenizer.get_vocab` 处复制
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 复制自 `transformers.models.bart.tokenization_bart.BartTokenizer.bpe`
    # 未提供具体的实现
    def bpe
    def bpe(self, token):
        # 如果缓存中已经存在该 token 的处理结果，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        
        # 将 token 转换为字符元组
        word = tuple(token)
        # 获取 token 的所有可能的字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则直接返回原始的 token
        if not pairs:
            return token
        
        # 进入循环，直到 token 不再有字符对为止
        while True:
            # 找到当前字符对中排名最低的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果找到的字符对不在字符对排名中，退出循环
            if bigram not in self.bpe_ranks:
                break
            # 分割字符对
            first, second = bigram
            new_word = []
            i = 0
            # 遍历 token 的字符
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到字符 first，则将剩余部分添加到新的 token 中
                    new_word.extend(word[i:])
                    break
                else:
                    # 将 first 之前的字符添加到新的 token 中
                    new_word.extend(word[i:j])
                    i = j

                # 检查是否匹配字符对 first 和 second
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    # 如果匹配，则将字符对添加到新的 token 中
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则将当前字符添加到新的 token 中
                    new_word.append(word[i])
                    i += 1
            # 更新 word 为新的字符元组
            new_word = tuple(new_word)
            word = new_word
            # 如果 token 的长度为 1，则退出循环
            if len(word) == 1:
                break
            else:
                # 获取更新后的字符对
                pairs = get_pairs(word)
        
        # 将字符元组转换为字符串
        word = " ".join(word)
        # 将处理结果添加到缓存中
        self.cache[token] = word
        # 返回处理后的字符串
        return word

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer._tokenize
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        # 使用正则表达式找到文本中的所有 token
        for token in re.findall(self.pat, text):
            # 将每个 token 编码为字节，并映射为 unicode 字符串，避免 BPE 中的控制符号（在我们的情况下是空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 使用 BPE 算法处理 token，并将处理结果拆分为多个子 token
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        # 返回处理后的所有子 token 列表
        return bpe_tokens

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将 token 转换为对应的 id，如果 token 不存在于词汇表中，则返回未知 token 的 id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将 id 转换为对应的 token
        return self.decoder.get(index)

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将所有 token 合并为一个字符串
        text = "".join(tokens)
        # 将合并后的字符串解码为 utf-8 格式的文本
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        # 返回解码后的文本
        return text

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer.save_vocabulary
    # 将词汇表保存到指定目录下的文件中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，若不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 构建词汇表文件名和合并文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器的内容以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将 BPE（Byte Pair Encoding）标记和其索引写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 若 BPE 合并索引不连续，记录警告信息
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                # 写入 BPE 标记
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的词汇表文件名和合并文件名
        return vocab_file, merge_file

    # 从 BARTTokenizer.build_inputs_with_special_tokens 复制并修改为 LEDTokenizer 的特殊标记构建方法
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过添加特殊标记，为序列分类任务构建模型输入。LED 序列有以下格式：

        - 单个序列：`<s> X </s>`
        - 序列对：`<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表（用于序列对任务）。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # 从 BARTTokenizer.get_special_tokens_mask 复制并修改为 LEDTokenizer 的特殊标记掩码方法
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
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
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer.create_token_type_ids_from_sequences with BART->LED
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. LED does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # Initialize separator and classification token IDs for sequence masking
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            # Return a list of zeros of length equal to the combined length of cls, token_ids_0, and sep
            return len(cls + token_ids_0 + sep) * [0]
        # Return a list of zeros of length equal to the combined length of cls, token_ids_0, sep, sep, token_ids_1, and sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    # Copied from transformers.models.bart.tokenization_bart.BartTokenizer.prepare_for_tokenization
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        """
        Prepare text for tokenization by adding a space prefix if specified and not already present.

        Args:
            text (str): The input text to be tokenized.
            is_split_into_words (bool, optional): Whether the text is already split into words.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the modified text and remaining keyword arguments.
        """
        # Check if a space prefix should be added and modify text accordingly
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)

    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        # 调用父类的 `_pad` 方法，对编码输入进行填充
        encoded_inputs = super()._pad(
            encoded_inputs=encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 如果 `return_attention_mask` 为 None，则根据模型默认值确定是否返回注意力掩码
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        # 如果需要返回注意力掩码，并且编码输入中存在 `global_attention_mask`
        if return_attention_mask and "global_attention_mask" in encoded_inputs:
            required_input = encoded_inputs[self.model_input_names[0]]
            # `global_attention_mask` 需要与其他（顺序）输入具有相同的长度
            needs_to_be_padded = len(encoded_inputs["global_attention_mask"]) != len(required_input)

            # 如果需要填充
            if needs_to_be_padded:
                difference = len(required_input) - len(encoded_inputs["global_attention_mask"])

                # 根据填充方向进行处理
                if self.padding_side == "right":
                    # 使用 `-1`，因为 `global_attention_mask` 中的 `0` 表示局部注意力而不是不需要注意
                    encoded_inputs["global_attention_mask"] = (
                        encoded_inputs["global_attention_mask"] + [-1] * difference
                    )
                elif self.padding_side == "left":
                    encoded_inputs["global_attention_mask"] = [-1] * difference + encoded_inputs[
                        "global_attention_mask"
                    ]
                else:
                    # 抛出异常，无效的填充策略
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        # 返回填充后的编码输入字典
        return encoded_inputs
```