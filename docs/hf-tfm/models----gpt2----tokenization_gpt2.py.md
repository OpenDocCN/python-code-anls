# `.\models\gpt2\tokenization_gpt2.py`

```py
# 设置脚本的编码格式为UTF-8

# 引入必要的模块和函数
import json  # 导入用于 JSON 操作的模块
import os  # 导入用于操作系统功能的模块
from functools import lru_cache  # 导入 lru_cache 装饰器，用于缓存函数调用结果
from typing import List, Optional, Tuple  # 导入类型提示相关的类和函数

import regex as re  # 导入 regex 模块，用于正则表达式操作

# 导入 tokenization_utils 模块中的 AddedToken 和 PreTrainedTokenizer 类
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 导入 utils 模块中的 logging 对象
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称常量
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表 JSON 文件的名称
    "merges_file": "merges.txt",  # 合并文件的名称
}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai-community/gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json",
        "openai-community/gpt2-medium": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/vocab.json",
        "openai-community/gpt2-large": "https://huggingface.co/openai-community/gpt2-large/resolve/main/vocab.json",
        "openai-community/gpt2-xl": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/vocab.json",
        "distilbert/distilgpt2": "https://huggingface.co/distilbert/distilgpt2/resolve/main/vocab.json",
    },
    "merges_file": {
        "openai-community/gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt",
        "openai-community/gpt2-medium": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/merges.txt",
        "openai-community/gpt2-large": "https://huggingface.co/openai-community/gpt2-large/resolve/main/merges.txt",
        "openai-community/gpt2-xl": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/merges.txt",
        "distilbert/distilgpt2": "https://huggingface.co/distilbert/distilgpt2/resolve/main/merges.txt",
    },
}

# 预训练位置嵌入的尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai-community/gpt2": 1024,
    "openai-community/gpt2-medium": 1024,
    "openai-community/gpt2-large": 1024,
    "openai-community/gpt2-xl": 1024,
    "distilbert/distilgpt2": 1024,
}

# 使用 lru_cache 装饰器缓存结果的函数，将字节转换为 Unicode 字符
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    """
    # 此函数返回一个 utf-8 字节列表和映射到 Unicode 字符串的字典，避免映射到空白或控制字符，以免 BPE 算法出错
    # 生成一个字典，将 UTF-8 字节与 Unicode 字符之间建立映射关系
    bs = (
        # ASCII 可见字符的 Unicode 码点范围
        list(range(ord("!"), ord("~") + 1)) +
        # 特殊符号的 Unicode 码点范围
        list(range(ord("¡"), ord("¬") + 1)) +
        # 特殊符号的 Unicode 码点范围
        list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]  # 复制 bs 列表到 cs 列表
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)  # 将不在 bs 中的字节添加到 bs 列表中
            cs.append(2**8 + n)  # 同时将其映射到 cs 列表中，通过添加 256 + n 的方式
            n += 1
    cs = [chr(n) for n in cs]  # 将 cs 列表中的整数转换为对应的 Unicode 字符
    return dict(zip(bs, cs))  # 返回由 bs 和 cs 列表组成的字典，表示 UTF-8 字节到 Unicode 字符的映射关系
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # Initialize an empty set to store symbol pairs
    pairs = set()
    # Initialize prev_char with the first symbol of the word
    prev_char = word[0]
    # Iterate through each character in the word starting from the second character
    for char in word[1:]:
        # Add a tuple representing the pair (prev_char, char) to the pairs set
        pairs.add((prev_char, char))
        # Update prev_char to the current character for the next iteration
        prev_char = char
    # Return the set of symbol pairs
    return pairs


class GPT2Tokenizer(PreTrainedTokenizer):
    """
    Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```
    >>> from transformers import GPT2Tokenizer

    >>> tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial beginning of sentence token to the input. This allows to treat the leading
            word just as any other word.
    """
    # The GPT2Tokenizer class provides methods to tokenize text based on GPT-2 model's byte-level BPE approach.
    # It handles various tokenization scenarios including handling spaces and special tokens like BOS and EOS.
    
    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: Optional[str] = None,
        add_prefix_space: bool = False,
        add_bos_token: bool = False,
    ):
        # Initialize the tokenizer by loading vocabulary and merges information
        pass

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Save the tokenizer's vocabulary and merges information to the specified directory
        pass

    def encode(
        self,
        text: Union[str, List[str], List[int]],
        text_pair: Optional[Union[str, List[str], List[int]]] = None,
        add_special_tokens: bool = True,
        is_split_into_words: bool = False,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        # Tokenize a single sequence or a pair of sequences and return the token IDs
        pass

    def decode(self, token_ids: Union[int, List[int]], skip_special_tokens: bool = False) -> str:
        # Convert token IDs back to text
        pass

    def decode_batch(self, token_ids_batch: Union[List[int], List[List[int]]], **kwargs) -> List[str]:
        # Convert batches of token IDs back to text
        pass

    def convert_tokens_to_string(self, tokens: Union[int, List[int]]) -> str:
        # Convert token IDs or list of token IDs to a single string
        pass

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        # Convert token IDs or list of token IDs to token strings
        pass

    def prepare_for_model(
        self,
        ids: Union[int, List[int]],
        pair_ids: Optional[Union[int, List[int]]] = None,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
        stride: int = 0,
        truncation_strategy: Union[TruncationStrategy, str] = "longest_first",
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        # Preprocess IDs and pair IDs for input to the model
        pass
    # 初始化变量 vocab_files_names，使用预定义的常量 VOCAB_FILES_NAMES
    vocab_files_names = VOCAB_FILES_NAMES
    # 初始化变量 pretrained_vocab_files_map，使用预定义的常量 PRETRAINED_VOCAB_FILES_MAP
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 初始化变量 max_model_input_sizes，使用预定义的常量 PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化变量 model_input_names，包含固定的字符串列表
    model_input_names = ["input_ids", "attention_mask"]

    # 定义类的初始化方法
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
        # 如果 bos_token 是字符串，则将其转换为 AddedToken 对象
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串，则将其转换为 AddedToken 对象
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果 unk_token 是字符串，则将其转换为 AddedToken 对象
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则将其转换为 AddedToken 对象
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 设置类属性 add_bos_token
        self.add_bos_token = add_bos_token

        # 使用 UTF-8 编码打开 vocab_file 文件，并加载其中的 JSON 数据到 self.encoder 中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 根据 self.encoder 创建反向映射 self.decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置错误处理策略
        self.errors = errors  # how to handle errors in decoding
        # 初始化 bytes_to_unicode 函数，并将其赋值给 self.byte_encoder
        self.byte_encoder = bytes_to_unicode()
        # 根据 self.byte_encoder 创建反向映射 self.byte_decoder
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 使用 UTF-8 编码打开 merges_file 文件，并读取其中的 BPE 合并规则
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将 BPE 合并规则列表转换为元组，并创建 BPE 合并规则到索引的映射 self.bpe_ranks
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存字典 self.cache
        self.cache = {}
        # 设置是否在前缀后添加空格的属性 self.add_prefix_space
        self.add_prefix_space = add_prefix_space

        # 编译正则表达式模式 self.pat，用于识别文本中的各种标记和空格
        # 添加 re.IGNORECASE 选项，以便处理大小写版本的缩略词合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.IGNORECASE)

        # 调用父类的初始化方法，传递额外的参数和关键字参数
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

    # 定义属性方法 vocab_size，返回 self.encoder 的长度
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 定义方法 get_vocab，返回包含 self.encoder 和 self.added_tokens_encoder 的字典
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
    # 根据给定的 token 查询缓存，如果已经缓存则直接返回结果
    if token in self.cache:
        return self.cache[token]

    # 将 token 转换为元组形式，每个字符作为一个元素
    word = tuple(token)

    # 获取 token 中所有可能的字符对
    pairs = get_pairs(word)

    # 如果不存在字符对，则直接返回原始 token
    if not pairs:
        return token

    # 循环处理直到无法继续拆分
    while True:
        # 找到权重最小的字符对
        bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))

        # 如果找到的字符对不在权重表中，则跳出循环
        if bigram not in self.bpe_ranks:
            break

        # 将 word 按照找到的字符对进行拆分和合并
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

        # 更新 word 为新的拆分后的元组形式
        new_word = tuple(new_word)
        word = new_word

        # 如果最终只剩一个元素，结束循环
        if len(word) == 1:
            break
        else:
            # 否则继续获取新的字符对
            pairs = get_pairs(word)

    # 将最终处理后的 word 转换为字符串形式
    word = " ".join(word)

    # 将处理后的结果加入缓存中
    self.cache[token] = word

    # 返回最终处理后的字符串形式的 token
    return word

# 根据给定的 token_ids_0 和 token_ids_1 （可选）构建带有特殊 token 的输入
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    # 如果需要添加 bos_token，则设置开始的特殊 token_ids
    if self.add_bos_token:
        bos_token_ids = [self.bos_token_id]
    else:
        bos_token_ids = []

    # 将 token_ids_0 添加到输出列表中
    output = bos_token_ids + token_ids_0

    # 如果 token_ids_1 为空，则直接返回构建好的输出
    if token_ids_1 is None:
        return output

    # 否则将 token_ids_1 也添加到输出列表中，并在两个句子之间添加 bos_token_ids
    return output + bos_token_ids + token_ids_1

# 返回一个特殊 token 的掩码，用于标识输入中哪些位置已经包含了特殊 token
def get_special_tokens_mask(
    self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的令牌列表中提取序列 ID。当使用分词器的 `prepare_for_model` 或 `encode_plus` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标识令牌列表是否已经包含模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，取值为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def _tokenize(self, text):
        """对字符串进行分词。"""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # 将所有字节映射为 Unicode 字符串，避免 BPE 的控制令牌（在本例中为空格）
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """使用词汇表将令牌（str）转换为 ID。"""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为令牌（str）。"""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """将令牌序列（字符串）转换为单个字符串。"""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 将词汇表保存到指定目录
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

        # 写入词汇表到vocab_file中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 写入合并信息到merge_file中
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 遍历并按照token_index排序写入BPE merges信息
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    # 如果BPE合并索引不连续，记录警告
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的文件路径
        return vocab_file, merge_file

    # 为进行分词准备文本
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本已经被分成单词或者需要在文本前添加前缀空格，则在文本前添加空格
        if is_split_into_words or add_prefix_space:
            text = " " + text
        # 返回处理后的文本和参数
        return (text, kwargs)

    # 默认的聊天模板
    @property
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        # 如果没有为这个分词器定义聊天模板，记录警告并返回默认模板
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
```