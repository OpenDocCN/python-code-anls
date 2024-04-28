# `.\transformers\models\qwen2\tokenization_qwen2.py`

```py
# coding=utf-8
# 声明文件编码格式和版权信息

# 导入必要的库
import json
import os
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple

# 导入 regex 库并命名为 re
import regex as re

# 导入 Hugging Face Transformers 库中的一些模块和函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 定义预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/vocab.json"},
    "merges_file": {"qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/merges.txt"},
}

# 定义最大模型输入大小
MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

# 定义预分词正则表达式
PRETOKENIZE_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# 使用 lru_cache 装饰器缓存函数结果，提高性能
@lru_cache()
# 从字节到 Unicode 的映射函数，来自于 GPT-2 模型的实现
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    # 定义 Unicode 编码范围
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # 将不在 bs 中的字节添加到 bs 和 cs 中，并进行编码
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    # 返回字节到 Unicode 的映射字典
    return dict(zip(bs, cs))


# 从 GPT-2 模型实现中复制的函数，用于获取词对
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


# 定义 Qwen2Tokenizer 类，继承自 PreTrainedTokenizer 类
class Qwen2Tokenizer(PreTrainedTokenizer):
    """
    Qwen2Tokenizer 类
    Construct a Qwen2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    Same with GPT2Tokenzier, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import Qwen2Tokenizer

    >>> tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21927, 1879]
    ```py
    This is expected.

    You should not use GPT2Tokenizer instead, because of the different pretokenization rules.

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
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not the model should cleanup the spaces that were added when splitting the input text during the
            tokenization process. Not applicable to this tokenizer, since tokenization does not add spaces.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the tokenization process. The default behavior is
            to not split special tokens. This means that if `<|endoftext|>` is the `eos_token`, then `tokenizer.tokenize("<|endoftext|>") =
            ['<|endoftext|>`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("<|endoftext|>")` will be give `['<',
            '|', 'endo', 'ft', 'ext', '|', '>']`. This argument is only supported for `slow` tokenizers for the moment.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    # Mapping of pretrained vocabulary files to names
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # Maximum model input sizes
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    # Input names for the model
    model_input_names = ["input_ids", "attention_mask"]
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        # 如果给定的开始标记不是字符串，则保持它不变；如果是字符串，则将其包装为特殊的 AddedToken 对象
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        # 如果给定的结束标记不是字符串，则保持它不变；如果是字符串，则将其包装为特殊的 AddedToken 对象
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        # 如果给定的未知标记不是字符串，则保持它不变；如果是字符串，则将其包装为特殊的 AddedToken 对象
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        # 如果给定的填充标记不是字符串，则保持它不变；如果是字符串，则将其包装为特殊的 AddedToken 对象
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        # 使用 utf-8 编码打开词汇文件，加载编码器（从单词到索引的映射）
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建解码器（从索引到单词的映射）
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设定在解码过程中处理错误的方式
        self.errors = errors  # how to handle errors in decoding
        # 创建字节到 Unicode 的映射
        self.byte_encoder = bytes_to_unicode()
        # 创建 Unicode 到字节的映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 初始化 BPE 合并列表为空
        bpe_merges = []
        # 使用 utf-8 编码打开合并文件，加载 BPE 合并列表
        with open(merges_file, encoding="utf-8") as merges_handle:
            for line in merges_handle:
                line = line.strip()
                # 忽略空行和以 # 开头的注释行
                if not line or line.startswith("#"):
                    continue
                # 将合并规则按空格分割并加入到 BPE 合并列表中
                bpe_merges.append(tuple(line.split()))
        # 创建 BPE 合并到其索引的映射
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存字典，用于存储编码后的文本和 BPE 单词的分割
        # 注意：缓存可能会无限增长，并且对于运行时间较长的进程（特别是对于不使用空格分隔单词的语言，例如中文的文本），缓存可能会变得非常大；
        # 技术上不算是内存泄漏，但看起来像是一个。
        # GPT2Tokenizer 也存在同样的问题，因此我们保持一致。
        self.cache = {}

        # 使用预定义的正则表达式编译标记化前正则表达式
        self.pat = re.compile(PRETOKENIZE_REGEX)

        # 如果 add_prefix_space 为 True，则发出警告，因为此类不支持该功能，设置为 True 没有任何效果
        if kwargs.get("add_prefix_space", False):
            logger.warning_once(
                f"{self.__class__.__name} does not support `add_prefix_space`, setting it to True has no effect."
            )

        # 调用父类的初始化方法，设置错误处理方式、开始标记、结束标记、填充标记、未知标记、清理标记化空格、分割特殊标记等参数
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )

    # 返回词汇表大小的属性方法
    @property
    def vocab_size(self) -> int:
        return len(self.encoder)
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.get_vocab复制而来，返回包含编码器和添加的特殊token编码器的字典
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
    
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe复制而来，将输入token进行BPE编码
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
    
        if not pairs:
            return token
    
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
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
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word
    
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._tokenize复制而来，将输入的字符串进行分词处理
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens
    
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_token_to_id复制而来，将输入的token转换为对应的id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))
    
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_id_to_token复制而来，将输入的id转换为对应的token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)
    
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string复制而来，将输入的token序列转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 定义一个方法用于将模型输出的token_ids解码成字符串
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = False,
        spaces_between_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        # 如果`spaces_between_special_tokens`没有被指定，则默认为 False
        # 这对于 _decode 在慢速分词器中默认为 True，但对于 Qwen2Tokenizer 应该默认为 False
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=spaces_between_special_tokens,
            **kwargs,
        )

    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.save_vocabulary 中复制的方法
    # 用于保存词汇表到磁盘
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存路径是否为目录
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建词汇表文件的路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构建BPE合并文件的路径
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将词汇表保存到文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        # 写入 BPE 合并操作到文件中
        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的文件路径
        return vocab_file, merge_file

    # 对文本进行标准化，准备进行分词
    def prepare_for_tokenization(self, text, **kwargs):
        # 使用 NFC 标准化文本
        text = unicodedata.normalize("NFC", text)
        # 返回标准化后的文本及其他参数
        return (text, kwargs)
```  
```