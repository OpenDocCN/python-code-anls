# `.\transformers\models\mvp\tokenization_mvp.py`

```py
# 导入必要的模块和类
import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 vocab 文件名和 merges 文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}

# 存储预训练的 vocab 文件和 merges 文件的 URL
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
}

# 存储预训练模型的 positional embeddings 大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "RUCAIBox/mvp": 1024,
}

# 定义一个缓存的函数，将 bytes 转换为 unicode 字符
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

# 定义一个函数，用于获取一个单词中的符号对
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

# 定义 MvpTokenizer 类，继承自 PreTrainedTokenizer
class MvpTokenizer(PreTrainedTokenizer):
    """
    Constructs a MVP tokenizer, which is smilar to the RoBERTa tokenizer, using byte-level Byte-Pair-Encoding.
    # 这个标记器经过训练，将空格视为标记的一部分（有点像 sentencepiece），因此一个单词会根据其是否在句子开头（没有空格）而被编码成不同的形式：

    ```python
    # 从 transformers 库中导入 MvpTokenizer
    >>> from transformers import MvpTokenizer

    # 从预训练模型 "RUCAIBox/mvp" 实例化标记器
    >>> tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
    # 对文本 "Hello world" 进行标记化，并获取其 input_ids
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    # 对文本 " Hello world" 进行标记化，并获取其 input_ids
    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```py

    # 当实例化此标记器时或在对文本进行标记化时传递 `add_prefix_space=True` 可以避免这种行为，但由于模型不是按这种方式预训练的，这可能会导致性能下降。

    <Tip>

    # 当使用 `is_split_into_words=True` 时，这个标记器会在每个单词前面（甚至第一个单词）添加一个空格。

    </Tip>

    # 这个标记器继承自 [`PreTrainedTokenizer`]，其中包含大部分主要方法。用户应参考这个超类以获取有关这些方法的更多信息。
    # 定义一个函数，用于加载词汇表和合并文件，并设置了一些默认参数
    def load_vocab(vocab_file, merges_file, errors="replace", bos_token="<s>", eos_token="</s>", sep_token="</s>", cls_token="<s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>", add_prefix_space=False):
        # 定义需要加载的文件名
        vocab_files_names = VOCAB_FILES_NAMES
        # 预训练模型的词汇文件映射
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 预训练模型的最大输入大小
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # 模型的输入名称
        model_input_names = ["input_ids", "attention_mask"]
    # 定义类的初始化方法，初始化词汇文件、合并文件等参数，并设置默认参数
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
        # 如果开始符号是字符串，则将其转换为特殊的 AddedToken 对象
        bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
        # 如果结束符号是字符串，则将其转换为特殊的 AddedToken 对象
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        # 如果分隔符号是字符串，则将其转换为特殊的 AddedToken 对象
        sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
        # 如果类别标识符号是字符串，则将其转换为特殊的 AddedToken 对象
        cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
        # 如果未知标记是字符串，则将其转换为特殊的 AddedToken 对象
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        # 如果填充标记是字符串，则将其转换为特殊的 AddedToken 对象
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token
    
        # 如果掩码标记是字符串，则将其转换为特殊的 AddedToken 对象，并移除左侧空格
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
        # 使用 utf-8 编码打开词汇文件
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            # 加载词汇文件到编码器
            self.encoder = json.load(vocab_handle)
        # 创建解码器，将编码器键值对颠倒
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置处理解码错误的策略
        self.errors = errors
        # 创建字节编码器和解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 使用 utf-8 编码打开合并文件
        with open(merges_file, encoding="utf-8") as merges_handle:
            # 读取合并文件内容并分割，排除空行
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将合并内容转换为元组列表
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建 BPE 合并的排名字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存
        self.cache = {}
        # 设置是否在前缀中添加空格的选项
        self.add_prefix_space = add_prefix_space
    
        # 编译正则表达式模式，用于匹配单词、数字、标点和空白字符
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
        # 调用父类的初始化方法，传递参数和关键字参数
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
    
    # 定义获取词汇表大小的属性方法
    @property
    def vocab_size(self):
        return len(self.encoder)
    
    # 定义获取词汇表的方法
    def get_vocab(self):
        # 复制编码器内容到词汇表
        vocab = self.encoder.copy()
        # 更新词汇表添加的标记编码
        vocab.update(self.added_tokens_encoder)
        return vocab
    def bpe(self, token):
        # 如果token已经在缓存中，则直接返回其对应的结果
        if token in self.cache:
            return self.cache[token]
        # 将token转换为元组形式
        word = tuple(token)
        # 获取token的所有可能的连续字符对
        pairs = get_pairs(word)

        # 如果没有连续字符对，则返回原始token
        if not pairs:
            return token

        # 循环直到无法再继续拆分token
        while True:
            # 获取拆分优先级最高的连续字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该字符对不在BPE词汇表中，则停止拆分
            if bigram not in self.bpe_ranks:
                break
            # 分割token为新的单词序列
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

                # 如果找到了连续字符对，则合并为一个新词，继续拆分
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            # 将新单词序列转换为元组形式，继续下一轮拆分
            new_word = tuple(new_word)
            word = new_word
            # 如果新的单词序列长度为1，则停止拆分
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # 将拆分后的单词序列用空格连接成字符串，并缓存结果
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        # 使用正则表达式将文本分词
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # 将每个token中的每个字节转换为unicode字符串，避免BPE中的控制标记（在本例中是空格）
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            # 使用BPE算法对token进行分词，并将分词结果添加到列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 根据词汇表将token转换为对应的id，若不存在则使用未知token的id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 根据词汇表将id转换为对应的token
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将token序列连接成字符串，并将字节转换为unicode字符串
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 将词汇表保存到指定目录中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则报错
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建保存词汇表和词汇合并文件的完整路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        # 将编码器保存到词汇表文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        # 将BPE合并规则保存到词汇合并文件中
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
        # 返回保存的两个文件路径
        return vocab_file, merge_file
    
    # 构建用于序列分类任务的输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        构建用于序列分类任务的输入，包括以下格式:
        - 单个序列: `<s> X </s>`
        - 成对序列: `<s> A </s></s> B </s>`
    
        Args:
            token_ids_0 (`List[int]`):
                需要添加特殊token的第一个序列的token id列表。
            token_ids_1 (`List[int]`, *optional*):
                需要添加特殊token的第二个序列的token id列表（如果存在）。
    
        Returns:
            `List[int]`: 包含适当特殊token的输入id列表。
        """
        # 如果只有一个序列，在开头添加cls token，在结尾添加sep token
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 如果有两个序列，在第一个序列前添加cls token，在两个序列之间和末尾添加sep token
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep
    
    # 获取输入序列的特殊token掩码
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
        # Check if special tokens are already present in the token list
        if already_has_special_tokens:
            # If special tokens are present, return the mask with special tokens information
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If special tokens are not present
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
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
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        # Extract the add_prefix_space argument or use the default value from self
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # Check if prefix space needs to be added to the text
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            # Add prefix space to the text if conditions are met
            text = " " + text
        # Return the text and kwargs
        return (text, kwargs)
```