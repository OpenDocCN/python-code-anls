# `.\models\clvp\tokenization_clvp.py`

```py
# coding=utf-8
# 版权声明
"""
- 代码声明使用 Apache 2.0 开源许可证
- utf-8 编码声明

从 typing 模块中引入 List、Optional 和 Tuple 类型
引入 regex 库，重命名为 re
从 ...tokenization_utils 模块中引入 AddedToken 和 PreTrainedTokenizer 类
从 ...utils 模块中引入 logging 函数
引入 number_normalizer 模块中的 EnglishNormalizer 类
"""
import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from .number_normalizer import EnglishNormalizer

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义文件名的字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 定义预训练模型的文件名映射的字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "clvp_dev": "https://huggingface.co/susnato/clvp_dev/blob/main/vocab.json",
    },
    "merges_file": {
        "clvp_dev": "https://huggingface.co/susnato/clvp_dev/blob/main/merges.txt",
    },
}

# 定义预训练模型的位置嵌入大小的字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "clvp_dev": 1024,
}

# 为 bytes_to_unicode 函数添加缓存装饰器
@lru_cache()
# 从 transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode 函数复制
def bytes_to_unicode():
    """
    返回 UTF-8 字节列表和字节到 Unicode 字符的映射。我们特别避免将空白/控制字符映射到 BPE 代码。

    可逆的 BPE 代码是作用于 Unicode 字符串的。这意味着如果你不想要 UNK 标记，你的词库中需要大量的 Unicode 字符。例如当你的数据集有大约 100 亿个标记时，你需要至少约 5000 个字符来覆盖较好。这相当于正常的 32k BPE 词库的很大一部分。为了避免这个问题，我们需要编写 utf-8 字节和 Unicode 字符之间的查找表。
    """
    # 获取 ASCII 码表的字节列表
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            # 将非 ASCII 字节添加到列表中
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # 将 ASCII 字节转换为 Unicode 字符
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# 从 transformers.models.gpt2.tokenization_gpt2.get_pairs 函数复制
def get_pairs(word):
    """
    返回单词中的符号对的集合。

    以符号元组的方式表示单词（符号是可变长度的字符串）。
    """
    # 初始化符号对集合
    pairs = set()
    # 遍历单词中的每个字符
    prev_char = word[0]
    for char in word[1:]:
        # 将前一个字符和当前字符添加到符号对集合中
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class ClvpTokenizer(PreTrainedTokenizer):
    """
    构造一个 CLVP 分词器。基于字节级别的 Byte-Pair-Encoding。
    """

    def __init__(
        self,
        vocab_file: str,
        merges_file: Optional[str] = None,
        normalization_rules: Optional[str] = None,
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
        sep_token: str = "<sep>",
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
        mask_token: str = "<mask>",
        **kwargs
    ):
        """
        Args:
            vocab_file (:obj:`str`):
                The vocabulary file path (ends with `.json` or `.txt`) or a URL pointing to a vocabulary file to copy from
                (ends with `.json`).
            merges_file (:obj:`str`, `optional`):
                The merges file path (ends with `.txt`) or a URL pointing to a merges file to copy from (ends with
                `.txt`).
            normalization_rules (:obj:`str`, `optional`):
                Path to a JSON file containing custom normalization rules.
            unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
                The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be
                this token instead.
            eos_token (:obj:`str`, `optional`, defaults to :obj:`"<eos>"`):
                The end of sequence token.
            sep_token (:obj:`str`, `optional`, defaults to :obj:`"<sep>"`):
                The separator token, which is used for separating context and response in an input sequence.
            pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
                The token used for padding, for example when batching sequences of different lengths.
            cls_token (:obj:`str`, `optional`, defaults to :obj:`"<cls>"`):
                The classification token which is used when adding sentence boundaries as input representation.
            mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
                The token used for masking values. This is the token used when training this model with masked language
                modeling.
        """
        # 调用父类构造方法
        super().__init__(
            unk_token=unk_token, eos_token=eos_token, pad_token=pad_token, **kwargs, mask_token=mask_token
        )
        # 设置用于分词的字节与 Unicode 字符之间映射的字典
        self._bytes_to_unicode = bytes_to_unicode()
        # 读取词汇表和合并文件
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        # 初始化词汇表和合并表
        self.encoder, self.decoder = self._load_vocab(vocab_file, merges_file)
        # 获取特殊标记的 ID
        self.sep_token_id = self.convert_tokens_to_ids(sep_token)
        self.cls_token_id = self.convert_tokens_to_ids(cls_token)
        self._custom_normalizer = None

    @property
    def vocab_size(self) -> int:
        """
        Return the size of the vocabulary, i.e. the number of different tokens.
        """
        return len(self.encoder)
    # 这个分词器经过训练，将空格视为词汇的一部分（有点像句子片段），因此一个词在句子开头（无空格）和非开头位置编码会不同：
    
    >>> from transformers import ClvpTokenizer
    
    >>> tokenizer = ClvpTokenizer.from_pretrained("susnato/clvp_dev")
    >>> tokenizer("Hello world")["input_ids"]
    [62, 84, 28, 2, 179, 79]
    
    >>> tokenizer(" Hello world")["input_ids"]
    [2, 62, 84, 28, 2, 179, 79]
    
    
    # 创建这个分词器实例时或调用时，通过传递 `add_prefix_space=True` 可以避免这种行为，但由于模型不是以这种方式进行预训练的，这可能会导致性能下降。
    
    <Tip>
    
    当与 `is_split_into_words=True` 一起使用时，该分词器会在每个词之前添加一个空格（即使是第一个词）。
    
    </Tip>
    
    # 这个分词器继承自 [`PreTrainedTokenizer`]，其中包含大部分主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    
    Args:
        vocab_file (`str`):
            词汇文件的路径。
        merges_file (`str`):
            合并文件的路径。
        errors (`str`, *optional*, 默认为 `"replace"`):
            将字节解码为 UTF-8 时要遵循的范例。有关更多信息，请参阅
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode)。
        unk_token (`str`, *optional*, 默认为 `"[UNK]"`):
            未知标记。词汇表中不存在的标记不能转换为 ID，并被设置为该标记。
        bos_token (`str`, *optional*, 默认为 `"<|endoftext|>"`):
            序列的开始标记。
        eos_token (`str`, *optional*, 默认为 `"[STOP]"`):
            序列的结束标记。
        pad_token (`str`, *optional*, 默认为 `"[STOP]"`):
            序列的填充标记。
        add_prefix_space (`bool`, *optional*, 默认为 `False`):
            是否在输入之前添加初始空格。这允许将前导词视为任何其他词。（CLVP 分词器通过前导空格检测单词的开头）。
        add_bos_token (`bool`, *optional*, 默认为 `False`):
            当 `add_special_tokens=True` 时，在序列前添加 `bos_token`。
        add_eos_token (`bool`, *optional*, 默认为 `False`):
            当 `add_special_tokens=True` 时，在序列末尾添加 `eos_token`。
    """
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = [
        "input_ids",
        "attention_mask",
    ]
    # 初始化方法，接受多个参数并设置相关属性
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="[UNK]",
        bos_token="<|endoftext|>",
        eos_token="[STOP]",
        pad_token="[STOP]",
        add_prefix_space=False,
        add_bos_token=False,
        add_eos_token=False,
        **kwargs,
    ):
        # 如果 `bos_token` 是字符串类型，转换为特殊的 AddedToken 对象
        bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
        # 如果 `eos_token` 是字符串类型，转换为特殊的 AddedToken 对象
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        # 如果 `unk_token` 是字符串类型，转换为特殊的 AddedToken 对象
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        # 如果 `pad_token` 是字符串类型，转换为特殊的 AddedToken 对象
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self._normalizer = None

        # 使用 UTF-8 编码打开 `vocab_file` 文件，读取内容并加载为字典
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 根据 `encoder` 创建 `decoder` 字典，键值互换
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 设置处理解码时的错误方式
        # 根据预定义函数生成字节到 Unicode 的映射
        self.byte_encoder = bytes_to_unicode()
        # 根据字节编码器创建字节解码器，键值互换
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 使用 UTF-8 编码打开 `merges_file` 文件，读取内容并按行划分
        with open(merges_file, encoding="utf-8") as merges_handle:
            # 跳过第一行，将剩余行按行分割并存储为列表
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将每个 BPE 合并操作字符串转换为元组，形成 BPE 合并操作列表
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建 BPE 合并操作及其对应排名的字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges)))
        # 初始化缓存字典
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # 使用正则表达式创建用于处理 BPE 合并的模式
        # 应该添加 `re.IGNORECASE`，以便对缩写的大写版本进行 BPE 合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类初始化方法，传递部分参数
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )

    # 获取词汇表大小的属性方法
    @property
    def vocab_size(self):
        return len(self.encoder)

    # 获取规范化实例的属性方法
    @property
    def normalizer(self):
        # 如果规范化实例未创建，则创建英文规范化实例并返回
        if self._normalizer is None:
            self._normalizer = EnglishNormalizer()
        return self._normalizer

    # 返回词汇表的方法
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe 复制的代码
    # 对输入的 token 进行 BPE（Byte Pair Encoding）编码处理
    def bpe(self, token):
        # 如果 token 已经存在于缓存中，直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将 token 转化为元组
        word = tuple(token)
        # 获取 token 的全部可能的 pairs
        pairs = get_pairs(word)

        # 如果没有 pairs，说明 token 无法再进行分割，直接返回 token
        if not pairs:
            return token

        # 循环处理 pairs，直到找不到更小的 bigram
        while True:
            # 找到当前 pairs 中按照 bpe_ranks 排序后的最小 bigram
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果当前 bigram 不在 bpe_ranks 中，结束循环
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

                # 如果 word 的下一个字符是 second，则将 first 和 second 合并成一个新的单词
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果 word 的长度为 1，则结束循环
            if len(word) == 1:
                break
            else:
                # 继续获取新的 pairs
                pairs = get_pairs(word)
        # 将处理后的 token 转化为字符串形式，并存入缓存
        word = " ".join(word)
        self.cache[token] = word
        return word

    # 从 LlamaTokenizer 的 build_inputs_with_special_tokens 方法复制而来
    # 构建包含特殊 token 的输入
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []  # 如果需要添加 bos_token，将 bos_token_id 加入到列表中
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []  # 如果需要添加 eos_token，将 eos_token_id 加入到列表中

        output = bos_token_id + token_ids_0 + eos_token_id  # 构建包含特殊 token 的输入列表

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id  # 如果有第二个 token_ids，构建包含特殊 token 的输入列表

        return output  # 返回最终的输入列表

    # 从 GPT2Tokenizer 的 get_special_tokens_mask 方法复制而来
    # 获取特殊 token 的 mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    # 定义方法，获取没有添加特殊令牌的令牌列表中的序列 ID，当使用分词器的`prepare_for_model`或`encode_plus`方法添加特殊令牌时调用此方法
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, 
                                already_has_special_tokens: bool = False) -> List[int]:
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
        
        # 如果已经有特殊令牌，直接调用父类的相应方法返回结果
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                )
        
        # 如果没有开始的特殊令牌，并且不需要添加“开头”的特殊令牌，直接调用父类的相应方法返回结果
        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
                )
        
        # 如果没有开始的特殊令牌，并且需要添加“开头”的特殊令牌，返回特殊令牌的掩码列表
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    # 将文本进行分词
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        text = self.normalizer(text)  # 规范化文本
        for token in re.findall(self.pat, text):  # 通过正则表达式将文本分割成 token
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # 将 token 的 utf-8 编码进行 BPE 编码

            # 如果 token 是 "Ġ"，则替换为 "[SPACE]"（如果词汇表中存在 "[SPACE]"），否则保持 "Ġ"
            bpe_tokens.extend(
                "[SPACE]" if bpe_token == "\u0120" and "[SPACE]" in self.encoder.keys() else bpe_token
                for bpe_token in self.bpe(token).split(" ")
            )

        return bpe_tokens

    # 将 token 转换为对应的 ID
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 将 ID 转换为对应的 token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # 将一系列 token 转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)  # 将一系列 token 连接为一个字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)  # 将字节解码为字符串
        return text
    # 清理 token 化后的文本，将所有字符连接在一起
    def clean_up_tokenization(self, text):
        text = "".join(text)
        # 获取编码器和额外添加的 token 编码器中的所有 token
        vocab_tokens = list(self.encoder.keys()) + list(self.added_tokens_encoder.keys())

        # 如果文本中包含 "[SPACE]"，则用空格替换
        text = text.replace("[SPACE]", " ") if "[SPACE]" in vocab_tokens else text
        # 如果文本中包含 "[STOP]"，则用空格替换
        text = text.replace("[STOP]", " ") if "[STOP]" in vocab_tokens else text

        # 替换文本中的特殊标记
        text = text.replace(self.unk_token, "").replace("   ", " ").replace("  ", " ")
        # 返回清理后的文本
        return text

    # 保存词汇表和合并规则
    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.save_vocabulary 复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，不存在则报错
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 获取词汇表文件和合并规则文件的完整路径
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器中的内容以 JSON 格式写入词汇表文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将合并规则写入合并规则文件
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

        # 返回词汇表文件和合并规则文件的路径
        return vocab_file, merge_file
```