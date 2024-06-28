# `.\models\clvp\tokenization_clvp.py`

```
# 设置文件编码为 UTF-8

# 版权声明

# 导入必要的库和模块
import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple

# 导入第三方库 regex 并重命名为 re
import regex as re

# 导入所需的 Hugging Face 库和模块
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 导入 CLVP 的数字归一化模块
from .number_normalizer import EnglishNormalizer

# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 定义预训练模型词汇文件的映射字典
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "clvp_dev": "https://huggingface.co/susnato/clvp_dev/blob/main/vocab.json",
    },
    "merges_file": {
        "clvp_dev": "https://huggingface.co/susnato/clvp_dev/blob/main/merges.txt",
    },
}

# 定义预训练位置嵌入大小的字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "clvp_dev": 1024,
}

# 使用 lru_cache 装饰器缓存 bytes_to_unicode 函数的结果
@lru_cache()
def bytes_to_unicode():
    """
    返回 utf-8 字节列表及其对应的 Unicode 字符映射。避免将空白字符和控制字符映射到 BPE 代码会出错的地方。

    可逆的 BPE 编码适用于 Unicode 字符串。这意味着如果要避免 UNKs，你需要在词汇表中包含大量的 Unicode 字符。
    当你处理像 10B 令牌数据集时，你最终需要大约 5K 个 Unicode 字符才能获得良好的覆盖率。
    这在你正常使用的 32K BPE 词汇表中占据了显著的比例。为了避免这种情况，我们需要 utf-8 字节和 Unicode 字符串之间的查找表。
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


# 复制自 transformers.models.gpt2.tokenization_gpt2 的 bytes_to_unicode 函数

# 复制自 transformers.models.gpt2.tokenization_gpt2 的 get_pairs 函数
def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词被表示为符号元组（符号是长度可变的字符串）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class ClvpTokenizer(PreTrainedTokenizer):
    """
    构建一个 CLVP 分词器。基于字节级字节对编码。
    """
    pass
    # 定义 CLVPTokenizer 类，它是基于 PreTrainedTokenizer 的一个子类，包含了大部分主要方法
    # 用户可以参考其父类 PreTrainedTokenizer 获取更多关于这些方法的信息
    class CLVPTokenizer(PreTrainedTokenizer):
        # 构造函数，初始化一个 CLVPTokenizer 对象
        def __init__(
            self,
            vocab_file: str,
            merges_file: str,
            errors: str = "replace",
            unk_token: str = "[UNK]",
            bos_token: str = "<|endoftext|>",
            eos_token: str = "[STOP]",
            pad_token: str = "[STOP]",
            add_prefix_space: bool = False,
            add_bos_token: bool = False,
            add_eos_token: bool = False,
        ):
            # 调用父类的构造函数，传入相关参数进行初始化
            super().__init__(
                vocab_file=vocab_file,
                merges_file=merges_file,
                errors=errors,
                unk_token=unk_token,
                bos_token=bos_token,
                eos_token=eos_token,
                pad_token=pad_token,
                add_prefix_space=add_prefix_space,
                add_bos_token=add_bos_token,
                add_eos_token=add_eos_token,
            )
    
        # 类变量，指定了预训练模型中的词汇文件名
        vocab_files_names = VOCAB_FILES_NAMES
        # 类变量，指定了预训练模型中的预训练词汇文件映射
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
        # 类变量，指定了预训练模型的最大输入尺寸
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
        # 类变量，定义了模型输入的名称列表
        model_input_names = [
            "input_ids",
            "attention_mask",
        ]
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
        bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token

        self.add_bos_token = add_bos_token  # 是否添加起始标记到词汇处理器
        self.add_eos_token = add_eos_token  # 是否添加结束标记到词汇处理器
        self._normalizer = None

        # 从给定的vocab_file读取JSON格式的词汇表并存储在self.encoder中
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建反向映射，将self.encoder的键值对颠倒，存储在self.decoder中
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # 设置解码时如何处理错误的策略
        self.byte_encoder = bytes_to_unicode()  # 初始化字节到Unicode字符的编码器
        # 创建字节解码器，将字节到Unicode字符的映射关系反转
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 从给定的merges_file中读取BPE合并列表，并建立合并的排名字典存储在self.bpe_ranks中
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))  # 将BPE合并转化为排名字典
        self.cache = {}  # 初始化缓存字典
        self.add_prefix_space = add_prefix_space  # 是否在前缀中添加空格

        # 创建正则表达式模式，用于识别不同类型的标记
        # 注意：应添加 re.IGNORECASE 以便对大小写版本的缩略词进行BPE合并
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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

    @property
    def vocab_size(self):
        return len(self.encoder)  # 返回词汇表大小

    @property
    def normalizer(self):
        if self._normalizer is None:
            self._normalizer = EnglishNormalizer()  # 如果没有正规化器，则创建一个英语正规化器
        return self._normalizer  # 返回正规化器对象

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe
    # 对输入的单词进行 BPE（Byte Pair Encoding）处理
    def bpe(self, token):
        # 如果 token 已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将单词转换为元组
        word = tuple(token)
        # 获取单词中的所有字符对
        pairs = get_pairs(word)

        # 如果没有字符对，则直接返回原始单词
        if not pairs:
            return token

        # 循环处理字符对，直到无法继续合并
        while True:
            # 选择出现频率最低的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果该字符对不在预定义的频率表中，则跳出循环
            if bigram not in self.bpe_ranks:
                break
            # 分割单词为新的列表
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

                # 如果找到了匹配的字符对，则合并并跳过这两个字符
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            # 将新单词转换为元组
            new_word = tuple(new_word)
            word = new_word
            # 如果新单词长度为1，则停止合并
            if len(word) == 1:
                break
            else:
                # 继续获取新的字符对
                pairs = get_pairs(word)
        # 将处理后的单词转换为字符串形式
        word = " ".join(word)
        # 将处理结果缓存起来
        self.cache[token] = word
        # 返回处理后的单词
        return word

    # 生成带有特殊标记的输入序列，用于模型输入
    # 从 transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens 复制而来
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 如果需要添加开始标记，则添加到输出中
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        # 如果需要添加结束标记，则添加到输出中
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # 将输入序列与特殊标记合并
        output = bos_token_id + token_ids_0 + eos_token_id

        # 如果有第二个输入序列，则将其也与特殊标记合并
        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        # 返回最终的带有特殊标记的输入序列
        return output

    # 获取带有特殊标记的特殊 token 掩码
    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.get_special_tokens_mask 复制而来
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
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
        # If the token list already has special tokens, delegate the masking to the superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # If `add_bos_token` is False, use the superclass method to get special tokens mask
        if not self.add_bos_token:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False
            )

        # If token_ids_1 is None, return a mask with one special token followed by sequence tokens
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0))
        
        # Otherwise, return a mask with special tokens followed by sequence tokens of both lists
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        text = self.normalizer(text)
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)

            # Replace special token "Ġ" with "[SPACE]" if "[SPACE]" is in the vocab
            bpe_tokens.extend(
                "[SPACE]" if bpe_token == "\u0120" and "[SPACE]" in self.encoder.keys() else bpe_token
                for bpe_token in self.bpe(token).split(" ")
            )

        return bpe_tokens

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) into an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocab."""
        return self.decoder.get(index)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) into a single string."""
        # Join tokens into a string and decode bytes to UTF-8 characters, handling errors
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 清理文本中的特殊标记和空格
    def clean_up_tokenization(self, text):
        # 将列表中的字符串连接成一个字符串
        text = "".join(text)
        # 获取所有编码器和新增编码器中的词汇标记
        vocab_tokens = list(self.encoder.keys()) + list(self.added_tokens_encoder.keys())

        # 替换文本中的特殊标记"[SPACE]"为普通空格" "，如果"[SPACE]"存在于词汇标记中
        text = text.replace("[SPACE]", " ") if "[SPACE]" in vocab_tokens else text
        # 替换文本中的特殊标记"[STOP]"为普通空格" "，如果"[STOP]"存在于词汇标记中
        text = text.replace("[STOP]", " ") if "[STOP]" in vocab_tokens else text

        # 替换文本中的未知标记为""，连续的三个空格为一个空格，连续的两个空格为一个空格
        text = text.replace(self.unk_token, "").replace("   ", " ").replace("  ", " ")
        return text

    # 保存词汇表到指定目录，参考了transformers库中GPT2Tokenizer的save_vocabulary方法
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构造词汇文件路径，结合前缀和文件名
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 构造合并文件路径，结合前缀和文件名
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        # 将编码器内容以JSON格式写入词汇文件
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将BPE标记和其索引写入合并文件
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            # 按照BPE索引排序并写入文件
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        # 返回保存的词汇文件路径和合并文件路径
        return vocab_file, merge_file
```