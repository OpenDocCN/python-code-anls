# `.\models\deberta\tokenization_deberta.py`

```
# 设置编码格式为 UTF-8
# 版权声明，指出代码版权和使用许可
# 声明 License，指出可以根据 Apache License, Version 2.0 许可使用代码
# 提供获取许可证的链接
# 明确软件分发基于 "AS IS" 原则，没有明确的担保或条件
# 规定具体语言，使得只有特定语言才符合权限，和在许可书面同意的情况下，才可以分发软件
# 查看HuggingFace Inc.团队和Microsoft 2020的DeBERTa模型的分词类
import json
import os
from typing import List, Optional, Tuple

import regex as re  # 导入正则表达式库

from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 从相对路径导入Tokenization相关模块
from ...utils import logging  # 从相对路径导入logging模块

# 获取logger对象
logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}  # 定义词汇文件名

# 定义预训练模型词汇文件路径映射
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

# 定义预训练模型位置embedding大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-base": 512,
    "microsoft/deberta-large": 512,
    "microsoft/deberta-xlarge": 512,
    "microsoft/deberta-base-mnli": 512,
    "microsoft/deberta-large-mnli": 512,
    "microsoft/deberta-xlarge-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {  # 定义预训练模型初始化配置
    "microsoft/deberta-base": {"do_lower_case": False},
    "microsoft/deberta-large": {"do_lower_case": False},
}

# 从transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode中复制bytes_to_unicode函数
def bytes_to_unicode():
    """
    返回 utf-8 字节列表和到 unicode 字符串的映射。我们特别避免映射到 bpe 代码无法处理的空格/控制字符。

    可逆的 bpe 代码适用于 unicode 字符串。这意味着如果你想避免 UNKs，你需要在你的词汇表中有大量的 unicode 字符。当你处理大约 100 亿标记的数据集时，你最终需要大约 5000 个字符才能达到良好的覆盖率。这在正常情况下，比如 32K bpe 词汇表中占据了相当大的比例。为了避免这种情况，我们需要 utf-8 字节和 unicode 字符串之间的查找表。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )  # 获取所有可打印字符的 utf-8 编码范围
    cs = bs[:]  # 复制 utf-8 编码列表
    n = 0  # 初始化计数器 n 为 0
    for b in range(2**8):  # 遍历 0 到 2^8 范围内的所有数字
        if b not in bs:  # 如果数字不在 utf-8 编码列表中
            bs.append(b)  # 将该数字添加到 utf-8 编码列表
            cs.append(2**8 + n)  # 将 2^8 + n 添加到 cs 列表
            n += 1  # 计数器 n 自增 1
    cs = [chr(n) for n in cs]  # 将 cs 列表中的数字转换成对应的 unicode 字符
    return dict(zip(bs, cs))  # 返回 bs 和 cs 的映射关系构成的字典
# 从transformers.models.gpt2.tokenization_gpt2.get_pairs复制过来的函数
def get_pairs(word):
    """
    返回单词中的符号对集合。

    单词表示为符号元组（符号是可变长度字符串）。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class DebertaTokenizer(PreTrainedTokenizer):
    """
    构造一个DeBERTa tokenizer，基于字节级的字对编码。

    这个tokenizer已经训练成将空格视为标记的一部分（有点像sentencepiece），因此一个词在句子开头（没有空格）和其他位置编码方式是不同的：

    ```python
    >>> from transformers import DebertaTokenizer

    >>> tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    >>> tokenizer("Hello world")["input_ids"]
    [1, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [1, 20920, 232, 2]
    ```

    当您在实例化这个tokenizer或在对一些文本调用它时，可以通过传递`add_prefix_space=True`来避免这种行为，但是由于模型没有以这种方式进行预训练，所以可能会导致性能下降。

    <提示>

    当与`is_split_into_words=True`一起使用时，该tokenizer将在每个单词之前都添加一个空格（甚至第一个单词也是如此）。

    </提示>

    这个tokenizer继承了[`PreTrainedTokenizer`]，其中包含了大部分主要方法。用户应该参考这个超类获取有关这些方法的更多信息。
    # 定义类的初始化方法，创建一个新的tokenizer对象
    # vocab_file: 词汇表文件的路径
    # merges_file: 合并文件的路径
    # errors: 解码字节为UTF-8时遇到错误时的处理方式，默认为"replace"
    # bos_token: 序列的开头标记，默认为"[CLS]"
    # eos_token: 序列的结尾标记，默认为"[SEP]"
    # sep_token: 用于构建多个序列时的分隔符，默认为"[SEP]"
    # cls_token: 用于序列分类时的分类器标记，默认为"[CLS]"
    # unk_token: 未知标记，不在词汇表中的token将被设置为此标记，默认为"[UNK]"
    # pad_token: 用于填充的token，当批量处理不同长度的序列时使用，默认为"[PAD]"
    # mask_token: 用于掩盖值的token，用于掩盖语言模型训练时的token，默认为"[MASK]"
    # add_prefix_space: 是否在输入的开头添加一个空格以便处理开头的单词，默认为False
    # add_bos_token: 是否在输入的开头添加一个<|endoftext|>，以便处理开头的单词，默认为False
    # vocab_files_names: 词汇表文件的名称，定义在VOCAB_FILES_NAMES中
    # pretrained_vocab_files_map: 预训练词汇表文件的映射，定义在PRETRAINED_VOCAB_FILES_MAP中
    # max_model_input_sizes: 预训练位置嵌入的最大输入尺寸，定义在PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES中
    # model_input_names: 模型输入的名称列表，包括"input_ids", "attention_mask", "token_type_ids"
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="[CLS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs,
    ):
        # 如果 bos_token 是字符串，则将其转换为特殊的 AddedToken 对象
        bos_token = AddedToken(bos_token, special=True) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串，则将其转换为特殊的 AddedToken 对象
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token
        # 如果 sep_token 是字符串，则将其转换为特殊的 AddedToken 对象
        sep_token = AddedToken(sep_token, special=True) if isinstance(sep_token, str) else sep_token
        # 如果 cls_token 是字符串，则将其转换为特殊的 AddedToken 对象
        cls_token = AddedToken(cls_token, special=True) if isinstance(cls_token, str) else cls_token
        # 如果 unk_token 是字符串，则将其转换为特殊的 AddedToken 对象
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则将其转换为特殊的 AddedToken 对象
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token

        # Mask token 的行为与普通单词相同，即在其之前包含空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        # 是否添加 bos_token
        self.add_bos_token = add_bos_token

        # 打开词汇文件并加载编码器
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建解码器
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置解码错误的处理方式
        self.errors = errors
        # 创建字节到 Unicode 的编码器和解码器
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 打开 merges 文件并读取 BPE 合并信息
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 解析 BPE 合并信息
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建 BPE 合并等级字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 缓存
        self.cache = {}
        # 是否在前缀之前添加空格
        self.add_prefix_space = add_prefix_space

        # 为了能够进行大小写不敏感的 BPE 合并，应该添加 re.IGNORECASE
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 调用父类的构造函数
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
        # 返回编码器的长度
        return len(self.encoder)

    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.get_vocab 复制而来
    def get_vocab(self):
        # 返回编码器和添加的特殊编码器的字典
        return dict(self.encoder, **self.added_tokens_encoder)

    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe 复制而来
    # 对给定的单词进行 BPE（字节对编码）处理，将其拆分为子词
    def bpe(self, token):
        # 如果单词已经在缓存中，则直接返回缓存中的结果
        if token in self.cache:
            return self.cache[token]
        # 将单词转换为元组形式
        word = tuple(token)
        # 获取单词的所有字节对
        pairs = get_pairs(word)

        # 如果没有字节对，则直接返回原单词
        if not pairs:
            return token

        # 进行字节对编码处理，直到无法再进行编码为止
        while True:
            # 选择出现频率最低的字节对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果字节对不在 BPE 词表中，则跳出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            # 遍历单词中的每个字符
            while i < len(word):
                try:
                    # 查找字节对的第一个字符的位置
                    j = word.index(first, i)
                except ValueError:
                    # 如果找不到，则将剩余字符加入新单词中并结束循环
                    new_word.extend(word[i:])
                    break
                else:
                    # 将第一个字符之前的字符加入新单词中
                    new_word.extend(word[i:j])
                    i = j

                # 如果当前字符与下一个字符组成字节对，则将字节对加入新单词中
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则将当前字符加入新单词中
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 如果新单词长度为1，则跳出循环
            if len(word) == 1:
                break
            else:
                # 否则继续进行字节对编码处理
                pairs = get_pairs(word)
        # 将单词的子词用空格连接起来形成新单词，并将结果加入缓存
        word = " ".join(word)
        self.cache[token] = word
        # 返回 BPE 处理后的单词
        return word

    # 构建带有特殊标记的输入序列，用于序列分类任务
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A DeBERTa sequence has the following format:

        - single sequence: [CLS] X [SEP]
        - pair of sequences: [CLS] A [SEP] B [SEP]

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果没有第二个序列，则直接在第一个序列的开头和结尾添加特殊标记
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 如果有第二个序列，则在两个序列之间添加特殊标记，并返回拼接后的结果
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    # 获取特殊标记的掩码，用于指示输入中哪些位置包含特殊标记
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    # 方法签名，接受两个 token ID 列表参数以及一个是否已经包含特殊 token 的标志
    # 如果 token ID 列表已经包含特殊 token，则直接返回特殊 token 掩码
    # 否则根据 token_ids_1 是否存在来生成特殊 token 掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        # 如果已经包含特殊 token，则调用父类方法来获取特殊 token 掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果 token_ids_1 为 None，则在 token_ids_0 前后加上特殊 token 掩码
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 在 token_ids_0 和 token_ids_1 前后加上特殊 token 掩码
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    # 创建一个用于序列对分类任务的 token 类型 ID 列表
    # 返回序列对分类任务的 token 类型 ID 列表，特定格式为 0 表示第一序列，1 表示第二序列
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # 初始化分隔符和类别标识
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果 token_ids_1 为 None，则返回只有第一个序列的 token 类型 ID 列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 返回合并两个序列的 token 类型 ID 列表
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    # 采用 GPT2Tokenizer 的 _tokenize 方法，对文本进行分词处理
    # 返回分词后的 BPE token
    def _tokenize(self, text):
        # 初始化空 BPE token 列表
        bpe_tokens = []
        # 遍历通过正则表达式找到的 token
        for token in re.findall(self.pat, text):
            # 将 token 转换为字节编码，然后通过 BPE 算法进行分词处理
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # 将所有字节映射为 Unicode 字符串，避免 BPE 中的控制标记（在这种情况下是空格）
            # 将分词后的 BPE token 加入列表中
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens
    # 将token转换为对应的id，如果token不在词汇表中，则使用unk_token代替
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token))
    
    # 从词汇表中将id转换为对应的token
    def _convert_id_to_token(self, index):
        return self.decoder.get(index)
    
    # 将token序列转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        text = "".join(tokens)
        # 使用byte_decoder将text中的字符转换为utf-8编码的字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    
    # 保存词汇表到指定的目录，包括词汇文件和合并文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 保存词汇文件
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 保存合并文件
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )
        # 将词汇表encoder保存到词汇文件中
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        # 将合并文件中包含的BPE token和对应的token_index按照token_index排序后保存到合并文件中
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
        return vocab_file, merge_file
    
    # 准备用于tokenization的文本，如果文本不是以空白字符开头且is_split_into_words为True或add_prefix_space为True，则在文本前添加空格
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if (is_split_into_words or add_prefix_space) and (len(text) > 0 and not text[0].isspace()):
            text = " " + text
        return (text, kwargs)
```