# `.\transformers\models\layoutlmv3\tokenization_layoutlmv3.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache License, Version 2.0 许可使用代码
# 详细许可信息可在 http://www.apache.org/licenses/LICENSE-2.0 找到
# 本代码在 "AS IS" 基础上分发，无任何明示或暗示的担保或条件
# 查看许可协议获取更多信息
"""LayoutLMv3 的分词类。与 LayoutLMv2 相同，但采用 RoBERTa 风格的 BPE 分词，而不是 WordPiece。"""

# 导入必要的库
import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import regex as re

# 导入 tokenization_utils.py 中的相关类和函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from ...utils import PaddingStrategy, TensorType, add_end_docstrings, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/raw/main/vocab.json",
        "microsoft/layoutlmv3-large": "https://huggingface.co/microsoft/layoutlmv3-large/raw/main/vocab.json",
    },
    "merges_file": {
        "microsoft/layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/raw/main/merges.txt",
        "microsoft/layoutlmv3-large": "https://huggingface.co/microsoft/layoutlmv3-large/raw/main/merges.txt",
    },
}

# 预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/layoutlmv3-base": 512,
    "microsoft/layoutlmv3-large": 512,
}

"""
此处是一个空行
"""

# 使用 lru_cache 装饰器缓存函数结果，提高性能
@lru_cache()
# 从 transformers.models.roberta.tokenization_roberta.bytes_to_unicode 复制的函数
def bytes_to_unicode():
    """
    返回 utf-8 字节列表和映射到 Unicode 字符串的映射。我们特意避免映射到空白字符/控制字符，以免引起 BPE 编码问题。
    
    可逆的 BPE 编码适用于 Unicode 字符串。这意味着如果要避免 UNKs，您需要在词汇表中拥有大量的 Unicode 字符。
    当您的数据集达到约 100 亿个标记时，您最终需要大约 5 千个字符才能得到良好的覆盖率。这占了正常情况下，例如，
    32 千个 bpe 词汇表的显着百分比。为了避免这种情况，我们希望在 utf-8 字节和 Unicode 字符串之间建立查找表。
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
    # 使用内置函数 zip() 将 bs 和 cs 中对应位置的元素一一配对组成元组，并构建一个迭代器对象
    # 再利用 dict() 将这些元组转换为字典并返回
    return dict(zip(bs, cs))
# 从transformers.models.roberta.tokenization_roberta.get_pairs复制过来的函数，用于返回单词中的符号对
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    # 初始化符号对集合
    pairs = set()
    # 获取单词中第一个符号
    prev_char = word[0]
    # 遍历单词中的每个符号
    for char in word[1:]:
        # 将前一个符号和当前符号组成一个符号对，加入符号对集合中
        pairs.add((prev_char, char))
        # 更新前一个符号
        prev_char = char
    # 返回符号对集合

class LayoutLMv3Tokenizer(PreTrainedTokenizer):
    r"""
    建立一个 LayoutLMv3 分词器。基于 RoBERTa 分词器（BPE）。
    LayoutLMv3Tokenizer 可以用于将单词、单词级别的边界框和可选的单词标签转换为令牌级别的 input_ids、attention_mask、token_type_ids、bbox，和可选的标签（用于令牌分类）。

    此分词器继承自 PreTrainedTokenizer，其中包含大部分主要方法。用户应参考该超类了解有关这些方法的更多信息。

    LayoutLMv3Tokenizer 运行端到端的分词：标点符号拆分和 wordpiece。它还将单词级别的边界框转换为令牌级别的边界框。
    """

    # 定义词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入名称
    model_input_names = ["input_ids", "attention_mask", "bbox"]

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
        add_prefix_space=True,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[0, 0, 0, 0],
        pad_token_box=[0, 0, 0, 0],
        pad_token_label=-100,
        only_label_first_subword=True,
        **kwargs,
    ):
        # 将特殊标记转换为AddedToken对象，保证左右没有空格
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        # 将特殊的遮罩标记转换为AddedToken对象，保证左侧有空格，右侧无空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 以UTF-8编码打开词汇文件并加载编码器
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        # 创建解码器，取键值对换
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置错误处理方式
        self.errors = errors  # how to handle errors in decoding
        # 创建字节到unicode的映射
        self.byte_encoder = bytes_to_unicode()
        # 创建unicode到字节的映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 以UTF-8编码打开合并文件并读取文件内容
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        # 将读取的内容按行分割，去掉首尾空行
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        # 创建bpe单词的等级字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 缓存字典
        self.cache = {}
        # 是否添加前缀空格
        self.add_prefix_space = add_prefix_space

        # 应该添加re.IGNORECASE，以便对缩写的大写版本进行BPE合并
        # 编译正则表达式模式
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # 添加属性
        self.cls_token_box = cls_token_box
        self.sep_token_box = sep_token_box
        self.pad_token_box = pad_token_box
        self.pad_token_label = pad_token_label
        self.only_label_first_subword = only_label_first_subword

        # 调用父类初始化方法
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
            cls_token_box=cls_token_box,
            sep_token_box=sep_token_box,
            pad_token_box=pad_token_box,
            pad_token_label=pad_token_label,
            only_label_first_subword=only_label_first_subword,
            **kwargs,
        )

    @property
    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.vocab_size中复制过来
    def vocab_size(self):
        # 返回编码器的长度
        return len(self.encoder)

    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_vocab中复制过来
    def get_vocab(self):
        # 复制编码器的内容并更新为已添加的特殊标记编码器内容，返回词汇表
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 从RobertaTokenizer.bpe复制而来
    def bpe(self, token):
        # 如果token在缓存中，则直接返回其值
        if token in self.cache:
            return self.cache[token]
        # 将token转换为元组形式
        word = tuple(token)
        # 获取token的所有子词对
        pairs = get_pairs(word)

        # 如果不存在子词对，则直接返回token
        if not pairs:
            return token

        # 重复直到找不到更低频的子词对
        while True:
            # 选取频率最低的子词对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            # 如果选取的子词对不在频率表中，则跳出循环
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    # 若未找到子词对，则将剩余部分添加到new_word中
                    new_word.extend(word[i:])
                    break
                else:
                    # 将子词对之前的部分添加到new_word中
                    new_word.extend(word[i:j])
                    i = j

                # 若当前位置处为子词对，则将其合并为一个词，并向后移动两个位置
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 若当前位置处非子词对，则将当前位置处字符添加到new_word中，并向后移动一个位置
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            # 若已将token分割为单个字符，则跳出循环
            if len(word) == 1:
                break
            else:
                # 获取新的子词对
                pairs = get_pairs(word)
        # 将分割后的单词用空格连接，并将结果添加到缓存中
        word = " ".join(word)
        self.cache[token] = word
        return word

    # 从RobertaTokenizer._tokenize复制而来
    def _tokenize(self, text):
        """Tokenize a string."""
        # 用正则表达式匹配文本中的所有token
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            # 将每个token编码为unicode字符串，并将其分割为子词，并将结果扩展到bpe_tokens列表中
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # 从RobertaTokenizer._convert_token_to_id复制而来
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将token转换为其对应的id，若不存在则使用未知标记的id
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # 从RobertaTokenizer._convert_id_to_token复制而来
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将索引转换为对应的token
        return self.decoder.get(index)

    # 从RobertaTokenizer.convert_tokens_to_string复制而来
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 将tokens列表中的token连接为单个字符串
        text = "".join(tokens)
        # 将每个字符的字节编码解码为unicode字符，并将结果转换为utf-8编码的字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text
    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.save_vocabulary复制而来的函数，用于保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则报错
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接词汇表文件路径，并保存词汇表字典内容
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        merge_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        # 将BPE标记与索引写入文件
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

    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.build_inputs_with_special_tokens复制而来的函数，用于构建带有特殊标记的模型输入
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        根据特殊标记，将一个或一对序列构建为用于序列分类任务的模型输入。RoBERTa序列的格式如下：
        
        - 单个序列： `<s> X </s>`
        - 一对序列： `<s> A </s></s> B </s>`

        参数:
            token_ids_0 (`List[int]`):
                用于添加特殊标记的ID列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的用于序列对的第二个ID列表。

        返回值:
            `List[int]`: 带有适当特殊标记的[input IDs](../glossary#input-ids)列表。
        """
        # 如果没有第二个ID列表，则返回含有特殊标记的输入
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    # 从transformers.models.roberta.tokenization_roberta.RobertaTokenizer.get_special_tokens_mask复制而来的函数，用于获取特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    # 定义一个方法，用于从没有添加特殊标记的标记列表中检索序列 ID。当使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。
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
        # 如果已经添加了特殊标记，则直接调用父类的 `get_special_tokens_mask` 方法
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果未添加特殊标记，则根据参数中的标记列表生成特殊标记掩码
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    # 从 `transformers.models.roberta.tokenization_roberta.RobertaTokenizer` 中复制过来的方法
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 初始化分隔和类别标记
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果只有一个序列，则返回由 0 组成的列表，因为 RoBERTa 不使用标记类型 ID
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有两个序列，则返回由 0 组成的列表，因为 RoBERTa 不使用标记类型 ID
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    # 准备文本进行分词化的方法
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        # 如果文本以不应分割的标记开头，则在文本之前不添加空格
        # 这对于匹配快速分词化是必要的
        if (
            (is_split_into_words or add_prefix_space)
            and (len(text) > 0 and not text[0].isspace())
            and sum([text.startswith(no_split_token) for no_split_token in self.added_tokens_encoder]) == 0
        ):
            text = " " + text
        return (text, kwargs)

    # 添加文档字符串的装饰器，用于将文档字符串添加到当前方法
    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从 `transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer` 中复制过来的方法
    # 定义一个方法，用于将输入文本进行编码处理
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        text_pair: Optional[Union[PreTokenizedInput, List[PreTokenizedInput]]] = None,
        boxes: Union[List[List[int]], List[List[List[int]]]] = None,
        word_labels: Optional[Union[List[int], List[List[int]]] = None,
        add_special_tokens: bool = True,  # 是否添加特殊的标记符
        padding: Union[bool, str, PaddingStrategy] = False,  # 是否进行填充操作
        truncation: Union[bool, str, TruncationStrategy] = None,  # 是否截断文本
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的长度
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型标识
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记符的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否打印详细信息
        **kwargs,
    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)  # 添加文档字符串
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer.batch_encode_plus 复制过来的代码块
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,  # 是否为文本对
        boxes: Optional[List[List[List[int]]] = None,  # 文本框坐标
        word_labels: Optional[Union[List[int], List[List[int]]] = None,  # 单词标签
        add_special_tokens: bool = True,  # 是否添加特殊的标记符
        padding: Union[bool, str, PaddingStrategy] = False,  # 是否进行填充操作
        truncation: Union[bool, str, TruncationStrategy] = None,  # 是否截断文本
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定的长度
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token 类型标识
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩��
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记符的掩码
        return_offsets_mapping: bool = False,  # 是否返回偏移映射
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否打印详细信息
        **kwargs,
    # 声明一个函数，用于将文本或文本对编码成批量编码
    ) -> BatchEncoding:
        # 为了向后兼容 'truncation_strategy'，'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 返回批量编码结果
        return self._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            is_pair=is_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    # 从transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer._batch_encode_plus复制
    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
        ],
        is_pair: bool = None,
        boxes: Optional[List[List[List[int]]] = None,
        word_labels: Optional[List[List[int]]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    # 定义函数，接受参数并返回BatchEncoding对象
    ) -> BatchEncoding:
        # 如果return_offsets_mapping为True，抛出NotImplementedError异常
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        # 调用_batch_prepare_for_model方法处理输入数据
        batch_outputs = self._batch_prepare_for_model(
            batch_text_or_text_pairs=batch_text_or_text_pairs,  # 输入文本或文本对
            is_pair=is_pair,  # 是否为文本对
            boxes=boxes,  # 文本框坐标
            word_labels=word_labels,  # 词标签
            add_special_tokens=add_special_tokens,  # 是否添加特殊标记
            padding_strategy=padding_strategy,  # 填充策略
            truncation_strategy=truncation_strategy,  # 截断策略
            max_length=max_length,  # 最大长度
            stride=stride,  # 步长
            pad_to_multiple_of=pad_to_multiple_of,  # 填充到指定长度的倍数
            return_attention_mask=return_attention_mask,  # 是否返回注意力掩码
            return_token_type_ids=return_token_type_ids,  # 是否返回token_type_ids
            return_overflowing_tokens=return_overflowing_tokens,  # 是否返回溢出的tokens
            return_special_tokens_mask=return_special_tokens_mask,  # 是否返回特殊标记掩码
            return_length=return_length,  # 是否返回长度
            return_tensors=return_tensors,  # 是否返回张量
            verbose=verbose,  # 是否输出详细信息
        )

        # 返回BatchEncoding对象
        return BatchEncoding(batch_outputs)

    # 添加文档字符串
    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer._batch_prepare_for_model进行复制
    def _batch_prepare_for_model(
        self,
        batch_text_or_text_pairs,  # 输入文本或文本对
        is_pair: bool = None,  # 是否为文本对
        boxes: Optional[List[List[int]]] = None,  # 文本框坐标
        word_labels: Optional[List[List[int]]] = None,  # 词标签
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,  # 填充策略
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,  # 截断策略
        max_length: Optional[int] = None,  # 最大长度
        stride: int = 0,  # 步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到指定长度的倍数
        return_tensors: Optional[str] = None,  # 是否返回张量
        return_token_type_ids: Optional[bool] = None,  # 是否返回token_type_ids
        return_attention_mask: Optional[bool] = None,  # 是否返回注意力掩码
        return_overflowing_tokens: bool = False,  # 是否返回溢出的tokens
        return_special_tokens_mask: bool = False,  # 是否返回特殊标记掩码
        return_length: bool = False,  # 是否返回长度
        verbose: bool = True,  # 是否输出详细信息
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens.

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        # 初始化一个空的字典，用于存储批处理输出
        batch_outputs = {}
        # 使用 enumerate() 遍历 batch_text_or_text_pairs 和 boxes 列表中的元素
        for idx, example in enumerate(zip(batch_text_or_text_pairs, boxes)):
            # 解压缩元组 example
            batch_text_or_text_pair, boxes_example = example
            # 使用模型的 prepare_for_model 方法处理输入，得到模型可以使用的格式
            outputs = self.prepare_for_model(
                # 如果是成对的输入，只获取第一个文本或者文本对
                batch_text_or_text_pair[0] if is_pair else batch_text_or_text_pair,
                # 如果是成对的输入，获取第二个文本，否则为 None
                batch_text_or_text_pair[1] if is_pair else None,
                # boxes_example 是当前输入对应的边界框信息
                boxes_example,
                # 如果提供了单词标签，传递给 prepare_for_model 方法
                word_labels=word_labels[idx] if word_labels is not None else None,
                # 是否添加特殊标记
                add_special_tokens=add_special_tokens,
                # 填充策略，此处设置为不填充，因为之后会在批处理中填充
                padding=PaddingStrategy.DO_NOT_PAD.value,
                # 截断策略
                truncation=truncation_strategy.value,
                # 最大长度
                max_length=max_length,
                # 步幅
                stride=stride,
                # 设置为 None，因为之后会在批处理中填充
                pad_to_multiple_of=None,
                # 设置为 False，因为之后会在批处理中填充
                return_attention_mask=False,
                # 是否返回 token 类型 ID
                return_token_type_ids=return_token_type_ids,
                # 是否返回溢出的 token
                return_overflowing_tokens=return_overflowing_tokens,
                # 是否返回特殊 token 掩码
                return_special_tokens_mask=return_special_tokens_mask,
                # 是否返回长度
                return_length=return_length,
                # 返回的张量类型，此处为 None，因为最后会将整个批次转换为张量
                return_tensors=None,
                # 是否在输出前添加批次轴
                prepend_batch_axis=False,
                # 是否详细输出
                verbose=verbose,
            )

            # 将每个输出添加到 batch_outputs 字典中
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        # 在批处理输出中执行填充
        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 将填充后的批处理输出转换为 BatchEncoding 对象
        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING)
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer.encode 复制而来
    # 定义一个方法用于将文本编码为模型可接受的输入
    def encode(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 文本输入，可以是原始文本或预分词输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的文本对输入，用于处理文本对任务
        boxes: Optional[List[List[int]]] = None,  # 文本框坐标信息
        word_labels: Optional[List[int]] = None,  # 单词级别的标签信息
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，控制是否进行填充
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略，控制是否进行截断
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 滑动窗口步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到的长度的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token type ids
        return_attention_mask: Optional[bool] = None,  # 是否返回 attention mask
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token
        return_special_tokens_mask: bool = False,  # 是否返回特殊 token 的 mask
        return_offsets_mapping: bool = False,  # 是否返回 token 的偏移映射
        return_length: bool = False,  # 是否返回编码后的长度
        verbose: bool = True,  # 是否显示详细信息
        **kwargs,  # 其它关键字参数
    ) -> List[int]:  # 返回编码后的输入 ID 列表
        # 调用 encode_plus 方法对输入进行编码
        encoded_inputs = self.encode_plus(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
    
        # 返回编码后的输入 ID 列表
        return encoded_inputs["input_ids"]
    
    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 从 transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer.encode_plus 复制注释
    def encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],  # 文本输入，可以是原始文本或预分词输入
        text_pair: Optional[PreTokenizedInput] = None,  # 可选的文本对输入，用于处理文本对任务
        boxes: Optional[List[List[int]]] = None,  # 文本框坐标信息
        word_labels: Optional[List[int]] = None,  # 单词级别的标签信息
        add_special_tokens: bool = True,  # 是否添加特殊标记
        padding: Union[bool, str, PaddingStrategy] = False,  # 填充策略，控制是否进行填充
        truncation: Union[bool, str, TruncationStrategy] = None,  # 截断策略，控制是否进行截断
        max_length: Optional[int] = None,  # 最大长度限制
        stride: int = 0,  # 滑动窗口步长
        pad_to_multiple_of: Optional[int] = None,  # 填充到的长度的倍数
        return_tensors: Optional[Union[str, TensorType]] = None,  # 返回的张量类型
        return_token_type_ids: Optional[bool] = None,  # 是否返回 token type ids
        return_attention_mask: Optional[bool] = None,  # 是否返回 attention mask
        return_overflowing_tokens: bool = False,  # 是否返回溢出的 token
        return_special_tokens_mask: bool = False,  # 是否返回特殊 token 的 mask
        return_offsets_mapping: bool = False,  # 是否返回 token 的偏移映射
        return_length: bool = False,  # 是否返回编码后的长度
        verbose: bool = True,  # 是否显示详细信息
        **kwargs,  # 其它关键字参数
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a sequence or a pair of sequences. .. warning:: This method is deprecated,
        `__call__` should be used instead.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The first sequence to be encoded. This can be a string, a list of strings or a list of list of strings.
            text_pair (`List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a list of strings (words of a single example) or a
                list of list of strings (words of a batch of examples).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # 获取填充和截断策略，以及其他相关参数
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        # 调用 _encode_plus 方法进行编码处理
        return self._encode_plus(
            text=text,
            boxes=boxes,
            text_pair=text_pair,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    # Copied from transformers.models.layoutlmv2.tokenization_layoutlmv2.LayoutLMv2Tokenizer._encode_plus
    # 对象方法，用于执行编码处理
    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    # 定义一个带有返回类型注解的方法，接受一个 BatchEncoding 对象，并根据参数进行处理
    ) -> BatchEncoding:
        # 如果参数 return_offsets_mapping 为 True，则抛出 NotImplementedError 异常，因为 Python tokenizers 不支持该功能
        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast. "
                "More information on available tokenizers at "
                "https://github.com/huggingface/transformers/pull/2674"
            )
        # 调用内部方法 prepare_for_model 处理输入参数并返回处理后的 BatchEncoding 对象
        return self.prepare_for_model(
            text=text,
            text_pair=text_pair,
            boxes=boxes,
            word_labels=word_labels,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )
    # 根据两个文档字符串添加额外的信息
    @add_end_docstrings(LAYOUTLMV3_ENCODE_KWARGS_DOCSTRING, LAYOUTLMV3_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # 定义一个方法用于准备模型输入数据
    def prepare_for_model(
        self,
        text: Union[TextInput, PreTokenizedInput],
        text_pair: Optional[PreTokenizedInput] = None,
        boxes: Optional[List[List[int]]] = None,
        word_labels: Optional[List[int]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs,
    # 从 LayoutLMv2Tokenizer.truncate_sequences 复制了一个方法来截断序列
    def truncate_sequences(
        self,
        ids: List[int],
        token_boxes: List[List[int]],
        pair_ids: Optional[List[int]] = None,
        pair_token_boxes: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        num_tokens_to_remove: int = 0,
        truncation_strategy: Union[str, TruncationStrategy] = "longest_first",
        stride: int = 0,
    # 从 LayoutLMv2Tokenizer._pad 复制了一个方法来填充
    # 定义一个内部方法用于填充输入序列的长度
    def _pad(
        self,
        # 编码后的输入，可以是字典形式的编码输入或批量编码对象
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        # 最大长度，如果不指定，默认为 None
        max_length: Optional[int] = None,
        # 填充策略，默认为不填充
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        # 填充到的长度的倍数，如果不指定，默认为 None
        pad_to_multiple_of: Optional[int] = None,
        # 是否返回注意力掩码，默认为 None
        return_attention_mask: Optional[bool] = None,
```