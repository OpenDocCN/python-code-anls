# `.\transformers\models\canine\tokenization_canine.py`

```py
# coding=utf-8
# 该文件的版权声明和许可证信息

# 导入所需的类型提示
from typing import Dict, List, Optional

# 导入预训练的分词器和相关工具函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练的位置嵌入大小字典，用于CANINE模型
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "nielsr/canine-s": 2048,
}

# Unicode定义了总共1,114,112个“码点”
UNICODE_VOCAB_SIZE = 1114112

# 下面是定义特殊的伪字符的规范码点的常量。
# 从 https://github.com/google-research/language/blob/master/language/canine/special_codepoints.py 复制而来

# 表示填充的码点
PAD = 0
# 表示序列的起始的码点
CLS = 0xE000
# 表示序列的终止的码点
SEP = 0xE001
# 表示开始的码点
BOS = 0xE002
# 表示掩码的码点
MASK = 0xE003
# 预留的码点
RESERVED = 0xE004

# 将特殊码点映射到人类可读的名称
SPECIAL_CODEPOINTS: Dict[int, str] = {
    # 特殊符号使用的码点值是有效的，但被指定为“私有使用”，这意味着它们永远不会被Unicode联盟分配字符，并且因此在这里使用是安全的。
    #
    # 注意：不要在这里添加任何类型的[UNK_CHAR]。它们被明确排除，应该以硬错误失败。
    CLS: "[CLS]",
    SEP: "[SEP]",
    BOS: "[BOS]",
    MASK: "[MASK]",
    PAD: "[PAD]",
    RESERVED: "[RESERVED]",
}

# 将特殊码点的人类可读名称映射到它们的码点值
SPECIAL_CODEPOINTS_BY_NAME: Dict[str, int] = {name: codepoint for codepoint, name in SPECIAL_CODEPOINTS.items()}


class CanineTokenizer(PreTrainedTokenizer):
    r"""
    构造一个CANINE分词器（即字符分割器）。它将文本转换为字符序列，然后将每个字符转换为其Unicode码点。

    [`CanineTokenizer`] 继承自 [`PreTrainedTokenizer`]。

    有关参数的用法示例和文档，请参阅超类 [`PreTrainedTokenizer`]。

    Args:
        model_max_length (`int`, *optional*, 默认为 2048):
            模型接受的最大句子长度。
    """

    # 模型输入的最大长度
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        bos_token=chr(CLS),
        eos_token=chr(SEP),
        sep_token=chr(SEP),
        cls_token=chr(CLS),
        pad_token=chr(PAD),
        mask_token=chr(MASK),
        add_prefix_space=False,
        model_max_length=2048,
        **kwargs,
    ):
        # 如果初始的特殊符号是字符串类型，则将其转换为AddedToken对象，并保留其两侧空白
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 如果mask_token是字符串类型，则将其转换为AddedToken对象，并在其左侧剥离空白
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 创建一个特殊符号ID的查找映射
        self._special_codepoints: Dict[str, int] = {}
        for codepoint, name in SPECIAL_CODEPOINTS.items():
            self._special_codepoints[name] = codepoint

        # 创建一个特殊符号ID的字符串形式的查找映射
        self._special_codepoint_strings: Dict[int, str] = {
            codepoint: name for name, codepoint in self._special_codepoints.items()
        }

        # 设置Unicode词汇表的大小和特殊符号的数量
        self._unicode_vocab_size = UNICODE_VOCAB_SIZE
        self._num_special_tokens = len(self._special_codepoints)

        # 调用父类的构造函数，初始化Tokenizer
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        # 返回Unicode词汇表的大小
        return self._unicode_vocab_size

    def get_vocab(self):
        # 构建词汇表，包括Unicode字符和额外添加的特殊符号
        vocab = {chr(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """将字符串分词（即按字符拆分）。"""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """将令牌（即Unicode字符）转换为ID（即其整数Unicode代码点值）。"""
        try:
            return ord(token)
        except TypeError:
            raise ValueError(f"invalid token: '{token}'")

    def _convert_id_to_token(self, index: int) -> str:
        """
        将Unicode代码点（整数）转换为令牌（str）。如果是特殊代码点，则转换为可读的格式。
        """
        try:
            if index in SPECIAL_CODEPOINTS:
                return SPECIAL_CODEPOINTS[index]
            return chr(index)
        except TypeError:
            raise ValueError(f"invalid id: {index}")

    def convert_tokens_to_string(self, tokens):
        # 将令牌列表连接成字符串
        return "".join(tokens)
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A CANINE sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 定义分隔符和类别标识符的 ID 列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 构建输入序列，将类别标识符、第一个序列的 token IDs 和分隔符连接起来
        result = cls + token_ids_0 + sep
        # 如果有第二个序列的 token IDs，将其添加到结果中，并加上分隔符
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

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
        # 如果已经有特殊标记，直接调用父类的方法返回特殊标记的 mask
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 构建特殊标记的 mask，1 表示特殊标记，0 表示序列 token
        result = [1] + ([0] * len(token_ids_0)) + [1]
        # 如果有第二个序列的 token IDs，将其相应的 mask 添加到结果中
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
    ) -> List[int]:
        """
        创建用于序列对分类任务的掩码。CANINE 序列对掩码的格式如下：

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | 第一个序列        | 第二个序列     |
        ```py

        如果 `token_ids_1` 是 `None`，则该方法仅返回掩码的第一部分（全为0）。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                序列对的第二个 ID 列表（可选）。

        Returns:
            `List[int]`: 根据给定序列(s)返回的[令牌类型ID](../glossary#token-type-ids)列表。
        """
        # 分隔符（SEP）令牌
        sep = [self.sep_token_id]
        # 分类（CLS）令牌
        cls = [self.cls_token_id]

        # 结果列表初始化为全0
        result = len(cls + token_ids_0 + sep) * [0]
        # 如果有第二个序列，将其添加到结果列表
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    # CanineTokenizer 没有词汇文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        # 返回空元组
        return ()
```