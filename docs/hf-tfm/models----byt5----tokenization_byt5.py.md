# `.\transformers\models\byt5\tokenization_byt5.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的具体语言
""" 用于 ByT5 模型的分词类 """

# 导入警告模块
import warnings
# 导入类型提示模块
from typing import List, Optional, Tuple

# 导入 HuggingFace 的 tokenization_utils 模块
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
# 导入日志模块
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# ByT5Tokenizer 类继承自 PreTrainedTokenizer 类
class ByT5Tokenizer(PreTrainedTokenizer):
    """
    构建一个 ByT5 分词器。ByT5 简单地使用原始字节 utf-8 编码。

    这个分词器继承自 [`PreTrainedTokenizer`]，其中包含大部分主要方法。用户应该参考
    这个超类以获取有关这些方法的更多信息。

    Args:
        eos_token (`str`, *optional*, 默认为 `"</s>"`):
            序列结束标记。

            <Tip>

            在使用特殊标记构建序列时，这不是用于序列结束的标记。
            使用的标记是 `sep_token`。

            </Tip>

        unk_token (`str`, *optional*, 默认为 `"<unk>"`):
            未知标记。词汇表中不存在的标记无法转换为 ID，而是设置为此标记。
        pad_token (`str`, *optional*, 默认为 `"<pad>"`):
            用于填充的标记，例如在批处理不同长度的序列时使用。
        extra_ids (`int`, *optional*, 默认为 125):
            添加一些额外的 ID 到词汇表的末尾，用作哨兵。这些标记可以作为 "<extra_id_{%d}>" 访问，其中 "{%d}" 是 0 到 extra_ids-1 之间的数字。
            额外的标记从词汇表的末尾向前索引 ("<extra_id_0>" 是词汇表中的最后一个标记，就像 ByT5 预处理中看到的那样
            [here](https://github.com/google-research/text-to-text-transfer-transformer/blob/9fd7b14a769417be33bc6c850f9598764913c833/t5/data/preprocessors.py#L2117)。
        additional_special_tokens (`List[str]`, *optional*):
            分词器使用的额外特殊标记。
    """

    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=125,
        additional_special_tokens=None,
        **kwargs,
    ) -> None:
        # 将额外的特殊标记添加到特殊标记列表中
        if extra_ids > 0 and additional_special_tokens is None:
            # 如果额外的标记数大于0，并且没有提供额外的特殊标记，则创建额外的特殊标记列表
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None and len(additional_special_tokens) > 0:
            # 如果额外的标记数大于0，并且提供了额外的特殊标记，并且额外的特殊标记列表不为空
            # 检查额外的特殊标记数量是否正确
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                # 如果额外的特殊标记数量与额外的标记数不匹配，则引发错误
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to ByT5Tokenizer. In this case the additional_special_tokens must include the"
                    " extra_ids tokens"
                )

        # 如果 pad_token 是字符串，则创建一个 AddedToken 对象，并去除左右空白
        pad_token = AddedToken(pad_token, lstrip=True, rstrip=True) if isinstance(pad_token, str) else pad_token
        # 强制左右去除空白以确保向后兼容性，byt5tests 依赖于此
        eos_token = AddedToken(eos_token, lstrip=True, rstrip=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=True, rstrip=True) if isinstance(unk_token, str) else unk_token
        # unk token 需要在词汇表中以正确的索引出现
        self._added_tokens_decoder = {0: pad_token, 1: eos_token, 2: unk_token}
        # 计算偏移量，以便将特殊标记的索引从字节编码转换为整数索引
        self.offset = len(self._added_tokens_decoder)
        # utf 编码使用 8 位
        self._utf_vocab_size = 2**8  # utf is 8 bits
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=0,
            additional_special_tokens=additional_special_tokens,  # TODO extra ids are not used :sweatywmile:
            **kwargs,
        )

    @property
    def vocab_size(self):
        # 返回词汇表大小，包括偏移量
        return self._utf_vocab_size

    def get_vocab(self):
        # 获取词汇表，包括额外添加的标记
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列标识符。在使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, 默认为 `False`):
                标记列表是否已经使用特殊标记格式化。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 正常情况：一些特殊标记
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """如果用户已经添加了 eos，则不再添加 eos。"""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"此序列已经有 {self.eos_token}。在未来版本中，此行为可能导致重复添加 eos 标记。"
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。ByT5 不使用标记类型 ID，因此返回一个由零组成的列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 零值列表。
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        从一个序列或一个序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。序列的格式如下：

        - 单个序列：`X </s>`
        - 序列对：`A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 具有适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        # 如果 token_ids_0 中没有结束标记，则添加结束标记
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        # 如果没有 token_ids_1，则返回 token_ids_0
        if token_ids_1 is None:
            return token_ids_0
        else:
            # 如果 token_ids_1 中没有结束标记，则添加结束标记
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            # 返回连接后的 token_ids_0 和 token_ids_1
            return token_ids_0 + token_ids_1

    def _tokenize(self, text: str) -> List[str]:
        """将字符串作为输入，返回单词/子词的字符串列表（标记）"""
        # 将文本编码为 UTF-8，每个字符转换为一个 token
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """使用词汇表将 token（str）转换为 id"""

        if len(token) != 1:
            token_id = None
        else:
            # 将 token 转换为对应的 id
            token_id = ord(token) + self.offset

        return token_id

    def _convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为 token（str）"""
        # 将索引转换为对应的字符 token
        token = chr(index - self.offset)
        return token

    def convert_tokens_to_string(self, tokens):
        """将一系列 token（字符串）转换为单个字符串"""
        bstring = b""
        for token in tokens:
            if token in self.added_tokens_decoder:
                # 如果 token 是特殊标记，则使用其对应的编码
                tok_string = self.added_tokens_decoder[token].encode("utf-8")
            elif token in self.added_tokens_encoder:
                # 如果 token 是特殊标记，则直接使用
                tok_string = token.encode("utf-8")
            else:
                # 否则将 token 转换为字节并添加到 bstring
                tok_string = bytes([ord(token)])
            bstring += tok_string
        # 将字节字符串解码为 UTF-8 编码的字符串
        string = bstring.decode("utf-8", errors="ignore")
        return string

    # ByT5Tokenizer 没有词汇表文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 返回空元组
        return ()
```