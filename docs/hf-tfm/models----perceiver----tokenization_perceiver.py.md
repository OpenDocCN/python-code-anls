# `.\models\perceiver\tokenization_perceiver.py`

```py
# coding=utf-8
# 版权 2021 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证版本 2.0 授权。
# 除非符合许可证要求或书面同意，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件按"原样"分发，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" Perceiver 的分词器类。"""


from typing import Dict, List, Optional, Tuple

# 导入父类 PreTrainedTokenizer 和一些其他必要的模块
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)


class PerceiverTokenizer(PreTrainedTokenizer):
    """
    构建一个 Perceiver 分词器。Perceiver 简单地使用原始字节 utf-8 编码。

    这个分词器继承自 [`PreTrainedTokenizer`]，该类包含大部分主要方法。用户应参考这个父类获取更多有关这些方法的信息。

    Args:
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            用于填充的标记，在批处理不同长度的序列时使用。
        bos_token (`str`, *optional*, defaults to `"[BOS]"`):
            BOS 标记（在词汇表中保留，但实际上不使用）。
        eos_token (`str`, *optional*, defaults to `"[EOS]"`):
            序列结束标记（在词汇表中保留，但实际上不使用）。

            <Tip>

            当使用特殊标记构建序列时，这不是实际用于序列结束的标记。
            实际用于结束序列的标记是 `sep_token`。

            </Tip>

        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            用于掩码语言建模的 MASK 标记。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            CLS 标记（在词汇表中保留，但实际上不使用）。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，在从两个序列构建一个序列时使用。

    """

    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=2048,
        **kwargs,
    ):
        # 初始化函数，设置分词器的各种特殊标记及其默认值
        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            **kwargs,
        )
    ) -> None:
        # 如果 pad_token 是字符串，则封装为 AddedToken 对象；否则直接使用传入的对象
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        # 如果 bos_token 是字符串，则封装为 AddedToken 对象；否则直接使用传入的对象
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串，则封装为 AddedToken 对象；否则直接使用传入的对象
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        # 如果 mask_token 是字符串，则封装为 AddedToken 对象；否则直接使用传入的对象
        mask_token = AddedToken(mask_token, lstrip=False, rstrip=False) if isinstance(mask_token, str) else mask_token
        # 如果 cls_token 是字符串，则封装为 AddedToken 对象；否则直接使用传入的对象
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        # 如果 sep_token 是字符串，则封装为 AddedToken 对象；否则直接使用传入的对象
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token

        # 初始化 UTF-8 编码的词汇表大小为 2 的 8 次方（256）
        self._utf_vocab_size = 2**8  # utf is 8 bits

        # 这些特殊 token 不在词汇表中，因此我们手动将它们添加到解码器中
        self._added_tokens_decoder: Dict[str, int] = {
            0: pad_token,
            1: bos_token,
            2: eos_token,
            3: mask_token,
            4: cls_token,
            5: sep_token,
        }
        # 特殊 token 的数量
        self._num_special_tokens = len(self._added_tokens_decoder)
        # 调用父类的构造方法，初始化基本特殊 token 和模型最大长度等参数
        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            model_max_length=model_max_length,
            **kwargs,
        )

    def get_vocab(self) -> Dict[str, int]:
        # 初始化一个空的词汇表字典
        vocab = {}
        # 遍历 UTF-8 编码范围内的所有字符
        for i in range(self._utf_vocab_size):
            # 将每个字符转换为对应的 token，索引从特殊 token 的数量开始递增
            token = chr(i)
            vocab[token] = i + self._num_special_tokens
        # 将已添加的特殊 token 编码器加入词汇表中
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def vocab_size(self):
        # 返回 UTF-8 编码的词汇表大小
        return self._utf_vocab_size

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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
        if already_has_special_tokens:
            # If the token list already has special tokens, delegate to superclass method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Normal case: adding special tokens to the token lists
        if token_ids_1 is None:
            # For a single sequence, add `[CLS]`, sequence tokens, and `[SEP]`
            return [1] + [0] * len(token_ids_0) + [1]
        else:
            # For a pair of sequences, add `[CLS]`, tokens from the first sequence, `[SEP]`, tokens from the second sequence, and `[SEP]`
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks. A sequence has the
        following format:

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
        if token_ids_1 is None:
            # If there is only one sequence, add `[CLS]`, tokens, and `[SEP]`
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        else:
            # If there are two sequences, add `[CLS]`, tokens from the first sequence, `[SEP]`, tokens from the second sequence, and `[SEP]`
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) into an ID using the vocabulary."""
        if len(token) != 1:
            token_id = self.unk_token_id
        else:
            token_id = ord(token) + self._num_special_tokens
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) into a token (str) using the vocabulary."""
        token = chr(index - self._num_special_tokens)
        return token

    # TODO @ArthurZ refactor this as well....
    # 将一系列的标记（字符串）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        # 初始化一个空的字节字符串
        bstring = b""
        # 遍历每个标记
        for token in tokens:
            # 如果标记在已添加标记的编码器中
            if token in self.added_tokens_encoder:
                # 将标记转换为 UTF-8 编码的字节序列
                tok_string = str(token).encode("utf-8")
            else:
                # 否则，将标记转换为对应的字节值
                tok_string = bytes([ord(token)])
            # 将处理后的字节串添加到总字节字符串中
            bstring += tok_string
        # 将字节串解码为 UTF-8 编码的字符串，使用替换错误处理方式
        string = bstring.decode("utf-8", errors="replace")
        # 返回最终的字符串
        return string

    # PerceiverTokenizer 没有词汇表文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 返回一个空元组，因为 PerceiverTokenizer 没有需要保存的词汇表
        return ()
```