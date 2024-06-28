# `.\models\barthez\tokenization_barthez.py`

```py
# coding=utf-8
# 版权所有 2020 年 Ecole Polytechnique 和 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本（“许可证”），您只有在遵守许可证的情况下才能使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" BARThez 模型的分词类。"""


import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm  # 导入 sentencepiece 库

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging  # 导入 logging 模块


logger = logging.get_logger(__name__)  # 获取当前模块的 logger 对象

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}  # 词汇文件名字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "moussaKam/mbarthez": "https://huggingface.co/moussaKam/mbarthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez": "https://huggingface.co/moussaKam/barthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez-orangesum-title": (
            "https://huggingface.co/moussaKam/barthez-orangesum-title/resolve/main/sentencepiece.bpe.model"
        ),
    },
}  # 预训练词汇文件映射

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "moussaKam/mbarthez": 1024,
    "moussaKam/barthez": 1024,
    "moussaKam/barthez-orangesum-title": 1024,
}  # 预训练位置嵌入的尺寸

SPIECE_UNDERLINE = "▁"  # SentencePiece 的特殊标记

# TODO this class is useless. This is the most standard sentencpiece model. Let's find which one is closest and nuke this.


class BarthezTokenizer(PreTrainedTokenizer):
    """
    从 `CamembertTokenizer` 和 `BartTokenizer` 改编而来。构建一个 BARThez 分词器。基于
    [SentencePiece](https://github.com/google/sentencepiece)。

    此分词器继承自 `PreTrainedTokenizer`，其中包含大多数主要方法。用户应参考
    此超类以获取关于这些方法的更多信息。

    Attributes:
        sp_model (`SentencePieceProcessor`):
            用于所有转换（字符串、标记和 ID）的 SentencePiece 处理器。
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 词汇文件名字典
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练位置嵌入的尺寸
    model_input_names = ["input_ids", "attention_mask"]  # 模型输入的名称列表

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",  # 开始标记
        eos_token="</s>",  # 结束标记
        sep_token="</s>",  # 分隔标记
        cls_token="<s>",  # 类别标记
        unk_token="<unk>",  # 未知标记
        pad_token="<pad>",  # 填充标记
        mask_token="<mask>",  # 掩码标记
        sp_model_kwargs: Optional[Dict[str, Any]] = None,  # SentencePiece 模型参数字典，默认为空
        **kwargs,  # 其他参数
    ) -> None:
        """
        初始化一个新的 BARThezTokenizer 对象。

        Args:
            mask_token (`Union[str, AddedToken]`):
                用作掩码标记的特殊令牌。如果是字符串，则 lstrip=True，special=True。
            sp_model_kwargs (`Optional[Dict]`, *optional*):
                SentencePiece 模型的额外参数，默认为空字典。
            vocab_file (`Optional[Union[str, Path]]`):
                词汇文件的路径。
            bos_token (`Optional[str]`, *optional*):
                用作开头（beginning of sequence）标记的特殊令牌。
            eos_token (`Optional[str]`, *optional*):
                用作结尾（end of sequence）标记的特殊令牌。
            unk_token (`Optional[str]`, *optional*):
                用作未知标记的特殊令牌。
            sep_token (`Optional[str]`, *optional*):
                用作分隔标记的特殊令牌。
            cls_token (`Optional[str]`, *optional*):
                用作类标记的特殊令牌。
            pad_token (`Optional[str]`, *optional*):
                用作填充标记的特殊令牌。
            **kwargs:
                其他参数传递给父类构造函数。
        """
        # 如果 mask_token 是字符串，则创建一个 AddedToken 对象，lstrip=True，special=True
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token

        # 如果 sp_model_kwargs 为 None，则设为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置词汇文件路径
        self.vocab_file = vocab_file
        # 使用 SentencePieceProcessor 加载并初始化模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        # 调用父类的初始化方法，传递参数并初始化对象
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        为序列分类任务构建模型输入，通过连接和添加特殊标记。BARThez 序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 序列对: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`Optional[List[int]]`, *optional*):
                第二个序列的 ID 列表，用于序列对输入。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """

        if token_ids_1 is None:
            # 返回只包含一个序列的特殊标记后的输入 ID 列表
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 构建包含两个序列的特殊标记后的输入 ID 列表
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        返回包含特殊标记的掩码列表，用于指示输入中的特殊标记位置。

        Args:
            token_ids_0 (`List[int]`):
                输入序列的 ID 列表。
            token_ids_1 (`Optional[List[int]]`, *optional*):
                第二个序列的 ID 列表，用于序列对输入。
            already_has_special_tokens (`bool`, *optional*):
                如果输入已包含特殊标记，则为 True。

        Returns:
            `List[int]`: 标记了特殊标记位置的掩码列表。
        """
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
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
            # If the token list already contains special tokens, delegate to superclass method
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            # Return a mask with 1 for the added special tokens (CLS and SEP) and 0 for sequence tokens
            return [1] + ([0] * len(token_ids_0)) + [1]
        else:
            # Return a mask with 1 for the added special tokens (CLS, SEP) for both sequences and 0s for sequence tokens
            return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        sep = [self.sep_token_id]  # Get the separator token ID
        cls = [self.cls_token_id]  # Get the classification token ID

        if token_ids_1 is None:
            # Return a list of zeros of the length of cls + token_ids_0 + sep
            return len(cls + token_ids_0 + sep) * [0]
        else:
            # Return a list of zeros of the length of cls + token_ids_0 + sep + sep + token_ids_1 + sep
            return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        # Return the size of the vocabulary
        return len(self.sp_model)

    def get_vocab(self):
        # Generate a vocabulary dictionary mapping tokens to IDs
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # Update with additional tokens
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # Tokenize input text into a list of strings (tokens)
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) into an ID using the vocabulary."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an ID (integer) into a token (str) using the vocabulary."""
        return self.sp_model.IdToPiece(index)

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string
    # 将一系列标记（字符串）转换为单个字符串。
    def convert_tokens_to_string(self, tokens):
        # 当前正在处理的子标记列表
        current_sub_tokens = []
        # 输出的字符串
        out_string = ""
        # 上一个标记是否是特殊标记
        prev_is_special = False
        # 遍历每个标记
        for token in tokens:
            # 检查特殊标记是否需要使用 sentencepiece 模型解码
            if token in self.all_special_tokens:
                # 如果上一个标记不是特殊标记，则在 out_string 后添加空格
                if not prev_is_special:
                    out_string += " "
                # 使用 sentencepiece 模型解码 current_sub_tokens，并添加当前特殊标记到 out_string
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                # 清空 current_sub_tokens，准备处理下一个标记序列
                current_sub_tokens = []
            else:
                # 将当前标记添加到 current_sub_tokens 中
                current_sub_tokens.append(token)
                prev_is_special = False
        # 处理剩余的 current_sub_tokens，并添加到 out_string
        out_string += self.sp_model.decode(current_sub_tokens)
        # 返回去除首尾空格的 out_string
        return out_string.strip()

    # 获取对象的状态信息，以便序列化保存
    def __getstate__(self):
        # 复制对象的字典属性
        state = self.__dict__.copy()
        # 将 sp_model 设置为 None，以便在序列化时排除
        state["sp_model"] = None
        return state

    # 设置对象的状态信息，以便反序列化恢复
    def __setstate__(self, d):
        # 使用字典 d 更新对象的属性
        self.__dict__ = d

        # 为了向后兼容
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 根据 sp_model_kwargs 重新创建 sp_model 对象，并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 设置输出的词汇文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇文件路径和输出路径不同，并且当前词汇文件存在，则复制当前词汇文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇文件不存在，则将 sentencepiece 模型序列化内容写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回保存的词汇文件路径的元组
        return (out_vocab_file,)
```