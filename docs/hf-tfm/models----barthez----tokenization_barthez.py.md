# `.\transformers\models\barthez\tokenization_barthez.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 Ecole Polytechnique 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" BARThez 模型的分词类。"""

# 导入所需的库
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

# 导入 HuggingFace 库中的相关模块
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "moussaKam/mbarthez": "https://huggingface.co/moussaKam/mbarthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez": "https://huggingface.co/moussaKam/barthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez-orangesum-title": (
            "https://huggingface.co/moussaKam/barthez-orangesum-title/resolve/main/sentencepiece.bpe.model"
        ),
    },
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "moussaKam/mbarthez": 1024,
    "moussaKam/barthez": 1024,
    "moussaKam/barthez-orangesum-title": 1024,
}

# SentencePiece 分词符号
SPIECE_UNDERLINE = "▁"

# 定义 BARThezTokenizer 类
# 继承自 PreTrainedTokenizer 类
class BarthezTokenizer(PreTrainedTokenizer):
    """
    从 CamembertTokenizer 和 BartTokenizer 改编而来。构建一个 BARThez 分词器。基于 SentencePiece。

    该分词器继承自 PreTrainedTokenizer，其中包含大多数主要方法。用户应参考该超类以获取有关这些方法的更多信息。

    属性:
        sp_model (`SentencePieceProcessor`):
            用于每次转换（字符串、标记和 ID）的 SentencePiece 处理器。
    """

    # 定义词汇文件名和预训练词汇文件映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 定义一个特殊的 mask token，类似于普通单词，即包括它前面的空格。默认情况下将 normalized=False，这样做
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token

        # 如果 sp_model_kwargs 为 None，则设置为空字典，否则使用传入的 sp_model_kwargs
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置词汇文件路径
        self.vocab_file = vocab_file
        # 使用 sp_model_kwargs 创建 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件到 sp_model
        self.sp_model.Load(str(vocab_file))
        # 调用父类的初始化方法，设置各种特殊 token
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
        从一个序列或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊 token。一个 BARThez 序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 一对序列: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊 token 的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊 token 的 [输入 ID](../glossary#input-ids) 列表。
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

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
        # 检查是否已经添加特殊标记
        if already_has_special_tokens:
            # 如果已经添加特殊标记，则调用父类的方法直接返回特殊标记掩码
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果未添加特殊标记
        if token_ids_1 is None:
            # 如果没有第二个句子的标记，则在首尾添加特殊标记
            return [1] + ([0] * len(token_ids_0)) + [1]
        # 如果有第二个句子的标记，则在首尾和两个句子之间添加特殊标记
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
        # 创建用于序列对分类任务的标记类型 ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            # 如果只有一个句子的标记，则返回全零序列
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有两个句子的标记，则返回全零序列
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        # 返回词汇表大小
        return len(self.sp_model)

    def get_vocab(self):
        # 获取词汇表
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        # 使用句子处理模型对文本进行标记化
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 将标记转换为 ID
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 将 ID 转换为标记
        return self.sp_model.IdToPiece(index)

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.convert_tokens_to_string
    # 将一系列标记（字符串）转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        # 初始化变量
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        # 遍历标记序列
        for token in tokens:
            # 确保特殊标记不使用 sentencepiece 模型解码
            if token in self.all_special_tokens:
                # 处理特殊标记
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()

    # 获取对象的状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 设置对象的状态
    def __setstate__(self, d):
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 加载 sentencepiece 模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 设置输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 复制词汇表文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果词汇表文件不存在，则写入内容
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)
```