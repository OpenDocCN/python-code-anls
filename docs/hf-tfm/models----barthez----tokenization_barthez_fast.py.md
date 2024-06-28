# `.\models\barthez\tokenization_barthez_fast.py`

```
# coding=utf-8
# 版权归 2020 年 Ecole Polytechnique 和 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证版本 2.0 进行许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“原样”分发的，没有任何形式的担保或条件。
# 有关更多信息，请参阅许可证。
""" BARThez 模型的分词类。"""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 检查是否安装了 sentencepiece
if is_sentencepiece_available():
    from .tokenization_barthez import BarthezTokenizer
else:
    BarthezTokenizer = None

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 预训练模型词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "moussaKam/mbarthez": "https://huggingface.co/moussaKam/mbarthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez": "https://huggingface.co/moussaKam/barthez/resolve/main/sentencepiece.bpe.model",
        "moussaKam/barthez-orangesum-title": (
            "https://huggingface.co/moussaKam/barthez-orangesum-title/resolve/main/sentencepiece.bpe.model"
        ),
    },
    "tokenizer_file": {
        "moussaKam/mbarthez": "https://huggingface.co/moussaKam/mbarthez/resolve/main/tokenizer.json",
        "moussaKam/barthez": "https://huggingface.co/moussaKam/barthez/resolve/main/tokenizer.json",
        "moussaKam/barthez-orangesum-title": (
            "https://huggingface.co/moussaKam/barthez-orangesum-title/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练模型位置嵌入的尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "moussaKam/mbarthez": 1024,
    "moussaKam/barthez": 1024,
    "moussaKam/barthez-orangesum-title": 1024,
}

# SentencePiece 使用的分词前缀
SPIECE_UNDERLINE = "▁"

class BarthezTokenizerFast(PreTrainedTokenizerFast):
    """
    从 `CamembertTokenizer` 和 `BartTokenizer` 改编而来。构建一个“快速”的 BARThez 分词器，基于
    [SentencePiece](https://github.com/google/sentencepiece)。

    该分词器继承自 `PreTrainedTokenizerFast`，其中包含大多数主要方法。用户应参考这个超类以获取更多关于这些方法的信息。
    """
    """
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
    """

    # 将文件名映射至文件名常量
    vocab_files_names = VOCAB_FILES_NAMES
    # 将预训练的词汇文件映射至词汇文件映射常量
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 将预训练位置嵌入大小映射至最大模型输入大小常量
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速分词器类别
    slow_tokenizer_class = BarthezTokenizer

    # 初始化方法，接受多个参数，包括词汇文件、tokenizer文件及各种特殊token
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs,
    """
    ):
        # 如果 mask_token 是字符串类型，将其包装为一个带有剥离左侧空格和不剥离右侧空格的 AddedToken 对象；否则保持不变
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的初始化方法，传入必要的参数和关键字参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        # 设置对象的 vocab_file 属性为传入的 vocab_file
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 如果 self.vocab_file 存在且是一个文件，则返回 True；否则返回 False
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        通过添加特殊 token 构建用于序列分类任务的模型输入。BARThez 序列的格式如下：

        - 单个序列: `<s> X </s>`
        - 序列对: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                需要添加特殊 token 的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的 ID 列表（对序列任务时使用）。

        Returns:
            `List[int]`: 包含适当特殊 token 的输入 ID 列表。
        """

        if token_ids_1 is None:
            # 如果没有第二个序列，返回包含 cls_token_id, token_ids_0 和 sep_token_id 的列表
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回包含 cls_token_id, token_ids_0, sep_token_id, sep_token_id, token_ids_1 和 sep_token_id 的列表
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从两个传入的序列创建一个用于序列对分类任务的掩码。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的 ID 列表（对序列任务时使用）。

        Returns:
            `List[int]`: 全为零的列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            # 如果没有第二个序列，返回长度为 cls_token_id, token_ids_0 和 sep 的列表，所有元素为零
            return len(cls + token_ids_0 + sep) * [0]
        # 返回长度为 cls_token_id, token_ids_0, sep, sep, token_ids_1 和 sep 的列表，所有元素为零
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
    # 定义一个方法用于保存词汇表到指定目录下的文件，返回文件路径元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果当前的快速分词器不具备保存慢速分词器所需的信息，则引发数值错误异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，记录错误日志并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构建输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出文件路径不一致，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回输出词汇表文件路径的元组
        return (out_vocab_file,)
```