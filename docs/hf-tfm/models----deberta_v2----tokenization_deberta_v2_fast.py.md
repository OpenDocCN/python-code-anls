# `.\models\deberta_v2\tokenization_deberta_v2_fast.py`

```py
# 设定文件编码为 UTF-8
# 版权声明部分
# 根据 Apache 许可证，使用此文件需要遵守相应协议
# 可以在下面链接查看详细协议内容
# 如果不遵守协议，则不能使用本文件
# 本文件遵从 "AS IS" 的分发方式，不提供任何形式的担保，包括但不限于任何明示担保和隐含担保
# 请在协议范围内使用本文件
# 导入必要的库和模块
import os
from shutil import copyfile
from typing import Optional, Tuple
# 导入自定义的工具函数和日志模块
from ...file_utils import is_sentencepiece_available
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
# 判断是否可用 sentencepiece，然后导入 DebertaV2Tokenizer 或为空
if is_sentencepiece_available():
    from .tokenization_deberta_v2 import DebertaV2Tokenizer
else:
    DebertaV2Tokenizer = None
# 获取日志记录器
logger = logging.get_logger(__name__)
# 定义词汇表文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "spm.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/spm.model",
        "microsoft/deberta-v2-xlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/spm.model"
        ),
        "microsoft/deberta-v2-xxlarge-mnli": (
            "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/spm.model"
        ),
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/deberta-v2-xlarge": 512,
    "microsoft/deberta-v2-xxlarge": 512,
    "microsoft/deberta-v2-xlarge-mnli": 512,
    "microsoft/deberta-v2-xxlarge-mnli": 512,
}

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/deberta-v2-xlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge": {"do_lower_case": False},
    "microsoft/deberta-v2-xlarge-mnli": {"do_lower_case": False},
    "microsoft/deberta-v2-xxlarge-mnli": {"do_lower_case": False},
}
# 定义 DebertaV2TokenizerFast 类，继承 PreTrainedTokenizerFast 类
# 用于构建 DeBERTa-v2 快速分词器，基于 SentencePiece
class DebertaV2TokenizerFast(PreTrainedTokenizerFast):
    r"""
    Constructs a DeBERTa-v2 fast tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    """
    # 设置词汇表文件名字典
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = DebertaV2Tokenizer
    # 初始化函数，设置参数并调用父类的初始化函数
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=False,
        split_by_punct=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ) -> None:
        # 调用父类的初始化函数，传入参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            split_by_punct=split_by_punct,
            **kwargs,
        )

        # 设置实例属性
        self.do_lower_case = do_lower_case
        self.split_by_punct = split_by_punct
        self.vocab_file = vocab_file

    # 返回是否可以保存慢速分词器
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 根据给定的token_ids构建特殊token的输入序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
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

        # 如果只有一个输入序列，则在首尾加上特殊token并返回
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # 如果存在两个输入序列，需要在首尾加上特殊token并返回
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        从没有添加特殊标记的令牌列表中检索序列 ID。当使用 tokenizer 的 `prepare_for_model` 或 `encode_plus` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                标记列表是否已经格式化为模型的特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。DeBERTa 序列对掩码的格式如下：

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | 第一个序列        | 第二个序列      |
        ```py

        如果 `token_ids_1` 为 `None`，则此方法仅返回掩码的第一部分（0）。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                序列对的可选第二个 ID 列表。

        Returns:
            `List[int]`: 根据给定序列返回 [token type IDs](../glossary#token-type-ids) 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查是否可以保存慢速分词器的词汇表
        if not self.can_save_slow_tokenizer:
            # 如果不能保存，抛出数值错误
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            # 如果目录不存在，记录错误信息
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return  # 返回空值

        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 检查输出词汇表文件路径是否与当前词汇表文件路径相同
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            # 如果不同，复制当前词汇表文件到输出词汇表文件路径
            copyfile(self.vocab_file, out_vocab_file)

        # 返回输出词汇表文件路径的元组
        return (out_vocab_file,)
```