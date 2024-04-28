# `.\transformers\models\t5\tokenization_t5_fast.py`

```
# 设置文件编码和版权信息
# 版权归 T5 作者和 HuggingFace Inc. 团队所有
# 根据 Apache 2.0 版本授权许可，除非符合许可条件，否则不得使用此文件
# 可以在以下链接获取许可内容：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依照许可条件分发的软件基于“AS IS”基础，
# 没有任何形式上的担保或条件，也没有明示或默示的保证
# 请查阅许可证以获取关于权限和限制的详细信息

""" 用于T5模型的标记化类 """

import os
import re
import warnings
from shutil import copyfile
from typing import List, Optional, Tuple

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 检查是否有可用的sentencepiece模块
if is_sentencepiece_available():
    # 如果可用，则导入T5Tokenizer
    from .tokenization_t5 import T5Tokenizer
else:
    # 否则将T5Tokenizer设为None
    T5Tokenizer = None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名称
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "t5-small": "https://huggingface.co/t5-small/resolve/main/spiece.model",
        "t5-base": "https://huggingface.co/t5-base/resolve/main/spiece.model",
        "t5-large": "https://huggingface.co/t5-large/resolve/main/spiece.model",
        "t5-3b": "https://huggingface.co/t5-3b/resolve/main/spiece.model",
        "t5-11b": "https://huggingface.co/t5-11b/resolve/main/spiece.model",
    },
    "tokenizer_file": {
        "t5-small": "https://huggingface.co/t5-small/resolve/main/tokenizer.json",
        "t5-base": "https://huggingface.co/t5-base/resolve/main/tokenizer.json",
        "t5-large": "https://huggingface.co/t5-large/resolve/main/tokenizer.json",
        "t5-3b": "https://huggingface.co/t5-3b/resolve/main/tokenizer.json",
        "t5-11b": "https://huggingface.co/t5-11b/resolve/main/tokenizer.json",
    },
}

# 预训练位置嵌入大小的映射（将在未来版本中移除）
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "t5-small": 512,
    "t5-base": 512,
    "t5-large": 512,
    "t5-3b": 512,
    "t5-11b": 512,
}


class T5TokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”T5 tokenizer（基于HuggingFace的*tokenizers*库）。基于
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models)。
    
    此tokenizer继承自[`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应该
    参考这个超类以获取关��这些方法的更多信息。
    # 参数说明：
    # vocab_file: SentencePiece文件，包含实例化分词器所需的词汇表
    # eos_token: 结束序列的特殊标记，默认为"</s>"
    # unk_token: 未知标记，不在词汇表中的标记将被设置为此标记
    # pad_token: 用于填充的标记，例如在将不同长度的序列进行批处理时使用
    # extra_ids: 词汇表中用作标记的额外ID数量，默认为100
    # additional_special_tokens: 分词器使用的其他特殊标记
    """

    # 定义变量
    # vocab_files_names: 词汇表文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇表文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 最大模型输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入的名称
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速分词器类别
    slow_tokenizer_class = T5Tokenizer

    # 前缀标记列表
    prefix_tokens: List[int] = []

    # 初始化方法，设置参数值
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        **kwargs,
        # Add extra_ids to the special token list
        # 如果有额外的特殊标记，将其添加到特殊标记列表中
        if additional_special_tokens is not None:
            extra_tokens = [x for x in additional_special_tokens if "<extra_id_" in str(x)]
            # 如果额外的特殊标记中没有包含额外ID，则自动添加额外ID的特殊标记
            if len(extra_tokens) < 1:
                additional_special_tokens += [f"<extra_id_{i}>" for i in range(extra_ids)]
            # 如果提供了额外ID并且额外特殊标记的数量与额外ID数量不一致，则引发异常
            elif extra_ids > 0 and extra_ids != len(extra_tokens):
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids"
                    " tokens"
                )
        else:
            # 如果没有提供额外特殊标记，则自动创建额外ID的特殊标记并赋值给additional_special_tokens
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            additional_special_tokens = extra_tokens

        # 调用父类的初始化方法，并传入相应参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # 将vocab_file和extra_ids保存至实例属性
        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

    # 定义can_save_slow_tokenizer属性的getter，判断vocab_file是否存在来确定是否可以保存慢速分词器
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 定义一个静态方法，用于调整T5模型的最大长度
    @staticmethod
    def _eventually_correct_t5_max_length(pretrained_model_name_or_path, max_model_length, init_max_model_length):
        if pretrained_model_name_or_path in T5TokenizerFast.max_model_input_sizes:
            deprecated_max_model_length = T5TokenizerFast.max_model_input_sizes[pretrained_model_name_or_path]
            # 如果初始化时指定了max_model_length，并且不等于已有的max_model_length，则返回初始化时指定的值
            if init_max_model_length is not None and init_max_model_length != max_model_length:
                return init_max_model_length
            # 如果初始化时未指定max_model_length，则发出警告，指出最大长度将在未来版本中修正
            elif init_max_model_length is None:
                warnings.warn(
                    "This tokenizer was incorrectly instantiated with a model max length of"
                    f" {deprecated_max_model_length} which will be corrected in Transformers v5.\nFor now, this"
                    " behavior is kept to avoid breaking backwards compatibility when padding/encoding with"
                    " `truncation is True`.\n- Be aware that you SHOULD NOT rely on"
                    f" {pretrained_model_name_or_path} automatically truncating your input to"
                    f" {deprecated_max_model_length} when padding/encoding.\n- If you want to encode/pad to sequences"
                    f" longer than {deprecated_max_model_length} you can either instantiate this tokenizer with"
                    " `model_max_length` or pass `max_length` when encoding/padding.\n- To avoid this warning, please"
                    " instantiate this tokenizer with `model_max_length` set to your preferred value.",
                    FutureWarning,
                )

        return max_model_length
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表，则引发值错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 设置输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果原词汇表文件路径和输出词汇表文件路径不同，则复制文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Copy vocab file to {out_vocab_file}")

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 为 token_ids_0 添加结束标记
        token_ids_0 = token_ids_0 + [self.eos_token_id]
        # 如果 token_ids_1 不存在，则返回前缀标记和 token_ids_0
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0
        else:
            # 为 token_ids_1 添加结束标记，返回前缀标记、token_ids_0 和 token_ids_1
            token_ids_1 = token_ids_1 + [self.eos_token_id]
            return self.prefix_tokens + token_ids_0 + token_ids_1

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 创建一个用于序列对分类任务的掩码，T5 不使用标记类型 id，因此返回一个由零组成的列表
        eos = [self.eos_token_id]

        # 如果 token_ids_1 不存在，则返回由零组成的列表（token_ids_0 和 eos 的长度）
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        # 返回由零组成的列表（token_ids_0、eos、token_ids_1 和 eos 的长度）
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def get_sentinel_tokens(self):
        # 返回包含特殊标记的列表
        return list(
            set(filter(lambda x: bool(re.search(r"<extra_id_\d+>", x)) is not None, self.additional_special_tokens))
        )
    # 获取特殊标记的标识符列表
    def get_sentinel_token_ids(self):
        # 获取特殊标记的标识符列表，通过遍历获取的特殊标记，转换成对应的标识符
        return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]
```