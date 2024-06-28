# `.\models\t5\tokenization_t5_fast.py`

```
# coding=utf-8
# 版权 2018 年 T5 作者和 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”分发的，没有任何形式的明示或暗示担保。
# 请查阅许可证了解特定的语言权限和限制。
""" T5 模型的分词类 """

import os                   # 导入操作系统功能模块
import re                   # 导入正则表达式模块
import warnings             # 导入警告模块
from shutil import copyfile # 导入复制文件功能
from typing import List, Optional, Tuple  # 导入类型提示相关的类

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 从快速分词工具中导入预训练分词器
from ...utils import is_sentencepiece_available, logging  # 从工具包中导入判断是否有句子片段可用的函数和日志记录功能

if is_sentencepiece_available():  # 如果句子片段可用
    from .tokenization_t5 import T5Tokenizer  # 从 T5 分词文件中导入 T5Tokenizer 类
else:
    T5Tokenizer = None  # 否则将 T5Tokenizer 设为 None

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}  # 词汇文件名称映射字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {  # 预训练词汇文件映射
        "google-t5/t5-small": "https://huggingface.co/google-t5/t5-small/resolve/main/spiece.model",
        "google-t5/t5-base": "https://huggingface.co/google-t5/t5-base/resolve/main/spiece.model",
        "google-t5/t5-large": "https://huggingface.co/google-t5/t5-large/resolve/main/spiece.model",
        "google-t5/t5-3b": "https://huggingface.co/google-t5/t5-3b/resolve/main/spiece.model",
        "google-t5/t5-11b": "https://huggingface.co/google-t5/t5-11b/resolve/main/spiece.model",
    },
    "tokenizer_file": {  # 预训练分词器文件映射
        "google-t5/t5-small": "https://huggingface.co/google-t5/t5-small/resolve/main/tokenizer.json",
        "google-t5/t5-base": "https://huggingface.co/google-t5/t5-base/resolve/main/tokenizer.json",
        "google-t5/t5-large": "https://huggingface.co/google-t5/t5-large/resolve/main/tokenizer.json",
        "google-t5/t5-3b": "https://huggingface.co/google-t5/t5-3b/resolve/main/tokenizer.json",
        "google-t5/t5-11b": "https://huggingface.co/google-t5/t5-11b/resolve/main/tokenizer.json",
    },
}

# TODO(PVP) - this should be removed in Transformers v5
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google-t5/t5-small": 512,   # 预训练位置嵌入大小映射，指定了每个 T5 模型的默认嵌入大小为 512
    "google-t5/t5-base": 512,
    "google-t5/t5-large": 512,
    "google-t5/t5-3b": 512,
    "google-t5/t5-11b": 512,
}


class T5TokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”T5分词器（基于HuggingFace的*tokenizers*库）。基于
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models)。

    这个分词器继承自[`PreTrainedTokenizerFast`]，该类包含大部分主要方法。用户应
    参考超类以获取更多关于这些方法的信息。
    """
    # 定义了一些常量和类变量，这些变量用于配置和初始化Tokenizer类的实例
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = T5Tokenizer

    # 前缀特殊标记的空列表
    prefix_tokens: List[int] = []

    # Tokenizer类的构造函数，用于实例化一个新的Tokenizer对象
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        add_prefix_space=None,
        **kwargs,
        # 如果 additional_special_tokens 参数不为空，则从中提取所有带有 "<extra_id_" 标记的额外特殊标记
        if additional_special_tokens is not None:
            extra_tokens = [x for x in additional_special_tokens if "<extra_id_" in str(x)]
            # 如果没有找到带有 "<extra_id_" 标记的额外特殊标记，且需要生成额外的标记，则添加相应数量的 "<extra_id_>" 标记
            if len(extra_tokens) < 1:
                additional_special_tokens += [f"<extra_id_{i}>" for i in range(extra_ids)]
            # 如果 extra_ids 大于 0 并且额外特殊标记的数量不等于 extra_ids，则抛出 ValueError 异常
            elif extra_ids > 0 and extra_ids != len(extra_tokens):
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids"
                    " tokens"
                )
        else:
            # 如果 additional_special_tokens 参数为空，则创建一个包含相应数量 "<extra_id_>" 标记的列表
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            # 将生成的额外特殊标记列表赋值给 additional_special_tokens 参数
            additional_special_tokens = extra_tokens

        # 如果 add_prefix_space 参数不为空，则发出一次警告日志
        if add_prefix_space is not None:
            logger.warning_once(
                "You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers"
            )
            # 将 from_slow 参数设置为 True，以便在初始化时使用
            kwargs["from_slow"] = True

        # 调用父类的初始化方法，传入必要的参数和关键字参数
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

        # 将 vocab_file 和 extra_ids 属性保存在当前对象中
        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
    def _eventually_correct_t5_max_length(pretrained_model_name_or_path, max_model_length, init_max_model_length):
        # 检查预训练模型名称或路径是否在T5TokenizerFast.max_model_input_sizes中
        if pretrained_model_name_or_path in T5TokenizerFast.max_model_input_sizes:
            # 获取过时的最大模型长度
            deprecated_max_model_length = T5TokenizerFast.max_model_input_sizes[pretrained_model_name_or_path]
            # 如果init_max_model_length不为空且与max_model_length不相等，则返回init_max_model_length
            if init_max_model_length is not None and init_max_model_length != max_model_length:
                return init_max_model_length
            # 如果init_max_model_length为空
            elif init_max_model_length is None:
                # 发出警告，指出实例化时的错误行为，并提示将在Transformers v5中更正
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

        # 返回当前的max_model_length
        return max_model_length

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果不能保存慢速tokenizer的词汇表，则引发错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不是一个有效的目录，记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 确定输出的词汇文件路径，并复制词汇文件
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Copy vocab file to {out_vocab_file}")

        # 返回输出的词汇文件路径
        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        # 该方法用于构建包含特殊令牌的输入序列
    def build_inputs_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens added.
        """
        # Add end-of-sequence token to the first sequence
        token_ids_0 = token_ids_0 + [self.eos_token_id]
        
        if token_ids_1 is None:
            # Return the prefix tokens followed by token_ids_0
            return self.prefix_tokens + token_ids_0
        else:
            # Add end-of-sequence token to the second sequence
            token_ids_1 = token_ids_1 + [self.eos_token_id]
            # Return the prefix tokens followed by concatenated token_ids_0 and token_ids_1
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
            `List[int]`: List of zeros as T5 does not use token type ids.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            # Return a list of zeros corresponding to the length of token_ids_0 + eos
            return len(token_ids_0 + eos) * [0]
        else:
            # Return a list of zeros corresponding to the length of token_ids_0 + eos + token_ids_1 + eos
            return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def get_sentinel_tokens(self):
        """
        Get sentinel tokens from the additional special tokens list based on a regex pattern matching.

        Returns:
            List[str]: List of sentinel tokens.
        """
        return list(
            set(filter(lambda x: bool(re.search(r"<extra_id_\d+>", x)) is not None, self.additional_special_tokens))
        )

    def get_sentinel_token_ids(self):
        """
        Convert sentinel tokens to their corresponding token IDs.

        Returns:
            List[int]: List of token IDs of sentinel tokens.
        """
        return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]
```