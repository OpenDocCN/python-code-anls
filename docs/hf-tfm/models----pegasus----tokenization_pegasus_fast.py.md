# `.\models\pegasus\tokenization_pegasus_fast.py`

```
# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for model PEGASUS."""

# 导入标准库和第三方库
import os
from shutil import copyfile
from typing import List, Optional, Tuple

# 导入所需的tokenization工具和logging
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 如果sentencepiece可用，则导入对应的PEGASUS tokenizer类，否则设为None
if is_sentencepiece_available():
    from .tokenization_pegasus import PegasusTokenizer
else:
    PegasusTokenizer = None

# 获取当前模块的logger
logger = logging.get_logger(__name__)

# 定义一个常量，用于表示词块的前缀
SPIECE_UNDERLINE = "▁"

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"google/pegasus-xsum": "https://huggingface.co/google/pegasus-xsum/resolve/main/spiece.model"},
    "tokenizer_file": {
        "google/pegasus-xsum": "https://huggingface.co/google/pegasus-xsum/resolve/main/tokenizer.json"
    },
}

# 预训练模型的位置编码大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/pegasus-xsum": 512,
}


class PegasusTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" PEGASUS tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        mask_token (`str`, *optional*, defaults to `"<mask_2>"`):
            The token used for masking single token values. This is the token used when training this model with masked
            language modeling (MLM). This is the token that the PEGASUS encoder will try to predict during pretraining.
            It corresponds to *[MASK2]* in [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive
            Summarization](https://arxiv.org/pdf/1912.08777.pdf).
        mask_token_sent (`str`, *optional*, defaults to `"<mask_1>"`):
            The token used for masking whole target sentences. This is the token used when training this model with gap
            sentences generation (GSG). This is the sentence that the PEGASUS decoder will try to predict during
            pretraining. It corresponds to *[MASK1]* in [PEGASUS: Pre-training with Extracted Gap-sentences for
            Abstractive Summarization](https://arxiv.org/pdf/1912.08777.pdf).
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer. If no additional_special_tokens are provided <mask_2> and
            <unk_2, ..., unk_102> are used as additional special tokens corresponding to the [original PEGASUS
            tokenizer](https://github.com/google-research/pegasus/blob/939830367bcf411193d2b5eca2f2f90f3f9260ca/pegasus/ops/pretrain_parsing_ops.cc#L66)
            that uses the tokens 2 - 104 only for pretraining
    ```

    vocab_files_names = VOCAB_FILES_NAMES
    # 获取存储了词汇文件名的常量列表

    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 获取存储了预训练模型词汇文件映射的常量字典

    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 获取存储了预训练模型输入最大尺寸的常量字典

    slow_tokenizer_class = PegasusTokenizer
    # 获取 PegasusTokenizer 类，用于慢速模式的分词器

    model_input_names = ["input_ids", "attention_mask"]
    # 定义了模型输入的名称列表，包括 input_ids 和 attention_mask
    # 初始化函数，用于创建一个新的对象
    def __init__(
        self,
        vocab_file=None,  # 词汇表文件路径，默认为None
        tokenizer_file=None,  # 分词器文件路径，默认为None
        pad_token="<pad>",  # 填充标记，默认为"<pad>"
        eos_token="</s>",  # 结束标记，默认为"</s>"
        unk_token="<unk>",  # 未知标记，默认为"<unk>"
        mask_token="<mask_2>",  # 掩码标记，默认为"<mask_2>"
        mask_token_sent="<mask_1>",  # 用于句子级别掩码的标记，默认为"<mask_1>"
        additional_special_tokens=None,  # 额外的特殊标记列表，默认为None
        offset=103,  # 前期训练中使用的偏移量，默认为103，索引2到104用于预训练
        **kwargs,  # 其他关键字参数
    ):
        self.offset = offset  # 初始化对象的偏移量属性为给定的offset值

        if additional_special_tokens is not None:
            if not isinstance(additional_special_tokens, list):
                # 如果额外特殊标记不是列表类型，则引发类型错误异常
                raise TypeError(
                    f"additional_special_tokens should be of type {type(list)}, but is"
                    f" {type(additional_special_tokens)}"
                )

            # 如果mask_token_sent不在additional_special_tokens中且不为None，则将其添加到额外特殊标记列表中
            additional_special_tokens_extended = (
                ([mask_token_sent] + additional_special_tokens)
                if mask_token_sent not in additional_special_tokens and mask_token_sent is not None
                else additional_special_tokens
            )

            # 填充额外特殊标记列表直到达到offset - 1的长度，并以"<unk_x>"命名
            additional_special_tokens_extended += [
                f"<unk_{i}>" for i in range(len(additional_special_tokens_extended), self.offset - 1)
            ]

            # 如果额外特殊标记列表中存在重复的标记，则引发值错误异常
            if len(set(additional_special_tokens_extended)) != len(additional_special_tokens_extended):
                raise ValueError(
                    "Please make sure that the provided additional_special_tokens do not contain an incorrectly"
                    f" shifted list of <unk_x> tokens. Found {additional_special_tokens_extended}."
                )
            additional_special_tokens = additional_special_tokens_extended
        else:
            # 如果额外特殊标记为None，则创建一个新的额外特殊标记列表，包含mask_token_sent和"<unk_x>"标记
            additional_special_tokens = [mask_token_sent] if mask_token_sent is not None else []
            additional_special_tokens += [f"<unk_{i}>" for i in range(2, self.offset)]

        # 如果from_slow参数未提供，则从kwargs中获取或初始化为None
        from_slow = kwargs.pop("from_slow", None)
        # 如果pad_token、eos_token、unk_token中有一个与默认值不同，则设置from_slow为True
        from_slow = from_slow or str(pad_token) != "<pad>" or str(eos_token) != "</s>" or str(unk_token) != "<unk>"

        # 从kwargs中移除added_tokens_decoder键的值
        kwargs.pop("added_tokens_decoder", {})

        # 调用父类的初始化方法，传递所需的参数和关键字参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            mask_token_sent=mask_token_sent,
            offset=offset,
            additional_special_tokens=additional_special_tokens,
            from_slow=from_slow,
            **kwargs,
        )
        self.vocab_file = vocab_file  # 设置对象的词汇表文件属性为给定的vocab_file路径

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查对象是否能够保存慢速分词器，前提是vocab_file文件存在
        return os.path.isfile(self.vocab_file) if self.vocab_file else False
    def _special_token_mask(self, seq):
        # 将所有特殊标记的 ID 放入集合中，并移除未知标记的 ID
        all_special_ids = set(self.all_special_ids)  # 一次性调用，而不是在列表推导式中调用
        all_special_ids.remove(self.unk_token_id)  # <unk> 只有在某些情况下才是特殊的

        # 检查特殊标记的数量是否正确
        if all_special_ids != set(range(len(self.additional_special_tokens) + 3)):
            raise ValueError(
                "There should be 3 special tokens: mask_token, pad_token, and eos_token +"
                f" {len(self.additional_special_tokens)} additional_special_tokens, but got {all_special_ids}"
            )

        # 返回序列中每个元素是否为特殊标记的掩码列表
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        获取特殊标记掩码列表，如果 token 是 [eos] 或 [pad] 则为 [1]，否则为 [0]。
        """
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """
        根据输入序列构建模型输入，末尾添加 eos 标记，不添加 bos 标记到开头。

        - 单个序列: `X </s>`
        - 序列对: `A B </s>`（不是预期的用法）

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表
            token_ids_1 (`List[int]`, *可选*):
                第二个序列 ID 列表（如果是序列对）

        Returns:
            `List[int]`: 带有适当特殊标记的输入 ID 列表。
        """
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # 虽然不期望处理序列对，但为了 API 一致性保留了对序列对的逻辑
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        保存词汇表到指定目录下，如果无法保存，则引发异常。

        Args:
            save_directory (str): 保存词汇表的目录路径
            filename_prefix (str, *可选*): 文件名前缀

        Returns:
            Tuple[str]: 保存的词汇表文件路径
        """
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 拼接输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不同，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
```