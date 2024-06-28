# `.\models\rembert\tokenization_rembert_fast.py`

```
# coding=utf-8
# 声明文件编码格式为UTF-8

# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
# 版权声明，版权属于Google AI、Google Brain和HuggingFace Inc.团队。

# Licensed under the Apache License, Version 2.0 (the "License");
# 以Apache License 2.0版本授权许可，详细信息可查阅License链接

# you may not use this file except in compliance with the License.
# 除非遵守许可证，否则不得使用此文件。

# You may obtain a copy of the License at
# 可在上述链接获取许可证的副本。

#     http://www.apache.org/licenses/LICENSE-2.0
#     许可证的URL地址

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and
# limitations under the License.
# 查阅许可证获取更多具体的权限和限制信息。

""" Tokenization classes for RemBERT model."""
# 注释：RemBERT模型的分词类

import os
# 导入标准库os中的功能
from shutil import copyfile
# 从shutil模块中导入copyfile函数
from typing import List, Optional, Tuple
# 导入typing模块中的List、Optional和Tuple类型

from ...tokenization_utils import AddedToken
# 从上级目录中的tokenization_utils模块导入AddedToken类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从上级目录中的tokenization_utils_fast模块导入PreTrainedTokenizerFast类
from ...utils import is_sentencepiece_available, logging
# 从上级目录中的utils模块导入is_sentencepiece_available和logging功能

if is_sentencepiece_available():
    # 如果SentencePiece可用
    from .tokenization_rembert import RemBertTokenizer
    # 从当前目录中的tokenization_rembert模块导入RemBertTokenizer类
else:
    # 如果SentencePiece不可用
    RemBertTokenizer = None
    # 将RemBertTokenizer设置为None

logger = logging.get_logger(__name__)
# 获取当前模块的logger对象
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.model", "tokenizer_file": "tokenizer.json"}
# 定义词汇文件和分词器文件的名称字典

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/rembert": "https://huggingface.co/google/rembert/resolve/main/sentencepiece.model",
    },
    "tokenizer_file": {
        "google/rembert": "https://huggingface.co/google/rembert/resolve/main/tokenizer.json",
    },
}
# 预训练词汇文件映射字典，包含google/rembert模型的词汇和分词器文件的URL

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/rembert": 256,
}
# 预训练位置嵌入大小字典，包含google/rembert模型的大小为256

SPIECE_UNDERLINE = "▁"
# 定义SentencePiece的特殊字符下划线

class RemBertTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" RemBert tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models). This
    tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods
    """
    # RemBertTokenizerFast类继承自PreTrainedTokenizerFast，构建一个“快速”的RemBert分词器，
    # 使用HuggingFace的tokenizers库支持。基于Unigram模型。此分词器继承自PreTrainedTokenizerFast类，
    # 包含大多数主要方法。用户应参考此超类以获取关于这些方法的更多信息。
    # 定义函数参数和说明
    Args:
        vocab_file (`str`):
            SentencePiece 文件的路径，通常以 *.spm* 扩展名，包含实例化分词器所需的词汇表。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            在分词时是否将输入转换为小写。
        remove_space (`bool`, *optional*, defaults to `True`):
            在分词时是否去除文本中的空格（去除字符串前后多余的空格）。
        keep_accents (`bool`, *optional*, defaults to `False`):
            在分词时是否保留重音符号。
        bos_token (`str`, *optional*, defaults to `"[CLS]"`):
            序列的开始标记，用于预训练。在构建序列时，实际使用的是 `cls_token`。
        eos_token (`str`, *optional*, defaults to `"[SEP]"`):
            序列的结束标记。在构建序列时，实际使用的是 `sep_token`。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记，表示词汇表中不存在的标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，在构建多个序列组成的序列时使用，例如序列分类或问答任务中的文本和问题之间的分隔。
            同时也是构建特殊序列的最后一个标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            填充标记，在将不同长度的序列进行批处理时使用。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，在进行序列分类时使用，是构建特殊序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            掩码标记，在掩码语言建模训练中使用，模型尝试预测的标记。
    
    # 设置特定的常量和映射
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = RemBertTokenizer
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs,
    ):
        # 如果 mask_token 是字符串类型，则创建一个 AddedToken 对象，用于处理左侧去除空格而右侧保留空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的初始化方法，设置各种属性和参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        # 设置对象的属性，用于保存配置参数
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查词汇表文件是否存在，以确定是否可以保存缓慢的分词器
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。RemBERT 序列的格式如下：

        - 单个序列：`[CLS] X [SEP]`
        - 序列对：`[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                将添加特殊标记的 ID 列表
            token_ids_1 (`List[int]`, *optional*, 默认为 `None`):
                第二个序列的 ID 列表（对序列任务）

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        获取特殊标记的掩码，用于标识哪些位置是特殊标记的位置。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的 ID 列表
            token_ids_1 (`List[int]`, *optional*, 默认为 `None`):
                第二个序列的 ID 列表（对序列任务）
            already_has_special_tokens (`bool`, 默认为 `False`):
                是否已经包含了特殊标记，如果是则为 True

        Returns:
            `List[int]`: 表示特殊标记位置的掩码列表。
        """
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. A RemBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*, defaults to `None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """

        # 定义特殊标记
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果没有第二个序列，则返回只有第一个序列部分的 mask (全为0)
        if token_ids_1 is None:
            # 构建第一个序列的 token type IDs，格式为 [0, ..., 0]，长度为 cls + token_ids_0 + sep 的总长度
            return len(cls + token_ids_0 + sep) * [0]

        # 如果有第二个序列
        # 构建第一个序列和第二个序列的 token type IDs
        # 第一个序列部分为 [0, ..., 0]，长度为 cls + token_ids_0 + sep 的总长度
        # 第二个序列部分为 [1, ..., 1]，长度为 token_ids_1 + sep 的总长度
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
    # 定义一个方法用于保存词汇表到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        
        # 构建输出词汇表文件路径，包括可选的文件名前缀和默认的词汇表文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与目标输出路径不同，复制当前词汇表文件到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回包含输出文件路径的元组
        return (out_vocab_file,)
```