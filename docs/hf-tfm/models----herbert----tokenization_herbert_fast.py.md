# `.\models\herbert\tokenization_herbert_fast.py`

```
# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Allegro.pl, Facebook Inc. and the HuggingFace Inc. team.
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

from typing import List, Optional, Tuple

# 从tokenization_utils_fast模块导入PreTrainedTokenizerFast类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从utils模块导入logging函数
from ...utils import logging
# 从当前目录的tokenization_herbert模块导入HerbertTokenizer类
from .tokenization_herbert import HerbertTokenizer

# 获取logger对象
logger = logging.get_logger(__name__)

# 定义常量，指定预训练模型相关的文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型文件映射，指定预训练模型及其对应的文件下载地址
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/vocab.json"
    },
    "merges_file": {
        "allegro/herbert-base-cased": "https://huggingface.co/allegro/herbert-base-cased/resolve/main/merges.txt"
    },
}

# 定义预训练模型的位置编码嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"allegro/herbert-base-cased": 514}
# 定义预训练模型的初始化配置为空字典
PRETRAINED_INIT_CONFIGURATION = {}

# HerbertTokenizerFast类，继承自PreTrainedTokenizerFast类
class HerbertTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "Fast" BPE tokenizer for HerBERT (backed by HuggingFace's *tokenizers* library).

    Peculiarities:

    - uses BERT's pre-tokenizer: BertPreTokenizer splits tokens on spaces, and also on punctuation. Each occurrence of
      a punctuation character will be treated separately.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the methods. Users should refer to the
    superclass for more information regarding methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
    """

    # 类属性，指定默认的文件名字典
    vocab_files_names = VOCAB_FILES_NAMES
    # 类属性，指定预训练模型文件的映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 类属性，指定预训练模型的初始化配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 类属性，指定预训练模型的最大输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 类属性，指定慢速tokenizer的类
    slow_tokenizer_class = HerbertTokenizer

    # 初始化方法，接受多个参数并调用父类的初始化方法
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sep_token="</s>",
        **kwargs,
    ):
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sep_token=sep_token,
            **kwargs,
        )
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个序列或者一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记。像BERT和HerBERT序列有如下格式：

        - 单个序列: `<s> X </s>`
        - 一对序列: `<s> A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的ID列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的ID列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的输入ID列表。
        """

        # 获取CLS和SEP标记的ID
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        # 如果只有一个序列，则返回CLS + 序列 + SEP
        if token_ids_1 is None:
            return cls + token_ids_0 + sep

        # 如果有一对序列，则返回CLS + 第一个序列 + SEP + 第二个序列 + SEP
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中提取序列ID。当使用分词器的 `prepare_for_model` 方法添加特殊标记时调用此方法。

        Args:
            token_ids_0 (`List[int]`):
                ID列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的ID列表，用于序列对。
            already_has_special_tokens (`bool`, *可选*, 默认为 `False`):
                标记列表是否已经按模型的要求格式化为特殊标记。

        Returns:
            `List[int]`: 一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """
        # 如果已经有特殊标记，则直接调用父类方法获取特殊标记掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 如果只有一个序列，则返回 [1] + [0] * len(token_ids_0) + [1]
        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]

        # 如果有一对序列，则返回 [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        从序列创建令牌类型ID列表，用于区分一对序列中的每个部分。

        Args:
            token_ids_0 (`List[int]`):
                第一个序列的ID列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的ID列表，用于序列对。

        Returns:
            `List[int]`: 一个整数列表，表示每个令牌的类型ID。
        """
        # 初始化类型ID列表
        token_type_ids = []

        # 遍历第一个序列的ID列表，标记为类型1
        for _ in token_ids_0:
            token_type_ids.append(0)

        # 如果有第二个序列，则遍历第二个序列的ID列表，标记为类型2
        if token_ids_1 is not None:
            for _ in token_ids_1:
                token_type_ids.append(1)

        return token_type_ids
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. HerBERT, like
        BERT sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        Args:
            token_ids_0 (`List[int]`):
                List of IDs for the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # Define the separation and classification tokens as lists containing their respective token IDs
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If token_ids_1 is None, return a list of zeros indicating the mask for token_ids_0 only
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # If token_ids_1 is provided, concatenate the masks for token_ids_0 and token_ids_1, with 0s for the first sequence and 1s for the second
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Save the tokenizer model's vocabulary files to the specified directory with the optional filename prefix
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # Return the saved file paths as a tuple
        return tuple(files)
```