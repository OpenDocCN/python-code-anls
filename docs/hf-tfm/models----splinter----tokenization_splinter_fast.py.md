# `.\models\splinter\tokenization_splinter_fast.py`

```
# coding=utf-8
# 上面的代码指定了文件编码格式为 UTF-8，确保能够正确解析包含非 ASCII 字符的内容
# Copyright 2021 Tel AViv University, AllenAI and The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 在 Apache License 2.0 下授权使用本文件
# you may not use this file except in compliance with the License.
# 除非遵循 License，否则不得使用此文件
# You may obtain a copy of the License at
# 获取 License 的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#     可以在上面的链接中找到详细的 License 内容
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据适用法律或书面同意，软件按"原样"分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看 License 获取详细的授权和限制条款
"""Fast Tokenization classes for Splinter."""
# 用于 Splinter 的快速分词类

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_splinter import SplinterTokenizer


logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "tau/splinter-base": "https://huggingface.co/tau/splinter-base/resolve/main/vocab.txt",
        "tau/splinter-base-qass": "https://huggingface.co/tau/splinter-base-qass/resolve/main/vocab.txt",
        "tau/splinter-large": "https://huggingface.co/tau/splinter-large/resolve/main/vocab.txt",
        "tau/splinter-large-qass": "https://huggingface.co/tau/splinter-large-qass/resolve/main/vocab.txt",
    }
}

# 预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "tau/splinter-base": 512,
    "tau/splinter-base-qass": 512,
    "tau/splinter-large": 512,
    "tau/splinter-large-qass": 512,
}

# 预训练模型的初始化配置映射
PRETRAINED_INIT_CONFIGURATION = {
    "tau/splinter-base": {"do_lower_case": False},
    "tau/splinter-base-qass": {"do_lower_case": False},
    "tau/splinter-large": {"do_lower_case": False},
    "tau/splinter-large-qass": {"do_lower_case": False},
}


class SplinterTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" Splinter tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    构建一个快速的 Splinter 分词器，基于 HuggingFace 的 tokenizers 库，基于 WordPiece。

    This class inherits from PreTrainedTokenizerFast, which includes most of the primary methods. Users should refer to
    the superclass for more information on those methods.
    此类继承自 PreTrainedTokenizerFast，该类包含大多数主要方法。用户应参考超类以获取有关这些方法的更多信息。
    ```
    # 定义函数参数和默认值的说明
    Args:
        vocab_file (`str`):
            Vocabulary 文件的路径。
        do_lower_case (`bool`, *optional*, defaults to `True`):
            是否在标记化时将输入转换为小写。
        unk_token (`str`, *optional*, defaults to `"[UNK]"`):
            未知标记。如果标记不在词汇表中，则无法转换为 ID，并将其设置为此标记。
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            分隔符标记，用于从多个序列构建一个序列，例如用于序列分类或问答任务中的问题与文本的分隔。
        pad_token (`str`, *optional*, defaults to `"[PAD]"`):
            填充标记，用于将不同长度的序列进行批处理时进行填充。
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            分类器标记，在序列分类任务中作为序列的第一个标记。
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            掩码标记，用于掩码语言建模任务中模型尝试预测的标记。
        question_token (`str`, *optional*, defaults to `"[QUESTION]"`):
            构建问题表示时使用的标记。
        clean_text (`bool`, *optional*, defaults to `True`):
            是否在标记化前清理文本，例如删除控制字符并替换所有空格。
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            是否标记化中文字符，对于日文可能需要禁用此选项。
        strip_accents (`bool`, *optional*):
            是否去除所有重音符号。如果未指定，则将根据 `lowercase` 的值来确定（与原始的 BERT 行为一致）。
        wordpieces_prefix (`str`, *optional*, defaults to `"##"`):
            子词的前缀。
    """
    
    # 导入预定义的常量和类
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = SplinterTokenizer
    # 初始化方法，用于实例化对象时设置各种参数和调用父类的初始化方法
    def __init__(
        self,
        vocab_file=None,  # 词汇文件路径，默认为None
        tokenizer_file=None,  # 分词器文件路径，默认为None
        do_lower_case=True,  # 是否将输入文本转为小写，默认为True
        unk_token="[UNK]",  # 未知token的表示，默认为"[UNK]"
        sep_token="[SEP]",  # 分隔token的表示，默认为"[SEP]"
        pad_token="[PAD]",  # 填充token的表示，默认为"[PAD]"
        cls_token="[CLS]",  # 类别token的表示，默认为"[CLS]"
        mask_token="[MASK]",  # 掩码token的表示，默认为"[MASK]"
        question_token="[QUESTION]",  # 问题token的表示，默认为"[QUESTION]"
        tokenize_chinese_chars=True,  # 是否分词中文字符，默认为True
        strip_accents=None,  # 是否去除重音符号，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法，设置各种参数
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            additional_special_tokens=(question_token,),  # 添加额外的特殊token，这里是问题token
            **kwargs,
        )

        # 获取前处理器的状态信息，并根据初始化参数更新其设置
        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))  # 获取前处理器类
            pre_tok_state["lowercase"] = do_lower_case  # 更新小写设置
            pre_tok_state["strip_accents"] = strip_accents  # 更新去重音符设置
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)  # 使用更新后的设置重新实例化前处理器

        self.do_lower_case = do_lower_case  # 保存是否转换为小写的设置

    @property
    def question_token_id(self):
        """
        `Optional[int]`: Id of the question token in the vocabulary, used to condition the answer on a question
        representation.
        """
        return self.convert_tokens_to_ids(self.question_token)  # 返回问题token在词汇表中的id

    # 构建带有特殊token的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a pair of sequences for question answering tasks by concatenating and adding special
        tokens. A Splinter sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences for question answering: `[CLS] question_tokens [QUESTION] . [SEP] context_tokens [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                The question token IDs if pad_on_right, else context tokens IDs
            token_ids_1 (`List[int]`, *optional*):
                The context token IDs if pad_on_right, else question token IDs

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """
        if token_ids_1 is None:
            # Return single sequence format: `[CLS] X [SEP]`
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if self.padding_side == "right":
            # Return question-then-context format: `[CLS] question_tokens [QUESTION] . [SEP] context_tokens [SEP]`
            return cls + token_ids_0 + question_suffix + sep + token_ids_1 + sep
        else:
            # Return context-then-question format: `[CLS] context_tokens [SEP] question_tokens [QUESTION] . [SEP]`
            return cls + token_ids_0 + sep + token_ids_1 + question_suffix + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create the token type IDs corresponding to the sequences passed. What are token type
        IDs? See glossary.

        Should be overridden in a subclass if the model has a special way of building those.

        Args:
            token_ids_0 (`List[int]`): The first tokenized sequence.
            token_ids_1 (`List[int]`, *optional*): The second tokenized sequence.

        Returns:
            `List[int]`: The token type ids.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        question_suffix = [self.question_token_id] + [self.convert_tokens_to_ids(".")]
        if token_ids_1 is None:
            # Return token type IDs for single sequence
            return len(cls + token_ids_0 + sep) * [0]

        if self.padding_side == "right":
            # Return token type IDs for question-then-context format
            return len(cls + token_ids_0 + question_suffix + sep) * [0] + len(token_ids_1 + sep) * [1]
        else:
            # Return token type IDs for context-then-question format
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + question_suffix + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer's model vocabulary to a directory.

        Args:
            save_directory (str): The directory where the vocabulary files will be saved.
            filename_prefix (str, *optional*): Optional prefix for the saved vocabulary files.

        Returns:
            `Tuple[str]`: Tuple of filenames saved.
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```