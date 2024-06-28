# `.\models\bert\tokenization_bert_fast.py`

```py
# coding=utf-8
# 上面是指定脚本的编码格式为 UTF-8

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# 版权声明，指明了代码的版权归属

#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证，可以自由使用本代码
# you may not use this file except in compliance with the License.
# 除非遵循许可证规定，否则不能使用该文件

# You may obtain a copy of the License at
# 可以在以下链接获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何形式的保证或条件。
# 请参阅许可证以获取详细的权限和限制信息。

"""Fast Tokenization classes for Bert."""
# 用于 Bert 的快速标记化类

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_bert import BertTokenizer

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称映射
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型所需的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    {
        "vocab_file": {
            "google-bert/bert-base-uncased": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt",
            "google-bert/bert-large-uncased": "https://huggingface.co/google-bert/bert-large-uncased/resolve/main/vocab.txt",
            "google-bert/bert-base-cased": "https://huggingface.co/google-bert/bert-base-cased/resolve/main/vocab.txt",
            "google-bert/bert-large-cased": "https://huggingface.co/google-bert/bert-large-cased/resolve/main/vocab.txt",
            "google-bert/bert-base-multilingual-uncased": (
                "https://huggingface.co/google-bert/bert-base-multilingual-uncased/resolve/main/vocab.txt"
            ),
            "google-bert/bert-base-multilingual-cased": "https://huggingface.co/google-bert/bert-base-multilingual-cased/resolve/main/vocab.txt",
            "google-bert/bert-base-chinese": "https://huggingface.co/google-bert/bert-base-chinese/resolve/main/vocab.txt",
            "google-bert/bert-base-german-cased": "https://huggingface.co/google-bert/bert-base-german-cased/resolve/main/vocab.txt",
            "google-bert/bert-large-uncased-whole-word-masking": (
                "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking/resolve/main/vocab.txt"
            ),
            "google-bert/bert-large-cased-whole-word-masking": (
                "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking/resolve/main/vocab.txt"
            ),
            "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": (
                "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
            ),
            "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": (
                "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/vocab.txt"
            ),
            "google-bert/bert-base-cased-finetuned-mrpc": (
                "https://huggingface.co/google-bert/bert-base-cased-finetuned-mrpc/resolve/main/vocab.txt"
            ),
            "google-bert/bert-base-german-dbmdz-cased": "https://huggingface.co/google-bert/bert-base-german-dbmdz-cased/resolve/main/vocab.txt",
            "google-bert/bert-base-german-dbmdz-uncased": (
                "https://huggingface.co/google-bert/bert-base-german-dbmdz-uncased/resolve/main/vocab.txt"
            ),
            "TurkuNLP/bert-base-finnish-cased-v1": (
                "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/vocab.txt"
            ),
            "TurkuNLP/bert-base-finnish-uncased-v1": (
                "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/vocab.txt"
            ),
            "wietsedv/bert-base-dutch-cased": (
                "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/vocab.txt"
            )
        }
    }
    
    
    
    # 注释：
    "vocab_file" 字典包含了多个键值对，每个键代表一个预训练的BERT模型，对应的值是该模型的词汇表（vocab.txt）的下载链接。
    这些链接可以通过Hugging Face模型中心获取，用于获取BERT模型的词汇表数据。
    {
        // Tokenizer文件的映射，键是模型名称，值是对应的Tokenizer.json文件的URL
        "tokenizer_file": {
            "google-bert/bert-base-uncased": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/tokenizer.json",
            "google-bert/bert-large-uncased": "https://huggingface.co/google-bert/bert-large-uncased/resolve/main/tokenizer.json",
            "google-bert/bert-base-cased": "https://huggingface.co/google-bert/bert-base-cased/resolve/main/tokenizer.json",
            "google-bert/bert-large-cased": "https://huggingface.co/google-bert/bert-large-cased/resolve/main/tokenizer.json",
            "google-bert/bert-base-multilingual-uncased": (
                "https://huggingface.co/google-bert/bert-base-multilingual-uncased/resolve/main/tokenizer.json"
            ),
            "google-bert/bert-base-multilingual-cased": (
                "https://huggingface.co/google-bert/bert-base-multilingual-cased/resolve/main/tokenizer.json"
            ),
            "google-bert/bert-base-chinese": "https://huggingface.co/google-bert/bert-base-chinese/resolve/main/tokenizer.json",
            "google-bert/bert-base-german-cased": "https://huggingface.co/google-bert/bert-base-german-cased/resolve/main/tokenizer.json",
            "google-bert/bert-large-uncased-whole-word-masking": (
                "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking/resolve/main/tokenizer.json"
            ),
            "google-bert/bert-large-cased-whole-word-masking": (
                "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking/resolve/main/tokenizer.json"
            ),
            "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": (
                "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"
            ),
            "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": (
                "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/tokenizer.json"
            ),
            "google-bert/bert-base-cased-finetuned-mrpc": (
                "https://huggingface.co/google-bert/bert-base-cased-finetuned-mrpc/resolve/main/tokenizer.json"
            ),
            "google-bert/bert-base-german-dbmdz-cased": (
                "https://huggingface.co/google-bert/bert-base-german-dbmdz-cased/resolve/main/tokenizer.json"
            ),
            "google-bert/bert-base-german-dbmdz-uncased": (
                "https://huggingface.co/google-bert/bert-base-german-dbmdz-uncased/resolve/main/tokenizer.json"
            ),
            "TurkuNLP/bert-base-finnish-cased-v1": (
                "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/tokenizer.json"
            ),
            "TurkuNLP/bert-base-finnish-uncased-v1": (
                "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/tokenizer.json"
            ),
            "wietsedv/bert-base-dutch-cased": (
                "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/tokenizer.json"
            )
        }
    }
}

# 首先定义了一个空的类 BertTokenizerFast，该类继承自 PreTrainedTokenizerFast
class BertTokenizerFast(PreTrainedTokenizerFast):
    # docstring: 构建一个“快速”BERT tokenizer，使用 HuggingFace 的 tokenizers 库支持，基于 WordPiece
    r"""
    Construct a "fast" BERT tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    # 定义类 BertTokenizer，用于处理 BERT 模型的分词器功能
    class BertTokenizer:
    
        # 类的初始化方法，用于设置分词器的各种参数和选项
        def __init__(
            self,
            vocab_file=None,  # 词汇表文件路径，用于加载模型的词汇表
            tokenizer_file=None,  # 分词器文件路径，可选，用于加载预训练的分词器模型
            do_lower_case=True,  # 是否将输入转换为小写
            unk_token="[UNK]",  # 未知标记，当词汇表中不存在某个词时使用
            sep_token="[SEP]",  # 分隔符标记，在构建多序列时使用
            pad_token="[PAD]",  # 填充标记，在对不同长度的序列进行批处理时使用
            cls_token="[CLS]",  # 分类器标记，用于序列分类任务中
            mask_token="[MASK]",  # 掩码标记，用于掩码语言模型任务中
            tokenize_chinese_chars=True,  # 是否分词中文字符
            strip_accents=None,  # 是否去除所有重音符号
            **kwargs,  # 其他参数，用于兼容未来可能添加的参数
    ):
        # 调用父类的构造函数，初始化模型的tokenizer
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
            **kwargs,
        )

        # 获取当前tokenizer的规范化器状态并转换为JSON格式
        normalizer_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        # 检查是否有用户设置的规范化器状态与当前初始化参数不匹配，如果不匹配则进行更新
        if (
            normalizer_state.get("lowercase", do_lower_case) != do_lower_case
            or normalizer_state.get("strip_accents", strip_accents) != strip_accents
            or normalizer_state.get("handle_chinese_chars", tokenize_chinese_chars) != tokenize_chinese_chars
        ):
            # 获取当前规范化器的类并进行实例化
            normalizer_class = getattr(normalizers, normalizer_state.pop("type"))
            # 更新规范化器的参数
            normalizer_state["lowercase"] = do_lower_case
            normalizer_state["strip_accents"] = strip_accents
            normalizer_state["handle_chinese_chars"] = tokenize_chinese_chars
            # 将更新后的规范化器应用于当前的tokenizer对象
            self.backend_tokenizer.normalizer = normalizer_class(**normalizer_state)

        # 更新当前对象的小写处理标志
        self.do_lower_case = do_lower_case

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 构建带有特殊标记的模型输入序列，用于序列分类任务
        output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        # 如果存在第二个序列token_ids_1，则连接第二个序列的特殊标记
        if token_ids_1 is not None:
            output += token_ids_1 + [self.sep_token_id]

        return output

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def create_token_type_ids_from_sequences(self,
                                            token_ids_0: List[int],
                                            token_ids_1: Optional[List[int]] = None) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of token IDs representing the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of token IDs representing the second sequence in sequence-pair tasks.

        Returns:
            `List[int]`: List of token type IDs according to the given sequence(s).
        """
        # Define the separator token ID and the classification token ID
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # If only one sequence is provided, return a mask with 0s for the first sequence
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]

        # If both sequences are provided, concatenate their lengths with separator and classification tokens
        # Return a mask with 0s for the first sequence and 1s for the second sequence
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]


    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary files associated with the tokenizer's model to a specified directory.

        Args:
            save_directory (str):
                Directory where the vocabulary files will be saved.
            filename_prefix (Optional[str]):
                Optional prefix to prepend to the saved vocabulary file names.

        Returns:
            Tuple[str]: Tuple containing the filenames of the saved vocabulary files.
        """
        # Call the model's save method to save the vocabulary files to the specified directory
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        
        # Return the filenames as a tuple
        return tuple(files)
```