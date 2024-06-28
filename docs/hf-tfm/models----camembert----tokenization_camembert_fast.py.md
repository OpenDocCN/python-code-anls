# `.\models\camembert\tokenization_camembert_fast.py`

```py
# coding=utf-8
# 上面的注释指定了文件的编码格式为 UTF-8

# 版权声明和许可证信息，说明了代码的使用权限和责任限制
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
# limitations under the License

""" Fast tokenization classes for Camembert model."""

# 导入必要的模块和库
import os
from shutil import copyfile
from typing import List, Optional, Tuple

# 导入 HuggingFace 库中的一些实用函数和类
from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging

# 检查是否安装了 sentencepiece 库
if is_sentencepiece_available():
    # 如果安装了 sentencepiece，则导入 CamembertTokenizer 类
    from .tokenization_camembert import CamembertTokenizer
else:
    # 否则，将 CamembertTokenizer 设为 None
    CamembertTokenizer = None

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 CamembertTokenizerFast 类的相关常量和映射
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 预训练模型使用的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "almanach/camembert-base": "https://huggingface.co/almanach/camembert-base/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_file": {
        "almanach/camembert-base": "https://huggingface.co/almanach/camembert-base/resolve/main/tokenizer.json",
    },
}

# 预训练模型的位置嵌入大小映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "almanach/camembert-base": 512,
}

# SentencePiece 模型中表示子词的前缀
SPIECE_UNDERLINE = "▁"

# CamembertTokenizerFast 类的定义
class CamembertTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" CamemBERT tokenizer (backed by HuggingFace's *tokenizers* library). Adapted from
    [`RobertaTokenizer`] and [`XLNetTokenizer`]. Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """

    # 类的主要功能和继承关系的说明
    # 构造一个基于 HuggingFace 的 tokenizers 库的 "快速" CamemBERT 分词器

    # 这里没有其他代码，因此没有需要添加注释的额外行
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
            SentencePiece文件的路径，用于实例化分词器的词汇表文件。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            序列的起始标记，用于预训练。可用作序列分类器的标记。

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            在使用特殊标记构建序列时，并非使用此标记作为序列的起始标记。实际使用的是 `cls_token`。

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
            序列的结束标记。

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            在使用特殊标记构建序列时，并非使用此标记作为序列的结束标记。实际使用的是 `sep_token`。

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            分隔符标记，在构建来自多个序列的序列时使用，例如用于序列分类的两个序列，或用于问答中的文本和问题序列。也用作使用特殊标记构建的序列的最后一个标记。

        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            在进行序列分类（整个序列而不是每个标记的分类）时使用的分类器标记。使用特殊标记构建序列时，它是序列的第一个标记。

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            未知标记。不在词汇表中的标记无法转换为ID，因此将被设置为此标记。

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
            用于填充的标记，例如在批处理不同长度的序列时使用。

        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
            用于掩码值的标记。在进行掩码语言建模训练时使用的标记。这是模型将尝试预测的标记。

        additional_special_tokens (`List[str]`, *optional*, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
            分词器使用的额外特殊标记列表。

    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = CamembertTokenizer
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
        additional_special_tokens=["<s>NOTUSED", "</s>NOTUSED", "<unk>NOTUSED"],
        **kwargs,
    ):
        # Mask token behavior is modified to strip left spaces and is marked as special
        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token
        # 调用父类的构造方法，初始化基类的属性
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        # 设置实例的属性，保存词汇表文件路径
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 判断词汇表文件是否存在，用于判断是否可以保存慢速分词器
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An CamemBERT sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of input IDs with the appropriate special tokens.
        """

        if token_ids_1 is None:
            # 返回只包含单个序列的输入 ID，包含特殊的开始和结束标记
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        # 返回包含两个序列的输入 ID，包含特殊的开始和结束标记
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        # 根据输入的序列创建 token type IDs，用于区分输入序列的类型（单个序列或序列对）
        # 在 CamemBERT 中，token type IDs 用于指示每个 token 属于哪个输入序列
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. CamemBERT, like
        RoBERTa, does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # Separator token ID for separating sequences
        sep = [self.sep_token_id]
        # CLS token ID for start of sequence classification
        cls = [self.cls_token_id]

        # If only one sequence is provided
        if token_ids_1 is None:
            # Return a list of zeros with the length of cls + token_ids_0 + sep
            return len(cls + token_ids_0 + sep) * [0]
        
        # If two sequences are provided
        # Return a list of zeros with the length of cls + token_ids_0 + 2 * sep + token_ids_1 + sep
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # Check if saving slow tokenizer vocabulary is possible
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # Check if save_directory exists and is a directory; log error if not
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # Define the output vocabulary file path
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # Copy the current vocabulary file to the specified directory if different
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # Return the path to the saved vocabulary file
        return (out_vocab_file,)
```