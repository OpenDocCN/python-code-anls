# `.\models\mpnet\tokenization_mpnet_fast.py`

```
# coding=utf-8
# 文件编码声明，指定文件采用UTF-8编码格式

# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 版权声明，指明版权归属及保留的所有权声明

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
# Apache License 2.0许可声明，指定了软件使用的许可条件

"""Fast Tokenization classes for MPNet."""
# 为MPNet提供快速分词的类声明

import json
from typing import List, Optional, Tuple

from tokenizers import normalizers

from ...tokenization_utils import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_mpnet import MPNetTokenizer

# 导入所需的模块和类

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}
# 定义了词汇表和分词器文件的名称及其默认文件名

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/tokenizer.json",
    },
}
# 预训练模型与其对应的词汇表和分词器文件的映射关系

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "microsoft/mpnet-base": 512,
}
# 预训练模型的位置嵌入大小配置

PRETRAINED_INIT_CONFIGURATION = {
    "microsoft/mpnet-base": {"do_lower_case": True},
}
# 预训练模型的初始化配置

class MPNetTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" MPNet tokenizer (backed by HuggingFace's *tokenizers* library). Based on WordPiece.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    # MPNetTokenizerFast类的定义，构建基于HuggingFace的tokenizers库支持的“快速”MPNet分词器，基于WordPiece算法
    # 词汇表文件名列表，定义了与不同模型相关联的预训练词汇表文件名
    vocab_files_names = VOCAB_FILES_NAMES
    
    # 预训练模型的词汇表文件映射，将预训练模型的名称映射到其对应的词汇表文件
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    
    # 预训练模型初始化的配置，包括词汇表、特殊标记等信息的配置
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    
    # 预训练位置嵌入大小，用于确定模型输入的最大长度限制
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
    # 使用的慢速分词器的类，这里指定了 MPNetTokenizer 作为慢速分词器的实现
    slow_tokenizer_class = MPNetTokenizer
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化函数，用于创建一个新的 Tokenizer 对象
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        do_lower_case=True,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="[UNK]",
        pad_token="<pad>",
        mask_token="<mask>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        # 将字符串类型的特殊 token 转换为 AddedToken 对象，保持左右空白不去除
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # 设置 mask_token 作为特殊 token，包含之前的空格
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        # 调用父类的初始化函数，初始化 Tokenizer
        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        # 获取当前标准化器的状态并根据初始化参数调整
        pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__())
        if (
            pre_tok_state.get("lowercase", do_lower_case) != do_lower_case
            or pre_tok_state.get("strip_accents", strip_accents) != strip_accents
        ):
            pre_tok_class = getattr(normalizers, pre_tok_state.pop("type"))
            pre_tok_state["lowercase"] = do_lower_case
            pre_tok_state["strip_accents"] = strip_accents
            self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)

        # 设置当前对象的小写参数
        self.do_lower_case = do_lower_case

    # 获取 mask_token 的属性方法
    @property
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        MPNet tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *<mask>*.
        """
        # 如果 mask_token 尚未设置，记录错误并返回 None
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)
    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.

        This is needed to preserve backward compatibility with all the previously used models based on MPNet.
        """
        # 将 mask token 设置为像普通单词一样，包含其前面的空格
        # 因此我们将 lstrip 设置为 True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Constructs input with special tokens by prepending bos_token_id, appending eos_token_id,
        and optionally adding token_ids_1 with an additional eos_token_id.

        Args:
            token_ids_0 (list of int):
                List of input token IDs.
            token_ids_1 (list of int, optional):
                Optional second list of token IDs for sequence pairs.

        Returns:
            list of int: Combined list of token IDs with special tokens.
        """
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. MPNet does not
        make use of token type IDs, therefore a list of zeros is returned.

        Args:
            token_ids_0 (List[int]):
                List of token IDs.
            token_ids_1 (List[int], optional):
                Optional second list of token IDs for sequence pairs.

        Returns:
            List[int]: List of zeros indicating token type IDs.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Saves the vocabulary and related model files to the specified directory.

        Args:
            save_directory (str):
                Directory where the vocabulary will be saved.
            filename_prefix (str, optional):
                Optional prefix for the saved vocabulary files.

        Returns:
            Tuple[str]: Tuple containing the paths of the saved files.
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
```