# `.\models\blenderbot_small\tokenization_blenderbot_small_fast.py`

```py
# coding=utf-8
# Copyright 2021, The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Fast tokenization class for BlenderbotSmall."""
from typing import List, Optional

from tokenizers import ByteLevelBPETokenizer

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入父类
from ...utils import logging  # 导入日志模块
from .tokenization_blenderbot_small import BlenderbotSmallTokenizer  # 导入慢速Tokenizer类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件名
    "merges_file": "merges.txt",  # 合并文件名
    "tokenizer_config_file": "tokenizer_config.json",  # Tokenizer配置文件名
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/vocab.json"
    },  # 预训练词汇文件映射
    "merges_file": {
        "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/merges.txt"
    },  # 预训练合并文件映射
    "tokenizer_config_file": {
        "facebook/blenderbot_small-90M": (
            "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/tokenizer_config.json"
        )
    },  # 预训练Tokenizer配置文件映射
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/blenderbot_small-90M": 512,  # 预训练位置嵌入尺寸
}


class BlenderbotSmallTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" BlenderbotSmall tokenizer (backed by HuggingFace's *tokenizers* library).

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇文件名属性
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 设置预训练词汇文件映射属性
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 设置最大模型输入尺寸属性
    slow_tokenizer_class = BlenderbotSmallTokenizer  # 设置慢速Tokenizer类属性

    def __init__(
        self,
        vocab_file=None,  # 词汇文件路径，默认为None
        merges_file=None,  # 合并文件路径，默认为None
        unk_token="<|endoftext|>",  # 未知标记，默认为"<|endoftext|>"
        bos_token="<|endoftext|>",  # 开始序列标记，默认为"<|endoftext|>"
        eos_token="<|endoftext|>",  # 结束序列标记，默认为"<|endoftext|>"
        add_prefix_space=False,  # 是否在前缀之前加空格，默认为False
        trim_offsets=True,  # 是否修剪偏移量，默认为True
        **kwargs,
    ):
        super().__init__(
            ByteLevelBPETokenizer(
                vocab=vocab_file,  # 使用指定的词汇文件初始化ByteLevelBPETokenizer
                merges=merges_file,  # 使用指定的合并文件初始化ByteLevelBPETokenizer
                add_prefix_space=add_prefix_space,  # 设置是否在前缀之前加空格
                trim_offsets=trim_offsets,  # 设置是否修剪偏移量
            ),
            bos_token=bos_token,  # 设置开始序列标记
            eos_token=eos_token,  # 设置结束序列标记
            unk_token=unk_token,  # 设置未知标记
            **kwargs,
        )
        self.add_prefix_space = add_prefix_space  # 初始化是否在前缀之前加空格属性
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 构建包含特殊标记的输入序列
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            # 如果只有一个输入序列，则返回包含特殊标记的序列
            return output

        # 如果有两个输入序列，将它们连接起来，并添加结束标记
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传入的两个序列创建用于序列对分类任务的类型标识。BlenderbotSmall 不使用 token type ids，因此返回一个全为零的列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 的列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的 ID 列表，用于序列对。

        Returns:
            `List[int]`: 全为零的列表。
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            # 如果只有一个输入序列，返回一个全为零的列表
            return len(cls + token_ids_0 + sep) * [0]
        # 如果有两个输入序列，返回一个全为零的列表，包含特殊标记和分隔符
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    # 从 transformers.models.blenderbot.tokenization_blenderbot.BlenderbotTokenizer.default_chat_template 复制过来
    def default_chat_template(self):
        """
        一个非常简单的聊天模板，只在消息之间添加空白。
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回一个用于聊天的默认模板字符串
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )
```