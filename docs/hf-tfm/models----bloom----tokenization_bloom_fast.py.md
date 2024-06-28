# `.\models\bloom\tokenization_bloom_fast.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Tokenization classes for Bloom."""


import pickle
from typing import Optional, Tuple

from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


logger = logging.get_logger(__name__)

# 定义用于存储tokenizer文件名的常量
VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

# 定义预训练模型到tokenizer文件映射的常量
PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "bigscience/tokenizer": "https://huggingface.co/bigscience/tokenizer/blob/main/tokenizer.json",
        "bigscience/bloom-560m": "https://huggingface.co/bigscience/bloom-560m/blob/main/tokenizer.json",
        "bigscience/bloom-1b1": "https://huggingface.co/bigscience/bloom-1b1/blob/main/tokenizer.json",
        "bigscience/bloom-1b7": "https://huggingface.co/bigscience/bloom-1b7/blob/main/tokenizer.json",
        "bigscience/bloom-3b": "https://huggingface.co/bigscience/bloom-3b/blob/main/tokenizer.json",
        "bigscience/bloom-7b1": "https://huggingface.co/bigscience/bloom-7b1/blob/main/tokenizer.json",
        "bigscience/bloom": "https://huggingface.co/bigscience/bloom/blob/main/tokenizer.json",
    },
}


class BloomTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Bloom tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import BloomTokenizerFast

    >>> tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
    >>> tokenizer("Hello world")["input_ids"]
    [59414, 8876]

    >>> tokenizer(" Hello world")["input_ids"]
    [86153, 8876]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    """
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Bloom tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether or not the post-processing step should trim offsets to avoid including whitespaces.
    """
    # 定义预训练模型所需的文件名称
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速分词器类，默认为 None
    slow_tokenizer_class = None
    # 没有 `max_model_input_sizes`，因为 BLOOM 使用 ALiBi 位置嵌入

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_prefix_space=False,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递必要的参数和可选参数
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )
        # 序列化后的预分词器和解码器状态
        pre_tok_state = pickle.dumps(self.backend_tokenizer.pre_tokenizer)
        decoder_state = pickle.dumps(self.backend_tokenizer.decoder)

        # 如果需要添加前缀空格，则更新序列化状态以匹配配置
        if add_prefix_space:
            pre_tok_state = pre_tok_state.replace(b'"add_prefix_space":false', b'"add_prefix_space": true')
            decoder_state = decoder_state.replace(b'"add_prefix_space":false', b'"add_prefix_space": true')
        # 反序列化并更新后端分词器的预分词器和解码器
        self.backend_tokenizer.pre_tokenizer = pickle.loads(pre_tok_state)
        self.backend_tokenizer.decoder = pickle.loads(decoder_state)

        # 设置类属性，记录是否添加前缀空格
        self.add_prefix_space = add_prefix_space
    # 定义一个方法 `_batch_encode_plus`，接受任意位置参数和关键字参数，并返回 `BatchEncoding` 对象
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 从关键字参数中获取 `is_split_into_words`，默认为 False
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 如果 `add_prefix_space` 为 False 并且 `is_split_into_words` 也为 False，则抛出异常
        if not (self.add_prefix_space or not is_split_into_words):
            raise Exception(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True to use it with"
                " pretokenized inputs."
            )

        # 调用父类的 `_batch_encode_plus` 方法，并传递所有位置参数和关键字参数
        return super()._batch_encode_plus(*args, **kwargs)

    # 定义一个方法 `_encode_plus`，接受任意位置参数和关键字参数，并返回 `BatchEncoding` 对象
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 从关键字参数中获取 `is_split_into_words`，默认为 False
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 如果 `add_prefix_space` 为 False 并且 `is_split_into_words` 也为 False，则抛出异常
        if not (self.add_prefix_space or not is_split_into_words):
            raise Exception(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True to use it with"
                " pretokenized inputs."
            )

        # 调用父类的 `_encode_plus` 方法，并传递所有位置参数和关键字参数
        return super()._encode_plus(*args, **kwargs)

    # 定义一个方法 `save_vocabulary`，接受一个保存目录路径 `save_directory` 和一个可选的文件名前缀 `filename_prefix`，返回一个包含文件名的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用 `_tokenizer` 对象的 `model.save` 方法，将模型保存到指定的 `save_directory` 中，并指定文件名前缀 `filename_prefix`
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名构成的元组
        return tuple(files)

    @property
    # 定义一个属性 `default_chat_template`，返回一个简单的聊天模板字符串，该模板忽略角色信息，并用 EOS 标记连接消息
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        # 发出警告日志，提示用户未定义聊天模板，使用默认模板
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板字符串，用于处理消息
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
```