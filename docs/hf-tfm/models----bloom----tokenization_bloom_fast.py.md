# `.\models\bloom\tokenization_bloom_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2022 年 HuggingFace Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 不附带任何明示或暗示的保证或条件。
# 有关具体语言下方法，请参阅许可证。
"""Bloom 的分词类。"""


# 导入所需库
import pickle
from typing import Optional, Tuple

from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


# 获取日志记录器
logger = logging.get_logger(__name__)

# 设置词汇文件名称
VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

# 预训练词汇文件映射
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


# BloomTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class BloomTokenizerFast(PreTrainedTokenizerFast):
    """
    构建“快速”Bloom 分词器（由 HuggingFace 的 *tokenizers* 库支持）。基于字节级别的 Byte-Pair-Encoding。

    此分词器已经训练过，将空格视为标记的一部分（有点类似 sentencepiece），因此单词的编码会根据它是否在句子开头（无空格）而不同：

    ```python
    >>> from transformers import BloomTokenizerFast

    >>> tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom")
    >>> tokenizer("Hello world")["input_ids"]
    [59414, 8876]

    >>> tokenizer(" Hello world")["input_ids"]
    [86153, 8876]
    ```

    当使用 `is_split_into_words=True` 时，可以通过在实例化分词器时传递 `add_prefix_space=True` 来绕过这种行为。

    <提示>

    当与 `is_split_into_words=True` 一起使用时，需要使用 `add_prefix_space=True` 实例化此分词器。

    </提示>

    此分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    """
    # 参数说明
    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        merges_file (`str`):
            合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            将字节解码为 UTF-8 时遵循的范例。更多信息请参见 [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode)。
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            未知标记。不在词汇表中的标记无法转换为 ID，而是设置为该标记。
        bos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            序列起始标记。
        eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            序列结束标记。
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            是否在输入开头添加一个空格。这允许将开头的单词视为任何其他单词。(Bloom 分词器通过前导空格检测单词的开头)。
        trim_offsets (`bool`, *optional*, defaults to `True`):
            后处理步骤是否应修剪偏移量，以避免包括空格。
    """

    # 定义变量
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None
    # No `max_model_input_sizes` as BLOOM uses ALiBi positional embeddings

    # 初始化函数
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
        # 更新后端分词器的状态
        pre_tok_state = pickle.dumps(self.backend_tokenizer.pre_tokenizer)
        decoder_state = pickle.dumps(self.backend_tokenizer.decoder)

        if add_prefix_space:
            # 如果 add_prefix_space 为真，则替换状态中的相应字段值
            pre_tok_state = pre_tok_state.replace(b'"add_prefix_space":false', b'"add_prefix_space": true')
            decoder_state = decoder_state.replace(b'"add_prefix_space":false', b'"add_prefix_space": true')
        self.backend_tokenizer.pre_tokenizer = pickle.loads(pre_tok_state)
        self.backend_tokenizer.decoder = pickle.loads(decoder_state)

        # 更新 add_prefix_space 属性
        self.add_prefix_space = add_prefix_space
    # 重写父类方法_batch_encode_plus()，使用*args和**kwargs作为参数，返回BatchEncoding类型的对象
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取参数中的is_split_into_words值，默认为False
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 如果add_prefix_space为False且is_split_into_words为False时，抛出异常
        if not (self.add_prefix_space or not is_split_into_words):
            raise Exception(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True to use it with"
                " pretokenized inputs."
            )

        # 调用父类的_batch_encode_plus()方法，并返回结果
        return super()._batch_encode_plus(*args, **kwargs)

    # 重写父类方法_encode_plus()，使用*args和**kwargs作为参数，返回BatchEncoding类型的对象
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取参数中的is_split_into_words值，默认为False
        is_split_into_words = kwargs.get("is_split_into_words", False)
        
        # 如果add_prefix_space为False且is_split_into_words为False时，抛出异常
        if not (self.add_prefix_space or not is_split_into_words):
            raise Exception(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True to use it with"
                " pretokenized inputs."
            )

        # 调用父类的_encode_plus()方法，并返回结果
        return super()._encode_plus(*args, **kwargs)

    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用_tokenizer.model.save()方法保存模型
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回文件名组成的元组
        return tuple(files)

    # 默认的聊天模板
    @property
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template复制过来的
    def default_chat_template(self):
        """
        Ignore role information and concatenate messages with EOS tokens.
        """
        # 打印警告信息
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回聊天模板字符串
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
```