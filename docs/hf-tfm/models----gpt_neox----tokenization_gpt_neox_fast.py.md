# `.\models\gpt_neox\tokenization_gpt_neox_fast.py`

```
# 设置脚本文件的编码格式为UTF-8
# 版权声明，指出此代码的版权归EleutherAI和The HuggingFace Inc.团队所有
#
# 根据Apache许可证2.0版进行许可，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”提供的，不提供任何明示或暗示的保证或条件
# 请参阅许可证以获取特定语言的许可证详细信息
"""GPTNeoX的标记类。"""
# 导入json模块，用于处理JSON格式的数据
import json
# 导入Optional和Tuple用于类型提示
from typing import Optional, Tuple

# 从tokenizers库中导入pre_tokenizers模块
from tokenizers import pre_tokenizers

# 从tokenization_utils_fast模块中导入PreTrainedTokenizerFast类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从utils模块中导入logging函数
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "EleutherAI/gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json",
    },
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt-neox-20b": 2048,
}


class GPTNeoXTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”的GPT-NeoX-20B标记生成器（由HuggingFace的*tokenizers*库支持）。基于字节级Byte-Pair-Encoding。

    这个标记生成器被训练成将空格视为标记的一部分（类似于sentencepiece），因此单词的编码会根据其是否位于句子开头（没有空格）而不同：

    ```python
    >>> from transformers import GPTNeoXTokenizerFast

    >>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("openai-community/gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    如果在实例化标记生成器时传递`add_prefix_space=True`，可以绕过此行为，但由于模型未用此方式进行预训练，可能会降低性能。

    <Tip>

    当与`is_split_into_words=True`一起使用时，应使用`add_prefix_space=True`实例化此标记生成器。

    </Tip>

    此标记生成器继承自[`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。

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
            other word. (GPTNeoX tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether or not the post-processing step should trim offsets to avoid including whitespaces.
    """

    # 定义常量：用于存储预定义的词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义常量：用于存储预定义的词汇文件映射字典
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义常量：用于存储预定义的最大模型输入大小列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义常量：用于存储预定义的模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递参数以配置tokenizer
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # 获取当前tokenizer的预处理状态，并更新其中的add_prefix_space选项
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            # 根据预处理器类型动态获取类，并根据更新后的状态重新配置预处理器
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置类属性，用于存储是否添加前导空格的标志
        self.add_prefix_space = add_prefix_space

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用底层tokenizer模型的保存方法，保存模型到指定目录，并返回保存的文件名元组
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    @property
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template复制而来
    # 定义一个默认的聊天模板函数，用于生成聊天内容，忽略角色信息，并使用 EOS 标记连接消息。
    logger.warning_once(
        # 发出一次性警告日志，指示没有为此分词器定义聊天模板，而是使用默认模板。
        "\nNo chat template is defined for this tokenizer - using the default template "
        f"for the {self.__class__.__name__} class. If the default is not appropriate for "
        "your model, please set `tokenizer.chat_template` to an appropriate template. "
        "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
    )
    # 返回一个字符串模板，用于格式化聊天消息，每条消息后附加 EOS 标记。
    return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
```