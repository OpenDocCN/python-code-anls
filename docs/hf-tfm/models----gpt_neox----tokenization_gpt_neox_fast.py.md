# `.\models\gpt_neox\tokenization_gpt_neox_fast.py`

```py
# 设置脚本编码为 UTF-8
# 版权声明
# Apache 许可证 2.0 版本
# 获取许可证的网址
# 在适用法律要求或书面同意的情况下，根据"AS IS"基础分发软件，不附带任何形式的担保或条件，无论是明示的还是暗示的
# 查看特定语言的限制和权限的许可证
"""GPTNeoX 的 Tokenization 类。"""
# 导入必要的库
import json
from typing import Optional, Tuple
# 从 tokenizers 库导入 pre_tokenizers
from tokenizers import pre_tokenizers
# 从 tokenization_utils_fast 中导入 PreTrainedTokenizerFast
# 从 utils 中导入 logging
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# VOCAB_FILES_NAMES 字典，存储 tokenizer 文件相关的文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# PRETRAINED_VOCAB_FILES_MAP 字典，存储预训练 tokenizer 和其文件的映射关系
PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "EleutherAI/gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/tokenizer.json",
    },
}

# PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES 字典，存储预训练 tokenizer 和其位置嵌入大小的映射关系
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt-neox-20b": 2048,
}

# 定义 GPTNeoXTokenizerFast 类，继承 PreTrainedTokenizerFast
class GPTNeoXTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个 "快速" GPT-NeoX-20B tokenizer（由 HuggingFace 的 *tokenizers* 库支持）。基于字节级 
    Byte-Pair-Encoding。
    
    这个 tokenizer 已经经过训练，以将空格视为词元的一部分（有点像 sentencepiece），因此一个单词被编码时，
    它在句子中是否位于开头（无空格）会有不同的编码：

    ```python
    >>> from transformers import GPTNeoXTokenizerFast

    >>> tokenizer = GPTNeoXTokenizerFast.from_pretrained("gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```py

    可以通过在实例化这个 tokenizer 时传递 `add_prefix_space=True` 来解决这个行为，但由于模型没有以这种方式
    预训练，这可能会降低性能。

    <Tip>

    当与 `is_split_into_words=True` 一起使用时，需要使用 `add_prefix_space=True` 实例化这个 tokenizer。

    </Tip>

    这个 tokenizer 继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应该参考这个超类了解这些方法的更多信息。
    
    # 初始化一个 GPT2Tokenizer 对象
    def __init__(
        self,
        vocab_file=None,  # 词汇表文件路径，默认为 None
        merges_file=None,  # 合并文件路径，默认为 None
        tokenizer_file=None,  # 分词器文件路径，默认为 None
        unk_token="<|endoftext|>",  # 未知 token，默认为 "<|endoftext|>"
        bos_token="<|endoftext|>",  # 序列起始 token，默认为 "<|endoftext|>"
        eos_token="<|endoftext|>",  # 序列结束 token，默认为 "<|endoftext|>"
        add_prefix_space=False,  # 是否在输入的开头添加空格，默认为 False
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化方法
        super().__init__(
            vocab_file,  # 词汇表文件路径
            merges_file,  # 合并文件路径
            tokenizer_file=tokenizer_file,  # 分词器文件路径
            unk_token=unk_token,  # 未知 token
            bos_token=bos_token,  # 序列起始 token
            eos_token=eos_token,  # 序列结束 token
            add_prefix_space=add_prefix_space,  # 是否在输入的开头添加空格
            **kwargs,  # 其他参数
        )
        
        # 从 backend_tokenizer 中加载预分词器状态
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果预分词器状态中的 add_prefix_space 参数不等于传入的 add_prefix_space 参数
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            # 获取预分词器类型并创建相应的预分词器类
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            # 更新预分词器状态中的 add_prefix_space 参数
            pre_tok_state["add_prefix_space"] = add_prefix_space
            # 重新设置 backend_tokenizer 的预分词器
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 保存是否添加前缀空格的参数
        self.add_prefix_space = add_prefix_space

    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用 tokenizer.model.save 方法保存词汇表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件名
        return tuple(files)

    # 返回默认的对话模板
    @property
    # 从 transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template 复制过来的
    # 默认的聊天模板，忽略角色信息，只是用 EOS 标记连接消息
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        # 如果没有为这个分词器定义聊天模板，则使用默认模板
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回用于连接消息的字符串模板
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
```