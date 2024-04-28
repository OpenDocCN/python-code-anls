# `.\models\gpt2\tokenization_gpt2_fast.py`

```
# 引入必要的模块
import json  # 导入 json 模块，用于处理 JSON 数据
from typing import Optional, Tuple  # 导入 typing 模块，用于类型提示

# 从 tokenizers 模块中导入预处理器
from tokenizers import pre_tokenizers  

# 从 tokenization_utils_base 模块中导入 BatchEncoding 类
from ...tokenization_utils_base import BatchEncoding  

# 从 tokenization_utils_fast 模块中导入 PreTrainedTokenizerFast 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  

# 从 utils 模块中导入 logging 函数
from ...utils import logging  

# 从 tokenization_gpt2 模块中导入 GPT2Tokenizer 类
from .tokenization_gpt2 import GPT2Tokenizer  

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/vocab.json",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/vocab.json",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/vocab.json",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/vocab.json",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/vocab.json",
    },
    "merges_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/merges.txt",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/merges.txt",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/merges.txt",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/merges.txt",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/tokenizer.json",
        "gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/tokenizer.json",
        "gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/tokenizer.json",
        "gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/tokenizer.json",
        "distilgpt2": "https://huggingface.co/distilgpt2/resolve/main/tokenizer.json",
    },
}

# 定义预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "gpt2": 1024,
    "gpt2-medium": 1024,
    "gpt2-large": 1024,
    "gpt2-xl": 1024,
    "distilgpt2": 1024,
}

# 定义 GPT2TokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class GPT2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT-2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import GPT2TokenizerFast

    >>> tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
    """

# 定义一系列参数，用于初始化 GPT2TokenizerFast 类
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = GPT2Tokenizer

# GPT2TokenizerFast 类的初始化方法
    def __init__(
        self,
        vocab_file=None,  # 词汇表文件的路径
        merges_file=None,  # merges 文件的路径
        tokenizer_file=None,  # tokenizers 文件的路径
        unk_token="<|endoftext|>",  # 未知 token，默认为 "<|endoftext|>"
        bos_token="<|endoftext|>",  # 序列的开始 token，默认为 "<|endoftext|>"
        eos_token="<|endoftext|>",  # 序列的结束 token，默认为 "<|endoftext|>"
        add_prefix_space=False,  # 是否在输入的开头加入空格，默认为 False
        **kwargs,  # 其他参数
    # 调用父类的初始化方法，传入相关参数
    def __init__(self, vocab_file, merges_file, tokenizer_file=tokenizer_file, unk_token=unk_token, bos_token=bos_token, eos_token=eos_token, add_prefix_space=add_prefix_space, **kwargs,
    ):
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

        # 为新加的参数添加默认值
        self.add_bos_token = kwargs.pop("add_bos_token", False)

        # 根据 JSON 数据创建前置切词器状态
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果前置切词器状态中的添加前缀空格参数与传入参数不一致，则修改为传入参数
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 保存传入的添加前缀空格参数值
        self.add_prefix_space = add_prefix_space

    # 批量编码
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 断言：如果添加前缀空格为真或未切分为单词，则满足条件；否则显示错误信息
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
        # 调用父类的批量编码方法
        return super()._batch_encode_plus(*args, **kwargs)

    # 编码
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 断言：如果添加前缀空格为真或未切分为单词，则满足条件；否则显示错误信息
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
        # 调用父类的编码方法
        return super()._encode_plus(*args, **kwargs)

    # 保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用_tokenizer.model.save()方法保存模型
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保��的文件名组成的元组
        return tuple(files)

    @property
    # 从transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template复制而来
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        # 警告：没有为这个分词器定义聊天模板，使用默认模板
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"
```