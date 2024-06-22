# `.\transformers\models\qwen2\tokenization_qwen2_fast.py`

```py
# 引入必要的模块和类
from typing import Optional, Tuple
# 引入日志记录模块
from ...utils import logging
# 引入 Qwen2Tokenizer 类
from .tokenization_qwen2 import Qwen2Tokenizer

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇表文件名和对应的文件名
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_file": "tokenizer.json",
}

# 预训练模型的词汇表文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/vocab.json"},
    "merges_file": {"qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/merges.txt"},
    "tokenizer_file": {
        "qwen/qwen-tokenizer": "https://huggingface.co/qwen/qwen-tokenizer/resolve/main/tokenizer.json"
    },
}

# 预训练模型的最大输入长度限制
MAX_MODEL_INPUT_SIZES = {"qwen/qwen-tokenizer": 32768}

# Qwen2TokenizerFast 类，继承自 PreTrainedTokenizerFast
class Qwen2TokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Qwen2 tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    Same with GPT2Tokenzier, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import Qwen2TokenizerFast

    >>> tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9707, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21927, 1879]
    ```py
    This is expected.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`, *optional*):
            词汇表文件的路径。
        merges_file (`str`, *optional*):
            合并文件的路径。
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers)文件的路径，通常具有 .json 扩展名，包含加载 tokenizer 所需的所有内容。
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            未知的标记。词汇表中没有的标记不能被转换为 ID，并且设置为此标记。不适用于该 tokenizer。
        bos_token (`str`, *optional*):
            序列开始的标记。不适用于该 tokenizer。
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            序列结束的标记。
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            用于填充的标记，例如当批处理不同长度的序列时。
    """

    # 词汇表文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型通过文件名映射词汇表文件
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 模型最大输入尺寸限制
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES
    # 模型输入名称
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速 tokenizer 的类
    slow_tokenizer_class = Qwen2Tokenizer

    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        **kwargs,
    ):
        # 在实例化基类之前，至少需要传递 vocab_file 和 merges_file 给基类，以防需要初始化慢速 tokenizer；其他设置可以通过文件进行配置
        # 类似 GPT2TokenizerFast，还添加了 unk_token、bos_token 和 eos_token

        # 将 bos_token 组装成 AddedToken 对象，确保是字符串类型且带有标记的特殊字符
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        # 将 eos_token 组装成 AddedToken 对象，确保是字符串类型且带有标记的特殊字符
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        # 将 unk_token 组装成 AddedToken 对象，确保是字符串类型且带有标记的特殊字符
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        # 将 pad_token 组装成 AddedToken 对象，确保是字符串类型且带有标记的特殊字符
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        # 调用基类的初始化方法
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
    # 从transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast中复制的方法，用于保存词汇表
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用_tokenizer模型对象的save方法，将词汇表保存到指定目录，返回保存的文件列表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        # 返回保存的文件列表作为元组
        return tuple(files)
```