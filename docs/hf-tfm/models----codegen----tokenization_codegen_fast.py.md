# `.\models\codegen\tokenization_codegen_fast.py`

```
# 设置文件编码格式为utf-8

# 导入所需的库
import json
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import numpy as np
from ...utils import is_tf_available, is_torch_available, logging

# 检查类型，如果类型是torch或tensorflow，导入相关库
if TYPE_CHECKING:
    if is_torch_available():
        import torch
    if is_tf_available():
        import tensorflow as tf

# 从tokenizers库中导入pre_tokenizers
from tokenizers import pre_tokenizers

# 导入其他相关文件和变量
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/vocab.json",
    },
    "merges_file": {
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "Salesforce/codegen-350M-mono": (
            "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "Salesforce/codegen-350M-mono": 2048,
}

# 定义类 CodeGenTokenizerFast，继承自 PreTrainedTokenizerFast
class CodeGenTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" CodeGen tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.
    
    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import CodeGenTokenizerFast
    
    >>> tokenizer = CodeGenTokenizerFast.from_pretrained("Salesforce/codegen-350M-mono")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]
    
    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```
    
    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since
    the model was not pretrained this way, it might yield a decrease in performance.
    
    <Tip>
    
    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.
    # 该 Tokenizer 类继承自 PreTrainedTokenizerFast，其中包含大多数主要方法。用户应参考这个超类以获取关于这些方法的更多信息。
    
    # 初始化 Tokenizer 类的参数说明：
    # - vocab_file（可选）：词汇文件的路径。
    # - merges_file（可选）：合并文件的路径。
    # - tokenizer_file（可选）：[tokenizers](https://github.com/huggingface/tokenizers) 文件的路径（通常具有 .json 扩展名），其中包含加载分词器所需的所有内容。
    # - unk_token（可选，默认为 `"<|endoftext|>"`）：未知标记。词汇表中没有的标记无法转换为 ID，并将被设置为此标记。
    # - bos_token（可选，默认为 `"<|endoftext|>"`）：序列的开始标记。
    # - eos_token（可选，默认为 `"<|endoftext|>"`）：序列的结束标记。
    # - add_prefix_space（可选，默认为 `False`）：是否在输入前添加一个初始空格。这允许将前导单词视为任何其他单词。（CodeGen 分词器通过前导空格检测单词的开头）。
    """
    
    # 定义类属性
    # 词汇文件名称列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 最大模型输入大小列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速分词器类（CodeGenTokenizer）
    slow_tokenizer_class = CodeGenTokenizer
    
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
    # 实例化一个子类对象，继承父类的__init__方法，并传入相应参数
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

        # 如果kwargs中有键"add_bos_token"，则弹出并返回值，否则返回False
        if kwargs.pop("add_bos_token", False):
            model_id = kwargs.pop("name_or_path", "")
            # 触发ValueError异常，提示GPT2的fast tokenizer不支持添加BOS token
            raise ValueError(
                "Currenty GPT2's fast tokenizer does NOT support adding a BOS token. "
                "Instead you should use GPT2's slow tokenizer class `CodeGenTokenizer` as follows: \n"
                f"`CodeGenTokenizer.from_pretrained('{model_id}')`\nor\n"
                f"`AutoTokenizer.from_pretrained('{model_id}', use_fast=False)`\n"
                "This issue will be fixed soon, see: https://github.com/huggingface/tokenizers/pull/1005."
                " so that the fast tokenizer works correctly."
            )

        # 将self.backend_tokenizer.pre_tokenizer序列化成json格式，存储到pre_tok_state中
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果pre_tok_state中的"add_prefix_space"不等于add_prefix_space，则更新add_prefix_space，并重新设定pre_tokenizer
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 将实例的add_prefix_space设为add_prefix_space
        self.add_prefix_space = add_prefix_space

    # 重新定义_batch_encode_plus方法，返回BatchEncoding对象
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取kwargs中的"is_split_into_words"，如果不存在则默认为False
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 断言如果self.add_prefix_space为True或is_split_into_words为False，则可以继续操作，否则触发异常
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        # 调用父类的_batch_encode_plus方法，返回结果
        return super()._batch_encode_plus(*args, **kwargs)

    # 重新定义_encode_plus方法，返回BatchEncoding对象
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取kwargs中的"is_split_into_words"，如果不存在则默认为False
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 断言如果self.add_prefix_space为True或is_split_into_words为False，则可以继续操作，否则触发异常
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        # 调用父类的_encode_plus方法，返回结果
        return super()._encode_plus(*args, **kwargs)

    # 定义保存词汇表的方法，返回文件名的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用tokenizer的model保存方法，将结果赋值给files
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    # 定义解码方法，接收token_ids和多个可选参数
    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        truncate_before_pattern: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        将一系列标识符转换为字符串，使用分词器和词汇表，同时具有删除特殊标记和清理分词空格的选项。

        类似于执行 `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`。

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                标记化输入 id 的列表。可以使用 `__call__` 方法获取。
            skip_special_tokens (`bool`, *optional*, 默认为 `False`):
                是否在解码时删除特殊标记。
            clean_up_tokenization_spaces (`bool`, *optional*):
                是否清理分词空格。如果为 `None`，将默认为 `self.clean_up_tokenization_spaces`（在 `tokenizer_config` 中可用）。
            truncate_before_pattern (`List[str]`, *optional*, 默认为 `None`):
                用于截断返回字符串的正则表达式字符串列表。这可用于删除额外的代码片段（例如，如果在新行开头观察到注释符号 "#"，则截断）。示例模式可以是 `["^#", re.escape("<|endoftext|>"), "^'''", "\n\n\n"]`。
            kwargs (additional keyword arguments, *optional*):
                将传递给底层模型特定的解码方法。

        Returns:
            `str`: 解码后的句子。
        """

        decoded_text = super().decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        if truncate_before_pattern is not None and len(truncate_before_pattern) > 0:
            decoded_text = self.truncate(decoded_text, truncate_before_pattern)

        return decoded_text

    def truncate(self, completion, truncate_before_pattern):
        def find_re(string, pattern, start_pos):
            m = pattern.search(string, start_pos)
            return m.start() if m else -1

        terminals = [re.compile(pattern, re.MULTILINE) for pattern in truncate_before_pattern]

        prints = list(re.finditer("^print", completion, re.MULTILINE))

        if len(prints) > 1:
            completion = completion[: prints[1].start()]

        defs = list(re.finditer("^def", completion, re.MULTILINE))

        if len(defs) > 1:
            completion = completion[: defs[1].start()]

        start_pos = 0

        terminals_pos = [
            pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1
        ]

        if len(terminals_pos) > 0:
            return completion[: min(terminals_pos)]
        else:
            return completion
```