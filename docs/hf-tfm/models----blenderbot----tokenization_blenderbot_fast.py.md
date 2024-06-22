# `.\transformers\models\blenderbot\tokenization_blenderbot_fast.py`

```py
# 导入所需模块和类
import json  # 导入用于 JSON 解析的模块
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

from tokenizers import pre_tokenizers, processors  # 导入 tokenizers 库中的预处理器和处理器类

from ...tokenization_utils_base import AddedToken, BatchEncoding  # 导入基础分词器相关类和函数
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入预训练的快速分词器类
from ...utils import logging  # 导入日志相关的工具类
from .tokenization_blenderbot import BlenderbotTokenizer  # 导入 Blenderbot 分词器类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件名
    "merges_file": "merges.txt",  # 合并文件名
    "tokenizer_config_file": "tokenizer_config.json",  # 分词器配置文件名
}

# 定义预训练词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/vocab.json"
    },  # 词汇表文件映射
    "merges_file": {
        "facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/merges.txt"
    },  # 合并文件映射
    "tokenizer_config_file": {
        "facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/tokenizer_config.json"
    },  # 分词器配置文件映射
}

# 定义预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/blenderbot-3B": 128}  # 预训练位置嵌入大小映射


class BlenderbotTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Blenderbot tokenizer (backed by HuggingFace's *tokenizers* library), derived from the GPT-2
    tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import BlenderbotTokenizerFast

    >>> tokenizer = BlenderbotTokenizerFast.from_pretrained("facebook/blenderbot-3B")
    >>> tokenizer("Hello world")["input_ids"]
    [6950, 1085, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [6950, 1085, 2]
    ```py

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    """
    # 从这个超类获取关于这些方法的更多信息。

    Args:
        vocab_file (`str`):
            词汇表文件的路径。
        merges_file (`str`):
            合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            在将字节解码为 UTF-8 时遵循的范例。参见
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) 获取更多信息。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            在预训练期间使用的序列开始标记。可以用作序列分类器的标记。

            <Tip>

            在使用特殊标记构建序列时，这不是用于序列开始的标记。使用的标记是 `cls_token`。

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            序列结束标记。

            <Tip>

            在使用特殊标记构建序列时，这不是用于序列结束的标记。使用的标记是 `sep_token`。

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            分隔符标记，用于从多个序列构建序列，例如用于序列分类的两个序列或用于文本和问题的问题回答。还用作使用特殊标记构建的序列的最后一个标记。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            分类器标记，用于执行序列分类（对整个序列进行分类而不是对每个标记进行分类）。使用特殊标记构建时，它是序列的第一个标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中没有的标记无法转换为 ID，而是设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如当批处理不同长度的序列时。
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            用于屏蔽值的标记。这是训练具有屏蔽语言建模的模型时使用的标记。这是模型将尝试预测的标记。
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            是否将初始空格添加到输入。这允许将前导单词视为任何其他单词。 （Blenderbot 标记器通过前导空格检测单词的开始）。
        trim_offsets (`bool`, *optional*, defaults to `True`):
            后处理步骤是否应修剪偏移量以避免包含空格。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义变量，最大模型输入尺寸为预训练位置嵌入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义变量，模型输入的名称列表为 ["input_ids", "attention_mask"]
    model_input_names = ["input_ids", "attention_mask"]
    # 定义变量，慢速分词器的类为 BlenderbotTokenizer
    slow_tokenizer_class = BlenderbotTokenizer

    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast.__init__ 中复制代码，并将 RoBERTa 替换为 Blenderbot
    def __init__(
        self,
        vocab_file=None,  # 词汇文件路径，默认为 None
        merges_file=None,  # 合并文件路径，默认为 None
        tokenizer_file=None,  # 分词器文件路径，默认为 None
        errors="replace",  # 错误处理方式，默认为 "replace"
        bos_token="<s>",  # 开始词标记，默认为 "<s>"
        eos_token="</s>",  # 结束词标记，默认为 "</s>"
        sep_token="</s>",  # 分隔词标记，默认为 "</s>"
        cls_token="<s>",  # 类别词标记，默认为 "<s>"
        unk_token="<unk>",  # 未知词标记，默认为 "<unk>"
        pad_token="<pad>",  # 填充词标记，默认为 "<pad>"
        mask_token="<mask>",  # 掩码词标记，默认为 "<mask>"
        add_prefix_space=False,  # 是否在前面添加空格，默认为 False
        trim_offsets=True,  # 是否修剪偏移量，默认为 True
        **kwargs,  # 其他关键字参数
    ):
        # 如果 mask_token 是字符串类型，则创建一个 AddedToken 对象
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )
        # 调用父类的初始化方法，传入参数
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )

        # 获取前处理器的状态，并转换为 JSON 格式
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果前处理器的 add_prefix_space 属性与当前值不同，则更新为当前值
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置 add_prefix_space 属性
        self.add_prefix_space = add_prefix_space

        # 获取后处理器的实例
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # 将列表 'sep' 和 'cls' 转换为元组，以便传递给 post_processor_class 对象
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            changes_to_apply = False

            # 如果 add_prefix_space 属性与当前值不同，则更新为当前值
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            # 如果 trim_offsets 属性与当前值不同，则更新为当前值
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            # 如果有更改需要应用，则创建新的组件类对象并设置给后处理器
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

    @property
    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast.mask_token 复制，将 Roberta->Blenderbot, RoBERTa->Blenderbot
    def mask_token(self) -> str:
        """
        `str`: 返回掩码标记，用于在进行掩码语言建模训练时使用。如果在未设置的情况下使用，则记录错误日志。

        Blenderbot 分词器有一个特殊的掩码标记，可用于填充掩码管道。掩码标记将贪婪地包括 *<mask>* 前面的空格。
        """
        # 如果掩码标记尚未设置，则记录错误日志并返回 None
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        # 返回掩码标记的字符串形式
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        重写掩码标记的默认行为，使其吞噬前面的空格。

        这是为了与基于 RoBERTa 的所有先前使用的模型保持向后兼容性。
        """
        # 控制掩码标记的行为，使其像普通单词一样，即包括前面的空格
        # 因此我们将 lstrip 设置为 True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast._batch_encode_plus 复制并修改为 Blenderbot
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast._encode_plus 复制并修改为 Blenderbot
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast.save_vocabulary 复制并修改为 Blenderbot
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast.create_token_type_ids_from_sequences 复制并修改为 Blenderbot
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. Blenderbot does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 定义特殊分隔符和类标记的 ID 列表
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        # 如果第二个序列的 ID 列表为空，则返回由特殊符号构成的列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则，返回由特殊符号和两个序列的 ID 列表构成的列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A Blenderbot sequence has the following format:
        - single sequence: ` X </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (`List[int]`, *optional*):
                Will be ignored
        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 将特殊结束符号添加到第一个序列的 ID 列表中
        return token_ids_0 + [self.eos_token_id]

    @property
    # Copied from transformers.models.blenderbot.tokenization_blenderbot.BlenderbotTokenizer.default_chat_template
    def default_chat_template(self):
        """
        A very simple chat template that just adds whitespace between messages.
        """
        # 如果未定义聊天模板，则使用默认模板，并发出警告
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )
```