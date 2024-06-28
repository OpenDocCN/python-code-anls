# `.\models\blenderbot\tokenization_blenderbot_fast.py`

```py
# 引入必要的模块和库
import json  # 用于处理 JSON 格式的数据
from typing import List, Optional, Tuple  # 引入类型提示相关的模块

from tokenizers import pre_tokenizers, processors  # 从 tokenizers 库引入预处理器和处理器

from ...tokenization_utils_base import AddedToken, BatchEncoding  # 从本地模块引入相应的类和函数
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 从本地模块引入 PreTrainedTokenizerFast 类
from ...utils import logging  # 从本地模块引入日志记录器
from .tokenization_blenderbot import BlenderbotTokenizer  # 从当前目录的 tokenization_blenderbot 模块引入 BlenderbotTokenizer 类

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义 BlenderbotTokenizerFast 类的静态属性：包含各个文件的名称
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "tokenizer_config_file": "tokenizer_config.json",
}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/vocab.json"},
    "merges_file": {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/merges.txt"},
    "tokenizer_config_file": {
        "facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/tokenizer_config.json"
    },
}

# 定义预训练模型的位置嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"facebook/blenderbot-3B": 128}


class BlenderbotTokenizerFast(PreTrainedTokenizerFast):
    """
    快速实现的 Blenderbot 分词器，基于 HuggingFace 的 tokenizers 库，衍生自 GPT-2 分词器，使用字节级别的 BPE。

    这个分词器经过训练，将空格视为词元的一部分（类似于 sentencepiece），因此一个词在句子开头（无空格）和其他位置编码会不同：

    ```
    >>> from transformers import BlenderbotTokenizerFast

    >>> tokenizer = BlenderbotTokenizerFast.from_pretrained("facebook/blenderbot-3B")
    >>> tokenizer("Hello world")["input_ids"]
    [6950, 1085, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [6950, 1085, 2]
    ```

    如果要避免这种行为，可以在实例化分词器时或调用时传递 add_prefix_space=True，但由于模型不是这样预训练的，可能会降低性能。

    <Tip>

    当使用 is_split_into_words=True 时，需要以 add_prefix_space=True 实例化这个分词器。

    </Tip>

    这个分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大部分主要方法。用户应该
    """
    pass  # 类定义结束，暂无额外的代码逻辑
    # 设置 Transformer 模型的词汇文件名称常量，这些文件包含了模型训练时使用的词汇表和合并文件
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射，指定了预训练模型使用的各类词汇文件的位置
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 使用预训练模型的位置嵌入尺寸作为最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    
    # 定义模型输入的名称列表，包括输入的标记和注意力掩码
    model_input_names = ["input_ids", "attention_mask"]
    
    # 指定使用的慢速标记化器类为BlenderbotTokenizer
    slow_tokenizer_class = BlenderbotTokenizer

    # 从transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast.__init__方法复制而来，
    # 用于初始化BlenderbotTokenizer类
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        tokenizer_file=None,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        trim_offsets=True,
        **kwargs,
    ):
        # 如果 mask_token 是字符串，则创建一个新的 AddedToken 对象，否则直接使用传入的 mask_token 对象
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )
        # 调用父类的构造函数，初始化 BlenderbotTokenizerFast 对象
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

        # 获取当前的预处理器（pre_tokenizer）状态，并将其转换为 JSON 格式
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果当前预处理器的 add_prefix_space 属性与传入的 add_prefix_space 不一致，则更新预处理器状态
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置对象属性 add_prefix_space
        self.add_prefix_space = add_prefix_space

        # 定义 tokenizer_component 变量为 "post_processor"，获取后处理器实例
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        # 如果后处理器实例存在，则获取其状态信息
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # 如果状态中包含 "sep"，则将其值转换为元组
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            # 如果状态中包含 "cls"，则将其值转换为元组
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            changes_to_apply = False

            # 如果状态中的 add_prefix_space 与传入的 add_prefix_space 不一致，则更新状态
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            # 如果状态中的 trim_offsets 与传入的 trim_offsets 不一致，则更新状态
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            # 如果有更新需要应用，则创建新的后处理器对象并设置回 backend_tokenizer
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)
    def mask_token(self) -> str:
        """
        `str`: 获取掩码标记，用于训练具有掩码语言建模功能的模型。如果在未设置的情况下使用，则记录错误信息。

        Blenderbot 分词器有一个特殊的掩码标记，用于在填充掩码流水线中使用。掩码标记将贪婪地包括 *<mask>* 前面的空格。
        """
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        重写掩码标记的默认行为，使其能够包含前置空格。

        这是为了与所有基于 Roberta 的先前使用的模型保持向后兼容性。
        """
        # 掩码标记行为类似普通单词，即包含前置空格，因此设置 lstrip 为 True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast._batch_encode_plus 复制，替换 RoBERTa 为 Blenderbot
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast._encode_plus 复制，替换 RoBERTa 为 Blenderbot
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast.save_vocabulary 复制，替换 RoBERTa 为 Blenderbot
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    # 从 transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast.create_token_type_ids_from_sequences 复制，替换 RoBERTa 为 Blenderbot
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ):
        """
        根据 token_ids_0 和（可选）token_ids_1 创建 token 类型 ID。

        如果使用预分词的输入，需要用 add_prefix_space=True 来实例化 {self.__class__.__name__}。
        """
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
        # Define special tokens
        sep = [self.sep_token_id]  # Separator token ID
        cls = [self.cls_token_id]  # Classification token ID

        # If only one sequence provided
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # Return mask of zeros
        # If two sequences provided
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]  # Return mask of zeros

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
        # Concatenate input tokens with end-of-sequence token
        return token_ids_0 + [self.eos_token_id]

    @property
    # Copied from transformers.models.blenderbot.tokenization_blenderbot.BlenderbotTokenizer.default_chat_template
    def default_chat_template(self):
        """
        A very simple chat template that just adds whitespace between messages.
        """
        # Issue a warning message if no chat template is defined
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # Return default chat template with placeholders
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}"
            "{{ message['content'] }}"
            "{% if not loop.last %}{{ '  ' }}{% endif %}"
            "{% endfor %}"
            "{{ eos_token }}"
        )
```