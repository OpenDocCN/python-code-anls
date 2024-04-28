# `.\transformers\models\led\tokenization_led_fast.py`

```
# 设置编码为 UTF-8
# 版权声明
# 根据 Apache 许可 2.0 版本获取许可文件
# 许可获取链接
# 仅在符合许可证规定的情况下使用此文件
# 根据许可证规定分发的文件以“AS IS”方式分发，没有任何形式的明示或暗示保证。
# 有关特定语言的条件，请查看许可证，控制权限和限制
"""LED 的 Tokenization 类"""

import json
from typing import Dict, List, Optional, Tuple, Union

from tokenizers import pre_tokenizers, processors

from ...tokenization_utils_base import AddedToken, BatchEncoding, EncodedInput
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import PaddingStrategy, logging
from .tokenization_led import LEDTokenizer

# 获取 logging 实例
logger = logging.get_logger(__name__)

# 词汇表文件名
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/vocab.json",
    },
    "merges_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/tokenizer.json",
    },
}

# 预训练位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allenai/led-base-16384": 16384,
}

class LEDTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Fast 版 LED 分词器，基于 HuggingFace 的 tokenizers 库，派生自 GPT-2 分词器，使用字节级 BPE。

    此分词器已经训练过，将空格视为令牌的一部分（有点像 sentencepiece），因此单词将根据是否在句子开头（没有空格）而编码不同:

    你可以通过实例化分词器时传递 `add_prefix_space=True` 参数或在处理文本时传递，来解决这个问题
    但是由于模型不是以这种方式进行的预训练，因此可能会降低性能。

    <Tip>

    当与 `is_split_into_words=True` 一起使用时，需要使用 `add_prefix_space=True` 实例化此分词器。

    </Tip>

    这个分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应
    # 请参考这个超类以获取有关这些方法的更多信息。

    Args:
        # 词汇文件的路径
        vocab_file (`str`):
        # 合并文件的路径
        merges_file (`str`):
        # 解码字节为 UTF-8 时遇到错误的处理方式，默认为 "replace"，参考 bytes.decode
        errors (`str`, *optional*, defaults to `"replace"`):
        # 在预训练期间使用的序列的开头标记，也可用作序列分类器的标记
        bos_token (`str`, *optional*, defaults to `"<s>"`):
        # 序列的结束标记
        eos_token (`str`, *optional*, defaults to `"</s>"`):
        # 用于连接序列的分隔符标记，如用于序列分类或问题回答的文本和问题连接
        sep_token (`str`, *optional*, defaults to `"</s>"`):
        # 用于序列分类时的分类器标记，特殊标记序列的第一个标记
        cls_token (`str`, *optional*, defaults to `"<s>"`):
        # 未知标记，当词汇中不存在某个标记时，转换为未知标记
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
        # 用于填充的标记，用于处理不同长度序列的批处理时
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
        # 用于蒙版标记的标记，用于训练带有蒙版语言建模的模型
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
        # 是否在输入前添加初始空格，这允许处理前导单词就像处理其他单词一样（LED 标记器通过前导空格检测单词的开始）
        add_prefix_space (`bool`, *optional*, defaults to `False`):
        # 后处理步骤是否应该修剪偏移以避免包含空格
        trim_offsets (`bool`, *optional*, defaults to `True`):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 初始化变量 max_model_input_sizes，赋值为预训练位置嵌入大小的列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化变量 slow_tokenizer_class，赋值为 LEDTokenizer 类
    slow_tokenizer_class = LEDTokenizer
    # 初始化变量 model_input_names，赋值为包含字符串 "input_ids" 和 "attention_mask" 的列表
    model_input_names = ["input_ids", "attention_mask"]

    # 从 transformers.models.bart.tokenization_bart_fast.BartTokenizerFast.__init__ 复制而来的方法
    # 定义类的初始化函数
    def __init__(
        self,
        # 词汇表文件路径，默认为 None
        vocab_file=None,
        # 合并文件路径，默认为 None
        merges_file=None,
        # 分词器文件路径，默认为 None
        tokenizer_file=None,
        # 错误处理方式，默认为 "replace"
        errors="replace",
        # 起始标记，默认为 "<s>"
        bos_token="<s>",
        # 结束标记，默认为 "</s>"
        eos_token="</s>",
        # 分隔标记，默认为 "</s>"
        sep_token="</s>",
        # 类起始标记，默认为 "<s>"
        cls_token="<s>",
        # 未知标记，默认为 "<unk>"
        unk_token="<unk>",
        # 填充标记，默认为 "<pad>"
        pad_token="<pad>",
        # 掩码标记，默认为 "<mask>"
        mask_token="<mask>",
        # 是否添加前缀空格，默认为 False
        add_prefix_space=False,
        # 是否修剪偏移量，默认为 True
        trim_offsets=True,
        # 其它关键字参数
        **kwargs,
    # we have to specify that this tokens is special otherwise adding it will reset the normalized flag to `False` in `add_special_tokens`
    # 指定这个标记是特殊的，否则在add_special_tokens中添加它将会将normalized标志重置为`False`
    mask_token = (
        AddedToken(mask_token, lstrip=True, normalized=True, special=True)
        if isinstance(mask_token, str)
        else mask_token
    )
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

    pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
    # 获取前置标记器的状态并转为json
    if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
        # 如果前置标记器的状态中的add_prefix_space值不等于当前add_prefix_space值
        pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
        # 获取前置标记器的类
        pre_tok_state["add_prefix_space"] = add_prefix_space
        # 更新前置标记器的add_prefix_space值
        self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
        # 更新后端标记器的前置标记器

    self.add_prefix_space = add_prefix_space
    # 更新当前类的add_prefix_space值

    # the pre_tokenizer is already updated in the GPT2TokenizerFast `__init__`
    # 前置标记器已经在GPT2TokenizerFast的`__init__`中更新了

    tokenizer_component = "post_processor"
    tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
    # 获取后端标记器中的post_processor实例

    if tokenizer_component_instance:
        # 如果post_processor实例存在
        state = json.loads(tokenizer_component_instance.__getstate__())
        # 获取post_processor实例的状态

        # The lists 'sep' and 'cls' must be cased in tuples for the object `post_processor_class`
        # 'sep'和'cls'列表必须以元组的形式给`post_processor_class`对象
        if "sep" in state:
            state["sep"] = tuple(state["sep"])
        if "cls" in state:
            state["cls"] = tuple(state["cls"])

        changes_to_apply = False

        if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            state["add_prefix_space"] = add_prefix_space
            changes_to_apply = True

        if state.get("trim_offsets", trim_offsets) != trim_offsets:
            state["trim_offsets"] = trim_offsets
            changes_to_apply = True

        if changes_to_apply:
            component_class = getattr(processors, state.pop("type"))
            new_value = component_class(**state)
            setattr(self.backend_tokenizer, tokenizer_component, new_value)


    @property
    # Copied from transformers.models.bart.tokenization_bart_fast.BartTokenizerFast.mask_token with BART->LED
    # 从transformers.models.bart.tokenization_bart_fast.BartTokenizerFast.mask_token拷贝，将BART->LED
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.

        LED tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *
    # 从 transformers.models.bart.tokenization_bart_fast.BartTokenizerFast.create_token_type_ids_from_sequences 复制而来
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传入的两个序列创建用于序列对分类任务的掩码。LED 不使用 token 类型 id，因此返回一个全零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 的列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个用于序列对的 ID 列表。

        Returns:
            `List[int]`: 全零列表。
        """
        sep = [self.sep_token_id]  # 分隔符的 ID 列表
        cls = [self.cls_token_id]  # 类别标记的 ID 列表

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # 返回一个长度为所有标记数加上特殊标记的全零列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]  # 返回一个长度为所有标记数加上特殊标记的全零列表

    # 从 transformers.models.led.tokenization_led.LEDTokenizer._pad 复制而来
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        # 调用父类方法 _pad() 对输入进行填充，并返回填充后的编码字典
        encoded_inputs = super()._pad(
            encoded_inputs=encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # 如果未指定返回注意力掩码，则根据模型默认值确定是否返回
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        # 如果要返回注意力掩码，并且编码字典中包含全局注意力掩码
        if return_attention_mask and "global_attention_mask" in encoded_inputs:
            # 获取输入中的第一个序列（通常是输入文本序列）
            required_input = encoded_inputs[self.model_input_names[0]]
            # 判断全局注意力掩码是否需要填充以匹配其他序列输入的长度
            needs_to_be_padded = len(encoded_inputs["global_attention_mask"]) != len(required_input)

            # 如果需要填充
            if needs_to_be_padded:
                # 计算需要填充的长度差异
                difference = len(required_input) - len(encoded_inputs["global_attention_mask"])

                # 根据填充位置调整全局注意力掩码
                if self.padding_side == "right":
                    # 使用 `-1` 表示不需要注意
                    encoded_inputs["global_attention_mask"] = (
                        encoded_inputs["global_attention_mask"] + [-1] * difference
                    )
                elif self.padding_side == "left":
                    encoded_inputs["global_attention_mask"] = [-1] * difference + encoded_inputs[
                        "global_attention_mask"
                    ]
                else:
                    # 如果填充策略无效，则引发 ValueError 异常
                    raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        # 返回填充后的编码字典
        return encoded_inputs
```