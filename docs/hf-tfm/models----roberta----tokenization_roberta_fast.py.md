# `.\models\roberta\tokenization_roberta_fast.py`

```py
# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Fast Tokenization classes for RoBERTa."""

# 导入必要的模块和类
import json
from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers, processors

from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_roberta import RobertaTokenizer

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "FacebookAI/roberta-base": "https://huggingface.co/FacebookAI/roberta-base/resolve/main/vocab.json",
        "FacebookAI/roberta-large": "https://huggingface.co/FacebookAI/roberta-large/resolve/main/vocab.json",
        "FacebookAI/roberta-large-mnli": "https://huggingface.co/FacebookAI/roberta-large-mnli/resolve/main/vocab.json",
        "distilbert/distilroberta-base": "https://huggingface.co/distilbert/distilroberta-base/resolve/main/vocab.json",
        "openai-community/roberta-base-openai-detector": "https://huggingface.co/openai-community/roberta-base-openai-detector/resolve/main/vocab.json",
        "openai-community/roberta-large-openai-detector": (
            "https://huggingface.co/openai-community/roberta-large-openai-detector/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "FacebookAI/roberta-base": "https://huggingface.co/FacebookAI/roberta-base/resolve/main/merges.txt",
        "FacebookAI/roberta-large": "https://huggingface.co/FacebookAI/roberta-large/resolve/main/merges.txt",
        "FacebookAI/roberta-large-mnli": "https://huggingface.co/FacebookAI/roberta-large-mnli/resolve/main/merges.txt",
        "distilbert/distilroberta-base": "https://huggingface.co/distilbert/distilroberta-base/resolve/main/merges.txt",
        "openai-community/roberta-base-openai-detector": "https://huggingface.co/openai-community/roberta-base-openai-detector/resolve/main/merges.txt",
        "openai-community/roberta-large-openai-detector": (
            "https://huggingface.co/openai-community/roberta-large-openai-detector/resolve/main/merges.txt"
        ),
    },
    # 定义一个字典，将预训练模型名称映射到其对应的 tokenizer.json 文件的 URL
    "tokenizer_file": {
        "FacebookAI/roberta-base": "https://huggingface.co/FacebookAI/roberta-base/resolve/main/tokenizer.json",
        "FacebookAI/roberta-large": "https://huggingface.co/FacebookAI/roberta-large/resolve/main/tokenizer.json",
        "FacebookAI/roberta-large-mnli": "https://huggingface.co/FacebookAI/roberta-large-mnli/resolve/main/tokenizer.json",
        "distilbert/distilroberta-base": "https://huggingface.co/distilbert/distilroberta-base/resolve/main/tokenizer.json",
        "openai-community/roberta-base-openai-detector": (
            "https://huggingface.co/openai-community/roberta-base-openai-detector/resolve/main/tokenizer.json"
        ),
        "openai-community/roberta-large-openai-detector": (
            "https://huggingface.co/openai-community/roberta-large-openai-detector/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练位置嵌入的尺寸映射，将模型名称映射到嵌入维度大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "FacebookAI/roberta-base": 512,
    "FacebookAI/roberta-large": 512,
    "FacebookAI/roberta-large-mnli": 512,
    "distilbert/distilroberta-base": 512,
    "openai-community/roberta-base-openai-detector": 512,
    "openai-community/roberta-large-openai-detector": 512,
}


class RobertaTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个“快速”RoBERTa分词器（基于HuggingFace的*tokenizers*库），从GPT-2分词器继承，
    使用字节级别的字节对编码。

    这个分词器已经训练过，将空格视为标记的一部分（类似于sentencepiece），因此一个单词会因为它是否在句子开头（没有空格）而编码不同：

    ```
    >>> from transformers import RobertaTokenizerFast

    >>> tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    通过在实例化或文本调用时传递`add_prefix_space=True`，可以避免这种行为，但由于模型不是这种方式预训练的，可能会降低性能。

    <Tip>

    当使用`is_split_into_words=True`时，这个分词器需要在实例化时使用`add_prefix_space=True`。

    </Tip>

    这个分词器继承自[`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应该参考这个超类获取有关这些方法的更多信息。

    ```
    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """

    # 定义模型所需的文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练模型词汇文件的映射关系
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义预训练位置嵌入大小的最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    # 定义模型输入的名称列表，包含"input_ids"和"attention_mask"

    slow_tokenizer_class = RobertaTokenizer
    # 指定一个名为RobertaTokenizer的慢速标记器类

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
        # 初始化函数，用于创建一个新的实例对象
        # 参数解释如下：
        # vocab_file: 词汇文件路径
        # merges_file: 合并文件路径
        # tokenizer_file: 标记器文件路径
        # errors: 处理错误的方式
        # bos_token, eos_token, sep_token, cls_token, unk_token, pad_token, mask_token:
        # 特殊标记，如起始、结束、分隔、类别、未知、填充、掩码标记
        # add_prefix_space: 是否在标记前加空格，默认为False
        # trim_offsets: 是否修剪偏移量，默认为True
        # **kwargs: 其他关键字参数

        # 将mask_token转换为AddedToken对象，根据其类型进行处理
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        # 调用父类的初始化方法，传递上述参数和关键字参数
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

        # 获取当前实例的后处理器状态
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())

        # 如果预处理器的add_prefix_space参数与指定的add_prefix_space不一致，则更新为指定值
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 将当前实例的add_prefix_space属性设置为指定值
        self.add_prefix_space = add_prefix_space

        # 指定后处理器组件的名称
        tokenizer_component = "post_processor"

        # 获取后处理器组件实例
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)

        # 如果后处理器组件存在
        if tokenizer_component_instance:
            # 获取后处理器组件实例的状态
            state = json.loads(tokenizer_component_instance.__getstate__())

            # 如果state中包含'sep'，则将其转换为元组
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            # 如果state中包含'cls'，则将其转换为元组
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            # 标记是否有待应用的更改
            changes_to_apply = False

            # 如果状态中的add_prefix_space参数与指定的add_prefix_space不一致，则更新为指定值
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            # 如果状态中的trim_offsets参数与指定的trim_offsets不一致，则更新为指定值
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            # 如果有待应用的更改，则创建新的组件类并设置为后处理器组件
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)
    def mask_token(self) -> str:
        """
        `str`: 获取掩码标记，用于训练带有掩码语言建模的模型。如果在未设置的情况下使用，则记录错误日志。
        
        Roberta 分词器具有特殊的掩码标记，可以在填充掩码管道中使用。掩码标记将贪婪地包括 *<mask>* 前面的空格。
        """
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        重写掩码标记的默认行为，使其可以包含前置空格。
        
        这是为了与之前基于 Roberta 的所有使用过的模型保持向后兼容性。
        """
        # 掩码标记表现得像普通单词，即包括前置空格
        # 因此我们将 lstrip 设置为 True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        保存词汇表到指定的目录。
        
        使用分词器模型将词汇表保存到指定的目录，可选地指定文件名前缀。
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        根据输入的 token_ids 构建带有特殊标记的输入序列。
        
        在 token_ids_0 前加入 bos_token_id，之后加入 eos_token_id；如果提供了 token_ids_1，则在其前后也加入 eos_token_id。
        """
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        根据输入的 token_ids 创造 token 类型 id。
        
        根据 token_ids_0 和 token_ids_1 创建对应的 token 类型 id，用于区分不同的序列类型。
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. RoBERTa does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 定义分隔符 `[SEP]` 的 token ID 列表
        sep = [self.sep_token_id]
        # 定义类别开始 `[CLS]` 的 token ID 列表
        cls = [self.cls_token_id]

        # 如果不存在第二个序列的 token IDs，返回只含第一个序列及特殊标记的列表长度的 0 值列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 如果存在第二个序列的 token IDs，返回含两个序列及特殊标记的列表长度的 0 值列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```