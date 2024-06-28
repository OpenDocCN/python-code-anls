# `.\models\bart\tokenization_bart_fast.py`

```
# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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

import json  # 导入 json 模块，用于处理 JSON 数据
from typing import List, Optional, Tuple  # 导入用于类型提示的模块

from tokenizers import pre_tokenizers, processors  # 导入 tokenizers 库中的预处理器和处理器

from ...tokenization_utils_base import AddedToken, BatchEncoding  # 导入基础的 tokenization_utils_base 中的类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入 tokenization_utils_fast 中的 PreTrainedTokenizerFast 类
from ...utils import logging  # 导入工具类 logging

from .tokenization_bart import BartTokenizer  # 导入当前目录下的 tokenization_bart 模块中的 BartTokenizer 类


logger = logging.get_logger(__name__)  # 获取当前文件名的日志记录器对象

# 定义用于存储文件名的字典常量
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 定义预训练模型的词汇文件映射字典常量
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/vocab.json",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/vocab.json",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/vocab.json",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/vocab.json",
    },
    "merges_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/merges.txt",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/merges.txt",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/merges.txt",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/merges.txt",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/merges.txt",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/merges.txt",
    },
    {
        # 定义一个字典，映射不同的 BART 模型到它们对应的 tokenizer.json 文件的 URL
        "tokenizer_file": {
            "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/tokenizer.json",
            "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/tokenizer.json",
            "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/tokenizer.json",
            "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json",
            "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/tokenizer.json",
            "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/tokenizer.json",
        },
    }
}

# 预训练位置嵌入的大小字典，映射不同的BART模型到对应的嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/bart-base": 1024,
    "facebook/bart-large": 1024,
    "facebook/bart-large-mnli": 1024,
    "facebook/bart-large-cnn": 1024,
    "facebook/bart-large-xsum": 1024,
    "yjernite/bart_eli5": 1024,
}


class BartTokenizerFast(PreTrainedTokenizerFast):
    r"""
    构建一个“快速”BART分词器（基于HuggingFace的*tokenizers*库），派生自GPT-2分词器，使用字节级别的字节对编码。

    此分词器已经训练成将空格视为标记的一部分（类似于sentencepiece），因此一个词会根据其是否位于句子开头而编码不同：

    ```python
    >>> from transformers import BartTokenizerFast

    >>> tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    当在实例化分词器或对文本调用时传递 `add_prefix_space=True`，可以避免这种行为，但由于模型未以这种方式进行预训练，可能会导致性能下降。

    <Tip>

    当与 `is_split_into_words=True` 一起使用时，需要使用 `add_prefix_space=True` 实例化此分词器。

    </Tip>

    此分词器继承自[`PreTrainedTokenizerFast`]，该类包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。
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
            other word. (BART tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """
    # 预训练模型中的词汇文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型中的词汇文件映射表
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型中的最大输入大小列表
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表，包括输入ID和注意力掩码
    model_input_names = ["input_ids", "attention_mask"]
    # 指定慢速分词器的类为BartTokenizer

    # 初始化方法，用于创建一个新的实例
    def __init__(
        self,
        vocab_file=None,         # 词汇表文件路径，用于设置分词器的词汇表
        merges_file=None,        # 合并文件路径，用于设置分词器的合并规则
        tokenizer_file=None,     # 分词器文件路径，用于加载预训练的分词器
        errors="replace",        # 处理编码错误的方式
        bos_token="<s>",         # 开始符号
        eos_token="</s>",        # 结束符号
        sep_token="</s>",        # 分隔符号
        cls_token="<s>",         # 类别符号
        unk_token="<unk>",       # 未知符号
        pad_token="<pad>",       # 填充符号
        mask_token="<mask>",     # 掩码符号
        add_prefix_space=False,  # 是否在词前加空格
        trim_offsets=True,       # 是否修剪偏移量
        **kwargs,                # 其他关键字参数
    ):
        # 如果 mask_token 是字符串类型，则创建一个特殊的 AddedToken 对象，保证 normalized=True
        mask_token = (
            AddedToken(mask_token, lstrip=True, normalized=True, special=True)
            if isinstance(mask_token, str)
            else mask_token
        )
        # 调用父类的初始化方法，设置 tokenizer 的基本参数
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

        # 获取当前的 pre_tokenizer 状态，并检查是否需要更新 add_prefix_space
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            # 如果需要更新 add_prefix_space，则更新 pre_tokenizer 的配置
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置实例属性 add_prefix_space
        self.add_prefix_space = add_prefix_space

        # 检查并更新 tokenizer 的后处理器组件状态
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # 如果 state 中有 "sep"，将其转换为元组形式
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            # 如果 state 中有 "cls"，将其转换为元组形式
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            changes_to_apply = False

            # 检查是否需要更新 add_prefix_space
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            # 检查是否需要更新 trim_offsets
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            # 如果有需要更新的内容，则创建新的后处理器组件实例并应用更新
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)
    def mask_token(self) -> str:
        """
        `str`: 返回用于训练模型的掩码标记。如果尚未设置，则记录错误信息。

        BART 分词器具有特殊的掩码标记，用于填充掩码管道。该掩码标记会贪婪地包括 *<mask>* 前面的空格。
        """
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        重写掩码标记的默认行为，使其能够吞掉前面的空格。

        这是为了与所有之前基于 BART 的模型保持向后兼容性。
        """
        # 掩码标记表现得像普通单词，即包括前面的空格，因此我们设置 lstrip=True
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        self._mask_token = value

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        is_split_into_words = kwargs.get("is_split_into_words", False)

        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )

        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        将词汇表保存到指定的目录中。

        调用底层分词器模型的保存方法，并返回保存的文件列表。
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        为输入构建包含特殊标记的序列。

        在 token_ids_0 前加入 bos_token_id，后加入 eos_token_id。如果提供 token_ids_1，则在其前后也加入 eos_token_id。
        """
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        根据序列创建 token_type_ids。

        token_ids_0 和 token_ids_1 用于创建对应的 token_type_ids，用于区分不同的句子或片段。
        """
    # 定义一个函数，用于生成用于序列对分类任务的掩码。BART 模型不使用token type ids，因此返回一个全零列表。

    Args:
        token_ids_0 (`List[int]`):
            第一个序列的ID列表。
        token_ids_1 (`List[int]`, *optional*):
            第二个序列的ID列表，用于序列对。

    Returns:
        `List[int]`: 全零列表，长度根据输入的序列长度动态计算。
    """
    # 分隔符 token 的 ID 列表
    sep = [self.sep_token_id]
    # 类别 token 的 ID 列表
    cls = [self.cls_token_id]

    # 如果第二个序列的 ID 列表为空
    if token_ids_1 is None:
        # 返回长度为 cls + token_ids_0 + sep 组合后的列表，每个元素都是 0
        return len(cls + token_ids_0 + sep) * [0]
    
    # 如果有第二个序列的 ID 列表
    # 返回长度为 cls + token_ids_0 + sep + sep + token_ids_1 + sep 组合后的列表，每个元素都是 0
    return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```