# `.\models\longformer\tokenization_longformer_fast.py`

```
# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
"""Fast Tokenization classes for Longformer."""
# 导入需要的模块
import json
from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers, processors

# 导入基础的 tokenization 类和 fast tokenization 类
from ...tokenization_utils_base import AddedToken, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging

# 导入 Longformer 的 tokenizer 类
from .tokenization_longformer import LongformerTokenizer

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义文件名常量
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/vocab.json",
        "allenai/longformer-large-4096": (
            "https://huggingface.co/allenai/longformer-large-4096/resolve/main/vocab.json"
        ),
        "allenai/longformer-large-4096-finetuned-triviaqa": (
            "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/vocab.json"
        ),
        "allenai/longformer-base-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/vocab.json"
        ),
        "allenai/longformer-large-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/merges.txt",
        "allenai/longformer-large-4096": (
            "https://huggingface.co/allenai/longformer-large-4096/resolve/main/merges.txt"
        ),
        "allenai/longformer-large-4096-finetuned-triviaqa": (
            "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/merges.txt"
        ),
        "allenai/longformer-base-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/merges.txt"
        ),
        "allenai/longformer-large-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/merges.txt"
        ),
    },
    # 定义一个字典，存储各种模型的 tokenizer 文件及其对应的 URL
    "tokenizer_file": {
        # AllenAI Longformer Base 4096 模型的 tokenizer 文件及 URL
        "allenai/longformer-base-4096": (
            "https://huggingface.co/allenai/longformer-base-4096/resolve/main/tokenizer.json"
        ),
        # AllenAI Longformer Large 4096 模型的 tokenizer 文件及 URL
        "allenai/longformer-large-4096": (
            "https://huggingface.co/allenai/longformer-large-4096/resolve/main/tokenizer.json"
        ),
        # AllenAI Longformer Large 4096 在 TriviaQA 数据集上微调的 tokenizer 文件及 URL
        "allenai/longformer-large-4096-finetuned-triviaqa": (
            "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/tokenizer.json"
        ),
        # AllenAI Longformer Base 4096 的额外位置嵌入模型的 tokenizer 文件及 URL
        "allenai/longformer-base-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/tokenizer.json"
        ),
        # AllenAI Longformer Large 4096 的额外位置嵌入模型的 tokenizer 文件及 URL
        "allenai/longformer-large-4096-extra.pos.embd.only": (
            "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练位置嵌入大小的映射，将模型名称映射到其预训练位置嵌入的长度
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "allenai/longformer-base-4096": 4096,
    "allenai/longformer-large-4096": 4096,
    "allenai/longformer-large-4096-finetuned-triviaqa": 4096,
    "allenai/longformer-base-4096-extra.pos.embd.only": 4096,
    "allenai/longformer-large-4096-extra.pos.embd.only": 4096,
}

# 从transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast中复制而来，
# 用于从FacebookAI/roberta-base转换为allenai/longformer-base-4096，RoBERTa转换为Longformer全大小写，Roberta转换为Longformer
class LongformerTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Longformer tokenizer (backed by HuggingFace's *tokenizers* library), derived from the GPT-2
    tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import LongformerTokenizerFast

    >>> tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.
    ```
    # 定义函数参数说明
    Args:
        vocab_file (`str`):
            # 词汇表文件的路径。
            Path to the vocabulary file.
        merges_file (`str`):
            # 合并文件的路径。
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            # 解码字节为 UTF-8 时的错误处理策略。详见 bytes.decode 的说明文档。
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            # 预训练过程中用作序列开头的特殊标记。
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            # 序列的结束标记。

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            # 分隔符标记，在构建多序列时使用，例如序列分类或问答时的文本和问题。
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            # 分类器标记，在序列分类时使用（整体序列分类而不是每个标记的分类）。
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            # 未知标记，词汇表中不存在的标记将被设置为此标记。
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            # 填充标记，在批处理不同长度序列时使用。
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            # 掩码标记，用于掩码语言建模训练。
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            # 是否在输入前添加初始空格，用于长序列处理。
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Longformer tokenizer detect beginning of words by the preceding space).
        trim_offsets (`bool`, *optional*, defaults to `True`):
            # 后处理步骤是否应修剪偏移量以避免包含空格。
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """

    # 从预定义的常量中获取相关文件名、映射和模型输入大小信息
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表，包括输入的标识和注意力掩码
    model_input_names = ["input_ids", "attention_mask"]
    # 慢速分词器的类，使用 LongformerTokenizer
    slow_tokenizer_class = LongformerTokenizer

    # 初始化函数，用于设置 tokenizer 的各种参数和状态
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
        # 如果 mask_token 是字符串，则创建一个 AddedToken 对象
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )
        # 调用父类的初始化方法，设置 tokenizer 的基本参数和文件路径
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

        # 获取 backend_tokenizer 的预处理器状态，并根据传入的参数进行调整
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            # 更新预处理器的类型和参数
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置当前对象的 add_prefix_space 属性
        self.add_prefix_space = add_prefix_space

        # 获取 tokenizer 的后处理器组件，并根据状态进行调整
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # 确保 'sep' 和 'cls' 的值为元组，以便于 post_processor_class 的对象处理
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            changes_to_apply = False

            # 检查是否需要更新 add_prefix_space 和 trim_offsets 参数
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            # 如果有改变，则更新后处理器的类型和参数
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

    @property
    def mask_token(self) -> str:
        """
        `str`: 返回掩码标记，用于在进行掩码语言建模训练时使用。如果在未设置的情况下使用，则记录错误日志。

        Longformer 分词器有一个特殊的掩码标记，可在填充掩码流程中使用。该掩码标记将贪婪地包括 *<mask>* 前面的空格。
        """
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        重写掩码标记的默认行为，使其能够吸收其前的空格。

        这是为了与基于 Longformer 的所有先前使用的模型保持向后兼容性。
        """
        # 掩码标记表现得像普通单词，即包括前面的空格，因此设置 lstrip 为 True
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
        将词汇表保存到指定目录，并返回保存的文件名。

        调用分词器模型的保存方法，将模型保存到指定目录中，并使用指定的文件名前缀。
        """
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        根据给定的 token_ids 构建包含特殊标记的输入序列。

        将序列开始标记 (bos_token_id)、序列 0 的 token_ids 和序列结束标记 (eos_token_id) 拼接成输出列表。
        如果存在第二个序列 (token_ids_1)，则将其与序列 0 后面的 eos_token_id 拼接。
        """
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        """
        根据给定的 token_ids 构建 token 类型 ID。

        用于指示每个 token 属于哪个序列的标识符，通常用于区分两个不同序列的 token。
        """
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. Longformer does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        # 定义分隔符和分类符的列表
        sep = [self.sep_token_id]  # 包含特殊分隔符的列表
        cls = [self.cls_token_id]  # 包含特殊分类符的列表

        # 如果没有第二个序列，返回第一个序列加上特殊符号的长度的零列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        # 如果有第二个序列，返回两个序列以及特殊符号的长度的零列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```