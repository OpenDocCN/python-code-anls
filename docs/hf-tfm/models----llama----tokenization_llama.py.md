# `.\models\llama\tokenization_llama.py`

```py
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

"""Tokenization classes for LLaMA."""
import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import sentencepiece as spm  # 导入 sentencepiece 库，用于处理分词

from ...convert_slow_tokenizer import import_protobuf  # 导入从 protobuf 格式转换的函数
from ...tokenization_utils import AddedToken, PreTrainedTokenizer  # 导入自定义的 token 类和预训练的 tokenizer 类
from ...utils import logging  # 导入日志工具模块


if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput  # 导入类型检查时所需的 TextInput 类型

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}  # 词汇表文件名映射

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "hf-internal-testing/llama-tokenizer": 2048,  # 预训练位置嵌入大小
}
SPIECE_UNDERLINE = "▁"  # 分词符号

B_INST, E_INST = "[INST]", "[/INST]"  # 实例开始和结束标记
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"  # 系统开始和结束标记

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class LlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.

    """

    vocab_files_names = VOCAB_FILES_NAMES  # 设置词汇表文件名
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇文件映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 最大模型输入尺寸映射
    # 定义模型输入的名称列表，包含 "input_ids" 和 "attention_mask"
    model_input_names = ["input_ids", "attention_mask"]
    
    # 初始化函数，用于初始化一个新的Tokenizer对象
    def __init__(
        self,
        vocab_file,  # 词汇表文件路径
        unk_token="<unk>",  # 未知token的表示，默认为"<unk>"
        bos_token="<s>",  # 起始token的表示，默认为"<s>"
        eos_token="</s>",  # 终止token的表示，默认为"</s>"
        pad_token=None,  # 填充token的表示，默认为None
        sp_model_kwargs: Optional[Dict[str, Any]] = None,  # SentencePiece模型的参数字典，默认为None
        add_bos_token=True,  # 是否添加起始token，默认为True
        add_eos_token=False,  # 是否添加终止token，默认为False
        clean_up_tokenization_spaces=False,  # 是否清理token化空格，默认为False
        use_default_system_prompt=False,  # 是否使用默认系统提示，默认为False
        spaces_between_special_tokens=False,  # 特殊token之间是否有空格，默认为False
        legacy=None,  # 是否使用旧版行为，默认为None
        add_prefix_space=True,  # 特殊token前是否添加空格，默认为True
        **kwargs,  # 其他关键字参数
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs  # 如果sp_model_kwargs为None，则设为空字典
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token  # 如果bos_token是字符串，则创建一个特殊的AddedToken对象
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token  # 如果eos_token是字符串，则创建一个特殊的AddedToken对象
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token  # 如果unk_token是字符串，则创建一个特殊的AddedToken对象
        pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token  # 如果pad_token是字符串，则创建一个特殊的AddedToken对象
    
        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565"
            )
            legacy = True  # 如果legacy为None，则设置为True
    
        self.legacy = legacy  # 将legacy属性设置为传入的legacy值
        self.vocab_file = vocab_file  # 设置词汇表文件路径
        self.add_bos_token = add_bos_token  # 设置是否添加起始token
        self.add_eos_token = add_eos_token  # 设置是否添加终止token
        self.use_default_system_prompt = use_default_system_prompt  # 设置是否使用默认系统提示
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))  # 获取SentencePiece模型处理器
        self.add_prefix_space = add_prefix_space  # 设置特殊token前是否添加空格
    
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
    
    # 属性装饰器，返回未知token在SentencePiece模型中的长度
    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))
    
    # 从transformers库中复制而来的函数，用于获取SentencePiece模型处理器
    # 获取 SentencePieceProcessor 对象用于处理文本
    def get_spm_processor(self, from_slow=False):
        # 根据给定的参数初始化 SentencePieceProcessor 对象
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        
        # 如果设置了 legacy 或者 from_slow 标志，不依赖于 protobuf，直接从文件加载词汇表
        if self.legacy or from_slow:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer

        # 否则，从文件中读取序列化后的 protobuf 模型，并进行必要的配置
        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            # 动态导入 protobuf 模块，并使用其对应的模型类进行反序列化
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            # 配置模型的规范化器
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            # 序列化模型为字符串，并加载到 tokenizer 中
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        
        return tokenizer

    # 序列化对象时需要调用的方法，保存对象的状态
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None  # 清空 sp_model，因为其无法直接序列化
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()  # 保存序列化后的模型 proto
        return state

    # 反序列化对象时需要调用的方法，加载对象的状态
    def __setstate__(self, d):
        self.__dict__ = d  # 恢复对象的属性
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)  # 重新创建 sp_model 对象
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)  # 从保存的 proto 数据中加载模型

    # 返回词汇表的大小
    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    # 返回词汇表作为字典
    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 根据指定的文本进行分词，返回 token 列表
    # 该方法源自 transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(self, text: "TextInput", **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        # 如果 legacy 设置为 True 或者文本长度为 0，则调用父类的 tokenize 方法处理
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        # 将特殊符号 SPIECE_UNDERLINE 替换为空格，并根据 add_prefix_space 设置添加前缀空格
        text = text.replace(SPIECE_UNDERLINE, " ")
        if self.add_prefix_space:
            text = SPIECE_UNDERLINE + text

        # 调用父类的 tokenize 方法获取 token 列表
        tokens = super().tokenize(text, **kwargs)

        # 如果 tokens 长度大于 1 并且第一个 token 是 SPIECE_UNDERLINE，并且第二个 token 是特殊 token，则去掉第一个 token
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    # 该方法复制自 transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 使用 sentencepiece 模型对文本进行编码，返回字符串形式的编码结果
        tokens = self.sp_model.encode(text, out_type=str)
        
        # 如果是传统模式或者文本不以 SPIECE_UNDERLINE 或空格开头，则直接返回编码结果
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. 对带有未知标记的文本进行编码，例如 "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从编码结果中移除 unk_token，例如 ['<','unk','>', '▁Hey'] 中移除 '<unk>'
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用词汇表将 token 转换为对应的 id
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用词汇表将 index 转换为对应的 token
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 因为我们手动添加了前缀空格，所以在解码时需要移除
        if tokens[0].startswith(SPIECE_UNDERLINE) and self.add_prefix_space:
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # 确保特殊标记不会使用 sentencepiece 模型进行解码
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0 and self.legacy:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string
    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 组合输出的词汇文件路径，如果指定了文件名前缀，则加在文件名之前
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇文件的绝对路径不等于输出路径的绝对路径，并且当前词汇文件存在，则复制文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇文件不存在，则将序列化后的模型写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回保存的文件路径的元组
        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # 构建包含特殊令牌的输入列表
        output = bos_token_id + token_ids_0 + eos_token_id

        # 如果存在第二个输入列表，则将其也加入到输出中
        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
        # 此处将创建一个特殊令牌掩码，用于标记特殊令牌的位置
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        # Check if the token list already has special tokens
        if already_has_special_tokens:
            # If yes, delegate to the parent class's method to get special tokens mask
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Determine the beginning of sentence (bos) and end of sentence (eos) token IDs
        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        # If token_ids_1 is not provided, return mask for single sequence
        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        
        # Return mask for sequence pairs
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        # Determine the beginning of sentence (bos) and end of sentence (eos) token IDs
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # Initialize the output list with zeros based on the length of the sequences with added tokens
        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        # If token_ids_1 is provided, extend the output list to accommodate the second sequence
        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output
```