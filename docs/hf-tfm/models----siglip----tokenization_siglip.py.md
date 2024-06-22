# `.\transformers\models\siglip\tokenization_siglip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息
""" Tokenization class for SigLIP model."""

# 导入所需的库
import os
import re
import string
import warnings
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import sentencepiece as spm

# 导入其他必要的模块
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken

# 如果是类型检查，则导入 TextInput 类
if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput
from ...utils import logging, requires_backends

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "google/siglip-base-patch16-224": "https://huggingface.co/google/siglip-base-patch16-224/resolve/main/spiece.model",
    }
}

# 预训练模型的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/siglip-base-patch16-224": 256,
}

# SentencePiece 的下划线符号
SPIECE_UNDERLINE = "▁"

# SiglipTokenizer 类，继承自 PreTrainedTokenizer
class SiglipTokenizer(PreTrainedTokenizer):
    """
    Construct a Siglip tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"</s>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
        model_max_length (`int`, *optional*, defaults to 64):
            The maximum length (in number of tokens) for model inputs.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="</s>",
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        model_max_length=64,
        do_lower_case=True,
        **kwargs,
    # 定义方法，初始化 Tokenizer 类
    ) -> None:
        # 检查是否需要后端支持 "protobuf"
        requires_backends(self, "protobuf")

        # 初始化 pad_token
        pad_token = (
            AddedToken(pad_token, rstrip=True, lstrip=True, normalized=False, special=True)
            if isinstance(pad_token, str)
            else pad_token
        )
        # 初始化 unk_token
        unk_token = (
            AddedToken(unk_token, rstrip=True, lstrip=True, normalized=False, special=True)
            if isinstance(unk_token, str)
            else unk_token
        )
        # 初始化 eos_token
        eos_token = (
            AddedToken(eos_token, rstrip=True, lstrip=True, normalized=False, special=True)
            if isinstance(eos_token, str)
            else eos_token
        )

        # 如果 sp_model_kwargs 为 None，则初始化为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置是否小写
        self.do_lower_case = do_lower_case
        self.vocab_file = vocab_file

        # 创建 sp_model 用于处理 SentencePiece
        self.sp_model = self.get_spm_processor()
        self.vocab_file = vocab_file

        # 调用父类的初始化方法，并传入相应参数
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            model_max_length=model_max_length,
            do_lower_case=do_lower_case,
            **kwargs,
        )

    # 定义方法，获取 spm 处理器
    def get_spm_processor(self):
        # 初始化 SentencePieceProcessor 实例
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从vocab_file读取序列化的模型
        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf()
            model = model_pb2.ModelProto.FromString(sp_model)
            # 设置正则化规范
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            # 加载序列化后的模型
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    @property
    # 获取词表大小
    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.vocab_size
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    # 获取词表
    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_vocab
    def get_vocab(self):
        # 转换词汇 ID 为词汇
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 更新特殊标记编码器
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 获取特殊标记掩码
    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
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
        # 如果已经有特殊标记，则直接调用父类的方法获取特殊标记掩码
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 一般情况：存在一些特殊标记
        if token_ids_1 is None:
            # 生成只含有序列标记的特殊标记掩码
            return ([0] * len(token_ids_0)) + [1]
        # 对于序列对，生成相应的特殊标记掩码
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    # 从 token_ids 列表中删除 EOS 标记，并确保不重复添加 EOS 标记
    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            # 如果最后一个 token 已经是 EOS 标记，则发出警告并直接返回 token_ids
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            # 如果最后一个 token 不是 EOS 标记，则添加 EOS 标记后返回
            return token_ids + [self.eos_token_id]

    # 创建用于序列对分类任务的 token_type_ids，T5 模型不使用 token_type_ids，因此返回全零列表
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        eos = [self.eos_token_id]

        if token_ids_1 is None:
            # 对于单个序列，返回全零列表
            return len(token_ids_0 + eos) * [0]
        # 对于序列对，返回全零列表
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    # 构建带有特殊标记的输入列表
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 添加序列结束符号（End of Sequence）到第一个序列中，如果不存在的话
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            # 如果只有一个序列，直接返回第一个序列
            return token_ids_0
        else:
            # 如果有两个序列，添加序列结束符号到第二个序列中，如果不存在的话，然后将两个序列连接返回
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    # 从transformers.models.t5.tokenization_t5.T5Tokenizer.__getstate__中复制过来的
    def __getstate__(self):
        # 复制Tokenizer对象的字典形式状态
        state = self.__dict__.copy()
        # 将特殊的SP模型设置为None
        state["sp_model"] = None
        return state

    # 从transformers.models.t5.tokenization_t5.T5Tokenizer.__setstate__中复制过来的
    def __setstate__(self, d):
        # 将Tokenizer对象的状态设置为指定的状态
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用特殊的参数初始化SP模型
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载SP模型的词汇文件
        self.sp_model.Load(self.vocab_file)

    # 移除文本中的标点符号
    def remove_punctuation(self, text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    # 来源: https://github.com/google-research/big_vision/blob/3b8e5ab6ad4f96e32b32826f9e1b8fd277914f9c/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94
    def canonicalize_text(self, text, *, keep_punctuation_exact_string=None):
        """Returns canonicalized `text` (puncuation removed).

        Args:
            text (`str`):
                String to be canonicalized.
            keep_punctuation_exact_string (`str`, *optional*):
                If provided, then this exact string is kept. For example providing '{}' will keep any occurrences of '{}'
                (but will still remove '{' and '}' that appear separately).
        """
        # 如果提供了一个特殊的字符串，将该字符串保留在文本中
        if keep_punctuation_exact_string:
            # 对文本按照特殊的字符串进行分割，然后移除标点符号，最后再用特殊字符串连接起来
            text = keep_punctuation_exact_string.join(self.remove_punctuation(part) for part in text.split(keep_punctuation_exact_string))
        else:
            # 移除文本中的标点符号
            text = self.remove_punctuation(text)
        # 替换文本中多个连续空格为一个空格，并去除首尾空格
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text
```  
    def tokenize(self, text: "TextInput", add_special_tokens=False, **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens.
        """
        # 使用父类的tokenize方法将文本转换为标记列表
        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        # 如果tokens的长度大于1并且第一个token等于SPIECE_UNDERLINE，且第二个token是特殊标记中的一个，则把第一个token去除
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        # 返回处理后的tokens
        return tokens

    @property
    # 从transformers.models.t5.tokenization_t5.T5Tokenizer.unk_token_length复制过来
    def unk_token_length(self):
        # 返回未知标记的长度
        return len(self.sp_model.encode(str(self.unk_token)))

    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE.

        For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give `['H', 'e', 'y']` instead of `['▁He', 'y']`.

        Thus we always encode `f"{unk_token}text"` and strip the `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 规范文本
        text = self.canonicalize_text(text, keep_punctuation_exact_string=None)
        # 使用sentencepiece模型对文本进行编码并以字符串形式输出
        tokens = self.sp_model.encode(text, out_type=str)

        # 1. 编码字符串+前缀 例如: "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从 ['<','unk','>', '▁Hey'] 中移除self.unk_token
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    # 从transformers.models.t5.tokenization_t5.T5Tokenizer._convert_token_to_id复制过来
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        # 使用vocab将token转换为id
        return self.sp_model.piece_to_id(token)

    # 从transformers.models.t5.tokenization_t5.T5Tokenizer._convert_id_to_token复制过来
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用vocab将id转换为token
        token = self.sp_model.IdToPiece(index)
        return token

    # 从transformers.models.t5.tokenization_t5.T5Tokenizer.convert_tokens_to_string复制过来
    # 将 tokens 序列转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 用于存储当前子 token
        current_sub_tokens = []
        # 删除手动添加的前缀空格
        tokens[0] = tokens[0].lstrip(SPIECE_UNDERLINE)
        # 初始化输出字符串
        out_string = ""
        # 初始化前一个 token 是否为特殊 token 的标志
        prev_is_special = False
        # 循环遍历 tokens
        for token in tokens:
            # 确保特殊 token 不是使用 sentencepiece 模型进行解码
            if token in self.all_special_tokens:
                # 如果前一个 token 不是特殊 token，则添加空格
                if not prev_is_special:
                    out_string += " "
                # 使用 sentencepiece 模型解码当前子 token，并添加到输出字符串中
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                # 添加当前 token 到当前子 token
                current_sub_tokens.append(token)
                prev_is_special = False
        # 使用 sentencepiece 模型解码当前子 token，并添加到输出字符串中
        out_string += self.sp_model.decode(current_sub_tokens)
        # 去除输出字符串两侧的空格并返回
        return out_string.strip()

    # 保存词汇表文件到指定目录
    # 从 transformers.models.t5.tokenization_t5.T5Tokenizer.save_vocabulary 复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则输出错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 设置输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        # 如果当前词汇表文件路径与输出路径不同，且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将 sentencepiece 模型序列化的内容写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)
        # 返回输出的词汇表文件路径
        return (out_vocab_file,)
```