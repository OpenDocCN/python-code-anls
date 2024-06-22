# `.\transformers\models\t5\tokenization_t5.py`

```py
# 定义编码为 UTF-8
# 版权声明及许可信息，本代码遵循 Apache 许可证 2.0 版本
""" T5 模型的分词类。"""

# 导入所需的库和模块
import os  # 导入操作系统相关的功能
import re  # 导入正则表达式模块
import warnings  # 导入警告模块
from shutil import copyfile  # 导入复制文件功能
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple  # 导入类型相关的功能

import sentencepiece as spm  # 导入 SentencePiece 分词模块

from ...convert_slow_tokenizer import import_protobuf  # 从转换慢速分词器模块中导入 import_protobuf 函数
from ...tokenization_utils import PreTrainedTokenizer  # 从分词工具模块中导入 PreTrainedTokenizer 类
from ...tokenization_utils_base import AddedToken  # 从基础分词工具模块中导入 AddedToken 类

# 如果类型检查开启，则导入 TextInput 类
if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput

# 导入日志模块
from ...utils import logging

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件名字典
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 定义预训练模型词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "t5-small": "https://huggingface.co/t5-small/resolve/main/spiece.model",
        "t5-base": "https://huggingface.co/t5-base/resolve/main/spiece.model",
        "t5-large": "https://huggingface.co/t5-large/resolve/main/spiece.model",
        "t5-3b": "https://huggingface.co/t5-3b/resolve/main/spiece.model",
        "t5-11b": "https://huggingface.co/t5-11b/resolve/main/spiece.model",
    }
}

# TODO(PVP) - 在 Transformers v5 中应该移除此部分
# 定义预训练位置嵌入大小字典
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "t5-small": 512,
    "t5-base": 512,
    "t5-large": 512,
    "t5-3b": 512,
    "t5-11b": 512,
}

# 定义 SentencePiece 分词器的特殊字符下划线
SPIECE_UNDERLINE = "▁"

# 定义 T5Tokenizer 类，继承自 PreTrainedTokenizer 类
class T5Tokenizer(PreTrainedTokenizer):
    """
    构建一个 T5 分词器，基于 SentencePiece。
    
    此分词器继承自 PreTrainedTokenizer，其中包含了大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。

    属性:
        sp_model (`SentencePieceProcessor`):
            用于每次转换（字符串、标记和 ID）的 SentencePiece 处理器。
    """

    # 定义词汇文件名字典
    vocab_files_names = VOCAB_FILES_NAMES
    # 定义预训练模型词汇文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义最大模型输入大小字典
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入名字列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化函数
    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        legacy=None,
        **kwargs,
    # 设置参数类型为 None 的函数返回值为 None
    ) -> None:
        # 如果 pad_token 是字符串，则将其封装成特殊的 AddedToken，否则保持原样
        pad_token = AddedToken(pad_token, special=True) if isinstance(pad_token, str) else pad_token
        # 如果 unk_token 是字符串，则将其封装成特殊的 AddedToken，否则保持原样
        unk_token = AddedToken(unk_token, special=True) if isinstance(unk_token, str) else unk_token
        # 如果 eos_token 是字符串，则将其封装成特殊的 AddedToken，否则保持原样
        eos_token = AddedToken(eos_token, special=True) if isinstance(eos_token, str) else eos_token

        # 如果 sp_model_kwargs 为 None，则设置为空字典，否则保持原样
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 设置属性值
        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        # 使用给定参数初始化 SentencePieceProcessor
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 处理额外的特殊 token
        if additional_special_tokens is not None:
            extra_tokens = [x for x in additional_special_tokens if "<extra_id_" in str(x)]
            if len(extra_tokens) < 1:
                additional_special_tokens += [f"<extra_id_{i}>" for i in range(extra_ids)]
            elif extra_ids > 0 and extra_ids != len(extra_tokens):
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids"
                    " tokens"
                )
        else:
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            additional_special_tokens = extra_tokens

        # 为了向后兼容，保留这里的代码，之后会移除并更新测试
        self._added_tokens_decoder = {}
        for i in range(len(extra_tokens)):
            self._added_tokens_decoder[len(self.sp_model) - 1 + extra_ids - i] = AddedToken(
                f"<extra_id_{i}>", single_word=False, lstrip=True, rstrip=True, special=True, normalized=False
            )

        # 处理 legacy 参数
        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565"
            )
            legacy = True

        # 设置 legacy 属性值
        self.legacy = legacy
        # 从缓慢加载获取 spm_processor
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        self.vocab_file = vocab_file
        self._extra_ids = extra_ids

        # 调用父类初始化函数
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            legacy=legacy,
            **kwargs,
        )
    # 从transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor中复制代码
    def get_spm_processor(self, from_slow=False):
        # 根据传入的sp_model_kwargs参数创建SentencePieceProcessor对象
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 如果self.legacy为True或者from_slow为True，则不依赖protobuf
        if self.legacy or from_slow:
            # 加载self.vocab_file，返回tokenizer对象
            tokenizer.Load(self.vocab_file)
            return tokenizer

        with open(self.vocab_file, "rb") as f:
            # 读取文件内容
            sp_model = f.read()
            # 导入protobuf模块
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            # 将读取的内容传入ModelProto对象中
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            # 从序列化的protobuf中加载模型到tokenizer对象
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    # 最终纠正T5模型的最大长度
    @staticmethod
    def _eventually_correct_t5_max_length(pretrained_model_name_or_path, max_model_length, init_max_model_length):
        # 如果pretrained_model_name_or_path在T5Tokenizer.max_model_input_sizes中
        if pretrained_model_name_or_path in T5Tokenizer.max_model_input_sizes:
            deprecated_max_model_length = T5Tokenizer.max_model_input_sizes[pretrained_model_name_or_path]
            # 如果init_max_model_length不为空且与max_model_length不相等，则返回init_max_model_length
            if init_max_model_length is not None and init_max_model_length != max_model_length:
                return init_max_model_length
            # 如果init_max_model_length为空
            elif init_max_model_length is None:
                # 引发警告，提醒即将在Transformers v5中修正此问题
                warnings.warn(
                    "This tokenizer was incorrectly instantiated with a model max length of"
                    f" {deprecated_max_model_length} which will be corrected in Transformers v5.\nFor now, this"
                    " behavior is kept to avoid breaking backwards compatibility when padding/encoding with"
                    " `truncation is True`.\n- Be aware that you SHOULD NOT rely on"
                    f" {pretrained_model_name_or_path} automatically truncating your input to"
                    f" {deprecated_max_model_length} when padding/encoding.\n- If you want to encode/pad to sequences"
                    f" longer than {deprecated_max_model_length} you can either instantiate this tokenizer with"
                    " `model_max_length` or pass `max_length` when encoding/padding.\n- To avoid this warning, please"
                    " instantiate this tokenizer with `model_max_length` set to your preferred value.",
                    FutureWarning,
                )

        return max_model_length

    # 返回vocab_size属性值
    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    # 获取词汇表
    def get_vocab(self):
        # 创建vocab字典存储token和对应的id
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 更新vocab字典中的特殊词汇
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 获取特殊token的mask
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
        # If the token list already has special tokens, call the superclass method to retrieve special token masks
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # normal case: some special tokens
        # If token_ids_1 is None, it means there is only one sequence; return a list of zeros for token_ids_0 length + 1 for special token
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        # If token_ids_1 is not None, return a list of zeros for token_ids_0 length + 1 for special token + zeros for token_ids_1 length + 1 for special token
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def get_sentinel_tokens(self):
        """
        Get the list of sentinel tokens from additional special tokens.

        Returns:
            `List[str]`: List of sentinel tokens.
        """
        return list(
            set(filter(lambda x: bool(re.search(r"<extra_id_\d+>", x)) is not None, self.additional_special_tokens))
        )

    def get_sentinel_token_ids(self):
        """
        Get the token IDs of sentinel tokens.

        Returns:
            `List[int]`: List of sentinel token IDs.
        """
        return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]

    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """
        Add an end-of-sequence token if it's not already present in the token IDs.

        Args:
            token_ids (`List[int]`): List of token IDs.

        Returns:
            `List[int]`: List of token IDs with end-of-sequence token added if not present.
        """
        # Check if the last token is already an end-of-sequence token
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            # Warn the user if an end-of-sequence token is already present
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            # Add end-of-sequence token to the token IDs if it's not present
            return token_ids + [self.eos_token_id]

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
        # Define end-of-sequence token list
        eos = [self.eos_token_id]

        # If token_ids_1 is None, return a list of zeros with length equal to token_ids_0 length + 1 for end-of-sequence token
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        # If token_ids_1 is not None, return a list of zeros with length equal to token_ids_0 length + 1 for end-of-sequence token + zeros for token_ids_1 length + 1 for end-of-sequence token
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: List[int] = None) -> List[int]:
        """
        从一个序列或者一对序列中构建用于序列分类任务的模型输入，通过连接和添加特殊标记。序列的格式如下:

        - 单个序列: `X </s>`
        - 一对序列: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                需要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                第二个序列的 ID 列表。

        Returns:
            `List[int]`: 包含适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # 为向后兼容性添加
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    # 从 transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize 复制而来
    def tokenize(self, text: "TextInput", add_special_tokens=False, **kwargs) -> List[str]:
        """
        将字符串转换为标记列表。如果 `self.legacy` 设置为 `False`，除非第一个标记是特殊的，否则会添加前缀标记。
        """
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))
    def _tokenize(self, text, **kwargs):
        """
        返回一个分词后的字符串。

        我们禁用了 `add_dummy_prefix` 选项，因此 sentencepiece 内部始终会去掉任何 SPIECE_UNDERLINE。
        例如：`self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` 将返回 `['H', 'e', 'y']` 而不是 `['▁He', 'y']`。
        因此我们总是编码 `f"{unk_token}text"` 并去掉 `unk_token`。以下是一个示例，其中 `unk_token = "<unk>"`， `unk_token_length = 4`。
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`。
        """
        tokens = self.sp_model.encode(text, out_type=str)
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. 编码字符串 + 前缀，例如："<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从 ['<','unk','>', '▁Hey'] 中删除 self.unk_token
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    def _convert_token_to_id(self, token):
        """使用词汇表将令牌（str）转换为ID。"""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """使用词汇表将索引（整数）转换为令牌（str）。"""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """将一系列令牌（字符串）转换为单个字符串。"""
        current_sub_tokens = []
        # 由于我们手动添加前缀空格，因此我们必须将其去除
        tokens[0] = tokens[0].lstrip(SPIECE_UNDERLINE)
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊令牌不使用 sentencepiece 模型进行解码
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string.strip()
    # 保存词汇表到指定目录下
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则输出错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构建输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果输入词汇表文件路径与输出词汇表文件路径不相同且输入词汇表文件存在，则将输入文件复制到输出文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果输入词汇表文件不存在，则将序列化后的词汇表模型写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出词汇表文件路径
        return (out_vocab_file,)
```