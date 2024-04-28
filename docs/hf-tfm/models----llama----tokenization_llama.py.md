# `.\transformers\models\llama\tokenization_llama.py`

```
# 设置文件编码为utf-8
# 版权声明
# 基于EleutherAI的GPT-NeoX库和该库中的GPT-NeoX和OPT实现的代码。已对其进行修改，以适应与训练模型的Meta AI团队相比存在的轻微架构差异。
# 根据Apache许可证第2.0版许可，您不得使用此文件，除非遵守许可证。您可以在http://www.apache.org/licenses/LICENSE-2.0获取许可证的副本。
#
# 除非适用法律要求或以书面方式同意，否则根据许可证分发的软件是基于“按原样”的基础分发的，
# 没有任何明示或暗示的担保或条件。有关特定语言管理权限和限制的细节，请参见许可证。

"""Tokenization classes for LLaMA."""
# 导入必要的库
import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
# 导入sentencepiece模块，并重命名为spm
import sentencepiece as spm
# 导入其他必要的库，包括import_protobuf，PreTrainedTokenizer，logging
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

# 如果是类型检查阶段，则导入TextInput
if TYPE_CHECKING:
    from ...tokenization_utils_base import TextInput

# 获取logger对象
logger = logging.get_logger(__name__)

# 定义vocab文件名
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

# 预训练模型的文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}
# 预训练位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "hf-internal-testing/llama-tokenizer": 2048,
}
# 定义特殊字符
SPIECE_UNDERLINE = "▁"

# 定义特殊标记
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# 默认系统提示信息
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# 定义LLaMA Tokenizer类
class LlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.

    """
    # 定义vocab文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的文件映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练模型输入的最大限制
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    
    # 初始化函数，用于创建一个新的Tokenizer对象
    def __init__(
        self,
        vocab_file,  # 词汇表文件路径
        unk_token="<unk>",  # 未知标记，默认为"<unk>"
        bos_token="<s>",  # 开始标记，默认为"<s>"
        eos_token="</s>",  # 结束标记，默认为"</s>"
        pad_token=None,  # 填充标记，默认为None
        sp_model_kwargs: Optional[Dict[str, Any]] = None,  # SentencePiece 模型参数，默认为None
        add_bos_token=True,  # 是否添加开始标记，默认为True
        add_eos_token=False,  # 是否添加结束标记，默认为False
        clean_up_tokenization_spaces=False,  # 是否清理标记化空格，默认为False
        use_default_system_prompt=False,  # 是否使用默认系统提示，默认为False
        spaces_between_special_tokens=False,  # 特殊标记之间是否添加空格，默认为False
        legacy=None,  # 是否使用旧版本行为，默认为None
        **kwargs,  # 其他参数
    ):
        # 如果 sp_model_kwargs 为 None，则设为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 如果 bos_token 是字符串，则创建一个特殊标记对象
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        # 如果 eos_token 是字符串，则创建一个特殊标记对象
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        # 如果 unk_token 是字符串，则创建一个特殊标记对象
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token
        # 如果 pad_token 是字符串，则创建一个特殊标记对象
        pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token
    
        # 如果 legacy 为 None，则发出警告提醒用户使用旧版本行为
        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565"
            )
            legacy = True
    
        # 将传入的参数赋值给对象属性
        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        # 获取 SentencePiece 模型处理器
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
    
        # 调用父类的初始化方法
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
            **kwargs,  # 其他参数
        )
    
    # 获取未知标记的长度
    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))
    
    # 从 transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor 复制过来的
    def get_spm_processor(self, from_slow=False):
        # 创建一个 SentencePieceProcessor 对象，使用指定的参数
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 如果设置了 legacy 或者 from_slow 标志，则不依赖于 protobuf
        if self.legacy or from_slow:  # no dependency on protobuf
            # 加载预训练的 SentencePiece 模型
            tokenizer.Load(self.vocab_file)
            return tokenizer

        # 使用二进制模式打开词汇文件
        with open(self.vocab_file, "rb") as f:
            # 读取模型数据
            sp_model = f.read()
            # 通过 import_protobuf 加载 protobuf 模块
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            # 将模型数据转换为 ModelProto 对象
            model = model_pb2.ModelProto.FromString(sp_model)
            # 创建 NormalizerSpec 对象
            normalizer_spec = model_pb2.NormalizerSpec()
            # 设置 add_dummy_prefix 属性为 False
            normalizer_spec.add_dummy_prefix = False
            # 将 NormalizerSpec 对象合并到模型中
            model.normalizer_spec.MergeFrom(normalizer_spec)
            # 将模型序列化为字符串
            sp_model = model.SerializeToString()
            # 从序列化的模型数据加载 tokenizer
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    def __getstate__(self):
        # 复制对象状态
        state = self.__dict__.copy()
        # 将 sp_model 置为 None
        state["sp_model"] = None
        # 将 sp_model_proto 设置为模型的序列化数据
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        # 恢复对象状态
        self.__dict__ = d
        # 重新创建 SentencePieceProcessor 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从序列化的模型数据中加载模型
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        # 返回词汇大小
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        # 创建词汇字典，将 token 转换为 ID
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        # 将额外添加的 token 加入词汇字典
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(self, text: "TextInput", add_special_tokens=False, **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        # 如果 legacy 设置为 True 或者文本为空，则调用超类的 tokenize 方法
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        # 在文本前添加 SPIECE_UNDERLINE，并调用超类的 tokenize 方法
        tokens = super().tokenize(SPIECE_UNDERLINE + text.replace(SPIECE_UNDERLINE, " "), **kwargs)

        # 如果 tokens 长度大于 1 并且第一个 token 是 SPIECE_UNDERLINE，并且第二个 token 是特殊 token，则移除第一个 token
        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 使用 SentencePiece 模型对文本进行编码，返回字符串形式的编码结果
        tokens = self.sp_model.encode(text, out_type=str)
        # 如果文本不是以 SPIECE_UNDERLINE 或空格开头，直接返回编码结果
        if self.legacy or not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens

        # 1. 将字符串加上前缀进行编码，例如 "<unk> Hey"
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 2. 从编码结果中移除 unk_token，例如 ['
    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接保存词汇表文件的路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇表文件路径与输出路径不同并且词汇表文件存在，则复制词汇表文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果词汇表文件不存在，则序列化模型并写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 根据是否添加bos和eos特殊标记对token_ids进行处理
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # 构建带有特殊标记的输入序列
        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            # 如果存在第二个序列，则合并第二个序列及其特殊标记到输出序列中
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

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
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []  # Create a list with bos_token_id value if add_bos_token is True
        eos_token_id = [1] if self.add_eos_token else []  # Create a list with eos_token_id value if add_eos_token is True

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id  # Return the list containing bos_token_id, sequence tokens, and eos_token_id
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
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []  # Create a list with bos_token_id value if add_bos_token is True
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []  # Create a list with eos_token_id value if add_eos_token is True

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)  # Create a mask with 0s for the first sequence

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)  # Add 1s to the mask for the second sequence

        return output

    @property
```  
```