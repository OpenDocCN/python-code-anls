# `.\models\code_llama\tokenization_code_llama.py`

```
# 设置文件编码为 UTF-8

# 版权声明和版权信息

# 导入必要的模块和库
import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

# 导入 sentencepiece 库，用于分词
import sentencepiece as spm

# 导入其他必要的自定义模块和函数
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging, requires_backends

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件名映射
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "hf-internal-testing/llama-code-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer.model",
    },
    "tokenizer_file": {
        "hf-internal-testing/llama-code-tokenizer": "https://huggingface.co/hf-internal-testing/llama-tokenizer/resolve/main/tokenizer_config.json",
    },
}

# 预训练模型的位置编码尺寸映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "hf-internal-testing/llama-code-tokenizer": 2048,
}

# SentencePiece 分词使用的特殊符号
SPIECE_UNDERLINE = "▁"

# 定义特殊标记，用于表示系统提示的起始和结束
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# 默认的系统提示信息，指导模型生成回复时的行为规范
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        prefix_token="▁<PRE>",
        middle_token="▁<MID>",
        suffix_token="▁<SUF>",
        eot_token="▁<EOT>",
        fill_token="<FILL_ME>",
        suffix_first=False,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        additional_special_tokens=None,
        use_default_system_prompt=False,
        **kwargs,
    ):
        # 要求依赖 protobuf 库
        requires_backends(self, "protobuf")

        # 如果未提供 sp_model_kwargs，则设置为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 如果 bos_token/eos_token/unk_token 是字符串，则创建相应的特殊 token 对象
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token

        # 设置是否使用默认系统提示
        self.use_default_system_prompt = use_default_system_prompt

        # 将特殊标记添加到 additional_special_tokens 列表中，用于跳过它们
        additional_special_tokens = additional_special_tokens or []
        for token in [prefix_token, middle_token, suffix_token, eot_token]:
            additional_special_tokens += [token] if token is not None else []

        # 初始化实例变量
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self._prefix_token = prefix_token
        self._middle_token = middle_token
        self._suffix_token = suffix_token
        self._eot_token = eot_token
        self.fill_token = fill_token
        self.suffix_first = suffix_first

        # 获取 SPM 处理器
        self.sp_model = self.get_spm_processor()

        # 调用父类初始化方法
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            prefix_token=prefix_token,
            middle_token=middle_token,
            suffix_token=suffix_token,
            eot_token=eot_token,
            fill_token=fill_token,
            sp_model_kwargs=self.sp_model_kwargs,
            suffix_first=suffix_first,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            additional_special_tokens=additional_special_tokens,
            use_default_system_prompt=use_default_system_prompt,
            **kwargs,
        )

    @property
    def unk_token_length(self):
        # 返回未知标记 token 的编码长度
        return len(self.sp_model.encode(str(self.unk_token)))
    # 返回一个 SentencePieceProcessor 对象，用于分词处理
    def get_spm_processor(self):
        # 使用给定的 sp_model_kwargs 初始化 SentencePieceProcessor 对象
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        
        # 打开并读取词汇文件，将其内容作为二进制数据加载到内存
        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            
            # 导入 protobuf 模块中的 import_protobuf 函数
            model_pb2 = import_protobuf()
            
            # 使用 protobuf 解析 sp_model，转换为 ModelProto 对象
            model = model_pb2.ModelProto.FromString(sp_model)
            
            # 创建一个 NormalizerSpec 对象，设定 add_dummy_prefix 属性为 False
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            
            # 将创建的 NormalizerSpec 对象合并到 ModelProto 对象的 normalizer_spec 中
            model.normalizer_spec.MergeFrom(normalizer_spec)
            
            # 将修改后的 ModelProto 对象序列化为字符串
            sp_model = model.SerializeToString()
            
            # 从序列化后的 ModelProto 字符串中加载 tokenizer
            tokenizer.LoadFromSerializedProto(sp_model)
        
        # 返回配置好的 tokenizer 对象
        return tokenizer

    @property
    def prefix_token(self):
        return self._prefix_token

    @property
    def prefix_id(self):
        # 如果 _prefix_token 为 None，则返回 None
        if self._prefix_token is None:
            return None
        # 否则，将 _prefix_token 转换为其对应的 id，并返回
        return self.convert_tokens_to_ids(self.prefix_token)

    @property
    def middle_token(self):
        return self._middle_token

    @property
    def middle_id(self):
        # 如果 _middle_token 为 None，则返回 None
        if self._middle_token is None:
            return None
        # 否则，将 _middle_token 转换为其对应的 id，并返回
        return self.convert_tokens_to_ids(self.middle_token)

    @property
    def suffix_token(self):
        return self._suffix_token

    @property
    def suffix_id(self):
        # 如果 _suffix_token 为 None，则返回 None
        if self._suffix_token is None:
            return None
        # 否则，将 _suffix_token 转换为其对应的 id，并返回
        return self.convert_tokens_to_ids(self.suffix_token)

    @property
    def eot_token(self):
        return self._eot_token

    @property
    def eot_id(self):
        # 如果 _eot_token 为 None，则返回 None
        if self._eot_token is None:
            return None
        # 否则，将 _eot_token 转换为其对应的 id，并返回
        return self.convert_tokens_to_ids(self.eot_token)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        # 返回 sp_model 的词汇大小，即词汇表中的词汇数量
        return self.sp_model.get_piece_size()

    # 从 transformers.models.llama.tokenization_llama.LlamaTokenizer.get_vocab 复制而来
    def get_vocab(self):
        """Returns vocab as a dict"""
        # 创建一个包含所有词汇及其对应 id 的字典 vocab
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        
        # 将 added_tokens_encoder 中的内容更新到 vocab 中
        vocab.update(self.added_tokens_encoder)
        
        # 返回构建好的词汇表字典 vocab
        return vocab
    # 将字符串 `prefix` 添加前缀空格
    def tokenize(self, prefix, suffix=None, suffix_first=False, **kwargs) -> List[int]:
        # 如果 `prefix` 中包含 `self.fill_token`，且没有 `suffix`，则拆分为 `prefix` 和 `suffix`
        if self.fill_token is not None and self.fill_token in prefix and suffix is None:
            prefix, suffix = prefix.split(self.fill_token)

        # 如果 `prefix` 长度大于 0，将 `SPIECE_UNDERLINE` 替换为空格并添加前缀 `_`
        if len(prefix) > 0:
            prefix = SPIECE_UNDERLINE + prefix.replace(SPIECE_UNDERLINE, " ")

        # 如果 `suffix` 为 None 或长度小于 1，则仅使用 `prefix` 进行分词
        if suffix is None or len(suffix) < 1:
            tokens = super().tokenize(prefix, **kwargs)
            # 如果 `tokens` 的长度大于 1 且第一个 token 是 `SPIECE_UNDERLINE`，并且第二个 token 是特殊 token 列表中的一部分，则移除第一个 token
            if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
                tokens = tokens[1:]
            return tokens

        # 对 `prefix` 进行分词，包含额外的 `SPIECE_UNDERLINE`
        prefix_tokens = self._tokenize(prefix)

        # 如果 `prefix_id`, `middle_id`, `suffix_id` 有任一为 None，则抛出 ValueError
        if None in (self.prefix_id, self.middle_id, self.suffix_id):
            raise ValueError(
                "The input either includes a `prefix` and a `suffix` used for the infilling task,"
                f"  or can be split on the {self.fill_token} token, creating a suffix and prefix,"
                " but the model does not support `infilling`."
            )

        # 对 `suffix` 进行分词，确保不会影响 CodeLlama sp 模型的结果
        suffix_tokens = self._tokenize(suffix)

        # 根据 `suffix_first` 参数决定返回的 token 排序顺序
        suffix_first = suffix_first if suffix_first is not None else self.suffix_first
        if suffix_first:
            # 格式化为 " <PRE> <SUF>{suf} <MID> {pre}"
            return [self.prefix_token, self.suffix_token] + suffix_tokens + [self.middle_token] + prefix_tokens
        else:
            # 格式化为 " <PRE> {pre} <SUF>{suf} <MID>"
            return [self.prefix_token] + prefix_tokens + [self.suffix_token] + suffix_tokens + [self.middle_token]

    # 返回经过分词后的字符串
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        We de-activated the `add_dummy_prefix` option, thus the sentencepiece internals will always strip any
        SPIECE_UNDERLINE. For example: `self.sp_model.encode(f"{SPIECE_UNDERLINE}Hey", out_type = str)` will give
        `['H', 'e', 'y']` instead of `['▁He', 'y']`. Thus we always encode `f"{unk_token}text"` and strip the
        `unk_token`. Here is an example with `unk_token = "<unk>"` and `unk_token_length = 4`.
        `self.tokenizer.sp_model.encode("<unk> Hey", out_type = str)[4:]`.
        """
        # 使用 sentencepiece 对文本进行编码，输出为字符串类型的 token 列表
        tokens = self.sp_model.encode(text, out_type=str)
        # 如果 `text` 不以 `SPIECE_UNDERLINE` 或空格开头，则直接返回 tokens
        if not text.startswith((SPIECE_UNDERLINE, " ")):
            return tokens
        # 在编码字符串前添加 `unk_token`，然后去除 `unk_token`
        tokens = self.sp_model.encode(self.unk_token + text, out_type=str)
        # 如果 `tokens` 的长度大于等于 `unk_token_length`，则去除前 `unk_token_length` 个 token；否则返回整个 tokens 列表
        return tokens[self.unk_token_length :] if len(tokens) >= self.unk_token_length else tokens

    # 从词汇表中将 token 转换为其对应的 id
    # 复制自 transformers.models.llama.tokenization_llama.LlamaTokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)
    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        # 使用 sentencepiece 模型将索引转换为对应的 token 字符串
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # 因为我们手动添加了前缀空格，所以在解码时需要去除
        if tokens[0].startswith(SPIECE_UNDERLINE):
            # 去除第一个 token 的前缀下划线
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        for _, token in enumerate(tokens):
            # 确保特殊 token 不使用 sentencepiece 模型解码
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        # 解码剩余的子 token 并添加到输出字符串中
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.save_vocabulary
    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            # 如果保存目录不存在，则记录错误并返回
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 构造输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与输出路径不同，并且当前文件是一个存在的文件，则复制当前文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将 sentencepiece 模型的序列化内容写入输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # 根据是否添加特殊 token 构建输入的 token 序列
        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            # 如果有第二个 token 序列，则连接第二个 token 序列的特殊 token 和 token_ids
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.get_special_tokens_mask
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

        # Initialize special tokens (BOS and EOS) IDs based on tokenizer settings
        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        # If only one sequence is provided
        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        
        # For sequence pairs, concatenate masks for both sequences
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.create_token_type_ids_from_sequences
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
        # Initialize BOS and EOS token IDs based on tokenizer settings
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # Initialize output list with zeros for the length of the first sequence with added special tokens
        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        # If there is a second sequence, concatenate its token type IDs after the first sequence
        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

    @property
    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.default_chat_template
    # 返回对象的状态字典，以便序列化
    def __getstate__(self):
        # 复制对象的字典属性，确保状态独立于实例
        state = self.__dict__.copy()
        # 将 sp_model 设为 None，因为不能直接序列化 SentencePieceProcessor 对象
        state["sp_model"] = None
        # 获取序列化后的 sp_model_proto 字符串表示，并存入状态字典
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        # 返回最终的状态字典
        return state

    # 根据给定的状态字典来恢复对象的状态
    def __setstate__(self, d):
        # 直接将对象的状态字典设置为传入的状态字典 d
        self.__dict__ = d
        # 使用 sp_model_kwargs 参数重新创建 sp_model 对象
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 从序列化后的 proto 字符串加载 sp_model 的状态
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)
```