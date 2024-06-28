# `.\models\marian\tokenization_marian.py`

```
# 导入所需的模块和库
import json  # 导入处理 JSON 数据的模块
import os  # 导入操作系统功能的模块
import re  # 导入正则表达式模块
import warnings  # 导入警告处理模块
from pathlib import Path  # 导入路径操作相关的模块
from shutil import copyfile  # 导入文件复制功能的模块
from typing import Any, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的模块

import sentencepiece  # 导入句子拼接（SentencePiece）模块

from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器基类
from ...utils import logging  # 导入日志记录模块


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例

# 定义用于保存词汇文件的名称映射
VOCAB_FILES_NAMES = {
    "source_spm": "source.spm",
    "target_spm": "target.spm",
    "vocab": "vocab.json",
    "target_vocab_file": "target_vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}

# 定义预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "source_spm": {
        "Helsinki-NLP/opus-mt-en-de": "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/source.spm"
    },
    "target_spm": {
        "Helsinki-NLP/opus-mt-en-de": "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/target.spm"
    },
    "vocab": {
        "Helsinki-NLP/opus-mt-en-de": "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/vocab.json"
    },
    "tokenizer_config_file": {
        "Helsinki-NLP/opus-mt-en-de": (
            "https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/tokenizer_config.json"
        )
    },
}

# 定义预训练位置嵌入大小的映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"Helsinki-NLP/opus-mt-en-de": 512}
PRETRAINED_INIT_CONFIGURATION = {}

SPIECE_UNDERLINE = "▁"

# MarianTokenizer 类，继承自 PreTrainedTokenizer
class MarianTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Marian tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    # 词汇文件的名称列表，用于指定每种语言的词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射，将语言名称映射到其对应的预训练模型的词汇文件路径
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 将预训练模型初始化配置复制给实例变量
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 将预训练位置编码的最大输入大小复制给实例变量
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入的名称列表，包括 "input_ids" 和 "attention_mask"
    model_input_names = ["input_ids", "attention_mask"]
    # 用于匹配语言代码的正则表达式对象，匹配形式为 ">>.+<<"
    language_code_re = re.compile(">>.+<<")  # type: re.Pattern

    def __init__(
        self,
        source_spm,
        target_spm,
        vocab,
        target_vocab_file=None,
        source_lang=None,
        target_lang=None,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=512,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        separate_vocabs=False,
        **kwargs,
    ) -> None:
        # 如果没有提供 sp_model_kwargs，则将其设置为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 断言源 SentencePiece 模型文件存在
        assert Path(source_spm).exists(), f"cannot find spm source {source_spm}"

        # 标记是否使用分离的词汇表
        self.separate_vocabs = separate_vocabs
        # 加载 JSON 格式的词汇表文件，并将其赋给实例变量 encoder
        self.encoder = load_json(vocab)
        # 如果 unk_token 不在 encoder 中，则抛出 KeyError 异常
        if str(unk_token) not in self.encoder:
            raise KeyError("<unk> token must be in the vocab")
        # 断言 pad_token 在 encoder 中
        assert str(pad_token) in self.encoder

        # 如果使用分离的词汇表
        if separate_vocabs:
            # 加载目标语言的词汇表文件，并将其赋给实例变量 target_encoder
            self.target_encoder = load_json(target_vocab_file)
            # 创建一个从值到键的映射，赋给实例变量 decoder
            self.decoder = {v: k for k, v in self.target_encoder.items()}
            # 初始化支持的语言代码列表为空
            self.supported_language_codes = []
        else:
            # 创建一个从值到键的映射，赋给实例变量 decoder
            self.decoder = {v: k for k, v in self.encoder.items()}
            # 初始化支持的语言代码列表，仅包含符合特定格式的语言代码
            self.supported_language_codes: list = [k for k in self.encoder if k.startswith(">>") and k.endswith("<<")]

        # 设置源语言和目标语言
        self.source_lang = source_lang
        self.target_lang = target_lang
        # 保存源和目标 SentencePiece 模型文件路径
        self.spm_files = [source_spm, target_spm]

        # 加载 SentencePiece 模型，用于预处理
        self.spm_source = load_spm(source_spm, self.sp_model_kwargs)
        self.spm_target = load_spm(target_spm, self.sp_model_kwargs)
        # 当前使用的 SentencePiece 模型，默认为源语言对应的模型
        self.current_spm = self.spm_source
        # 当前使用的编码器，默认为 encoder
        self.current_encoder = self.encoder

        # 设置规范化器
        self._setup_normalizer()

        super().__init__(
            # bos_token=bos_token, 未使用。开始解码时使用配置中的 decoder_start_token_id
            source_lang=source_lang,
            target_lang=target_lang,
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
            model_max_length=model_max_length,
            sp_model_kwargs=self.sp_model_kwargs,
            target_vocab_file=target_vocab_file,
            separate_vocabs=separate_vocabs,
            **kwargs,
        )

    def _setup_normalizer(self):
        try:
            # 尝试导入 MosesPunctNormalizer，并设置为实例方法 self.punc_normalizer
            from sacremoses import MosesPunctNormalizer

            self.punc_normalizer = MosesPunctNormalizer(self.source_lang).normalize
        except (ImportError, FileNotFoundError):
            # 如果导入失败，给出警告信息，并设置一个简单的标识符匿名函数作为 self.punc_normalizer
            warnings.warn("Recommended: pip install sacremoses.")
            self.punc_normalizer = lambda x: x
    def normalize(self, x: str) -> str:
        """规范化输入字符串，处理空字符串的情况。如果输入为空字符串，则返回空字符串。"""
        return self.punc_normalizer(x) if x else ""

    def _convert_token_to_id(self, token):
        """将给定的 token 转换为对应的 ID。如果 token 不在当前编码器中，则使用未知 token 的 ID。"""
        return self.current_encoder.get(token, self.current_encoder[self.unk_token])

    def remove_language_code(self, text: str):
        """移除文本中的语言代码，例如 >>fr<<，以便后续处理使用 SentencePiece。"""
        match = self.language_code_re.match(text)
        code: list = [match.group(0)] if match else []
        return code, self.language_code_re.sub("", text)

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行标记化处理，返回标记化后的列表。"""
        code, text = self.remove_language_code(text)
        pieces = self.current_spm.encode(text, out_type=str)
        return code + pieces

    def _convert_id_to_token(self, index: int) -> str:
        """使用解码器将索引（整数）转换为对应的 token（字符串）。"""
        return self.decoder.get(index, self.unk_token)

    def batch_decode(self, sequences, **kwargs):
        """
        将一组 token ID 列表转换为字符串列表，通过调用 decode 方法实现。

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                tokenized input ids 的列表。可以使用 `__call__` 方法获得。
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                是否在解码时移除特殊 token。
            clean_up_tokenization_spaces (`bool`, *optional*):
                是否清理 tokenization 空格。如果为 `None`，则默认使用 `self.clean_up_tokenization_spaces`。
            use_source_tokenizer (`bool`, *optional*, defaults to `False`):
                是否使用源 tokenizer 解码序列（仅适用于序列到序列问题）。
            kwargs (additional keyword arguments, *optional*):
                将传递给底层模型特定的解码方法。

        Returns:
            `List[str]`: 解码后的句子列表。
        """
        return super().batch_decode(sequences, **kwargs)
    def decode(self, token_ids, **kwargs):
        """
        Converts a sequence of ids into a string using the tokenizer and vocabulary,
        with options to remove special tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            use_source_tokenizer (`bool`, *optional*, defaults to `False`):
                Whether or not to use the source tokenizer to decode sequences (only applicable in sequence-to-sequence
                problems).
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        return super().decode(token_ids, **kwargs)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Uses the source SentencePiece model (`spm_source`) if `_decode_use_source_tokenizer` is True,
        otherwise uses the target model (`spm_target`) to convert tokens back into a string.

        Args:
            tokens (List[str]): List of tokens to be converted into a string.

        Returns:
            str: The reconstructed string from tokens.
        """
        # Determine whether to use the source or target tokenizer based on the flag `_decode_use_source_tokenizer`
        sp_model = self.spm_source if self._decode_use_source_tokenizer else self.spm_target
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # Check if the token is a special token that should not be decoded using SentencePiece
            if token in self.all_special_tokens:
                # Decode accumulated sub-tokens and append the current special token with a space
                out_string += sp_model.decode_pieces(current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                # Accumulate tokens that are not special tokens
                current_sub_tokens.append(token)
        # Final decode of remaining sub-tokens and replace SentencePiece underline with a space
        out_string += sp_model.decode_pieces(current_sub_tokens)
        out_string = out_string.replace(SPIECE_UNDERLINE, " ")
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """
        Builds model inputs from token ids by appending `eos_token_id` to token_ids_0 or to both token_ids_0 and token_ids_1.

        Args:
            token_ids_0 (List[int]): List of token ids for the first sequence.
            token_ids_1 (List[int], optional): List of token ids for the second sequence, if processing pairs.

        Returns:
            List[int]: Concatenated list of token ids with `eos_token_id` appended.
        """
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        else:
            # If processing pairs, concatenate token_ids_0 and token_ids_1 with eos_token_id appended
            return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def _switch_to_input_mode(self):
        """
        Sets current SentencePiece model (`current_spm`) and encoder (`current_encoder`) to use the source versions.
        """
        self.current_spm = self.spm_source
        self.current_encoder = self.encoder

    def _switch_to_target_mode(self):
        """
        Sets current SentencePiece model (`current_spm`) to use the target version (`spm_target`).
        Sets current encoder (`current_encoder`) to use `target_encoder` if `separate_vocabs` is True.
        """
        self.current_spm = self.spm_target
        if self.separate_vocabs:
            self.current_encoder = self.target_encoder
    # 返回编码器的大小，即其包含的词汇量大小
    def vocab_size(self) -> int:
        return len(self.encoder)

    # 将词汇保存到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        saved_files = []

        # 如果设置了分离的词汇表选项
        if self.separate_vocabs:
            # 构建源语言词汇表文件名
            out_src_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"],
            )
            # 构建目标语言词汇表文件名
            out_tgt_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["target_vocab_file"],
            )
            # 分别保存源语言和目标语言的编码器到JSON文件
            save_json(self.encoder, out_src_vocab_file)
            save_json(self.target_encoder, out_tgt_vocab_file)
            saved_files.append(out_src_vocab_file)
            saved_files.append(out_tgt_vocab_file)
        else:
            # 构建词汇表文件名（未分离词汇表时）
            out_vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"]
            )
            # 将编码器保存到JSON文件
            save_json(self.encoder, out_vocab_file)
            saved_files.append(out_vocab_file)

        # 复制或保存SentencePiece模型文件
        for spm_save_filename, spm_orig_path, spm_model in zip(
            [VOCAB_FILES_NAMES["source_spm"], VOCAB_FILES_NAMES["target_spm"]],
            self.spm_files,
            [self.spm_source, self.spm_target],
        ):
            # 构建保存路径
            spm_save_path = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + spm_save_filename
            )
            # 如果原始路径和保存路径不同且原始路径是文件，则复制文件
            if os.path.abspath(spm_orig_path) != os.path.abspath(spm_save_path) and os.path.isfile(spm_orig_path):
                copyfile(spm_orig_path, spm_save_path)
                saved_files.append(spm_save_path)
            # 如果原始路径不是文件，则将序列化后的模型写入保存路径
            elif not os.path.isfile(spm_orig_path):
                with open(spm_save_path, "wb") as fi:
                    content_spiece_model = spm_model.serialized_model_proto()
                    fi.write(content_spiece_model)
                saved_files.append(spm_save_path)

        # 返回保存的文件路径组成的元组
        return tuple(saved_files)

    # 获取编码器字典，等同于获取源语言编码器字典
    def get_vocab(self) -> Dict:
        return self.get_src_vocab()

    # 获取源语言编码器字典
    def get_src_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 获取目标语言编码器字典
    def get_tgt_vocab(self):
        return dict(self.target_encoder, **self.added_tokens_decoder)

    # 定义对象的序列化状态
    def __getstate__(self) -> Dict:
        # 复制对象的字典属性
        state = self.__dict__.copy()
        # 将指定键的值设置为None，以便对象可以进行序列化
        state.update(
            {k: None for k in ["spm_source", "spm_target", "current_spm", "punc_normalizer", "target_vocab_file"]}
        )
        return state
    # 重写对象的状态，将给定的字典作为对象的新状态
    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d

        # 为了向后兼容性
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 加载和设置子词模型来源和目标文件的路径
        self.spm_source, self.spm_target = (load_spm(f, self.sp_model_kwargs) for f in self.spm_files)
        # 设置当前的子词模型为来源模型
        self.current_spm = self.spm_source
        # 初始化正则化器
        self._setup_normalizer()

    # 返回特殊令牌（例如 EOS）的数量
    def num_special_tokens_to_add(self, *args, **kwargs):
        """Just EOS"""
        return 1

    # 创建一个掩码，标记特殊令牌（非 EOS 或 PAD ）
    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # 仅调用一次以避免在列表推导内部多次调用
        all_special_ids.remove(self.unk_token_id)  # <unk> 仅在某些情况下是特殊的
        return [1 if x in all_special_ids else 0 for x in seq]

    # 获取特殊令牌的掩码列表，其中条目为 [1] 表示为 [eos] 或 [pad]，否则为 0
    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]
# 根据给定的路径加载 SentencePiece 模型，并使用提供的参数配置
def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    # 创建一个 SentencePieceProcessor 对象，使用传入的参数进行配置
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    # 加载指定路径下的 SentencePiece 模型文件
    spm.Load(path)
    return spm

# 将给定的数据保存为 JSON 格式到指定的文件路径
def save_json(data, path: str) -> None:
    # 以写入模式打开指定路径的文件
    with open(path, "w") as f:
        # 将数据以漂亮的格式（缩进为2）写入 JSON 文件
        json.dump(data, f, indent=2)

# 加载指定路径的 JSON 文件，并返回其中的数据结构（可以是字典或列表）
def load_json(path: str) -> Union[Dict, List]:
    # 以读取模式打开指定路径的 JSON 文件
    with open(path, "r") as f:
        # 解析 JSON 文件并将其内容返回
        return json.load(f)
```