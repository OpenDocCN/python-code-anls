# `.\transformers\models\marian\tokenization_marian.py`

```py
# 版权声明以及许可信息
# 版权所有 © 2020 The HuggingFace Team
# 根据 Apache 许可证第 2 版（“许可证”）许可
# 除非符合许可条件，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件根据“原样”提供，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取关于特定语言的权限和限制
import json  # 导入用于处理 JSON 格式的模块
import os  # 导入用于提供与操作系统交互的功能的模块
import re  # 导入用于支持正则表达式的模块
import warnings  # 导入用于处理警告的模块
from pathlib import Path  # 导入处理路径相关操作的模块
from shutil import copyfile  # 导入文件复制相关功能的模块
from typing import Any, Dict, List, Optional, Tuple, Union  # 引入类型提示相关模块

import sentencepiece  # 导入分词器模块

from ...tokenization_utils import PreTrainedTokenizer  # 从 Hugging Face 的 tokenization_utils 模块中导入 PreTrainedTokenizer 类
from ...utils import logging  # 从 Hugging Face 的 utils 模块中导入 logging 模块

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义各种文件的名称常量
VOCAB_FILES_NAMES = {
    "source_spm": "source.spm",
    "target_spm": "target.spm",
    "vocab": "vocab.json",
    "target_vocab_file": "target_vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}

# 定义预训练模型对应的词汇文件映射关系
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

# 定义预训练模型对应的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"Helsinki-NLP/opus-mt-en-de": 512}

# 定义预训练模型初始化配置信息
PRETRAINED_INIT_CONFIGURATION = {}

# 定义特殊的 SPIECE_UNDERLINE 常量
SPIECE_UNDERLINE = "▁"

# 以下为类的注释
# 构造一个 MarianTokenizer。基于 SentencePiece。
# 该分词器是 PreTrainedTokenizer 的子类，继承了大部分主要方法。用户应参考超类以获取有关这些方法的更多信息。
class MarianTokenizer(PreTrainedTokenizer):
    """Construct a Marian tokenizer. Based on SentencePiece.

    This tokenizer inherits from `PreTrainedTokenizer` which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.
    """
    Args:
        source_spm (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary for the source language.
        target_spm (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that
            contains the vocabulary for the target language.
        source_lang (`str`, *optional*):
            A string representing the source language.
        target_lang (`str`, *optional*):
            A string representing the target language.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        model_max_length (`int`, *optional*, defaults to 512):
            The maximum sentence length the model accepts.
        additional_special_tokens (`List[str]`, *optional*, defaults to `["<eop>", "<eod>"]`):
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

    Examples:

    ```python
    >>> from transformers import MarianForCausalLM, MarianTokenizer

    >>> model = MarianForCausalLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    >>> tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    >>> src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
    >>> tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
    >>> inputs = tokenizer(src_texts, text_target=tgt_texts, return_tensors="pt", padding=True)

    >>> outputs = model(**inputs)  # should work
    ```py"""

    # Define the names of the vocabulary files
    vocab_files_names = VOCAB_FILES_NAMES
    # Define the map of pretrained vocabulary files
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 将预训练模型的初始配置赋值给变量
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    # 将预训练位置嵌入的最大模型输入尺寸赋值给变量
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 编译正则表达式，用于匹配语言代码
    language_code_re = re.compile(">>.+<<")  # type: re.Pattern

    # 初始化方法，接受多个参数
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
        # 如果没有传入 sp_model_kwargs，则将其初始化为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        # 检查源分词模型文件是否存在，如果不存在则抛出异常
        assert Path(source_spm).exists(), f"cannot find spm source {source_spm}"

        # 设置是否使用分离的词汇表
        self.separate_vocabs = separate_vocabs
        # 加载源文本的编码器词典
        self.encoder = load_json(vocab)
        # 如果 unk_token 不在编码器词典中，则抛出异常
        if str(unk_token) not in self.encoder:
            raise KeyError("<unk> token must be in the vocab")
        # 检查 pad_token 是否在编码器词典中
        assert str(pad_token) in self.encoder

        # 如果使用分离的词汇表
        if separate_vocabs:
            # 加载目标文本的编码器词典
            self.target_encoder = load_json(target_vocab_file)
            # 构建目标解码器词典，key 和 value 互换
            self.decoder = {v: k for k, v in self.target_encoder.items()}
            # 初始化支持的语言代码列表为空
            self.supported_language_codes = []
        else:
            # 构建解码器词典，key 和 value 互换
            self.decoder = {v: k for k, v in self.encoder.items()}
            # 获取支持的语言代码列表
            self.supported_language_codes: list = [k for k in self.encoder if k.startswith(">>") and k.endswith("<<")]

        # 设置源语言和目标语言
        self.source_lang = source_lang
        self.target_lang = target_lang
        # 将源分词模型和目标分词模型文件路径组成列表
        self.spm_files = [source_spm, target_spm]

        # 加载 SentencePiece 分���模型用于预处理
        self.spm_source = load_spm(source_spm, self.sp_model_kwargs)
        self.spm_target = load_spm(target_spm, self.sp_model_kwargs)
        self.current_spm = self.spm_source
        self.current_encoder = self.encoder

        # 多语言目标端：默认使用第一个支持的语言代码
        self._setup_normalizer()

        # 调用父类的初始化方法，传入各种参数
        super().__init__(
            # bos_token=bos_token,  unused. Start decoding with config.decoder_start_token_id
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

    # 设置标准化器
    def _setup_normalizer(self):
        try:
            # 尝试导入 MosesPunctNormalizer，并设置标点标准化方法
            from sacremoses import MosesPunctNormalizer
            self.punc_normalizer = MosesPunctNormalizer(self.source_lang).normalize
        except (ImportError, FileNotFoundError):
            # 如果导入失败，则警告安装 sacremoses 库
            warnings.warn("Recommended: pip install sacremoses.")
            self.punc_normalizer = lambda x: x
    # 将输入字符串规范化，处理空字符串的特例情况
    def normalize(self, x: str) -> str:
        """Cover moses empty string edge case. They return empty list for '' input!"""
        return self.punc_normalizer(x) if x else ""

    # 将 token 转换为对应的 id
    def _convert_token_to_id(self, token):
        return self.current_encoder.get(token, self.current_encoder[self.unk_token])

    # 移除文本中的语言代码，如 >>fr<<
    def remove_language_code(self, text: str):
        """Remove language codes like >>fr<< before sentencepiece"""
        match = self.language_code_re.match(text)
        code: list = [match.group(0)] if match else []
        return code, self.language_code_re.sub("", text)

    # 对文本进行分词处理
    def _tokenize(self, text: str) -> List[str]:
        code, text = self.remove_language_code(text)
        pieces = self.current_spm.encode(text, out_type=str)
        return code + pieces

    # 将 id 转换为对应的 token
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        return self.decoder.get(index, self.unk_token)

    # 将一组 token id 列表转换为字符串列表
    def batch_decode(self, sequences, **kwargs):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
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
            `List[str]`: The list of decoded sentences.
        """
        return super().batch_decode(sequences, **kwargs)
    # 将一系列标识符转换为字符串，使用标记器和词汇表，还可以选择移除特殊标记和清理标记化空格
    def decode(self, token_ids, **kwargs):
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

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

    # 将标记转换为字符串
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Uses source spm if _decode_use_source_tokenizer is True, and target spm otherwise"""
        sp_model = self.spm_source if self._decode_use_source_tokenizer else self.spm_target
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # 确保特殊标记不会使用sentencepiece模型进行解码
            if token in self.all_special_tokens:
                out_string += sp_model.decode_pieces(current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += sp_model.decode_pieces(current_sub_tokens)
        out_string = out_string.replace(SPIECE_UNDERLINE, " ")
        return out_string.strip()

    # 通过添加eos_token_id从序列构建模型输入
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # 我们不希望处理成对，但为了API一致性保留成对逻辑
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    # 切换到输入模式
    def _switch_to_input_mode(self):
        self.current_spm = self.spm_source
        self.current_encoder = self.encoder

    # 切换到目标模式
    def _switch_to_target_mode(self):
        self.current_spm = self.spm_target
        if self.separate_vocabs:
            self.current_encoder = self.target_encoder

    @property
    # 返回编码器中的词汇表大小
    def vocab_size(self) -> int:
        return len(self.encoder)

    # 保存词汇表到指定目录，返回保存的文件名元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果保存目录不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        saved_files = []

        # 如果使用分开的词汇表
        if self.separate_vocabs:
            # 保存源语言的词汇表
            out_src_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"],
            )
            save_json(self.encoder, out_src_vocab_file)
            saved_files.append(out_src_vocab_file)
            # 保存目标语言的词汇表
            out_tgt_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["target_vocab_file"],
            )
            save_json(self.target_encoder, out_tgt_vocab_file)
            saved_files.append(out_tgt_vocab_file)
        else:
            # 保存共享的词汇表
            out_vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"]
            )
            save_json(self.encoder, out_vocab_file)
            saved_files.append(out_vocab_file)

        # 保存句子片模型相关文件
        for spm_save_filename, spm_orig_path, spm_model in zip(
            [VOCAB_FILES_NAMES["source_spm"], VOCAB_FILES_NAMES["target_spm"]],
            self.spm_files,
            [self.spm_source, self.spm_target],
        ):
            spm_save_path = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + spm_save_filename
            )
            # 如果句子片模型原始路径与保存路径不同且原始路径是一个文件
            if os.path.abspath(spm_orig_path) != os.path.abspath(spm_save_path) and os.path.isfile(spm_orig_path):
                copyfile(spm_orig_path, spm_save_path)
                saved_files.append(spm_save_path)
            # 如果句子片模型原始路径不是一个文件
            elif not os.path.isfile(spm_orig_path):
                with open(spm_save_path, "wb") as fi:
                    # 写入句子片模型序列化数据
                    content_spiece_model = spm_model.serialized_model_proto()
                    fi.write(content_spiece_model)
                saved_files.append(spm_save_path)

        # 返回保存的文件名元组
        return tuple(saved_files)

    # 获取词汇表，返回源语言的词汇表
    def get_vocab(self) -> Dict:
        return self.get_src_vocab()

    # 获取源语言的词汇表
    def get_src_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # 获取目标语言的词汇表
    def get_tgt_vocab(self):
        return dict(self.target_encoder, **self.added_tokens_decoder)

    # 序列化对象状态以便pickle
    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        # 更新状态，部分数据置空以减小序列化对象的大小
        state.update(
            {k: None for k in ["spm_source", "spm_target", "current_spm", "punc_normalizer", "target_vocab_file"]}
        )
        return state
    # 定义特殊方法 __setstate__，用于从序列化的状态中恢复对象的属性
    def __setstate__(self, d: Dict) -> None:
        # 将对象的 __dict__ 属性设置为给定的字典
        self.__dict__ = d

        # 为了向后兼容性
        # 如果对象没有 sp_model_kwargs 属性，则将其设置为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用 load_spm 函数加载 spm_files 中的文件，并分别赋值给 spm_source 和 spm_target
        self.spm_source, self.spm_target = (load_spm(f, self.sp_model_kwargs) for f in self.spm_files)
        # 将当前 spm 设置为 spm_source
        self.current_spm = self.spm_source
        # 设置标准化器
        self._setup_normalizer()

    # 定义方法 num_special_tokens_to_add，用于返回要添加的特殊标记数量
    def num_special_tokens_to_add(self, *args, **kwargs):
        """Just EOS"""
        # 只添加一个特殊标记 <eos>
        return 1

    # 定义方法 _special_token_mask，用于生成特殊标记的掩码
    def _special_token_mask(self, seq):
        # 将所有特殊标记的 ID 放入集合中，只调用一次而不是在列表推导式中多次调用
        all_special_ids = set(self.all_special_ids)
        # 从集合中移除 <unk>，因为它只在某些情况下是特殊的
        all_special_ids.remove(self.unk_token_id)
        # 生成特殊标记的掩码列表，如果序列中的元素是特殊标记，则为 1，否则为 0
        return [1 if x in all_special_ids else 0 for x in seq]

    # 定义方法 get_special_tokens_mask，用于生成特殊标记的掩码列表
    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        # 如果已经存在特殊标记，则直接调用 _special_token_mask 生成掩码列表
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        # 如果不存在特殊标记，则将 token_ids_0 作为输入调用 _special_token_mask，最后添加一个 1，表示 [eos]
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        # 如果有两个输入序列，则将它们合并后调用 _special_token_mask，最后添加一个 1，表示 [eos]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]
# 加载 SentencePiece 模型
def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    # 创建 SentencePieceProcessor 对象，使用给定的参数
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    # 加载指定路径下的 SentencePiece 模型
    spm.Load(path)
    # 返回加载后的 SentencePieceProcessor 对象
    return spm


# 将数据保存为 JSON 格式到指定路径
def save_json(data, path: str) -> None:
    # 以写入模式打开指定路径的文件
    with open(path, "w") as f:
        # 将数据以 JSON 格式写入文件，缩进为 2 个空格
        json.dump(data, f, indent=2)


# 从指定路径加载 JSON 格式的数据
def load_json(path: str) -> Union[Dict, List]:
    # 以只读模式打开指定路径的文件
    with open(path, "r") as f:
        # 从文件中加载 JSON 格式的数据并返回
        return json.load(f)
```