# `.\transformers\models\speech_to_text\tokenization_speech_to_text.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可信息
"""Tokenization classes for Speech2Text."""
# 导入所需模块和库
import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece

# 导入 Hugging Face 库中的通用 Tokenizer 类和日志记录工具
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 设置词片分隔符
SPIECE_UNDERLINE = "▁"

# 定义词汇文件名
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "spm_file": "sentencepiece.bpe.model",
}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/s2t-small-librispeech-asr": (
            "https://huggingface.co/facebook/s2t-small-librispeech-asr/resolve/main/vocab.json"
        ),
    },
    "spm_file": {
        "facebook/s2t-small-librispeech-asr": (
            "https://huggingface.co/facebook/s2t-small-librispeech-asr/resolve/main/sentencepiece.bpe.model"
        )
    },
}

# 预训练模型的最大输入长度映射
MAX_MODEL_INPUT_SIZES = {
    "facebook/s2t-small-librispeech-asr": 1024,
}

# MUSTC 数据集支持的语言列表
MUSTC_LANGS = ["pt", "fr", "ru", "nl", "ro", "it", "es", "de"]

# 支持的语言列表
LANGUAGES = {"mustc": MUSTC_LANGS}

# 定义 Speech2TextTokenizer 类，继承自 PreTrainedTokenizer
class Speech2TextTokenizer(PreTrainedTokenizer):
    """
    Construct an Speech2Text tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains some of the main methods. Users should refer to
    the superclass for more information regarding such methods.
    """
    # 定义初始化函数，接受以下参数
    Args:
        vocab_file (`str`):
            包含词汇表的文件。
        spm_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) 模型文件的路径。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            句子开头的标记。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            句子结尾的标记。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中不存在的标记将无法转换为ID，并被设置为此标记。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的标记，例如当批处理不同长度的序列时使用。
        do_upper_case (`bool`, *optional*, defaults to `False`):
            在解码时是否将输出转换为大写。
        do_lower_case (`bool`, *optional*, defaults to `False`):
            在标记化时是否将输入转换为小写。
        tgt_lang (`str`, *optional*):
            表示目标语言的字符串。
        sp_model_kwargs (`dict`, *optional*):
            将被传递给 `SentencePieceProcessor.__init__()` 方法的参数。
            [SentencePiece的Python封装](https://github.com/google/sentencepiece/tree/master/python) 可以用于设置：
            - `enable_sampling`: 启用子词正则化。
            - `nbest_size`: 用于unigram的采样参数。BPE-Dropout无效。
                - `nbest_size = {0,1}`: 不进行采样。
                - `nbest_size > 1`: 从前nbest_size个结果中进行采样。
                - `nbest_size < 0`: 假设nbest_size为无穷大，并使用前向过滤和后向采样算法从所有假设（网格）中进行采样。
            - `alpha`: unigram采样的平滑参数，以及BPE-dropout的合并操作的丢失概率。
        **kwargs
            传递给 [`PreTrainedTokenizer`] 的额外关键字参数。
    """

    vocab_files_names = VOCAB_FILES_NAMES  # 获取词汇文件名
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练的词汇文件映射
    max_model_input_sizes = MAX_MODEL_INPUT_SIZES  # 最大模型输入尺寸
    model_input_names = ["input_ids", "attention_mask"]  # 模型输入名称列表

    prefix_tokens: List[int] = []  # 前缀标记列表，初始化为空列表

    # 初始化函数
    def __init__(
        self,
        vocab_file,
        spm_file,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        do_upper_case=False,
        do_lower_case=False,
        tgt_lang=None,
        lang_codes=None,
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    # 定义初始化方法，设置默认参数为空字典，且保存参数值
    def __init__(
        self,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        do_upper_case: bool = False,
        do_lower_case: bool = False,
        vocab_file: str,
        spm_file: str,
        lang_codes: Optional[str] = None,
        additional_special_tokens: Optional[List[str]] = None,
        tgt_lang: Optional[str] = None,
        **kwargs,
    ) -> None:
        # 如果 sp_model_kwargs 为 None，则设置为空字典，否则保留值
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        
        # 保存参数值
        self.do_upper_case = do_upper_case
        self.do_lower_case = do_lower_case
        
        # 从文件中加载编码器和解码器，创建 spm 模型
        self.encoder = load_json(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.spm_file = spm_file
        self.sp_model = load_spm(spm_file, self.sp_model_kwargs)

        # 如果提供了语言代码，则设置语言相关变量，包括特殊 token 和语言关联
        if lang_codes is not None:
            self.lang_codes = lang_codes
            self.langs = LANGUAGES[lang_codes]
            self.lang_tokens = [f"<lang:{lang}>" for lang in self.langs]
            self.lang_code_to_id = {lang: self.sp_model.PieceToId(f"<lang:{lang}>") for lang in self.langs}
            if additional_special_tokens is not None:
                additional_special_tokens = self.lang_tokens + additional_special_tokens
            else:
                additional_special_tokens = self.lang_tokens
            self._tgt_lang = tgt_lang if tgt_lang is not None else self.langs[0]

            # 设置特殊 token
            self.set_tgt_lang_special_tokens(self._tgt_lang)
        else:
            self.lang_code_to_id = {}

        # 调用父类的初始化方法，传递参数
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            do_upper_case=do_upper_case,
            do_lower_case=do_lower_case,
            tgt_lang=tgt_lang,
            lang_codes=lang_codes,
            sp_model_kwargs=self.sp_model_kwargs,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    # 定义属性，返回词汇表大小
    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    # 返回词汇表
    def get_vocab(self) -> Dict:
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    # 定义属性，返回目标语言
    @property
    def tgt_lang(self) -> str:
        return self._tgt_lang

    # 设置目标语言
    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang) -> None:
        self._tgt_lang = new_tgt_lang
        self.set_tgt_lang_special_tokens(new_tgt_lang)

    # 重置特殊 token 为目标语言设置
    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[eos, tgt_lang_code] and suffix=[eos]."""
        lang_code_id = self.lang_code_to_id[tgt_lang]
        self.prefix_tokens = [lang_code_id]

    # 对文本进行分词
    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    # 转换 token 到 ID
    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])

    # 转换 ID 到 token
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        return self.decoder.get(index, self.unk_token)
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """将一系列标记（子词的字符串）转换为单个字符串。"""
        current_sub_tokens = []  # 当前子词列表
        out_string = ""  # 输出字符串
        for token in tokens:
            # 确保特殊标记不会使用 sentencepiece 模型进行解码
            if token in self.all_special_tokens:
                # 使用 sentencepiece 模型对当前子词列表进行解码
                decoded = self.sp_model.decode(current_sub_tokens)
                out_string += (decoded.upper() if self.do_upper_case else decoded) + token + " "  # 如果 do_upper_case 为真，则将解码后的内容转换为大写
                current_sub_tokens = []  # 清空当前子词列表
            else:
                current_sub_tokens.append(token)  # 将标记添加到当前子词列表
        decoded = self.sp_model.decode(current_sub_tokens)  # 对剩余的子词列表进行解码
        out_string += decoded.upper() if self.do_upper_case else decoded  # 如果 do_upper_case 为真，则将解码后的内容转换为大写
        return out_string.strip()  # 返回去除两侧空格的结果字符串

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """通过添加 eos_token_id 从序列构建模型输入。"""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]  # 如果只有一个输入序列，则在前缀标记后添加输入序列和结束标记
        # 我们不希望处理成对序列，但为了 API 一致性保留成对逻辑
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]  # 如果有两个输入序列，则在前缀标记后添加两个输入序列和结束标记

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        从没有添加特殊标记的标记列表中检索序列 id。这个方法在使用 tokenizer `prepare_for_model` 方法添加特殊标记时调用。

        参数：
            token_ids_0（List[int]）：
                ID 列表。
            token_ids_1（List[int]，*可选*）：
                可选的第二个 ID 列表，用于序列对。
            already_has_special_tokens（bool，*可选*，默认为 `False`）：
                标记列表是否已经格式化为模型的特殊标记。

        返回：
            `List[int]`：一个整数列表，范围为 [0, 1]：1 表示特殊标记，0 表示序列标记。
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)  # 前缀标记的全 1 列表
        suffix_ones = [1]  # 后缀标记的全 1 列表
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones  # 如果只有一个输入序列，则返回前缀全 1 列表 + 输入序列全 0 列表 + 后缀全 1 列表
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones  # 如果有两个输入序列，则返回前缀全 1 列表 + 输入序列全 0 列表 + 输入序列全 0 列表 + 后缀全 1 列表

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()  # 复制当前对象的字典形式
        state["sp_model"] = None  # 将 sp_model 属性置为 None
        return state  # 返回更新后的对象状态字典
    # 重新定义对象的状态，将其属性字典替换为给定字典
    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d  # 使用给定的字典 d 替换对象的属性字典

        # 为了向后兼容性，在对象中添加 sp_model_kwargs 属性（如果不存在）
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 加载 SentencePiece 模型，使用给定的 spm_file 和 sp_model_kwargs
        self.sp_model = load_spm(self.spm_file, self.sp_model_kwargs)

    # 将词汇表保存到指定的目录中
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 将保存目录路径转换为 Path 对象
        save_dir = Path(save_directory)
        # 确保保存目录是一个目录，否则抛出断言错误
        assert save_dir.is_dir(), f"{save_directory} should be a directory"

        # 构建词汇表和 SentencePiece 模型的保存路径
        vocab_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        spm_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["spm_file"]
        )

        # 将编码器（encoder）保存为 JSON 文件到指定的 vocab_save_path
        save_json(self.encoder, vocab_save_path)

        # 如果 self.spm_file 的绝对路径与 spm_save_path 不同且 self.spm_file 是一个文件，则复制 self.spm_file 到 spm_save_path
        if os.path.abspath(self.spm_file) != os.path.abspath(spm_save_path) and os.path.isfile(self.spm_file):
            copyfile(self.spm_file, spm_save_path)
        # 否则，如果 self.spm_file 不是一个文件，则将序列化后的 SentencePiece 模型写入 spm_save_path
        elif not os.path.isfile(self.spm_file):
            with open(spm_save_path, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回保存的词汇表和 SentencePiece 模型的路径
        return (str(vocab_save_path), str(spm_save_path))
# 加载 SentencePiece 模型
def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    # 创建 SentencePieceProcessor 对象，并传入指定的参数
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    # 加载指定路径的 SentencePiece 模型
    spm.Load(str(path))
    # 返回初始化的 SentencePieceProcessor 对象
    return spm


# 加载 JSON 文件
def load_json(path: str) -> Union[Dict, List]:
    # 打开指定路径的 JSON 文件，以只读方式
    with open(path, "r") as f:
        # 解析 JSON 文件并返回其内容
        return json.load(f)


# 保存数据到 JSON 文件
def save_json(data, path: str) -> None:
    # 打开指定路径的文件，以写入方式
    with open(path, "w") as f:
        # 将数据以 JSON 格式写入文件，设置缩进为 2 个空格
        json.dump(data, f, indent=2)
```