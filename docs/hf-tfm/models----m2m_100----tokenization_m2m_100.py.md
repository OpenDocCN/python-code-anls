# `.\transformers\models\m2m_100\tokenization_m2m_100.py`

```py
# 该代码是 M2M100 模型的分词器实现,基于 SentencePiece 算法
# 包含了以下主要功能:
# 1. 定义了分词器需要的文件名和预训练模型的 URL
# 2. 定义了 M2M100 模型支持的语言代码
# 3. 实现了 M2M100 分词器的核心类 M2M100Tokenizer

# Copyright 声明,指定了代码的许可协议

# 导入必要的库和模块
import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece

from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义分词时使用的特殊标记
SPIECE_UNDERLINE = "▁"

# 定义分词器需要的文件名
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "spm_file": "sentencepiece.bpe.model",
    "tokenizer_config_file": "tokenizer_config.json",
}

# 定义预训练模型对应的 URL
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/m2m100_418M": "https://huggingface.co/facebook/m2m100_418M/resolve/main/vocab.json",
        "facebook/m2m100_1.2B": "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/vocab.json",
    },
    "spm_file": {
        "facebook/m2m100_418M": "https://huggingface.co/facebook/m2m100_418M/resolve/main/sentencepiece.bpe.model",
        "facebook/m2m100_1.2B": "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_config_file": {
        "facebook/m2m100_418M": "https://huggingface.co/facebook/m2m100_418M/resolve/main/tokenizer_config.json",
        "facebook/m2m100_1.2B": "https://huggingface.co/facebook/m2m100_1.2B/resolve/main/tokenizer_config.json",
    },
}

# 定义预训练模型支持的最大序列长度
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/m2m100_418M": 1024,
}

# 定义 M2M100 模型支持的语言代码
FAIRSEQ_LANGUAGE_CODES = {
    "m2m100": ["af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs", "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv", "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns", "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk", "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl", "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh", "zu"],
    "wmt21": ['en', 'ha', 'is', 'ja', 'cs', 'ru', 'zh', 'de']
}

# 定义 M2M100Tokenizer 类,继承自 PreTrainedTokenizer
class M2M100Tokenizer(PreTrainedTokenizer):
    """
    Construct an M2M100 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).
    """
    # 这个分词器继承自`PreTrainedTokenizer`，其中包含了大多数主要方法。用户应该参考这个超类以获得有关这些方法的更多信息。
    
    # 参数:
    # vocab_file(`str`)：词汇文件的路径。
    # spm_file(`str`)：包含词汇的[SentencePiece](https://github.com/google/sentencepiece)文件的路径(通常具有.spm扩展名)。
    # src_lang(`str`，*可选*)：表示源语言的字符串。
    # tgt_lang(`str`，*可选*)：表示目标语言的字符串。
    # eos_token(`str`，*可选*，默认为`"</s>"`)：序列的结束符号。
    # sep_token(`str`，*可选*，默认为`"</s>"`)：分隔符标记，用于从多个序列构建序列，例如用于序列分类的两个序列，或用于文本和问题生成的文本序列。它也用作使用特殊标记构建的序列的最后一个标记。
    # unk_token(`str`，*可选*，默认为`"<unk>"`)：未知标记。词汇中没有的标记无法转换为ID，并将设置为该标记。
    # pad_token(`str`，*可选*，默认为`"<pad>"`)：用于填充的标记，例如在批处理不同长度的序列时使用。
    # language_codes(`str`，*可选*，默认为`"m2m100"`）：要使用的语言代码。应该是`"m2m100"`或`"wmt21"`之一。
    # sp_model_kwargs(`dict`，*可选*）：将传递给`SentencePieceProcessor.__init__()`方法的参数。[SentencePiece的Python封装](https://github.com/google/sentencepiece/tree/master/python)可以用来设置：
    
    #     - `enable_sampling`：启用子字规范化。
    #     - `nbest_size`：unigram的采样参数。对于BPE-Dropout无效。
    
    #       - `nbest_size = {0,1}`：不执行采样。
    #       - `nbest_size > 1`：从nbest_size结果中采样。
    #       - `nbest_size < 0`：假设nbest_size无限大，并使用前向过滤和后向采样算法从所有假设(网格)中采样。
    
    #     - `alpha`：unigram采样的平滑参数，以及BPE-dropout合并操作的dropout概率。
    
    # 示例:
    # ```python
    # >>> from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    
    # >>> model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    # >>> tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", tgt_lang="ro")
    # >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
    # >>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
    # ```py
    # 使用给定的文本进行标记化，生成模型输入
    model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
    # 使用模型处理生成的输入，输出结果
    outputs = model(**model_inputs)  # 应该正常工作

    # 初始化变量，存储词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 初始化变量，存储预训练模型的最大输入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 初始化变量，存储预训练词汇文件的映射关系
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 初始化变量，存储模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化前缀标记列表
    prefix_tokens: List[int] = []
    # 初始化后缀标记列表
    suffix_tokens: List[int] = []

    # 初始化方法，构造函数
    def __init__(
        self,
        vocab_file,
        spm_file,
        src_lang=None,
        tgt_lang=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        language_codes="m2m100",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        num_madeup_words=8,
        **kwargs,
    ) -> None:
        # 如果未提供 sp_model_kwargs，则初始化为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 初始化语言代码
        self.language_codes = language_codes
        # 从 FAIRSEQ_LANGUAGE_CODES 获取 fairseq 语言代码
        fairseq_language_code = FAIRSEQ_LANGUAGE_CODES[language_codes]
        # 根据 fairseq 语言代码创建语言代码到特殊标记的映射
        self.lang_code_to_token = {lang_code: f"__{lang_code}__" for lang_code in fairseq_language_code}

        # 获取额外的特殊标记
        additional_special_tokens = kwargs.pop("additional_special_tokens", [])
        # 遍历 fairseq 语言代码，添加缺失的特殊标记
        for lang_code in fairseq_language_code:
            token = self.get_lang_token(lang_code)
            if token not in additional_special_tokens and lang_code not in str(token) not in self.added_tokens_encoder:
                additional_special_tokens.append(token)

        # 初始化词汇文件和 SPM 文件
        self.vocab_file = vocab_file
        self.encoder = load_json(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.spm_file = spm_file
        self.sp_model = load_spm(spm_file, self.sp_model_kwargs)

        # 计算编码器大小
        self.encoder_size = len(self.encoder)

        # 创建语言代码到 ID 的映射
        self.lang_token_to_id = {
            self.get_lang_token(lang_code): self.encoder_size + i for i, lang_code in enumerate(fairseq_language_code)
        }
        # 创建语言代码到 ID 的映射
        self.lang_code_to_id = {lang_code: self.encoder_size + i for i, lang_code in enumerate(fairseq_language_code)}
        # 创建 ID 到语言代码的映射
        self.id_to_lang_token = {v: k for k, v in self.lang_token_to_id.items()}

        # 设置源语言和目标语言
        self._src_lang = src_lang if src_lang is not None else "en"
        self.tgt_lang = tgt_lang
        # 获取当前语言 ID
        self.cur_lang_id = self.get_lang_id(self._src_lang)

        # 设置虚构词汇数目
        self.num_madeup_words = num_madeup_words

        # 调用父类构造函数
        super().__init__(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            unk_token=unk_token,
            pad_token=pad_token,
            language_codes=language_codes,
            sp_model_kwargs=self.sp_model_kwargs,
            additional_special_tokens=additional_special_tokens,
            num_madeup_words=num_madeup_words,
            **kwargs,
        )
        # 设置源语言特殊标记
        self.set_src_lang_special_tokens(self._src_lang)

    # 属性装饰器，用于获取属性值
    @property
    # 返回词汇表大小
    def vocab_size(self) -> int:
        return len(self.encoder)

    # 返回词汇表
    def get_vocab(self) -> Dict:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}  # 创建词汇表字典
        vocab.update(self.added_tokens_encoder)  # 更新添加的特殊标记的编码器
        return vocab

    # 返回源语言
    @property
    def src_lang(self) -> str:
        return self._src_lang

    # 设置源语言
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang  # 更新源语言
        self.set_src_lang_special_tokens(self._src_lang)  # 设置源语言的特殊标记

    # 对文本进行分词
    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)  # 使用sp_model对文本进行编码，返回字符串列表

    # 将标记转换为id
    def _convert_token_to_id(self, token):
        if token in self.lang_token_to_id:  # 如果标记在语言标记到id的映射中
            return self.lang_token_to_id[token]  # 返回语言标记对应的id
        return self.encoder.get(token, self.encoder[self.unk_token])  # 否则返回默认的未知标记的id

    # 将id转换为标记
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        if index in self.id_to_lang_token:  # 如果id在id到语言标记的映射中
            return self.id_to_lang_token[index]  # 返回id对应的语言标记
        return self.decoder.get(index, self.unk_token)  # 否则返回默认的未知标记

    # 将标记序列转换为单个字符串
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []  # 存储当前的子标记
        out_string = ""  # 输出字符串
        for token in tokens:
            if token in self.all_special_tokens:  # 如果标记是特殊标记
                out_string += self.sp_model.decode(current_sub_tokens) + token  # 解码当前子标记并添加到输出字符串中
                current_sub_tokens = []  # 重置当前子标记
            else:
                current_sub_tokens.append(token)  # 否则将标记添加到当前子标记中
        out_string += self.sp_model.decode(current_sub_tokens)  # 解码剩余的子标记并添加到输出字符串中
        return out_string.strip()  # 返回去除首尾空格的输出字符串

    # 获取特殊标记的掩码
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
            """
            从没有添加特殊标记的标记列表中检索序列ID。当使用分词器的`prepare_for_model`方法添加特殊标记时，将调用此方法。
    
            Args:
                token_ids_0 (`List[int]`):
                    ID列表。
                token_ids_1 (`List[int]`, *可选*):
                    第二个ID列表，用于序列对。
                already_has_special_tokens (`bool`, *可选*，默认为`False`):
                    标记列表是否已经有模型的特殊标记。
    
            Returns:
                `List[int]`: 包含整数的列表，范围为[0, 1]：1表示特殊标记，0表示序列标记。
            """
    
            if already_has_special_tokens:
                return super().get_special_tokens_mask(
                    token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
                )
    
            prefix_ones = [1] * len(self.prefix_tokens)
            suffix_ones = [1] * len(self.suffix_tokens)
            if token_ids_1 is None:
                return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
            return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones
    
        def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
            """
            通过连接和添加特殊标记从序列或序列对构建用于序列分类任务的模型输入。MBART序列的格式如下，其中`X`表示序列：
    
            - `input_ids`（用于编码器）`X [eos, src_lang_code]`
            - `decoder_input_ids`：（用于解码器）`X [eos, tgt_lang_code]`
    
            不使用BOS。我们不期望处理序列对，但为了API一致性，它们将被处理而不使用分隔符。
    
            Args:
                token_ids_0 (`List[int]`):
                    将添加特殊标记的ID列表。
                token_ids_1 (`List[int]`, *可选*):
                    第二个ID列表，用于序列对。
    
            Returns:
                `List[int]`: 包含适当特殊标记的[input IDs](../glossary#input-ids)列表。
            """
            if token_ids_1 is None:
                return self.prefix_tokens + token_ids_0 + self.suffix_tokens
            # 我们不期望处理序列对，但为了API一致性，保留成对逻辑
            return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens
    
        def __getstate__(self) -> Dict:
            state = self.__dict__.copy()
            state["sp_model"] = None
            return state
    
        def __setstate__(self, d: Dict) -> None:
            self.__dict__ = d
    
            # 为了向后兼容性
            if not hasattr(self, "sp_model_kwargs"):
                self.sp_model_kwargs = {}
    
            self.sp_model = load_spm(self.spm_file, self.sp_model_kwargs)
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 将保存目录转换为 Path 对象
        save_dir = Path(save_directory)
        # 如果保存目录不是一个目录，抛出异常
        if not save_dir.is_dir():
            raise OSError(f"{save_directory} should be a directory")
        # 生成词汇表和 SentencePiece 模型文件的保存路径
        vocab_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        spm_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["spm_file"]
        )

        # 保存编码器到 JSON 文件
        save_json(self.encoder, vocab_save_path)

        # 如果当前 SentencePiece 模型文件路径和目标保存路径不相同且当前路径下存在 SentencePiece 模型文件，复制文件
        if os.path.abspath(self.spm_file) != os.path.abspath(spm_save_path) and os.path.isfile(self.spm_file):
            copyfile(self.spm_file, spm_save_path)
        # 如果当前路径下不存在 SentencePiece 模型文件，将模型序列化后写入目标保存路径
        elif not os.path.isfile(self.spm_file):
            with open(spm_save_path, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回词汇表和 SentencePiece 模型文件的保存路径
        return (str(vocab_save_path), str(spm_save_path))

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言，并为源语言设置特殊符号
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self.src_lang)
        # 准备 Seq2Seq 批次数据
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _build_translation_inputs(self, raw_inputs, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        # 检查是否提供了源语言和目标语言
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        # 设置源语言、生成模型输入，添加特殊符号后返回
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, **extra_kwargs)
        tgt_lang_id = self.get_lang_id(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def _switch_to_input_mode(self):
        # 切换到输入模式，为源语言设置特殊符号
        self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        # 切换到目标模式，为目标语言设置特殊符号
        self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        # 重置特殊符号为源语言设置，无前缀和后缀=[eos, src_lang_code]符号
        lang_token = self.get_lang_token(src_lang)
        self.cur_lang_id = self.lang_token_to_id[lang_token]
        self.prefix_tokens = [self.cur_lang_id]
        self.suffix_tokens = [self.eos_token_id]
    # 重置特殊标记为目标语言设置。无前缀和后缀=[eos, tgt_lang_code]。
    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        # 获取目标语言的语言标记
        lang_token = self.get_lang_token(tgt_lang)
        # 设置当前语言 ID 为目标语言的语言标记对应的 ID
        self.cur_lang_id = self.lang_token_to_id[lang_token]
        # 将前缀标记设置为当前语言 ID
        self.prefix_tokens = [self.cur_lang_id]
        # 将后缀标记设置为结束标记 ID
        self.suffix_tokens = [self.eos_token_id]

    # 获取指定语言的语言标记
    def get_lang_token(self, lang: str) -> str:
        return self.lang_code_to_token[lang]

    # 获取指定语言的语言 ID
    def get_lang_id(self, lang: str) -> int:
        # 获取指定语言的语言标记
        lang_token = self.get_lang_token(lang)
        # 返回指定语言的语言标记对应的 ID
        return self.lang_token_to_id[lang_token]
# 加载 SentencePiece 模型
def load_spm(path: str, sp_model_kwargs: Dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    # 使用给定的参数实例化 SentencePieceProcessor 对象
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    # 从指定路径加载 SentencePiece 模型文件
    spm.Load(str(path))
    # 返回加载后的 SentencePieceProcessor 对象
    return spm

# 从 JSON 文件加载数据
def load_json(path: str) -> Union[Dict, List]:
    # 以只读方式打开 JSON 文件
    with open(path, "r") as f:
        # 使用 json 模块加载文件中的数据
        return json.load(f)

# 将数据保存到 JSON 文件
def save_json(data, path: str) -> None:
    # 以写入方式打开 JSON 文件
    with open(path, "w") as f:
        # 使用 json 模块将数据写入文件，缩进为 2
        json.dump(data, f, indent=2)
```