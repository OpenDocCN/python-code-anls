# `.\models\mbart50\tokenization_mbart50_fast.py`

```py
# coding=utf-8
# 设置文件编码为 UTF-8，确保正确处理各种字符集
# Copyright 2021 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
# 版权声明，指明代码版权归属
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 进行许可
# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用此文件
# You may obtain a copy of the License at
# 详细许可证信息可在下述链接获取
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，本软件以“原样”分发，无论明示或暗示，均不包含任何担保或条件
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 根据许可证“原样”分发软件，不提供任何担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看许可证以了解权限和限制详情

import os
# 导入操作系统模块，用于与操作系统交互
from shutil import copyfile
# 导入 shutil 模块的 copyfile 函数，用于文件复制操作
from typing import List, Optional, Tuple
# 导入 typing 模块，用于类型提示

from tokenizers import processors
# 从 tokenizers 库中导入 processors 模块

from ...tokenization_utils import AddedToken, BatchEncoding
# 从 tokenization_utils 模块中导入 AddedToken 和 BatchEncoding 类
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从 tokenization_utils_fast 模块中导入 PreTrainedTokenizerFast 类
from ...utils import is_sentencepiece_available, logging
# 从 utils 模块中导入 is_sentencepiece_available 和 logging 函数

if is_sentencepiece_available():
    from .tokenization_mbart50 import MBart50Tokenizer
else:
    MBart50Tokenizer = None
# 如果 sentencepiece 库可用，则从 tokenization_mbart50 模块导入 MBart50Tokenizer 类，否则将 MBart50Tokenizer 设置为 None

logger = logging.get_logger(__name__)
# 获取当前模块的 logger 实例

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}
# 定义词汇表文件名字典，包含 "vocab_file" 和 "tokenizer_file" 键

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-50-one-to-many-mmt": (
            "https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt/resolve/main/sentencepiece.bpe.model"
        ),
    },
    "tokenizer_file": {
        "facebook/mbart-large-50-one-to-many-mmt": (
            "https://huggingface.co/facebook/mbart-large-50-one-to-many-mmt/resolve/main/tokenizer.json"
        ),
    },
}
# 预训练词汇文件映射，包含 "vocab_file" 和 "tokenizer_file" 键，以及对应的 URL

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-50-one-to-many-mmt": 1024,
}
# 预训练位置嵌入大小，以模型名称为键，大小为值的字典

FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]
# Fairseq 语言代码列表，用于支持多种语言和区域设置
    # 导入 MBart50TokenizerFast 类，该类用于处理 MBart 系列模型的快速分词和编码
    from transformers import MBart50TokenizerFast

    # 定义一个 MBart50TokenizerFast 类的实例，并从预训练模型 "facebook/mbart-large-50" 加载
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")

    # 定义一个源文本字符串
    src_text = " UN Chief Says There Is No Military Solution in Syria"

    # 定义一个目标文本字符串
    tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"

    # 使用 tokenizer 处理源文本和目标文本，返回 PyTorch 张量格式的模型输入
    model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")

    # model(**model_inputs) 应该能够正常工作，此处提供了一个示例用法
    # 定义一个特殊的 Token，其行为类似于普通单词，即在其前包含空格
    mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
    
    # 设定额外的特殊 Token，如果不存在则创建空列表并添加 Fairseq 语言代码
    kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
    kwargs["additional_special_tokens"] += [
        code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
    ]
    
    # 调用父类的初始化方法，传入参数
    super().__init__(
        vocab_file,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        tokenizer_file=tokenizer_file,
        eos_token=eos_token,
        sep_token=sep_token,
        cls_token=cls_token,
        unk_token=unk_token,
        pad_token=pad_token,
        mask_token=mask_token,
        **kwargs,
    )
    
    # 设置属性 vocab_file 为传入的 vocab_file
    self.vocab_file = vocab_file
    
    # 设置属性 lang_code_to_id，循环 FAIRSEQ_LANGUAGE_CODES 并将其转换为对应的 id
    self.lang_code_to_id = {
        lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
    }
    
    # 设置属性 _src_lang 为传入的 src_lang，如果为 None 则设置为 "en_XX"
    self._src_lang = src_lang if src_lang is not None else "en_XX"
    # 设置属性 tgt_lang 为传入的 tgt_lang
    self.tgt_lang = tgt_lang
    # 设置属性 cur_lang_code_id 为 lang_code_to_id 的 _src_lang 对应的值
    self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
    # 调用方法 set_src_lang_special_tokens 并传入 _src_lang
    self.set_src_lang_special_tokens(self._src_lang)
    
    # 定义属性 can_save_slow_tokenizer，判断是否存在 vocab_file
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False
    
    # 定义属性 src_lang，返回 _src_lang
    @property
    def src_lang(self) -> str:
        return self._src_lang
    
    # 定义属性 src_lang 的 setter 方法，将传入的值设置给 _src_lang，并调用方法 set_src_lang_special_tokens 并传入新值
    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)
    
    # 定义方法 build_inputs_with_special_tokens，用于构建带有特殊 Token 的输入序列
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. The special tokens depend on calling set_lang.
    
        An MBART-50 sequence has the following format, where `X` represents the sequence:
    
        - `input_ids` (for encoder) `[src_lang_code] X [eos]`
        - `labels`: (for decoder) `[tgt_lang_code] X [eos]`
    
        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.
    
        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
    
        Returns:
            `List[int]`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # 如果 token_ids_1 不为空，则将两个 token_ids 连接并加上特殊 Token
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言属性
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # 调用父类方法，准备序列到序列的批次编码
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    def _switch_to_input_mode(self):
        # 切换到输入模式，设置特殊的源语言令牌
        return self.set_src_lang_special_tokens(self.src_lang)

    def _switch_to_target_mode(self):
        # 切换到目标模式，设置特殊的目标语言令牌
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        # 将当前语言代码转换为对应的 ID
        self.cur_lang_code_id = self.convert_tokens_to_ids(src_lang)
        # 设置前缀特殊令牌为源语言代码 ID，后缀特殊令牌为 EOS 令牌 ID
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

        # 将 ID 转换为对应的字符串表示
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新分词器的后处理器，使用模板处理器设置特殊令牌
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[src_lang_code] and suffix=[eos]."""
        # 将当前语言代码转换为对应的 ID
        self.cur_lang_code_id = self.convert_tokens_to_ids(tgt_lang)
        # 设置前缀特殊令牌为目标语言代码 ID，后缀特殊令牌为 EOS 令牌 ID
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

        # 将 ID 转换为对应的字符串表示
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新分词器的后处理器，使用模板处理器设置特殊令牌
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        # 检查是否提供了源语言和目标语言，若未提供则抛出异常
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        # 设置当前的源语言属性
        self.src_lang = src_lang
        # 调用自身，使用指定参数调用模型，添加特殊令牌并返回结果
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        # 将目标语言转换为对应的 ID
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        # 将强制的 BOS 令牌 ID 添加到输入中
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs
    # 定义一个方法用于保存词汇表到文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果当前快速分词器无法提供保存慢速分词器所需的信息，则引发数值错误异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则记录错误日志并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        # 构建输出词汇表文件路径，根据前缀确定文件名，使用全局常量 VOCAB_FILES_NAMES 中的键来生成文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与目标输出路径不一致，则复制当前词汇表文件到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回一个包含输出词汇表文件路径的元组
        return (out_vocab_file,)
```