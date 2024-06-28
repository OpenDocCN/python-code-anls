# `.\models\mbart\tokenization_mbart_fast.py`

```py
# coding=utf-8
# 设置文件编码为 UTF-8

# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
# 版权声明，版权归Facebook AI研究团队和HuggingFace公司所有。

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版本授权许可

# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# distributed under the License is distributed on an "AS IS" BASIS, # 除非适用法律要求或书面同意，否则按“原样”分发软件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 无论是明示的还是暗示的，软件都不附带任何保证或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看许可证以了解具体的语言授权和限制

import os
# 导入操作系统模块

from shutil import copyfile
# 导入文件复制函数copyfile

from typing import List, Optional, Tuple
# 导入类型提示：List（列表），Optional（可选类型），Tuple（元组）

from tokenizers import processors
# 从tokenizers模块导入processors

from ...tokenization_utils import AddedToken, BatchEncoding
# 导入tokenization_utils模块中的AddedToken和BatchEncoding类

from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从tokenization_utils_fast模块导入PreTrainedTokenizerFast类

from ...utils import is_sentencepiece_available, logging
# 从utils模块导入is_sentencepiece_available函数和logging对象

if is_sentencepiece_available():
    from .tokenization_mbart import MBartTokenizer
else:
    MBartTokenizer = None
# 如果sentencepiece可用，则导入.tokenization_mbart模块中的MBartTokenizer类，否则将MBartTokenizer设为None

logger = logging.get_logger(__name__)
# 使用logging模块获取与当前模块名对应的logger对象

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}
# 设置字典，指定词汇文件和标记器文件的名称

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/mbart-large-en-ro": (
            "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/sentencepiece.bpe.model"
        ),
        "facebook/mbart-large-cc25": (
            "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/sentencepiece.bpe.model"
        ),
    },
    "tokenizer_file": {
        "facebook/mbart-large-en-ro": "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/tokenizer.json",
        "facebook/mbart-large-cc25": "https://huggingface.co/facebook/mbart-large-cc25/resolve/main/tokenizer.json",
    },
}
# 预训练模型的词汇文件和标记器文件的映射字典，指定了各个模型的下载链接

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/mbart-large-en-ro": 1024,
    "facebook/mbart-large-cc25": 1024,
}
# 预训练模型的位置嵌入大小字典，指定了各个模型的位置嵌入大小

FAIRSEQ_LANGUAGE_CODES = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN"]  # fmt: skip
# FAIRSEQ语言代码列表，指定了支持的语言代码

class MBartTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" MBART tokenizer (backed by HuggingFace's *tokenizers* library). Based on
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    The tokenization method is `<tokens> <eos> <language code>` for source language documents, and `<language code>
    <tokens> <eos>` for target language documents.

    Examples:

    ```
    >>> from transformers import MBartTokenizerFast

    >>> tokenizer = MBartTokenizerFast.from_pretrained(

    ```

    注释：
    创建一个“快速”MBART标记器（由HuggingFace的*tokenizers*库支持）。基于BPE模型。
    继承自PreTrainedTokenizerFast类，该类包含大多数主要方法。用户应参考此超类以获取有关这些方法的更多信息。
    用于源语言文档的标记化方法是`<tokens> <eos> <language code>`，用于目标语言文档的是`<language code> <tokens> <eos>`。
    ```
    # 定义一个类，用于处理 MBart 模型的特定 tokenizer
    class MBartTokenizer:
        # 类属性：定义一些常量
        vocab_files_names = VOCAB_FILES_NAMES  # MBartTokenizer 实例的词汇文件名
        max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES  # 预训练位置嵌入大小的最大模型输入尺寸
        pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP  # 预训练词汇文件映射表
        model_input_names = ["input_ids", "attention_mask"]  # 模型输入的名称列表
        slow_tokenizer_class = MBartTokenizer  # MBartTokenizer 类本身作为慢速 tokenizer 类
    
        prefix_tokens: List[int] = []  # 前缀 token 列表
        suffix_tokens: List[int] = []  # 后缀 token 列表
    
        def __init__(
            self,
            vocab_file=None,
            tokenizer_file=None,
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            src_lang=None,
            tgt_lang=None,
            additional_special_tokens=None,
            **kwargs,
        ):
            # 如果 mask_token 是字符串，将其转换为 AddedToken 对象，处理前后空格
            mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
            
            # 复制 FAIRSEQ_LANGUAGE_CODES 到 _additional_special_tokens
            _additional_special_tokens = FAIRSEQ_LANGUAGE_CODES.copy()
    
            # 如果有额外的特殊 token，且它们不在 _additional_special_tokens 中，则添加到 _additional_special_tokens
            if additional_special_tokens is not None:
                _additional_special_tokens.extend(
                    [t for t in additional_special_tokens if t not in _additional_special_tokens]
                )
    
            # 调用父类的初始化方法，设置实例属性
            super().__init__(
                vocab_file=vocab_file,
                tokenizer_file=tokenizer_file,
                bos_token=bos_token,
                eos_token=eos_token,
                sep_token=sep_token,
                cls_token=cls_token,
                unk_token=unk_token,
                pad_token=pad_token,
                mask_token=mask_token,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                additional_special_tokens=_additional_special_tokens,
                **kwargs,
            )
    
            # 设置实例属性：词汇文件名、语言代码到 ID 的映射、当前源语言代码、目标语言和特殊 token
            self.vocab_file = vocab_file
            self.lang_code_to_id = {
                lang_code: self.convert_tokens_to_ids(lang_code) for lang_code in FAIRSEQ_LANGUAGE_CODES
            }
            self._src_lang = src_lang if src_lang is not None else "en_XX"
            self.cur_lang_code = self.convert_tokens_to_ids(self._src_lang)
            self.tgt_lang = tgt_lang
            self.set_src_lang_special_tokens(self._src_lang)
    
        @property
        def can_save_slow_tokenizer(self) -> bool:
            # 检查词汇文件是否存在，用于判断是否可以保存慢速 tokenizer
            return os.path.isfile(self.vocab_file) if self.vocab_file else False
    
        @property
        def src_lang(self) -> str:
            # 返回当前源语言
            return self._src_lang
    
        @src_lang.setter
        def src_lang(self, new_src_lang: str) -> None:
            # 设置新的源语言，并更新特殊 token
            self._src_lang = new_src_lang
            self.set_src_lang_special_tokens(self._src_lang)
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从一个或一对序列构建模型输入，用于序列分类任务，通过连接和添加特殊标记来完成。特殊标记取决于调用set_lang。

        一个MBART序列具有以下格式，其中 `X` 表示序列：

        - `input_ids`（用于编码器）：`X [eos, src_lang_code]`
        - `decoder_input_ids`：（用于解码器）：`X [eos, tgt_lang_code]`

        BOS 永远不会被使用。序列对不是预期的使用情况，但它们会在没有分隔符的情况下处理。

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的可选 ID 列表，用于序列对。

        Returns:
            `List[int]`: 带有适当特殊标记的输入 ID 列表。
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # 对于序列对，我们不希望处理它们，但为了 API 一致性保留了处理序列对的逻辑
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建一个在序列对分类任务中使用的掩码。mBART 不使用标记类型 ID，因此返回一个零列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                第二个序列的可选 ID 列表，用于序列对。

        Returns:
            `List[int]`: 零列表。
        """

        sep = [self.sep_token_id]  # 分隔符标记的 ID
        cls = [self.cls_token_id]  # 类别标记的 ID

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]  # 返回一个全零列表，长度为特殊标记和输入长度的总和
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]  # 返回一个全零列表，长度包括特殊标记和两个序列的总和

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """由翻译管道使用，准备生成函数的输入"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang  # 设置源语言属性
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)  # 使用模型处理原始输入，添加特殊标记
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)  # 将目标语言转换为其对应的 ID
        inputs["forced_bos_token_id"] = tgt_lang_id  # 在输入中添加强制的 BOS 标记 ID
        return inputs  # 返回处理后的输入
    # 准备用于序列到序列模型的批处理数据，设置源语言和目标语言，默认源语言为英语，目标语言为罗马尼亚语
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang  # 设置源语言
        self.tgt_lang = tgt_lang  # 设置目标语言
        # 调用父类方法准备序列到序列模型的批处理数据
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    # 切换到输入模式，使用当前源语言设置特殊标记
    def _switch_to_input_mode(self):
        return self.set_src_lang_special_tokens(self.src_lang)

    # 切换到目标模式，使用当前目标语言设置特殊标记
    def _switch_to_target_mode(self):
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    # 设置特殊标记为当前源语言的设定，包括结束标记和源语言编码
    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting. No prefix and suffix=[eos, src_lang_code]."""
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)  # 转换源语言为对应的语言编码
        self.prefix_tokens = []  # 重置前缀标记为空列表
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]  # 设置后缀标记为结束标记和当前语言编码

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)  # 将前缀标记转换为字符串形式
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)  # 将后缀标记转换为字符串形式

        # 设置 Tokenizer 的后处理器，使用模板处理
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    # 设置特殊标记为当前目标语言的设定，包括结束标记和目标语言编码
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target language setting. No prefix and suffix=[eos, tgt_lang_code]."""
        self.cur_lang_code = self.convert_tokens_to_ids(lang)  # 转换目标语言为对应的语言编码
        self.prefix_tokens = []  # 重置前缀标记为空列表
        self.suffix_tokens = [self.eos_token_id, self.cur_lang_code]  # 设置后缀标记为结束标记和当前语言编码

        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)  # 将前缀标记转换为字符串形式
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)  # 将后缀标记转换为字符串形式

        # 设置 Tokenizer 的后处理器，使用模板处理
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )
    # 定义一个保存词汇表的方法，接受一个保存目录路径和可选的文件名前缀作为参数，并返回一个包含文件路径字符串的元组
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果当前快速分词器不能保存慢速分词器所需的信息，则引发值错误异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return
        
        # 构造输出词汇文件的路径，结合文件名前缀和常量中定义的词汇文件名
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇文件的绝对路径与输出词汇文件的绝对路径不同，则复制当前词汇文件到输出词汇文件
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        # 返回包含输出词汇文件路径的元组
        return (out_vocab_file,)
```