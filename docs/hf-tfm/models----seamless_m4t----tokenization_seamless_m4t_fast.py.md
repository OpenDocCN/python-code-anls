# `.\transformers\models\seamless_m4t\tokenization_seamless_m4t_fast.py`

```py
# 设置编码方式
# 版权声明
#
# 基于 Apache 许可证 2.0 发布的一款快速分词类，适用于 SeamlessM4T。
import os  # 导入 os 模块，用于处理文件路径操作
from shutil import copyfile  # 导入 copyfile 方法，用于复制文件
from typing import List, Optional, Tuple, Union  # 导入类型提示需要用到的类型
from tokenizers import processors  # 导入 tokenizers 模块的 processors

from ...tokenization_utils import (  # 从 tokenization_utils 模块导入 BatchEncoding, PreTokenizedInput, TextInput 类
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 从 tokenization_utils_fast 模块导入 PreTrainedTokenizerFast 类
from ...utils import PaddingStrategy, is_sentencepiece_available, logging  # 从 utils 模块导入 PaddingStrategy, is_sentencepiece_available, logging

# 如果 sentencepiece 可用
if is_sentencepiece_available():
    # 导入 tokenization_seamless_m4t 模块的 SeamlessM4TTokenizer 类
    from .tokenization_seamless_m4t import SeamlessM4TTokenizer
else:
    SeamlessM4TTokenizer = None  # 否则设置为 None

logger = logging.get_logger(__name__)  # 获取 logger 对象

# 定义词汇文件的名称字典
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

# 预训练的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/hf-seamless-m4t-medium": "https://huggingface.co/facebook/hf-seamless-m4t-medium/resolve/main/vocab.txt",
    },
    "tokenizer_file": {
        "facebook/hf-seamless-m4t-medium": "https://huggingface.co/facebook/hf-seamless-m4t-medium/resolve/main/tokenizer.json",
    },
}

# 预训练的位置嵌入大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/hf-seamless-m4t-medium": 2048,  # 设置 facebook/hf-seamless-m4t-medium 的位置嵌入大小为 2048
}


class SeamlessM4TTokenizerFast(PreTrainedTokenizerFast):  # 定义 SeamlessM4TTokenizerFast 类，继承自 PreTrainedTokenizerFast
    """
    构建一个“快速”的 SeamlessM4T 分词器（由 HuggingFace 的 *tokenizers* 库支持）。基于
    [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models)。

    该分词器继承自[`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应该
    参考这个超类以获取有关这些方法的更多信息。

    分词方法是 `<language code> <tokens> <eos>` 用于源语言文档，和 `<eos> <language
    code> <tokens> <eos>` 用于目标语言文档。
    """

    # 示例
    '''
    >>> from transformers import SeamlessM4TTokenizerFast

    >>> tokenizer = SeamlessM4TTokenizerFast.from_pretrained(
    ...     "facebook/hf-seamless-m4t-medium", src_lang="eng", tgt_lang="fra"
    ... )
    >>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
    >>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
    >>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
    '''
    Args:
        vocab_file (`str`, *optional*):
            词汇表文件的路径。
        tokenizer_file (`str`, *optional*):
            一个替代词汇表文件的令牌生成器文件的路径。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            在预训练过程中用作序列开头的令牌。可以用作序列分类令牌。

            <Tip>

            当使用特殊令牌构建序列时，`cls_token` 才是用作序列开头的令牌。

            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            序列结尾的令牌。

            <Tip>

            当使用特殊令牌构建序列时，`sep_token` 才是用作序列结尾的令牌。

            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            在多个序列组合成一个序列时使用的分隔符，例如用于序列分类或问题回答的两个序列。也用作用特殊令牌构建的序列的最后一个令牌。
        cls_token (`str`, *optional*, defaults to `"<s>"`):
            在进行序列分类（整个序列而不是每个令牌的分类）时使用的分类器令牌。这是使用特殊令牌构建序列时的第一个令牌。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知令牌。词汇表中不存在的令牌无法转换为ID，将被替换为这个令牌。
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            用于填充的令牌，例如当批处理不同长度的序列时。
        src_lang (`str`, *optional*, defaults to `"eng"`):
            用作翻译的源语言。
        tgt_lang (`str`, *optional*, defaults to `"fra"`):
            用作翻译的目标语言。
        additional_special_tokens (tuple or list of `str` or `tokenizers.AddedToken`, *optional*):
            附加特殊令牌的元组或列表。
    """

    # 词汇表文件名字的映射
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇表文件的映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 预训练位置嵌入大小的映射
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 慢速的令牌生成器类
    slow_tokenizer_class = SeamlessM4TTokenizer
    # 模型输入的名字列表
    model_input_names = ["input_ids", "attention_mask"]

    # 前缀令牌列表的初始化
    prefix_tokens: List[int] = []
    # 后缀令牌列表的初始化
    suffix_tokens: List[int] = []
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
        src_lang="eng",
        tgt_lang="fra",
        additional_special_tokens=None,
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file
        self._src_lang = f"__{src_lang}__" if "__" not in src_lang else src_lang
        self._tgt_lang = f"__{tgt_lang}__" if "__" not in tgt_lang else tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)
        self.set_tgt_lang_special_tokens(self._tgt_lang)



        # 创建一个新的对象实例
        def __init__(
            # 初始化函数的参数
            self,
            # 词汇表文件路径
            vocab_file=None,
            # 分词器文件路径
            tokenizer_file=None,
            # 句子开始标记
            bos_token="<s>",
            # 句子结束标记
            eos_token="</s>",
            # 句子分隔标记
            sep_token="</s>",
            # 分类标记
            cls_token="<s>",
            # 未知标记
            unk_token="<unk>",
            # 填充标记
            pad_token="<pad>",
            # 源语言默认为英语
            src_lang="eng",
            # 目标语言默认为法语
            tgt_lang="fra",
            # 额外特殊标记
            additional_special_tokens=None,
            # 其他关键字参数
            **kwargs,
        ):
        
        # 调用父类的初始化函数，将参数传递给父类
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        
        # 设置词汇表文件路径
        self.vocab_file = vocab_file
        # 将源语��添加到特殊标记中
        self._src_lang = f"__{src_lang}__" if "__" not in src_lang else src_lang
        # 将目标语言添加到特殊标记中
        self._tgt_lang = f"__{tgt_lang}__" if "__" not in tgt_lang else tgt_lang
        # 设置源语言的特殊标记
        self.set_src_lang_special_tokens(self._src_lang)
        # 设置目标语言的特殊标记
        self.set_tgt_lang_special_tokens(self._tgt_lang)



    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False



        # 返回是否可以保存缓慢的分词器
        @property
        def can_save_slow_tokenizer(self) -> bool:
            # 判断词汇表文件是否存在
            return os.path.isfile(self.vocab_file) if self.vocab_file else False


    @property
    # Copied from transformers.models.nllb.tokenization_nllb.NllbTokenizer.src_lang
    def src_lang(self) -> str:
        return self._src_lang



        # 返回源语言
        @property
        def src_lang(self) -> str:
            return self._src_lang



    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        if "__" not in new_src_lang:
            self._src_lang = f"__{new_src_lang}__"
        else:
            self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)


        # 设置源语言
        @src_lang.setter
        def src_lang(self, new_src_lang: str) -> None:
            # 检查源语言是否已经是特殊标记
            if "__" not in new_src_lang:
                self._src_lang = f"__{new_src_lang}__"
            else:
                self._src_lang = new_src_lang
            # 为源语言设置特殊标记
            self.set_src_lang_special_tokens(self._src_lang)



    @property
    def tgt_lang(self) -> str:
        return self._tgt_lang



        # 返回目标语言
        @property
        def tgt_lang(self) -> str:
            return self._tgt_lang



    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang: str) -> None:
        if "__" not in new_tgt_lang:
            self._tgt_lang = f"__{new_tgt_lang}__"
        else:
            self._tgt_lang = new_tgt_lang
        self.set_tgt_lang_special_tokens(self._tgt_lang)



        # 设置目标语言
        @tgt_lang.setter
        def tgt_lang(self, new_tgt_lang: str) -> None:
            # 检查目标语言是否已经是特殊标记
            if "__" not in new_tgt_lang:
                self._tgt_lang = f"__{new_tgt_lang}__"
            else:
                self._tgt_lang = new_tgt_lang
            # 为目标语言设置特殊标记
            self.set_tgt_lang_special_tokens(self._tgt_lang)



    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):



        # 使用特殊的标记构建输入
        def build_inputs_with_special_tokens(
            # 第一个句子的token ID列表
            self, token_ids_0: List[int],
            # 第二个句子的token ID列表（可选）
            token_ids_1: Optional[List[int]] = None
        ):
    def build_model_inputs(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """
        从一个序列或一个序列对构建模型输入，用于序列分类任务，通过连接和添加特殊标记。特殊标记取决于调用 set_lang。

        一个 SeamlessM4T 序列具有以下格式，其中 `X` 表示序列：

        - `input_ids`（用于编码器）`[src_lang_code] X [eos]`
        - `decoder_input_ids`：（用于解码器）`[eos, tgt_lang_code] X [eos]`

        BOS 从未被使用。序列对不是预期的使用情况，但它们将被处理，没有分隔符。

        参数:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                用于序列对的可选第二个 ID 列表。

        返回:
            `List[int]`: 具有适当特殊标记的 [输入 ID](../glossary#input-ids) 列表。
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # 我们不希望处理序列对，但为了 API 的一致性，保留对序列对的逻辑
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    # 从 transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast.create_token_type_ids_from_sequences 复制而来
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        从传递的两个序列创建一个用于序列对分类任务的掩码。nllb 不使用 token type ids，因此返回一个全零列表。

        参数:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *可选*):
                用于序列对的可选第二个 ID 列表。

        返回:
            `List[int]`: 全零列表。

        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        """由翻译流水线使用，用于为 generate 函数准备输入"""
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        if "__" not in tgt_lang:
            tgt_lang = f"__{tgt_lang}__"
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs
```  
    # 从transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast.prepare_seq2seq_batch中复制代码，并将"fra_Latn"->"fra"，"eng_Latn"->"eng"
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "eng",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "fra",
        **kwargs,
    ) -> BatchEncoding:
        # 设置源语言和目标语言
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        # 调用父类方法准备序列到序列的批处理
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    # 从transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast._switch_to_input_mode中复制代码
    def _switch_to_input_mode(self):
        # 切换至输入模式并设置源语言的特殊标记
        return self.set_src_lang_special_tokens(self.src_lang)

    # 从transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast._switch_to_target_mode中复制代码
    def _switch_to_target_mode(self):
        # 切换至目标模式并设置目标语言的特殊标记
        return self.set_tgt_lang_special_tokens(self.tgt_lang)

    def set_src_lang_special_tokens(self, src_lang) -> None:
        """Reset the special tokens to the source lang setting.
        Prefix=[src_lang_code], suffix = [eos]
        """
        # 设置当前语言代码为源语言代码，并检查是否在词汇表中存在
        self.cur_lang_code = self.convert_tokens_to_ids(src_lang)
        
        # 如果当前语言代码等于未知标记，记录警告信息
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`tgt_lang={src_lang}` has not be found in the `vocabulary`. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        # 更新初始化参数中的源语言
        self.init_kwargs["src_lang"] = src_lang

        # 设置前缀和后缀特殊标记
        self.prefix_tokens = [self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

        # 转换特殊标记为字符串
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新tokenizer的后处理器，将特殊标记作为模板处理
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )
    # 设置目标语言的特殊标记
    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        """Reset the special tokens to the target lang setting.
        Prefix=[eos, tgt_lang_code] and suffix=[eos].
        """
        # 将当前语言代码转换为对应的token id
        self.cur_lang_code = self.convert_tokens_to_ids(lang)

        # 若当前语言代码为未知token id，则记录警告
        if self.cur_lang_code == self.unk_token_id:
            logger.warning_once(
                f"`tgt_lang={lang}` has not be found in the `vocabulary`. Behaviour will probably be unexpected because the language token id will be replaced by the unknown token id."
            )

        # 更新初始化参数中的目标语言
        self.init_kwargs["tgt_lang"] = lang

        # 设置前缀标记为[eos, 当前语言代码]，后缀标记为[eos]
        self.prefix_tokens = [self.eos_token_id, self.cur_lang_code]
        self.suffix_tokens = [self.eos_token_id]

        # 将前缀标记和后缀标记转换为字符串
        prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # 更新tokenizer的后处理器
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=prefix_tokens_str + ["$A"] + suffix_tokens_str,
            pair=prefix_tokens_str + ["$A", "$B"] + suffix_tokens_str,
            special_tokens=list(zip(prefix_tokens_str + suffix_tokens_str, self.prefix_tokens + self.suffix_tokens)),
        )

    # 保存词汇表的方法
    # Copied from transformers.models.nllb.tokenization_nllb_fast.NllbTokenizerFast.save_vocabulary
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表信息，则抛出错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory.")
            return
        # 拼接输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与目标输出路径不同，则复制词汇表文件到输出目录
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    @classmethod
    # 从预训练模型中加载模型的方法
    def _from_pretrained(
        cls,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        token=None,
        cache_dir=None,
        local_files_only=False,
        _commit_hash=None,
        _is_local=False,
        **kwargs,
    # 覆盖父类的_from_pretrained方法，用于从预训练模型加载tokenizer
    def _from_pretrained(
        self,
        resolved_vocab_files,
        pretrained_model_name_or_path,
        init_configuration,
        *init_inputs,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        _commit_hash=_commit_hash,
        _is_local=_is_local,
        **kwargs,
    ):
        # 调用父类的_from_pretrained方法，返回tokenizer对象
        tokenizer = super()._from_pretrained(
            resolved_vocab_files,
            pretrained_model_name_or_path,
            init_configuration,
            *init_inputs,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            _commit_hash=_commit_hash,
            _is_local=_is_local,
            **kwargs,
        )

        # 确保在从预训练模型加载后也设置特殊语言标记
        tokenizer.set_src_lang_special_tokens(tokenizer._src_lang)
        tokenizer.set_tgt_lang_special_tokens(tokenizer._tgt_lang)

        # 返回tokenizer对象
        return tokenizer

    # 用于对文本进行tokenization的方法
    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        text_pair_target: Optional[
            Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
        ] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = 2,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        **kwargs,
```