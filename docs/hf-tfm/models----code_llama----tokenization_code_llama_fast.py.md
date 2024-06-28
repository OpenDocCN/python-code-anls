# `.\models\code_llama\tokenization_code_llama_fast.py`

```
# 导入所需的库和模块
import os  # 导入操作系统模块
from shutil import copyfile  # 导入复制文件函数
from typing import List, Optional, Tuple  # 导入类型提示相关的工具

from tokenizers import normalizers, processors  # 导入 tokenizers 库中的规范化和处理器模块

from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 从本地库导入预训练的快速分词器
from ...utils import is_sentencepiece_available, logging  # 导入判断是否可用 SentencePiece 和日志模块
from ...utils.versions import require_version  # 导入版本要求函数


require_version("tokenizers>=0.13.3")  # 确保 tokenizers 版本在 0.13.3 或以上

if is_sentencepiece_available():
    from .tokenization_code_llama import CodeLlamaTokenizer  # 如果可用，导入 CodeLlamaTokenizer
else:
    CodeLlamaTokenizer = None  # 否则设为 None

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}  # 词汇文件和分词器文件的名称定义

SPIECE_UNDERLINE = "▁"  # 定义特定的空格字符

B_INST, E_INST = "[INST]", "[/INST]"  # 定义实例开始和结束标记
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"  # 定义系统提示的开始和结束标记


# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on


class CodeLlamaTokenizerFast(PreTrainedTokenizerFast):
    """
    构建 Llama 快速分词器。基于字节级别的字节对编码。

    这里特别使用了 ByteFallback 和没有规范化。

    ```python
    >>> from transformers import CodeLlamaTokenizerFast

    >>> tokenizer = CodeLlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    >>> tokenizer.encode("Hello this is a test")
    [1, 15043, 445, 338, 263, 1243]
    ```

    如果要更改 `bos_token` 或 `eos_token`，请在初始化模型时指定它们，或调用 `tokenizer.update_post_processor()` 来确保后处理正确执行
    （否则编码序列的第一个标记和最后一个标记的值将不正确）。有关更多详细信息，请查看 [后处理器文档](https://huggingface.co/docs/tokenizers/api/post-processors)。

    该分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应该
    """
    # 定义默认的词汇文件名列表，用于加载模型
    vocab_files_names = VOCAB_FILES_NAMES

    # 指定慢速分词器的类，这里使用 CodeLlamaTokenizer
    slow_tokenizer_class = CodeLlamaTokenizer

    # 指定填充的位置在左侧
    padding_side = "left"

    # 指定模型的输入名称列表，包括输入的标识符和注意力掩码
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化函数，用于初始化一个自定义的Tokenizer对象
    def __init__(
        self,
        vocab_file=None,  # 词汇表文件路径，默认为None
        tokenizer_file=None,  # 分词器文件路径，默认为None
        clean_up_tokenization_spaces=False,  # 是否清理分词后的空格，默认为False
        unk_token="<unk>",  # 未知标记，默认为"<unk>"
        bos_token="<s>",  # 开始标记，默认为"<s>"
        eos_token="</s>",  # 结束标记，默认为"</s>"
        prefix_token="▁<PRE>",  # 前缀标记，默认为"▁<PRE>"
        middle_token="▁<MID>",  # 中间标记，默认为"▁<MID>"
        suffix_token="▁<SUF>",  # 后缀标记，默认为"▁<SUF>"
        eot_token="▁<EOT>",  # 结束标记，默认为"▁<EOT>"
        fill_token="<FILL_ME>",  # 填充标记，默认为"<FILL_ME>"
        additional_special_tokens=None,  # 额外的特殊标记列表，默认为None
        add_bos_token=True,  # 是否添加开始标记，默认为True
        add_eos_token=False,  # 是否添加结束标记，默认为False
        use_default_system_prompt=False,  # 是否使用默认系统提示，默认为False
        **kwargs,
    ):
        # 标记需要特别处理的特殊标记
        additional_special_tokens = additional_special_tokens or []
        for token in [prefix_token, middle_token, suffix_token, eot_token]:
            additional_special_tokens += [token] if token is not None else []
        # 记录是否使用默认系统提示
        self.use_default_system_prompt = use_default_system_prompt

        # 调用父类的初始化方法，传递所有参数
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            additional_special_tokens=additional_special_tokens,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            prefix_token=prefix_token,
            middle_token=middle_token,
            suffix_token=suffix_token,
            eot_token=eot_token,
            fill_token=fill_token,
            use_default_system_prompt=use_default_system_prompt,
            **kwargs,
        )
        
        # 初始化是否添加开始标记和结束标记的标志位
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        
        # 更新后处理器
        self.update_post_processor()

        # 记录词汇表文件路径
        self.vocab_file = vocab_file

        # 记录各种特殊标记的值
        self._prefix_token = prefix_token
        self._middle_token = middle_token
        self._suffix_token = suffix_token
        self._eot_token = eot_token
        self.fill_token = fill_token

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查词汇表文件是否存在，从而判断是否可以保存慢速分词器
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 从 transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.update_post_processor 复制
    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        # 获取当前的 `bos_token` 和 `bos_token_id`
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        # 如果 `bos_token` 为 None 且需要添加 `bos_token`，则引发数值错误
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        # 获取当前的 `eos_token` 和 `eos_token_id`
        eos = self.eos_token
        eos_token_id = self.eos_token_id
        # 如果 `eos_token` 为 None 且需要添加 `eos_token`，则引发数值错误
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        # 构建单句和双句模板，包含 `bos_token` 和 `eos_token`
        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        # 准备特殊标记列表，包括 `bos_token` 和 `eos_token`，用于后处理器
        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        
        # 更新 tokenizer 的后处理器使用新的模板和特殊标记
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def prefix_token(self):
        return self._prefix_token

    @property
    def prefix_id(self):
        # 如果 `_prefix_token` 为 None，则返回 None，否则将 `_prefix_token` 转换为对应的 id
        if self._prefix_token is None:
            return None
        return self.convert_tokens_to_ids(self.prefix_token)

    @property
    def middle_token(self):
        return self._middle_token

    @property
    def middle_id(self):
        # 如果 `_middle_token` 为 None，则返回 None，否则将 `_middle_token` 转换为对应的 id
        if self._middle_token is None:
            return None
        return self.convert_tokens_to_ids(self.middle_token)

    @property
    def suffix_token(self):
        return self._suffix_token

    @property
    def suffix_id(self):
        # 如果 `_suffix_token` 为 None，则返回 None，否则将 `_suffix_token` 转换为对应的 id
        if self._suffix_token is None:
            return None
        return self.convert_tokens_to_ids(self.suffix_token)

    @property
    def eot_id(self):
        # 如果 `_eot_token` 为 None，则返回 None，否则将 `_eot_token` 转换为对应的 id
        if self._eot_token is None:
            return None
        return self.convert_tokens_to_ids(self.eot_token)

    @property
    def eot_token(self):
        return self._eot_token

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        # 设置 `_add_eos_token` 的值，然后更新后处理器
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        # 设置 `_add_bos_token` 的值，然后更新后处理器
        self._add_bos_token = value
        self.update_post_processor()
    def set_infilling_processor(self, reset, suffix_first=False, add_special_tokens=True):
        """
        Updates the normalizer to ensure the prompt format for `infilling` is respected. The infilling format is as follows:
        if `suffix_first`:
            " <PRE> <SUF>{suf} <MID> {pre}"
        else:
            " <PRE> {pre} <SUF>{suf} <MID>"

        If `reset` is `True`, resets `normalizer` and `post_processor` to their default behaviors:
        normalizer adds a prefix space, post_processor adds a `bos_token`.

        Args:
            reset (bool): Indicates whether to reset the processors.
            suffix_first (bool, optional): Flag indicating the order of suffix and prefix in the format.
            add_special_tokens (bool, optional): Whether to add special tokens.

        Returns:
            None
        """
        # Resetting the processors if `reset` is `True`
        if reset:
            self._tokenizer.normalizer = normalizers.Sequence(
                [
                    normalizers.Prepend(prepend="▁"),  # Add a prefix space if resetting
                    normalizers.Replace(pattern=" ", content="▁"),  # Replace spaces with underscores
                ]
            )
            # Update post processor
            self.update_post_processor()
            return

        # Setting normalizer to replace spaces with underscores
        self._tokenizer.normalizer = normalizers.Replace(pattern=" ", content="▁")

        # Building `pair` and `special_tokens` based on `suffix_first` flag
        pair = [self.bos_token] if self.add_bos_token and add_special_tokens else []
        special_tokens = [(self.bos_token, self.bos_token_id)] if self.add_bos_token and add_special_tokens else []
        
        if suffix_first:
            # Format as " <PRE> <SUF>{suf} <MID> {pre}"
            pair += [self.prefix_token, self.suffix_token, "$B", self.middle_token, "$A"]
            special_tokens += [
                (self.prefix_token, self.prefix_id),
                (self.suffix_token, self.suffix_id),
                (self.middle_token, self.middle_id),
            ]
        else:
            # Format as " <PRE> {pre} <SUF>{suf} <MID>"
            pair += [self.prefix_token, "$A", self.suffix_token, "$B", self.middle_token]
            special_tokens += [
                (self.prefix_token, self.prefix_id),
                (self.suffix_token, self.suffix_id),
                (self.middle_token, self.middle_id),
            ]

        # Adding `eos_token` if required
        if self.add_eos_token and add_special_tokens:
            pair += [self.eos_token]
            special_tokens += [(self.eos_token, self.eos_token_id)]

        # Setting `post_processor` using TemplateProcessing
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single="$A", pair=pair, special_tokens=special_tokens
        )
    def encode_plus(self, text, text_pair=None, suffix_first=False, add_special_tokens=True, **kwargs):
        # 用于确保输入预处理在 Rust 外完成的一个小技巧
        text_pair = kwargs.pop("suffix", text_pair)
        # 如果存在填充标记并且在文本中找到了该标记但没有提供 text_pair，则将 text 拆分成 text 和 text_pair
        if self.fill_token is not None and self.fill_token in text and text_pair is None:
            text, text_pair = text.split(self.fill_token)

        # 如果 text_pair 为 None 或者长度小于1，则调用父类方法返回编码结果
        if text_pair is None or len(text_pair) < 1:
            return super().encode_plus(text, text_pair, add_special_tokens=add_special_tokens, **kwargs)

        # 如果 self.prefix_id, self.middle_id, self.suffix_id 中有任何一个为 None，则抛出 ValueError 异常
        if None in (self.prefix_id, self.middle_id, self.suffix_id):
            raise ValueError(
                "Then input includes a `prefix` and a `suffix` used for the infilling task,"
                " the `prefix_id, middle_id, suffix_id` must all be initialized. Current"
                f" values : {self.prefix_id, self.middle_id, self.suffix_id}"
            )

        # 设置 infilling 处理器，根据 suffix_first 和 add_special_tokens 参数决定是否添加特殊标记
        self.set_infilling_processor(False, suffix_first=suffix_first, add_special_tokens=add_special_tokens)
        # 调用父类方法编码 text 和 text_pair，并返回结果 tokens
        tokens = super().encode_plus(" " + text, text_pair=text_pair, add_special_tokens=True, **kwargs)
        # 恢复默认的 infilling 处理器设置
        self.set_infilling_processor(True)
        return tokens

    # 从 transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.save_vocabulary 复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速 tokenizer 的词汇表，则抛出 ValueError 异常
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存路径不是目录，则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 确定输出的词汇表文件路径，并复制当前词汇表文件到该路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与目标路径不同，则执行复制操作
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    @property
    # 从 transformers.models.llama.tokenization_llama.LlamaTokenizer.default_chat_template 复制而来
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        # 此方法应返回将特殊标记添加到输入 token_ids_0 和 token_ids_1 中的结果
    def set_lang(self, src_lang_code: int, tgt_lang_code: int = None) -> List[int]:
        """
        通过连接和添加特殊标记，从序列或序列对构建用于序列分类任务的模型输入。特殊标记取决于调用 set_lang。

        对于 NLLB 序列，其格式如下，其中 `X` 表示序列：

        - `input_ids`（用于编码器）：`X [eos, src_lang_code]`
        - `decoder_input_ids`（用于解码器）：`X [eos, tgt_lang_code]`

        BOS 永远不会被使用。序列对不是预期的使用情况，但将会在没有分隔符的情况下处理。

        Args:
            token_ids_0 (`List[int]`):
                要添加特殊标记的 ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                可选的第二个 ID 列表，用于序列对。

        Returns:
            `List[int]`: 包含适当特殊标记的输入 ID 列表。
        """
        if token_ids_1 is None:
            # 如果没有第二个序列对，返回加上特殊标记的第一个序列的输入 ID 列表
            return self.bos_token_id + token_ids_0 + self.eos_token_id
        # 如果有第二个序列对，返回加上特殊标记的两个序列的输入 ID 列表
        return self.bos_token_id + token_ids_0 + token_ids_1 + self.eos_token_id
```