# `.\models\code_llama\tokenization_code_llama_fast.py`

```
# 导入所需的包和模块
import os
from shutil import copyfile
from typing import List, Optional, Tuple

# 导入 Tokenizers 中需要使用的特定类和函数
from tokenizers import normalizers, processors

# 导入其他相关模块
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
from ...utils.versions import require_version

# 检查 Tokenizers 的版本要求
require_version("tokenizers>=0.13.3")

# 根据当前环境是否具备 sentencepiece 库的功能性，决定是否导入相关的 tokenization_code_llama 模块
if is_sentencepiece_available():
    from .tokenization_code_llama import CodeLlamaTokenizer
else:
    CodeLlamaTokenizer = None

# 设置日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}

# 设置 SPIECE_UNDERLINE 为特定字符串
SPIECE_UNDERLINE = "▁"

# 定义一些特定标记的起始和结束字符串
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# 格式化渲染默认的系统提示文本
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""

# 定义 Llama 快速分词器的类
class CodeLlamaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.
    
    This uses notably ByteFallback and no normalization.
    
    ```python
    >>> from transformers import CodeLlamaTokenizerFast
    
    >>> tokenizer = CodeLlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    >>> tokenizer.encode("Hello this is a test")
    [1, 15043, 445, 338, 263, 1243]
    ```
    
    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.
    
    
    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should

    ```
    """
    # 可以参考这个超类获取有关这些方法的更多信息。默认配置与 [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf/blob/main/tokenizer_config.json) 相匹配，在支持提示填充的情况下。

    Args:
        vocab_file (`str`, *optional*):
            包含实例化分词器所需词汇表的 [SentencePiece](https://github.com/google/sentencepiece) 文件（通常具有 .model 扩展名）。
        tokenizer_file (`str`, *optional*):
            包含加载分词器所需的所有内容的 [tokenizers](https://github.com/huggingface/tokenizers) 文件（通常具有 .json 扩展名）。
        clean_up_tokenization_spaces (`str`, *optional*, defaults to `False`):
            是否在解码后清理空格，清理包括删除可能的额外空格等伪影。
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            未知标记。词汇表中没有的标记无法转换为 ID，并且将被设置为此标记。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            在预训练期间使用的序列开始标记。可用作序列分类器标记。
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            序列结束标记。
        prefix_token (`str`, *optional*, defaults to `"▁<PRE>"`):
            用于填充的前缀标记。
        middle_token (`str`, *optional*, defaults to `"▁<MID>"`):
            用于填充的中间标记。
        suffix_token (`str`, *optional*, defaults to `"▁<SUF>"`):
            用于填充的后缀标记。
        eot_token (`str`, *optional*, defaults to `"▁<EOT>"`):
            用于填充的文本结束标记。
        fill_token (`str`, *optional*, defaults to `"<FILL_ME>"`):
            用于在前缀和后缀之间拆分输入的标记。
        additional_special_tokens (`List[str]`, *optional*):
            分词器使用的额外特殊标记。
        add_bos_token (`bool`, *optional*, defaults to `True`):
            是否在序列开头添加一个序列开始标记。
        add_eos_token (`bool`, *optional*, defaults to `False`):
            是否在序列结尾添加一个序列结束标记。
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            是否使用 Llama 的默认系统提示。

    """

    # 词汇表文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 慢速分词器类
    slow_tokenizer_class = CodeLlamaTokenizer
    # 填充位置设为左侧
    padding_side = "left"
    # 模型输入名称
    model_input_names = ["input_ids", "attention_mask"]
    # 初始化函数，包括设置词汇文件（vocab_file）、分词器文件（tokenizer_file）等参数
    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        prefix_token="▁<PRE>",
        middle_token="▁<MID>",
        suffix_token="▁<SUF>",
        eot_token="▁<EOT>",
        fill_token="<FILL_ME>",
        additional_special_tokens=None,
        add_bos_token=True,
        add_eos_token=False,
        use_default_system_prompt=False,
        **kwargs,
    ):
        # 标记特殊标记以跳过它们
        additional_special_tokens = additional_special_tokens or []
        for token in [prefix_token, middle_token, suffix_token, eot_token]:
            additional_special_tokens += [token] if token is not None else []
        # 设置是否使用默认系统提示
        self.use_default_system_prompt = use_default_system_prompt

        # 调用父类的初始化函数，并传入相应参数
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
        # 设置是否添加 bos_token 和 eos_token
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        # 更新后处理器
        self.update_post_processor()

        # 设置词汇文件
        self.vocab_file = vocab_file

        # 设置特殊标记
        self._prefix_token = prefix_token
        self._middle_token = middle_token
        self._suffix_token = suffix_token
        self._eot_token = eot_token
        self.fill_token = fill_token

    # 返回是否可以保存慢速分词器
    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 从 transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.update_post_processor 中复制过来的
    # 更新底层的后处理器，使用当前的`bos_token`和`eos_token`
    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        # 获取`bos_token`和`bos_token_id`
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        # 如果`add_bos_token`为True，但`bos_token`为None，则抛出数值错误
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        # 获取`eos_token`和`eos_token_id`
        eos = self.eos_token
        eos_token_id = self.eos_token_id
        # 如果`add_eos_token`为True，但`eos_token`为None，则抛出数值错误
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        # 根据`add_bos_token`和`add_eos_token`的值构建单个和成对的模板
        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        # 如果`add_bos_token`为True，将`bos_token`和`bos_token_id`添加到special_tokens列表中
        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        # 如果`add_eos_token`为True，将`eos_token`和`eos_token_id`添加到special_tokens列表中
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        # 使用模板处理器构建后处理器
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def prefix_token(self):
        return self._prefix_token

    @property
    def prefix_id(self):
        # 如果`_prefix_token`为None，返回None，否则将`_prefix_token`转换为id返回
        if self._prefix_token is None:
            return None
        return self.convert_tokens_to_ids(self.prefix_token)

    # 同理，获取中间token和id、后缀token和id、结束标记token和id的方法
    # 设置添加eos token和bos token的方法，同时更新处理器
    def set_infilling_processor(self, reset, suffix_first=False, add_special_tokens=True):
        """
        更新标准化器，确保 `infilling` 的提示格式得到遵守。infilling 格式如下：如果 suffix_first
            " <PRE> <SUF>{suf} <MID> {pre}"
        否则
            " <PRE> {pre} <SUF>{suf} <MID>"

        如果 `reset` 设为 `True`，则将 `normalizer` 和 `post_processor` 重置为它们的 "正常" 行为，即为标准化器添加前缀空格，为 `post_processor` 的输入文本添加 `bos_token`。
        """
        if reset:
            # 设置标准化器以确保提示格式的前缀空格被添加
            self._tokenizer.normalizer = normalizers.Sequence(
                [
                    normalizers.Prepend(prepend="▁"),
                    normalizers.Replace(pattern=" ", content="▁"),
                ]
            )
            # 更新后处理器
            self.update_post_processor()
            return

        # 设置标准化器以替换空格为特定标记
        self._tokenizer.normalizer = normalizers.Replace(pattern=" ", content="▁")
        # 创建用于标记对应 id 的初始列表
        pair = [self.bos_token] if self.add_bos_token and add_special_tokens else []
        special_tokens = [(self.bos_token, self.bos_token_id)] if self.add_bos_token and add_special_tokens else []
        if suffix_first:
            # 格式为 " <PRE> <SUF>{suf} <MID> {pre}"
            pair += [self.prefix_token, self.suffix_token, "$B", self.middle_token, "$A"]
            special_tokens += [
                (self.prefix_token, self.prefix_id),
                (self.suffix_token, self.suffix_id),
                (self.middle_token, self.middle_id),
            ]
        else:
            # 格式为 " <PRE> {pre} <SUF>{suf} <MID>"
            pair += [self.prefix_token, "$A", self.suffix_token, "$B", self.middle_token]
            special_tokens += [
                (self.prefix_token, self.prefix_id),
                (self.suffix_token, self.suffix_id),
                (self.middle_token, self.middle_id),
            ]

        if self.add_eos_token and add_special_tokens:
            pair += [self.eos_token]
            special_tokens += [(self.eos_token, self.eos_token_id)]
        # 设置后处理器以处理模板
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single="$A", pair=pair, special_tokens=special_tokens
        )
    # 根据输入的文本对进行编码，返回编码后的 token
    def encode_plus(self, text, text_pair=None, suffix_first=False, add_special_tokens=True, **kwargs):
        # 为了确保输入在 Rust 外部进行预处理，采取的 hack
        text_pair = kwargs.pop("suffix", text_pair)
        # 如果 fill_token 存在且在 text 中出现，并且 text_pair 为空，则将 text 拆分成 text_pair
        if self.fill_token is not None and self.fill_token in text and text_pair is None:
            text, text_pair = text.split(self.fill_token)

        # 如果 text_pair 为空或长度小于 1，则直接调用父类的 encode_plus 方法
        if text_pair is None or len(text_pair) < 1:
            return super().encode_plus(text, text_pair, add_special_tokens=add_special_tokens, **kwargs)

        # 如果 prefix_id, middle_id, suffix_id 有任何一个为 None，则抛出 ValueError
        if None in (self.prefix_id, self.middle_id, self.suffix_id):
            raise ValueError(
                "Then input includes a `prefix` and a `suffix` used for the infilling task,"
                " the `prefix_id, middle_id, suffix_id` must all be initialized. Current"
                f" values : {self.prefix_id, self.middle_id, self.suffix_id}"
            )

        # 设置 infilling_processor 为 False，处理 token
        self.set_infilling_processor(False, suffix_first=suffix_first, add_special_tokens=add_special_tokens)
        # 调用父类的 encode_plus 方法，处理 token
        tokens = super().encode_plus(" " + text, text_pair=text_pair, add_special_tokens=True, **kwargs)
        # 设置 infilling_processor 为 True
        self.set_infilling_processor(True)
        return tokens

    # 保存词汇表
    # 从 transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.LlamaTokenizerFast 类中复制
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表，则抛出 ValueError
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存目录不存在，则返回错误信息
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与目标路径不一致，则复制文件到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    # 构建带有特殊 token 的输入
    # 从 transformers.models.llama.tokenization_llama.LlamaTokenizer.default_chat_template 类中复制
    @property
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    # 定义一个方法，用于构建用于序列分类任务的模型输入，通过连接和添加特殊标记
    def build_model_inputs(self, token_ids_0: List[int], token_ids_1: List[int] = None) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. The special tokens depend on calling set_lang.

        An NLLB sequence has the following format, where `X` represents the sequence:

        - `input_ids` (for encoder) `X [eos, src_lang_code]`
        - `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int`: list of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        # 如果没有第二个列表传入，则返回带特殊标记的第一个列表
        if token_ids_1 is None:
            return self.bos_token_id + token_ids_0 + self.eos_token_id
        # 如果有第二个列表传入，则返回带特殊标记的两个列表连接后的结果
        return self.bos_token_id + token_ids_0 + token_ids_1 + self.eos_token_id
```