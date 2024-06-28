# `.\models\gemma\tokenization_gemma_fast.py`

```
# 导入必要的模块和函数
import os  # 导入操作系统模块
from shutil import copyfile  # 从 shutil 模块导入 copyfile 函数
from typing import Optional, Tuple  # 导入类型提示相关的类和函数

from tokenizers import processors  # 导入 tokenizers 库中的 processors 模块

# 导入相对路径下的模块和函数
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入预训练的快速分词器
from ...utils import is_sentencepiece_available, logging  # 导入工具函数和日志模块
from ...utils.versions import require_version  # 导入版本控制相关函数

# 确保 tokenizers 的版本符合要求
require_version("tokenizers>=0.13.3")

# 如果系统支持 sentencepiece，则导入 GemmaTokenizer
if is_sentencepiece_available():
    from .tokenization_gemma import GemmaTokenizer
else:
    GemmaTokenizer = None  # 否则将 GemmaTokenizer 设置为 None

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}


class GemmaTokenizerFast(PreTrainedTokenizerFast):
    """
    构建一个快速的 Gemma 分词器，基于字节级别的 Byte-Pair-Encoding。

    这个分词器使用了 ByteFallback 和没有前缀空格。标准化应用于将 `" "` 替换为 `"▁"`

    ```python
    >>> from transformers import GemmaTokenizerFast

    >>> tokenizer = GemmaTokenizerFast.from_pretrained("hf-internal-testing/dummy-gemma")
    >>> tokenizer.encode("Hello this is a test")
    [2, 4521, 736, 603, 476, 2121]
    ```

    如果您想要更改 `bos_token` 或 `eos_token`，请确保在初始化模型时指定它们，或者调用 `tokenizer.update_post_processor()` 
    确保后处理正确完成（否则编码序列的第一个令牌和最后一个令牌的值将不正确）。更多详情，请查看
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) 文档。

    这个分词器继承自 [`PreTrainedTokenizerFast`]，其中包含大多数主要方法。用户应该参考这个超类获取更多关于这些方法的信息。
    """
    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<bos>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<eos>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The padding token
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add a `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
    """
    # 定义用于加载和保存的文件名列表
    vocab_files_names = VOCAB_FILES_NAMES
    # 使用 GemmaTokenizer 作为慢速分词器的类
    slow_tokenizer_class = GemmaTokenizer
    # 填充位置设为左侧
    padding_side = "left"
    # 模型输入名称列表
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        add_bos_token=True,
        add_eos_token=False,
        **kwargs,
    ):
        # 调用父类的初始化方法，传递参数以配置分词器
        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )
        # 设置是否添加 bos_token 到序列开始的标志
        self._add_bos_token = add_bos_token
        # 设置是否添加 eos_token 到序列结尾的标志
        self._add_eos_token = add_eos_token
        # 更新分词器的后处理器
        self.update_post_processor()
        # 保存词汇文件的路径
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        # 检查是否可以保存慢速分词器，需要有有效的词汇文件路径
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    # 从 transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.update_post_processor 复制
    # 更新后处理器，使用当前的 `bos_token` 和 `eos_token` 更新底层后处理器
    def update_post_processor(self):
        bos = self.bos_token  # 获取开始标记符
        bos_token_id = self.bos_token_id  # 获取开始标记符的 ID
        # 如果 `add_bos_token` 为 True 但 `bos_token` 为 None，抛出数值错误
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token  # 获取结束标记符
        eos_token_id = self.eos_token_id  # 获取结束标记符的 ID
        # 如果 `add_eos_token` 为 True 但 `eos_token` 为 None，抛出数值错误
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        # 构建单句模板
        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        # 构建双句模板
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        special_tokens = []
        # 如果需要添加开始标记符，则将其添加到特殊标记列表中
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        # 如果需要添加结束标记符，则将其添加到特殊标记列表中
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        
        # 将后处理器设为模板处理器，使用构建好的模板和特殊标记列表
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        return self._add_eos_token  # 返回是否添加结束标记符的属性值

    @property
    def add_bos_token(self):
        return self._add_bos_token  # 返回是否添加开始标记符的属性值

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value  # 设置是否添加结束标记符的属性值
        self.update_post_processor()  # 更新后处理器

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value  # 设置是否添加开始标记符的属性值
        self.update_post_processor()  # 更新后处理器

    # 从 transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast.save_vocabulary 复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 如果无法保存慢速分词器的词汇表，则抛出数值错误
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        # 如果保存路径不是目录，则记录错误信息并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        # 设置输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径与目标路径不同，则复制词汇表文件到目标路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
    # 构建带有特殊令牌的输入序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 如果需要添加起始令牌，将起始令牌 ID 添加到列表中
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        # 如果需要添加结束令牌，将结束令牌 ID 添加到列表中
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        # 构建输出序列，连接起始令牌、token_ids_0、结束令牌
        output = bos_token_id + token_ids_0 + eos_token_id

        # 如果存在第二个输入序列 token_ids_1，进行相同的处理
        if token_ids_1 is not None:
            # 连接起始令牌、token_ids_1、结束令牌
            output = output + bos_token_id + token_ids_1 + eos_token_id

        # 返回构建好的输出序列
        return output
```