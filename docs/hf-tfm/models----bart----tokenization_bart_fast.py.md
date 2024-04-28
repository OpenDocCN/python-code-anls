# `.\transformers\models\bart\tokenization_bart_fast.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Facebook AI Research Team 作者和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入必要的库
import json
from typing import List, Optional, Tuple
# 从 tokenizers 库中导入 pre_tokenizers 和 processors
from tokenizers import pre_tokenizers, processors

# 从 tokenization_utils_base 中导入 AddedToken 和 BatchEncoding
from ...tokenization_utils_base import AddedToken, BatchEncoding
# 从 tokenization_utils_fast 中导入 PreTrainedTokenizerFast
from ...tokenization_utils_fast import PreTrainedTokenizerFast
# 从 utils 中导入 logging
from ...utils import logging
# 从当前目录中的 tokenization_bart 文件中导入 BartTokenizer
from .tokenization_bart import BartTokenizer

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

# 预训练模型的词汇文件映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/vocab.json",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/vocab.json",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/vocab.json",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/vocab.json",
    },
    "merges_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/merges.txt",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/merges.txt",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/merges.txt",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/merges.txt",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/merges.txt",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/merges.txt",
    },
    # 定义一个字典，用于存储各个 BART 模型的 tokenizer 文件的 URL
    "tokenizer_file": {
        "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/tokenizer.json",
        "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/tokenizer.json",
        "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/tokenizer.json",
        "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/tokenizer.json",
        "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/tokenizer.json",
        "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/tokenizer.json",
    },
# 预训练位置嵌入的大小字典，键为模型名称，值为位置嵌入的大小
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "facebook/bart-base": 1024,
    "facebook/bart-large": 1024,
    "facebook/bart-large-mnli": 1024,
    "facebook/bart-large-cnn": 1024,
    "facebook/bart-large-xsum": 1024,
    "yjernite/bart_eli5": 1024,
}

# 定义 BartTokenizerFast 类，继承自 PreTrainedTokenizerFast 类
class BartTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" BART tokenizer (backed by HuggingFace's *tokenizers* library), derived from the GPT-2 tokenizer,
    using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import BartTokenizerFast

    >>> tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```py

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    """
    Args:
        vocab_file (`str`):
            Path to the vocabulary file. 词汇表文件的路径。
        merges_file (`str`):
            Path to the merges file. 合并文件的路径。
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
            解码字节到 UTF-8 时要遵循的范例。有关更多信息，请参阅 [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode)。
        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            
            <Tip>
            
            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.
            
            </Tip>
            
            用于预训练期间使用的序列起始标记。可以用作序列分类器标记。
            
            <Tip>
            
            使用特殊标记构建序列时，这不是用于序列开头的标记。使用的标记是 `cls_token`。
            
            </Tip>

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
            
            <Tip>
            
            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.
            
            </Tip>
            
            序列结束标记。
            
            <Tip>
            
            使用特殊标记构建序列时，这不是用于序列结尾的标记。使用的标记是 `sep_token`。
            
            </Tip>

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            
            分隔符标记，用于从多个序列构建序列，例如，用于序列分类的两个序列或用于问答的文本和问题。它也用作使用特殊标记构建的序列的最后一个标记。

        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            
            序列分类时使用的分类器标记（对整个序列进行分类而不是对每个标记进行分类）。在使用特殊标记构建时，它是序列的第一个标记。

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            
            未知标记。不在词汇表中的标记无法转换为 ID，并被设置为此标记。

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
            
            用于填充的标记，例如当批处理不同长度的序列时。

        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
            
            用于掩码值的标记。这是在使用掩码语言建模对该模型进行训练时使用的标记。这是模型将尝试预测的标记。

        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (BART tokenizer detect beginning of words by the preceding space).
            
            是否添加初始空格到输入。这样可以将首单词视为任何其他单词。（BART 分词器通过前导空格检测单词的开头）。

        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
            
            后处理步骤是否应修剪偏移量以避免包含空格。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表，包括 input_ids 和 attention_mask
    model_input_names = ["input_ids", "attention_mask"]
    # 指定慢速分词器的类为 BartTokenizer
    slow_tokenizer_class = BartTokenizer

    # 初始化方法
    def __init__(
        self,
        # 词汇表文件路径，默认为 None
        vocab_file=None,
        # 合并文件路径，默认为 None
        merges_file=None,
        # 分词器文件路径，默认为 None
        tokenizer_file=None,
        # 错误处理方式，默认为 "replace"
        errors="replace",
        # 开始词标记，默认为 "<s>"
        bos_token="<s>",
        # 结束词标记，默认为 "</s>"
        eos_token="</s>",
        # 分隔词标记，默认为 "</s>"
        sep_token="</s>",
        # 类别标记，默认为 "<s>"
        cls_token="<s>",
        # 未知词标记，默认为 "<unk>"
        unk_token="<unk>",
        # 填充词标记，默认为 "<pad>"
        pad_token="<pad>",
        # 掩码词标记，默认为 "<mask>"
        mask_token="<mask>",
        # 是否添加前缀空格，默认为 False
        add_prefix_space=False,
        # 是否修剪偏移量，默认为 True
        trim_offsets=True,
        # 其它参数，使用 **kwargs 接收
        **kwargs,
        # 如果 mask_token 是字符串类型，则创建一个特殊的 AddedToken 对象，否则直接使用给定的 mask_token
        mask_token = (
            AddedToken(mask_token, lstrip=True, normalized=True, special=True)
            if isinstance(mask_token, str)
            else mask_token
        )
        # 调用父类的初始化方法，传入各种参数以初始化 Tokenizer
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )

        # 将前置分词器的状态序列化为 JSON 对象
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果给定的 add_prefix_space 与前置分词器中的状态不一致，则更新前置分词器的状态
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置 add_prefix_space 属性
        self.add_prefix_space = add_prefix_space

        # 获取后置处理器组件的状态
        tokenizer_component = "post_processor"
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # 'sep' 和 'cls' 列表必须用元组包装，以适应 post_processor_class 对象的需求
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            # 检查是否有要应用的更改
            changes_to_apply = False

            # 如果给定的 add_prefix_space 与状态中的不一致，则更新状态
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            # 如果给定的 trim_offsets 与状态中的不一致，则更新状态
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            # 如果有更改要应用，则创建新的后置处理器对象并应用更改
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

    @property
    def mask_token(self) -> str:
        """
        `str`: 返回掩码标记，用于训练具有掩码语言建模的模型。如果在未设置的情况下使用，则记录错误。
        
        BART 分词器有一个特殊的掩码标记，可用于填充掩码管道。掩码标记将贪婪地包括*<mask>*之前的空格。
        """
        # 如果掩码标记尚未设置，则记录错误并返回None
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        # 返回掩码标记的字符串表示形式
        return str(self._mask_token)

    @mask_token.setter
    def mask_token(self, value):
        """
        重写掩码标记的默认行为，使其包含前面的空格。
        
        这是为了与基于Bart的所有先前使用的模型保持向后兼容性。
        """
        # 如果值是字符串，则将其封装为 AddedToken 对象，保留左侧空格，右侧不保留
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        # 设置掩码标记
        self._mask_token = value

    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 检查是否已将输入分割为单词
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 如果已经将输入分割为单词，但未设置前缀空格，则引发错误
        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )

        # 调用父类的_batch_encode_plus方法
        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 检查是否已将输入分割为单词
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 如果已经将输入分割为单词，但未设置前缀空格，则引发错误
        if is_split_into_words and not self.add_prefix_space:
            raise ValueError(
                f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
                "to use it with pretokenized inputs."
            )

        # 调用父类的_encode_plus方法
        return super()._encode_plus(*args, **kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 保存词汇表文件
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 构建带有特殊标记的输入序列
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        # 如果存在第二个输入序列，则将其添加到输出中
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
```  
    # 定义一个函数，用于在序列对分类任务中创建一个掩码。BART模型不使用token类型id，因此返回一个全零列表。
    def create_mask(self, token_ids_0: List[int], token_ids_1: List[int] = None) -> List[int]:
        # 定义分隔符token的id列表
        sep = [self.sep_token_id]
        # 定义类别token的id列表
        cls = [self.cls_token_id]

        # 如果没有第二个序列的token id列表，则返回第一个序列的token id列表长度加上类别token、分隔符token的长度的全零列表
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        # 否则返回两个序列的token id列表长度加上类别token、分隔符token的长度的全零列表
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```