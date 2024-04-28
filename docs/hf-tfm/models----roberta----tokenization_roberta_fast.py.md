# `.\transformers\models\roberta\tokenization_roberta_fast.py`

```
# 设置文件编码为 UTF-8
# 版权声明，引用了 Apache License 2.0
# 导入必要的模块
"""Fast Tokenization classes for RoBERTa."""
import json  # 导入 json 模块
from typing import List, Optional, Tuple  # 导入类型提示模块

from tokenizers import pre_tokenizers, processors  # 导入 tokenizers 模块

from ...tokenization_utils_base import AddedToken, BatchEncoding  # 导入基础 Tokenization 模块
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入 Fast Tokenization 模块
from ...utils import logging  # 导入日志模块
from .tokenization_roberta import RobertaTokenizer  # 导入 RoBERTa Tokenizer 模块

logger = logging.get_logger(__name__)  # 获取日志记录器

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}  # 定义词汇表文件名字典

PRETRAINED_VOCAB_FILES_MAP = {  # 预训练词汇表文件映射
    "vocab_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/vocab.json",  # RoBERTa Base 模型的词汇表文件
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/vocab.json",  # RoBERTa Large 模型的词汇表文件
        "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/vocab.json",  # RoBERTa Large MNLI 模型的词汇表文件
        "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/vocab.json",  # DistilRoBERTa Base 模型的词汇表文件
        "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/vocab.json",  # RoBERTa Base OpenAI Detector 模型的词汇表文件
        "roberta-large-openai-detector": (  # RoBERTa Large OpenAI Detector 模型的词汇表文件
            "https://huggingface.co/roberta-large-openai-detector/resolve/main/vocab.json"
        ),
    },
    "merges_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/merges.txt",  # RoBERTa Base 模型的合并文件
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/merges.txt",  # RoBERTa Large 模型的合并文件
        "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/merges.txt",  # RoBERTa Large MNLI 模型的合并文件
        "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/merges.txt",  # DistilRoBERTa Base 模型的合并文件
        "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/merges.txt",  # RoBERTa Base OpenAI Detector 模型的合并文件
        "roberta-large-openai-detector": (  # RoBERTa Large OpenAI Detector 模型的合并文件
            "https://huggingface.co/roberta-large-openai-detector/resolve/main/merges.txt"
        ),
    },
    # 定义一个字典，键为模型名称，值为对应的 tokenizer 文件链接
    "tokenizer_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/tokenizer.json",
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/tokenizer.json",
        "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/tokenizer.json",
        "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/tokenizer.json",
        "roberta-base-openai-detector": (
            "https://huggingface.co/roberta-base-openai-detector/resolve/main/tokenizer.json"
        ),
        "roberta-large-openai-detector": (
            "https://huggingface.co/roberta-large-openai-detector/resolve/main/tokenizer.json"
        ),
    },
}

# 预训练位置嵌入的尺寸字典，用于不同 RoBERTa 模型的尺寸配置
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
    "roberta-large": 512,
    "roberta-large-mnli": 512,
    "distilroberta-base": 512,
    "roberta-base-openai-detector": 512,
    "roberta-large-openai-detector": 512,
}

# 定义一个 RoBERTa 快速分词器类，继承自 PreTrainedTokenizerFast
class RobertaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" RoBERTa tokenizer (backed by HuggingFace's *tokenizers* library), derived from the GPT-2
    tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import RobertaTokenizerFast

    >>> tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    >>> tokenizer("Hello world")["input_ids"]
    [0, 31414, 232, 2]

    >>> tokenizer(" Hello world")["input_ids"]
    [0, 20920, 232, 2]
    ```

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
            Path to the vocabulary file.       // 指向词汇文件的路径

        merges_file (`str`):
            Path to the merges file.           // 指向合并文件的路径

        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
            // 解码字节为 UTF-8 时遵循的范例

        bos_token (`str`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            // 在预训练期间使用的序列开始标记

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the beginning of
            sequence. The token used is the `cls_token`.

            </Tip>
            // 当使用特殊标记构建序列时，这不是使用的序列开始标记

        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
            // 序列结束标记

            <Tip>

            When building a sequence using special tokens, this is not the token that is used for the end of sequence.
            The token used is the `sep_token`.

            </Tip>
            // 当使用特殊标记构建序列时，这不是用于序列结束的标记

        sep_token (`str`, *optional*, defaults to `"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            // 用于构建多个序列时的分隔标记

        cls_token (`str`, *optional*, defaults to `"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            // 用于进行序列分类时使用的分类器标记

        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
            // 未知标记

        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
            // 用于填充的标记，例如在批处理不同长度的序列时

        mask_token (`str`, *optional*, defaults to `"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
            // 用于屏蔽值的标记，用于进行遮罩语言建模时

        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
            // 是否在输入前加一个初始空格

        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
            // 后处理步骤是否应该修剪偏移量以避免包含空格
    """

    // 保存词汇文件名的常量
    vocab_files_names = VOCAB_FILES_NAMES
    // 预先训练的词汇文件的映射
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    // 保存预先训练位置嵌入大小的常量
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称
    model_input_names = ["input_ids", "attention_mask"]

    # 指定慢速分词器的类
    slow_tokenizer_class = RobertaTokenizer

    def __init__(
        self,
        # 定义各种参数和令牌的默认值
        vocab_file=None,  # 词汇文件
        merges_file=None,  # 合并文件
        tokenizer_file=None,  # 分词器文件
        errors="replace",  # 错误处理策略
        bos_token="<s>",  # 开始标记
        eos_token="</s>",  # 结束标记
        sep_token="</s>",  # 分隔标记
        cls_token="<s>",  # 类标记
        unk_token="<unk>",  # 未知标记
        pad_token="<pad>",  # 填充标记
        mask_token="<mask>",  # 掩码标记
        add_prefix_space=False,  # 是否添加前缀空格
        trim_offsets=True,  # 是否修剪偏移量
        **kwargs,  # 其他额外参数
    ):
        # 如果掩码标记是字符串，则用 `AddedToken` 类来创建，并指定 lstrip 为 True，rstrip 为 False
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False, normalized=False)
            if isinstance(mask_token, str)
            else mask_token
        )
        # 调用父类的构造函数，传入各种参数和标记
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

        # 获取 `pre_tokenizer` 的状态，并将其转为 JSON 格式
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())

        # 如果 `add_prefix_space` 与预期的不符，则更改
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            # 获取预分词器的类，并调整其状态
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            # 设置新的 `pre_tokenizer`，并传入调整后的状态
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 设置 `add_prefix_space` 的实例变量
        self.add_prefix_space = add_prefix_space

        # 定义需要检查的分词器组件
        tokenizer_component = "post_processor"

        # 获取指定组件的实例
        tokenizer_component_instance = getattr(self.backend_tokenizer, tokenizer_component, None)

        # 如果组件实例存在，则获取其状态，并进行必要的调整
        if tokenizer_component_instance:
            state = json.loads(tokenizer_component_instance.__getstate__())

            # `sep` 和 `cls` 列表需要转换为元组，以满足组件类的要求
            if "sep" in state:
                state["sep"] = tuple(state["sep"])
            if "cls" in state:
                state["cls"] = tuple(state["cls"])

            # 标记是否需要应用更改
            changes_to_apply = False

            # 如果 `add_prefix_space` 不一致，则需要更改
            if state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
                state["add_prefix_space"] = add_prefix_space
                changes_to_apply = True

            # 如果 `trim_offsets` 不一致，则需要更改
            if state.get("trim_offsets", trim_offsets) != trim_offsets:
                state["trim_offsets"] = trim_offsets
                changes_to_apply = True

            # 如果需要更改，则创建新的组件类实例，并应用到分词器
            if changes_to_apply:
                component_class = getattr(processors, state.pop("type"))
                new_value = component_class(**state)
                setattr(self.backend_tokenizer, tokenizer_component, new_value)

    # 定义一个属性，后续可扩展使用
    @property
    # 该函数定义了 mask_token 属性的 getter 方法
    def mask_token(self) -> str:
        """
        `str`: Mask token, to use when training a model with masked-language modeling. Log an error if used while not
        having been set.
    
        Roberta tokenizer has a special mask token to be usable in the fill-mask pipeline. The mask token will greedily
        comprise the space before the *<mask>*.
        """
        # 如果 _mask_token 属性还未设置，输出警告日志并返回 None
        if self._mask_token is None:
            if self.verbose:
                logger.error("Using mask_token, but it is not set yet.")
            return None
        # 返回 _mask_token 属性的字符串表示
        return str(self._mask_token)
    
    # 该函数定义了 mask_token 属性的 setter 方法
    @mask_token.setter
    def mask_token(self, value):
        """
        Overriding the default behavior of the mask token to have it eat the space before it.
    
        This is needed to preserve backward compatibility with all the previously used models based on Roberta.
        """
        # 如果输入的 value 是字符串，将其转换为 AddedToken 对象，并设置 lstrip=True、rstrip=False
        value = AddedToken(value, lstrip=True, rstrip=False) if isinstance(value, str) else value
        # 将转换后的 value 赋值给 _mask_token 属性
        self._mask_token = value
    
    # 该函数覆盖了父类的 _batch_encode_plus 方法
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取 is_split_into_words 参数的值
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 如果 is_split_into_words 为 False 且 add_prefix_space 为 False，抛出异常
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
        # 调用父类的 _batch_encode_plus 方法并返回结果
        return super()._batch_encode_plus(*args, **kwargs)
    
    # 该函数覆盖了父类的 _encode_plus 方法
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取 is_split_into_words 参数的值
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 如果 is_split_into_words 为 False 且 add_prefix_space 为 False，抛出异常
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
        # 调用父类的 _encode_plus 方法并返回结果
        return super()._encode_plus(*args, **kwargs)
    
    # 该函数保存词汇表到指定路径
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用 _tokenizer.model.save 方法保存词汇表，并返回文件路径列表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
    
    # 该函数构建包含特殊令牌的输入序列
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # 构建包含开始和结束标记的输入序列
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        # 如果提供了第二个输入序列，在其后添加结束标记
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
    
    # 该函数根据输入序列创建对应的 token_type_ids
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
        ) -> List[int]:
        """
        从传递的两个序列中创建一个用于序列对分类任务的掩码。RoBERTa 不使用 token 类型 id，因此返回一个由零组成的列表。

        Args:
            token_ids_0 (`List[int]`):
                ID 列表。
            token_ids_1 (`List[int]`, *optional*):
                序列对的可选第二个 ID 列表。

        Returns:
            `List[int]`: 零值列表。
        """
        # 设置 [SEP] token 的 ID
        sep = [self.sep_token_id]
        # 设置 [CLS] token 的 ID
        cls = [self.cls_token_id]

        # 如果没有传入第二个序列的 ID 列表
        if token_ids_1 is None:
            # 返回一个长度为 [CLS] + 第一个序列 + [SEP] 的列表，并填充为 0
            return len(cls + token_ids_0 + sep) * [0]
        # 否则
        # 返回一个长度为 [CLS] + 第一个序列 + [SEP] + [SEP] + 第二个序列 + [SEP] 的列表，并填充为 0
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]
```