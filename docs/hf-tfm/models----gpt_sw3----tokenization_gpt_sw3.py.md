# `.\models\gpt_sw3\tokenization_gpt_sw3.py`

```py
"""The tokenizer used by the GPT-SW3 models."""

# 导入所需的模块和库
import os
import re
import unicodedata
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple, Union

import sentencepiece as spm

from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_torch_available, logging

# 检查是否有 torch 可用
if is_torch_available():
    import torch

# 获取 logger 实例
logger = logging.get_logger(__name__)
# 词汇文件的名称
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}

# 预训练模型词汇文件的映射
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "AI-Sweden-Models/gpt-sw3-126m": "https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m/resolve/main/spiece.model",
        "AI-Sweden-Models/gpt-sw3-356m": "https://huggingface.co/AI-Sweden-Models/gpt-sw3-356m/resolve/main/spiece.model",
        "AI-Sweden-Models/gpt-sw3-1.3b": "https://huggingface.co/AI-Sweden-Models/gpt-sw3-1.3b/resolve/main/spiece.model",
        "AI-Sweden-Models/gpt-sw3-6.7b": "https://huggingface.co/AI-Sweden-Models/gpt-sw3-6.7b/resolve/main/spiece.model",
        "AI-Sweden-Models/gpt-sw3-6.7b-v2": "https://huggingface.co/AI-Sweden-Models/gpt-sw3-6.7b-v2/resolve/main/spiece.model",
        "AI-Sweden-Models/gpt-sw3-20b": "https://huggingface.co/AI-Sweden-Models/gpt-sw3-20b/resolve/main/spiece.model",
        "AI-Sweden-Models/gpt-sw3-40b": "https://huggingface.co/AI-Sweden-Models/gpt-sw3-20b/resolve/main/spiece.model",
    }
}

# 预训练位置嵌入大小的映射
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "AI-Sweden-Models/gpt-sw3-126m": 2048,
    "AI-Sweden-Models/gpt-sw3-356m": 2048,
    "AI-Sweden-Models/gpt-sw3-1.3b": 2048,
    "AI-Sweden-Models/gpt-sw3-6.7b": 2048,
    "AI-Sweden-Models/gpt-sw3-6.7b-v2": 2048,
    "AI-Sweden-Models/gpt-sw3-20b": 2048,
    "AI-Sweden-Models/gpt-sw3-40b": 2048,
}


class GPTSw3Tokenizer(PreTrainedTokenizer):
    """
    Construct an GPTSw3 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Example usage:
    ```python
    >>> from transformers import GPTSw3Tokenizer

    >>> tokenizer = GPTSw3Tokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-126m")
    >>> tokenizer("Svenska är kul!")["input_ids"]
    [1814, 377, 3617, 63504]
    ```py
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
            实例化分词器所需的词汇文件路径，通常具有 *.spm* 扩展名。
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
            在分词时是否将输入转换为小写，默认为 `False`。
        remove_space (`bool`, *optional*, defaults to `False`):
            Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
            在分词时是否删除文本中的空格（删除字符串前后的多余空格），默认为 `False`。
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether or not to keep accents when tokenizing.
            在分词时是否保留重音符号，默认为 `False`。
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths. If not provided, will
            default to '<pad>' or '<unk>' depending on model size.
            用于填充的标记，例如当批处理不同长度的序列时。如果未提供，则默认为 '<pad>' 或 '<unk>'，取决于模型大小。
        unk_token (`str`, *optional*):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. If not provided, will default to '<unk>'.
            未知标记。不在词汇表中的标记无法转换为 ID，并将其设置为此标记。如果未提供，则默认为 '<unk>'。
        eos_token (`str`, *optional*):
            The end of sequence token seen during pretraining. If not provided, will default to '<|endoftext|>'
            预训练期间遇到的序列结束标记。如果未提供，则默认为 '<|endoftext|>'。
        bos_token (`str`, *optional*):
            The beginning of sequence token that can be used for downstream task, was not seen during pretraining. If
            not provided, will default to '<s>' or '<|endoftext|>', depending on model size.
            用于下游任务的序列开始标记，在预训练期间未见过。如果未提供，则默认为 '<s>' 或 '<|endoftext|>'，取决于模型大小。
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
              启用子词正则化。
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
                不执行采样。
              - `nbest_size > 1`: samples from the nbest_size results.
                从 nbest_size 个结果中采样。
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.
                假设 nbest_size 为无穷大，并使用前向过滤和后向采样算法从所有假设（格）中进行采样。

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
              一元采样的平滑参数，以及 BPE-dropout 的合并操作的丢弃概率。

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
            用于每次转换（字符串、标记和 ID）的 *SentencePiece* 处理器。
        whitespaces (`set`):
            The whitespaces that are replaced in the whitespace normalization in preprocessing.
            在预处理中用于空格归一化的空格。
        non_printing_characters_re (`Pattern`):
            The compiled regular expression to remove non-printing characters in preprocessing.
            用于在预处理中删除非打印字符的编译正则表达式。
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义最大模型输入尺寸为预训练位置嵌入大小
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化函数，接受多个参数
    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=False,
        keep_accents=False,
        pad_token=None,
        unk_token=None,
        eos_token=None,
        bos_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # 如果 sp_model_kwargs 为 None，则设置为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 获取参数中的 name_or_path，如果为 None，则设置为 "None"，并发出警告
        name_or_path = kwargs.get("name_or_path")
        if name_or_path is None:
            logger.warning(
                "name_or_path not provided, will work for all GPTSw3 models except gpt-sw3-7b,"
                " you are testing the model, this can safely be ignored"
            )
            name_or_path = "None"

        # 根据条件设置 eos_token 和 unk_token 的默认值
        eos_token = "<|endoftext|>" if eos_token is None else eos_token
        unk_token = "<unk>" if unk_token is None else unk_token
        if "gpt-sw3-7b" in name_or_path:
            pad_token = unk_token if pad_token is None else pad_token
            bos_token = eos_token if bos_token is None else bos_token
        else:
            pad_token = "<pad>" if pad_token is None else pad_token
            bos_token = "<s>" if bos_token is None else bos_token

        # 设置对象的属性值
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        # 使用 spm.SentencePieceProcessor 初始化 self.sp_model
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 用于输入文本中空格的规范化
        # fmt : off
        self.whitespaces = {" ", " ", " ", " ", " ", "　", " ", " ", " ", " ", "￼", ""}
        # fmt : on

        # 用于在预处理中移除非打印字符（例如一些 Unicode 控制字符）的正则表达式
        self.non_printing_characters_re = re.compile(
            f"[{''.join(map(chr, list(range(0, 9)) + list(range(11, 32)) + list(range(127, 160)) + [160, 173, 8203]))}]"
        )

        # 调用父类的初始化函数
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.__getstate__ 复制的函数
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.__setstate__ 复制的函数
    # 重写 __setstate__ 方法，将传入的字典赋值给对象的 __dict__ 属性
    def __setstate__(self, d):
        self.__dict__ = d

        # 为了向后兼容性，如果对象没有 sp_model_kwargs 属性，则设置为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用 sp_model_kwargs 创建 SentencePieceProcessor 对象，并加载词汇文件
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    @property
    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.vocab_size 复制属性
    def vocab_size(self) -> int:
        # 返回 SentencePieceProcessor 对象的长度作为词汇量大小
        return len(self.sp_model)

    # 预处理文本，去除非打印字符，规范化空白字符，进行 NFC Unicode 规范化
    def preprocess_text(self, text: str) -> str:
        """
        Returns the preprocessed text. This procedure is identical to what was used when training the tokenizer.
        """

        # 去除非打印字符
        text = self.non_printing_characters_re.sub("", text)

        # 规范化空白字符
        text = "".join([char if char not in self.whitespaces else " " for char in text])

        # NFC Unicode 规范化
        text = unicodedata.normalize("NFC", text)
        return text

    # 对文本进行分词处理
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        text = self.preprocess_text(text)
        return self.sp_model.encode(text, out_type=str)

    # 将 token 转换为 id
    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id (int) using the vocab."""
        return self.sp_model.PieceToId(token)

    # 将 id 转换为 token
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (int) to a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    @staticmethod
    # 清理分词结果的方法，覆盖默认的清理方法
    def clean_up_tokenization(out_string: str) -> str:
        """Returns the input string, this function is overridden to remove the default clean up."""
        return out_string

    # 将一系列 token 转换为单个字符串
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings) to a single string. Special tokens remain intact."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊 token 不会被 SentencePiece 模型解码
            if token in self.all_special_tokens:
                # TODO: 检查是否需要，因为它确保了 decode(encode(doc)) != doc，通过在解码文档中添加额外的空格
                if not prev_is_special:
                    out_string += " "

                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)

        return out_string

    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.get_vocab 复制方法
    def get_vocab(self) -> Dict[str, int]:
        # 创建词汇表字典，将 token 转换为 id
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    # 从transformers.models.albert.tokenization_albert.AlbertTokenizer.save_vocabulary中复制的方法
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 设置输出词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果词汇表文件路径与当前词汇表文件路径不同且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化的模型内容写入输出路径
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def encode_fast(
        self, text: Union[str, List[str]], return_tensors: Union[str, bool] = False
    ) -> Union[List[int], List[List[int]], "torch.Tensor"]:
        """
        Encodes a text or batch of texts to token ids using preprocessing and the raw SP tokenizer. This has reduced
        functionality but is often much faster.

        Does NOT handle special tokens correctly, these can manually be added as ids afterwards.

        Does NOT support padding, these can manually be added as ids afterwards.

        Use default HuggingFace tokenization methods for full functionality.

        Args:
            text (`str` or `List[str]`): One or several text(s) to convert to token ids.
            return_tensors (`str` or `bool`): Returns PyTorch tensors if set to True or "pt"

        Returns:
            `List[int]`, `List[List[int]]`, or `torch.Tensor`: The encoded text(s) as token ids.
        """

        # 如果输入为单个字符串，则预处理文本并使用原始SP tokenizer对其进行编码
        if isinstance(text, str):
            text = self.preprocess_text(text)
            token_ids = self.sp_model.encode(text)
        # 如果输入为字符串列表，则对每个字符串进行预处理并使用原始SP tokenizer对其进行编码
        else:
            text = [self.preprocess_text(t) for t in text]
            token_ids = self.sp_model.encode(text)

        # 如果return_tensors为True或"pt"，则将token_ids转换为PyTorch张量
        if return_tensors is True or return_tensors == "pt":
            token_ids = torch.tensor(token_ids)

        return token_ids

    def decode_fast(self, token_ids: Union[int, List[int]]) -> str:
        """
        Encodes a text or batch of texts to token ids using preprocessing and the raw SP tokenizer. This has reduced
        functionality but is often much faster.

        Args:
            token_ids (`int` or `List[int]`): Encoded token or text as token id(s).

        Returns:
            `str`: Decoded text
        """

        # 使用原始SP tokenizer对token_ids进行解码
        return self.sp_model.decode(token_ids)

    @property
    # 默认的聊天模板，格式化消息类似即时通讯聊天记录，使用"User:"和"Bot:"字符串标记消息前缀，消息之间添加BOS标记
    def default_chat_template(self):
        # 如果未定义聊天模板，则使用默认模板，并给出警告信息
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回默认的聊天模板
        return (
            "{{ eos_token }}{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ 'User: ' + message['content']}}"
            "{% else %}{{ 'Bot: ' + message['content']}}{% endif %}"
            "{{ message['text'] }}{{ bos_token }}"
            "{% endfor %}"
            "Bot:"
        )
```