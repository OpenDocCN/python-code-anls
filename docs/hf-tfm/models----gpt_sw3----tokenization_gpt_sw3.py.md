# `.\models\gpt_sw3\tokenization_gpt_sw3.py`

```
"""The tokenizer used by the GPT-SW3 models."""

# 引入所需的模块和库
import os  # 提供了与操作系统交互的功能
import re  # 提供了正则表达式操作支持
import unicodedata  # 提供了对 Unicode 数据库的访问
from shutil import copyfile  # 提供了文件和目录的操作函数
from typing import Any, Dict, List, Optional, Tuple, Union  # 提供了类型提示支持

import sentencepiece as spm  # 引入 SentencePiece 库用于分词

# 引入所需的自定义模块和函数
from ...tokenization_utils import PreTrainedTokenizer  # 引入预训练的 tokenizer 类
from ...utils import is_torch_available, logging  # 引入用于检查 Torch 是否可用和日志记录的工具函数


if is_torch_available():  # 检查是否安装了 Torch，如果是则引入 Torch
    import torch  # 引入 Torch 库

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}  # 定义词汇文件名字典

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
    ```
    """
    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (`bool`, *optional*, defaults to `False`):
            Whether or not to lowercase the input when tokenizing.
        remove_space (`bool`, *optional*, defaults to `False`):
            Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (`bool`, *optional*, defaults to `False`):
            Whether or not to keep accents when tokenizing.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths. If not provided, will
            default to '<pad>' or '<unk>' depending on model size.
        unk_token (`str`, *optional*):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead. If not provided, will default to '<unk>'.
        eos_token (`str`, *optional*):
            The end of sequence token seen during pretraining. If not provided, will default to '<|endoftext|>'
        bos_token (`str`, *optional*):
            The beginning of sequence token that can be used for downstream task, was not seen during pretraining. If
            not provided, will default to '<s>' or '<|endoftext|>', depending on model size.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor that is used for every conversion (string, tokens and IDs).
        whitespaces (`set`):
            The whitespaces that are replaced in the whitespace normalization in preprocessing.
        non_printing_characters_re (`Pattern`):
            The compiled regular expression to remove non-printing characters in preprocessing.
    """

    # List of vocabulary files' names
    vocab_files_names = VOCAB_FILES_NAMES
    # Map of pretrained vocabulary files
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 将预训练模型的位置编码嵌入大小赋值给变量
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]

    # 初始化函数，接收多个参数来配置分词器
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
        # 如果 sp_model_kwargs 是 None，则设置为空字典
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # 获取 kwargs 中的 name_or_path 参数，如果不存在则警告并设置为 "None"
        name_or_path = kwargs.get("name_or_path")
        if name_or_path is None:
            logger.warning(
                "name_or_path not provided, will work for all GPTSw3 models except gpt-sw3-7b,"
                " you are testing the model, this can safely be ignored"
            )
            name_or_path = "None"

        # 根据情况设置 eos_token 和 unk_token 的默认值
        eos_token = "<|endoftext|>" if eos_token is None else eos_token
        unk_token = "<unk>" if unk_token is None else unk_token
        
        # 如果 name_or_path 包含 "gpt-sw3-7b"，则设置 pad_token 和 bos_token 的值
        if "gpt-sw3-7b" in name_or_path:
            pad_token = unk_token if pad_token is None else pad_token
            bos_token = eos_token if bos_token is None else bos_token
        else:
            pad_token = "<pad>" if pad_token is None else pad_token
            bos_token = "<s>" if bos_token is None else bos_token

        # 设置对象的属性
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file

        # 使用 SentencePieceProcessor 初始化 self.sp_model
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        # 用于输入文本中空白字符的规范化
        # fmt : off
        self.whitespaces = {" ", " ", " ", " ", " ", "　", " ", " ", " ", " ", "￼", ""}
        # fmt : on

        # 正则表达式，用于在预处理中移除非打印字符（例如某些 Unicode 控制字符）
        self.non_printing_characters_re = re.compile(
            f"[{''.join(map(chr, list(range(0, 9)) + list(range(11, 32)) + list(range(127, 160)) + [160, 173, 8203]))}]"
        )

        # 调用父类的初始化方法
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

    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.__getstate__ 复制的方法
    # 返回对象的状态字典，将 sp_model 设为 None
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    # Copied from transformers.models.albert.tokenization_albert.AlbertTokenizer.__setstate__
    # 用于反序列化对象状态，将给定的字典 d 直接赋给对象的 __dict__ 属性
    def __setstate__(self, d):
        self.__dict__ = d

        # 用于向后兼容性检查，如果对象没有属性 "sp_model_kwargs"，则设置为空字典
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        # 使用 SentencePieceProcessor 初始化 self.sp_model 对象，传入 self.sp_model_kwargs 的参数
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        # 加载词汇文件到 self.sp_model
        self.sp_model.Load(self.vocab_file)

    @property
    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.vocab_size 处复制的属性
    # 返回当前对象 self.sp_model 的词汇大小，即词汇表中的条目数
    def vocab_size(self) -> int:
        return len(self.sp_model)

    # 对给定文本进行预处理，返回处理后的文本
    def preprocess_text(self, text: str) -> str:
        """
        返回预处理后的文本。该过程与训练标记器时使用的过程相同。
        """

        # 移除非打印字符
        text = self.non_printing_characters_re.sub("", text)

        # 规范化空白字符
        text = "".join([char if char not in self.whitespaces else " " for char in text])

        # NFC Unicode 规范化
        text = unicodedata.normalize("NFC", text)
        return text

    # 将给定文本进行标记化处理，返回标记化后的列表
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        text = self.preprocess_text(text)
        return self.sp_model.encode(text, out_type=str)

    # 将给定的 token (str) 转换为其对应的 id (int)，使用当前对象的词汇表进行转换
    def _convert_token_to_id(self, token: str) -> int:
        """将 token (str) 转换为 id (int)，使用词汇表进行转换。"""
        return self.sp_model.PieceToId(token)

    # 将给定的 id (int) 转换为其对应的 token (str)，使用当前对象的词汇表进行转换
    def _convert_id_to_token(self, index: int) -> str:
        """将 id (int) 转换为 token (str)，使用词汇表进行转换。"""
        return self.sp_model.IdToPiece(index)

    @staticmethod
    # 返回输入字符串本身，用于覆盖默认的清理函数
    def clean_up_tokenization(out_string: str) -> str:
        """返回输入的字符串，此函数用于移除默认的清理行为。"""
        return out_string

    # 将一系列 token (字符串) 转换为单个字符串，特殊 token 保持不变
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """将一系列 token (字符串) 转换为单个字符串。特殊 token 保持不变。"""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # 确保特殊 token 不使用 sentencepiece 模型解码
            if token in self.all_special_tokens:
                # TODO: 检查是否需要这一步骤，它确保 decode(encode(doc)) != doc，通过在解码文档中添加额外的空格来实现
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

    # 从 transformers.models.albert.tokenization_albert.AlbertTokenizer.get_vocab 处复制的方法
    # 返回当前对象的词汇表，包含 token 到 id 的映射
    def get_vocab(self) -> Dict[str, int]:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab
    # 从transformers.models.albert.tokenization_albert.AlbertTokenizer.save_vocabulary方法复制而来
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 检查保存目录是否存在，如果不存在则记录错误并返回
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        # 拼接输出的词汇表文件路径
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        # 如果当前词汇表文件路径不等于输出路径，并且当前词汇表文件存在，则复制当前词汇表文件到输出路径
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        # 如果当前词汇表文件不存在，则将序列化的 sp_model 内容写入到输出文件
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        # 返回输出文件路径的元组
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

        # 如果输入是单个字符串，则预处理文本并使用 sp_model 进行编码
        if isinstance(text, str):
            text = self.preprocess_text(text)
            token_ids = self.sp_model.encode(text)
        # 如果输入是字符串列表，则分别预处理每个文本并使用 sp_model 进行编码
        else:
            text = [self.preprocess_text(t) for t in text]
            token_ids = self.sp_model.encode(text)

        # 如果需要返回 PyTorch 张量，则转换编码后的结果为张量
        if return_tensors is True or return_tensors == "pt":
            token_ids = torch.tensor(token_ids)

        # 返回编码后的 token_ids
        return token_ids

    def decode_fast(self, token_ids: Union[int, List[int]]) -> str:
        """
        Decodes a text or batch of texts from token ids using preprocessing and the raw SP tokenizer. This has reduced
        functionality but is often much faster.

        Args:
            token_ids (`int` or `List[int]`): Encoded token or text as token id(s).

        Returns:
            `str`: Decoded text
        """

        # 使用 sp_model 对 token_ids 进行解码，返回解码后的文本
        return self.sp_model.decode(token_ids)

    @property
    # 定义一个默认的聊天模板函数，用于格式化消息，类似即时通讯的聊天记录，消息前面有 "User:" 和 "Bot:" 字符串。消息之间使用 BOS 标记分隔。
    def default_chat_template(self):
        """
        This chat template formats messages like an instant messenger chat log, with "User:" and "Bot:" strings
        preceding messages. BOS tokens are added between all messages.
        """
        # 记录一次警告，提示没有为该分词器定义聊天模板，将使用该类的默认模板。如果默认模板不适合您的模型，请设置 `tokenizer.chat_template` 为合适的模板。
        # 查看 https://huggingface.co/docs/transformers/main/chat_templating 获取更多信息。
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # 返回格式化后的聊天模板字符串，包括 EOS 标记和 BOS 标记在每条消息之间，处理 messages 列表中的每一条消息并添加对应的角色前缀。
        return (
            "{{ eos_token }}{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ 'User: ' + message['content']}}"
            "{% else %}{{ 'Bot: ' + message['content']}}{% endif %}"
            "{{ message['text'] }}{{ bos_token }}"
            "{% endfor %}"
            "Bot:"  # 最后追加一个固定的 "Bot:" 字符串，表示消息结束
        )
```