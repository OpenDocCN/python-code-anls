# `.\models\whisper\tokenization_whisper_fast.py`

```py
# coding=utf-8
# 设置文件编码为UTF-8，确保可以正确处理中文等特殊字符

# 版权声明，指明此代码的版权归The HuggingFace Inc.团队所有，保留所有权利

# 导入需要的模块和库
import json  # 导入用于处理JSON格式的模块
import os  # 导入操作系统功能的模块
import re  # 导入正则表达式模块，用于文本匹配和处理
import warnings  # 导入警告处理模块，用于处理警告信息
from functools import lru_cache  # 导入缓存函数装饰器，用于缓存函数的返回值
from typing import List, Optional, Tuple  # 导入类型提示相关的模块

import numpy as np  # 导入数值计算库Numpy
from tokenizers import AddedToken, pre_tokenizers, processors  # 从tokenizers库中导入特定的类和函数

from ...tokenization_utils_base import BatchEncoding  # 导入来自tokenization_utils_base模块的BatchEncoding类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入来自tokenization_utils_fast模块的PreTrainedTokenizerFast类
from ...utils import logging  # 从utils模块中导入logging函数
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer  # 导入本地english_normalizer模块中的文本规范化类
from .tokenization_whisper import LANGUAGES, TASK_IDS, TO_LANGUAGE_CODE, WhisperTokenizer, _decode_asr  # 导入本地tokenization_whisper模块中的特定内容

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

# 定义常量，指明文件名称和其对应的内容类型
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件的名称为vocab.json
    "tokenizer_file": "tokenizer.json",  # 分词器文件的名称为tokenizer.json
    "merges_file": "merges.txt",  # 合并文件的名称为merges.txt
    "normalizer_file": "normalizer.json",  # 规范化器文件的名称为normalizer.json
}

# 定义预训练模型的文件映射，包含不同模型及其对应的词汇表文件URL
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/vocab.json",
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/vocab.json",
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/vocab.json",
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/vocab.json",
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/vocab.json",
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/vocab.json",
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/vocab.json",
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/vocab.json",
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/vocab.json",
    },
}
    "merges_file": {
        # merges_file 是一个字典，包含了各种不同模型的名称和对应的 merges.txt 文件的 URL
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/merges.txt",
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/merges.txt",
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/merges.txt",
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/merges.txt",
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/merges.txt",
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/merges.txt",
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/merges.txt",
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/merges.txt",
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        # tokenizer_file 是一个字典，包含了各种不同模型的名称和对应的 tokenizer.json 文件的 URL
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json",
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/tokenizer.json",
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json",
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/tokenizer.json",
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/tokenizer.json",
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/tokenizer.json",
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/tokenizer.json",
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/tokenizer.json",
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    # 定义预训练位置嵌入的大小，对应不同模型的键值对
    "openai/whisper-tiny": 1500,
    "openai/whisper-base": 1500,
    "openai/whisper-small": 1500,
    "openai/whisper-medium": 1500,
    "openai/whisper-large": 1500,
    "openai/whisper-tiny.en": 1500,
    "openai/whisper-base.en": 1500,
    "openai/whisper-small.en": 1500,
    "openai/whisper-medium.en": 1500,
}

class WhisperTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Whisper tokenizer (backed by HuggingFace's *tokenizers* library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        normalizer_file (`str`, *optional*):
            Path to the normalizer_file file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
            `"<|startoftranscript|>"` when generating.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Whisper tokenizer detect beginning of words by the preceding space).
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 使用预先定义的全局变量来初始化最大模型输入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 设置慢速标记器的类为WhisperTokenizer
    slow_tokenizer_class = WhisperTokenizer

    # 初始化方法，接受多个可选参数和关键字参数
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        normalizer_file=None,
        tokenizer_file=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        add_prefix_space=False,
        language=None,
        task=None,
        predict_timestamps=False,
        **kwargs,
    ):
        # 根据传入的参数初始化起始标记（bos_token）、结束标记（eos_token）和未知标记（unk_token）
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(unk_token, str)
            else unk_token
        )

        # 调用父类的初始化方法，传入相关参数和关键字参数
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        # 从关键字参数中移除并设置add_bos_token属性，默认为False
        self.add_bos_token = kwargs.pop("add_bos_token", False)

        # 获取前处理器的状态，确保前缀空格的一致性，并更新到后端标记器的前处理器中
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            pre_tok_state["add_prefix_space"] = add_prefix_space
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)

        # 如果提供了正常化文件，则加载英语拼写规范化器
        if normalizer_file is not None:
            with open(normalizer_file, encoding="utf-8") as vocab_handle:
                self.english_spelling_normalizer = json.load(vocab_handle)
        else:
            self.english_spelling_normalizer = None

        # 设置是否添加前缀空格的属性
        self.add_prefix_space = add_prefix_space
        # 编译用于匹配时间戳的正则表达式模式
        self.timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")

        # 设置语言、任务和是否预测时间戳的属性
        self.language = language
        self.task = task
        self.predict_timestamps = predict_timestamps

    # 从transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast._batch_encode_plus复制而来
    # 批量编码方法，返回BatchEncoding对象
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 检查是否已分割为单词，确保如果使用预分词的输入则需要设置add_prefix_space=True
        is_split_into_words = kwargs.get("is_split_into_words", False)
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        # 调用父类的_batch_encode_plus方法进行编码处理
        return super()._batch_encode_plus(*args, **kwargs)

    # 从transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast._encode_plus复制而来
    # 检查是否传入了参数 is_split_into_words，并获取其值，默认为 False
    is_split_into_words = kwargs.get("is_split_into_words", False)

    # 断言条件：如果 add_prefix_space 为 True 或者 is_split_into_words 为 False，否则抛出错误信息
    assert self.add_prefix_space or not is_split_into_words, (
        f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
        "to use it with pretokenized inputs."
    )

    # 调用父类的 _encode_plus 方法，并返回其结果
    return super()._encode_plus(*args, **kwargs)

# 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer._decode_with_timestamps 复制而来
def _decode_with_timestamps(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
    """
    Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
    given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
    """
    # 计算时间戳的起始位置，即特殊标记的最后一个 ID 加一
    timestamp_begin = self.all_special_ids[-1] + 1
    # 初始化输出列表
    outputs = [[]]

    # 当前最大时间戳和前一个段落的长度
    cur_max_timestamp = 0.0
    prev_segments_len = 0.0

    # 遍历 token_ids 中的每个 token
    for token in token_ids:
        # 如果 token 大于等于 timestamp_begin，表示是时间戳的 token
        if token >= timestamp_begin:
            # 计算时间戳值，根据时间精度 time_precision 进行计算
            timestamp = float((token - timestamp_begin) * time_precision)

            # 如果时间戳小于当前最大时间戳
            if timestamp < cur_max_timestamp:
                # 下一个段落已开始，更新前一个段落的长度
                prev_segments_len += cur_max_timestamp

            # 更新当前最大时间戳
            cur_max_timestamp = timestamp

            # 将时间戳标记添加到输出列表中
            outputs.append(f"<|{(timestamp + prev_segments_len):.2f}|>")
            # 添加一个空列表，用于存储下一个段落的 token
            outputs.append([])
        else:
            # 如果不是时间戳 token，直接添加到当前段落的列表中
            outputs[-1].append(token)

    # 对 outputs 列表中的每个子列表进行处理，如果是字符串则保持不变，否则调用 decode 方法解码
    outputs = [
        s if isinstance(s, str) else self.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
    ]
    # 将所有子列表合并为一个字符串并返回
    return "".join(outputs)

# 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer._compute_offsets 复制而来
    def _compute_offsets(self, token_ids, time_precision=0.02):
        """
        Compute offsets for a given tokenized input

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        offsets = []
        
        # ensure torch tensor of token ids is placed on cpu
        if "torch" in str(type(token_ids)) and (hasattr(token_ids, "cpu") and callable(token_ids.cpu)):
            token_ids = token_ids.cpu()
        
        token_ids = np.array(token_ids)  # Convert token_ids to a numpy array
        
        # Check if token_ids contains more than one input or is multi-dimensional
        if token_ids.shape[0] > 1 and len(token_ids.shape) > 1:
            raise ValueError("Can only process a single input at a time")
        
        # Define the beginning token index for timestamps
        timestamp_begin = self.all_special_ids[-1] + 1
        
        # Identify tokens that represent timestamps
        timestamp_tokens = token_ids >= timestamp_begin
        
        # Find consecutive timestamp tokens
        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        
        # Handle cases where there are no timestamps or no consecutive timestamps
        if consecutive.shape[0] == 0 and timestamp_tokens.sum() <= 1:
            return []  # Return an empty list if no valid offsets are found
        elif np.where(timestamp_tokens)[0][-1] + 1 not in consecutive:
            consecutive = np.append(consecutive, np.where(timestamp_tokens)[0][-1] + 1)
        
        # Initialize variables to track slices of token_ids
        last_slice = np.where(timestamp_tokens)[0][0]
        
        # Iterate over consecutive timestamp segments
        for current_slice in consecutive:
            sliced_tokens = token_ids[last_slice:current_slice]
            
            # Compute start and end positions of timestamps
            start_timestamp_position = sliced_tokens[0].item() - timestamp_begin
            end_timestamp_position = sliced_tokens[-1].item() - timestamp_begin
            
            # Preprocess token ids to strip timestamp tokens from text output
            sliced_tokens = self._preprocess_token_ids(sliced_tokens)
            
            # Decode token ids into text
            text = self._decode(sliced_tokens)
            
            # Filter out timestamp identifiers from text
            text = self._filter_timestamp_ids(text)
            
            # Calculate and store offset information
            offsets.append(
                {
                    "text": text,
                    "timestamp": (
                        start_timestamp_position * time_precision,
                        end_timestamp_position * time_precision,
                    ),
                }
            )
            
            last_slice = current_slice  # Update last slice index
        
        return offsets  # Return computed offsets
    # 计算给定精度的时间戳标记 ID，并保存到最近最少使用 (LRU) 缓存中
    def timestamp_ids(self, time_precision=0.02):
        return self.convert_tokens_to_ids([("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)])

    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer._preprocess_token_ids 复制而来
    # 预处理令牌 ID，通过移除提示令牌 ID 和时间戳令牌 ID 来为解码做准备
    def _preprocess_token_ids(self, token_ids, skip_special_tokens: bool = False):
        if skip_special_tokens:
            prompt_token_id = self.convert_tokens_to_ids("<|startofprev|>")
            decoder_start_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
            token_ids = self._strip_prompt(token_ids, prompt_token_id, decoder_start_token_id)

        return token_ids

    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer._filter_timestamp_ids 复制而来
    # 过滤时间戳令牌 ID
    def _filter_timestamp_ids(self, token_ids):
        return re.sub(self.timestamp_pat, "", token_ids)

    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer.decode 复制而来
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        output_offsets: bool = False,
        time_precision: float = 0.02,
        decode_with_timestamps: bool = False,
        normalize: bool = False,
        basic_normalize: bool = False,
        remove_diacritics: bool = False,
        **kwargs,
    ):
        """
        解码令牌 ID 为文本。

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                Tokenized input ids list.
            skip_special_tokens (`bool`, optional, defaults to `False`):
                Whether or not to remove special tokens from the token ids. If `True`, the prompt token ids will be
                removed.
            clean_up_tokenization_spaces (`bool`, optional):
                Whether or not to clean up tokenization spaces in the output text.
            output_offsets (`bool`, optional):
                Whether to return the token-level offsets in the original input text.
            time_precision (`float`, optional, defaults to 0.02):
                The time precision used for decoding timestamps.
            decode_with_timestamps (`bool`, optional):
                Whether to decode timestamps along with the tokens.
            normalize (`bool`, optional):
                Whether to normalize the decoded text.
            basic_normalize (`bool`, optional):
                Whether to perform basic normalization on the decoded text.
            remove_diacritics (`bool`, optional):
                Whether to remove diacritics from the decoded text.

        Returns:
            Decoded text as a string.
        """
        if normalize:
            clean_text = self._normalize(text)
            return clean_text
        elif basic_normalize:
            clean_text = self._basic_normalize(text, remove_diacritics=remove_diacritics)
            return clean_text
        else:
            return text

    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer._decode 复制而来
    def _decode(
        self, *args, normalize: bool = False, basic_normalize: bool = False, remove_diacritics: bool = False, **kwargs
    ) -> str:
        """
        解码操作的内部实现。

        Args:
            *args: 传递给超类的参数。
            normalize (`bool`, optional):
                是否对解码后的文本进行规范化处理。
            basic_normalize (`bool`, optional):
                是否对解码后的文本进行基础规范化处理。
            remove_diacritics (`bool`, optional):
                是否移除解码后文本中的变音符号。

        Returns:
            解码后的字符串。
        """
        text = super()._decode(*args, **kwargs)

        if normalize:
            clean_text = self._normalize(text)
            return clean_text
        elif basic_normalize:
            clean_text = self._basic_normalize(text, remove_diacritics=remove_diacritics)
            return clean_text
        else:
            return text

    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer._normalize 复制而来
    # 对文本进行规范化处理
    # 对输入的文本进行规范化处理，已被废弃，在 Transformers 的 v5 版本中将会移除
    def _normalize(self, text):
        warnings.warn(
            "The private method `_normalize` is deprecated and will be removed in v5 of Transformers."
            "You can normalize an input string using the Whisper English normalizer using the `normalize` method."
        )
        return self.normalize(text)

    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer._basic_normalize 复制而来
    # 对输入的文本进行基本规范化处理，已被废弃，在 Transformers 的 v5 版本中将会移除
    def _basic_normalize(self, text, remove_diacritics=False):
        warnings.warn(
            "The private method `_basic_normalize` is deprecated and will be removed in v5 of Transformers."
            "You can normalize an input string using the Whisper basic normalizer using the `basic_normalize` method."
        )
        return self.basic_normalize(text, remove_diacritics=remove_diacritics)

    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer.normalize 复制而来
    # 使用 `EnglishTextNormalizer` 类对给定的字符串进行规范化处理，执行常见的英文文本转换
    def normalize(self, text):
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        return normalizer(text)

    @staticmethod
    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer.basic_normalize 复制而来
    # 使用 `BasicTextNormalizer` 类对给定的字符串进行规范化处理，执行常见的多语言文本转换
    def basic_normalize(text, remove_diacritics=False):
        normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
        return normalizer(text)

    # 保存词汇表到指定的目录，返回保存的文件列表和规范化文件
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 调用 tokenizer 对象的 model 的 save 方法，将词汇保存到指定目录中
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)

        # 初始化规范化文件路径
        normalizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["normalizer_file"]
        )

        # 如果存在英文拼写规范化器，则将其保存到规范化文件中
        if self.english_spelling_normalizer is not None:
            with open(normalizer_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )

        # 返回保存的文件列表和规范化文件路径
        return tuple(files) + (normalizer_file,)
    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```
        >>> # instantiate the tokenizer and set the prefix token to Spanish
        >>> tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", language="spanish")
        >>> # now switch the prefix token from Spanish to French
        >>> tokenizer.set_prefix_tokens(language="french")
        ```

        Args:
            language (`str`, *optional*, defaults to `None`):
                The language of the transcription text.
            task (`str`, *optional*, defaults to `None`):
                Task identifier to append at the start of sequence (if any).
            predict_timestamps (`bool`, *optional*, defaults to `None`):
                Whether to omit the `<|notimestamps|>` token at the start of the sequence.
        """
        # 设置语言参数，如果未提供则使用当前实例的语言
        self.language = language if language is not None else self.language
        # 设置任务参数，如果未提供则使用当前实例的任务
        self.task = task if task is not None else self.task
        # 设置预测时间戳参数，如果未提供则使用当前实例的预测时间戳设置
        self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps

        # 获取当前实例的前缀 token IDs
        prefix_token_ids = self.prefix_tokens
        # 将前缀 token IDs 转换为 token 列表
        prefixes = self.convert_ids_to_tokens(prefix_token_ids)
        # 获取结束符号（EOS）
        eos = self.eos_token
        # 获取结束符号（EOS）的 token ID
        eos_token_id = self.eos_token_id
        # 构建前缀模板，用于后续的处理
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        # 设置后处理器，使用模板处理器，定义单个和对称序列的格式
        self.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{prefix_template} $A:0 {eos}:0",
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            special_tokens=[
                (eos, eos_token_id),  # 添加结束符号（EOS）和其 token ID 到特殊 token 列表
                *zip(prefixes, prefix_token_ids),  # 添加前缀 token 和对应的 token ID 到特殊 token 列表
            ],
        )

    @property
    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer.prefix_tokens 复制而来
    # 返回一个列表，包含特定的起始标记的 ID
    def prefix_tokens(self) -> List[int]:
        # 将 "<|startoftranscript|>" 转换为对应的 ID
        bos_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
        # 将 "<|translate|>" 转换为对应的 ID
        translate_token_id = self.convert_tokens_to_ids("<|translate|>")
        # 将 "<|transcribe|>" 转换为对应的 ID
        transcribe_token_id = self.convert_tokens_to_ids("<|transcribe|>")
        # 将 "<|notimestamps|>" 转换为对应的 ID
        notimestamps_token_id = self.convert_tokens_to_ids("<|notimestamps|>")
        # 获取所有语言的键值对的元组
        langs = tuple(LANGUAGES.keys())

        if self.language is not None:
            # 将语言名称转换为小写
            self.language = self.language.lower()
            # 检查语言是否在 TO_LANGUAGE_CODE 中存在，如果是则获取对应的语言 ID
            if self.language in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[self.language]
            # 如果语言名称不在 TO_LANGUAGE_CODE 中，则检查是否是语言 ID，如果是则直接使用
            elif self.language in TO_LANGUAGE_CODE.values():
                language_id = self.language
            else:
                # 如果语言既不是名称也不是 ID，则抛出错误
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )

        if self.task is not None:
            # 检查任务是否在支持的任务列表 TASK_IDS 中，如果不是则抛出错误
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        bos_sequence = [bos_token_id]
        if self.language is not None:
            # 如果有指定语言，则将其对应的特定 ID 添加到序列中
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        if self.task is not None:
            # 如果有指定任务，则根据任务类型添加对应的特定 ID 到序列中
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        if not self.predict_timestamps:
            # 如果不需要预测时间戳，则添加对应的特定 ID 到序列中
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence

    # 从给定的 token_ids_0 和 token_ids_1 构建模型输入，同时添加结束标记 eos_token_id
    # 来自 transformers.models.whisper.tokenization_whisper.WhisperTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            # 如果没有第二个序列，则只需将 token_ids_0 与前缀序列和结束标记连接起来
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # 对于存在第二个序列的情况，按照API的一般性保留了对成对逻辑的处理
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    # 获取特殊 token 的掩码，来自 transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_special_tokens_mask
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ):
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        # Check if special tokens are already added; if so, delegate to superclass method
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # Initialize prefix tokens as all 1s (special tokens)
        prefix_ones = [1] * len(self.prefix_tokens)
        # Suffix token is always 1 (special token)
        suffix_ones = [1]

        # If token_ids_1 is None, return concatenated list of prefix, sequence tokens, and suffix
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        
        # If token_ids_1 exists, return concatenated list of prefix, sequence tokens from both lists, and suffix
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    @property
    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        # Issue a warning if no specific chat template is defined
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        # Return the default chat template string
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_decoder_prompt_ids
    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        # Set prefix tokens based on task, language, and timestamp settings
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
        
        # Prefix tokens format: <|startoftranscript|> <|lang_id|> <|task|> <|notimestamps|>
        # Exclude the first token (`<|startoftranscript|>`) as it is the starting token for generation
        # Extract the remaining prefix tokens for decoder prompt IDs
        forced_tokens = self.prefix_tokens[1:]
        # Pair each token with its rank shifted by one (rank + 1) for decoder prompt IDs
        forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]
        
        # Return the list of forced decoder prompt IDs
        return forced_decoder_ids
    # 将指定的 ASR 模型输出解码为文本，根据参数选择是否返回时间戳、语言和时间精度
    def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
        return _decode_asr(
            self,
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )

    # 从WhisperTokenizer类中复制的方法，用于获取文本的提示词ID
    def get_prompt_ids(self, text: str, return_tensors="np"):
        """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
        # 使用WhisperTokenizer实例的编码方法将特殊标记加入到输入文本前，并进行批处理编码
        batch_encoding = self("<|startofprev|>", " " + text.strip(), add_special_tokens=False)

        # 检查特殊标记
        prompt_text_ids = batch_encoding["input_ids"][1:]  # 移除首个特殊标记
        special_token_id = next((x for x in prompt_text_ids if x >= self.all_special_ids[0]), None)
        if special_token_id is not None:
            token = self.convert_ids_to_tokens(special_token_id)
            raise ValueError(f"Encountered text in the prompt corresponding to disallowed special token: {token}.")

        # 将批处理编码转换为指定类型的张量（如numpy）
        batch_encoding.convert_to_tensors(tensor_type=return_tensors)
        return batch_encoding["input_ids"]  # 返回输入文本对应的ID列表

    @staticmethod
    # 从WhisperTokenizer类中复制的方法，用于从标记ID列表中删除提示标记
    def _strip_prompt(token_ids: List[int], prompt_token_id: int, decoder_start_token_id: int):
        has_prompt = isinstance(token_ids, list) and token_ids and token_ids[0] == prompt_token_id
        if has_prompt:
            if decoder_start_token_id in token_ids:
                return token_ids[token_ids.index(decoder_start_token_id) :]  # 返回从解码开始标记到结尾的标记ID列表
            else:
                return []  # 如果没有解码开始标记，则返回空列表

        return token_ids  # 如果没有提示标记，则直接返回标记ID列表
```