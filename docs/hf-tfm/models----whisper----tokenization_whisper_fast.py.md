# `.\transformers\models\whisper\tokenization_whisper_fast.py`

```
# 设置文件编码为 UTF-8

# 版权声明和许可证信息

# 引入必要的库和模块
"""Tokenization classes for Whisper."""
import json  # 导入用于 JSON 操作的模块
import os  # 导入用于操作系统路径的模块
import re  # 导入正则表达式模块
from functools import lru_cache  # 导入用于缓存函数调用结果的装饰器
from typing import List, Optional, Tuple  # 引入类型提示

import numpy as np  # 导入 NumPy 库
from tokenizers import AddedToken, pre_tokenizers, processors  # 导入 tokenizers 库中的一些类

from ...tokenization_utils_base import BatchEncoding  # 导入用于处理批编码的类
from ...tokenization_utils_fast import PreTrainedTokenizerFast  # 导入用于快速分词的预训练分词器基类
from ...utils import logging  # 导入日志记录工具
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer  # 导入英文文本规范化类
from .tokenization_whisper import LANGUAGES, TASK_IDS, TO_LANGUAGE_CODE, WhisperTokenizer, _decode_asr  # 导入 Whisper 分词器及相关辅助函数

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义与文件相关的常量
VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",  # 词汇表文件名
    "tokenizer_file": "tokenizer.json",  # 分词器文件名
    "merges_file": "merges.txt",  # 合并文件名
    "normalizer_file": "normalizer.json",  # 规范化器文件名
}

# 定义预训练模型与其对应文件的映射关系
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/vocab.json",  # 微型预训练模型的词汇表文件
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/vocab.json",  # 基础预训练模型的词汇表文件
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/vocab.json",  # 小型预训练模型的词汇表文件
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/vocab.json",  # 中型预训练模型的词汇表文件
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/vocab.json",  # 大型预训练模型的词汇表文件
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/vocab.json",  # 英文微型预训练模型的词汇表文件
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/vocab.json",  # 英文基础预训练模型的词汇表文件
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/vocab.json",  # 英文小型预训练模型的词汇表文件
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/vocab.json",  # 英文中型预训练模型的词汇表文件
    },
    "merges_file": {
        # merges文件的键值对，键为模型名称，值为merges文件的URL
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
        # tokenizer文件的键值对，键为模型名称，值为tokenizer文件的URL
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
# 定义预训练模型的位置编码嵌入尺寸字典，将模型名称映射到其对应的位置编码嵌入尺寸
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
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

# 定义 WhisperTokenizerFast 类，继承自 PreTrainedTokenizerFast
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

    # 词汇文件名列表，用于定义词汇文件名
    vocab_files_names = VOCAB_FILES_NAMES
    # 预训练模型的词汇文件映射，将模型名称映射到其对应的词汇文件路径
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    # 定义静态变量：最大模型输入尺寸为预训练位置嵌入尺寸
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # 定义静态变量：模型输入的名称列表
    model_input_names = ["input_ids", "attention_mask"]
    # 定义静态变量：慢速分词器的类名为WhisperTokenizer
    slow_tokenizer_class = WhisperTokenizer
    
    # 初始化函数，接收多个参数
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
        # 如果bos_token是字符串类型，将其封装成特殊标记的AddedToken
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(bos_token, str)
            else bos_token
        )
        # 如果eos_token是字符串类型，将其封装成特殊标记的AddedToken
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(eos_token, str)
            else eos_token
        )
        # 如果unk_token是字符串类型，将其封装成特殊标记的AddedToken
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, normalized=False, special=True)
            if isinstance(unk_token, str)
            else unk_token
        )
    
        # 调用父类的初始化函数，传入参数进行初始化
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
    
        # 将参数kwargs中的add_bos_token取出并赋值给self.add_bos_token
        self.add_bos_token = kwargs.pop("add_bos_token", False)
    
        # 获取当前backend_tokenizer的pre_tokenizer状态，转为字典格式
        pre_tok_state = json.loads(self.backend_tokenizer.pre_tokenizer.__getstate__())
        # 如果pre_tok_state中的add_prefix_space与输入参数add_prefix_space不一致
        if pre_tok_state.get("add_prefix_space", add_prefix_space) != add_prefix_space:
            # 获取pre_tokenizers中对应类型的类
            pre_tok_class = getattr(pre_tokenizers, pre_tok_state.pop("type"))
            # 更新pre_tok_state中的add_prefix_space为输入参数add_prefix_space
            pre_tok_state["add_prefix_space"] = add_prefix_space
            # 重新创建backend_tokenizer的pre_tokenizer对象
            self.backend_tokenizer.pre_tokenizer = pre_tok_class(**pre_tok_state)
    
        # 如果normalizer_file不为None
        if normalizer_file is not None:
            # 打开normalizer_file，读取其中的���容，保存到self.english_spelling_normalizer中
            with open(normalizer_file, encoding="utf-8") as vocab_handle:
                self.english_spelling_normalizer = json.load(vocab_handle)
        else:
            # normalizer_file为None时，self.english_spelling_normalizer设置为None
            self.english_spelling_normalizer = None
    
        # 将add_prefix_space参数赋值给self.add_prefix_space
        self.add_prefix_space = add_prefix_space
        # 编译时间戳的正则表达式模式
        self.timestamp_pat = re.compile(r"<\|(\d+\.\d+)\|>")
    
        # 将参数language、task、predict_timestamps赋值给self.language、self.task、self.predict_timestamps
        self.language = language
        self.task = task
        self.predict_timestamps = predict_timestamps
    
    # 从transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast中复制的_batch_encode_plus函数
    def _batch_encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取is_split_into_words参数，默认为False
        is_split_into_words = kwargs.get("is_split_into_words", False)
        # 如果add_prefix_space为False且is_split_into_words为True，抛出异常
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )
    
        # 调用父类的_batch_encode_plus函数，并返回结果
        return super()._batch_encode_plus(*args, **kwargs)
    
    # 从transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast中复制的_encode_plus函数
    # 对文本进行编码，返回编码后的 BatchEncoding 对象
    def _encode_plus(self, *args, **kwargs) -> BatchEncoding:
        # 获取是否已经分成单词的参数
        is_split_into_words = kwargs.get("is_split_into_words", False)

        # 检查是否需要在单词前添加空格，如果已经分成单词则不需要
        assert self.add_prefix_space or not is_split_into_words, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        # 调用父类的 _encode_plus 方法对输入进行编码
        return super()._encode_plus(*args, **kwargs)

    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer._decode_with_timestamps 复制而来
    # 根据带有时间戳的标记进行解码，生成文本结果
    def _decode_with_timestamps(self, token_ids, skip_special_tokens=False, time_precision=0.02) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`. This method decodes
        given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        # 定义时间戳开始的位置
        timestamp_begin = self.all_special_ids[-1] + 1
        outputs = [[]]

        cur_max_timestamp = 0.0
        prev_segments_len = 0.0

        # 遍历标记列表
        for token in token_ids:
            if token >= timestamp_begin:
                # 计算时间戳
                timestamp = float((token - timestamp_begin) * time_precision)

                if timestamp < cur_max_timestamp:
                    # 下一个段落开始
                    prev_segments_len += cur_max_timestamp

                cur_max_timestamp = timestamp

                # 添加带有时间戳的标记
                outputs.append(f"<|{(timestamp + prev_segments_len):.2f}|>")
                outputs.append([])
            else:
                # 添加标记
                outputs[-1].append(token)
        
        # 对标记列表进行解码，跳过特殊标记
        outputs = [
            s if isinstance(s, str) else self.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
        ]
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
        # 初始化一个空列表用于存储偏移量
        offsets = []
        # 确保 torch tensor 的 token ids 放置在 CPU 上
        if "torch" in str(type(token_ids)) and (hasattr(token_ids, "cpu") and callable(token_ids.cpu)):
            # 如果 token_ids 是 torch tensor，则将其移到 CPU 上
            token_ids = token_ids.cpu()
        # 将 token_ids 转换为 numpy 数组
        token_ids = np.array(token_ids)
        # 如果 token_ids 的行数大于1且维度大于1，则抛出异常
        if token_ids.shape[0] > 1 and len(token_ids.shape) > 1:
            raise ValueError("Can only process a single input at a time")
        # 定义时间戳开始位置
        timestamp_begin = self.all_special_ids[-1] + 1
        # 找到 token 中的时间戳位置
        timestamp_tokens = token_ids >= timestamp_begin

        # 找出连续的时间戳
        consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
        # 如果没有连续的时间戳或者时间戳数量小于等于1，则返回空列表
        if consecutive.shape[0] == 0 and timestamp_tokens.sum() <= 1:
            return []
        # 如果时间戳位置不在连续时间戳列表中，则将最后一个时间戳位置添加到列表中
        elif np.where(timestamp_tokens)[0][-1] + 1 not in consecutive:
            consecutive = np.append(consecutive, np.where(timestamp_tokens)[0][-1] + 1)

        # 初始化上一个切片的位置
        last_slice = np.where(timestamp_tokens)[0][0]
        for current_slice in consecutive:
            # 切割 token_ids 以获取每个时间戳区间的 tokens
            sliced_tokens = token_ids[last_slice:current_slice]
            if len(sliced_tokens) > 1:
                # 计算开始和结束时间戳的位置
                start_timestamp_position = sliced_tokens[0].item() - timestamp_begin
                end_timestamp_position = sliced_tokens[-1].item() - timestamp_begin
                # 处理 token_ids 中的时间戳，获取文本输出
                sliced_tokens = self._preprocess_token_ids(sliced_tokens)
                text = self._decode(sliced_tokens)
                text = self._filter_timestamp_ids(text)
                # 将文本和时间戳位置存储到偏移量列表中
                offsets.append(
                    {
                        "text": text,
                        "timestamp": (
                            start_timestamp_position * time_precision,
                            end_timestamp_position * time_precision,
                        ),
                    }
                )
            last_slice = current_slice

        return offsets

    @lru_cache
    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer.timestamp_ids 复制而来
    def timestamp_ids(self, time_precision=0.02):
        """
        Compute the timestamp token ids for a given precision and save to least-recently used (LRU) cache.

        Args:
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        # 返回时间戳 token ids 列表，根据给定精度并保存到最近最少使用 (LRU) 缓存
        return self.convert_tokens_to_ids([("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)])

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._preprocess_token_ids
    def _preprocess_token_ids(self, token_ids, skip_special_tokens: bool = False):
        """
        Pre-process the token ids for decoding by removing the prompt tokens ids and timestamp token ids.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Typically, obtained using the `__call__` method of the tokenizer.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens from the token ids. If `True`, the prompt token ids will be
                removed.
        """
        # 如果需要跳过特殊 token，则移除提示 token ids 和时间戳 token ids
        if skip_special_tokens:
            prompt_token_id = self.convert_tokens_to_ids("<|startofprev|>")
            decoder_start_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
            token_ids = self._strip_prompt(token_ids, prompt_token_id, decoder_start_token_id)

        return token_ids

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._filter_timestamp_ids
    def _filter_timestamp_ids(self, token_ids):
        # 过滤时间戳 token ids
        return re.sub(self.timestamp_pat, "", token_ids)

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.decode
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
    def _decode(
        self, *args, normalize: bool = False, basic_normalize: bool = False, remove_diacritics: bool = False, **kwargs
    ) -> str:
        # 调用父类方法对 token ids 进行解码
        text = super()._decode(*args, **kwargs)

        # 根据参数进行文本清洗处理
        if normalize:
            clean_text = self._normalize(text)
            return clean_text
        elif basic_normalize:
            clean_text = self._basic_normalize(text, remove_diacritics=remove_diacritics)
            return clean_text
        else:
            return text

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer._normalize
    # 使用 EnglishTextNormalizer 类对文本进行标准化处理
    def _normalize(self, text):
        """
        Normalize a given string using the `EnglishTextNormalizer` class, which preforms commons transformation on
        english text.
        """
        # 创建 EnglishTextNormalizer 对象，传入拼写标准化器
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        # 使用 normalizer 对象对文本进行标准化处理，并返回结果
        return normalizer(text)
    
    # 使用 BasicTextNormalizer 类对文本进行基本标准化处理
    @staticmethod
    def _basic_normalize(text, remove_diacritics=False):
        """
        Normalize a given string using the `BasicTextNormalizer` class, which preforms commons transformation on
        multilingual text.
        """
        # 创建 BasicTextNormalizer 对象，传入是否移除音调符号的参数
        normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
        # 使用 normalizer 对象对文本进行标准化处理，并返回结果
        return normalizer(text)
    
    # 保存分词器及其配置到指定目录
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        # 保存分词器模型到指定目录，返回保存的文件名列表
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
    
        # 构建标准化器配置文件的路径
        normalizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["normalizer_file"]
        )
    
        # 如果存在英语拼写标准化器，则将其保存到标准化器配置文件
        if self.english_spelling_normalizer is not None:
            with open(normalizer_file, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(self.english_spelling_normalizer, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                )
    
        # 返回保存的所有文件名
        return tuple(files) + (normalizer_file,)
    # 设置自定义前缀标记，可用于更新细微调整
    def set_prefix_tokens(self, language: str = None, task: str = None, predict_timestamps: bool = None):
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```python
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
        # 如果 language 参数不为 None，则使用 language 参数，否则使用 self.language
        self.language = language if language is not None else self.language
        # 如果 task 参数不为 None，则使用 task 参数，否则使用 self.task
        self.task = task if task is not None else self.task
        # 如果 predict_timestamps 参数不为 None，则使用 predict_timestamps 参数，否则使用 self.predict_timestamps
        self.predict_timestamps = predict_timestamps if predict_timestamps is not None else self.predict_timestamps

        # 获取前缀标记的 ID
        prefix_token_ids = self.prefix_tokens
        # 将前缀标记的 ID 转换为标记
        prefixes = self.convert_ids_to_tokens(prefix_token_ids)
        # 获取结束标记
        eos = self.eos_token
        # 获取结束标记的 ID
        eos_token_id = self.eos_token_id
        # 生成前缀模板
        prefix_template = " ".join([f"{token}:0" for token in prefixes])
        # 设置后端处理器的模板处理
        self.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{prefix_template} $A:0 {eos}:0",
            pair=f"{prefix_template} $A:0 $B:1 {eos}:1",
            special_tokens=[
                (eos, eos_token_id),
                *zip(prefixes, prefix_token_ids),
            ],
        )

    @property
    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.prefix_tokens
    # 返回一个列表，包含特殊标记的 token_id
    def prefix_tokens(self) -> List[int]:
        # 将“<|startoftranscript|>”转换为其对应的 token_id
        bos_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
        # 将“<|translate|>”转换为其对应的 token_id
        translate_token_id = self.convert_tokens_to_ids("<|translate|>")
        # 将“<|transcribe|>”转换为其对应的 token_id
        transcribe_token_id = self.convert_tokens_to_ids("<|transcribe|>")
        # 将“<|notimestamps|>”转换为其对应的 token_id
        notimestamps_token_id = self.convert_tokens_to_ids("<|notimestamps|>")
        # 获取语言列表的元组
        langs = tuple(LANGUAGES.keys())

        # 如果语言不为空
        if self.language is not None:
            # 将语言转换为小写
            self.language = self.language.lower()
            # 如果语言在 TO_LANGUAGE_CODE 中
            if self.language in TO_LANGUAGE_CODE:
                # 获取语言的 ID
                language_id = TO_LANGUAGE_CODE[self.language]
            # 如果语言在 TO_LANGUAGE_CODE 值中
            elif self.language in TO_LANGUAGE_CODE.values():
                # 获取语言的 ID
                language_id = self.language
            else:
                # 检查语言代码的长度是否为 2，如果不是则抛出异常
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )

        # 如果任务不为空
        if self.task is not None:
            # 如果任务不在 TASK_IDS 中，则抛出异常
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        # 构建 bos_sequence，包含特殊标记的 token_id
        bos_sequence = [bos_token_id]
        # 如果语言不为空
        if self.language is not None:
            # 添加语言的特殊标记 token_id
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        # 如果任务不为空
        if self.task is not None:
            # 根据任务类型添加特殊标记 token_id
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else translate_token_id)
        # 如果不需要预测时间戳，则添加对应的特殊标记 token_id
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        # 返回特殊标记的 token_id 列表
        return bos_sequence

    # 从一个序列中构建模型的输入，末尾添加 eos_token_id
    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer.build_inputs_with_special_tokens 复制而来
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        # 如果 token_ids_1 为空
        if token_ids_1 is None:
            # 返回包含特殊标记 token_id 的列表，以及 token_ids_0 和 eos_token_id
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # 不期望处理成对的序列，但为了 API 一致性，保留成对逻辑
        # 返回包含特殊标记 token_id 的列表，以及 token_ids_0、token_ids_1 和 eos_token_id
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

    # 获取特殊标记的 token 掩码
    # 从 transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_special_tokens_mask 复制而来
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

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)  # 创建一个长度为前缀标记数量的元素都为1的列表
        suffix_ones = [1]  # 创建一个包含单个1的列表
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones  # 如果没有第二个ID列表，返回前缀标记，后缀标记和零填充的第一个ID列表
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones  # 否则返回前缀标记，零填充的第一个ID列表，零填充的第二个ID列表和后缀标记

    @property
    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.default_chat_template
    def default_chat_template(self):
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        return "{% for message in messages %}" "{{ message.content }}{{ eos_token }}" "{% endfor %}"  # 返回一个简单的聊天模板，忽略角色信息，只是使用EOS标记连接消息

    # Copied from transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_decoder_prompt_ids
    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)  # 设置前缀标记
        forced_tokens = self.prefix_tokens[1:]  # 忽略了BOS标记的前缀标记
        forced_decoder_ids = [(rank + 1, token) for rank, token in enumerate(forced_tokens)]  # 获取强制解码器ID
        return forced_decoder_ids  # 返回强制解码器ID
    # 将ASR模型输出解码成可读文本，根据参数决定是否返回时间戳、语言和时间精度
    def _decode_asr(self, model_outputs, *, return_timestamps, return_language, time_precision):
        return _decode_asr(
            self,
            model_outputs,
            return_timestamps=return_timestamps,
            return_language=return_language,
            time_precision=time_precision,
        )

    # 从transformers.models.whisper.tokenization_whisper.WhisperTokenizer.get_prompt_ids复制过来的方法
    def get_prompt_ids(self, text: str, return_tensors="np"):
        """将提示文本转换为可以传递给[`~WhisperForConditionalGeneration.generate`]的ID。"""
        # 使用WhisperTokenizer对文本进行编码，删除前后空格并添加特殊标记
        batch_encoding = self("<|startofprev|>", " " + text.strip(), add_special_tokens=False)

        # 检查特殊标记
        prompt_text_ids = batch_encoding["input_ids"][1:]  # 跳过特殊标记，获取 prompt 文本的 ID
        special_token_id = next((x for x in prompt_text_ids if x >= self.all_special_ids[0]), None)  # 获取第一个特殊标记的 ID

        # 如果遇到不允许的特殊标记，抛出异常
        if special_token_id is not None:
            token = self.convert_ids_to_tokens(special_token_id)
            raise ValueError(f"遇到提示文本中对应不允许的特殊标记: {token}.")

        # 将编码转换成张量并返回输入 ID
        batch_encoding.convert_to_tensors(tensor_type=return_tensors)
        return batch_encoding["input_ids"]

    @staticmethod
    # 从transformers.models.whisper.tokenization_whisper.WhisperTokenizer._strip_prompt复制过来的方法
    def _strip_prompt(token_ids: List[int], prompt_token_id: int, decoder_start_token_id: int):
        # 检查文本是否包含提示
        has_prompt = isinstance(token_ids, list) and token_ids and token_ids[0] == prompt_token_id
        
        # 如果文本包含提示，则返回从解码器开始标记到最后的部分
        if has_prompt:
            if decoder_start_token_id in token_ids:
                return token_ids[token_ids.index(decoder_start_token_id) :]
            else:
                return []

        return token_ids  # 如果文本不包含提示，则原样返回
```