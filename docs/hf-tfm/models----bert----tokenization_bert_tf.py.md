# `.\transformers\models\bert\tokenization_bert_tf.py`

```py
import os  # 导入操作系统模块
from typing import List, Union  # 导入类型提示模块

import tensorflow as tf  # 导入 TensorFlow 库
from tensorflow_text import BertTokenizer as BertTokenizerLayer  # 从 tensorflow_text 模块导入 BertTokenizerLayer 类
from tensorflow_text import FastBertTokenizer, ShrinkLongestTrimmer, case_fold_utf8, combine_segments, pad_model_inputs  # 导入其他相关功能模块

from .tokenization_bert import BertTokenizer  # 从当前包中导入 tokenization_bert 模块的 BertTokenizer 类


class TFBertTokenizer(tf.keras.layers.Layer):  # 定义 TFBertTokenizer 类，继承自 TensorFlow 的 Layer 类
    """
    This is an in-graph tokenizer for BERT. It should be initialized similarly to other tokenizers, using the
    `from_pretrained()` method. It can also be initialized with the `from_tokenizer()` method, which imports settings
    from an existing standard tokenizer object.

    In-graph tokenizers, unlike other Hugging Face tokenizers, are actually Keras layers and are designed to be run
    when the model is called, rather than during preprocessing. As a result, they have somewhat more limited options
    than standard tokenizer classes. They are most useful when you want to create an end-to-end model that goes
    straight from `tf.string` inputs to outputs.
    """
```  
    Args:
        vocab_list (`list`):
            List containing the vocabulary. 词汇表列表
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing. 是否在标记化时将输入转换为小写
        cls_token_id (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
            用于序列分类时使用的分类器标记（对整个序列进行分类而不是对每个标记进行分类）。当使用特殊标记构建序列时，它是序列的第一个标记。
        sep_token_id (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
            用于从多个序列构建序列时使用的分隔符标记，例如，用于序列分类的两个序列或用于文本和问题的问题回答。它也用作使用特殊标记构建的序列的最后一个标记。
        pad_token_id (`str`, *optional*, defaults to `"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
            用于填充的标记，例如在批处理不同长度的序列时使用。
        padding (`str`, defaults to `"longest"`):
            The type of padding to use. Can be either `"longest"`, to pad only up to the longest sample in the batch,
            or `"max_length", to pad all inputs to the maximum length supported by the tokenizer.
            要使用的填充类型。可以是“longest”，仅填充到批处理中最长的样本，或“max_length”，将所有输入填充到令牌化器支持的最大长度。
        truncation (`bool`, *optional*, defaults to `True`):
            Whether to truncate the sequence to the maximum length. 是否将序列截断到最大长度
        max_length (`int`, *optional*, defaults to `512`):
            The maximum length of the sequence, used for padding (if `padding` is "max_length") and/or truncation (if
            `truncation` is `True`).
            序列的最大长度，用于填充（如果`padding`为“max_length”）和/或截断（如果`truncation`为`True`）。
        pad_to_multiple_of (`int`, *optional*, defaults to `None`):
            If set, the sequence will be padded to a multiple of this value.
            如果设置，序列将填充到此值的倍数。
        return_token_type_ids (`bool`, *optional*, defaults to `True`):
            Whether to return token_type_ids. 是否返回 token_type_ids
        return_attention_mask (`bool`, *optional*, defaults to `True`):
            Whether to return the attention_mask. 是否返回 attention_mask
        use_fast_bert_tokenizer (`bool`, *optional*, defaults to `True`):
            If True, will use the FastBertTokenizer class from Tensorflow Text. If False, will use the BertTokenizer
            class instead. BertTokenizer supports some additional options, but is slower and cannot be exported to
            TFLite.
            如果为 True，则将使用来自 Tensorflow Text 的 FastBertTokenizer 类。如果为 False，则将使用 BertTokenizer 类。BertTokenizer 支持一些额外选项，但速度较慢且无法导出到 TFLite。
    """

    def __init__(
        self,
        vocab_list: List,
        do_lower_case: bool,
        cls_token_id: int = None,
        sep_token_id: int = None,
        pad_token_id: int = None,
        padding: str = "longest",
        truncation: bool = True,
        max_length: int = 512,
        pad_to_multiple_of: int = None,
        return_token_type_ids: bool = True,
        return_attention_mask: bool = True,
        use_fast_bert_tokenizer: bool = True,
        **tokenizer_kwargs,
        ):
        # 调用父类的构造函数
        super().__init__()
        # 如果使用快速的BERT分词器
        if use_fast_bert_tokenizer:
            # 使用FastBertTokenizer创建tf_tokenizer对象
            self.tf_tokenizer = FastBertTokenizer(
                vocab_list, token_out_type=tf.int64, lower_case_nfd_strip_accents=do_lower_case, **tokenizer_kwargs
            )
        else:
            # 创建静态词汇表查找表
            lookup_table = tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=vocab_list,
                    key_dtype=tf.string,
                    values=tf.range(tf.size(vocab_list, out_type=tf.int64), dtype=tf.int64),
                    value_dtype=tf.int64,
                ),
                num_oov_buckets=1,
            )
            # 使用BertTokenizerLayer创建tf_tokenizer对象
            self.tf_tokenizer = BertTokenizerLayer(
                lookup_table, token_out_type=tf.int64, lower_case=do_lower_case, **tokenizer_kwargs
            )

        # 设置对象的属性
        self.vocab_list = vocab_list
        self.do_lower_case = do_lower_case
        self.cls_token_id = cls_token_id or vocab_list.index("[CLS]")
        self.sep_token_id = sep_token_id or vocab_list.index("[SEP]")
        self.pad_token_id = pad_token_id or vocab_list.index("[PAD]")
        self.paired_trimmer = ShrinkLongestTrimmer(max_length - 3, axis=1)  # Allow room for special tokens
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_token_type_ids = return_token_type_ids
        self.return_attention_mask = return_attention_mask

    # 类方法
    @classmethod
    def from_tokenizer(cls, tokenizer: "PreTrainedTokenizerBase", **kwargs):  # noqa: F821
        """
        从现有的 `Tokenizer` 初始化一个 `TFBertTokenizer`。

        Args:
            tokenizer (`PreTrainedTokenizerBase`):
                用于初始化 `TFBertTokenizer` 的分词器。

        Examples:

        ```py
        from transformers import AutoTokenizer, TFBertTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tf_tokenizer = TFBertTokenizer.from_tokenizer(tokenizer)
        ```
        """
        # 获取参数中的 do_lower_case，如果没有则使用 tokenizer 的值
        do_lower_case = kwargs.pop("do_lower_case", None)
        do_lower_case = tokenizer.do_lower_case if do_lower_case is None else do_lower_case
        # 获取参数中的 cls_token_id，如果没有则使用 tokenizer 的值
        cls_token_id = kwargs.pop("cls_token_id", None)
        cls_token_id = tokenizer.cls_token_id if cls_token_id is None else cls_token_id
        # 获取参数中的 sep_token_id，如果没有则使用 tokenizer 的值
        sep_token_id = kwargs.pop("sep_token_id", None)
        sep_token_id = tokenizer.sep_token_id if sep_token_id is None else sep_token_id
        # 获取参数中的 pad_token_id，如果没有则使用 tokenizer 的值
        pad_token_id = kwargs.pop("pad_token_id", None)
        pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id

        # 获取 tokenizer 的词汇表
        vocab = tokenizer.get_vocab()
        # 按照词汇表中的索引排序
        vocab = sorted(vocab.items(), key=lambda x: x[1])
        # 提取词汇表中的词项
        vocab_list = [entry[0] for entry in vocab]
        # 使用参数和提取的词汇表初始化 TFBertTokenizer
        return cls(
            vocab_list=vocab_list,
            do_lower_case=do_lower_case,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        """
        从预训练的分词器实例化一个 `TFBertTokenizer`。

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                预训练分词器的名称或路径。

        Examples:

        ```py
        from transformers import TFBertTokenizer

        tf_tokenizer = TFBertTokenizer.from_pretrained("bert-base-uncased")
        ```
        """
        try:
            # 尝试使用 BertTokenizer 实例化分词器
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        except:  # noqa: E722
            from .tokenization_bert_fast import BertTokenizerFast

            # 如果出错，则使用 BertTokenizerFast 实例化分词器
            tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        # 从分词器实例化 TFBertTokenizer
        return cls.from_tokenizer(tokenizer, **kwargs)

    def unpaired_tokenize(self, texts):
        # 如果设置了 do_lower_case，则进行小写转换
        if self.do_lower_case:
            texts = case_fold_utf8(texts)
        # 使用 tf_tokenizer 进行分词
        tokens = self.tf_tokenizer.tokenize(texts)
        # 合并维度
        return tokens.merge_dims(1, -1)

    def call(
        self,
        text,
        text_pair=None,
        padding=None,
        truncation=None,
        max_length=None,
        pad_to_multiple_of=None,
        return_token_type_ids=None,
        return_attention_mask=None,
    # 获取配置信息的方法，返回一个包含各项配置信息的字典
    def get_config(self):
        # 返回包含以下配置信息的字典：
        return {
            "vocab_list": self.vocab_list,  # 词汇表列表
            "do_lower_case": self.do_lower_case,  # 是否进行小写处理的布尔值
            "cls_token_id": self.cls_token_id,  # [CLS] 标记的 ID
            "sep_token_id": self.sep_token_id,  # [SEP] 标记的 ID
            "pad_token_id": self.pad_token_id,  # 填充标记的 ID
        }
```