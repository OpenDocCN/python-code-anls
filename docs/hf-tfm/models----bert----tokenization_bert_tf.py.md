# `.\models\bert\tokenization_bert_tf.py`

```py
    # 导入所需的标准库和模块
    import os
    from typing import List, Union

    # 导入 TensorFlow 库
    import tensorflow as tf
    # 导入 TensorFlow Text 库中的 BERT 分词器
    from tensorflow_text import BertTokenizer as BertTokenizerLayer
    from tensorflow_text import FastBertTokenizer, ShrinkLongestTrimmer, case_fold_utf8, combine_segments, pad_model_inputs

    # 导入自定义的 Keras 辅助函数
    from ...modeling_tf_utils import keras
    # 导入自定义的 BERT 分词器
    from .tokenization_bert import BertTokenizer

    # 定义一个 Keras 层，用于在图中进行 BERT 分词
    class TFBertTokenizer(keras.layers.Layer):
        """
        This is an in-graph tokenizer for BERT. It should be initialized similarly to other tokenizers, using the
        `from_pretrained()` method. It can also be initialized with the `from_tokenizer()` method, which imports settings
        from an existing standard tokenizer object.

        In-graph tokenizers, unlike other Hugging Face tokenizers, are actually Keras layers and are designed to be run
        when the model is called, rather than during preprocessing. As a result, they have somewhat more limited options
        than standard tokenizer classes. They are most useful when you want to create an end-to-end model that goes
        straight from `tf.string` inputs to outputs.
        """
    # 初始化函数，用于创建一个 Tokenizer 对象
    def __init__(
        self,
        vocab_list: List,                   # 词汇表列表，包含了 Tokenizer 所需的词汇
        do_lower_case: bool,                # 是否将输入文本转换为小写进行分词
        cls_token_id: int = None,           # 分类器标记的 ID，在序列分类中用作序列的第一个标记
        sep_token_id: int = None,           # 分隔符标记的 ID，在构建序列时用于多序列的分隔
        pad_token_id: int = None,           # 填充标记的 ID，在批处理不同长度的序列时使用
        padding: str = "longest",           # 填充类型，可以是"longest"或"max_length"
        truncation: bool = True,            # 是否对序列进行截断，使其不超过最大长度
        max_length: int = 512,              # 序列的最大长度，用于填充和截断
        pad_to_multiple_of: int = None,     # 如果设置，序列将填充到此值的倍数
        return_token_type_ids: bool = True, # 是否返回 token_type_ids
        return_attention_mask: bool = True, # 是否返回 attention_mask
        use_fast_bert_tokenizer: bool = True,  # 是否使用 FastBertTokenizer 类（Tensorflow Text）进行分词
        **tokenizer_kwargs,                 # 其他可能传递给 tokenizer 的参数
        ):
            super().__init__()
            # 调用父类的初始化方法

            if use_fast_bert_tokenizer:
                # 如果使用快速的 BERT 分词器
                self.tf_tokenizer = FastBertTokenizer(
                    vocab_list, token_out_type=tf.int64, lower_case_nfd_strip_accents=do_lower_case, **tokenizer_kwargs
                )
            else:
                # 否则使用静态词汇表创建查找表
                lookup_table = tf.lookup.StaticVocabularyTable(
                    tf.lookup.KeyValueTensorInitializer(
                        keys=vocab_list,
                        key_dtype=tf.string,
                        values=tf.range(tf.size(vocab_list, out_type=tf.int64), dtype=tf.int64),
                        value_dtype=tf.int64,
                    ),
                    num_oov_buckets=1,
                )
                # 使用查找表创建 BERT 分词器层
                self.tf_tokenizer = BertTokenizerLayer(
                    lookup_table, token_out_type=tf.int64, lower_case=do_lower_case, **tokenizer_kwargs
                )

            self.vocab_list = vocab_list
            self.do_lower_case = do_lower_case
            # 设置特殊 token 的索引，如果未提供则从 vocab_list 中获取
            self.cls_token_id = vocab_list.index("[CLS]") if cls_token_id is None else cls_token_id
            self.sep_token_id = vocab_list.index("[SEP]") if sep_token_id is None else sep_token_id
            self.pad_token_id = vocab_list.index("[PAD]") if pad_token_id is None else pad_token_id
            # 初始化用于截断最长序列的 paired_trimmer
            self.paired_trimmer = ShrinkLongestTrimmer(max_length - 3, axis=1)  # Allow room for special tokens
            self.max_length = max_length
            self.padding = padding
            self.truncation = truncation
            self.pad_to_multiple_of = pad_to_multiple_of
            self.return_token_type_ids = return_token_type_ids
            self.return_attention_mask = return_attention_mask
    def from_tokenizer(cls, tokenizer: "PreTrainedTokenizerBase", **kwargs):  # noqa: F821
        """
        Initialize a `TFBertTokenizer` from an existing `Tokenizer`.

        Args:
            tokenizer (`PreTrainedTokenizerBase`):
                The tokenizer to use to initialize the `TFBertTokenizer`.

        Examples:

        ```
        from transformers import AutoTokenizer, TFBertTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        tf_tokenizer = TFBertTokenizer.from_tokenizer(tokenizer)
        ```
        """
        # Retrieve the 'do_lower_case' parameter from kwargs; if not provided, use tokenizer's setting
        do_lower_case = kwargs.pop("do_lower_case", None)
        do_lower_case = tokenizer.do_lower_case if do_lower_case is None else do_lower_case
        # Retrieve the 'cls_token_id' parameter from kwargs; if not provided, use tokenizer's setting
        cls_token_id = kwargs.pop("cls_token_id", None)
        cls_token_id = tokenizer.cls_token_id if cls_token_id is None else cls_token_id
        # Retrieve the 'sep_token_id' parameter from kwargs; if not provided, use tokenizer's setting
        sep_token_id = kwargs.pop("sep_token_id", None)
        sep_token_id = tokenizer.sep_token_id if sep_token_id is None else sep_token_id
        # Retrieve the 'pad_token_id' parameter from kwargs; if not provided, use tokenizer's setting
        pad_token_id = kwargs.pop("pad_token_id", None)
        pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id

        # Get the vocabulary dictionary from the tokenizer and sort it by indices
        vocab = tokenizer.get_vocab()
        vocab = sorted(vocab.items(), key=lambda x: x[1])
        # Extract just the vocabulary tokens into a list
        vocab_list = [entry[0] for entry in vocab]
        # Instantiate a new TFBertTokenizer using the retrieved parameters and vocab_list
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
        Instantiate a `TFBertTokenizer` from a pre-trained tokenizer.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The name or path to the pre-trained tokenizer.

        Examples:

        ```
        from transformers import TFBertTokenizer

        tf_tokenizer = TFBertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        ```
        """
        try:
            # Attempt to create a BertTokenizer instance from the provided pretrained_model_name_or_path
            tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        except:  # noqa: E722
            # If the above fails, fall back to using BertTokenizerFast
            from .tokenization_bert_fast import BertTokenizerFast

            tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        # Call from_tokenizer to create a TFBertTokenizer instance using the obtained tokenizer
        return cls.from_tokenizer(tokenizer, **kwargs)

    def unpaired_tokenize(self, texts):
        # If do_lower_case is True, convert texts to lowercase using case_fold_utf8
        if self.do_lower_case:
            texts = case_fold_utf8(texts)
        # Tokenize texts using tf_tokenizer's tokenize method
        tokens = self.tf_tokenizer.tokenize(texts)
        # Merge dimensions from 1 to -1 in tokens
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
    # 定义一个方法，用于获取配置信息的字典
    def get_config(self):
        # 返回包含各种配置项的字典
        return {
            "vocab_list": self.vocab_list,       # 返回实例的词汇表列表
            "do_lower_case": self.do_lower_case, # 返回是否执行小写转换的布尔值
            "cls_token_id": self.cls_token_id,   # 返回类别标记的 ID
            "sep_token_id": self.sep_token_id,   # 返回分隔标记的 ID
            "pad_token_id": self.pad_token_id,   # 返回填充标记的 ID
        }
```