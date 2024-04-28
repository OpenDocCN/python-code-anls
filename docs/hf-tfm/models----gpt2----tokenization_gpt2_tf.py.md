# `.\models\gpt2\tokenization_gpt2_tf.py`

```
# 导入所需的模块
import os
from typing import Dict, List, Union

import tensorflow as tf
from keras_nlp.tokenizers import BytePairTokenizer
from tensorflow_text import pad_model_inputs
from .tokenization_gpt2 import GPT2Tokenizer

# 创建名为 TFGPT2Tokenizer 的类，用来表示 GPT2 的 in-graph tokenizer
class TFGPT2Tokenizer(tf.keras.layers.Layer):
    """
    This is an in-graph tokenizer for GPT2. It should be initialized similarly to other tokenizers, using the
    `from_pretrained()` method. It can also be initialized with the `from_tokenizer()` method, which imports settings
    from an existing standard tokenizer object.

    In-graph tokenizers, unlike other Hugging Face tokenizers, are actually Keras layers and are designed to be run
    when the model is called, rather than during preprocessing. As a result, they have somewhat more limited options
    than standard tokenizer classes. They are most useful when you want to create an end-to-end model that goes
    straight from `tf.string` inputs to outputs.

    Args:
        vocab (Dict[str, int]): Vocabulary dict for Byte Pair Tokenizer
        merges (List[str]): Merges list for Byte Pair Tokenizer
    """
    # 初始化函数，初始化一些参数和字节对编码器的字典和分词列表
    def __init__(self, vocab: Dict[str, int], merges: List[str], max_length: int = None, pad_token_id: int = None):
        super().__init__()
        # 设置一些参数
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.vocab = vocab
        self.merges = merges
        # 使用字节对编码器初始化文本流 tokenizer
        self.tf_tokenizer = BytePairTokenizer(vocab, merges, sequence_length=max_length)

    # 类方法，从标准 tokenizer 创建 TFGPT2Tokenizer
    @classmethod
    def from_tokenizer(cls, tokenizer: GPT2Tokenizer, *args, **kwargs):
        """Creates TFGPT2Tokenizer from GPT2Tokenizer

        Args:
            tokenizer (GPT2Tokenizer)

        Examples:

        ```python
        from transformers import AutoTokenizer, TFGPT2Tokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tf_tokenizer = TFGPT2Tokenizer.from_tokenizer(tokenizer)
        ```
        """
        # 从 GPT2Tokenizer 中提取字典和分词列表生成 TFGPT2Tokenizer
        merges = [" ".join(m) for m in tokenizer.bpe_ranks.keys()]
        vocab = tokenizer.get_vocab()
        return cls(vocab, merges, *args, **kwargs)

    # 类方法，从预训练模型创建 TFGPT2Tokenizer
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        """Creates TFGPT2Tokenizer from pretrained GPT2Tokenizer

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): Path to pretrained model

        Examples:

        ```python
        from transformers import TFGPT2Tokenizer

        tf_tokenizer = TFGPT2Tokenizer.from_pretrained("gpt2")
        ```
        """
        # 从预训练的 GPT2Tokenizer 创建 TFGPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        return cls.from_tokenizer(tokenizer, *init_inputs, **kwargs)

    # 类方法，从配置中创建 TFGPT2Tokenizer
    @classmethod
    def from_config(cls, config):
        """Creates TFGPT2Tokenizer from configurations

        Args:
            config (Dict): Dictionary with keys such as stated in `get_config`.
        """
        # 从已有的配置文件中创建 TFGPT2Tokenizer
        return cls(**config)
    # 返回当前配置信息，包括词汇表、合并规则、最大长度和填充标记编号
    def get_config(self):
        return {
            "vocab": self.vocab,
            "merges": self.merges,
            "max_length": self.max_length,
            "pad_token_id": self.pad_token_id,
        }

    # 调用函数，对输入进行处理，返回注意力掩码和输入标识
    def call(self, x, max_length: int = None):
        # 使用 TF Tokenizer 处理输入 x，得到输入标识
        input_ids = self.tf_tokenizer(x)
        # 创建一个全为1的注意力掩码
        attention_mask = tf.ones_like(input_ids)

        if self.pad_token_id is not None:
            # 填充标记编号不为空时，对标识进行填充，使其达到最大长度
            max_length = max_length if max_length is not None else self.max_length

            if max_length is not None:
                # 使用 pad_model_inputs 函数对标识和注意力掩码进行填充
                input_ids, attention_mask = pad_model_inputs(
                    input_ids, max_seq_length=max_length, pad_value=self.pad_token_id
                )

        # 返回注意力掩码和输入标识
        return {"attention_mask": attention_mask, "input_ids": input_ids}
```