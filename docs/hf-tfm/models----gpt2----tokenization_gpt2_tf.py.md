# `.\models\gpt2\tokenization_gpt2_tf.py`

```py
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        """Creates TFGPT2Tokenizer from pretrained GPT2Tokenizer

        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): Path to pretrained model

        Examples:

        ```
        from transformers import TFGPT2Tokenizer

        tf_tokenizer = TFGPT2Tokenizer.from_pretrained("openai-community/gpt2")
        ```
        """
        # 使用给定的模型名或路径加载预训练的GPT2Tokenizer对象
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
        # 使用加载的GPT2Tokenizer对象创建TFGPT2Tokenizer对象
        return cls.from_tokenizer(tokenizer, *init_inputs, **kwargs)
    # 从配置信息创建 TFGPT2Tokenizer 的类方法
    def from_config(cls, config):
        """Creates TFGPT2Tokenizer from configurations

        Args:
            config (Dict): Dictionary with keys such as stated in `get_config`.
        """
        # 使用传入的配置参数创建 TFGPT2Tokenizer 实例并返回
        return cls(**config)

    # 返回当前实例的配置信息字典
    def get_config(self):
        return {
            "vocab": self.vocab,                # 返回词汇表
            "merges": self.merges,              # 返回合并信息
            "max_length": self.max_length,      # 返回最大长度
            "pad_token_id": self.pad_token_id,  # 返回填充标记的ID
        }

    # 对输入的文本进行处理，生成模型的输入
    def call(self, x, max_length: int = None):
        # 使用 TensorFlow Tokenizer 处理输入文本得到输入的ID
        input_ids = self.tf_tokenizer(x)
        # 创建一个全为1的注意力掩码
        attention_mask = tf.ones_like(input_ids)

        if self.pad_token_id is not None:
            # 如果存在填充标记ID，则将输入ID填充至最大长度
            max_length = max_length if max_length is not None else self.max_length

            if max_length is not None:
                # 使用 pad_model_inputs 函数填充输入ID和注意力掩码
                input_ids, attention_mask = pad_model_inputs(
                    input_ids, max_seq_length=max_length, pad_value=self.pad_token_id
                )

        # 返回注意力掩码和填充后的输入ID作为字典形式的结果
        return {"attention_mask": attention_mask, "input_ids": input_ids}
```