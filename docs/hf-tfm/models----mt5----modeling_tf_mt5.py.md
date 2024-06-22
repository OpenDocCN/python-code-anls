# `.\transformers\models\mt5\modeling_tf_mt5.py`

```py
# 指定编码方式为 UTF-8
# 版权声明及许可证信息
# 导入日志记录工具
# 导入 mT5 模型相关类和配置
# 导入 mT5 模型配置类
# 获取日志记录器
# 用于文档的配置名称
# mT5 模型的 TensorFlow 实现
class TFMT5Model(TFT5Model):
    r"""
    此类覆盖了 [`TFT5Model`]。请查看超类以获取适当的文档和使用示例。

    示例：

    ```python
    >>> from transformers import TFMT5Model, AutoTokenizer

    >>> model = TFMT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="tf")
    >>> labels = tokenizer(text_target=summary, return_tensors="tf")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```py"""

    # 模型类型
    model_type = "mt5"
    # 配置类
    config_class = MT5Config


class TFMT5ForConditionalGeneration(TFT5ForConditionalGeneration):
    r"""
    此类覆盖了 [`TFT5ForConditionalGeneration`]。请查看超类以获取适当的文档和使用示例。

    示例：

    ```python
    >>> from transformers import TFMT5ForConditionalGeneration, AutoTokenizer

    >>> model = TFMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="tf")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```py"""

    # 模型类型
    model_type = "mt5"
    # 配置类
    config_class = MT5Config


class TFMT5EncoderModel(TFT5EncoderModel):
    r"""
    此类覆盖了 [`TFT5EncoderModel`]。请查看超类以获取适当的文档和使用示例。

    示例：

    ```python
    >>> from transformers import TFMT5EncoderModel, AutoTokenizer

    >>> model = TFMT5EncoderModel.from_pretrained("google/mt5-small")
    # 使用预训练的 Google MT5-small 模型的分词器
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    # 给定待处理的文章内容
    article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    # 使用分词器对文章进行编码得到输入张量
    input_ids = tokenizer(article, return_tensors="tf").input_ids
    # 使用模型进行推断
    outputs = model(input_ids)
    # 获取模型输出中最后一层的隐藏状态
    hidden_state = outputs.last_hidden_state
    
    
    
    # 模型类型为 MT5
    model_type = "mt5"
    # 配置类为 MT5Config
    config_class = MT5Config
```