# `.\models\mt5\modeling_tf_mt5.py`

```py
# 设置文件编码为UTF-8
# 版权声明，指出代码的版权归属
# 版权使用协议，告知可在Apache License Version 2.0下使用
# 获取Apache License Version 2.0的具体内容链接
# 如果不是根据许可证中规定的，不得使用此文件
# 在适用法律下，本软件按"原样"提供，没有任何明示或暗示的担保或条件
# 参见许可证以了解特定语言的权限
""" Tensorflow mT5 model."""

# 从相对路径导入logging工具
from ...utils import logging
# 从T5的TensorFlow模型中导入编码器模型、有条件生成模型和基础模型
from ..t5.modeling_tf_t5 import TFT5EncoderModel, TFT5ForConditionalGeneration, TFT5Model
# 从当前目录下的配置文件中导入MT5的配置类
from .configuration_mt5 import MT5Config

# 获取logger对象
logger = logging.get_logger(__name__)

# 文档字符串，用于生成文档
_CONFIG_FOR_DOC = "T5Config"

# TFMT5Model类，继承自TFT5Model类，用于MT5模型的TensorFlow实现
class TFMT5Model(TFT5Model):
    r"""
    This class overrides [`TFT5Model`]. Please check the superclass for the appropriate documentation alongside usage
    examples.

    Examples:

    ```
    >>> from transformers import TFMT5Model, AutoTokenizer

    >>> model = TFMT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="tf")
    >>> labels = tokenizer(text_target=summary, return_tensors="tf")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```"""

    # 模型类型为"mt5"
    model_type = "mt5"
    # 配置类为MT5Config
    config_class = MT5Config


# TFMT5ForConditionalGeneration类，继承自TFT5ForConditionalGeneration类，用于带条件生成的MT5模型的TensorFlow实现
class TFMT5ForConditionalGeneration(TFT5ForConditionalGeneration):
    r"""
    This class overrides [`TFT5ForConditionalGeneration`]. Please check the superclass for the appropriate
    documentation alongside usage examples.

    Examples:

    ```
    >>> from transformers import TFMT5ForConditionalGeneration, AutoTokenizer

    >>> model = TFMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="tf")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```"""

    # 模型类型为"mt5"
    model_type = "mt5"
    # 配置类为MT5Config
    config_class = MT5Config


# TFMT5EncoderModel类，继承自TFT5EncoderModel类，用于MT5编码器模型的TensorFlow实现
class TFMT5EncoderModel(TFT5EncoderModel):
    r"""
    This class overrides [`TFT5EncoderModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.

    Examples:

    ```
    >>> from transformers import TFMT5EncoderModel, AutoTokenizer

    >>> model = TFMT5EncoderModel.from_pretrained("google/mt5-small")
    # 设置tokenizer为从预训练模型"google/mt5-small"加载的自动分词器
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    
    # 设置一个新闻文章的示例文本
    article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    
    # 使用tokenizer对文章进行分词并返回TensorFlow格式的输入ID
    input_ids = tokenizer(article, return_tensors="tf").input_ids
    
    # 对输入ID进行模型推理，获取模型的输出
    outputs = model(input_ids)
    
    # 从模型的输出中提取最后一个隐藏状态的表示
    hidden_state = outputs.last_hidden_state
    
    # 设置模型类型为"mt5"，这里暂存了模型的类型信息
    model_type = "mt5"
    
    # 设置配置类为MT5Config，用于模型配置的加载和管理
    config_class = MT5Config
```