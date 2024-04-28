# `.\transformers\models\mt5\modeling_flax_mt5.py`

```
# 使用 UTF-8 编码格式
# 版权声明和许可证信息，指明代码版权和使用许可
# 引入所需的库和模块
import jax.numpy as jnp
# 引入日志记录工具
from ...utils import logging
# 引入 T5 模型相关的类和函数
from ..t5.modeling_flax_t5 import FlaxT5EncoderModel, FlaxT5ForConditionalGeneration, FlaxT5Model
# 引入 MT5 模型的配置类
from .configuration_mt5 import MT5Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中的配置信息
_CONFIG_FOR_DOC = "T5Config"

# 定义一个函数，将输入的 token 向右移动一位
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    # 创建一个全零数组，与输入 token 形状相同
    shifted_input_ids = jnp.zeros_like(input_ids)
    # 将输入 token 向右移动一位
    shifted_input_ids = shifted_input_ids.at[:, 1:].set(input_ids[:, :-1])
    # 将移动后的第一个位置设置为解码器的起始 token id
    shifted_input_ids = shifted_input_ids.at[:, 0].set(decoder_start_token_id)
    # 将值为-100的位置替换为 pad token id
    shifted_input_ids = jnp.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids

# 定义 FlaxMT5Model 类，继承自 FlaxT5Model
class FlaxMT5Model(FlaxT5Model):
    r"""
    This class overrides [`FlaxT5Model`]. Please check the superclass for the appropriate documentation alongside usage
    examples.

    Examples:

    ```python
    >>> from transformers import FlaxMT5Model, AutoTokenizer

    >>> model = FlaxMT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="np")

    >>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids)
    >>> hidden_states = outputs.last_hidden_state
    ```"""

    # 模型类型
    model_type = "mt5"
    # 配置类
    config_class = MT5Config

# 定义 FlaxMT5EncoderModel 类，继承自 FlaxT5EncoderModel
class FlaxMT5EncoderModel(FlaxT5EncoderModel):
    r"""
    This class overrides [`FlaxT5EncoderModel`]. Please check the superclass for the appropriate documentation
    alongside usage examples.

    Examples:

    ```python
    >>> from transformers import FlaxT5EncoderModel, AutoTokenizer

    >>> model = FlaxT5EncoderModel.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    ```    
    # 使用分词器将输入文章转换为张量格式
    >>> inputs = tokenizer(article, return_tensors="np")
    
    # 使用分词器将目标摘要转换为张量格式的输入ID
    >>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids
    
    # 使用模型对输入文章进行编码，获取最后一层隐藏状态
    >>> outputs = model(input_ids=inputs["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    
    # 声明模型类型为 mt5
    model_type = "mt5"
    # 声明配置类为 MT5Config
    config_class = MT5Config
# 定义了一个自定义的 MT5 条件生成器类，继承自 FlaxT5ForConditionalGeneration 类
class FlaxMT5ForConditionalGeneration(FlaxT5ForConditionalGeneration):
    r"""
    这个类覆盖了 [`FlaxT5ForConditionalGeneration`]。请查看超类以获取适当的文档以及使用示例。

    示例:

    ```python
    >>> from transformers import FlaxMT5ForConditionalGeneration, AutoTokenizer

    >>> model = FlaxMT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="np")

    >>> decoder_input_ids = tokenizer(text_target=summary, return_tensors="np").input_ids

    >>> outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
    >>> logits = outputs.logits
    ```"""

    # 指定模型类型为 "mt5"
    model_type = "mt5"
    # 指定配置类为 MT5Config
    config_class = MT5Config
```