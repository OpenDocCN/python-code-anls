# `.\models\speech_to_text_2\configuration_speech_to_text_2.py`

```
# coding=utf-8
# 定义脚本编码为 UTF-8

# 版权声明和许可信息
# 版权所有 2021 年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 按“原样”分发，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" Speech2Text model configuration"""

# 从父级目录导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 从工具模块导入日志记录功能
from ...utils import logging

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 预训练配置存档映射，将预训练模型名称映射到其配置文件的 URL
SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/s2t-wav2vec2-large-en-de": (
        "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/config.json"
    ),
    # 查看所有 Speech2Text 模型：https://huggingface.co/models?filter=speech2text2
}


# 定义 Speech2Text2Config 类，继承自 PretrainedConfig 类
class Speech2Text2Config(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`Speech2Text2ForCausalLM`] 的配置。根据指定的参数实例化 Speech2Text2 模型，
    定义模型架构。使用默认值实例化配置将生成类似于 Speech2Text2
    [facebook/s2t-wav2vec2-large-en-de](https://huggingface.co/facebook/s2t-wav2vec2-large-en-de) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。
    ```
    # 定义了 Speech2Text 模型的配置类 Speech2Text2Config 的默认参数
    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            语音到文本模型的词汇表大小，定义了在调用 Speech2TextModel 时传递的 `inputs_ids` 可表示的不同标记数量。
        d_model (`int`, *optional*, defaults to 1024):
            层和池化层的维度。
        decoder_layers (`int`, *optional*, defaults to 12):
            解码器层数。
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Transformer 解码器中每个注意力层的注意力头数。
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            解码器中“中间”（通常称为前馈）层的维度。
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"silu"` 和 `"gelu_new"`。
        dropout (`float`, *optional*, defaults to 0.1):
            嵌入层和池化器中所有全连接层的丢弃概率。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的丢弃比率。
        activation_dropout (`float`, *optional*, defaults to 0.0):
            全连接层内部激活的丢弃比率。
        init_std (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
            参考 https://arxiv.org/abs/1909.11556 进一步了解详情。
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            解码器的 LayerDrop 概率。参见 LayerDrop 论文 https://arxiv.org/abs/1909.11556 了解更多详情。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后一个键/值注意力（并非所有模型都使用）。
        max_target_positions (`int`, *optional*, defaults to 1024):
            模型可能会用到的最大序列长度。通常将其设置为一个较大的值（例如 512、1024 或 2048）。

    Example:

    ```python
    >>> from transformers import Speech2Text2Config, Speech2Text2ForCausalLM

    >>> # 初始化一个 Speech2Text2Config 配置类实例
    >>> configuration = Speech2Text2Config()

    >>> # 从 Speech2Text2Config 配置类实例初始化一个带有随机权重的模型
    >>> model = Speech2Text2ForCausalLM(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```

    # 设置模型类型
    model_type = "speech_to_text_2"
    # 在推断过程中需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，将模型配置的字段映射到其他命名约定
    attribute_map = {"num_attention_heads": "decoder_attention_heads", "hidden_size": "d_model"}
    # 初始化函数，用于初始化 TransformerDecoderModel 类的实例
    def __init__(
        self,
        vocab_size=10000,  # 词汇表大小，默认为10000
        decoder_layers=6,  # 解码器层数，默认为6层
        decoder_ffn_dim=2048,  # 解码器中间层的维度，默认为2048
        decoder_attention_heads=4,  # 解码器注意力头数，默认为4个头
        decoder_layerdrop=0.0,  # 解码器层级随机丢弃的概率，默认为0.0（不丢弃）
        use_cache=True,  # 是否使用缓存，默认为True
        activation_function="relu",  # 激活函数，默认为ReLU
        d_model=256,  # 模型维度，默认为256
        dropout=0.1,  # 全连接层和注意力层的dropout概率，默认为0.1
        attention_dropout=0.0,  # 注意力机制的dropout概率，默认为0.0（不丢弃）
        activation_dropout=0.0,  # 激活函数的dropout概率，默认为0.0（不丢弃）
        init_std=0.02,  # 参数初始化的标准差，默认为0.02
        decoder_start_token_id=2,  # 解码器起始token的ID，默认为2
        scale_embedding=True,  # 是否对embedding进行缩放，默认为True
        pad_token_id=1,  # 填充token的ID，默认为1
        bos_token_id=0,  # 开始token的ID，默认为0
        eos_token_id=2,  # 结束token的ID，默认为2
        max_target_positions=1024,  # 目标序列的最大长度，默认为1024
        **kwargs,  # 其他关键字参数
    ):
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.d_model = d_model  # 设置模型维度
        self.decoder_ffn_dim = decoder_ffn_dim  # 设置解码器中间层维度
        self.decoder_layers = decoder_layers  # 设置解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 设置解码器注意力头数
        self.dropout = dropout  # 设置全连接层和注意力层的dropout概率
        self.attention_dropout = attention_dropout  # 设置注意力机制的dropout概率
        self.activation_dropout = activation_dropout  # 设置激活函数的dropout概率
        self.activation_function = activation_function  # 设置激活函数类型
        self.init_std = init_std  # 设置参数初始化的标准差
        self.decoder_layerdrop = decoder_layerdrop  # 设置解码器层级随机丢弃的概率
        self.use_cache = use_cache  # 设置是否使用缓存
        self.num_hidden_layers = decoder_layers  # 设置隐藏层的数量为解码器层数
        self.scale_embedding = scale_embedding  # 设置是否对embedding进行缩放，若True则缩放因子为sqrt(d_model)
        self.max_target_positions = max_target_positions  # 设置目标序列的最大长度

        # 调用父类的初始化方法，传入填充、起始和结束token的ID以及其他关键字参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
```