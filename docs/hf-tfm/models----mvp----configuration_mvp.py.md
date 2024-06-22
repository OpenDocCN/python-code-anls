# `.\transformers\models\mvp\configuration_mvp.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版，你不得使用本文件，除非遵守许可证的规定。
# 你可以从下面链接获取许可证的副本：
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，否则根据许可证分发的软件都是基于"原样"的基础上,
# 没有任何明示或暗示的担保或条件。请查看许可证以获取有关权限和限制的详细信息。

""" MVP模型配置 """

# 导入警告模块
import warnings
# 从...包中的配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 从...工具中导入日志模块
from ...utils import logging

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 定义 MVP 预训练配置存档映射
MVP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "RUCAIBox/mvp": "https://huggingface.co/RUCAIBox/mvp/resolve/main/config.json",
}

# MVP配置类继承自PretrainedConfig
class MvpConfig(PretrainedConfig):
    r"""
    这是用于存储[`MvpModel`]配置的配置类。它用于根据指定的参数实例化 MVP 模型，定义模型架构。
    使用默认值实例化配置将生成类似于 MVP [RUCAIBox/mvp](https://huggingface.co/RUCAIBox/mvp) 架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读来自[`PretrainedConfig`]的文档以获取更多信息。

    示例:
    ```python
    >>> from transformers import MvpConfig, MvpModel

    >>> # 初始化MVP RUCAIBox/mvp风格的配置
    >>> configuration = MvpConfig()

    >>> # 从RUCAIBox/mvp风格的配置初始化一个模型(带有随机权重)
    >>> model = MvpModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    model_type = "mvp"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=50267,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        use_prompt=False,
        prompt_length=100,
        prompt_mid_dim=800,
        **kwargs,
    # 初始化模型参数
            # 设置词汇表大小
            self.vocab_size = vocab_size
            # 设置最大位置嵌入数量
            self.max_position_embeddings = max_position_embeddings
            # 设置模型维度
            self.d_model = d_model
            # 设置编码器FFN维度
            self.encoder_ffn_dim = encoder_ffn_dim
            # 设置编码器层数
            self.encoder_layers = encoder_layers
            # 设置编码器注意力头数
            self.encoder_attention_heads = encoder_attention_heads
            # 设置解码器FFN维度
            self.decoder_ffn_dim = decoder_ffn_dim
            # 设置解码器层数
            self.decoder_layers = decoder_layers
            # 设置解码器注意力头数
            self.decoder_attention_heads = decoder_attention_heads
            # 设置全局dropout率
            self.dropout = dropout
            # 设置注意力dropout率
            self.attention_dropout = attention_dropout
            # 设置激活函数的dropout率
            self.activation_dropout = activation_dropout
            # 设置激活函数
            self.activation_function = activation_function
            # 设置初始化标准差
            self.init_std = init_std
            # 设置编码器层丢弃率
            self.encoder_layerdrop = encoder_layerdrop
            # 设置解码器层丢弃率
            self.decoder_layerdrop = decoder_layerdrop
            # 设置分类器丢弃率
            self.classifier_dropout = classifier_dropout
            # 设置是否使用缓存
            self.use_cache = use_cache
            # 设置隐藏层数量
            self.num_hidden_layers = encoder_layers
            # 设置嵌入缩放因子，如果为True，则缩放因子为sqrt(d_model)
            self.scale_embedding = scale_embedding
            # 设置是否使用提示
            self.use_prompt = use_prompt
            # 设置提示长度
            self.prompt_length = prompt_length
            # 设置提示中间维度
            self.prompt_mid_dim = prompt_mid_dim
    
            # 调用父类的初始化方法，传入参数
            super().__init__(
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                is_encoder_decoder=is_encoder_decoder,
                decoder_start_token_id=decoder_start_token_id,
                forced_eos_token_id=forced_eos_token_id,
                **kwargs,  # 剩余的参数使用kwargs传入
            )
    
            # 如果forced_bos_token_id为None并且kwargs中包含"force_bos_token_to_be_generated"为True的键值对
            if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
                # 设置forced_bos_token_id为bos_token_id
                self.forced_bos_token_id = self.bos_token_id
                # 发出警告
                warnings.warn(
                    f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions. "
                    "The config can simply be saved and uploaded again to be fixed."
                )
```