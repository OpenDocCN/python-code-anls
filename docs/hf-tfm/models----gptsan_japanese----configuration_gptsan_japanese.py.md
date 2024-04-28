# `.\models\gptsan_japanese\configuration_gptsan_japanese.py`

```py
# 设置文件编码为 UTF-8
# 版权归 HuggingFace 公司所有，2023 年
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的条款，否则不得使用此文件。
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或以书面形式同意，否则未经许可的软件
# 基于 "原样" 分发，在任何情况下都没有担保或条件，
# 无论是明示的还是隐含的担保或条件。
# 请查阅许可证以查看特定语言的权限和限制。
"""  GPTSAN-japanese 模型配置"""
# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志工具
from ...utils import logging


# 获取 logger 对象
logger = logging.get_logger(__name__)

# 预训练配置文件映射
GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tanreinama/GPTSAN-2.8B-spout_is_uniform": (
        "https://huggingface.co/tanreinama/GPTSAN-2.8B-spout_is_uniform/resolve/main/config.json"
    ),
}


# GPTSanJapanese 配置类
class GPTSanJapaneseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTSanJapaneseModel`]. It is used to instantiate
    a GPTSANJapanese model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTSANJapanese
    [Tanrei/GPTSAN-japanese](https://huggingface.co/Tanrei/GPTSAN-japanese) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    """
    
    # 模型类型
    model_type = "gptsan-japanese"
    # 推断时需要忽略的键
    keys_to_ignore_at_inference = [
        "past_key_values",
    ]
    # 属性映射
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    # 初始化方法
    def __init__(
        self,
        vocab_size=36000,
        max_position_embeddings=1280,
        d_model=1024,
        d_ff=8192,
        d_ext=4096,
        d_spout=128,
        num_switch_layers=10,
        num_ext_layers=0,
        num_heads=16,
        num_experts=16,
        expert_capacity=128,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-5,
        router_bias=False,
        router_jitter_noise=0.0,
        router_dtype="float32",
        router_ignore_padding_tokens=False,
        output_hidden_states=False,
        output_attentions=False,
        initializer_factor=0.002,
        output_router_logits=False,
        use_cache=True,
        separator_token_id=35998,
        pad_token_id=35995,
        eos_token_id=35999,
        **kwargs,
    ):
        # 设置模型的词汇量大小
        self.vocab_size = vocab_size
        # 设置模型的最大位置编码长度
        self.max_position_embeddings = max_position_embeddings
        # 设置模型的隐藏层大小
        self.d_model = d_model
        # 设置模型前馈网络隐藏层大小
        self.d_ff = d_ff
        # 设置模型扩展输出的隐藏层大小
        self.d_ext = d_ext
        # 设置模型选择器输出的隐藏层大小
        self.d_spout = d_spout
        # 设置模型中的开关层数量
        self.num_switch_layers = num_switch_layers
        # 设置模型中扩展层数量
        self.num_ext_layers = num_ext_layers
        # 设置模型的总层数量
        self.num_layers = num_switch_layers + num_ext_layers
        # 设置模型注意力头的数量
        self.num_heads = num_heads
        # 设置模型的专家数量
        self.num_experts = num_experts
        # 设置模型每个专家的容量
        self.expert_capacity = expert_capacity
        # 设置模型的dropout率
        self.dropout_rate = dropout_rate
        # 设置层归一化的 epsilon 值
        self.layer_norm_epsilon = layer_norm_epsilon
        # 设置路由器的偏置
        self.router_bias = router_bias
        # 设置路由器的抖动噪声
        self.router_jitter_noise = router_jitter_noise
        # 设置路由器的数据类型
        self.router_dtype = router_dtype
        # 设置路由器是否忽略填充标记
        self.router_ignore_padding_tokens = router_ignore_padding_tokens
        # 设置是否输出隐藏状态
        self.output_hidden_states = output_hidden_states
        # 设置是否输出注意力分布
        self.output_attentions = output_attentions
        # 设置初始化因子
        self.initializer_factor = initializer_factor
        # 设置是否输出路由器的 logits
        self.output_router_logits = output_router_logits
        # 设置是否使用缓存
        self.use_cache = use_cache

        # 调用父类的构造函数，并传递相关参数
        super().__init__(
            separator_token_id=separator_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
```