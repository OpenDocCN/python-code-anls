# `.\models\falcon\configuration_falcon.py`

```
# 设定文件编码为 UTF-8
# 版权声明及许可协议信息
#
# 在 Apache 许可证 2.0 版本下许可
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何形式的明示或暗示担保或条件，包括但不限于
# 特定用途的适销性或适用性以及任何特定用途的非侵权性担保。
# 有关详细信息，请参见许可证。
"""Falcon 配置"""
# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射字典
FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tiiuae/falcon-40b": "https://huggingface.co/tiiuae/falcon-40b/resolve/main/config.json",
    "tiiuae/falcon-7b": "https://huggingface.co/tiiuae/falcon-7b/resolve/main/config.json",
}


class FalconConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`FalconModel`] 的配置信息。它用于根据指定的参数实例化 Falcon 模型，定义模型架构。使用默认值初始化配置将生成与 [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。


    示例:

    ```python
    >>> from transformers import FalconModel, FalconConfig

    >>> # 初始化一个小型（2层）Falcon配置
    >>> configuration = FalconConfig(num_hidden_layers=2)

    >>> # 从小型配置初始化一个模型
    >>> model = FalconModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "falcon"
    # 推断时要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65024,
        hidden_size=4544,
        num_hidden_layers=32,
        num_attention_heads=71,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_kv_heads=None,
        alibi=False,
        new_decoder_architecture=False,
        multi_query=True,
        parallel_attn=True,
        bias=False,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=11,
        eos_token_id=11,
        **kwargs,
    # 初始化 TransformerDecoderConfig 类
    ):
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 与 n_embed 关键字参数保持向后兼容
        n_embed = kwargs.pop("n_embed", None)
        # 设置隐藏层大小，如果 n_embed 为 None，则使用 hidden_size
        self.hidden_size = hidden_size if n_embed is None else n_embed
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置层归一化的 epsilon
        self.layer_norm_epsilon = layer_norm_epsilon
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置隐藏层的 dropout
        self.hidden_dropout = hidden_dropout
        # 设置注意力的 dropout
        self.attention_dropout = attention_dropout

        # 设置起始符号的 token ID
        self.bos_token_id = bos_token_id
        # 设置终止符号的 token ID
        self.eos_token_id = eos_token_id
        # 如果 num_kv_heads 为 None，则设置为 num_attention_heads
        self.num_kv_heads = num_attention_heads if num_kv_heads is None else num_kv_heads
        # 设置 alibi 属性
        self.alibi = alibi
        # 设置是否使用新的解码器架构
        self.new_decoder_architecture = new_decoder_architecture
        # 设置是否允许多查询（在新的解码器架构为真时被忽略）
        self.multi_query = multi_query  # Ignored when new_decoder_architecture is True
        # 设置是否并行注意力
        self.parallel_attn = parallel_attn
        # 设置是否包含偏置
        self.bias = bias
        # 设置最大位置嵌入
        self.max_position_embeddings = max_position_embeddings
        # 设置 ROPE 中的 theta
        self.rope_theta = rope_theta
        # 设置 ROPE 的缩放
        self.rope_scaling = rope_scaling
        # 执行 ROPE 的缩放验证
        self._rope_scaling_validation()

        # 调用父类的构造函数，传递起始符号 token ID 和终止符号 token ID 以及任何其他关键字参数
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    # 计算头维度
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    # 返回是否使用旋转位置编码
    def rotary(self):
        return not self.alibi

    # 执行 ROPE 缩放验证
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 如果 ROPE 缩放为 None，则直接返回
        if self.rope_scaling is None:
            return

        # 如果 alibi 为真，则不支持 ROPE 缩放
        if self.alibi:
            raise ValueError("`rope_scaling` is not supported when `alibi` is `True`.")

        # 如果 ROPE 缩放不是字典或长度不为 2，则抛出错误
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        # 获取 ROPE 缩放类型和因子
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        # 如果 ROPE 缩放类型为空或不是 ['linear', 'dynamic'] 中的一种，则抛出错误
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        # 如果 ROPE 缩放因子为空或不是大于 1 的浮点数，则抛出错误
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```