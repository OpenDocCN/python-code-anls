# `.\models\decision_transformer\configuration_decision_transformer.py`

```py
# 设置文件编码为 utf-8
# 版权声明，保留所有权利
# 根据 Apache 许可证，除非符合许可证条件，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，根据许可证分发的软件是基于“按现状”分发的
# 没有任何明示或暗示的保证或条件，无论是明示或暗示
# 有关许可证的具体内容，输出数据的限制和
# 在许可证中规定的条件
# 决策变压器模型配置

# 导入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 决策变压器预训练配置的映射表
DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "edbeeching/decision-transformer-gym-hopper-medium": (
        "https://huggingface.co/edbeeching/decision-transformer-gym-hopper-medium/resolve/main/config.json"
    ),
    # 查看所有决策变换器模型请访问 https://huggingface.co/models?filter=decision_transformer
}

# 决策变换器配置类继承自 PretrainedConfig
class DecisionTransformerConfig(PretrainedConfig):
    """
    这是配置类，用于存储[`DecisionTransformerModel`]的配置。它用于根据指定参数实例化一个决策变换器模型，从而定义模型体系结构。
    使用默认值实例化配置将产生类似于标准DecisionTransformer体系结构的配置。许多配置选项用于实例化作为体系结构一部分
    使用的GPT2模型。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读
    [`PretrainedConfig`]的文档以获取更多信息。
    
    例子：
    
    ```python
    >>> from transformers import DecisionTransformerConfig, DecisionTransformerModel

    >>> # 初始化一个决策变换器配置
    >>> configuration = DecisionTransformerConfig()

    >>> # 使用配置初始化一个模型（带随机权重）
    >>> model = DecisionTransformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
   
    # 模型类型为"decision_transformer"
    model_type = "decision_transformer"
    # 推理时需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，用于将配置中的某些属性映射到其他属性
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    # 初始化函数，用于创建一个类实例并初始化其属性
    # state_dim: 状态维度，默认为17
    # act_dim: 动作维度，默认为4
    # hidden_size: 隐藏层大小，默认为128
    # max_ep_len: 最大回合长度，默认为4096
    # action_tanh: 是否对动作进行tanh激活，默认为True
    # vocab_size: 词汇大小，默认为1
    # n_positions: 位置编码的最大长度，默认为1024
    # n_layer: Transformer中的层数，默认为3
    # n_head:  Transformer中的头数，默认为1
    # n_inner: Feedforward层中的内部大小，如果为None，则设置为hidden_size * 4，默认为None
    # activation_function: 激活函数，默认为"relu"
    # resid_pdrop: 残差连接的dropout概率，默认为0.1
    # embd_pdrop: 位置编码和token嵌入的dropout概率，默认为0.1
    # attn_pdrop: 自注意力层的dropout概率，默认为0.1
    # layer_norm_epsilon: 归一化层epsilon的值，默认为1e-5
    # initializer_range: 初始化权重的范围，默认为0.02
    # scale_attn_weights: 是否按层索引对注意力权重进行缩放，默认为True
    # use_cache: 是否缓存输入，默认为True
    # bos_token_id: 开始标记的token id，默认为50256
    # eos_token_id: 结束标记的token id，默认为50256
    # scale_attn_by_inverse_layer_idx: 是否按层索引对注意力权重进行缩放的倒数，默认为False
    # reorder_and_upcast_attn: 在attention时是否要重新排序并提升， 默认为False
    # kwargs: 其他参数
    def __init__(
        self,
        state_dim=17,
        act_dim=4,
        hidden_size=128,
        max_ep_len=4096,
        action_tanh=True,
        vocab_size=1,
        n_positions=1024,
        n_layer=3,
        n_head=1,
        n_inner=None,
        activation_function="relu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        **kwargs,
    ):
        # 初始化类属性
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len
        self.action_tanh = action_tanh
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
    
        # 调用父类的初始化函数并传入参数
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
```