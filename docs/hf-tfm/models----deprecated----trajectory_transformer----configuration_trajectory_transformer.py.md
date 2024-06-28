# `.\models\deprecated\trajectory_transformer\configuration_trajectory_transformer.py`

```py
# 设置文件编码为UTF-8
# 版权声明，声明此代码版权归Trajectory Transformers论文作者和HuggingFace Inc.团队所有
#
# 根据Apache许可证2.0版授权，除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于"按原样"分发的，不提供任何明示或暗示的担保或条件。
# 有关特定语言的详细信息，请参阅许可证。

""" TrajectoryTransformer模型配置"""

# 导入必要的配置类和日志记录工具
from ....configuration_utils import PretrainedConfig
from ....utils import logging

# 获取全局日志记录器实例
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件映射字典
TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "CarlCochet/trajectory-transformer-halfcheetah-medium-v2": (
        "https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2/resolve/main/config.json"
    ),
    # 查看所有TrajectoryTransformer模型请访问 https://huggingface.co/models?filter=trajectory_transformer
}


class TrajectoryTransformerConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储[`TrajectoryTransformerModel`]的配置。根据指定的参数实例化TrajectoryTransformer模型，
    定义模型架构。使用默认参数实例化一个配置将产生类似于TrajectoryTransformer
    [CarlCochet/trajectory-transformer-halfcheetah-medium-v2](https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2)
    架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型的输出。有关更多信息，请阅读[`PretrainedConfig`]的文档。

    ```
    >>> from transformers import TrajectoryTransformerConfig, TrajectoryTransformerModel

    >>> # 初始化一个TrajectoryTransformer模型，以CarlCochet/trajectory-transformer-halfcheetah-medium-v2风格的配置
    >>> configuration = TrajectoryTransformerConfig()

    >>> # 使用随机权重从CarlCochet/trajectory-transformer-halfcheetah-medium-v2风格的配置初始化一个模型
    >>> model = TrajectoryTransformerModel(configuration)

    >>> # 访问模型的配置
    >>> configuration = model.config
    ```
    """

    # 模型类型为trajectory_transformer
    model_type = "trajectory_transformer"
    
    # 推断过程中需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # 属性映射字典，将配置属性映射到模型中的实际名称
    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    def __init__(
        self,
        vocab_size=100,                           # 初始化函数，设定类的初始属性值，其中包括词汇表大小，默认为100
        action_weight=5,                          # 动作权重，默认为5
        reward_weight=1,                          # 奖励权重，默认为1
        value_weight=1,                           # 值权重，默认为1
        block_size=249,                           # 块大小，默认为249
        action_dim=6,                             # 动作维度，默认为6
        observation_dim=17,                       # 观察维度，默认为17
        transition_dim=25,                        # 过渡维度，默认为25
        n_layer=4,                                # 层数，默认为4
        n_head=4,                                 # 头数，默认为4
        n_embd=128,                               # 嵌入维度，默认为128
        embd_pdrop=0.1,                           # 嵌入层dropout率，默认为0.1
        attn_pdrop=0.1,                           # 注意力dropout率，默认为0.1
        resid_pdrop=0.1,                          # 残差连接dropout率，默认为0.1
        learning_rate=0.0006,                     # 学习率，默认为0.0006
        max_position_embeddings=512,              # 最大位置嵌入数，默认为512
        initializer_range=0.02,                   # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,                     # 层归一化epsilon，默认为1e-12
        kaiming_initializer_range=1,              # Kaiming初始化范围，默认为1
        use_cache=True,                           # 是否使用缓存，默认为True
        pad_token_id=1,                           # 填充标记ID，默认为1
        bos_token_id=50256,                       # 开始标记ID，默认为50256
        eos_token_id=50256,                       # 结束标记ID，默认为50256
        **kwargs,                                 # 其他关键字参数
    ):
        self.vocab_size = vocab_size               # 设置对象属性：词汇表大小
        self.action_weight = action_weight         # 设置对象属性：动作权重
        self.reward_weight = reward_weight         # 设置对象属性：奖励权重
        self.value_weight = value_weight           # 设置对象属性：值权重
        self.max_position_embeddings = max_position_embeddings  # 设置对象属性：最大位置嵌入数
        self.block_size = block_size               # 设置对象属性：块大小
        self.action_dim = action_dim               # 设置对象属性：动作维度
        self.observation_dim = observation_dim     # 设置对象属性：观察维度
        self.transition_dim = transition_dim       # 设置对象属性：过渡维度
        self.learning_rate = learning_rate         # 设置对象属性：学习率
        self.n_layer = n_layer                     # 设置对象属性：层数
        self.n_head = n_head                       # 设置对象属性：头数
        self.n_embd = n_embd                       # 设置对象属性：嵌入维度
        self.embd_pdrop = embd_pdrop               # 设置对象属性：嵌入层dropout率
        self.attn_pdrop = attn_pdrop               # 设置对象属性：注意力dropout率
        self.resid_pdrop = resid_pdrop             # 设置对象属性：残差连接dropout率
        self.initializer_range = initializer_range # 设置对象属性：初始化范围
        self.layer_norm_eps = layer_norm_eps       # 设置对象属性：层归一化epsilon
        self.kaiming_initializer_range = kaiming_initializer_range  # 设置对象属性：Kaiming初始化范围
        self.use_cache = use_cache                 # 设置对象属性：是否使用缓存
        super().__init__(                           # 调用父类构造函数，传递填充、开始和结束标记ID以及其他关键字参数
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
```